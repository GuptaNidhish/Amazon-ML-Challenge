import os
import time
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

from config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_RESULTS, CANDIDATE_PAIRS, TEST_DIR, MODEL_DIR, OUTPUT_DIR
)
from data_loader import (
    load_source_df, load_ground_truth, load_train_data, load_test_data,
    stream_source_file_by_country
)
import re
from rapidfuzz import fuzz
from normalization import normalize_name, normalize_address, clean_core_name
from evaluation import compute_macro_f05, compute_detailed_metrics, create_validation_split
from blocking import CountryBlocker, SqliteCountryBlocker, int_to_eid, eid_to_int, extract_blocking_keys
from features import batch_compute_features, compute_pair_features, FEATURE_NAMES
from model import train_lgbm, predict_proba
from thresholding import optimize_threshold, apply_threshold
from inference import save_submission_files, run_official_validator, StreamingSubmissionWriter

CITY_STATE_WORDS = {
    'kolkata', 'howrah', 'bengal', 'delhi', 'mumbai', 'maharashtra', 'bangalore', 'karnataka',
    'chennai', 'tamil', 'nadu', 'hyderabad', 'telangana', 'gujarat', 'ahmedabad', 'pune',
    'jaipur', 'rajasthan', 'lucknow', 'uttar', 'pradesh', 'india', 'texas', 'houston',
    'dallas', 'california', 'florida', 'york', 'illinois', 'chicago', 'georgia', 'ohio',
    'pennsylvania', 'north', 'carolina', 'south', 'virginia', 'france', 'paris', 'lyon'
}


def process_s1_batch(
    batch: List[Tuple[str, str, str]],
    blocker: SqliteCountryBlocker,
    booster,
    threshold: float,
    writer: StreamingSubmissionWriter,
    max_cands: int = 40
) -> int:
    """Processes a chunk of S1 entities, predicts matches with LightGBM, and streams to TSV."""
    pairs = []
    s1_slices = []
    
    for s1_id, name, addr in batch:
        cands = blocker.query_candidates(name, addr, max_cands=max_cands)
        norm_n1 = normalize_name(name)
        norm_a1 = normalize_address(addr)
        
        start_idx = len(pairs)
        cand_strs = []
        for eid_int, norm_n2, norm_a2 in cands:
            cand_id = int_to_eid(eid_int)
            cand_strs.append(cand_id)
            pairs.append((norm_n1, norm_a1, norm_n2, norm_a2, cand_id))
            
        s1_slices.append((s1_id, start_idx, len(pairs), cand_strs))
        
    if pairs:
        X_b = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
        for i, (n1, a1, n2, a2, cid) in enumerate(pairs):
            X_b[i] = compute_pair_features(n1, a1, n2, a2, cid)
            
        probas = booster.predict(X_b)
        del X_b
        
        # Precision gating & False Positive filtering
        for i, (n1, a1, n2, a2, cid) in enumerate(pairs):
            n_sim = fuzz.token_sort_ratio(n1, n2) / 100.0
            c1_core = clean_core_name(n1)
            c2_core = clean_core_name(n2)
            core_sim = fuzz.token_sort_ratio(c1_core, c2_core) / 100.0
            
            nums1 = {x.lstrip('0') or '0' for x in re.findall(r'\d+', a1)}
            nums2 = {x.lstrip('0') or '0' for x in re.findall(r'\d+', a2)}
            num_conflict = bool(nums1 and nums2 and len(nums1 & nums2) == 0)
            
            # Distinctive Multi-word Name Exact Boost (min 2 words or len >= 6)
            c1 = n1.replace(' ', '')
            c2 = n2.replace(' ', '')
            is_multiword = len(n1.split()) >= 2 and len(n2.split()) >= 2
            if ((n1 == n2 and is_multiword) or (c1 == c2 and len(c1) >= 6) or (c1_core == c2_core and len(c1_core) >= 5 and len(c1_core.split()) >= 2)) and not num_conflict:
                probas[i] = max(probas[i], 0.98)
                
            # Filter weak name matches that only match on city/state
            if max(n_sim, core_sim) < 0.45:
                shared_nums = nums1 & nums2
                words1 = {w for w in a1.split() if len(w) >= 4 and w not in CITY_STATE_WORDS}
                words2 = {w for w in a2.split() if len(w) >= 4 and w not in CITY_STATE_WORDS}
                shared_words = words1 & words2
                
                if (nums1 and nums2 and not shared_nums) or not shared_words:
                    probas[i] = 0.0
        
        for s1_id, start_idx, end_idx, cand_strs in s1_slices:
            matches = []
            for idx, p_idx in enumerate(range(start_idx, end_idx)):
                if probas[p_idx] >= threshold:
                    matches.append(cand_strs[idx])
            writer.write_record(s1_id, matches, cand_strs)
    else:
        for s1_id, _, _, cand_strs in s1_slices:
            writer.write_record(s1_id, [], cand_strs)
            
    return len(pairs)


def run_full_ml_inference(threshold: float = 0.80, batch_size: int = 5000, max_cands: int = 40):
    """
    Submission 2: End-to-End Scalable, Disk-Backed, Zero-OOM ML Pipeline.
    1. Loads trained LightGBM model.
    2. Streams country partitions sequentially (France, US, India).
    3. Disk-backed SQLite inverted indexing: ultra-fast and RAM strictly < 200 MB.
    4. Streams S1 entities in batches of 5000: queries candidates, computes 36 features,
       predicts probabilities, and writes directly to TSV files.
    5. Cleans up temporary disk indices after each country partition.
    6. Runs official validate_submission.py to verify 100% compliance.
    """
    print("\n" + "="*70)
    print(f"RUNNING SUBMISSION 2: FULL ML PIPELINE (Threshold tau = {threshold:.2f})")
    print("="*70)
    t0 = time.time()
    
    import lightgbm as lgb
    model_path = os.path.join(MODEL_DIR, 'lgbm_model.txt')
    if not os.path.isfile(model_path):
        print(f"Error: Model not found at {model_path}. Run training first!")
        return False
        
    print(f"Loading LightGBM model from {model_path}...")
    booster = lgb.Booster(model_file=model_path)
    
    countries = ['France', 'US', 'India']
    
    with StreamingSubmissionWriter(MATCHING_RESULTS, CANDIDATE_PAIRS) as writer:
        for country in countries:
            print(f"\n{'='*40}")
            print(f"PROCESSING PARTITION: {country}")
            print(f"{'='*40}")
            t_c = time.time()
            
            # Step 1: Fit SqliteCountryBlocker
            db_file = os.path.join(OUTPUT_DIR, f'blocker_{country.lower()}.db')
            blocker = SqliteCountryBlocker(country, db_path=db_file)
            print("1. Building disk-backed SQLite blocker index...")
            
            def stream_targets():
                for path in [TEST_S2, TEST_S3]:
                    for eid, name, addr in stream_source_file_by_country(path, country):
                        yield eid, name, addr
                        
            n_target = blocker.fit_stream(stream_targets())
            db_size_mb = os.path.getsize(db_file) / 1e6 if os.path.exists(db_file) else 0.0
            print(f"   Indexed {n_target:,} target records in {time.time()-t_c:.1f}s, DB size: {db_size_mb:.1f} MB")
            
            # Step 2: Stream S1 records in batches, query, score, and write
            print(f"2. Querying, scoring, and streaming output in batches of {batch_size:,} S1 entities...")
            t_s = time.time()
            s1_batch = []
            total_s1 = 0
            total_pairs = 0
            
            for eid, name, addr in stream_source_file_by_country(TEST_S1, country):
                s1_batch.append((eid, name, addr))
                total_s1 += 1
                if len(s1_batch) >= batch_size:
                    pairs_scored = process_s1_batch(s1_batch, blocker, booster, threshold, writer, max_cands=max_cands)
                    total_pairs += pairs_scored
                    s1_batch.clear()
                    
            if s1_batch:
                pairs_scored = process_s1_batch(s1_batch, blocker, booster, threshold, writer, max_cands=max_cands)
                total_pairs += pairs_scored
                s1_batch.clear()
                
            writer.flush()
            print(f"   Processed {total_s1:,} S1 entities ({total_pairs:,} candidate pairs) in {time.time()-t_s:.1f}s")
            
            # Step 3: Cleanup blocker & temp DB
            blocker.close()
            gc.collect()
            print(f"Completed partition {country} in {time.time()-t_c:.1f}s")
            
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE! Total time: {time.time()-t0:.1f}s")
    print(f"{'='*70}")
    
    print("\nRunning official validator...")
    is_valid = run_official_validator()
    return is_valid


def run_baseline_pipeline():
    """
    Submission 1: High-precision Exact & Compact Name Match Baseline.
    Streams across countries (US, India, and France) with minimal RAM.
    Validates output with official validator.
    """
    print("\n" + "="*70)
    print("RUNNING SUBMISSION 1: HIGH-PRECISION NAME MATCH BASELINE")
    print("="*70)
    t0 = time.time()
    
    countries = ['France', 'US', 'India']
    
    with StreamingSubmissionWriter(MATCHING_RESULTS, CANDIDATE_PAIRS) as writer:
        for country in countries:
            print(f"\n--- Processing {country} ---")
            t_c = time.time()
            
            name_idx = {}
            compact_idx = {}
            target_count = 0
            
            for path in [TEST_S2, TEST_S3]:
                for eid, name, _ in stream_source_file_by_country(path, country):
                    target_count += 1
                    norm_n = normalize_name(name)
                    if norm_n:
                        if norm_n not in name_idx:
                            name_idx[norm_n] = []
                        if len(name_idx[norm_n]) < 25:
                            name_idx[norm_n].append(eid)
                            
                        compact = norm_n.replace(' ', '')
                        if len(compact) >= 5:
                            if compact not in compact_idx:
                                compact_idx[compact] = []
                            if len(compact_idx[compact]) < 25:
                                compact_idx[compact].append(eid)
                                
            print(f"  Indexed {target_count:,} target records in {time.time()-t_c:.1f}s")
            
            s1_count = 0
            t_m = time.time()
            for eid, name, _ in stream_source_file_by_country(TEST_S1, country):
                s1_count += 1
                norm_n = normalize_name(name)
                compact = norm_n.replace(' ', '') if norm_n else ''
                
                cands = []
                if norm_n in name_idx:
                    cands.extend(name_idx[norm_n])
                elif len(compact) >= 5 and compact in compact_idx:
                    cands.extend(compact_idx[compact])
                    
                if len(cands) > 20:
                    cands = cands[:20]
                    
                writer.write_record(eid, cands, cands)
                
            print(f"  Processed {s1_count:,} S1 entities in {time.time()-t_m:.1f}s")
            del name_idx, compact_idx
            gc.collect()
            
    print(f"\nBaseline generation completed in {time.time()-t0:.1f}s")
    print("Running official validator...")
    is_valid = run_official_validator()
    return is_valid


def run_training_and_validation(sample_size: int = 35000):
    """
    Optimized, low-RAM streaming training pipeline:
    1. Samples S1 entities and true targets directly from ground truth.
    2. Streams only required records from TSV files (<150 MB RAM).
    3. Builds training and validation pairs with blocking.
    4. Computes 36 C++ RapidFuzz & Soundex features.
    5. Trains LightGBM and evaluates macro F0.5 across thresholds.
    6. Saves booster to models/lgbm_model.txt.
    """
    import random
    import lightgbm as lgb
    from collections import defaultdict
    
    print("\n" + "="*70)
    print(f"TRAINING PIPELINE (Sample Size: {sample_size:,} entities)")
    print("="*70)
    t0 = time.time()
    
    random.seed(42)
    np.random.seed(42)
    
    print("1. Sampling entities from ground truth...")
    gt_all = {}
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        next(f, None)
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            p = line.rstrip('\n').split('\t')
            gt_all[p[0]] = set(p[1].split(',')) if len(p) > 1 and p[1] else set()
            
    s1_list = list(gt_all.keys())
    random.shuffle(s1_list)
    n_train = int(len(s1_list) * 0.8)
    train_s1_ids = set(s1_list[:n_train])
    val_s1_ids = set(s1_list[n_train:])
    needed_s1 = train_s1_ids | val_s1_ids
    all_true_target_ids = {mid for mlist in gt_all.values() for mid in mlist}
    
    print(f"   Train S1: {len(train_s1_ids):,}, Val S1: {len(val_s1_ids):,}, True Targets: {len(all_true_target_ids):,}")
    
    print("2. Streaming required records from disk...")
    s1_records = {}
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        next(f, None)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in needed_s1:
                s1_records[p[0]] = (p[1], p[2], p[3] if len(p) > 3 else '')
                
    target_records = {}
    for fn in [TRAIN_S2, TRAIN_S3]:
        with open(fn, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                p = line.rstrip('\n').split('\t')
                if p[0] in all_true_target_ids:
                    target_records[p[0]] = (p[1], p[2], p[3] if len(p) > 3 else '')
                    
    print(f"   Extracted {len(s1_records):,} S1 and {len(target_records):,} target records in {time.time()-t0:.1f}s")
    
    print("3. Indexing targets for candidate retrieval...")
    t_idx = time.time()
    target_index = defaultdict(list)
    norm_targets = {}
    for tid, (tn, ta, _) in target_records.items():
        norm_n = normalize_name(tn)
        norm_a = normalize_address(ta)
        norm_targets[tid] = (norm_n, norm_a, 1.0 if tid.startswith('S3') else 0.0)
        keys = extract_blocking_keys(tn, ta)
        for k in keys:
            target_index[k].append(tid)
            
    norm_s1 = {sid: (normalize_name(rec[0]), normalize_address(rec[1])) for sid, rec in s1_records.items()}
    print(f"   Indexed targets in {time.time()-t_idx:.1f}s")
    
    print("4. Generating training pairs and computing feature matrix...")
    t_tr = time.time()
    train_pairs = []
    y_train = []
    train_seen = set()
    
    for s1_id in train_s1_ids:
        s1_n, s1_a, _ = s1_records[s1_id]
        true_targets = gt_all[s1_id]
        for tid in true_targets:
            if tid in target_records:
                train_pairs.append((s1_id, tid))
                y_train.append(1)
                train_seen.add((s1_id, tid))
                
        keys = extract_blocking_keys(s1_n, s1_a)
        cands = []
        for k in keys:
            for cid in target_index[k][:25]:
                if (s1_id, cid) not in train_seen and cid not in true_targets:
                    train_seen.add((s1_id, cid))
                    train_pairs.append((s1_id, cid))
                    y_train.append(0)
                    cands.append(cid)
                    if len(cands) >= 20:
                        break
            if len(cands) >= 20:
                break
                
    y_train = np.array(y_train, dtype=np.int32)
    print(f"   Train pairs: {len(train_pairs):,} (positives: {(y_train==1).sum():,}, negatives: {(y_train==0).sum():,})")
    
    X_train = np.empty((len(train_pairs), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (s1_id, tid) in enumerate(train_pairs):
        n1, a1 = norm_s1[s1_id]
        n2, a2, is_s3 = norm_targets[tid]
        X_train[i] = compute_pair_features(n1, a1, n2, a2, 'S3' if is_s3 else 'S2')
        
    print(f"   Feature matrix computed in {time.time()-t_tr:.1f}s. Shape: {X_train.shape}")
    
    print("\n5. Training LightGBM Booster...")
    t_m = time.time()
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.04,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'min_child_samples': 25,
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    booster = lgb.train(params, dtrain, num_boost_round=800)
    
    model_path = os.path.join(MODEL_DIR, 'lgbm_model.txt')
    booster.save_model(model_path)
    print(f"   Trained and saved booster to {model_path} in {time.time()-t_m:.1f}s")
    
    print("\n6. Evaluating on Validation Split...")
    val_pairs = []
    val_slices = {}
    
    for s1_id in val_s1_ids:
        s1_n, s1_a, _ = s1_records[s1_id]
        keys = extract_blocking_keys(s1_n, s1_a)
        cands = []
        seen = set()
        for k in keys:
            for cid in target_index[k][:35]:
                if cid not in seen:
                    seen.add(cid)
                    cands.append(cid)
                    if len(cands) >= 35:
                        break
            if len(cands) >= 35:
                break
                
        start_p = len(val_pairs)
        for cid in cands:
            val_pairs.append((s1_id, cid))
        val_slices[s1_id] = (start_p, len(val_pairs), cands)
        
    X_val = np.empty((len(val_pairs), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (s1_id, tid) in enumerate(val_pairs):
        n1, a1 = norm_s1[s1_id]
        n2, a2, is_s3 = norm_targets[tid]
        X_val[i] = compute_pair_features(n1, a1, n2, a2, 'S3' if is_s3 else 'S2')
        
    val_probas = booster.predict(X_val)
    
    # Precision gating & False Positive filtering
    for i, (s1_id, tid) in enumerate(val_pairs):
        n1, a1 = norm_s1[s1_id]
        n2, a2, _ = norm_targets[tid]
        n_sim = fuzz.token_sort_ratio(n1, n2) / 100.0
        c1_core = clean_core_name(n1)
        c2_core = clean_core_name(n2)
        core_sim = fuzz.token_sort_ratio(c1_core, c2_core) / 100.0
        
        nums1 = {x.lstrip('0') or '0' for x in re.findall(r'\d+', a1)}
        nums2 = {x.lstrip('0') or '0' for x in re.findall(r'\d+', a2)}
        num_conflict = bool(nums1 and nums2 and len(nums1 & nums2) == 0)
        
        c1 = n1.replace(' ', '')
        c2 = n2.replace(' ', '')
        is_multiword = len(n1.split()) >= 2 and len(n2.split()) >= 2
        if ((n1 == n2 and is_multiword) or (c1 == c2 and len(c1) >= 6) or (c1_core == c2_core and len(c1_core) >= 5 and len(c1_core.split()) >= 2)) and not num_conflict:
            val_probas[i] = max(val_probas[i], 0.98)
            
        if max(n_sim, core_sim) < 0.45:
            shared_nums = nums1 & nums2
            words1 = {w for w in a1.split() if len(w) >= 4 and w not in CITY_STATE_WORDS}
            words2 = {w for w in a2.split() if len(w) >= 4 and w not in CITY_STATE_WORDS}
            shared_words = words1 & words2
            
            if (nums1 and nums2 and not shared_nums) or not shared_words:
                val_probas[i] = 0.0
                
    val_gt = {sid: gt_all[sid] for sid in val_s1_ids}
    
    print("\n--- THRESHOLD OPTIMIZATION RESULTS ---")
    best_tau = 0.80
    best_f05 = -1
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        preds = {}
        for s1_id, (sp, ep, cands) in val_slices.items():
            matches = {cands[idx] for idx, p_idx in enumerate(range(sp, ep)) if val_probas[p_idx] >= tau}
            preds[s1_id] = matches
        f05 = compute_macro_f05(preds, val_gt)
        singleton_pct = sum(1 for m in preds.values() if len(m) == 0) / len(preds) * 100
        mark = "  <-- OPTIMAL TAU*" if f05 > best_f05 else ""
        print(f"  tau = {tau:.2f} -> Macro F0.5 = {f05:.4f} | Singletons: {singleton_pct:.2f}%{mark}")
        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau
            
    print(f"\nOPTIMAL TAU: {best_tau:.2f} with Macro F0.5 = {best_f05:.4f}")
    print(f"Total elapsed time: {time.time()-t0:.1f}s")
    return booster, best_tau, best_f05
