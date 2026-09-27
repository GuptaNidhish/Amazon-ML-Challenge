import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

from config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_RESULTS, CANDIDATE_PAIRS, TEST_DIR, MODEL_DIR
)
from data_loader import load_source_df, load_ground_truth, load_train_data, load_test_data
from normalization import normalize_name, normalize_address
from evaluation import compute_macro_f05, compute_detailed_metrics, create_validation_split
from blocking import CountryBlocker
from features import batch_compute_features
from model import train_lgbm, predict_proba
from thresholding import optimize_threshold, apply_threshold
from inference import save_submission_files, run_official_validator


def run_baseline_pipeline():
    """
    Submission 1: High-precision Exact & Compact Name Match Baseline.
    Processes all 1.73M test S1 entities across US, India, and France.
    Matches S1 to S2/S3 entities that share the normalized or compact business name.
    Validates output with official validator.
    """
    print("\n" + "="*70)
    print("RUNNING SUBMISSION 1: HIGH-PRECISION NAME MATCH BASELINE")
    print("="*70)
    t0 = time.time()
    
    print("1. Loading Test Sources...")
    te1, te2, te3 = load_test_data()
    print(f"Loaded: {len(te1)} S1, {len(te2)} S2, {len(te3)} S3 in {time.time()-t0:.1f}s")
    
    matched_dict = {}
    candidate_dict = {}
    
    for country in te1['country'].unique():
        t_c = time.time()
        c_te1 = te1[te1['country'] == country]
        c_te2 = te2[te2['country'] == country]
        c_te3 = te3[te3['country'] == country]
        print(f"\nProcessing {country}: {len(c_te1)} S1, {len(c_te2)} S2, {len(c_te3)} S3...")
        
        target_ids = list(c_te2['entity_id']) + list(c_te3['entity_id'])
        target_names = list(c_te2['business_name']) + list(c_te3['business_name'])
        
        name_idx = {}
        compact_idx = {}
        
        for eid, name in zip(target_ids, target_names):
            norm_n = normalize_name(name)
            if norm_n:
                if norm_n not in name_idx:
                    name_idx[norm_n] = []
                name_idx[norm_n].append(eid)
                
                compact = norm_n.replace(' ', '')
                if len(compact) >= 5:
                    if compact not in compact_idx:
                        compact_idx[compact] = []
                    compact_idx[compact].append(eid)
                    
        print(f"  Indexed in {time.time()-t_c:.1f}s. Matching S1 entities...")
        
        t_m = time.time()
        for s1_id, s1_name in zip(c_te1['entity_id'], c_te1['business_name']):
            norm_s1 = normalize_name(s1_name)
            compact_s1 = norm_s1.replace(' ', '') if norm_s1 else ''
            
            cands = []
            if norm_s1 in name_idx:
                cands.extend(name_idx[norm_s1])
            elif len(compact_s1) >= 5 and compact_s1 in compact_idx:
                cands.extend(compact_idx[compact_s1])
                
            if len(cands) > 20:
                cands = cands[:20]
                
            c_set = set(cands)
            candidate_dict[s1_id] = list(c_set)
            matched_dict[s1_id] = c_set
            
        print(f"  Matched in {time.time()-t_m:.1f}s")
        
    all_test_s1_ids = list(te1['entity_id'])
    save_submission_files(all_test_s1_ids, matched_dict, candidate_dict)
    
    print("\nRunning official validator...")
    is_valid = run_official_validator()
    print(f"\nBaseline pipeline finished in {time.time()-t0:.1f}s. Valid: {is_valid}")
    return is_valid


def run_training_and_validation(sample_size: int = 30000):
    """
    Optimized training pipeline:
    1. Loads train data.
    2. Builds stratified train/val split.
    3. Indexes target (S2+S3) ONCE per country and queries both train and val together.
    4. Computes pairwise features only for retrieved candidate pairs.
    5. Trains LightGBM and optimizes threshold for macro F0.5.
    """
    print("\n" + "="*70)
    print(f"TRAINING PIPELINE (Sample Size: {sample_size} entities)")
    print("="*70)
    t0 = time.time()
    
    print("1. Loading training data...")
    s1, s2, s3, gt = load_train_data()
    print(f"Data loaded in {time.time()-t0:.1f}s")
    
    print("2. Creating stratified validation split...")
    train_ids, val_ids = create_validation_split(s1, gt, val_ratio=0.2, seed=42)
    
    np.random.seed(42)
    n_train = int(sample_size * 0.8)
    n_val = int(sample_size * 0.2)
    sampled_train_ids = set(np.random.choice(list(train_ids), min(n_train, len(train_ids)), replace=False))
    sampled_val_ids = set(np.random.choice(list(val_ids), min(n_val, len(val_ids)), replace=False))
    
    sub_s1_train = s1[s1['entity_id'].isin(sampled_train_ids)]
    sub_s1_val = s1[s1['entity_id'].isin(sampled_val_ids)]
    print(f"Sampled train: {len(sub_s1_train)}, val: {len(sub_s1_val)}")
    
    # 3. Country-partitioned blocking (Fit ONCE, query BOTH)
    print("\n3. Running Country-Partitioned Blocking...")
    train_cands = {}
    val_cands = {}
    
    for country in s1['country'].unique():
        t_c = time.time()
        c_tr_s1 = sub_s1_train[sub_s1_train['country'] == country]
        c_val_s1 = sub_s1_val[sub_s1_val['country'] == country]
        c_s2 = s2[s2['country'] == country]
        c_s3 = s3[s3['country'] == country]
        
        print(f"\n--- {country} ---")
        print(f"Fitting blocker on {len(c_s2)} S2 + {len(c_s3)} S3 records...")
        blocker = CountryBlocker(country)
        target_ids = list(c_s2['entity_id']) + list(c_s3['entity_id'])
        target_names = list(c_s2['business_name']) + list(c_s3['business_name'])
        target_addrs = list(c_s2['business_address']) + list(c_s3['business_address'])
        blocker.fit(target_ids, target_names, target_addrs)
        
        print(f"Querying {len(c_tr_s1)} train entities...")
        train_map = blocker.query(
            list(c_tr_s1['entity_id']), list(c_tr_s1['business_name']), list(c_tr_s1['business_address']),
            max_candidates=40
        )
        train_cands.update(train_map)
        
        print(f"Querying {len(c_val_s1)} val entities...")
        val_map = blocker.query(
            list(c_val_s1['entity_id']), list(c_val_s1['business_name']), list(c_val_s1['business_address']),
            max_candidates=40
        )
        val_cands.update(val_map)
        print(f"Completed {country} in {time.time()-t_c:.1f}s")
        
    # 4. Prepare entity features lookup ONLY for needed entities
    print("\n4. Preparing entity features lookup for candidates...")
    needed_ids = set()
    for s1_id, cands in train_cands.items():
        needed_ids.add(s1_id)
        needed_ids.update(cands)
        needed_ids.update(gt.get(s1_id, set()))
    for s1_id, cands in val_cands.items():
        needed_ids.add(s1_id)
        needed_ids.update(cands)
        needed_ids.update(gt.get(s1_id, set()))
        
    print(f"Total unique entities needed for features: {len(needed_ids):,}")
    entity_dict = {}
    for df in [s1, s2, s3]:
        sub_df = df[df['entity_id'].isin(needed_ids)]
        for eid, name, addr in zip(sub_df['entity_id'], sub_df['business_name'], sub_df['business_address']):
            entity_dict[eid] = (normalize_name(name), normalize_address(addr))
            
    # 5. Build pairwise training data
    print("\n5. Generating pairwise training feature matrix...")
    train_pairs = []
    y_train = []
    seen = set()
    for s1_id, cands in train_cands.items():
        true_matches = gt.get(s1_id, set())
        for cid in cands:
            pair = (s1_id, cid)
            if pair not in seen:
                seen.add(pair)
                train_pairs.append(pair)
                y_train.append(1 if cid in true_matches else 0)
        for tid in true_matches:
            if tid in entity_dict:
                pair = (s1_id, tid)
                if pair not in seen:
                    seen.add(pair)
                    train_pairs.append(pair)
                    y_train.append(1)
                
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Train pairs: {len(train_pairs):,} (positives: {(y_train==1).sum():,}, negatives: {(y_train==0).sum():,})")
    
    t_f = time.time()
    X_train = batch_compute_features(train_pairs, entity_dict)
    print(f"Train features computed in {time.time()-t_f:.1f}s. Shape: {X_train.shape}")
    
    # 6. Build pairwise validation data
    print("\n6. Generating pairwise validation feature matrix...")
    val_pairs = []
    for s1_id, cands in val_cands.items():
        for cid in cands:
            val_pairs.append((s1_id, cid))
            
    print(f"Val pairs: {len(val_pairs):,}")
    X_val = batch_compute_features(val_pairs, entity_dict)
    print(f"Val features computed. Shape: {X_val.shape}")
    
    # 7. Train LightGBM Model
    print("\n7. Training LightGBM Model...")
    booster, feat_imp = train_lgbm(X_train, y_train)
    
    print("\n--- TOP 10 FEATURE IMPORTANCES (GAIN) ---")
    for feat, imp in list(feat_imp.items())[:10]:
        print(f"  {feat:<25}: {imp:.1f}")
        
    # 8. Score Validation Pairs & Optimize Threshold
    print("\n8. Scoring Validation Pairs & Optimizing Threshold for Macro F0.5...")
    val_probas = predict_proba(booster, X_val)
    
    val_scored = {s1_id: [] for s1_id in sampled_val_ids}
    for (s1_id, cid), proba in zip(val_pairs, val_probas):
        val_scored[s1_id].append((cid, float(proba)))
        
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in sampled_val_ids}
    best_t, best_f05, hist = optimize_threshold(val_scored, val_gt)
    
    print("\n--- THRESHOLD OPTIMIZATION RESULTS ---")
    for t_val, score in hist.items():
        mark = "  <-- OPTIMAL TAU*" if t_val == best_t else ""
        print(f"  tau = {t_val:.2f} -> Macro F0.5 = {score:.4f}{mark}")
        
    best_preds = apply_threshold(val_scored, best_t)
    
    # Country breakdown
    val_countries = {row['entity_id']: row['country'] for _, row in sub_s1_val.iterrows()}
    metrics = compute_detailed_metrics(best_preds, val_gt, val_countries)
    
    print(f"\n{'='*70}\nVALIDATION PERFORMANCE SUMMARY (tau* = {best_t:.2f})\n{'='*70}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")
            
    print(f"\nEntire training & validation completed in {time.time()-t0:.1f}s")
    return booster, best_t, metrics


def run_full_ml_inference(threshold: float = 0.65):
    """
    Submission 2: End-to-End ML Pipeline.
    1. Loads test sources (US, India, France).
    2. Runs CountryBlocker across each country partition.
    3. Extracts pairwise features using trained LightGBM model.
    4. Predicts match probabilities and applies optimal threshold tau*.
    5. Saves matching_results.tsv and candidate_pairs.tsv.
    6. Runs official validate_submission.py.
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
    
    print("Loading test sources...")
    te1, te2, te3 = load_test_data()
    print(f"Loaded: {len(te1)} S1, {len(te2)} S2, {len(te3)} S3")
    
    matched_dict = {}
    candidate_dict = {}
    
    for country in te1['country'].unique():
        t_c = time.time()
        c_te1 = te1[te1['country'] == country]
        c_te2 = te2[te2['country'] == country]
        c_te3 = te3[te3['country'] == country]
        print(f"\nProcessing {country}: {len(c_te1)} S1, {len(c_te2)} S2, {len(c_te3)} S3...")
        
        # 1. Blocking
        blocker = CountryBlocker(country)
        target_ids = list(c_te2['entity_id']) + list(c_te3['entity_id'])
        target_names = list(c_te2['business_name']) + list(c_te3['business_name'])
        target_addrs = list(c_te2['business_address']) + list(c_te3['business_address'])
        blocker.fit(target_ids, target_names, target_addrs)
        
        s1_ids = list(c_te1['entity_id'])
        s1_names = list(c_te1['business_name'])
        s1_addrs = list(c_te1['business_address'])
        cands_map = blocker.query(s1_ids, s1_names, s1_addrs, max_candidates=35)
        
        # 2. Entity lookup for this country's candidates
        c_needed_ids = set(s1_ids)
        for clist in cands_map.values():
            c_needed_ids.update(clist)
            
        print(f"  Entities needed for features: {len(c_needed_ids):,}")
        entity_dict = {}
        for df in [c_te1, c_te2, c_te3]:
            sub_df = df[df['entity_id'].isin(c_needed_ids)]
            for eid, name, addr in zip(sub_df['entity_id'], sub_df['business_name'], sub_df['business_address']):
                entity_dict[eid] = (normalize_name(name), normalize_address(addr))
                
        # 3. Generate pairwise features in batches
        test_pairs = []
        for s1_id, clist in cands_map.items():
            for cid in clist:
                test_pairs.append((s1_id, cid))
                
        print(f"  Candidate pairs to score: {len(test_pairs):,}...")
        X_test = batch_compute_features(test_pairs, entity_dict)
        
        # 4. Predict probabilities
        probas = predict_proba(booster, X_test)
        
        # 5. Apply threshold
        pair_idx = 0
        for s1_id in s1_ids:
            clist = cands_map.get(s1_id, [])
            candidate_dict[s1_id] = clist
            matches = set()
            for cid in clist:
                prob = probas[pair_idx]
                if prob >= threshold:
                    matches.add(cid)
                pair_idx += 1
            matched_dict[s1_id] = matches
            
        print(f"  Completed {country} in {time.time()-t_c:.1f}s")
        
    # Write submission files
    all_test_s1_ids = list(te1['entity_id'])
    save_submission_files(all_test_s1_ids, matched_dict, candidate_dict)
    
    # Run official validator
    print("\nRunning official validator...")
    is_valid = run_official_validator()
    print(f"\nML inference pipeline finished in {time.time()-t0:.1f}s. Valid: {is_valid}")
    return is_valid
