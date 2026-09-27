import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

def compute_macro_f05(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> float:
    """
    Compute macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    
    Singletons:
      - true empty, pred empty: 1.0
      - true empty, pred non-empty: 0.0
      - true non-empty, pred empty: 0.0
    Non-singletons:
      - standard F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    total_score = 0.0
    n = len(ground_truth)
    if n == 0:
        return 0.0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        len_t = len(true_set)
        len_p = len(pred_set)

        if len_t == 0:
            if len_p == 0:
                total_score += 1.0
            # else: 0.0
        elif len_p == 0:
            # missed all true matches
            pass # 0.0
        else:
            tp = len(true_set & pred_set)
            if tp == 0:
                pass # 0.0
            else:
                precision = tp / len_p
                recall = tp / len_t
                denom = 0.25 * precision + recall
                if denom > 0:
                    total_score += (1.25 * precision * recall) / denom

    return total_score / n


def compute_detailed_metrics(
    predictions: Dict[str, Set[str]], 
    ground_truth: Dict[str, Set[str]], 
    s1_countries: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
    """
    Returns breakdown of performance:
    - macro_f05
    - micro precision, recall, f05
    - singleton accuracy
    - per-country macro_f05 (if s1_countries provided)
    """
    n = len(ground_truth)
    if n == 0:
        return {}

    total_f05 = 0.0
    singleton_total = 0
    singleton_correct = 0
    
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    country_scores = defaultdict(list)

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        len_t = len(true_set)
        len_p = len(pred_set)
        score = 0.0

        if len_t == 0:
            singleton_total += 1
            if len_p == 0:
                score = 1.0
                singleton_correct += 1
            else:
                micro_fp += len_p
        elif len_p == 0:
            micro_fn += len_t
        else:
            tp = len(true_set & pred_set)
            fp = len_p - tp
            fn = len_t - tp
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            if tp > 0:
                p = tp / len_p
                r = tp / len_t
                denom = 0.25 * p + r
                if denom > 0:
                    score = (1.25 * p * r) / denom

        total_f05 += score
        if s1_countries and s1_id in s1_countries:
            country_scores[s1_countries[s1_id]].append(score)

    macro_f05 = total_f05 / n
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_denom = 0.25 * micro_p + micro_r
    micro_f05 = (1.25 * micro_p * micro_r) / micro_denom if micro_denom > 0 else 0.0

    metrics = {
        'macro_f05': macro_f05,
        'singleton_accuracy': singleton_acc,
        'singleton_count': singleton_total,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f05': micro_f05,
        'total_evaluated': n
    }

    if s1_countries:
        for c, scores in country_scores.items():
            metrics[f'macro_f05_{c}'] = float(np.mean(scores))

    return metrics


def evaluate_blocking_recall(
    candidate_dict: Dict[str, Set[str]], 
    ground_truth: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Evaluates blocking quality:
    - true match recall: what fraction of true matches are in the candidate set
    - entity recall: what fraction of entities have ALL true matches in candidate set
    - mean candidates per entity
    """
    total_true = 0
    found_true = 0
    all_found_entities = 0
    non_singleton_entities = 0
    cand_counts = []

    for s1_id, true_set in ground_truth.items():
        cands = candidate_dict.get(s1_id, set())
        cand_counts.append(len(cands))
        if len(true_set) > 0:
            non_singleton_entities += 1
            total_true += len(true_set)
            found = len(true_set & cands)
            found_true += found
            if found == len(true_set):
                all_found_entities += 1

    return {
        'candidate_recall': found_true / total_true if total_true > 0 else 1.0,
        'all_matches_found_rate': all_found_entities / non_singleton_entities if non_singleton_entities > 0 else 1.0,
        'mean_candidates': float(np.mean(cand_counts)) if cand_counts else 0.0,
        'max_candidates': int(np.max(cand_counts)) if cand_counts else 0,
        'total_candidates': sum(cand_counts)
    }


def create_validation_split(
    s1_df: pd.DataFrame, 
    gt_dict: Dict[str, Set[str]], 
    val_ratio: float = 0.2, 
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Stratified train/validation split by country × singleton status.
    Returns (train_s1_ids, val_s1_ids).
    """
    np.random.seed(seed)
    
    # Group entity IDs by stratification bucket
    strata = defaultdict(list)
    for _, row in s1_df[['entity_id', 'country']].iterrows():
        s1_id = row['entity_id']
        country = row['country']
        is_singleton = int(len(gt_dict.get(s1_id, set())) == 0)
        strata[(country, is_singleton)].append(s1_id)

    train_ids = set()
    val_ids = set()

    for key, ids in strata.items():
        np.random.shuffle(ids)
        split_idx = int(len(ids) * (1 - val_ratio))
        train_ids.update(ids[:split_idx])
        val_ids.update(ids[split_idx:])

    return train_ids, val_ids
