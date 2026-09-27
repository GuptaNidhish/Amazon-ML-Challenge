import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from evaluation import compute_macro_f05, compute_detailed_metrics

def optimize_threshold(
    scored_candidates: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None
) -> Tuple[float, float, Dict[float, float]]:
    """
    Finds the exact decision threshold tau* that maximizes macro-averaged F_0.5.
    Returns: (best_threshold, best_f05, all_threshold_scores)
    """
    if thresholds is None:
        thresholds = list(np.arange(0.30, 0.96, 0.05))

    best_thresh = 0.50
    best_score = -1.0
    history = {}

    for t in thresholds:
        preds = {}
        for s1_id, cands in scored_candidates.items():
            matches = {cid for cid, score in cands if score >= t}
            preds[s1_id] = matches
            
        score = compute_macro_f05(preds, ground_truth)
        history[float(round(t, 2))] = float(score)
        if score > best_score:
            best_score = score
            best_thresh = float(round(t, 2))

    return best_thresh, best_score, history


def apply_threshold(
    scored_candidates: Dict[str, List[Tuple[str, float]]],
    threshold: float
) -> Dict[str, Set[str]]:
    """
    Applies decision threshold tau to candidate probability scores.
    Singletons are represented as empty set.
    """
    predictions = {}
    for s1_id, cands in scored_candidates.items():
        predictions[s1_id] = {cid for cid, score in cands if score >= threshold}
    return predictions
