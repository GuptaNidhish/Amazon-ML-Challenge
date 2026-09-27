import numpy as np
from typing import Dict, List, Tuple
from rapidfuzz import fuzz

FEATURE_NAMES = [
    'name_exact',
    'name_compact_exact',
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_token_jaccard',
    'name_common_tokens',
    'name_first_token_match',
    'name_len_diff_ratio',
    'addr_exact',
    'addr_ratio',
    'addr_partial_ratio',
    'addr_token_sort_ratio',
    'addr_token_set_ratio',
    'addr_token_jaccard',
    'addr_common_tokens',
    'addr_num_jaccard',
    'addr_shared_numbers',
    'addr_is_empty',
    'name_addr_mult',
    'name_addr_max',
    'name_addr_min',
    'name_strong_addr_weak',
    'name_weak_addr_strong',
    'both_strong',
    'source_is_s3'
]


def compute_pair_features(
    norm_n1: str,
    norm_a1: str,
    norm_n2: str,
    norm_a2: str,
    cand_id: str
) -> List[float]:
    """
    Computes 27 fast C++ pairwise features between an S1 entity and an S2/S3 candidate.
    """
    w1 = norm_n1.split()
    w2 = norm_n2.split()
    s1_w = set(w1)
    s2_w = set(w2)
    
    # 1. Name features
    name_exact = 1.0 if (norm_n1 == norm_n2 and norm_n1) else 0.0
    c1 = norm_n1.replace(' ', '')
    c2 = norm_n2.replace(' ', '')
    name_compact_exact = 1.0 if (c1 == c2 and c1) else 0.0
    
    n_ratio = fuzz.ratio(norm_n1, norm_n2) / 100.0
    n_partial = fuzz.partial_ratio(norm_n1, norm_n2) / 100.0
    n_sort = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0
    n_set = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0
    
    name_jaccard = len(s1_w & s2_w) / len(s1_w | s2_w) if (s1_w | s2_w) else 0.0
    name_common = float(len(s1_w & s2_w))
    name_first_match = 1.0 if (w1 and w2 and w1[0] == w2[0]) else 0.0
    len_max = max(len(norm_n1), len(norm_n2), 1)
    name_len_diff = abs(len(norm_n1) - len(norm_n2)) / len_max
    
    # 2. Address features
    aw1 = norm_a1.split()
    aw2 = norm_a2.split()
    s1_aw = set(aw1)
    s2_aw = set(aw2)
    
    addr_empty = 1.0 if not norm_a2 else 0.0
    addr_exact = 1.0 if (norm_a1 == norm_a2 and norm_a1) else 0.0
    
    if addr_empty:
        a_ratio = 0.0
        a_partial = 0.0
        a_sort = 0.0
        a_set = 0.0
        addr_jaccard = 0.0
        addr_common = 0.0
        num_jaccard = 0.0
        num_common = 0.0
    else:
        a_ratio = fuzz.ratio(norm_a1, norm_a2) / 100.0
        a_partial = fuzz.partial_ratio(norm_a1, norm_a2) / 100.0
        a_sort = fuzz.token_sort_ratio(norm_a1, norm_a2) / 100.0
        a_set = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0
        
        addr_jaccard = len(s1_aw & s2_aw) / len(s1_aw | s2_aw) if (s1_aw | s2_aw) else 0.0
        addr_common = float(len(s1_aw & s2_aw))
        
        nums1 = {x for x in aw1 if x.isdigit()}
        nums2 = {x for x in aw2 if x.isdigit()}
        num_jaccard = len(nums1 & nums2) / len(nums1 | nums2) if (nums1 | nums2) else 0.0
        num_common = float(len(nums1 & nums2))
        
    # 3. Cross-field interaction features
    mult = n_sort * a_sort
    max_sim = max(n_sort, a_sort)
    min_sim = min(n_sort, a_sort)
    strong_n_weak_a = 1.0 if (n_sort >= 0.85 and a_sort < 0.30) else 0.0
    weak_n_strong_a = 1.0 if (n_sort < 0.40 and a_sort >= 0.75) else 0.0
    both_strong = 1.0 if (n_sort >= 0.75 and a_sort >= 0.60) else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3') else 0.0
    
    return [
        name_exact,
        name_compact_exact,
        n_ratio,
        n_partial,
        n_sort,
        n_set,
        name_jaccard,
        name_common,
        name_first_match,
        name_len_diff,
        addr_exact,
        a_ratio,
        a_partial,
        a_sort,
        a_set,
        addr_jaccard,
        addr_common,
        num_jaccard,
        num_common,
        addr_empty,
        mult,
        max_sim,
        min_sim,
        strong_n_weak_a,
        weak_n_strong_a,
        both_strong,
        is_s3
    ]


def batch_compute_features(
    candidate_pairs: List[Tuple[str, str]],
    entity_dict: Dict[str, Tuple[str, str]]
) -> np.ndarray:
    """
    Computes feature matrix for a list of (s1_id, cand_id) pairs.
    entity_dict maps entity_id -> (norm_name, norm_address).
    """
    n_pairs = len(candidate_pairs)
    if n_pairs == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        
    X = np.empty((n_pairs, len(FEATURE_NAMES)), dtype=np.float32)
    for i, (s1_id, cand_id) in enumerate(candidate_pairs):
        n1, a1 = entity_dict.get(s1_id, ('', ''))
        n2, a2 = entity_dict.get(cand_id, ('', ''))
        X[i] = compute_pair_features(n1, a1, n2, a2, cand_id)
        
    return X
