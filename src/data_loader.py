import os
import pandas as pd
from typing import Dict, Set, Tuple, Iterator
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, TEST_S1, TEST_S2, TEST_S3

def load_source_df(path: str) -> pd.DataFrame:
    """
    Load a source TSV file into pandas DataFrame.
    Returns DataFrame with columns ['entity_id', 'business_name', 'business_address', 'country']
    with missing values filled as empty strings.
    """
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    for col in ['entity_id', 'business_name', 'business_address', 'country']:
        if col not in df.columns:
            df[col] = ''
    return df

def load_ground_truth(path: str = TRAIN_GT) -> Dict[str, Set[str]]:
    """
    Load ground truth TSV into a mapping: {s1_id: set of matched s2/s3 ids}.
    Singletons map to empty set.
    """
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    gt_dict = {}
    for s1_id, matched in zip(df['source1_entity_id'], df['matched_entity_ids']):
        m = matched.strip()
        if not m:
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = {x.strip() for x in m.split(',') if x.strip()}
    return gt_dict

def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Load train sources 1, 2, 3 and train ground truth."""
    s1 = load_source_df(TRAIN_S1)
    s2 = load_source_df(TRAIN_S2)
    s3 = load_source_df(TRAIN_S3)
    gt = load_ground_truth(TRAIN_GT)
    return s1, s2, s3, gt

def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load test sources 1, 2, 3."""
    s1 = load_source_df(TEST_S1)
    s2 = load_source_df(TEST_S2)
    s3 = load_source_df(TEST_S3)
    return s1, s2, s3

def stream_source_file_by_country(path: str, target_country: str) -> Iterator[Tuple[str, str, str]]:
    """
    Streams records (entity_id, business_name, business_address) for a given country
    directly from disk with ~0 RAM overhead.
    """
    with open(path, 'r', encoding='utf-8') as f:
        next(f, None)  # Skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4 and parts[3] == target_country:
                yield parts[0], parts[1], parts[2]

def stream_all_s1_records(path: str = TEST_S1) -> Iterator[Tuple[str, str, str, str]]:
    """
    Streams all S1 records (entity_id, business_name, business_address, country) in file order.
    """
    with open(path, 'r', encoding='utf-8') as f:
        next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4:
                yield parts[0], parts[1], parts[2], parts[3]
