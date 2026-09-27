import os
import subprocess
import pandas as pd
from typing import Dict, List, Set
from config import (
    MATCHING_RESULTS, CANDIDATE_PAIRS, TEST_DIR, TEST_S1, VALIDATE_SCRIPT
)

def save_submission_files(
    test_s1_ids: List[str],
    matched_dict: Dict[str, Set[str]],
    candidate_dict: Dict[str, List[str]],
    matching_path: str = MATCHING_RESULTS,
    candidate_path: str = CANDIDATE_PAIRS
):
    """
    Writes matching_results.tsv and candidate_pairs.tsv with exact format verification.
    """
    os.makedirs(os.path.dirname(matching_path), exist_ok=True)
    os.makedirs(os.path.dirname(candidate_path), exist_ok=True)

    print(f"Writing {matching_path} for {len(test_s1_ids)} test S1 entities...")
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1_ids:
            matches = matched_dict.get(s1_id, set())
            # Ensure matched IDs are sorted, deduplicated, and clean
            m_str = ",".join(sorted(list(set(matches))))
            f.write(f"{s1_id}\t{m_str}\n")

    print(f"Writing {candidate_path}...")
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in test_s1_ids:
            cands = candidate_dict.get(s1_id, [])
            # Must ensure final matches are a subset of candidates
            all_cands = set(cands) | matched_dict.get(s1_id, set())
            c_str = ",".join(sorted(list(all_cands)))
            f.write(f"{s1_id}\t{c_str}\n")

    print("Submission files written successfully.")


def run_official_validator(
    matching_path: str = MATCHING_RESULTS,
    candidate_path: str = CANDIDATE_PAIRS,
    test_dir: str = TEST_DIR
) -> bool:
    """
    Runs official utils/validate_submission.py to ensure zero formatting issues.
    """
    cmd = [
        "python3", VALIDATE_SCRIPT,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]
    print(f"Running validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    if result.returncode == 0:
        print(">>> OFFICIAL VALIDATION PASSED! Safe to submit.")
        return True
    else:
        print(">>> VALIDATION FAILED! Issues need to be fixed.")
        return False
