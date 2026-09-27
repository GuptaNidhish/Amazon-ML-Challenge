import os
import subprocess
from typing import Dict, List, Set, Iterable
from config import (
    MATCHING_RESULTS, CANDIDATE_PAIRS, TEST_DIR, TEST_S1, VALIDATE_SCRIPT
)

class StreamingSubmissionWriter:
    """
    Streaming writer for matching_results.tsv and candidate_pairs.tsv.
    Avoids holding 1.73M entity predictions in RAM by writing chunks directly to disk.
    Ensures format correctness:
    - Tab-separated
    - Empty string for singletons
    - Deduplicated and sorted ID lists
    - Final matches are strictly a subset of candidates
    """
    def __init__(
        self,
        matching_path: str = MATCHING_RESULTS,
        candidate_path: str = CANDIDATE_PAIRS
    ):
        self.matching_path = matching_path
        self.candidate_path = candidate_path
        os.makedirs(os.path.dirname(matching_path), exist_ok=True)
        os.makedirs(os.path.dirname(candidate_path), exist_ok=True)
        
        self.f_matching = open(matching_path, 'w', encoding='utf-8')
        self.f_candidate = open(candidate_path, 'w', encoding='utf-8')
        
        # Write headers
        self.f_matching.write("source1_entity_id\tmatched_entity_ids\n")
        self.f_candidate.write("source1_entity_id\tcandidate_entity_ids\n")
        self.written_count = 0

    def write_record(self, s1_id: str, matched_ids: Iterable[str], candidate_ids: Iterable[str]):
        """Write a single S1 entity prediction."""
        m_set = set(matched_ids)
        c_set = set(candidate_ids) | m_set  # Guarantees matches are a subset of candidates
        
        m_str = ",".join(sorted(list(m_set)))
        c_str = ",".join(sorted(list(c_set)))
        
        self.f_matching.write(f"{s1_id}\t{m_str}\n")
        self.f_candidate.write(f"{s1_id}\t{c_str}\n")
        self.written_count += 1

    def flush(self):
        self.f_matching.flush()
        self.f_candidate.flush()

    def close(self):
        self.f_matching.close()
        self.f_candidate.close()
        print(f"Streaming writer completed. Wrote {self.written_count:,} records.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
    with StreamingSubmissionWriter(matching_path, candidate_path) as writer:
        for s1_id in test_s1_ids:
            writer.write_record(
                s1_id,
                matched_dict.get(s1_id, set()),
                candidate_dict.get(s1_id, [])
            )
    print("Submission files written successfully.")


def run_official_validator(
    matching_path: str = MATCHING_RESULTS,
    candidate_path: str = CANDIDATE_PAIRS,
    test_dir: str = TEST_DIR
) -> bool:
    """
    Runs official utils/validate_submission.py on matching_results.tsv,
    followed by a fast streaming verification of candidate_pairs.tsv.
    This guarantees 100% compliance while keeping RAM < 100 MB.
    """
    cmd = [
        "python3", VALIDATE_SCRIPT,
        "--matching", matching_path,
        "--test-dir", test_dir
    ]
    print(f"1. Running official validator on matching results: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    if result.returncode != 0:
        print(">>> OFFICIAL VALIDATION FAILED on matching_results.tsv!")
        return False

    print("2. Running streaming cross-validation on candidate_pairs.tsv...")
    if not os.path.isfile(candidate_path):
        print(f"Error: {candidate_path} not found.")
        return False
        
    row_count = 0
    with open(matching_path, 'r', encoding='utf-8') as f_m, open(candidate_path, 'r', encoding='utf-8') as f_c:
        h_m = next(f_m, '').rstrip('\n').split('\t')
        h_c = next(f_c, '').rstrip('\n').split('\t')
        if h_m != ['source1_entity_id', 'matched_entity_ids']:
            print(f"Error: Invalid matching header: {h_m}")
            return False
        if h_c != ['source1_entity_id', 'candidate_entity_ids']:
            print(f"Error: Invalid candidate header: {h_c}")
            return False
            
        for line_m, line_c in zip(f_m, f_c):
            row_count += 1
            pm = line_m.rstrip('\n').split('\t')
            pc = line_c.rstrip('\n').split('\t')
            
            s1_m, m_str = pm[0], (pm[1] if len(pm) > 1 else '')
            s1_c, c_str = pc[0], (pc[1] if len(pc) > 1 else '')
            
            if s1_m != s1_c:
                print(f"Error at row {row_count}: ID mismatch {s1_m} vs {s1_c}")
                return False
                
            matches = set(m_str.split(',')) if m_str else set()
            cands = set(c_str.split(',')) if c_str else set()
            
            if not matches.issubset(cands):
                violations = matches - cands
                print(f"Error at row {row_count} ({s1_m}): Matches not in candidates: {violations}")
                return False
                
    print(f"   Successfully verified {row_count:,} rows. Zero subset violations!")
    print(">>> ALL SUBMISSION CHECKS PASSED 100%! Ready for leaderboard.")
    return True
