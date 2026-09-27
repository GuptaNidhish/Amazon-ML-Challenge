import os

# Base paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, 'student_resource', 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
EXP_DIR = os.path.join(PROJECT_DIR, 'experiments')

# Specific file paths
TRAIN_S1 = os.path.join(TRAIN_DIR, 'train_source1.tsv')
TRAIN_S2 = os.path.join(TRAIN_DIR, 'train_source2.tsv')
TRAIN_S3 = os.path.join(TRAIN_DIR, 'train_source3.tsv')
TRAIN_GT = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

TEST_S1 = os.path.join(TEST_DIR, 'test_source1.tsv')
TEST_S2 = os.path.join(TEST_DIR, 'test_source2.tsv')
TEST_S3 = os.path.join(TEST_DIR, 'test_source3.tsv')

# Submission files
SUBMISSION_MATCHING = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
SUBMISSION_CANDIDATE = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
MATCHING_RESULTS = SUBMISSION_MATCHING
CANDIDATE_PAIRS = SUBMISSION_CANDIDATE

# Validation script
VALIDATE_SCRIPT = os.path.join(PROJECT_DIR, 'student_resource', 'utils', 'validate_submission.py')

# Ensure directories exist
for d in [OUTPUT_DIR, MODEL_DIR, EXP_DIR]:
    os.makedirs(d, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# LightGBM Default Parameters
LGBM_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'n_estimators': 1500,
    'verbosity': -1,
    'n_jobs': -1,
    'random_state': RANDOM_SEED,
}
