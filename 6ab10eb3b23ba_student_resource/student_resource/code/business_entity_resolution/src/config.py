import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# File paths
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

OUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CANDIDATES = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "lgb_matcher.txt")

# Blocking configuration
MAX_CANDIDATES_PER_ENTITY = 12
MAX_KEY_BUCKET_SIZE = 400

# Optimal inference threshold calibrated on validation set for Macro F_0.5
DEFAULT_THRESHOLD = 0.88
