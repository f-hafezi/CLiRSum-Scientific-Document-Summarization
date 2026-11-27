"""Configuration for the BART multi-task training project.

Set paths through environment variables or edit the defaults below.
"""
import os

MODEL_PATH = os.getenv("BART_MODEL_PATH", "./pretrained/bart_large")
DATASET_PATH = os.getenv("BART_DATASET_PATH", "./data/train_dataset.csv")
OUTPUT_DIR = os.getenv("BART_OUTPUT_DIR", "./outputs")

SEED = 42
EVAL_SIZE = 0.10

# Stage 1: single-task warm-up (Task 0)
STAGE1_EPOCHS = 4
STAGE1_BATCH_SIZE = 32
STAGE1_WARMUP_STEPS = 200

# Stage 2: multi-task training
STAGE2_EPOCHS = 5
STAGE2_BATCH_SIZE = 48
STAGE2_WARMUP_STEPS = 500

GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 3
FP16 = True
LR_SCHEDULER_TYPE = "cosine"

TASK_PREFIXES = {
    0: "generate_introduction: ",
    1: "generate_abstract: ",
    2: "generate_abstract_from_intro: ",
    3: "generate_title: ",
}
INPUT_MAX_LENGTHS = {0: 1024, 1: 1024, 2: 512, 3: 256}
TARGET_MAX_LENGTHS = {0: 512, 1: 256, 2: 256, 3: 32}
# Initial weights for dynamic task weighting
TASK_WEIGHTS = {
    0: 0.25,
    1: 0.25,
    2: 0.25,
    3: 0.25
}

DYNAMIC_TEMPERATURE = 2.0
DYNAMIC_MOMENTUM = 0.9
TASK_LOSS_HISTORY_SIZE = 20

GENERATION_CONFIG = {
    "max_length": 512,
    "min_length": 32,
    "num_beams": 8,
    "length_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "forced_bos_token_id": 0,
    "forced_eos_token_id": 2,
}
