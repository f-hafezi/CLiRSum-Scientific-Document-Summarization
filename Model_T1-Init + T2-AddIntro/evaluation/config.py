import os

MODEL_PATH = os.getenv("BART_MODEL_PATH", "./models/t1_init_t2_addintro_multitask")
DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "./data/test_dataset.csv")
OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", "./results/t1_init_t2_addintro_multitask")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
PAD_TO_MULTIPLE_OF = 8
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

TASK_PREFIXES = {
    0: "generate_introduction: ",
    1: "generate_abstract: ",
    2: "generate_abstract_from_intro: ",
    3: "generate_title: ",
}
INPUT_MAX_LENGTHS = {0: 1024, 1: 1024, 2: 512, 3: 256}
TARGET_MAX_LENGTHS = {0: 512, 1: 256, 2: 256, 3: 32}

GENERATION_CONFIGS = {
    0: dict(max_length=512, min_length=250, num_beams=4, length_penalty=1.5,
            no_repeat_ngram_size=3, early_stopping=True, use_cache=True),
    1: dict(max_length=256, min_length=150, num_beams=4, length_penalty=1.5,
            no_repeat_ngram_size=3, early_stopping=True, use_cache=True),
    2: dict(max_length=256, min_length=150, num_beams=4, length_penalty=1.5,
            no_repeat_ngram_size=3, early_stopping=True, use_cache=True),
    3: dict(max_length=32, min_length=10, num_beams=4, length_penalty=1.0,
            no_repeat_ngram_size=2, early_stopping=True, use_cache=True),
}

TASK_NAMES = {
    0: "Task 0 - Article to Introduction",
    1: "Task 1 - Full Article to Abstract",
    2: "Task 2 - Introduction to Abstract",
    3: "Task 3 - Abstract to Title",
}
REQUIRED_COLUMNS = [
    "article_text_no_intro", "introduction_text", "article_text",
    "abstract_text", "title",
]
