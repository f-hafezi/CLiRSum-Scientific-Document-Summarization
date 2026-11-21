"""Evaluate the saved model's validation loss on the configured CSV split.

This script reports the held-out validation loss. It does not calculate ROUGE/BLEU,
because the supplied source code did not define generation-based evaluation metrics.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments

from bart_multitask.config import (
    MODEL_PATH, DATASET_PATH, OUTPUT_DIR, SEED, EVAL_SIZE,
    TASK_WEIGHTS, TARGET_MAX_LENGTHS,
)
from bart_multitask.datasets import MultiTaskDataset
from bart_multitask.collator import SimpleDataCollator
from bart_multitask.modeling import load_model_and_tokenizer
from bart_multitask.trainer import CustomTrainer


def main():
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Set BART_DATASET_PATH."
        )
    model_dir = os.path.join(OUTPUT_DIR, "final_model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Trained model not found: {model_dir}")

    model, tokenizer = load_model_and_tokenizer(model_dir)
    data = pd.read_csv(DATASET_PATH)
    indices = list(range(len(data)))
    _, eval_indices = train_test_split(
        indices, test_size=EVAL_SIZE, random_state=SEED
    )
    eval_dataset = MultiTaskDataset(DATASET_PATH, tokenizer, eval_indices)

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "evaluation"),
        per_device_eval_batch_size=8,
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
    )
    trainer = CustomTrainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        data_collator=SimpleDataCollator(tokenizer),
        task_weights=TASK_WEIGHTS,
        pad_token_id=tokenizer.pad_token_id,
        target_max_lengths=TARGET_MAX_LENGTHS,
    )
    results = trainer.evaluate()
    print("Validation results:", results)


if __name__ == "__main__":
    main()
