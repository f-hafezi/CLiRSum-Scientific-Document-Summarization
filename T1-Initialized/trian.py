"""Run Stage 1 task-0 warm-up followed by Stage 2 multi-task training."""
import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, GenerationConfig

from bart_multitask.config import (
    MODEL_PATH, DATASET_PATH, OUTPUT_DIR, SEED, EVAL_SIZE,
    STAGE1_EPOCHS, STAGE1_BATCH_SIZE, STAGE1_WARMUP_STEPS,
    STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_WARMUP_STEPS,
    GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, WEIGHT_DECAY,
    LOGGING_STEPS, SAVE_TOTAL_LIMIT, FP16, LR_SCHEDULER_TYPE,
    TASK_WEIGHTS, TARGET_MAX_LENGTHS, GENERATION_CONFIG,
)
from bart_multitask.datasets import MultiTaskDataset, SingleTaskDataset
from bart_multitask.collator import SimpleDataCollator
from bart_multitask.modeling import load_model_and_tokenizer
from bart_multitask.trainer import CustomTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None
    checkpoints = [
        name for name in os.listdir(directory)
        if name.startswith("checkpoint-")
        and name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda name: int(name.split("-")[-1]))
    return os.path.join(directory, latest)


def build_training_args(output_dir, epochs, batch_size, warmup_steps, save_limit):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=warmup_steps,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=save_limit,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=FP16 and torch.cuda.is_available(),
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )


def main():
    set_seed(SEED)
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Set BART_DATASET_PATH."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    model.gradient_checkpointing_enable()

    # Keep the same article-level 90/10 split for both training stages.
    import pandas as pd
    data = pd.read_csv(DATASET_PATH)
    all_indices = list(range(len(data)))
    train_indices, eval_indices = train_test_split(
        all_indices,
        test_size=EVAL_SIZE,
        random_state=SEED,
    )

    train_dataset = MultiTaskDataset(DATASET_PATH, tokenizer, train_indices)
    eval_dataset = MultiTaskDataset(DATASET_PATH, tokenizer, eval_indices)
    task0_train = SingleTaskDataset(
        DATASET_PATH, tokenizer, task_id=0, indices=train_indices
    )
    task0_eval = SingleTaskDataset(
        DATASET_PATH, tokenizer, task_id=0, indices=eval_indices
    )
    collator = SimpleDataCollator(tokenizer)

    # Stage 1: warm-up on Task 0 (introduction generation).
    stage1_dir = os.path.join(OUTPUT_DIR, "stage1_task0")
    stage1_trainer = CustomTrainer(
        model=model,
        args=build_training_args(
            stage1_dir, STAGE1_EPOCHS, STAGE1_BATCH_SIZE,
            STAGE1_WARMUP_STEPS, 1,
        ),
        train_dataset=task0_train,
        eval_dataset=task0_eval,
        data_collator=collator,
        task_weights=TASK_WEIGHTS,
        pad_token_id=tokenizer.pad_token_id,
        target_max_lengths=TARGET_MAX_LENGTHS,
    )
    stage1_resume = latest_checkpoint(stage1_dir)
    print("Stage 1: training Task 0...")
    stage1_trainer.train(resume_from_checkpoint=stage1_resume)

    # Stage 2: train jointly on all four tasks.
    stage2_dir = os.path.join(OUTPUT_DIR, "stage2_multitask")
    trainer = CustomTrainer(
        model=model,
        args=build_training_args(
            stage2_dir, STAGE2_EPOCHS, STAGE2_BATCH_SIZE,
            STAGE2_WARMUP_STEPS, SAVE_TOTAL_LIMIT,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        task_weights=TASK_WEIGHTS,
        pad_token_id=tokenizer.pad_token_id,
        target_max_lengths=TARGET_MAX_LENGTHS,
    )
    stage2_resume = latest_checkpoint(stage2_dir)
    print("Stage 2: multi-task training...")
    trainer.train(resume_from_checkpoint=stage2_resume)

    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    GenerationConfig(**GENERATION_CONFIG).save_pretrained(final_dir)
    print(f"Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
