import torch
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, GenerationConfig
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import evaluate

from dataset import MultiTaskDataset, SingleTaskDataset
from trainer import CustomTrainer
from collator import SimpleDataCollator

# Load model and tokenizer
model_path = "E:/../bart_large"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path, use_cache=False)
model.to("cuda")
model.gradient_checkpointing_enable()

# Load dataset
dataset_path = "train_dataset_article_text_have_intro_20000/train_dataset_article_text_have_intro_20000.csv"
data = pd.read_csv(dataset_path)
all_indices = list(range(len(data)))
train_indices, eval_indices = train_test_split(all_indices, test_size=0.1, random_state=42)

# Datasets
train_dataset = MultiTaskDataset(dataset_path, tokenizer, train_indices)
eval_dataset = MultiTaskDataset(dataset_path, tokenizer, eval_indices)

task0_train_dataset = SingleTaskDataset(dataset_path, tokenizer, task_id=0, indices=train_indices)
task0_eval_dataset = SingleTaskDataset(dataset_path, tokenizer, task_id=0, indices=eval_indices)

# Collator
data_collator = SimpleDataCollator(tokenizer, max_input_length=1024, max_target_length=512)

# Checkpoint
checkpoint_dir = "model_seri1/model_t1t2_6/model/2end_multi_task_bart_final_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

task_weights = {0: 2.0, 1: 1.5, 2: 1.0, 3: 0.5}
target_max_lengths = {0: 512, 1: 256, 2: 256, 3: 32}

# ---- Stage 1 ----
print("Starting training for task 0")

task0_trainer = CustomTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_steps=300,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        report_to="none",
        fp16=True,
        lr_scheduler_type="cosine",
        dataloader_drop_last=True,
        remove_unused_columns=False,
    ),
    train_dataset=task0_train_dataset,
    eval_dataset=task0_eval_dataset,
    data_collator=data_collator,
    task_weights=task_weights,
    pad_token_id=tokenizer.pad_token_id,
    target_max_lengths=target_max_lengths
)

task0_trainer.train()

# Save
final_model_path = "model_seri1/model_t1t2_6/model/model"
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print("DONE")
