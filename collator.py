"""Batch collation for task-specific sequence lengths."""
import torch


class SimpleDataCollator:
    def __init__(self, tokenizer, max_input_length=1024, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, features):
        input_ids = [
            torch.nn.functional.pad(
                f["input_ids"],
                (0, self.max_input_length - f["input_ids"].size(0)),
                value=self.tokenizer.pad_token_id,
            )
            for f in features
        ]
        attention_masks = [
            torch.nn.functional.pad(
                f["attention_mask"],
                (0, self.max_input_length - f["attention_mask"].size(0)),
                value=0,
            )
            for f in features
        ]
        labels = [
            torch.nn.functional.pad(
                f["labels"],
                (0, self.max_target_length - f["labels"].size(0)),
                value=self.tokenizer.pad_token_id,
            )
            for f in features
        ]

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "task_id": torch.stack([f["task_id"] for f in features]),
        }
