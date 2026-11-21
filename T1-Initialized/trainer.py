"""Custom Hugging Face Trainer for task-weighted multi-task loss."""
from transformers import Trainer
from .losses import MultiTaskLoss


class CustomTrainer(Trainer):
    def __init__(
        self,
        task_weights,
        pad_token_id,
        target_max_lengths,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.custom_loss = MultiTaskLoss(
            task_weights=task_weights,
            pad_token_id=pad_token_id,
            target_max_lengths=target_max_lengths,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = inputs.copy()
        labels = inputs.pop("labels")
        task_ids = inputs.pop("task_id")
        outputs = model(**inputs, labels=labels)
        loss = self.custom_loss(outputs, labels, task_ids)
        return (loss, outputs) if return_outputs else loss
