"""Task-weighted loss used by the custom Trainer."""
import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights, pad_token_id, target_max_lengths):
        super().__init__()
        self.task_weights = task_weights
        self.pad_token_id = pad_token_id
        self.target_max_lengths = target_max_lengths
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, outputs, labels, task_ids):
        total_loss = outputs.logits.new_zeros(())
        batch_size = labels.size(0)

        for i in range(batch_size):
            task_id = int(task_ids[i].item())
            max_length = self.target_max_lengths[task_id]
            sample_logits = outputs.logits[i, :max_length, :].contiguous().view(
                -1, outputs.logits.size(-1)
            )
            sample_labels = labels[i, :max_length].contiguous().view(-1)
            sample_loss = self.loss_fn(sample_logits, sample_labels)
            total_loss = total_loss + self.task_weights.get(task_id, 1.0) * sample_loss

        return total_loss / max(batch_size, 1)
