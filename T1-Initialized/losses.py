
"""Dynamic multi-task loss computation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        task_weights,
        pad_token_id,
        target_max_lengths
    ):
        super().__init__()

        self.task_weights = task_weights or {}
        self.pad_token_id = pad_token_id
        self.target_max_lengths = target_max_lengths

    def compute_task_losses(self, outputs, labels, task_ids):

        logits = outputs.logits
        batch_size = labels.size(0)

        sample_losses = []

        for i in range(batch_size):

            task_id = int(task_ids[i].item())

            max_length = self.target_max_lengths[task_id]

            current_length = min(
                max_length,
                labels.size(1),
                logits.size(1)
            )

            sample_logits = logits[i, :current_length, :]
            sample_labels = labels[i, :current_length]

            loss_sum = F.cross_entropy(
                sample_logits.contiguous().view(
                    -1, logits.size(-1)
                ),
                sample_labels.contiguous().view(-1),
                ignore_index=self.pad_token_id,
                reduction="sum"
            )

            valid_tokens = (
                sample_labels != self.pad_token_id
            ).sum().clamp(min=1)

            sample_loss = loss_sum / valid_tokens

            sample_losses.append(sample_loss)

        return torch.stack(sample_losses)

    def forward(self, outputs, labels, task_ids):

        sample_losses = self.compute_task_losses(
            outputs,
            labels,
            task_ids
        )

        return sample_losses.mean()
