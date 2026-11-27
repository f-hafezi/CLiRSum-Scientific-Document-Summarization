
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        task_weights=None,
        pad_token_id=1,
        target_max_lengths=None
    ):
        super().__init__()

        self.task_weights = task_weights
        self.pad_token_id = pad_token_id

        self.target_max_lengths = target_max_lengths or {
            0: 512,
            1: 256,
            2: 256,
            3: 32
        }

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

            loss = F.cross_entropy(
                sample_logits.reshape(-1, logits.size(-1)),
                sample_labels.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction="sum"
            )

            valid_tokens = (
                sample_labels != self.pad_token_id
            ).sum().clamp(min=1)

            loss = loss / valid_tokens

            sample_losses.append(loss)

        return torch.stack(sample_losses)

    def forward(self, outputs, labels, task_ids):

        task_losses = self.compute_task_losses(
            outputs,
            labels,
            task_ids
        )

        return task_losses.mean()
