
import torch
import torch.nn.functional as F
from transformers import Trainer

from loss import MultiTaskLoss


class CustomTrainer(Trainer):

    def __init__(
        self,
        task_weights,
        pad_token_id,
        target_max_lengths,
        temperature=2.0,
        momentum=0.9,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.custom_loss = MultiTaskLoss(
            task_weights,
            pad_token_id,
            target_max_lengths
        )

        self.num_tasks = 4
        self.temperature = temperature
        self.momentum = momentum

        # Initial task weights
        self.task_weights = {
            i: 1.0 / self.num_tasks
            for i in range(self.num_tasks)
        }

        # Store recent losses
        self.task_loss_history = {
            i: []
            for i in range(self.num_tasks)
        }

    def update_task_weights(self, task_ids, task_losses):

        for task_id, loss_value in zip(
            task_ids.tolist(),
            task_losses.detach().tolist()
        ):

            self.task_loss_history[task_id].append(
                loss_value
            )

            # Keep recent 20 losses
            self.task_loss_history[task_id] = (
                self.task_loss_history[task_id][-20:]
            )

        # Wait until all tasks have observations
        if any(
            len(self.task_loss_history[i]) == 0
            for i in range(self.num_tasks)
        ):
            return

        # Average recent loss of each task
        avg_losses = torch.tensor(
            [
                sum(self.task_loss_history[i])
                / len(self.task_loss_history[i])
                for i in range(self.num_tasks)
            ],
            dtype=torch.float32
        )

        # Dynamic Softmax weighting
        new_weights = F.softmax(
            avg_losses / self.temperature,
            dim=0
        )

        # Exponential moving average
        for i in range(self.num_tasks):

            self.task_weights[i] = (
                self.momentum * self.task_weights[i]
                +
                (1 - self.momentum) * new_weights[i].item()
            )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")
        task_ids = inputs.pop("task_id")

        outputs = model(
            **inputs,
            labels=labels
        )

        # Calculate individual task losses
        task_losses = self.custom_loss.compute_task_losses(
            outputs,
            labels,
            task_ids
        )

        # Update dynamic weights
        self.update_task_weights(
            task_ids,
            task_losses
        )

        # Weighted multi-task loss
        loss = torch.tensor(
            0.0,
            device=outputs.logits.device
        )

        for i in range(self.num_tasks):

            mask = task_ids == i

            if mask.any():

                task_loss = task_losses[mask].mean()

                loss = loss + (
                    self.task_weights[i] * task_loss
                )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )
