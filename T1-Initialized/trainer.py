
"""Custom Trainer with dynamic task weighting."""

import torch
import torch.nn.functional as F

from transformers import Trainer
from .losses import MultiTaskLoss


class CustomTrainer(Trainer):

    def __init__(
        self,
        task_weights,
        pad_token_id,
        target_max_lengths,
        dynamic_task_weighting=False,
        temperature=2.0,
        momentum=0.9,
        num_tasks=4,
        history_size=20,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.custom_loss = MultiTaskLoss(
            task_weights=task_weights,
            pad_token_id=pad_token_id,
            target_max_lengths=target_max_lengths
        )

        self.dynamic_task_weighting = dynamic_task_weighting
        self.temperature = temperature
        self.momentum = momentum
        self.num_tasks = num_tasks
        self.history_size = history_size

        # Initialize task weights
        initial_weights = task_weights or {
            i: 1.0 for i in range(num_tasks)
        }

        weight_sum = sum(
            initial_weights.get(i, 0.0)
            for i in range(num_tasks)
        )

        if weight_sum <= 0:
            raise ValueError("Initial task weights must be positive.")

        self.task_weights = {
            i: initial_weights.get(i, 0.0) / weight_sum
            for i in range(num_tasks)
        }

        # Recent loss history for each task
        self.task_loss_history = {
            i: []
            for i in range(num_tasks)
        }

    def update_task_weights(self, task_ids, sample_losses):

        # Store recent losses separately for each task
        for task_id in range(self.num_tasks):

            mask = task_ids == task_id

            if mask.any():

                current_loss = sample_losses[mask].mean().item()

                self.task_loss_history[task_id].append(
                    current_loss
                )

                self.task_loss_history[task_id] = (
                    self.task_loss_history[task_id][
                        -self.history_size:
                    ]
                )

        # Wait until every task has at least one observation
        if any(
            len(self.task_loss_history[i]) == 0
            for i in range(self.num_tasks)
        ):
            return

        # Calculate average recent loss
        average_losses = torch.tensor(
            [
                sum(self.task_loss_history[i])
                / len(self.task_loss_history[i])
                for i in range(self.num_tasks)
            ],
            dtype=torch.float32
        )

        # Softmax-based dynamic weighting
        new_weights = F.softmax(
            average_losses / self.temperature,
            dim=0
        )

        # Exponential moving average
        updated_weights = {}

        for i in range(self.num_tasks):

            updated_weights[i] = (
                self.momentum * self.task_weights[i]
                +
                (1.0 - self.momentum) * new_weights[i].item()
            )

        # Normalize weights
        total = sum(updated_weights.values())

        self.task_weights = {
            i: updated_weights[i] / total
            for i in range(self.num_tasks)
        }

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        inputs = inputs.copy()

        labels = inputs.pop("labels")
        task_ids = inputs.pop("task_id")

        outputs = model(
            **inputs,
            labels=labels
        )

        # Calculate individual sample losses
        sample_losses = self.custom_loss.compute_task_losses(
            outputs,
            labels,
            task_ids
        )

        # Dynamic updates only during Stage 2 training
        if self.dynamic_task_weighting and model.training:

            self.update_task_weights(
                task_ids,
                sample_losses.detach()
            )

        # Calculate weighted loss
        total_loss = outputs.logits.new_zeros(())

        for task_id in range(self.num_tasks):

            mask = task_ids == task_id

            if mask.any():

                task_loss = sample_losses[mask].mean()

                if self.dynamic_task_weighting:

                    weight = self.task_weights[task_id]

                else:

                    # Stage 1: no multi-task weighting
                    weight = 1.0

                total_loss = total_loss + weight * task_loss

        # Log current weights
        if (
            self.dynamic_task_weighting
            and model.training
            and self.state.global_step > 0
            and self.state.global_step % 100 == 0
        ):

            print(
                "\nStep:",
                self.state.global_step,
                "\nDynamic Task Weights:",
                self.task_weights
            )

        return (
            (total_loss, outputs)
            if return_outputs
            else total_loss
        )
