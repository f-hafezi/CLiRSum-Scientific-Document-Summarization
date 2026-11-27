
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

        # Loss function
        self.custom_loss = MultiTaskLoss(
            task_weights=task_weights,
            pad_token_id=pad_token_id,
            target_max_lengths=target_max_lengths
        )

        # Dynamic weighting settings
        self.dynamic_task_weighting = dynamic_task_weighting
        self.temperature = temperature
        self.momentum = momentum
        self.num_tasks = num_tasks
        self.history_size = history_size

        if self.temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        if not 0 <= self.momentum < 1:
            raise ValueError("Momentum must be in the range [0, 1).")

        # Initialize normalized task weights
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

        # Store recent losses for each task
        self.task_loss_history = {
            i: [] for i in range(num_tasks)
        }

    def update_task_weights(self, task_ids, sample_losses):
        """
        Update task weights using recent task losses,
        softmax normalization, and exponential moving average.
        """

        # Collect the average loss for each task in this batch
        for task_id in range(self.num_tasks):

            mask = task_ids == task_id

            if mask.any():

                current_loss = sample_losses[mask].mean().item()

                self.task_loss_history[task_id].append(
                    current_loss
                )

                # Keep only the most recent observations
                self.task_loss_history[task_id] = (
                    self.task_loss_history[task_id][-self.history_size:]
                )

        # Do not update until all tasks have observations
        if any(
            len(self.task_loss_history[i]) == 0
            for i in range(self.num_tasks)
        ):
            return

        # Calculate average recent loss for each task
        average_losses = torch.tensor(
            [
                sum(self.task_loss_history[i])
                / len(self.task_loss_history[i])
                for i in range(self.num_tasks)
            ],
            dtype=torch.float32
        )

        # Higher-loss tasks receive higher weights
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

        # Normalize the updated weights
        total_weight = sum(updated_weights.values())

        self.task_weights = {
            i: updated_weights[i] / total_weight
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

        # Forward pass
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

        # Update weights only during multi-task training
        if self.dynamic_task_weighting and model.training:

            self.update_task_weights(
                task_ids,
                sample_losses.detach()
            )

        # Calculate total task-weighted loss
        total_loss = outputs.logits.new_zeros(())

        for task_id in range(self.num_tasks):

            mask = task_ids == task_id

            if mask.any():

                task_loss = sample_losses[mask].mean()

                if self.dynamic_task_weighting:
                    weight = self.task_weights[task_id]
                else:
                    # Stage 1: ordinary single-task training
                    weight = 1.0

                total_loss = total_loss + weight * task_loss

        # Print dynamic weights every 100 optimizer steps
        if (
            self.dynamic_task_weighting
            and model.training
            and self.state.global_step > 0
            and self.state.global_step % 100 == 0
        ):

            print(
                f"\nStep: {self.state.global_step}"
                f"\nDynamic Task Weights: {self.task_weights}"
            )

        return (
            (total_loss, outputs)
            if return_outputs
            else total_loss
        )
