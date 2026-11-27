
import torch
import torch.nn.functional as F

from transformers import Trainer
from loss import MultiTaskLoss


class CustomTrainer(Trainer):

    def __init__(
        self,
        task_weights=None,
        pad_token_id=1,
        target_max_lengths=None,
        dynamic_task_weighting=False,
        temperature=2.0,
        momentum=0.9,
        num_tasks=4,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.custom_loss = MultiTaskLoss(
            task_weights,
            pad_token_id,
            target_max_lengths
        )

        self.dynamic_task_weighting = dynamic_task_weighting
        self.temperature = temperature
        self.momentum = momentum
        self.num_tasks = num_tasks

        self.task_weights = {
            i: 1.0 / num_tasks
            for i in range(num_tasks)
        }

        self.task_loss_history = {
            i: []
            for i in range(num_tasks)
        }

    def update_task_weights(self, task_ids, task_losses):

        for task_id, loss_value in zip(
            task_ids.detach().cpu().tolist(),
            task_losses.detach().float().cpu().tolist()
        ):

            self.task_loss_history[task_id].append(loss_value)

            self.task_loss_history[task_id] = (
                self.task_loss_history[task_id][-20:]
            )

        # Wait until every task has at least one observation
        if any(
            len(self.task_loss_history[i]) == 0
            for i in range(self.num_tasks)
        ):
            return

        avg_losses = torch.tensor(
            [
                sum(self.task_loss_history[i])
                / len(self.task_loss_history[i])
                for i in range(self.num_tasks)
            ],
            dtype=torch.float32
        )

        new_weights = F.softmax(
            avg_losses / self.temperature,
            dim=0
        )

        # Smooth dynamic weight updates
        for i in range(self.num_tasks):

            self.task_weights[i] = (
                self.momentum * self.task_weights[i]
                +
                (1 - self.momentum) * new_weights[i].item()
            )

        if self.state.global_step % 100 == 0:

            print(
                f"\nStep {self.state.global_step} "
                f"Dynamic Task Weights: {self.task_weights}"
            )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        inputs = dict(inputs)

        labels = inputs.pop("labels")
        task_ids = inputs.pop("task_id")

        outputs = model(
            **inputs,
            labels=labels
        )

        task_losses = self.custom_loss.compute_task_losses(
            outputs,
            labels,
            task_ids
        )

        # Update weights only during multi-task training
        if self.dynamic_task_weighting and model.training:

            self.update_task_weights(
                task_ids,
                task_losses
            )

        total_loss = torch.tensor(
            0.0,
            device=outputs.logits.device
        )

        present_tasks = torch.unique(task_ids).detach().cpu().tolist()

        if self.dynamic_task_weighting:

            # Use only the weights of tasks present in this batch
            weight_sum = sum(
                self.task_weights[int(i)]
                for i in present_tasks
            )

            for task_id in present_tasks:

                task_id = int(task_id)

                mask = task_ids == task_id

                current_loss = task_losses[mask].mean()

                weight = self.task_weights[task_id]

                total_loss = total_loss + weight * current_loss

            total_loss = total_loss / max(weight_sum, 1e-8)

        else:

            # Single-task or ordinary unweighted training
            total_loss = task_losses.mean()

        return (
            (total_loss, outputs)
            if return_outputs
            else total_loss
        )
