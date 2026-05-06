import torch

class MultiTaskLoss(torch.nn.Module):
    def __init__(self, task_weights, pad_token_id, target_max_lengths):
        super().__init__()
        self.task_weights = task_weights
        self.pad_token_id = pad_token_id
        self.target_max_lengths = target_max_lengths
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, outputs, labels, task_ids):
        total_loss = 0
        batch_size = labels.size(0)
        
        for i in range(batch_size):
            task_id = task_ids[i].item()
            max_length = self.target_max_lengths[task_id]
            
            sample_logits = outputs.logits[i, :max_length, :].view(-1, outputs.logits.size(-1))
            sample_labels = labels[i, :max_length].view(-1)
            
            sample_loss = self.loss_fn(sample_logits, sample_labels)
            weight = self.task_weights.get(task_id, 1.0)
            total_loss += weight * sample_loss
        
        return total_loss / batch_size