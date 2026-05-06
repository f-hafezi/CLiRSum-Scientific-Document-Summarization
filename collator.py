import torch

class SimpleDataCollator:
    def __init__(self, tokenizer, max_input_length=1024, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, features):
        batch = {}
        
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        input_ids_padded = torch.stack([
            torch.nn.functional.pad(
                ids, (0, self.max_input_length - ids.size(0)), value=self.tokenizer.pad_token_id
            ) for ids in input_ids
        ])
        attention_mask_padded = torch.stack([
            torch.nn.functional.pad(
                mask, (0, self.max_input_length - mask.size(0)), value=0
            ) for mask in attention_mask
        ])
        
        labels_padded = torch.stack([
            torch.nn.functional.pad(
                label, (0, self.max_target_length - label.size(0)), value=self.tokenizer.pad_token_id
            ) for label in labels
        ])
        
        batch["input_ids"] = input_ids_padded
        batch["attention_mask"] = attention_mask_padded
        batch["labels"] = labels_padded
        batch["task_id"] = torch.stack([f["task_id"] for f in features])
        
        return batch