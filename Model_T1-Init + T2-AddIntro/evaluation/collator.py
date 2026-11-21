import numpy as np
import torch

class FastDataCollator:
    """Pads inputs and labels to the longest sequence in each batch."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _rounded_length(self, length):
        multiple = self.pad_to_multiple_of
        if not multiple:
            return length
        return ((length + multiple - 1) // multiple) * multiple

    def __call__(self, features):
        input_ids = [item["input_ids"] for item in features]
        attention_masks = [item["attention_mask"] for item in features]
        labels = [item["labels"] for item in features]

        max_input = self._rounded_length(max(map(len, input_ids)))
        max_label = self._rounded_length(max(map(len, labels)))
        batch_size = len(features)

        padded_inputs = np.full(
            (batch_size, max_input), self.pad_token_id, dtype=np.int64
        )
        padded_masks = np.zeros((batch_size, max_input), dtype=np.int64)
        padded_labels = np.full((batch_size, max_label), -100, dtype=np.int64)

        for i, (ids, mask, target) in enumerate(
            zip(input_ids, attention_masks, labels)
        ):
            padded_inputs[i, :len(ids)] = ids
            padded_masks[i, :len(mask)] = mask
            padded_labels[i, :len(target)] = target

        return {
            "input_ids": torch.from_numpy(padded_inputs),
            "attention_mask": torch.from_numpy(padded_masks),
            "labels": torch.from_numpy(padded_labels),
            "task_id": torch.tensor(
                [item["task_id"] for item in features], dtype=torch.long
            ),
        }
