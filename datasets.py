"""Dataset classes for the four BART generation tasks."""
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TASK_PREFIXES, INPUT_MAX_LENGTHS, TARGET_MAX_LENGTHS

REQUIRED_COLUMNS = [
    "article_text_no_intro",
    "abstract_text",
    "title",
    "introduction_text",
]


def _safe_text(value):
    return "" if pd.isna(value) else str(value)


class _BaseTaskDataset(Dataset):
    def __init__(self, data_path, tokenizer, indices=None):
        self.data = pd.read_csv(data_path)
        missing = [col for col in REQUIRED_COLUMNS if col not in self.data.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        self.tokenizer = tokenizer
        self.indices = list(indices) if indices is not None else list(range(len(self.data)))

        if self.indices and (min(self.indices) < 0 or max(self.indices) >= len(self.data)):
            raise IndexError("One or more dataset indices are out of range.")

    def _encode(self, row, task_id):
        article = _safe_text(row["article_text_no_intro"])
        abstract = _safe_text(row["abstract_text"])
        title = _safe_text(row["title"])
        introduction = _safe_text(row["introduction_text"])

        if task_id == 0:
            source, target = article, introduction
        elif task_id == 1:
            source, target = article, abstract
        elif task_id == 2:
            source, target = introduction, abstract
        elif task_id == 3:
            source, target = abstract, title
        else:
            raise ValueError(f"Unsupported task_id: {task_id}")

        input_text = TASK_PREFIXES[task_id] + source
        input_encoding = self.tokenizer(
            input_text,
            max_length=INPUT_MAX_LENGTHS[task_id],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target,
            max_length=TARGET_MAX_LENGTHS[task_id],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
            "task_id": torch.tensor(task_id, dtype=torch.long),
        }


class MultiTaskDataset(_BaseTaskDataset):
    """Returns all four tasks for each article, in task order 0, 1, 2, 3."""
    def __len__(self):
        return len(self.indices) * 4

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        sample_idx = self.indices[idx // 4]
        task_id = idx % 4
        return self._encode(self.data.iloc[sample_idx], task_id)


class SingleTaskDataset(_BaseTaskDataset):
    """Returns one selected task for each article."""
    def __init__(self, data_path, tokenizer, task_id=0, indices=None):
        super().__init__(data_path, tokenizer, indices)
        if task_id not in TASK_PREFIXES:
            raise ValueError(f"Unsupported task_id: {task_id}")
        self.task_id = task_id

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        sample_idx = self.indices[idx]
        return self._encode(self.data.iloc[sample_idx], self.task_id)
