import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .config import (
    REQUIRED_COLUMNS, TASK_PREFIXES, INPUT_MAX_LENGTHS, TARGET_MAX_LENGTHS
)

logger = logging.getLogger(__name__)

class EvaluationDataset(Dataset):
    """Expands every source article into the four evaluation tasks."""

    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        logger.info("Loading evaluation data from: %s", data_path)
        self.df = pd.read_csv(data_path, encoding="utf-8")

        missing = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")

        original_count = len(self.df)
        self.df = self.df.dropna(subset=REQUIRED_COLUMNS)
        non_empty = self.df[REQUIRED_COLUMNS].apply(
            lambda column: column.astype(str).str.strip().str.len().gt(0)
        ).all(axis=1)
        self.df = self.df.loc[non_empty].reset_index(drop=True)
        logger.info(
            "Retained %d/%d rows after removing missing or empty fields",
            len(self.df), original_count
        )

    def __len__(self):
        return len(self.df) * 4

    def __getitem__(self, idx):
        sample_idx, task_id = divmod(idx, 4)
        row = self.df.iloc[sample_idx]

        if task_id == 0:
            content, target = row["article_text_no_intro"], row["introduction_text"]
        elif task_id == 1:
            content, target = row["article_text"], row["abstract_text"]
        elif task_id == 2:
            content, target = row["introduction_text"], row["abstract_text"]
        else:
            content, target = row["abstract_text"], row["title"]

        prefix = TASK_PREFIXES[task_id]
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        max_content = INPUT_MAX_LENGTHS[task_id] - len(prefix_ids)
        content_ids = self.tokenizer(
            str(content), add_special_tokens=False, truncation=True,
            max_length=max_content
        )["input_ids"]

        input_ids = prefix_ids + content_ids
        attention_mask = [1] * len(input_ids)

        target_ids = self.tokenizer(
            str(target), add_special_tokens=False, truncation=True,
            max_length=TARGET_MAX_LENGTHS[task_id] - 2
        )["input_ids"]
        labels = [self.tokenizer.bos_token_id] + target_ids + [
            self.tokenizer.eos_token_id
        ]

        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "task_id": task_id,
        }
