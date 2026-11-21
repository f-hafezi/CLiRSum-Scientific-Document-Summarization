import torch
from torch.utils.data import Dataset
import pandas as pd

# Simplified Multi-Task Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, indices=None):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.indices = indices if indices is not None else list(range(len(self.data)))
        
        self.task_prefixes = {
            0: "generate_introduction: ",
            1: "generate_abstract: ",
            2: "generate_abstract_from_intro: ",
            3: "generate_title: "
        }
        self.input_max_lengths = {0: 1024, 1: 1024, 2: 512, 3: 256}
        self.target_max_lengths = {0: 512, 1: 256, 2: 256, 3: 32}
        self.max_input_length = max(self.input_max_lengths.values())

    def __len__(self):
        return len(self.indices) * 4

    def __getitem__(self, idx):
        sample_idx = self.indices[idx // 4]
        task_id = idx % 4
        
        if sample_idx >= len(self.data):
            raise IndexError(f"Sample index {sample_idx} out of range for dataset of size {len(self.data)}")
        
        row = self.data.iloc[sample_idx]
        
        article_text = str(row.get("article_text_no_intro", ""))
        article_text1 = str(row.get("article_text", ""))
        abstract_text = str(row.get("abstract_text", ""))
        title = str(row.get("title", ""))
        intro_text = str(row.get("introduction_text", ""))

        if task_id == 0:
            input_text = self.task_prefixes[0] + article_text
            target_text = intro_text
        elif task_id == 1:
            input_text = self.task_prefixes[1] + article_text1
            target_text = abstract_text
        elif task_id == 2:
            input_text = self.task_prefixes[2] + intro_text
            target_text = abstract_text
        else:
            input_text = self.task_prefixes[3] + abstract_text
            target_text = title

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.input_max_lengths[task_id],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.target_max_lengths[task_id],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "task_id": torch.tensor(task_id, dtype=torch.long)
        }


# Single Task Dataset
class SingleTaskDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, task_id=0, indices=None):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.task_id = task_id
        self.indices = indices if indices is not None else list(range(len(self.data)))
        
        self.task_prefixes = {
            0: "generate_introduction: ",
            1: "generate_abstract: ",
            2: "generate_abstract_from_intro: ",
            3: "generate_title: "
        }
        self.input_max_lengths = {0: 1024, 1: 1024, 2: 512, 3: 256}
        self.target_max_lengths = {0: 512, 1: 256, 2: 256, 3: 32}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        
        if sample_idx >= len(self.data):
            raise IndexError(f"Sample index {sample_idx} out of range for dataset of size {len(self.data)}")
        
        row = self.data.iloc[sample_idx]
        
        article_text = str(row.get("article_text_no_intro", ""))
        article_text1 = str(row.get("article_text", ""))
        abstract_text = str(row.get("abstract_text", ""))
        title = str(row.get("title", ""))
        intro_text = str(row.get("introduction_text", ""))

        if self.task_id == 0:
            input_text = self.task_prefixes[0] + article_text
            target_text = intro_text
        elif self.task_id == 1:
            input_text = self.task_prefixes[1] + article_text1
            target_text = abstract_text
        elif self.task_id == 2:
            input_text = self.task_prefixes[2] + intro_text
            target_text = abstract_text
        else:
            input_text = self.task_prefixes[3] + abstract_text
            target_text = title

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.input_max_lengths[self.task_id],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.target_max_lengths[self.task_id],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "task_id": torch.tensor(self.task_id, dtype=torch.long)
        }
