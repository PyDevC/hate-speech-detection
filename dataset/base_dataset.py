import torch
from torch.utils.data import Dataset
from datasets import Dataset


class baseDataset(Dataset):
    def __init__(self, data,text_column, label_column ,tokenizer, max_length=128):
        self.texts = list(data[text_column]) 
        self.labels = list(data[label_column])  
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):  
            texts = [self.texts[i] for i in idx]
            labels = [self.labels[i] for i in idx]
        else:
            texts = [self.texts[idx]]
            labels = [self.labels[idx]]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),  
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
