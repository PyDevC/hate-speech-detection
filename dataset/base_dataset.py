import torch
from torch.utils.data import Dataset
from datasets import Dataset


class baseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.text = list(data["tweet"]) 
        self.labels = list(data["class"])  
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):  
            texts = [self.text[i] for i in idx]
            labels = [self.labels[i] for i in idx]
        else:
            texts = [self.text[idx]]
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

    def load_data(self, data):
        texts = list[data["tweet"]]
        labels = list[data["class"]]
        return texts, labels
