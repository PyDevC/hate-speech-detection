import torch
from torch.utils.data import Dataset


class baseDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.text, self.labels = self.load_data(data)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            text = [self.text[i] for i in idx]
            labels = [self.labels[i] for i in idx]
        else:
            text = [self.text[idx]]
            labels = [self.labels[idx]]

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def load_data(self, data):
        texts = list[data["tweets"]]
        labels = list[data["class"]]
        return texts, labels
