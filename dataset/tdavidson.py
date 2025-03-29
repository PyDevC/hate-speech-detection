from .base_dataset import baseDataset

class tdavidson_hate_offensive(baseDataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.texts = list(data["tweet"]) 
        self.labels = data["label"]
        self.tokenizer = tokenizer
        self.max_length = max_length
