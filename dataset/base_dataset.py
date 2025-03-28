from datasets import load_dataset, load_dataset_builder

class Dataset:
    def __init__(self, Key:str):
        self.dataset_builder = load_dataset_builder(Key)
        self.data = load_dataset(Key)
        self.column = self.data.column_names

class hateexplain.py
