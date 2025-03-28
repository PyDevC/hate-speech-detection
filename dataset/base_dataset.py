from datasets import load_dataset, load_dataset_builder

class Dataset:
    """Base class of every dataset
    parameters:
        Key: path to data or hugging face path
    """
    def __init__(self, Key:str):
        self.dataset_builder = load_dataset_builder(Key)
        self.data = load_dataset(Key)
