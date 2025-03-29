from .base_dataset import baseDataset

class tdavidson_hate_offensive(baseDataset):
    def load_data(self, data):
        texts = list[data["tweets"]]
        labels = list[data["class"]]
        return texts, labels
