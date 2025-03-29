from .base_dataset import baseDataset
from sklearn.model_selection import train_test_split

class tdavidson_hate_offensive(baseDataset):
    def load_data(self, data):
        texts = list[data[""]]
        labels = list[data[""]]
        return texts, labels
