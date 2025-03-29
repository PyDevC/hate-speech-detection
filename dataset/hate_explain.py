from .base_dataset import Dataset
from sklearn.model_selection import train_test_split


class HateExplain(Dataset):
    """Create a DataFrame of HateExplain Dataset
    """
    def __init__(self, Key):
        super().__init__(self)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data,
                                                                      test_size=0.3, 
                                                                      random_state=42)
