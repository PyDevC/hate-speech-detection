import pandas as pd
import numpy as np
import pyarrow


class Dataset:
    """Base class of every dataset
    """
    def __init__(self, Key):
        self.data = self.load_data(Key)
        self.shape = self.data.shape
        self.columns = self.data.columns

    def load_data(self, Key)->pd.DataFrame:
        """Read parquet file from a path Key
        Return: DataFrame from parquet file
        """
        return pd.read_parquet(Key)
