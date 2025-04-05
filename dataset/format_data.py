import pandas as pd
import numpy as np
import sys
import os

raw_data_path = os.path.join(os.getcwd(), "dataset", "data")

def read_dataset(data_path: str, dataset_name: str)-> pd.DataFrame:
    """Currently only read json and parquet
    """
    if ".json" in dataset_name:
        df = pd.read_json(os.path.join(data_path, dataset_name))
    elif ".parquet" in dataset_name:
        df = pd.read_parquet(os.path.join(data_path, dataset_name))
    else:
        raise TypeError(
            "create a function to read this format file"
        )

    return df
