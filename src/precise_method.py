import pandas as pd
import numpy as np
from data_load import split_dataset_to_xy, one_hot_encoding, remove_nan


def find_weights(x_train: pd.DataFrame,
                 y_train: pd.DataFrame) -> np.ndarray:
    x = remove_nan(x_train).to_numpy()
    y = y_train.to_numpy()

    w = np.linalg.inv(x.T @ x) @ x.T @ y
    return w

if __name__ == "__main__":
    df = split_dataset_to_xy()
    weights = find_weights(one_hot_encoding(df[0], ['ocean_proximity']), df[2])
    print(weights.ravel())
