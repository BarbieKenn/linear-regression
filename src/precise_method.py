import pandas as pd
import numpy as np
from data_load import split_dataset_to_xy, one_hot_encoding, add_bias


def find_weights(x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 reg_rate: float) -> np.ndarray:
    x = x_train.to_numpy()
    y = y_train.to_numpy()

    identity = np.eye(x_train.shape[1])
    identity[x_train.columns.get_loc('bias'), 0] = 0

    w = np.linalg.inv(x.T @ x + reg_rate * identity) @ x.T @ y
    return w

if __name__ == "__main__":
    df = split_dataset_to_xy()
    weights = find_weights(add_bias(one_hot_encoding(df[0], ['ocean_proximity'])), df[2], 3e-4)
    print(weights.ravel())
