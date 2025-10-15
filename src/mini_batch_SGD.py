import pandas as pd
import numpy as np
from data_load import split_dataset_to_xy, one_hot_encoding, add_bias, standardization


def gradient_descent_mse(
        x: pd.DataFrame,
        y: pd.DataFrame,
        a: float = 10 ** (-3),
        n_iter: int = 1000,
        batch: float = 0.25,
        reg_rate: float = 3 * 10 ** (-4)
) -> tuple:
    n = x.to_numpy().shape[1]
    w = np.zeros((n, 1), dtype=float)

    y_mean = float(y.mean())
    y_centered = y - y_mean
    y_std = float(y_centered.std(ddof=0)) or 1.0
    y_scaled = y_centered / y_std

    cols = list(x.columns)
    mask = np.ones((len(cols), 1))
    if 'bias' in cols:
        mask[cols.index('bias'), 0] = 0.0

    for i in range(n_iter):
        m = len(x)
        batch_size = max(1, int(np.ceil(batch * m)))
        idx = x.sample(n=batch_size, replace=False).index
        x_b = x.loc[idx].to_numpy(dtype=float)
        y_b = y_scaled.loc[idx].to_numpy(dtype=float).reshape(-1, 1)

        y_prediction = (x_b @ w)

        grad = (2 / batch_size) * (x_b.T @ (y_prediction - y_b)) + 2 * reg_rate * (w * mask)
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e3:
            grad *= 1e3 / (gnorm + 1e-12)
        w -= a * grad

    return w, y_std, y_mean

if __name__ == '__main__':
    df = split_dataset_to_xy()
    x_t, y_t = one_hot_encoding(df[0], ['ocean_proximity']), df[2]
    print(gradient_descent_mse(add_bias(standardization(x_t)[0]), y_t))

