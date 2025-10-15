import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
    return np.sqrt(mse(y_pred, y_true))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    return np.mean(np.abs(y_pred - y_true))


def mape(y_pred: np.ndarray, y_true: np.ndarray):
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def r2_score(y_pred: np.ndarray, y_true: np.ndarray):
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
