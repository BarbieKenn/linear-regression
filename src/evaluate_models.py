from data_load import split_dataset_to_xy, add_bias, standardization, one_hot_encoding
from precise_method import find_weights
from mini_batch_SGD import gradient_descent_mse
from metrics import rmse, mae, mape, r2_score

if __name__ == '__main__':
    dataset = split_dataset_to_xy(dataset_directory='housing.csv', target='median_house_value', frac=0.8)
    x_train, x_test, y_train, y_test = dataset
    x_train, x_test = (one_hot_encoding(x_train, ['ocean_proximity']),
                       one_hot_encoding(x_test, ['ocean_proximity']))

    x_train_std, mean, std = standardization(x_train)
    x_test_std = standardization(x_test, mean, std)

    x_test_std, x_train_std = add_bias(x_test_std).astype('float64'), add_bias(x_train_std).astype('float64')

    x_train_std = x_train_std.astype('float64')
    x_test_std = x_test_std.astype('float64')

    x_test_std = x_test_std.reindex(columns=x_train_std.columns, fill_value=0)

    w_exact = find_weights(x_train_std, y_train, 3 * 10 ** (-4))
    w_sgd, y_std, y_mean = gradient_descent_mse(x_train_std, y_train, 10 ** (-3), 2500, 0.25, 10 ** (-3))

    y_pred_exact = (x_test_std @ w_exact).to_numpy().ravel()
    y_pred_sgd = ((x_test_std @ w_sgd).to_numpy().ravel() * float(y_std)) + float(y_mean)

    print(' Metrics for exact method:')
    metrics_exact = {
        "RMSE": rmse(y_pred_exact, y_test),
        "MAE":  mae(y_pred_exact, y_test),
        "MAPE": mape(y_pred_exact, y_test),
        "R2":   r2_score(y_pred_exact, y_test),
    }
    for name, value in metrics_exact.items():
        print(f"{name}: {value:.3f}")

    print('\n', 'Metrics for SGD:')
    metrics_sgd = {
        "RMSE": rmse(y_pred_sgd, y_test),
        "MAE":  mae(y_pred_sgd, y_test),
        "MAPE": mape(y_pred_sgd, y_test),
        "R2":   r2_score(y_pred_sgd, y_test),
    }
    for name, value in metrics_sgd.items():
        print(f"{name}: {value:.3f}")
