import pandas as pd


def split_dataset_to_xy(dataset_directory: str = 'housing.csv',
                        target: str = 'median_house_value',
                        frac: float = 0.8) -> tuple:
    dataset = remove_nan(pd.read_csv(dataset_directory))
    train_df = dataset.sample(frac=frac, random_state=42)
    test_df = dataset.drop(train_df.index)

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return x_train, x_test, y_train, y_test


def add_bias(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.assign(bias=1)


def one_hot_encoding(dataset: pd.DataFrame,
                     categorical_feats: list) -> pd.DataFrame:
    encoded_dataset = dataset.copy()
    for feat in categorical_feats:
        uniq_feat = dataset[feat].unique()
        for new_column in uniq_feat:
            encoded_dataset[new_column] = (dataset[feat] == new_column).astype(int)

        encoded_dataset = encoded_dataset.drop(columns=[feat])
        encoded_dataset = encoded_dataset.drop(columns=uniq_feat[0])

    return encoded_dataset


def remove_nan(dataset: pd.DataFrame) -> pd.DataFrame:
    nan_cols = dataset.columns[dataset.isna().any()].tolist()
    if nan_cols:
        for nan_col in nan_cols:
            dataset[nan_col] = dataset[nan_col].fillna(dataset[nan_col].median())
        return dataset
    return dataset


def normalization(dataset: pd.DataFrame) -> pd.DataFrame:
    num_feats = dataset.select_dtypes(include=['int64', 'float64'])
    cat_feats = dataset.select_dtypes(exclude=['int64', 'float64'])

    num_scaled = (num_feats - num_feats.min()) / (num_feats.max() - num_feats.min())

    dataset_normalized = pd.concat([num_scaled, cat_feats], axis=1)

    return dataset_normalized


def standardization(dataset: pd.DataFrame) -> pd.DataFrame:
    num_feats = dataset.select_dtypes(include=['int64', 'float64'])
    cat_feats = dataset.select_dtypes(exclude=['int64', 'float64'])

    num_mean = num_feats.mean()
    n = len(num_feats)
    num_deviation = (((num_feats - num_mean) ** 2).sum() / (n - 1)) ** 0.5

    num_standard = (num_feats - num_mean) / num_deviation

    dataset_standard = pd.concat([num_standard, cat_feats], axis=1)
    return dataset_standard


if __name__ == "__main__":
    df = split_dataset_to_xy()
    print((df[0]))
