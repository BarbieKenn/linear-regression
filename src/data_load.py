import pandas as pd


def split_dataset_to_xy(dataset_directory: str = 'housing.csv',
                        target: str = 'median_house_value',
                        frac: float = 0.8) -> tuple:
    dataset = pd.read_csv(dataset_directory)
    train_df = dataset.sample(frac=frac, random_state=42)
    test_df = dataset.drop(train_df.index)

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return add_bias(x_train), add_bias(x_test), y_train, y_test


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


if __name__ == "__main__":
    df = pd.read_csv('housing.csv')
    print(df.head(10))

