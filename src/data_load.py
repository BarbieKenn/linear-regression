import pandas as pd


def split_dataset_to_xy(dataset: pd.DataFrame = pd.read_csv('housing.csv'),
                        target: str = 'median_house_value',
                        frac: float = 0.8) -> tuple:
    train_df = dataset.sample(frac=frac, random_state=42)
    test_df = dataset.drop(train_df.index)

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return x_train, x_test, y_train, y_test


def add_bias(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.assign(bias=1)


