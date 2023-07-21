import pandas as pd
import numpy as np


def unique_values(df: pd.DataFrame, column: str) -> np.ndarray:
    return df[column].unique()


def count_values(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].value_counts()


def count_nan(df: pd.DataFrame, column: str) -> int:
    return df[column].isna().sum()


def to_one_hot(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert the given columns to one hot encoding
    The original columns are dropped and the new ones should be named according to the category they represent

    :param data: The data to convert
    :param columns: The columns to convert

    :return: The converted data
    """
    for column in columns:
        one_hot = pd.get_dummies(data[column], prefix=column)
        data = data.drop(column, axis=1)
        data = data.join(one_hot)
    return data


def to_binary_one_hot(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert the given column to one hot encoding. The column should only have two values
    The original column is dropped and the new ones should be named according to the category they represent

    :param data: The data to convert
    :param column: The column to convert

    :return: The converted data
    """
    for column in columns:
        one_hot = pd.get_dummies(data[column], prefix=column)
        data = data.drop(column, axis=1)

        positive_value = one_hot.columns[0]
        data[f"{column}_{positive_value}"] = one_hot[positive_value]

    return data
