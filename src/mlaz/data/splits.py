from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame,
    *,
    features: list[str],
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    features : list[str]
        Feature column names.
    target : str
        Target column name.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """
    X = df[features]
    y = df[target]
    return X, y


def train_test_split_xy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion of data to use for test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
