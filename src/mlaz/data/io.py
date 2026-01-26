from __future__ import annotations

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)
