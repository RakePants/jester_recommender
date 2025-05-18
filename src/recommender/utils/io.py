import pickle
from pathlib import Path

import pandas as pd


def save_pickle(obj, path: Path) -> None:
    """
    Serialize an object to the given path using pickle.

    Parameters:
        obj: Python object to serialize
        path: filesystem path (including filename) to write to
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    """
    Load and return a pickled object from the given path.

    Parameters:
        path: filesystem path to the pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Save a pandas DataFrame to CSV at the given path.

    Parameters:
        df: DataFrame to save
        path: filesystem path to write (should end in .csv)
        index: whether to write row index
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a pandas DataFrame from CSV at the given path.

    Parameters:
        path: filesystem path to read (CSV file)

    Returns:
        DataFrame read from CSV
    """
    return pd.read_csv(path)
