from pathlib import Path

import pandas as pd

from .preprocess import preprocess_interactions, preprocess_jokes


def load_interactions(path: Path) -> pd.DataFrame:
    """
    Reads an Excel interaction matrix:
      - Drops the first column (row IDs)
      - Renames columns to 0..n_items-1
      - Applies preprocessing (NaN replacement, scaling)

    Returns a DataFrame of shape (n_users, n_items) with ratings in [0,1].
    """
    path = Path(path)
    df = pd.read_excel(path, header=None)
    # remove the index column and reset numeric column labels
    df = df.iloc[:, 1:]
    df.columns = range(df.shape[1])
    return preprocess_interactions(df)


def load_jokes(path: Path) -> pd.DataFrame:
    """
    Reads an Excel file of jokes (one per row) and renames column 0 to 'text'.
    """
    path = Path(path)
    df = pd.read_excel(path, header=None)
    df = df.rename(columns={0: "text"})
    return preprocess_jokes(df)
