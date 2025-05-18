import numpy as np
import pandas as pd

from ..config import RATING_MAX, RATING_MIN


def preprocess_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing sentinel (99) with NaN and scale ratings to [0,1].

    If original ratings are in [-10,10] with 99 as missing,
    scaled = (raw + 10) / 20.
    """
    # replace sentinel values with NaN
    df = df.replace(99, np.nan)
    # scale to [0,1]
    df = (df + 10) / 20
    # clip just in case
    return df.clip(lower=RATING_MIN, upper=RATING_MAX)


def preprocess_jokes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure jokes DataFrame has a 'text' column of type string.

    Placeholder for future text cleaning steps.
    """
    df = df.copy()
    df["text"] = df["text"].astype(str)
    return df
