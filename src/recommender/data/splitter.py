from typing import Tuple

import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split

from ..config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, RATING_MAX, RATING_MIN


def train_test_split(
    interactions: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_RANDOM_STATE,
):  # trainset and surprise-style testlist
    """
    Splits interactions into Surprise Trainset and test list.

    Parameters:
    - interactions: DataFrame indexed by user_id, columns=item_id, values=rating
    - test_size: fraction for test hold-out
    - seed: random state for reproducibility

    Returns:
    - trainset: Surprise Trainset for fitting algorithms
    - testset: list of (user_id, item_id, true_rating)
    """
    # stack to long format
    df = interactions.stack().reset_index()
    df.columns = ["user_id", "item_id", "rating"]

    reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    dataset = Dataset.load_from_df(df, reader)
    trainset, testset = surprise_split(dataset, test_size=test_size, random_state=seed)
    return trainset, testset
