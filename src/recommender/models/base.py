"""
Abstract base class for recommenders.
"""

from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """
    Interface for recommender models.
    """

    @abstractmethod
    def fit(self, trainset):  # trainset: Surprise Trainset
        """
        Fit the recommender to training data.
        """
        pass

    @abstractmethod
    def predict(self, user_id):  # raw user_id
        """
        Generate ranked (item_id, score) list for a given user.
        """
        pass
