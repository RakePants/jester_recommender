"""
SVD-based collaborative filtering recommender.
"""

from surprise import SVD

from .base import BaseRecommender


class SVDRecommender(BaseRecommender):
    def __init__(
        self,
        n_factors=5,
        n_epochs=5,
        lr_all=0.02,
        reg_all=0.03,
        biased=True,
    ):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            biased=biased,
        )
        self.trainset = None
        self.items = []

    def fit(self, trainset):
        self.model.fit(trainset)
        self.trainset = trainset
        # record all raw item ids
        inner_items = trainset.all_items()
        self.items = [trainset.to_raw_iid(i) for i in inner_items]
        return self

    def predict(self, user_id):
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return []
        seen = {iid for iid, _ in self.trainset.ur[inner_uid]}
        preds = []
        for item in self.items:
            inner_iid = self.trainset.to_inner_iid(item)
            if inner_iid in seen:
                continue
            est = self.model.predict(user_id, item).est
            preds.append((item, est))
        return sorted(preds, key=lambda x: x[1], reverse=True)

    def predict_one(self, user_id, item_id):
        return self.model.predict(user_id, item_id).est
