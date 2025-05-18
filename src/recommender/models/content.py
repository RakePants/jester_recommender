"""
Content-based recommender using precomputed ColBERT vectors.
"""

import numpy as np

from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """
    Ranks items for a user by semantic similarity to positively rated items.
    """

    def __init__(self, colbert_vecs, threshold: float = 0.5):
        self.colbert_vecs = colbert_vecs
        self.threshold = threshold
        self.score_matrix = self._compute_score_matrix()
        self.user_interactions = {}

    def _compute_score_matrix(self):
        n = len(self.colbert_vecs)
        mat = np.full((n, n), -np.inf, dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sim = np.dot(self.colbert_vecs[i], self.colbert_vecs[j].T)
                mat[i, j] = np.sum(np.max(sim, axis=1))
        return mat

    def fit(self, trainset):
        # build raw user -> [(item_id, rating), ...]
        for inner_uid, interactions in trainset.ur.items():
            raw_uid = trainset.to_raw_uid(inner_uid)
            raw_inter = [
                (trainset.to_raw_iid(inner_iid), rating)
                for inner_iid, rating in interactions
            ]
            self.user_interactions[raw_uid] = raw_inter
        return self

    def get_topn(self, item_id, n=10):
        row = self.score_matrix[item_id]
        indices = np.argsort(row)[::-1][:n]
        return indices.tolist()

    def predict(self, user_id):
        interactions = self.user_interactions.get(user_id, [])
        interacted = {iid for iid, _ in interactions}
        positive = {iid for iid, r in interactions if r >= self.threshold}

        # gather similar candidates
        candidates = set()
        for iid in positive:
            candidates.update(self.get_topn(iid))
        candidates -= interacted

        # score by max similarity across positive items
        item_scores = {}
        for c in candidates:
            item_scores[c] = max(self.score_matrix[p, c] for p in positive)
        # sort descending
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
