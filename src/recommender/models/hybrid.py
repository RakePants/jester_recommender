from .base import BaseRecommender
from .content import ContentBasedRecommender
from .svd import SVDRecommender


class HybridRecommender(BaseRecommender):
    def __init__(
        self,
        content_rec: ContentBasedRecommender,
        svd_rec: SVDRecommender,
        threshold: float = 0.5,
        n_similar: int = 10,
    ):
        self.content_rec = content_rec
        self.svd_rec = svd_rec
        self.threshold = threshold
        self.n_similar = n_similar
        self.user_interactions = {}

    def fit(self, trainset):
        # fit both
        self.content_rec.fit(trainset)
        self.svd_rec.fit(trainset)
        # build user_interactions mapping raw -> [(item, rating)]
        for inner_uid, interactions in trainset.ur.items():
            raw_uid = trainset.to_raw_uid(inner_uid)
            raw_inter = [
                (trainset.to_raw_iid(inner_iid), rating)
                for inner_iid, rating in interactions
            ]
            self.user_interactions[raw_uid] = raw_inter
        return self

    def predict(self, user_id):
        interactions = self.user_interactions.get(user_id, [])
        interacted = {iid for iid, _ in interactions}
        positive = {iid for iid, r in interactions if r >= self.threshold}

        # content-based candidates
        content_cands = set()
        for iid in positive:
            content_cands.update(self.content_rec.get_topn(iid, n=self.n_similar))
        content_cands -= interacted

        # rerank by SVD
        content_scores = {
            i: self.svd_rec.predict_one(user_id, i) for i in content_cands
        }
        # only keep those above threshold
        content_scores = {
            i: s for i, s in content_scores.items() if s >= self.threshold
        }
        items_by_content = sorted(
            content_scores.items(), key=lambda x: x[1], reverse=True
        )

        # remaining items
        all_items = set(self.svd_rec.items)
        rest = all_items - interacted - set(i for i, _ in items_by_content)
        rest_scores = {i: self.svd_rec.predict_one(user_id, i) for i in rest}
        rest_scores = {i: s for i, s in rest_scores.items() if s >= self.threshold}
        rest_items = sorted(rest_scores.items(), key=lambda x: x[1], reverse=True)

        return items_by_content + rest_items
