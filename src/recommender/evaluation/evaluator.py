from typing import Any, Dict, List, Tuple

import numpy as np

from ..config import DEFAULT_TOP_K
from .metrics import (
    diversity,
    mae,
    mapk,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    serendipity_at_k,
)


class Evaluator:
    """
    Given a fitted recommender, run evaluation on train/test split.
    """

    def __init__(self, recommender):
        self.recommender = recommender

    def evaluate(
        self, trainset, testset: List[Tuple[Any, Any, float]], k: int = DEFAULT_TOP_K
    ) -> Dict[str, float]:
        # Build true ratings and recommendation dicts
        test_user_ratings: Dict[Any, Dict[Any, float]] = {}
        for uid, iid, true_r in testset:
            test_user_ratings.setdefault(uid, {})[iid] = true_r

        # User-level dicts
        true_dict: Dict[Any, set] = {
            u: set(i_dict.keys()) for u, i_dict in test_user_ratings.items()
        }
        pred_dict: Dict[Any, List[Any]] = {}

        # For MAE
        all_true: List[float] = []
        all_pred: List[float] = []

        for user, true_items in true_dict.items():
            preds = self.recommender.predict(user)
            pred_ids = [iid for iid, _ in preds]
            pred_dict[user] = pred_ids
            # collect MAE
            for iid, p in preds:
                if iid in true_items:
                    all_true.append(test_user_ratings[user][iid])
                    all_pred.append(p)

        # Compute metrics
        results: Dict[str, float] = {}
        results["mae"] = mae(all_true, all_pred)
        results["precision@k"] = float(
            np.mean([precision_at_k(true_dict[u], pred_dict[u], k) for u in true_dict])
        )
        results["recall@k"] = float(
            np.mean([recall_at_k(true_dict[u], pred_dict[u], k) for u in true_dict])
        )
        results["map@k"] = mapk(true_dict, pred_dict, k)
        results["ndcg@k"] = (
            ndcg_at_k(true_dict, pred_dict, k)
            if False
            else float(
                np.mean([ndcg_at_k(true_dict[u], pred_dict[u], k) for u in true_dict])
            )
        )

        # total items
        total_items = len(trainset.all_items())
        # popularity per item
        pop_counts: Dict[Any, int] = {}
        for inner_uid, interactions in trainset.ur.items():
            for inner_iid, _ in interactions:
                item = trainset.to_raw_iid(inner_iid)
                pop_counts[item] = pop_counts.get(item, 0) + 1
        max_pop = max(pop_counts.values()) if pop_counts else 1
        popularity_norm = {item: cnt / max_pop for item, cnt in pop_counts.items()}

        results["diversity@k"] = diversity(pred_dict, total_items, k)
        results["novelty@k"] = novelty(pred_dict, popularity_norm, k)

        results["serendipity@k"] = float(
            np.mean(
                [
                    serendipity_at_k(true_dict[u], pred_dict[u], popularity_norm, k)
                    for u in true_dict
                ]
            )
        )

        return results
