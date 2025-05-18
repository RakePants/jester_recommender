from typing import Any, Dict, List, Set

import numpy as np


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    return float(np.mean([abs(t - p) for t, p in zip(y_true, y_pred)]))


def precision_at_k(true_items: Set[Any], predicted_items: List[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = predicted_items[:k]
    hits = len(set(top_k) & true_items)
    return hits / float(k)


def recall_at_k(true_items: Set[Any], predicted_items: List[Any], k: int) -> float:
    if not true_items or k <= 0:
        return 0.0
    top_k = predicted_items[:k]
    return len(set(top_k) & true_items) / float(len(true_items))


def average_precision_at_k(
    true_items: Set[Any], predicted_items: List[Any], k: int
) -> float:
    if not true_items or k <= 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for idx in range(1, k + 1):
        if idx - 1 < len(predicted_items) and predicted_items[idx - 1] in true_items:
            hits += 1
            sum_prec += hits / float(idx)
    return sum_prec / float(len(true_items))


def mapk(
    true_dict: Dict[Any, Set[Any]], pred_dict: Dict[Any, List[Any]], k: int
) -> float:
    ap_values = []
    for user, true_items in true_dict.items():
        predicted = pred_dict.get(user, [])
        ap = average_precision_at_k(true_items, predicted, k)
        ap_values.append(ap)
    return float(np.mean(ap_values)) if ap_values else 0.0


def dcg_at_k(true_items: Set[Any], predicted_items: List[Any], k: int) -> float:
    dcg = 0.0
    for idx in range(1, k + 1):
        if idx - 1 < len(predicted_items) and predicted_items[idx - 1] in true_items:
            rel = 1
        else:
            rel = 0
        if idx == 1:
            dcg += rel
        else:
            dcg += rel / np.log2(idx + 1)
    return dcg


def ndcg_at_k(true_items: Set[Any], predicted_items: List[Any], k: int) -> float:
    if not true_items or k <= 0:
        return 0.0
    dcg = dcg_at_k(true_items, predicted_items, k)
    ideal_rels = [1] * min(len(true_items), k)
    idcg = sum(
        (1 if i == 1 else 1 / np.log2(i + 1)) for i in range(1, len(ideal_rels) + 1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def diversity(pred_dict: Dict[Any, List[Any]], total_items: int, k: int) -> float:
    all_rec = set()
    for recs in pred_dict.values():
        all_rec.update(recs[:k])
    return len(all_rec) / float(total_items) if total_items > 0 else 0.0


def novelty(
    pred_dict: Dict[Any, List[Any]], popularity: Dict[Any, float], k: int
) -> float:
    total = 0.0
    count = 0
    for recs in pred_dict.values():
        for item in recs[:k]:
            pop = popularity.get(item, 0.0)
            total += 1.0 - pop
            count += 1
    return total / count if count > 0 else 0.0


def serendipity_at_k(
    true_items: Set[Any],
    predicted_items: List[Any],
    popularity: Dict[Any, float],
    k: int,
) -> float:
    """
    Compute serendipity@k as the average unexpected relevance:
      sum_{i <= k, rec in true_items} (1 - popularity[item]) / k
    """
    if k <= 0:
        return 0.0
    score = 0.0
    for idx in range(1, k + 1):
        if idx - 1 < len(predicted_items) and predicted_items[idx - 1] in true_items:
            pop = popularity.get(predicted_items[idx - 1], 0.0)
            score += 1.0 - pop
    return score / float(k)
