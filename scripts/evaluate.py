import argparse
import json
from pathlib import Path

from recommender.config import DEFAULT_TOP_K
from recommender.evaluation.evaluator import Evaluator
from recommender.models.content import ContentBasedRecommender
from recommender.models.hybrid import HybridRecommender
from recommender.utils.io import load_pickle
from recommender.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Load splits and artifacts, evaluate hybrid recommender, and save all metrics."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing trainset.pkl, testset.pkl, colbert_vecs.pkl, svd_model.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save evaluation metrics",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Cutoff for top-K metrics",
    )
    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info("Loading data splits and artifacts...")
    trainset = load_pickle(args.artifacts_dir / "trainset.pkl")
    testset = load_pickle(args.artifacts_dir / "testset.pkl")
    colbert_vecs = load_pickle(args.artifacts_dir / "colbert_vecs.pkl")
    svd_rec = load_pickle(args.artifacts_dir / "svd_model.pkl")

    logger.info("Instantiating recommenders...")
    content_rec = ContentBasedRecommender(colbert_vecs)
    hybrid_rec = HybridRecommender(content_rec, svd_rec)

    logger.info("Running evaluation...")
    evaluator = Evaluator(hybrid_rec)
    metrics = evaluator.evaluate(trainset, testset, k=args.k)

    for name, val in metrics.items():
        logger.info(f"{name}: {val:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"All metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
