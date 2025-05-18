import argparse
import logging
from pathlib import Path

from recommender.data.loader import load_interactions, load_jokes
from recommender.data.splitter import train_test_split
from recommender.embeddings.colbert import ColbertEmbedder
from recommender.models.svd import SVDRecommender
from recommender.utils.io import save_pickle
from recommender.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train recommender models, split data, and save artifacts."
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        required=True,
        help="Path to interactions matrix (.xlsx)",
    )
    parser.add_argument(
        "--jokes",
        type=Path,
        required=True,
        help="Path to jokes file (.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save trained artifacts and splits",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting",
    )
    args = parser.parse_args()

    logger = get_logger(__name__)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading interaction matrix and jokes metadata")
    interactions = load_interactions(args.interactions)
    jokes = load_jokes(args.jokes)

    logger.info(f"Splitting data: test_size={args.test_size}, seed={args.seed}")
    trainset, testset = train_test_split(
        interactions,
        test_size=args.test_size,
        seed=args.seed,
    )

    logger.info("Training ColBERT embedder")
    embedder = ColbertEmbedder()
    colbert_vecs = embedder.fit_transform(jokes.text.tolist())
    save_pickle(colbert_vecs, args.output_dir / "colbert_vecs.pkl")

    logger.info("Training SVD model")
    svd_rec = SVDRecommender(n_factors=5, n_epochs=5, lr_all=0.02, reg_all=0.03)
    svd_rec.fit(trainset)
    save_pickle(svd_rec, args.output_dir / "svd_model.pkl")

    logger.info("Saving data splits")
    save_pickle(trainset, args.output_dir / "trainset.pkl")
    save_pickle(testset, args.output_dir / "testset.pkl")

    logger.info(f"All artifacts saved in {args.output_dir}")


if __name__ == "__main__":
    main()
