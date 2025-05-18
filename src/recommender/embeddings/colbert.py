from typing import Any, List

import numpy as np
from FlagEmbedding import BGEM3FlagModel


class ColbertEmbedder:
    """
    Generate ColBERT embeddings for a corpus of texts.

    Parameters:
        model_name: HuggingFace model identifier for BGEM3FlagModel
        use_fp16: whether to use half-precision
    """

    def __init__(
        self,
        model_name: str = "TatonkaHF/bge-m3_en_ru",
        use_fp16: bool = False,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.colbert_vecs: List[Any] = []

    def fit(self, texts: List[str]) -> "ColbertEmbedder":  # noqa: F821
        """
        Encode a list of texts and store their ColBERT vectors.

        Returns:
            self
        """
        self.colbert_vecs = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )["colbert_vecs"]
        return self

    def transform(self, texts: List[str]) -> List[Any]:
        """
        Encode new texts and return their ColBERT vectors.
        """
        return self.model.encode(
            texts,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )["colbert_vecs"]

    def fit_transform(self, texts: List[str]) -> List[Any]:
        """
        Convenience method: fit on texts and return vectors.
        """
        self.fit(texts)
        return self.colbert_vecs

    def get_index(self) -> np.ndarray:
        """
        Return the matrix of pairwise colbert scores for all fitted texts.

        Note: computing this is O(n^2) in number of texts.
        """
        if not self.colbert_vecs:
            raise ValueError("Must fit embedder before computing score matrix.")

        n = len(self.colbert_vecs)
        score_matrix = np.full((n, n), -np.inf, dtype=float)

        for i, query_vec in enumerate(self.colbert_vecs):
            for j, doc_vec in enumerate(self.colbert_vecs):
                if i == j:
                    continue
                # dot per-token and sum of max similarities
                sim = np.dot(query_vec, doc_vec.T)
                score_matrix[i, j] = np.sum(np.max(sim, axis=1))

        return score_matrix
