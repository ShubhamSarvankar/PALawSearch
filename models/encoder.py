"""
SentenceTransformer-based dual encoder.
Preserves the DualEncoder interface so dense_searcher and embed need no changes.
Loads from a trained checkpoint dir via settings.encoder_model_path.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import settings


class Encoder:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        path = model_path or settings.encoder_model_path
        self.model = SentenceTransformer(path, device=device)

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
        max_length: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if max_length is not None:
            self.model.max_seq_length = max_length
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode(query)[0]

    def encode_document(self, document: str) -> np.ndarray:
        return self.encode(document)[0]

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        d = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9)
        return d @ q
