from __future__ import annotations

import logging

from chromadb.utils import embedding_functions


def build_sentence_transformer_ef(
    model_name: str,
    *,
    disable_progress: bool = True,
):
    """Build a SentenceTransformer embedding function with progress bars disabled."""
    if disable_progress:
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            show_progress_bar=not disable_progress,
        )
    except TypeError:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
