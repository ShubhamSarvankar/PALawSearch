"""
Embed the full corpus with the production encoder and write embeddings.jsonl.

Reads:  data/parsed/cases.jsonl
Writes: embeddings.jsonl  (one JSON line per doc, includes dense_vector)

Usage:
    just embed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from indexing.bm25_indexer import normalize_decision_date

CORPUS_PATH   = ROOT / "data/parsed/cases.jsonl"
OUTPUT_PATH   = ROOT / "embeddings.jsonl"
EMBED_BATCH   = 256
DOC_EMBED_MAX = 4096  # chars fed to encoder per doc


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = settings.encoder_model_path
    if not Path(model_path).is_absolute():
        model_path = str(ROOT / model_path)

    print(f"encoder_model_path : {settings.encoder_model_path}")
    print(f"dense_vector_dim   : {settings.dense_vector_dim}")
    print(f"Resolved path      : {model_path}")
    print(f"Device             : {device}")

    model = SentenceTransformer(model_path, device=device)
    actual_dim = model.get_sentence_embedding_dimension()
    print(f"Encoder loaded. Embedding dim = {actual_dim}")
    if actual_dim != settings.dense_vector_dim:
        raise ValueError(
            f"Encoder outputs dim={actual_dim} but settings.dense_vector_dim={settings.dense_vector_dim}"
        )

    print(f"\nReading corpus from {CORPUS_PATH} ...")
    docs: list[dict] = []
    texts: list[str] = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading corpus"):
            case = json.loads(line)
            text = (case.get("full_text") or case.get("head_matter") or "")[:DOC_EMBED_MAX]
            docs.append({
                "id":                case.get("id"),
                "name":              case.get("name", ""),
                "decision_date":     normalize_decision_date(case.get("decision_date")),
                "court_name":        case.get("court_name", ""),
                "jurisdiction_name": case.get("jurisdiction", ""),
                "parties":           case.get("parties", ""),
                "judges":            case.get("judges", ""),
                "word_count":        case.get("word_count", 0),
                "full_text":         text,
            })
            texts.append(text)

    print(f"Loaded {len(docs):,} docs. Embedding with batch_size={EMBED_BATCH} ...")
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print(f"Embeddings shape: {vecs.shape}")

    print(f"Writing {OUTPUT_PATH} ...")
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for doc, vec in tqdm(zip(docs, vecs), total=len(docs), desc="Writing embeddings.jsonl"):
            doc["dense_vector"] = vec.tolist()
            f.write(json.dumps(doc) + "\n")
    tmp.rename(OUTPUT_PATH)

    print(f"\nDone. {len(docs):,} embeddings written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
