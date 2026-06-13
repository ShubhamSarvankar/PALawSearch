"""
Hand-implemented IR metrics for the citation-grounded eval harness.

All metrics operate on binary relevance (cited = 1, not cited = 0).

Inputs:
  qrels : dict[str, set[str]]  -- {query_id: {relevant_doc_id, ...}}
  run   : dict[str, list[str]] -- {query_id: [doc_id_rank1, doc_id_rank2, ...]}

Cross-checked against pytrec_eval in tests/test_metrics.py.
"""
from __future__ import annotations

import math


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def recall_at_k(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
    k: int,
) -> float:
    """Fraction of relevant docs found in top-k, averaged over queries."""
    scores = []
    for qid, rel in qrels.items():
        if not rel:
            continue
        retrieved = set(run.get(qid, [])[:k])
        scores.append(len(retrieved & rel) / len(rel))
    return _mean(scores)


def ndcg_at_k(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k, binary relevance."""
    scores = []
    for qid, rel in qrels.items():
        if not rel:
            continue
        ranked = run.get(qid, [])[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in rel)
        ideal_k = min(len(rel), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
        scores.append(dcg / idcg if idcg > 0.0 else 0.0)
    return _mean(scores)


def mrr(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
) -> float:
    """Mean Reciprocal Rank: 1/(rank of first relevant doc), averaged over queries."""
    scores = []
    for qid, rel in qrels.items():
        if not rel:
            continue
        rr = 0.0
        for i, doc_id in enumerate(run.get(qid, [])):
            if doc_id in rel:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)
    return _mean(scores)


def map_score(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
) -> float:
    """Mean Average Precision.

    AP = (1 / |R|) * sum_{k : ranked_k is relevant} P@k
    """
    scores = []
    for qid, rel in qrels.items():
        if not rel:
            continue
        ranked = run.get(qid, [])
        hits, ap = 0, 0.0
        for i, doc_id in enumerate(ranked):
            if doc_id in rel:
                hits += 1
                ap += hits / (i + 1)
        scores.append(ap / len(rel) if hits > 0 else 0.0)
    return _mean(scores)


def compute_all(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
) -> dict[str, float]:
    return {
        "recall@1":  recall_at_k(qrels, run, 1),
        "recall@5":  recall_at_k(qrels, run, 5),
        "recall@10": recall_at_k(qrels, run, 10),
        "recall@20": recall_at_k(qrels, run, 20),
        "ndcg@10":   ndcg_at_k(qrels, run, 10),
        "mrr":       mrr(qrels, run),
        "map":       map_score(qrels, run),
    }


def per_query_ndcg10(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
) -> dict[str, float]:
    """Per-query nDCG@10 dict -- used for paired significance testing."""
    out: dict[str, float] = {}
    for qid, rel in qrels.items():
        if not rel:
            out[qid] = 0.0
            continue
        ranked = run.get(qid, [])[:10]
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in rel)
        ideal_k = min(len(rel), 10)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
        out[qid] = dcg / idcg if idcg > 0.0 else 0.0
    return out
