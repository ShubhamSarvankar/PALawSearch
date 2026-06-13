"""
Encoder fine-tuning (dual-encoder, MNRL).

Workflow:
  1. python -m training.train_encoder --base minilm --smoke   # confirm VRAM + loss
  2. python -m training.train_encoder --base minilm           # full run
  3. python -m training.train_encoder --base legalbert --smoke
  4. python -m training.train_encoder --base legalbert
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models as st_models,
)
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from transformers import EarlyStoppingCallback

# -- per-base config ------------------------------------------------------------

BASE_CONFIGS: dict[str, dict] = {
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "is_st_model": True,   # already packaged as a sentence-transformer
        "max_seq_length": 256,
        "dim": 384,
        "batch_size": 32,
        "grad_accum": 2,       # effective batch = 64
        "epochs": 2,
        "lr": 2e-5,
    },
    "legalbert": {
        "model_id": "nlpaueb/legal-bert-base-uncased",
        "is_st_model": False,  # raw BERT; we wrap with explicit mean-pooling
        "max_seq_length": 384,
        "dim": 768,
        "batch_size": 16,
        "grad_accum": 4,       # effective batch = 64
        "epochs": 2,
        "lr": 2e-5,
    },
}

TRIPLETS_PATH = Path("data/train/triplets.jsonl")
CHECKPOINTS_ROOT = Path("models/checkpoints/encoder")
SEED = 42
DEV_QUERIES = 500    # unique query-ids for the in-loop evaluator (train split only)
SMOKE_N = 300        # triplets loaded in smoke test
SMOKE_STEPS = 20     # max training steps in smoke test
SMOKE_DEV_Q = 50     # dev queries used during smoke


# -- model construction ---------------------------------------------------------

def build_model(cfg: dict) -> SentenceTransformer:
    """Construct a SentenceTransformer for the given base.

    LegalBERT is explicitly wrapped with mean-pooling to match v1 lineage.
    MiniLM is loaded directly (already ships with correct pooling).
    """
    if cfg["is_st_model"]:
        model = SentenceTransformer(cfg["model_id"])
        model.max_seq_length = cfg["max_seq_length"]
    else:
        transformer = st_models.Transformer(
            cfg["model_id"], max_seq_length=cfg["max_seq_length"]
        )
        pooling = st_models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        model = SentenceTransformer(modules=[transformer, pooling])
    return model


# -- data helpers ---------------------------------------------------------------

def load_triplets(path: Path, n: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def build_dev_evaluator(
    triplets: list[dict], n_queries: int = DEV_QUERIES
) -> InformationRetrievalEvaluator:
    """Build an IRE from a random sample of unique TRAIN-split queries.

    Uses only the training triplets - never touches data/eval/.
    """
    by_src: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in triplets:
        by_src[row["src_id"]].append((row["pos_id"], row["query"], row["positive"]))

    rng = random.Random(SEED)
    sampled_ids = rng.sample(list(by_src.keys()), min(n_queries, len(by_src)))

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for src_id in sampled_ids:
        pairs = by_src[src_id]
        queries[src_id] = pairs[0][1]          # same query text across all its positives
        relevant_docs[src_id] = set()
        for pos_id, _, pos_text in pairs:
            corpus[pos_id] = pos_text
            relevant_docs[src_id].add(pos_id)

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="dev",
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10],
        mrr_at_k=[10],
        batch_size=64,
        show_progress_bar=False,
    )


def triplets_to_dataset(triplets: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "anchor":   [t["query"]    for t in triplets],
        "positive": [t["positive"] for t in triplets],
        "negative": [t["hard_neg"] for t in triplets],
    })


# -- VRAM helpers ---------------------------------------------------------------

def vram_used_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def preflight_vram_check() -> None:
    """Warn if another process is already holding VRAM (e.g. Ollama still running)."""
    used = vram_used_gb()
    if used > 0.5:
        print(f"WARNING: {used:.2f} GB VRAM already in use. Run 'just gpu-free' first.")
        print("         Training while Ollama/other models hold VRAM risks OOM.\n")


# -- shared training args factory -----------------------------------------------

def make_args(
    out_dir: Path,
    cfg: dict,
    *,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    use_evaluator: bool = False,
) -> SentenceTransformerTrainingArguments:
    common = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        bf16=True,
        gradient_checkpointing=True,
        seed=SEED,
        dataloader_num_workers=0,   # Windows: fork-based workers hang
        report_to="none",
    )
    if max_steps is not None:
        common["max_steps"] = max_steps
        common["eval_strategy"] = "no"
        common["save_strategy"] = "no"
        common["logging_steps"] = max(1, max_steps // 4)
    else:
        common["num_train_epochs"] = cfg["epochs"]
        common["warmup_ratio"] = 0.10
        common["logging_steps"] = 50
        if use_evaluator and eval_steps:
            common["eval_strategy"] = "steps"
            common["eval_steps"] = eval_steps
            common["save_strategy"] = "steps"
            common["save_steps"] = eval_steps
            common["load_best_model_at_end"] = True
            common["metric_for_best_model"] = "dev_cosine_ndcg@10"
            common["greater_is_better"] = True
        else:
            common["eval_strategy"] = "no"
            common["save_strategy"] = "no"

    return SentenceTransformerTrainingArguments(**common)


# -- smoke test -----------------------------------------------------------------

def run_smoke(base: str, cfg: dict, run_id: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  SMOKE TEST - {base.upper()}")
    print(f"  {SMOKE_N} triplets | {SMOKE_STEPS} steps | bf16 | seq {cfg['max_seq_length']}")
    print(f"{'='*62}\n")

    preflight_vram_check()
    triplets = load_triplets(TRIPLETS_PATH, SMOKE_N)
    model = build_model(cfg)
    dataset = triplets_to_dataset(triplets)
    loss = MultipleNegativesRankingLoss(model)

    out_dir = CHECKPOINTS_ROOT / f"{base}-smoke-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = make_args(out_dir, cfg, max_steps=SMOKE_STEPS)

    reset_peak_vram()
    t0 = time.time()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    train_result = trainer.train()
    wall = time.time() - t0
    peak = peak_vram_gb()
    headroom = 8.5 - peak

    print(f"\n-- Smoke test complete ------------------------------------------")
    print(f"  Wall-clock  : {wall:.1f}s")
    print(f"  Peak VRAM   : {peak:.2f} GB  (headroom: {headroom:.2f} GB)")
    print(f"  Train loss  : {train_result.training_loss:.4f}")
    if headroom >= 0.5:
        print(f"  VRAM OK - safe to launch full run")
    else:
        print(f"  VRAM TIGHT - reduce batch_size or max_seq_length before full run")
    print(f"{'='*62}\n")

    manifest = {
        "base": base,
        "model_id": cfg["model_id"],
        "dim": cfg["dim"],
        "run_id": run_id,
        "mode": "smoke",
        "max_steps": SMOKE_STEPS,
        "n_triplets": len(triplets),
        "batch_size": cfg["batch_size"],
        "grad_accum": cfg["grad_accum"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum"],
        "max_seq_length": cfg["max_seq_length"],
        "lr": cfg["lr"],
        "seed": SEED,
        "peak_vram_gb": round(peak, 3),
        "headroom_gb": round(headroom, 3),
        "wall_clock_s": round(wall, 1),
        "training_loss": round(train_result.training_loss, 4),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest ? {out_dir / 'train_manifest.json'}")
    return manifest


# -- full training run ----------------------------------------------------------

def run_full(base: str, cfg: dict, run_id: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  FULL TRAINING - {base.upper()}")
    print(f"  Loading all triplets ...")
    print(f"{'='*62}\n")

    preflight_vram_check()
    triplets = load_triplets(TRIPLETS_PATH)
    print(f"  Loaded {len(triplets):,} triplets")

    model = build_model(cfg)
    evaluator = build_dev_evaluator(triplets)
    dataset = triplets_to_dataset(triplets)
    loss = MultipleNegativesRankingLoss(model)

    out_dir = CHECKPOINTS_ROOT / f"{base}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # eval ~6x per epoch so early-stopping has enough signal within a single epoch
    steps_per_epoch = len(triplets) // (cfg["batch_size"] * cfg["grad_accum"])
    eval_steps = max(steps_per_epoch // 6, 200)

    # frozen baseline before any weight update
    print("-- Evaluating frozen baseline ...")
    reset_peak_vram()
    before_metrics = evaluator(model, output_path=str(out_dir))
    before_ndcg = before_metrics["dev_cosine_ndcg@10"]
    print(f"   Frozen baseline  dev nDCG@10 = {before_ndcg:.4f}\n")

    args = make_args(
        out_dir, cfg, eval_steps=eval_steps, use_evaluator=True
    )

    reset_peak_vram()
    t0 = time.time()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_result = trainer.train()
    wall = time.time() - t0
    peak = peak_vram_gb()

    # save best model (load_best_model_at_end restored it into `model`)
    model_dir = out_dir / "model"
    model.save(str(model_dir))

    # eval after training
    print("\n-- Evaluating trained model ...")
    after_metrics = evaluator(model, output_path=str(out_dir))
    after_ndcg = after_metrics["dev_cosine_ndcg@10"]
    lift = after_ndcg - before_ndcg

    expected_steps = (len(triplets) // (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
    actual_steps = train_result.global_step
    early_stopped = actual_steps < expected_steps

    print(f"\n{'='*62}")
    print(f"  RESULTS - {base.upper()}")
    print(f"  Frozen baseline  nDCG@10 : {before_ndcg:.4f}")
    print(f"  Trained model    nDCG@10 : {after_ndcg:.4f}  (lift: {lift:+.4f})")
    print(f"  Peak VRAM                : {peak:.2f} GB")
    print(f"  Wall-clock               : {wall/3600:.2f}h ({wall:.0f}s)")
    print(f"  Early stopped            : {early_stopped} (step {actual_steps}/{expected_steps})")
    if lift > 0:
        print(f"  LIFT POSITIVE - proceed to next base / reranker")
    else:
        print(f"  NO LIFT - debug data/mining before continuing")
    print(f"{'='*62}\n")

    manifest = {
        "base": base,
        "model_id": cfg["model_id"],
        "dim": cfg["dim"],
        "run_id": run_id,
        "mode": "full",
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "per_device_batch": cfg["batch_size"],
        "grad_accum": cfg["grad_accum"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum"],
        "max_seq_length": cfg["max_seq_length"],
        "seed": SEED,
        "num_triplets": len(triplets),
        "eval_steps": eval_steps,
        "early_stopped": early_stopped,
        "early_stop_step": actual_steps if early_stopped else None,
        "expected_total_steps": expected_steps,
        "dev_ndcg10_before": round(before_ndcg, 4),
        "dev_ndcg10_after": round(after_ndcg, 4),
        "dev_ndcg10_lift": round(lift, 4),
        "peak_vram_gb": round(peak, 3),
        "wall_clock_s": round(wall, 1),
        "training_loss": round(train_result.training_loss, 4),
        "checkpoint_dir": str(model_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_note": (
            "dev_ndcg10 is measured on a small positives-only dev corpus sampled "
            "from training triplets. It is NOT comparable to the full-corpus "
            "citation-grounded eval (qrels_citation.jsonl). Use the citation-grounded nDCG@10 as the headline metric."
        ),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest ? {out_dir / 'train_manifest.json'}")
    print(f"Model    ? {model_dir}")
    return manifest


# -- entry point ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder fine-tuning (dual-encoder, MNRL)")
    parser.add_argument(
        "--base", choices=list(BASE_CONFIGS), required=True,
        help="Which encoder base to fine-tune",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: verify VRAM + loss on a tiny slice before full run",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override per-device train batch size (default: base config value)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None,
        help="Override gradient accumulation steps (default: base config value)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs (default: base config value)",
    )
    args = parser.parse_args()

    cfg = dict(BASE_CONFIGS[args.base])  # copy so we don't mutate the global
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg["grad_accum"] = args.grad_accum
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        run_smoke(args.base, cfg, run_id)
    else:
        run_full(args.base, cfg, run_id)


if __name__ == "__main__":
    main()
