"""
Cross-encoder reranker fine-tuning.

Base: BAAI/bge-reranker-base (~110M, BERT-base), replacing v1's frozen 560M bge-reranker-large.
Loss: BinaryCrossEntropyLoss on (query, doc, label) pairs.
Data: triplets.jsonl from pair mining -- each triplet yields 1 positive + 1 negative CE example.

Workflow:
  1. python -m training.train_reranker --smoke   # verify VRAM + loss drop (~3 min)
  2. python -m training.train_reranker           # full run (~12-15 h for 1 epoch)
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
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from transformers import EarlyStoppingCallback

# -- config --------------------------------------------------------------------

MODEL_ID = "BAAI/bge-reranker-base"   # BERT-base cross-encoder, ~110M params

CFG = {
    "model_id": MODEL_ID,
    "max_seq_length": 384,  # spec cap: 256-384; matches encoder runs, cuts per-step cost
    "batch_size": 32,
    "grad_accum": 4,         # effective batch = 128
    "epochs": 1,
    "lr": 2e-5,
    "warmup_ratio": 0.10,
    "early_stopping_patience": 2,
}

TRIPLETS_PATH = Path("data/train/triplets.jsonl")
CHECKPOINTS_ROOT = Path("models/checkpoints/reranker")
SEED = 42

DEV_QUERIES = 500    # unique query-ids for the dev evaluator (train split only)
SMOKE_N = 200        # triplets loaded in smoke mode
SMOKE_STEPS = 20     # max training steps in smoke mode


# -- data helpers --------------------------------------------------------------

def load_triplets(path: Path, n: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def triplets_to_ce_dataset(triplets: list[dict]) -> Dataset:
    """Expand triplets -> 2x CE (query, doc, label) examples: 1 positive + 1 negative each."""
    queries, docs, labels = [], [], []
    for t in triplets:
        queries.append(t["query"])
        docs.append(t["positive"])
        labels.append(1.0)
        queries.append(t["query"])
        docs.append(t["hard_neg"])
        labels.append(0.0)
    return Dataset.from_dict({"query": queries, "response": docs, "label": labels})


def build_dev_evaluator(triplets: list[dict]) -> CrossEncoderRerankingEvaluator:
    """Build reranking eval samples from a random train-split sample. Never uses data/eval/."""
    by_src: dict[str, list[dict]] = defaultdict(list)
    for t in triplets:
        by_src[t["src_id"]].append(t)

    rng = random.Random(SEED)
    sampled_ids = rng.sample(list(by_src.keys()), min(DEV_QUERIES, len(by_src)))

    samples = []
    for src_id in sampled_ids:
        rows = by_src[src_id]
        positives = list(dict.fromkeys(r["positive"] for r in rows))   # dedup, order-stable
        negatives = list(dict.fromkeys(r["hard_neg"] for r in rows))
        if positives and negatives:
            samples.append({"query": rows[0]["query"], "positive": positives, "negative": negatives})

    return CrossEncoderRerankingEvaluator(
        samples=samples, at_k=10, name="dev", show_progress_bar=False
    )


# -- VRAM helpers (same pattern as train_encoder) ------------------------------

def vram_used_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def preflight_vram_check() -> None:
    used = vram_used_gb()
    if used > 0.5:
        print(f"WARNING: {used:.2f} GB VRAM already in use. Run 'just gpu-free' first.")
        print("         Training while Ollama/other models hold VRAM risks OOM.\n")


# -- training args factory -----------------------------------------------------

def make_args(
    out_dir: Path,
    cfg: dict,
    *,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    use_evaluator: bool = False,
) -> CrossEncoderTrainingArguments:
    common: dict = dict(
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
        common["warmup_ratio"] = cfg["warmup_ratio"]
        common["logging_steps"] = 100
        if use_evaluator and eval_steps:
            common["eval_strategy"] = "steps"
            common["eval_steps"] = eval_steps
            common["save_strategy"] = "steps"
            common["save_steps"] = eval_steps
            common["load_best_model_at_end"] = True
            # CrossEncoderRerankingEvaluator with name="dev", at_k=10 returns "dev_ndcg@10"
            common["metric_for_best_model"] = "dev_ndcg@10"
            common["greater_is_better"] = True
        else:
            common["eval_strategy"] = "no"
            common["save_strategy"] = "no"
    return CrossEncoderTrainingArguments(**common)


# -- smoke test ----------------------------------------------------------------

def run_smoke(cfg: dict, run_id: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  SMOKE TEST -- reranker")
    print(f"  base : {cfg['model_id']}")
    print(f"  {SMOKE_N} triplets -> {SMOKE_N * 2} CE pairs | {SMOKE_STEPS} steps")
    print(f"  bf16 | seq {cfg['max_seq_length']} | batch {cfg['batch_size']} x accum {cfg['grad_accum']}")
    print(f"{'='*62}\n")

    preflight_vram_check()
    triplets = load_triplets(TRIPLETS_PATH, SMOKE_N)

    model = CrossEncoder(cfg["model_id"])
    model.max_seq_length = cfg["max_seq_length"]

    dataset = triplets_to_ce_dataset(triplets)
    loss = BinaryCrossEntropyLoss(model)

    out_dir = CHECKPOINTS_ROOT / f"smoke-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = make_args(out_dir, cfg, max_steps=SMOKE_STEPS)

    reset_peak_vram()
    t0 = time.time()
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    result = trainer.train()
    wall = time.time() - t0
    peak = peak_vram_gb()
    headroom = 8.5 - peak

    print(f"\n-- Smoke complete -----------------------------------------------")
    print(f"  Wall-clock  : {wall:.1f}s  ({wall / SMOKE_STEPS:.2f}s/step)")
    print(f"  Peak VRAM   : {peak:.2f} GB  (headroom: {headroom:.2f} GB)")
    print(f"  Train loss  : {result.training_loss:.4f}")
    if headroom >= 0.5:
        print(f"  VRAM OK -- safe to launch full run")
    else:
        print(f"  VRAM TIGHT -- reduce batch_size or max_seq_length before full run")
    print(f"{'='*62}\n")

    manifest = {
        "model_id": cfg["model_id"],
        "run_id": run_id,
        "mode": "smoke",
        "max_steps": SMOKE_STEPS,
        "max_seq_length": cfg["max_seq_length"],
        "batch_size": cfg["batch_size"],
        "grad_accum": cfg["grad_accum"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum"],
        "lr": cfg["lr"],
        "seed": SEED,
        "peak_vram_gb": round(peak, 3),
        "headroom_gb": round(headroom, 3),
        "wall_clock_s": round(wall, 1),
        "step_time_s": round(wall / SMOKE_STEPS, 2),
        "training_loss": round(result.training_loss, 4),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest -> {out_dir / 'train_manifest.json'}")
    return manifest


# -- full training run ---------------------------------------------------------

def run_full(cfg: dict, run_id: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  FULL TRAINING -- reranker ({cfg['model_id']})")
    print(f"  Loading triplets from {TRIPLETS_PATH} ...")
    print(f"{'='*62}\n")

    preflight_vram_check()
    triplets = load_triplets(TRIPLETS_PATH)
    n_triplets = len(triplets)
    n_ce_examples = n_triplets * 2
    print(f"  Loaded {n_triplets:,} triplets -> {n_ce_examples:,} CE (query, doc, label) examples\n")

    model = CrossEncoder(cfg["model_id"])
    model.max_seq_length = cfg["max_seq_length"]

    dataset = triplets_to_ce_dataset(triplets)
    evaluator = build_dev_evaluator(triplets)
    loss = BinaryCrossEntropyLoss(model)

    out_dir = CHECKPOINTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # eval ~10x per epoch so early-stopping has enough signal
    steps_per_epoch = n_ce_examples // (cfg["batch_size"] * cfg["grad_accum"])
    eval_steps = max(steps_per_epoch // 10, 200)

    # frozen baseline before any weight update
    print("-- Evaluating frozen baseline ...")
    reset_peak_vram()
    before_metrics = evaluator(model, output_path=str(out_dir))
    before_ndcg = before_metrics.get("dev_ndcg@10", 0.0)
    print(f"   Frozen baseline  dev nDCG@10 = {before_ndcg:.4f}\n")

    args = make_args(out_dir, cfg, eval_steps=eval_steps, use_evaluator=True)

    reset_peak_vram()
    t0 = time.time()
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )
    train_result = trainer.train()
    wall = time.time() - t0
    peak = peak_vram_gb()

    model_dir = out_dir / "model"
    model.save(str(model_dir))

    print("\n-- Evaluating trained reranker ...")
    after_metrics = evaluator(model, output_path=str(out_dir))
    after_ndcg = after_metrics.get("dev_ndcg@10", 0.0)
    lift = after_ndcg - before_ndcg

    expected_steps = steps_per_epoch * cfg["epochs"]
    actual_steps = train_result.global_step
    early_stopped = actual_steps < expected_steps

    print(f"\n{'='*62}")
    print(f"  RESULTS -- reranker")
    print(f"  Frozen baseline  nDCG@10 : {before_ndcg:.4f}")
    print(f"  Trained reranker nDCG@10 : {after_ndcg:.4f}  (lift: {lift:+.4f})")
    print(f"  Peak VRAM                : {peak:.2f} GB")
    print(f"  Wall-clock               : {wall / 3600:.2f}h ({wall:.0f}s)")
    print(f"  Early stopped            : {early_stopped} (step {actual_steps}/{expected_steps})")
    if lift <= 0:
        print(f"  NO LIFT -- debug data / negatives before running eval")
    print(f"{'='*62}\n")

    manifest = {
        "model_id": cfg["model_id"],
        "run_id": run_id,
        "mode": "full",
        "max_seq_length": cfg["max_seq_length"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "warmup_ratio": cfg["warmup_ratio"],
        "per_device_batch": cfg["batch_size"],
        "grad_accum": cfg["grad_accum"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum"],
        "seed": SEED,
        "num_triplets": n_triplets,
        "num_ce_examples": n_ce_examples,
        "steps_per_epoch": steps_per_epoch,
        "eval_steps": eval_steps,
        "early_stop_patience": cfg["early_stopping_patience"],
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
            "dev_ndcg10 is a training thermometer measured on a dev slice of training "
            "triplets (500 query-ids, train split only). It is NOT comparable to the "
            "full-corpus citation-grounded eval. Use the citation-grounded nDCG@10 as the "
            "headline metric."
        ),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest -> {out_dir / 'train_manifest.json'}")
    print(f"Model    -> {model_dir}")
    return manifest


# -- entry point ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-encoder reranker fine-tuning")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: VRAM + loss check")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()

    cfg = dict(CFG)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg["grad_accum"] = args.grad_accum
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        run_smoke(cfg, run_id)
    else:
        run_full(cfg, run_id)


if __name__ == "__main__":
    main()
