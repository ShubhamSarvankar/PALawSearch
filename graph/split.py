"""
Case-level train / eval / holdout split.

Decided ONCE here, seeded, committed to disk before any mining begins.
Mining reads this file; it never modifies or regenerates it.

Usage: python -m graph.split   (or: just split)

Reads:  data/parsed/cases.jsonl
Writes: data/graph/split.json           {case_id: "train"|"eval"|"holdout"}
        data/graph/split_manifest.json
"""

import json
import random
from pathlib import Path

PARSED_JSONL  = Path("data/parsed/cases.jsonl")
GRAPH_DIR     = Path("data/graph")
SPLIT_JSON    = GRAPH_DIR / "split.json"
MANIFEST      = GRAPH_DIR / "split_manifest.json"

TRAIN_FRAC    = 0.80
EVAL_FRAC     = 0.10
# holdout gets the remainder so rounding never leaves cases unassigned
SEED          = 42


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading case ids ...")
    case_ids: list[str] = []
    with PARSED_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case_ids.append(json.loads(line)["id"])

    n = len(case_ids)
    print(f"  Total cases: {n:,}")

    rng = random.Random(SEED)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)

    n_train   = int(n * TRAIN_FRAC)
    n_eval    = int(n * EVAL_FRAC)
    n_holdout = n - n_train - n_eval

    split: dict[str, str] = {}
    for cid in shuffled[:n_train]:
        split[cid] = "train"
    for cid in shuffled[n_train : n_train + n_eval]:
        split[cid] = "eval"
    for cid in shuffled[n_train + n_eval :]:
        split[cid] = "holdout"

    assert len(split) == n

    SPLIT_JSON.write_text(json.dumps(split), "utf-8")

    manifest = {
        "total"        : n,
        "train"        : n_train,
        "eval"         : n_eval,
        "holdout"      : n_holdout,
        "train_frac"   : TRAIN_FRAC,
        "eval_frac"    : EVAL_FRAC,
        "seed"         : SEED,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), "utf-8")

    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Train   : {n_train:>10,}  ({n_train/n*100:.1f}%)")
    print(f"  Eval    : {n_eval:>10,}  ({n_eval/n*100:.1f}%)")
    print(f"  Holdout : {n_holdout:>10,}  ({n_holdout/n*100:.1f}%)")
    print(f"  Seed    : {SEED}")
    print(sep)
    print(f"\n  Split    → {SPLIT_JSON}")
    print(f"  Manifest → {MANIFEST}")


if __name__ == "__main__":
    main()
