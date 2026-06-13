"""Compute Cohen's kappa between human labels and LLM judge labels.

Fill in my_label (RELEVANT or NOT_RELEVANT) in data/eval/judge_validation_sheet.jsonl,
then run:
    python -m eval.compute_kappa
"""

import json
from pathlib import Path

from eval.judge import CONTENT_FREE_EXCLUDED

ROOT = Path(__file__).parent.parent
VALIDATION_SHEET_PATH = ROOT / "data/eval/judge_validation_sheet.jsonl"
JUDGE_LABELS_PATH = ROOT / "data/eval/judge_labels.jsonl"
META_PATH = ROOT / "data/eval/judge_validation_meta.jsonl"

CLASSES = ["RELEVANT", "NOT_RELEVANT"]


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    classes = sorted(set(labels_a) | set(labels_b))
    cm: dict = {}
    for a, b in zip(labels_a, labels_b):
        cm[(a, b)] = cm.get((a, b), 0) + 1
    po = sum(cm.get((c, c), 0) for c in classes) / n
    pe = sum(
        (sum(cm.get((c, x), 0) for x in classes) / n)
        * (sum(cm.get((x, c), 0) for x in classes) / n)
        for c in classes
    )
    return 0.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def main() -> None:
    human: dict = {}
    with open(VALIDATION_SHEET_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("my_label"):
                human[obj["pair_id"]] = obj["my_label"].strip().upper()

    judge: dict = {}
    with open(JUDGE_LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            judge[obj["pair_id"]] = obj["judge_label"]

    meta: dict = {}
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                meta[obj["pair_id"]] = obj["band"]

    common = [
        pid for pid in human
        if pid in judge
        and pid not in CONTENT_FREE_EXCLUDED
        and judge[pid] != "PARSE_ERROR"
    ]
    parse_errors = [pid for pid in judge if judge[pid] == "PARSE_ERROR"]
    if parse_errors:
        print(f"WARNING: {len(parse_errors)} PARSE_ERROR judge labels excluded from kappa: {parse_errors}")
    if not common:
        print("No labeled pairs found. Fill in my_label in the validation sheet first.")
        return

    h = [human[pid] for pid in common]
    j = [judge[pid] for pid in common]
    n = len(common)
    agreed = sum(a == b for a, b in zip(h, j))

    print(f"\n=== Judge Validation ({n} pairs) ===")
    print(f"Raw agreement : {agreed/n:.3f}  ({agreed}/{n})")
    print(f"Cohen's kappa : {cohen_kappa(h, j):.3f}")

    print(f"\nConfusion matrix (rows=human, cols=judge):")
    header = f"{'':>16}" + "  ".join(f"{c:>14}" for c in CLASSES)
    print(header)
    for hc in CLASSES:
        vals = [sum(1 for a, b in zip(h, j) if a == hc and b == jc) for jc in CLASSES]
        print(f"{hc:>16}" + "  ".join(f"{v:>14d}" for v in vals))

    if meta:
        print("\nBand breakdown:")
        for band in ("relevant", "borderline", "irrelevant"):
            bids = [pid for pid in common if meta.get(pid) == band]
            if not bids:
                continue
            bh = [human[pid] for pid in bids]
            bj = [judge[pid] for pid in bids]
            ba = sum(a == b for a, b in zip(bh, bj)) / len(bids)
            bk = cohen_kappa(bh, bj)
            print(f"  {band:>12}: n={len(bids):2d}  agreement={ba:.3f}  kappa={bk:.3f}")


if __name__ == "__main__":
    main()
