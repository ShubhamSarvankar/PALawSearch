"""Interactive terminal labeler for judge_validation_sheet.jsonl.

Keys: r = RELEVANT  |  n = NOT_RELEVANT  |  s = skip  |  q = quit
Resume-safe: already-labeled pairs are skipped on restart.
Writes each label back immediately (atomic .tmp → rename).
"""

import json
import os
import sys
import textwrap
from pathlib import Path

SHEET_PATH = Path(__file__).parent.parent / "data/eval/judge_validation_sheet.jsonl"
QUERY_SHOW = 2000   # chars displayed from query_text
DOC_SHOW = 3000     # chars displayed from doc_text

try:
    _cols = os.get_terminal_size().columns
except OSError:
    _cols = 100
WIDTH = min(_cols, 100)


def _load(path: Path) -> list:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def _save(pairs: list, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _wrap(text: str, indent: str = "  ") -> str:
    w = WIDTH - len(indent)
    return textwrap.fill(text, width=w, initial_indent=indent, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


def _show(pair: dict, position: int, total: int, n_labeled: int) -> None:
    sep = "-" * WIDTH
    print(f"\n{sep}")
    print(f"  {pair['pair_id']}   [{position}/{total}]   labeled: {n_labeled}/{total}")
    print(sep)

    qt = pair["query_text"]
    qt_cut = len(qt) > QUERY_SHOW
    print(f"\n  QUERY")
    print(_wrap(qt[:QUERY_SHOW]))
    if qt_cut:
        print(f"  [... query truncated at {QUERY_SHOW} of {len(qt)} chars]")

    print()

    dt = pair["doc_text"]
    dt_cut = len(dt) > DOC_SHOW
    print(f"  CANDIDATE DOCUMENT")
    print(_wrap(dt[:DOC_SHOW]))
    if dt_cut:
        print(f"  [... doc truncated at {DOC_SHOW} of {len(dt)} chars]")

    print(f"\n{sep}")


def main() -> None:
    # Make stdout UTF-8 tolerant on Windows (smart quotes, dashes, etc.)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if not SHEET_PATH.exists():
        print(f"Sheet not found: {SHEET_PATH}")
        return

    pairs = _load(SHEET_PATH)
    total = len(pairs)
    n_done = sum(1 for p in pairs if p.get("my_label") is not None)
    remaining = total - n_done

    print(f"\nLoaded {total} pairs  ({n_done} already labeled, {remaining} remaining)")
    print("Keys:  r = RELEVANT   n = NOT_RELEVANT   s = skip   q = quit\n")

    if remaining == 0:
        print("All pairs are labeled. Run: python -m eval.compute_kappa")
        return

    this_session = 0
    position = 0   # counts only unlabeled pairs shown

    for pair in pairs:
        if pair.get("my_label") is not None:
            continue

        position += 1
        n_labeled = sum(1 for p in pairs if p.get("my_label") is not None)
        _show(pair, position, remaining, n_labeled)

        while True:
            try:
                raw = input("  Label [r/n/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted — progress saved.")
                _save(pairs, SHEET_PATH)
                return

            if raw == "r":
                pair["my_label"] = "RELEVANT"
                _save(pairs, SHEET_PATH)
                this_session += 1
                print("  -> RELEVANT")
                break
            elif raw == "n":
                pair["my_label"] = "NOT_RELEVANT"
                _save(pairs, SHEET_PATH)
                this_session += 1
                print("  -> NOT_RELEVANT")
                break
            elif raw == "s":
                print("  Skipped.")
                break
            elif raw == "q":
                print(f"\nQuit. Labeled {this_session} pairs this session.")
                _save(pairs, SHEET_PATH)
                return
            else:
                print("  Unrecognized key. Type r, n, s, or q.")

    final = sum(1 for p in pairs if p.get("my_label") is not None)
    skipped = total - final
    print(f"\nAll unlabeled pairs shown. Labeled: {final}/{total}")
    if skipped:
        print(f"  ({skipped} still skipped — re-run to fill them in)")
    else:
        print("  Labeling complete! Run: python -m eval.compute_kappa")


if __name__ == "__main__":
    main()
