"""LLM judge for PA legal case relevance.

Distinct from the training-pair mining pipeline.
"""

import argparse
import json
import random
from pathlib import Path

import anthropic

from config.settings import settings

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


# ── Constants ─────────────────────────────────────────────────────────────────

RUBRIC_VERSION = "v2"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

RUBRIC = """\
You are evaluating Pennsylvania legal case relevance for a legal research system.

TASK: Determine whether the candidate opinion is relevant to the query opinion.

RELEVANT: The candidate opinion addresses the same legal issue, rule, statute, or doctrine \
as the query and would be useful precedent for the query's legal question. This includes \
cases establishing the same legal rule or test, cases with closely analogous facts that set \
applicable precedent, cases interpreting the same statute or constitutional provision, and \
cases on the same procedural or substantive legal question.

NOT_RELEVANT: The candidate opinion does not share a meaningful legal connection with the \
query. Superficial similarities (same court, same time period, or same parties type) without \
shared legal substance do not qualify.

Respond in exactly this format — no preamble, nothing after REASON:
LABEL: [RELEVANT or NOT_RELEVANT]
REASON: [One sentence explaining the key legal connection or its absence]

---
Query opinion:
{query_text}

---
Candidate opinion:
{doc_text}
"""

QUERY_TEXT_MAX = 1500   # chars fed to the judge for the query side
DOC_TEXT_MAX = 2500     # chars fed to the judge for the doc side

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
QRELS_PATH = ROOT / "data/eval/qrels_citation.jsonl"
BM25_RUN_PATH = ROOT / "data/eval/runs/bm25.jsonl"
CASES_PATH = ROOT / "data/parsed/cases.jsonl"
VALIDATION_SHEET_PATH = ROOT / "data/eval/judge_validation_sheet.jsonl"
JUDGE_LABELS_PATH = ROOT / "data/eval/judge_labels.jsonl"
META_PATH = ROOT / "data/eval/judge_validation_meta.jsonl"

# ── Sample configuration ──────────────────────────────────────────────────────

SAMPLE_SEED = 42
N_RELEVANT = 27
N_BORDERLINE = 27
N_IRRELEVANT = 26

# Per-curiam orders / memorandum affirmances with no legal analysis — unjudgeable.
CONTENT_FREE_EXCLUDED: frozenset[str] = frozenset(
    {"pair_0043", "pair_0044", "pair_0074", "pair_0075"}
)


# ── Core judge function ───────────────────────────────────────────────────────

def call_judge(query_text: str, doc_text: str, model: str = JUDGE_MODEL) -> dict:
    """Call the LLM judge. Returns {label, raw, model, rubric_version}."""
    prompt = RUBRIC.format(
        query_text=query_text[:QUERY_TEXT_MAX],
        doc_text=doc_text[:DOC_TEXT_MAX],
    )
    message = _get_client().messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    label = "PARSE_ERROR"
    for line in raw.splitlines():
        if line.upper().startswith("LABEL:"):
            val = line.split(":", 1)[1].strip().upper()
            if val == "RELEVANT":
                label = "RELEVANT"
            elif "NOT_RELEVANT" in val or val == "NOT RELEVANT":
                label = "NOT_RELEVANT"
            break

    return {"label": label, "raw": raw, "model": model, "rubric_version": RUBRIC_VERSION}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_qrels() -> dict:
    """Load qrels as {query_id: {relevant_doc_ids: set}}.
    Query text is intentionally omitted — loaded from cases.jsonl instead.
    """
    qrels: dict = {}
    with open(QRELS_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qrels[obj["query_id"]] = {
                "relevant_doc_ids": set(obj["relevant_doc_ids"]),
            }
    return qrels


def _load_bm25_run() -> dict:
    runs: dict = {}
    with open(BM25_RUN_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            runs[obj["qid"]] = obj["doc_ids"]
    return runs


def _load_case_texts(needed_ids: set) -> dict:
    """Single pass through cases.jsonl; returns {id: {head_matter, full_text}}."""
    texts: dict = {}
    print(f"  Scanning cases.jsonl for {len(needed_ids)} cases...")
    found = 0
    with open(CASES_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 50_000 == 0 and i > 0:
                print(f"    {i:,} lines scanned, {found}/{len(needed_ids)} found", flush=True)
            obj = json.loads(line)
            if obj["id"] in needed_ids:
                texts[obj["id"]] = {
                    "head_matter": obj.get("head_matter", ""),
                    "full_text": obj.get("full_text", ""),
                }
                found += 1
                if found == len(needed_ids):
                    break
    print(f"  Done. Found {found}/{len(needed_ids)} cases.")
    return texts


# ── Era-aware text extraction ─────────────────────────────────────────────────

def _extract_text(head_matter: str, full_text: str, max_chars: int) -> str:
    """Return the most legally diagnostic N chars for a case.

    Old cases (head_matter > 800 chars): head_matter is a rich numbered
    syllabus — it is the gold summary of legal holdings.

    Modern cases (head_matter <= 800 chars): head_matter is pure caption
    (parties, court, dates, attorneys). Skip it; take the opinion body.
    """
    if len(head_matter) > 800:
        return head_matter[:max_chars]
    else:
        return full_text[len(head_matter):][:max_chars]


# ── Sample construction ───────────────────────────────────────────────────────

def build_validation_sample() -> list:
    """Build stratified ~80-pair sample across three difficulty bands."""
    rng = random.Random(SAMPLE_SEED)

    print("Loading qrels...")
    qrels = _load_qrels()
    print("Loading BM25 run...")
    bm25_runs = _load_bm25_run()

    qids = list(qrels.keys())
    qid_list = list(qrels.keys())   # stable copy for donor selection
    used_pairs: set = set()
    pairs: list = []
    counts = {"relevant": 0, "borderline": 0, "irrelevant": 0}

    # Band 1: clearly-relevant — citation-linked pairs
    rng.shuffle(qids)
    for qid in qids:
        if counts["relevant"] >= N_RELEVANT:
            break
        rel_docs = list(qrels[qid]["relevant_doc_ids"])
        if not rel_docs:
            continue
        doc_id = rng.choice(rel_docs)
        if (qid, doc_id) in used_pairs:
            continue
        used_pairs.add((qid, doc_id))
        pairs.append({"band": "relevant", "query_id": qid, "doc_id": doc_id})
        counts["relevant"] += 1

    # Band 2: borderline — BM25 ranks 20-150 (0-indexed 19-149), not in relevant set
    rng.shuffle(qids)
    for qid in qids:
        if counts["borderline"] >= N_BORDERLINE:
            break
        if qid not in bm25_runs:
            continue
        bm25_docs = bm25_runs[qid]
        rel_set = qrels[qid]["relevant_doc_ids"]
        candidates = [d for d in bm25_docs[19:150] if d not in rel_set]
        if not candidates:
            continue
        doc_id = rng.choice(candidates)
        if (qid, doc_id) in used_pairs:
            continue
        used_pairs.add((qid, doc_id))
        pairs.append({"band": "borderline", "query_id": qid, "doc_id": doc_id})
        counts["borderline"] += 1

    # Band 3: clearly-irrelevant — relevant doc of a different query, not retrieved for Q1
    rng.shuffle(qids)
    for qid in qids:
        if counts["irrelevant"] >= N_IRRELEVANT:
            break
        rel_set = qrels[qid]["relevant_doc_ids"]
        bm25_top = set(bm25_runs.get(qid, []))
        for _ in range(30):
            qid2 = rng.choice(qid_list)
            if qid2 == qid:
                continue
            donor_docs = [
                d for d in qrels[qid2]["relevant_doc_ids"]
                if d not in rel_set and d not in bm25_top
            ]
            if donor_docs:
                doc_id = rng.choice(donor_docs)
                if (qid, doc_id) not in used_pairs:
                    used_pairs.add((qid, doc_id))
                    pairs.append({"band": "irrelevant", "query_id": qid, "doc_id": doc_id})
                    counts["irrelevant"] += 1
                    break

    print(
        f"\nSampled: {counts['relevant']} relevant, "
        f"{counts['borderline']} borderline, "
        f"{counts['irrelevant']} irrelevant  (total={len(pairs)})"
    )

    # Load head_matter + full_text for ALL query and doc cases in one pass
    needed_ids = {p["doc_id"] for p in pairs} | {p["query_id"] for p in pairs}
    case_texts = _load_case_texts(needed_ids)

    # Assemble with era-aware extraction, shuffle, assign pair_ids
    result = []
    for p in pairs:
        qcase = case_texts.get(p["query_id"], {})
        dcase = case_texts.get(p["doc_id"], {})
        result.append({
            "band": p["band"],
            "query_id": p["query_id"],
            "doc_id": p["doc_id"],
            "query_text": _extract_text(
                qcase.get("head_matter", ""),
                qcase.get("full_text", ""),
                QUERY_TEXT_MAX,
            ),
            "doc_text": _extract_text(
                dcase.get("head_matter", ""),
                dcase.get("full_text", ""),
                DOC_TEXT_MAX,
            ),
        })

    rng.shuffle(result)
    for i, p in enumerate(result):
        p["pair_id"] = f"pair_{i:04d}"

    return result


# ── Sheet rebuild from existing meta ─────────────────────────────────────────

def rebuild_validation_sheet(meta_path: Path, sheet_path: Path) -> None:
    """Rebuild the labeling sheet from existing meta using era-aware extraction.

    Preserves the 80-pair composition and pair_ids from meta. Resets
    my_label to null — text representation changed so re-labeling is required.
    """
    meta_pairs = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            meta_pairs.append(json.loads(line))

    needed_ids = {m["query_id"] for m in meta_pairs} | {m["doc_id"] for m in meta_pairs}
    case_texts = _load_case_texts(needed_ids)

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sheet_path, "w", encoding="utf-8") as f:
        for m in meta_pairs:
            qcase = case_texts.get(m["query_id"], {})
            dcase = case_texts.get(m["doc_id"], {})
            row = {
                "pair_id": m["pair_id"],
                "query_text": _extract_text(
                    qcase.get("head_matter", ""),
                    qcase.get("full_text", ""),
                    QUERY_TEXT_MAX,
                ),
                "doc_text": _extract_text(
                    dcase.get("head_matter", ""),
                    dcase.get("full_text", ""),
                    DOC_TEXT_MAX,
                ),
                "my_label": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Rebuilt {len(meta_pairs)}-pair validation sheet -> {sheet_path}")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_validation_sheet(pairs: list, sheet_path: Path, meta_path: Path) -> None:
    """Write the labeling sheet (no band) and a metadata file (with band)."""
    sheet_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sheet_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "pair_id": p["pair_id"],
                "query_text": p["query_text"],
                "doc_text": p["doc_text"],
                "my_label": None,
            }, ensure_ascii=False) + "\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "pair_id": p["pair_id"],
                "band": p["band"],
                "query_id": p["query_id"],
                "doc_id": p["doc_id"],
            }, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)}-pair validation sheet -> {sheet_path}")
    print(f"Saved band metadata                      -> {meta_path}")


def run_judge_on_sample(
    sheet_path: Path, labels_path: Path, model: str = JUDGE_MODEL
) -> None:
    """Run the judge on all pairs in the sheet; write judge_labels.jsonl."""
    pairs = []
    with open(sheet_path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    excluded = [p["pair_id"] for p in pairs if p["pair_id"] in CONTENT_FREE_EXCLUDED]
    pairs = [p for p in pairs if p["pair_id"] not in CONTENT_FREE_EXCLUDED]
    if excluded:
        print(f"Excluded {len(excluded)} content-free pairs: {excluded}")

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, pair in enumerate(pairs):
        print(f"  [{i+1:3d}/{len(pairs)}] {pair['pair_id']} ...", end=" ", flush=True)
        result = call_judge(pair["query_text"], pair["doc_text"], model=model)
        results.append({
            "pair_id": pair["pair_id"],
            "judge_label": result["label"],
            "judge_raw": result["raw"],
            "model": result["model"],
            "rubric_version": result["rubric_version"],
        })
        print(result["label"])

    with open(labels_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rel = sum(1 for r in results if r["judge_label"] == "RELEVANT")
    print(f"\nSaved {len(results)} labels -> {labels_path}")
    print(f"  RELEVANT={rel}  NOT_RELEVANT={len(results)-rel}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM judge for PA legal case relevance")
    parser.add_argument(
        "--build-sample", action="store_true",
        help="Build stratified validation sample from scratch (saves sheet + metadata)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild sheet from existing metadata with era-aware text extraction"
    )
    parser.add_argument(
        "--run-judge", action="store_true",
        help="Run judge on the validation sheet (makes Anthropic API calls)"
    )
    parser.add_argument(
        "--model", default=JUDGE_MODEL,
        help=f"Judge model string (default: {JUDGE_MODEL})"
    )
    args = parser.parse_args()

    if args.build_sample:
        pairs = build_validation_sample()
        save_validation_sheet(pairs, VALIDATION_SHEET_PATH, META_PATH)
    elif args.rebuild:
        rebuild_validation_sheet(META_PATH, VALIDATION_SHEET_PATH)
    elif args.run_judge:
        run_judge_on_sample(VALIDATION_SHEET_PATH, JUDGE_LABELS_PATH, model=args.model)
    else:
        parser.print_help()
