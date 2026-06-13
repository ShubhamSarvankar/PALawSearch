"""
Canonical case parser — streams raw downloaded zips to cases.jsonl.

From v1 BM25/dense indexers (kept):
  - full_text construction: head_matter + newline + joined opinion texts
  - normalize_decision_date (YYYY / YYYY-MM / YYYY-MM-DD → same or None)

Added:
  - Single CaseRecord dataclass
  - cites_to extraction: top-level only (inspection confirmed opinions[].cites_to
    and extracted_citations are both absent in this CAP export)
  - case_ids are integers in the JSON — stringified to match CaseRecord.target_case_ids
  - PA jurisdiction filter: drops non-PA cases from regional reporters at parse time
  - Dedup by CAP case id (two-pass): keep the record with the highest word_count
  - Citation-density gate numbers printed at end

Usage:
    python ingest/parse.py
    (or: just ingest)

Output:
    data/parsed/cases.jsonl          canonical, deduplicated CaseRecord objects
    data/parsed/ingest_manifest.json counts + citation-density gate numbers
"""

import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

RAW_DIR    = Path("data/raw")
PARSED_DIR = Path("data/parsed")
TEMP_JSONL = PARSED_DIR / "cases_raw.jsonl"    # temp; deleted after dedup
OUT_JSONL  = PARSED_DIR / "cases.jsonl"
MANIFEST   = PARSED_DIR / "ingest_manifest.json"

PA_NAME_LONG = "Pennsylvania"   # jurisdiction.name_long value kept; all others dropped

_DATE_FULL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_YM   = re.compile(r"^\d{4}-\d{2}$")
_DATE_Y    = re.compile(r"^\d{4}$")


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class Citation:
    raw_cite: str
    reporter: str | None
    target_case_ids: list[str]   # CAP-resolved ids, stringified; [] when absent
    category: str | None


@dataclass
class CaseRecord:
    id: str                      # CAP case id, stringified; ES _id
    name: str
    name_abbreviation: str
    decision_date: str | None    # normalized ISO string or None
    court_name: str | None
    jurisdiction: str | None     # name_long, e.g. "Pennsylvania"
    reporter: str                # source reporter slug, e.g. "a3d"
    volume: int
    citations: list[str]         # this case's own citation strings
    cites_to: list[Citation]     # outbound citations (the graph edges)
    parties: str
    judges: str
    word_count: int              # from analysis.word_count
    head_matter: str
    full_text: str               # head_matter + joined opinion texts


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if _DATE_FULL.match(s) or _DATE_YM.match(s) or _DATE_Y.match(s):
        return s
    return None


def _parse_cites_to(raw_list: list | None) -> list[Citation]:
    if not raw_list:
        return []
    out: list[Citation] = []
    for c in raw_list:
        raw_ids = c.get("case_ids")   # list[int] | key absent → None
        out.append(Citation(
            raw_cite        = c.get("cite", ""),
            reporter        = c.get("reporter"),
            # case_ids are integers in the export; stringify to match CaseRecord
            target_case_ids = [str(i) for i in raw_ids] if raw_ids else [],
            category        = c.get("category"),
        ))
    return out


def parse_case(raw: dict, reporter: str, volume: int) -> "CaseRecord | None":
    """
    Parse one raw CAP case JSON into a CaseRecord.
    Returns None when: id missing, not Pennsylvania, or full_text is empty.
    """
    case_id = str(raw.get("id", "")).strip()
    if not case_id:
        return None

    jur      = raw.get("jurisdiction") or {}
    jur_long = jur.get("name_long", "")
    if jur_long != PA_NAME_LONG:
        return None

    casebody    = raw.get("casebody") or {}
    opinions    = casebody.get("opinions") or []
    head_matter = (casebody.get("head_matter") or "").strip()
    opinion_texts = [op.get("text", "").strip() for op in opinions if op.get("text")]
    full_text     = (head_matter + "\n" + "\n".join(opinion_texts)).strip()
    if not full_text:
        return None

    analysis   = raw.get("analysis") or {}
    word_count = int(analysis.get("word_count") or 0)

    own_cites = [c.get("cite", "") for c in (raw.get("citations") or []) if c.get("cite")]

    parties_raw = casebody.get("parties") or []
    judges_raw  = casebody.get("judges") or []

    return CaseRecord(
        id                = case_id,
        name              = raw.get("name", ""),
        name_abbreviation = raw.get("name_abbreviation", ""),
        decision_date     = _normalize_date(raw.get("decision_date")),
        court_name        = (raw.get("court") or {}).get("name"),
        jurisdiction      = jur_long or None,
        reporter          = reporter,
        volume            = volume,
        citations         = own_cites,
        cites_to          = _parse_cites_to(raw.get("cites_to")),
        parties           = " ".join(parties_raw),
        judges            = "; ".join(judges_raw),
        word_count        = word_count,
        head_matter       = head_matter,
        full_text         = full_text,
    )


# ── pass 1: stream all zips → cases_raw.jsonl ────────────────────────────────

def _stream_to_temp(raw_dir: Path, temp: Path) -> tuple[dict, dict[str, int]]:
    """
    Returns:
        counts:  {reporter: {raw, skipped_non_pa, skipped_empty}}
        best_wc: {case_id: best word_count seen} — used to pick canonical record
    """
    counts: dict[str, dict[str, int]] = {}
    best_wc: dict[str, int] = {}

    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} does not exist — run the downloader first")

    with temp.open("w", encoding="utf-8") as out:
        for reporter_dir in sorted(raw_dir.iterdir()):
            if not reporter_dir.is_dir():
                continue
            reporter = reporter_dir.name
            c = counts.setdefault(reporter, {"raw": 0, "skipped_non_pa": 0, "skipped_empty": 0})

            zips = sorted(reporter_dir.glob("*.zip"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            for zip_path in zips:
                try:
                    vol = int(zip_path.stem)
                except ValueError:
                    continue

                print(f"  parsing {reporter}/{zip_path.name} ...", flush=True)
                try:
                    zf_obj = zipfile.ZipFile(zip_path)
                except zipfile.BadZipFile:
                    print(f"    [warn] bad zip, skipping")
                    continue

                with zf_obj as zf:
                    case_files = [n for n in zf.namelist()
                                  if n.startswith("json/") and n.endswith(".json")]
                    for name in case_files:
                        try:
                            raw = json.loads(zf.read(name))
                        except Exception:
                            continue

                        c["raw"] += 1

                        # jurisdiction pre-check for accurate per-reporter counting
                        jur_long = (raw.get("jurisdiction") or {}).get("name_long", "")
                        if jur_long != PA_NAME_LONG:
                            c["skipped_non_pa"] += 1
                            continue

                        record = parse_case(raw, reporter, vol)
                        if record is None:
                            c["skipped_empty"] += 1
                            continue

                        d = asdict(record)
                        out.write(json.dumps(d, ensure_ascii=False) + "\n")

                        if record.word_count > best_wc.get(record.id, -1):
                            best_wc[record.id] = record.word_count

    return counts, best_wc


# ── pass 2: dedup by case id, keep richest full_text ─────────────────────────

def _dedup(temp: Path, out: Path, best_wc: dict[str, int]) -> tuple[int, int]:
    """
    Emits the first record seen at the best word_count for each case id.
    Returns (written, collapsed).
    """
    seen: set[str] = set()
    written = collapsed = 0
    with temp.open("r", encoding="utf-8") as inp, out.open("w", encoding="utf-8") as outp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            d   = json.loads(line)
            cid = d["id"]
            if cid in seen:
                collapsed += 1
                continue
            if d["word_count"] >= best_wc.get(cid, 0):
                outp.write(line + "\n")
                seen.add(cid)
                written += 1
            else:
                collapsed += 1
    return written, collapsed


# ── passes 3+4: citation density gate ────────────────────────────────────────

def _density_gate(jsonl: Path) -> dict:
    """
    Pass 3: build corpus_ids + raw tallies.
    Pass 4: count in-corpus edges using the complete corpus_ids set.
    The two-pass approach is needed because in-corpus check requires
    the full corpus to be known before checking individual edges.
    """
    corpus_ids: set[str] = set()
    total_cases          = 0
    cases_with_cites     = 0
    total_entries        = 0
    resolved_entries     = 0
    # (src_id, [target_ids]) — kept in memory only to avoid a third file pass.
    # Memory: ~2 M integers max for a full PA corpus; well within 32 GB RAM.
    src_targets: list[list[str]] = []

    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d   = json.loads(line)
            cid = d["id"]
            corpus_ids.add(cid)
            total_cases += 1

            ct = d.get("cites_to") or []
            if ct:
                cases_with_cites += 1
            total_entries += len(ct)

            resolved_here = [tid for c in ct for tid in (c.get("target_case_ids") or [])]
            resolved_entries += len(resolved_here)
            src_targets.append(resolved_here)

    # Pass 4: in-corpus edges
    in_corpus_edges = sum(
        1 for targets in src_targets for tid in targets if tid in corpus_ids
    )
    # Query cases: ≥1 in-corpus cited target (can form an eval pair)
    query_cases = sum(
        1 for targets in src_targets if any(tid in corpus_ids for tid in targets)
    )

    frac_with_cites = cases_with_cites / total_cases       if total_cases      else 0.0
    frac_resolved   = resolved_entries  / total_entries     if total_entries    else 0.0
    frac_in_corpus  = in_corpus_edges   / resolved_entries  if resolved_entries else 0.0

    return {
        "total_corpus_cases"     : total_cases,
        "cases_with_cites_to"    : cases_with_cites,
        "frac_with_cites_to"     : round(frac_with_cites, 4),
        "total_cites_to_entries" : total_entries,
        "resolved_entries"       : resolved_entries,
        "frac_resolved"          : round(frac_resolved, 4),
        "in_corpus_edges"        : in_corpus_edges,
        "frac_in_corpus"         : round(frac_in_corpus, 4),
        "query_cases"            : query_cases,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Pass 1 — streaming all zips → cases_raw.jsonl")
    counts, best_wc = _stream_to_temp(RAW_DIR, TEMP_JSONL)

    total_raw    = sum(c["raw"]             for c in counts.values())
    total_non_pa = sum(c["skipped_non_pa"]  for c in counts.values())
    total_empty  = sum(c["skipped_empty"]   for c in counts.values())
    total_kept   = total_raw - total_non_pa - total_empty
    discard_pct  = total_non_pa / total_raw * 100 if total_raw else 0.0

    print(f"\n  raw cases read      : {total_raw:>10,}")
    print(f"  skipped non-PA      : {total_non_pa:>10,}  ({discard_pct:.1f}% discard from regional filter)")
    print(f"  skipped empty text  : {total_empty:>10,}")
    print(f"  passed to dedup     : {total_kept:>10,}")

    print("\nPass 2 — deduplicating by CAP case id (keep richest full_text) ...")
    written, collapsed = _dedup(TEMP_JSONL, OUT_JSONL, best_wc)
    TEMP_JSONL.unlink()
    print(f"  canonical cases     : {written:>10,}")
    print(f"  dedup-collapsed     : {collapsed:>10,}")

    print("\nPasses 3+4 — citation density gate ...")
    density = _density_gate(OUT_JSONL)

    manifest = {
        "by_reporter": counts,
        "totals": {
            "raw"             : total_raw,
            "skipped_non_pa"  : total_non_pa,
            "skipped_empty"   : total_empty,
            "dedup_kept"      : written,
            "dedup_collapsed" : collapsed,
        },
        "citation_density_gate": density,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), "utf-8")

    sep = "─" * 62
    print(f"\n{sep}")
    print("  CITATION DENSITY GATE")
    print(sep)
    print(f"  Corpus size                       : {density['total_corpus_cases']:>10,}")
    print(f"  Cases with ≥1 cites_to            : {density['cases_with_cites_to']:>10,}"
          f"  ({density['frac_with_cites_to']*100:.1f} %)")
    print(f"  Total cites_to entries            : {density['total_cites_to_entries']:>10,}")
    print(f"  Entries with resolved case_id     : {density['resolved_entries']:>10,}"
          f"  ({density['frac_resolved']*100:.1f} %)")
    print(f"  In-corpus edges                   : {density['in_corpus_edges']:>10,}")
    print(f"  In-corpus fraction of resolved    : {density['frac_in_corpus']*100:>9.1f} %")
    print(f"  Query cases (≥1 in-corpus target) : {density['query_cases']:>10,}")
    print(sep)

    floor = 5_000
    if density["query_cases"] < floor:
        print(f"\n  ✗ GATE FAIL — only {density['query_cases']:,} query cases (floor: {floor:,}).")
        print("    Stop and report to owner — graph and eval depend on sufficient citation density.")
    else:
        print(f"\n  ✓ Gate looks viable ({density['query_cases']:,} query cases ≥ floor of {floor:,}).")
        print("    Review numbers with owner before proceeding with graph and training.")

    print(f"\n  Full manifest: {MANIFEST}")
    print("  → Hand these numbers to owner and stop here.")


if __name__ == "__main__":
    main()
