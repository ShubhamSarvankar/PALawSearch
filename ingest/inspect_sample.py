"""
Download one volume each from `pa` and `a3d` and print the exact JSON structure
of cites_to (all known locations) and casebody before writing the parser.

Run:
    python ingest/inspect_sample.py

Paste the output back; the parser will be built against real shapes, not assumed ones.
Nothing is written to disk — both zips are fetched into memory.
"""
import io
import json
import time
import urllib.request
import urllib.error
import zipfile

BASE = "https://static.case.law"
UA = "PA-LawSearch-v2/inspect (research; contact ssarvankar1311@gmail.com)"

# pa/1  — oldest PA reporter, smallest volume, baseline structure check
# a3d/1 — Atlantic 3d starts ~2010; most likely to have populated case_ids
TARGETS = [("pa", 1), ("a3d", 1)]


# ── fetch ─────────────────────────────────────────────────────────────────────

def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    print(f"  GET {url} ... ", end="", flush=True)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    print(f"{len(data)/1_048_576:.2f} MB")
    return data


# ── printers ──────────────────────────────────────────────────────────────────

def show_cites_to(entries: list | None, label: str, indent: int = 6) -> None:
    pad = " " * indent
    if entries is None:
        print(f"{pad}{label}: key ABSENT")
        return
    if not entries:
        print(f"{pad}{label}: [] (empty list)")
        return
    print(f"{pad}{label}: {len(entries)} entries — showing up to 3")
    for i, c in enumerate(entries[:3]):
        print(f"{pad}  [{i}] all keys : {sorted(c.keys())}")
        print(f"{pad}      cite        : {c.get('cite')!r}")
        print(f"{pad}      category    : {c.get('category')!r}")
        print(f"{pad}      reporter    : {c.get('reporter')!r}")
        print(f"{pad}      case_ids    : {c.get('case_ids')!r}")
        print(f"{pad}      case_paths  : {c.get('case_paths')!r}")
        print(f"{pad}      opinion_index: {c.get('opinion_index')!r}")
        print(f"{pad}      weight      : {c.get('weight')!r}")
        print(f"{pad}      pin_cites   : {c.get('pin_cites')!r}")
    if len(entries) > 3:
        print(f"{pad}  ... +{len(entries) - 3} more")


def show_case(raw: dict, zip_path: str) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  file            : {zip_path}")
    print(f"  top-level keys  : {sorted(raw.keys())}")
    print(f"  id              : {raw.get('id')!r}")
    print(f"  name            : {raw.get('name')!r}")
    print(f"  decision_date   : {raw.get('decision_date')!r}")

    jur = raw.get("jurisdiction") or {}
    print(f"  jurisdiction    : name={jur.get('name')!r}  name_long={jur.get('name_long')!r}")

    ana = raw.get("analysis") or {}
    print(f"  analysis.word_count : {ana.get('word_count')!r}")

    own = raw.get("citations") or []
    print(f"  citations (own) : {[c.get('cite') for c in own]}")

    # ── location 1: top-level cites_to ────────────────────────────────────────
    print("\n  LOCATION 1 — top-level cites_to")
    show_cites_to(raw.get("cites_to"), "cites_to", indent=4)

    # ── location 2: extracted_citations (seen in some CAP exports) ────────────
    print("\n  LOCATION 2 — top-level extracted_citations")
    show_cites_to(raw.get("extracted_citations"), "extracted_citations", indent=4)

    # ── location 3: casebody ──────────────────────────────────────────────────
    cb = raw.get("casebody")
    if cb is None:
        print("\n  LOCATION 3 — casebody: KEY ABSENT")
        return

    print(f"\n  LOCATION 3 — casebody  (keys: {sorted(cb.keys())})")

    hm = (cb.get("head_matter") or "").strip()
    print(f"    head_matter[:150] : {hm[:150]!r}")
    print(f"    parties           : {cb.get('parties')!r}")
    print(f"    judges            : {cb.get('judges')!r}")

    opinions = cb.get("opinions") or []
    print(f"\n    opinions          : {len(opinions)} entries")
    for oi, op in enumerate(opinions[:2]):
        print(f"\n    opinions[{oi}]")
        print(f"      keys   : {sorted(op.keys())}")
        print(f"      type   : {op.get('type')!r}")
        text = (op.get("text") or "").strip()
        print(f"      text[:200] : {text[:200]!r}")
        # per-opinion cites_to (the plan says it may live here too)
        show_cites_to(op.get("cites_to"), "opinions[].cites_to", indent=6)


# ── main ──────────────────────────────────────────────────────────────────────

def inspect_volume(reporter: str, volume: int) -> None:
    print(f"\n{'═'*64}")
    print(f"  Reporter: {reporter}   Volume: {volume}")
    print(f"{'═'*64}")

    url = f"{BASE}/{reporter}/{volume}.zip"
    try:
        data = fetch(url)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} — skipping")
        return
    except Exception as e:
        print(f"  ERROR: {e} — skipping")
        return

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        all_names = zf.namelist()
        case_files = [n for n in all_names if n.endswith(".json") and n.startswith("json/")]
        print(f"  zip entries total  : {len(all_names)}")
        print(f"  case JSON files    : {len(case_files)}")
        if not case_files:
            print("  WARNING — no case JSON files found; zip layout may differ")
            print(f"  first 10 entries   : {all_names[:10]}")
            return

        inspected = 0
        for name in case_files:
            if inspected >= 2:
                break
            try:
                raw = json.loads(zf.read(name))
            except Exception as e:
                print(f"  JSON parse error in {name}: {e}")
                continue
            cb = raw.get("casebody") or {}
            opinions = cb.get("opinions") or []
            has_text = any((op.get("text") or "").strip() for op in opinions)
            if not has_text:
                continue   # prefer cases with actual opinion text
            show_case(raw, name)
            inspected += 1

        if inspected == 0:
            # fallback: show the first parseable case regardless of text
            for name in case_files[:3]:
                try:
                    raw = json.loads(zf.read(name))
                    show_case(raw, name)
                    break
                except Exception:
                    continue


if __name__ == "__main__":
    for reporter, volume in TARGETS:
        inspect_volume(reporter, volume)
        time.sleep(1)   # polite gap between reporters

    print("\n\n✓ Inspection done. Paste this output before the parser is written.")
