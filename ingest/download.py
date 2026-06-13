"""
Full PA corpus downloader.

From batch_download.py (kept):
  - urllib.request + User-Agent header (no new deps)
  - Exponential backoff: sleep(2**attempt) on transient errors
  - 404 detection as terminal — recorded as 'missing', not retried
  - Per-volume skip when already 'done' in the manifest
  - Inter-volume sleep for polite crawling

Added:
  - Six reporters: pa, pa-super, pa-commw, a, a2d, a3d
  - Volume discovery from {reporter}/VolumesMetadata.json — no hardcoded NUM_VOLUMES
  - Size estimation via HEAD samples (SAMPLE_N evenly-spaced volumes per reporter)
  - shutil.disk_usage gate — stops if estimate > 90 % of free space
  - --estimate: print sizes and stop; --go: run the full download
  - Streaming chunked writes (CHUNK bytes at a time, never response.read() into memory)
  - Atomic writes: .tmp rename — no corrupt partial files on Ctrl-C
  - MD5 per file written into manifest
  - Per-reporter manifest at data/download_manifest.json:
      {reporter: {volume_str: {status, size_bytes, md5}}}
  - Output: data/raw/{reporter}/{volume}.zip  (pathlib, no backslash literals)

Usage:
    python ingest/download.py --estimate   # show sizes, do NOT download
    python ingest/download.py --go         # run after reviewing the estimate
    (or: just estimate / just download)
"""

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://static.case.law"
UA   = "PA-LawSearch-v2/download (research; contact ssarvankar1311@gmail.com)"

# PA-named reporters + regional Atlantic Reporter (includes most modern PA appellate decisions).
# Volumes are discovered dynamically; this list is the only thing hard-coded.
REPORTERS = ["pa", "pa-super", "pa-commw", "a", "a2d", "a3d"]

DATA_DIR    = Path("data/raw")
MANIFEST    = Path("data/download_manifest.json")
MAX_RETRIES = 3
SLEEP_VOL   = 0.4     # seconds between volumes
CHUNK       = 1 << 16  # 64 KB streaming chunks
SAMPLE_N    = 5        # volumes to HEAD-sample per reporter for the size estimate

# Conservative per-volume size fallbacks when HEAD returns no Content-Length.
# PA-named reporters have older, shorter opinions; Atlantic volumes are larger.
_FALLBACK_MB: dict[str, float] = {
    "pa": 2.0, "pa-super": 2.5, "pa-commw": 2.0,
    "a": 6.0,  "a2d": 8.0,     "a3d": 8.0,
}


def _req(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": UA}, method=method)


def _fetch_json(url: str) -> list | dict:
    with urllib.request.urlopen(_req(url), timeout=60) as r:
        return json.loads(r.read())


def discover_volumes(reporter: str) -> list[int]:
    """Return sorted volume numbers from the reporter's VolumesMetadata.json."""
    url = f"{BASE}/{reporter}/VolumesMetadata.json"
    try:
        meta = _fetch_json(url)
    except Exception as exc:
        print(f"  [warn] {reporter}/VolumesMetadata.json unavailable: {exc}")
        return []
    entries = meta if isinstance(meta, list) else list(meta.values())
    nums: list[int] = []
    for entry in entries:
        vn = entry.get("volume_number") or entry.get("volumeNumber")
        try:
            nums.append(int(vn))
        except (TypeError, ValueError):
            pass
    return sorted(set(nums))


def _head_bytes(url: str) -> int | None:
    try:
        with urllib.request.urlopen(_req(url, "HEAD"), timeout=20) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def estimate_sizes(volumes_map: dict[str, list[int]]) -> tuple[dict[str, int], int]:
    """
    HEAD SAMPLE_N evenly-spaced volumes per reporter to estimate total bytes.
    Returns (per_reporter_bytes, grand_total_bytes).
    """
    print("\nSampling HEAD requests for size estimate...")
    per_reporter: dict[str, int] = {}
    for reporter, vols in volumes_map.items():
        if not vols:
            per_reporter[reporter] = 0
            continue
        step   = max(1, len(vols) // SAMPLE_N)
        sample = vols[::step][:SAMPLE_N]
        sizes: list[int] = []
        for v in sample:
            url = f"{BASE}/{reporter}/{v}.zip"
            sz  = _head_bytes(url)
            tag = f"{sz / 1_048_576:.2f} MB" if sz else "no Content-Length"
            print(f"  HEAD {reporter}/{v}.zip → {tag}")
            if sz:
                sizes.append(sz)
            time.sleep(0.15)
        avg = (sum(sizes) / len(sizes)) if sizes else _FALLBACK_MB[reporter] * 1_048_576
        per_reporter[reporter] = int(avg * len(vols))
    return per_reporter, sum(per_reporter.values())


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else {}


def _save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2), "utf-8")


def _download_vol(reporter: str, vol: int, dest: Path) -> tuple[str, int | None, str | None]:
    """
    Download one volume zip with retries.
    Returns (status, size_bytes, md5); status in {'done', 'failed', 'missing'}.
    """
    url = f"{BASE}/{reporter}/{vol}.zip"
    tmp = dest.with_suffix(".tmp")
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(_req(url), timeout=120) as r, tmp.open("wb") as f:
                md5   = hashlib.md5()
                total = 0
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    md5.update(chunk)
                    total += len(chunk)
            tmp.rename(dest)
            return "done", total, md5.hexdigest()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                tmp.unlink(missing_ok=True)
                return "missing", None, None
            print(f"    HTTP {exc.code} (attempt {attempt + 1}/{MAX_RETRIES})")
        except Exception as exc:
            print(f"    {exc!r} (attempt {attempt + 1}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    tmp.unlink(missing_ok=True)
    return "failed", None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Download PA Law corpus from static.case.law")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--estimate", action="store_true",
                     help="Show size estimate only; do not download")
    grp.add_argument("--go", action="store_true",
                     help="Run the full download (after reviewing the estimate)")
    args = ap.parse_args()

    # ── discover volumes ─────────────────────────────────────────────────────
    print("Discovering volumes from VolumesMetadata.json ...")
    volumes_map: dict[str, list[int]] = {}
    for r in REPORTERS:
        vols = discover_volumes(r)
        volumes_map[r] = vols
        print(f"  {r:12s}: {len(vols):4d} volumes")
    total_vols = sum(len(v) for v in volumes_map.values())
    print(f"  {'TOTAL':12s}: {total_vols:4d} volumes")

    # ── size estimate + disk gate ────────────────────────────────────────────
    per_reporter, total_bytes = estimate_sizes(volumes_map)

    # Walk up to find the nearest existing ancestor for disk_usage.
    probe = DATA_DIR
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free

    print("\n── Download size estimate ───────────────────────────────────────────")
    for r in REPORTERS:
        est  = per_reporter[r]
        n    = len(volumes_map[r])
        print(f"  {r:12s}: ~{est / 1_073_741_824:.2f} GB  ({n} volumes)")
    print(f"\n  TOTAL estimated : ~{total_bytes / 1_073_741_824:.2f} GB")
    print(f"  Free disk       :  {free_bytes / 1_073_741_824:.2f} GB")

    if total_bytes * 1.1 > free_bytes:
        print("\n✗ STOP — estimate exceeds 90 % of free disk. Free space and re-run.")
        return

    if not args.go:
        print("\n→ Review the estimate, then run with --go (or: just download) to proceed.")
        return

    # ── full download ────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    for reporter in REPORTERS:
        vols   = volumes_map[reporter]
        rdir   = DATA_DIR / reporter
        rdir.mkdir(parents=True, exist_ok=True)
        rm     = manifest.setdefault(reporter, {})
        done_n = sum(1 for v in rm.values() if v.get("status") == "done")
        print(f"\n── {reporter}  ({len(vols)} vols, {done_n} already done) " + "─" * 20)

        for vol in vols:
            key = str(vol)
            if rm.get(key, {}).get("status") == "done":
                continue
            dest = rdir / f"{vol}.zip"
            print(f"  {reporter}/{vol}.zip ... ", end="", flush=True)
            status, sz, md5 = _download_vol(reporter, vol, dest)
            rm[key] = {"status": status, "size_bytes": sz, "md5": md5}
            label   = f"done  ({sz / 1_048_576:.1f} MB)" if status == "done" else status
            print(label)
            _save_manifest(manifest)
            time.sleep(SLEEP_VOL)

    manifest = _load_manifest()
    print("\n── Final summary ────────────────────────────────────────────────────")
    for r in REPORTERS:
        counts: dict[str, int] = {}
        for v in manifest.get(r, {}).values():
            s = v.get("status", "?")
            counts[s] = counts.get(s, 0) + 1
        print(f"  {r:12s}: {counts}")


if __name__ == "__main__":
    main()
