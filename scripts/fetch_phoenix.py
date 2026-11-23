"""Stream RWTH-PHOENIX-Weather 2014-T and extract a reproducible subset.

The official archive (41.7 GB, CC BY-NC-SA 3.0, research only) is never written to disk.
It is decompressed on the fly, and only the selected members are extracted:

* everything outside ``features/`` (annotations, evaluation scripts), up to ``--max-other-mb`` per file;
* all ``dev`` and ``test`` frame folders;
* ``train`` frame folders whose name hashes into the kept bucket (``--train-mod``),
  so the subset is reproducible and independent of archive order.

An inventory of *every* member (kept or not) is written to ``inventory.json`` so the
audit can report real counts and sizes for the full release.

Usage::

    python -I -X utf8 scripts/fetch_phoenix.py               # default subset into ../datasets/phoenix2014T
    python -I -X utf8 scripts/fetch_phoenix.py --train-mod 1 # all of train (needs ~45 GB free)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

URL = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
EXPECTED_BYTES = 41_699_758_035  # Content-Length reported by the server on 2026-11-23
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "datasets" / "phoenix2014T"
FEATURES = "features/"
DATASET_DIR = "PHOENIX-2014-T/"


class CountingReader:
    """File-like wrapper that counts compressed bytes read from the network."""

    def __init__(self, raw):
        self.raw = raw
        self.n = 0

    def read(self, size=-1):
        data = self.raw.read(size)
        self.n += len(data)
        return data


def train_bucket(video: str, mod: int) -> bool:
    return int(hashlib.md5(video.encode("utf-8")).hexdigest(), 16) % mod == 0


def classify(rel: str) -> tuple[str, str | None, str | None]:
    """Return (kind, split, video) for a path relative to the dataset root."""
    if not rel.startswith(FEATURES):
        return "other", None, None
    parts = rel.split("/")
    # features/<feature_type>/<split>/<video>/<frame>
    if len(parts) < 5:
        return "feature_meta", None, None
    return parts[1], parts[2], parts[3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--train-mod", type=int, default=8, help="keep train videos with md5(name) %% mod == 0 (1 = all)")
    ap.add_argument("--splits", default="dev,test", help="splits kept in full")
    ap.add_argument("--feature", default="fullFrame-210x260px")
    ap.add_argument("--max-other-mb", type=float, default=200.0)
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="abort if free space drops below this")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    splits = {s for s in args.splits.split(",") if s}

    videos: dict[tuple[str, str, str], dict] = defaultdict(lambda: {"frames": 0, "bytes": 0, "kept": False})
    other: list[dict] = []
    root_prefix: str | None = None
    kept_bytes = 0
    kept_files = 0
    t0 = time.time()
    last_report = t0

    req = urllib.request.Request(args.url, headers={"User-Agent": "research-dataset-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        if total and total != EXPECTED_BYTES:
            print(f"WARNING: Content-Length {total} != expected {EXPECTED_BYTES}", flush=True)
        reader = CountingReader(resp)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for m in tar:
                name = m.name
                # Dataset root inside the archive. The first member is the bare top-level
                # directory, so the prefix must be found from a member that contains it.
                if root_prefix is None and DATASET_DIR in name:
                    root_prefix = name[: name.index(DATASET_DIR) + len(DATASET_DIR)]
                rel = name[len(root_prefix):] if root_prefix and name.startswith(root_prefix) else name
                if not m.isfile():
                    continue
                kind, split, video = classify(rel)
                keep = False
                if kind == "other" or kind == "feature_meta":
                    keep = m.size <= args.max_other_mb * 1e6
                    other.append({"path": rel, "bytes": m.size, "kept": keep})
                else:
                    v = videos[(kind, split, video)]
                    v["frames"] += 1
                    v["bytes"] += m.size
                    if kind == args.feature and (
                        split in splits or (split == "train" and train_bucket(video, args.train_mod))
                    ):
                        keep = True
                        v["kept"] = True
                if keep:
                    tar.extract(m, path=out, filter="data")
                    kept_bytes += m.size
                    kept_files += 1

                now = time.time()
                if now - last_report > 60:
                    last_report = now
                    free_gb = shutil.disk_usage(out).free / 1e9
                    pct = 100 * reader.n / total if total else float("nan")
                    rate = reader.n / (now - t0) / 1e6
                    print(
                        f"[{(now - t0) / 60:6.1f} min] {pct:5.1f}% read ({rate:.1f} MB/s), "
                        f"kept {kept_files} files / {kept_bytes / 1e9:.2f} GB, free {free_gb:.1f} GB",
                        flush=True,
                    )
                    if free_gb < args.min_free_gb:
                        print(f"ABORT: free space {free_gb:.1f} GB < {args.min_free_gb} GB", flush=True)
                        return 3

    summary: dict[str, dict] = defaultdict(lambda: {"videos": 0, "frames": 0, "bytes": 0, "kept_videos": 0, "kept_bytes": 0})
    for (kind, split, _), v in videos.items():
        s = summary[f"{kind}/{split}"]
        s["videos"] += 1
        s["frames"] += v["frames"]
        s["bytes"] += v["bytes"]
        if v["kept"]:
            s["kept_videos"] += 1
            s["kept_bytes"] += v["bytes"]

    inventory = {
        "url": args.url,
        "compressed_bytes_read": reader.n,
        "root_prefix": root_prefix,
        "selection": {"feature": args.feature, "full_splits": sorted(splits), "train_mod": args.train_mod},
        "minutes": round((time.time() - t0) / 60, 1),
        "kept_files": kept_files,
        "kept_bytes": kept_bytes,
        "summary": dict(summary),
        "other_files": other,
        "videos": [
            {"feature": k, "split": s, "video": vid, **v} for (k, s, vid), v in sorted(videos.items())
        ],
    }
    (out / "inventory.json").write_text(json.dumps(inventory, indent=1), encoding="utf-8")
    print(json.dumps({k: inventory[k] for k in ("compressed_bytes_read", "kept_files", "kept_bytes", "minutes")}))
    print(json.dumps(inventory["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
