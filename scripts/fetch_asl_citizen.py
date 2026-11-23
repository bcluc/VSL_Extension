"""Fetch a subset of ASL Citizen (Microsoft Research) without downloading the 42.8 GB zip.

ASL Citizen license: Microsoft Research License Terms. Non-commercial research only, no
redistribution, delete personal data when the research ends.
https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/

The server supports HTTP byte ranges, so the zip is opened remotely: ``zipfile`` reads the
central directory through range requests, and only the selected members are fetched.

    python -I -X utf8 scripts/fetch_asl_citizen.py list                 # index + split CSVs
    python -I -X utf8 scripts/fetch_asl_citizen.py fetch --glosses HELLO NAME ...
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = ("https://download.microsoft.com/download/b/8/8/"
       "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "datasets" / "asl_citizen"
UA = {"User-Agent": "research-dataset-fetch/1.0"}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP byte ranges."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("server does not advertise byte ranges")
        self.pos = 0
        self.requests = 0
        self.bytes = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        end = self.pos + n - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"expected 206, got {r.status}")
                    data = r.read()
                break
            except Exception:  # noqa: BLE001 - network retry
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        b[: len(data)] = data
        self.pos += len(data)
        self.requests += 1
        self.bytes += len(data)
        return len(data)


def open_remote_zip(url: str) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    raw = HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20)), raw


def safe_name(member: str) -> str:
    """Flatten a zip member path into a safe local file name (no traversal)."""
    p = PurePosixPath(member)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe member path: {member}")
    return p.name


def cmd_list(args) -> int:
    zf, raw = open_remote_zip(args.url)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    infos = zf.infolist()
    listing = [{"name": i.filename, "size": i.file_size, "csize": i.compress_size, "method": i.compress_type}
               for i in infos]
    (out / "zip_index.json").write_text(json.dumps(listing), encoding="utf-8")
    small = [i for i in infos if not i.is_dir() and not i.filename.lower().endswith(".mp4")]
    for i in small:
        if i.file_size > 50e6:
            print(f"skip large non-video member {i.filename} ({i.file_size} B)")
            continue
        dest = out / "meta" / safe_name(i.filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(i))
        print(f"saved {dest.name} ({i.file_size} B)")
    vids = [i for i in infos if i.filename.lower().endswith(".mp4")]
    print(json.dumps({"zip_bytes": raw.size, "members": len(infos), "videos": len(vids),
                      "video_bytes": sum(i.file_size for i in vids),
                      "http_requests": raw.requests, "bytes_transferred": raw.bytes}))
    return 0


def read_split_rows(meta: Path) -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        p = meta / f"{split}.csv"
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                r["split"] = split
                rows.append(r)
    return rows


def cmd_fetch(args) -> int:
    out: Path = args.out
    rows = read_split_rows(out / "meta")
    gloss_col = next(c for c in rows[0] if c.strip().lower() == "gloss")
    file_col = next(c for c in rows[0] if "video" in c.strip().lower())
    wanted = {g.upper() for g in args.glosses}
    sel = [r for r in rows if r[gloss_col].upper() in wanted]
    missing = wanted - {r[gloss_col].upper() for r in sel}
    if missing:
        print(f"WARNING: glosses not in ASL Citizen: {sorted(missing)}")
    zf, raw = open_remote_zip(args.url)
    by_base = {PurePosixPath(i.filename).name: i for i in zf.infolist() if i.filename.lower().endswith(".mp4")}
    vdir = out / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fetched = skipped = 0
    for k, r in enumerate(sel, 1):
        base = PurePosixPath(r[file_col]).name
        info = by_base.get(base)
        if info is None:
            print(f"WARNING: {base} listed in CSV but not in zip")
            continue
        dest = vdir / safe_name(base)
        if dest.exists() and dest.stat().st_size == info.file_size:
            skipped += 1
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(zf.read(info))  # zipfile verifies the CRC
        tmp.replace(dest)
        fetched += 1
        if k % 50 == 0:
            print(f"{k}/{len(sel)} clips, {raw.bytes / 1e6:.0f} MB, {time.time() - t0:.0f} s", flush=True)
    with open(out / "subset.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel[0].keys()))
        w.writeheader()
        w.writerows(sel)
    print(json.dumps({"selected": len(sel), "fetched": fetched, "already_present": skipped,
                      "bytes_transferred": raw.bytes, "seconds": round(time.time() - t0)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("--glosses", nargs="+", required=True)
    args = ap.parse_args()
    return cmd_list(args) if args.cmd == "list" else cmd_fetch(args)


if __name__ == "__main__":
    sys.exit(main())
