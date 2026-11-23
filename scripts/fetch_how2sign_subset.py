"""Fetch a *small subset* of How2Sign files without downloading whole archives.

How2Sign distributes clips/videos as large ZIP archives on Google Drive
(val frontal clips: ~1.7 GB, val frontal full videos: ~16 GB). Google Drive
honours HTTP Range requests, so this tool:

1. reads only the ZIP central directory (tail of the archive),
2. lists the members,
3. downloads only the byte ranges of the requested members,
4. verifies each member's CRC32 before writing it to disk.

Annotation CSVs are small (~300 KB) and are downloaded whole.

Nothing here is committed to Git: the default output root is
``<repo>/../datasets/How2Sign`` (outside the repository).

License reminder: How2Sign is CC BY-NC 4.0 - non-commercial research only.

Usage examples
--------------
    python scripts/fetch_how2sign_subset.py annotations
    python scripts/fetch_how2sign_subset.py list val_rgb_front_clips
    python scripts/fetch_how2sign_subset.py fetch val_rgb_front_clips --n 24 --seed 0
    python scripts/fetch_how2sign_subset.py fetch val_rgb_front_videos --names <video>.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import time
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT.parent / "datasets" / "How2Sign"

# Google Drive file IDs taken from https://how2sign.github.io/ (download table,
# checked 2026-11-23). The IDs in how2sign/how2sign-data/download_how2sign.sh
# for clips and keypoints returned HTTP 404 on that date.
RESOURCES = {
    "val_rgb_front_clips": {"id": "1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO", "kind": "zip",
                            "dest": "sentence_level/val/rgb_front/raw_videos"},
    "val_rgb_front_videos": {"id": "1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g", "kind": "zip",
                             "dest": "video_level/val/rgb_front/raw_videos"},
    "val_2D_keypoints": {"id": "1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w", "kind": "tar.gz",
                         "dest": "sentence_level/val/rgb_front/features"},
    "how2sign_val.csv": {"id": "1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3", "kind": "file",
                         "dest": "sentence_level/val/text/en/raw_text"},
    "how2sign_realigned_val.csv": {"id": "1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI", "kind": "file",
                                   "dest": "sentence_level/val/text/en/raw_text/re_aligned"},
}
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


# --------------------------------------------------------------------------- HTTP
def http_range(file_id: str, start: int, end: int, retries: int = 4) -> tuple[bytes, int]:
    """Return (bytes[start:end+1], total_size)."""
    req = urllib.request.Request(URL.format(id=file_id),
                                 headers={"Range": f"bytes={start}-{end}",
                                          "User-Agent": "signbridge-research-fetcher"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {r.status}")
                total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                data = r.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"short read {len(data)} != {end - start + 1}")
            return data, total
        except Exception as exc:  # noqa: BLE001 - network: retry then re-raise
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def http_full(file_id: str) -> bytes:
    req = urllib.request.Request(URL.format(id=file_id),
                                 headers={"User-Agent": "signbridge-research-fetcher"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# --------------------------------------------------------------------------- ZIP
@dataclass
class ZipMember:
    name: str
    method: int
    crc32: int
    comp_size: int
    size: int
    local_header_offset: int


def read_central_directory(file_id: str) -> tuple[list[ZipMember], int]:
    _, total = http_range(file_id, 0, 0)
    tail_len = min(total, (1 << 16) + 22 + 20)  # max comment + EOCD + ZIP64 locator
    tail, _ = http_range(file_id, total - tail_len, total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD not found - not a ZIP archive?")
    (_, _, _, _, n_entries, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", tail[eocd:eocd + 22])
    if cd_offset == 0xFFFFFFFF or n_entries == 0xFFFF:  # ZIP64
        loc = tail.rfind(b"PK\x06\x07")
        z64_off = struct.unpack("<IIQI", tail[loc:loc + 20])[2]
        rec, _ = http_range(file_id, z64_off, z64_off + 55)
        n_entries, cd_size, cd_offset = struct.unpack("<QQQ", rec[32:56])
    cd, _ = http_range(file_id, cd_offset, cd_offset + cd_size - 1)
    members, p = [], 0
    for _ in range(n_entries):
        (sig, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen,
         _, _, _, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        if sig != 0x02014B50:
            raise RuntimeError("bad central directory signature")
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, lho):  # ZIP64 extended info
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = list(struct.unpack("<" + "Q" * (hlen // 8), extra[q + 4:q + 4 + hlen]))
                    if usize == 0xFFFFFFFF: usize = vals.pop(0)
                    if csize == 0xFFFFFFFF: csize = vals.pop(0)
                    if lho == 0xFFFFFFFF: lho = vals.pop(0)
                q += 4 + hlen
        members.append(ZipMember(name, method, crc, csize, usize, lho))
        p += 46 + nlen + xlen + clen
    return members, total


def fetch_member(file_id: str, m: ZipMember) -> bytes:
    hdr, _ = http_range(file_id, m.local_header_offset, m.local_header_offset + 29)
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    data_start = m.local_header_offset + 30 + nlen + xlen
    raw, _ = http_range(file_id, data_start, data_start + m.comp_size - 1)
    if m.method == 0:
        data = raw
    elif m.method == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"unsupported compression method {m.method}")
    if zlib.crc32(data) & 0xFFFFFFFF != m.crc32:
        raise RuntimeError(f"CRC mismatch for {m.name}")
    return data


# --------------------------------------------------------------------------- CLI
def cmd_annotations(root: Path) -> None:
    for name in ("how2sign_val.csv", "how2sign_realigned_val.csv"):
        res = RESOURCES[name]
        out = root / res["dest"] / name
        out.parent.mkdir(parents=True, exist_ok=True)
        data = http_full(res["id"])
        out.write_bytes(data)
        print(f"{name}: {len(data)} bytes -> {out}")
        _log(root, {"resource": name, "file": str(out), "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()})


def cmd_list(resource: str, root: Path) -> list[ZipMember]:
    res = RESOURCES[resource]
    if res["kind"] != "zip":
        raise SystemExit(f"{resource} is {res['kind']}, member listing needs a ZIP")
    members, total = read_central_directory(res["id"])
    files = [m for m in members if not m.name.endswith("/")]
    listing = root / "_listings" / f"{resource}.json"
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text(json.dumps({"resource": resource, "archive_bytes": total,
                                   "members": [asdict(m) for m in files]}, indent=1))
    print(f"{resource}: archive {total / 1e9:.2f} GB, {len(files)} files -> {listing}")
    return files


def cmd_fetch(resource: str, root: Path, n: int, seed: int, names: list[str] | None,
              max_mb: float) -> None:
    files = cmd_list(resource, root)
    res = RESOURCES[resource]
    if names:
        wanted = set(names)
        chosen = [m for m in files if Path(m.name).name in wanted]
        missing = wanted - {Path(m.name).name for m in chosen}
        if missing:
            print(f"WARNING: not in archive: {sorted(missing)}")
    else:
        pool = [m for m in files if m.comp_size <= max_mb * 1e6]
        chosen = sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda m: m.name)
    dest = root / res["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for m in chosen:
        out = dest / Path(m.name).name
        if out.exists() and out.stat().st_size == m.size:
            print(f"  skip (exists) {out.name}")
            continue
        t0 = time.perf_counter()
        data = fetch_member(res["id"], m)
        out.write_bytes(data)
        total += m.comp_size
        print(f"  {out.name}: {m.size / 1e6:.2f} MB in {time.perf_counter() - t0:.1f}s (CRC ok)")
        _log(root, {"resource": resource, "member": m.name, "file": str(out), "bytes": m.size,
                    "crc32": f"{m.crc32:08x}", "selection": "names" if names else f"random seed={seed}"})
    print(f"downloaded {total / 1e6:.1f} MB compressed for {len(chosen)} members")


def cmd_peek_tar(resource: str, root: Path, peek_mb: float) -> None:
    """Stream only the first `peek_mb` MB of a .tar.gz and extract the members that are
    complete within that prefix. Used to inspect the keypoint schema without the full
    1.2 GB download."""
    import io
    import tarfile
    res = RESOURCES[resource]
    raw, total = http_range(res["id"], 0, int(peek_mb * 1e6) - 1)
    data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    dest = root / res["dest"] / "_peek"
    dest.mkdir(parents=True, exist_ok=True)
    names, n_files = [], 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as tf:
        try:
            for ti in tf:
                names.append(ti.name)
                if ti.isfile():
                    f = tf.extractfile(ti)
                    if f is None:
                        continue
                    content = f.read()
                    if len(content) != ti.size:
                        break  # truncated by the peek boundary
                    safe = dest / Path(ti.name).as_posix().replace("/", "__")
                    safe.write_bytes(content)
                    n_files += 1
        except (tarfile.ReadError, EOFError):
            pass  # reached the end of the downloaded prefix
    (dest / "_members_seen.json").write_text(json.dumps(
        {"archive_bytes": total, "peek_bytes": len(raw), "decompressed_bytes": len(data),
         "members_seen": names}, indent=1))
    print(f"{resource}: archive {total / 1e9:.2f} GB, peeked {len(raw) / 1e6:.1f} MB, "
          f"{len(names)} tar entries seen, {n_files} complete files -> {dest}")
    _log(root, {"resource": resource, "peek_bytes": len(raw), "entries_seen": len(names)})


def _log(root: Path, entry: dict) -> None:
    entry["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(root / "_provenance.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["annotations", "list", "fetch", "peek"])
    ap.add_argument("resource", nargs="?", choices=[k for k, v in RESOURCES.items() if v["kind"] != "file"])
    ap.add_argument("--peek-mb", type=float, default=8.0, help="prefix size for 'peek' (tar.gz)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--n", type=int, default=24, help="number of random members to fetch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", nargs="*", help="explicit member basenames to fetch")
    ap.add_argument("--max-mb", type=float, default=15.0, help="skip members larger than this (random mode)")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if a.command == "annotations":
        cmd_annotations(a.root)
    elif a.command == "list":
        cmd_list(a.resource, a.root)
    elif a.command == "peek":
        cmd_peek_tar(a.resource, a.root, a.peek_mb)
    else:
        cmd_fetch(a.resource, a.root, a.n, a.seed, a.names, a.max_mb)


if __name__ == "__main__":
    main()
