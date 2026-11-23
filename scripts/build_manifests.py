"""Generate common-schema manifests (JSONL + CSV view) for every adapter/view.

    python scripts/build_manifests.py [--h2s-root ../datasets/How2Sign] [--out outputs/manifests]

Outputs stay in outputs/ (git-ignored): How2Sign text is CC BY-NC data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.data import How2SignDataset, VSLDataset, write_manifest, write_manifest_csv  # noqa: E402


def build(h2s_root: Path, vsl_root: Path | None, out: Path) -> dict:
    summary = {}
    jobs = [(f"how2sign_val_{v}", How2SignDataset(h2s_root, "val", v)) for v in
            ("clips", "original_full", "realigned_full")]
    if vsl_root:
        jobs.append(("vsl_letter_frames", VSLDataset(vsl_root)))
    for name, ds in jobs:
        recs = ds.load_manifest()
        avail = [r for r in recs if r.video_available]
        write_manifest(recs, out / f"{name}.jsonl")
        write_manifest_csv(recs, out / f"{name}.csv")
        write_manifest(avail, out / f"{name}.available.jsonl")
        summary[name] = {
            "records": len(recs), "video_available": len(avail),
            "annotation_status": dict(Counter(r.annotation_status for r in recs)),
            "with_gloss": sum(r.has_gloss for r in recs), "with_target_text": sum(r.has_translation for r in recs),
            "with_signer_id": sum(r.signer_id is not None for r in recs),
            "with_fps": sum(r.fps is not None for r in recs),
            "fps_values": dict(Counter(str(r.fps) for r in avail)),
            "with_segment_times": sum(r.start_time is not None for r in recs),
            "build_warnings": ds.build_warnings,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2s-root", type=Path, default=REPO.parent / "datasets" / "How2Sign")
    ap.add_argument("--vsl-root", type=Path, default=None, help="optional VSL letter_frames folder")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "manifests")
    a = ap.parse_args()
    print(json.dumps(build(a.h2s_root, a.vsl_root, a.out), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
