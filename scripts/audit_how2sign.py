"""How2Sign subset audit: every number in docs/dataset_audit.md comes from this script.

    python scripts/audit_how2sign.py [--root ../datasets/How2Sign] [--out outputs/audit/how2sign_audit.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.data.how2sign import ORIGINAL_COLUMNS, REALIGNED_COLUMNS, How2SignDataset, read_how2sign_csv  # noqa: E402
from signbridge.data.validation import locate_clip_in_video  # noqa: E402
from signbridge.video.io import probe_video  # noqa: E402
from signbridge.video.sampling import interval_iou  # noqa: E402


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "min": round(min(xs), 3), "median": round(st.median(xs), 3),
            "mean": round(st.mean(xs), 3), "max": round(max(xs), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO.parent / "datasets" / "How2Sign")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "audit" / "how2sign_audit.json")
    a = ap.parse_args()
    t0 = time.perf_counter()
    ds = How2SignDataset(a.root, a.split, "clips", probe_videos=False)
    rep: dict = {"root": str(a.root), "split": a.split, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # 1. files on disk ---------------------------------------------------------------
    files = [p for p in a.root.rglob("*") if p.is_file()]
    by_dir = Counter(str(p.parent.relative_to(a.root)).replace("\\", "/") for p in files)
    rep["files_on_disk"] = {"n_files": len(files), "total_bytes": sum(p.stat().st_size for p in files),
                            "by_directory": dict(sorted(by_dir.items()))}

    # 2. annotation files ------------------------------------------------------------
    orig = read_how2sign_csv(ds.original_csv, ORIGINAL_COLUMNS)
    real = read_how2sign_csv(ds.realigned_csv, REALIGNED_COLUMNS)
    with open(ds.original_csv, encoding="utf-8", newline="") as f:    # quoting sensitivity check
        no_quoting_rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    quote_diff = [x["SENTENCE_NAME"] for x, y in zip(orig, no_quoting_rows) if x["SENTENCE"] != y.get("SENTENCE")]
    ann = {}
    for name, rows, cols, s_key, e_key in (("original", orig, ORIGINAL_COLUMNS, "START", "END"),
                                           ("realigned", real, REALIGNED_COLUMNS, "START_REALIGNED", "END_REALIGNED")):
        durs = [float(r[e_key]) - float(r[s_key]) for r in rows]
        ann[name] = {
            "file": str((ds.original_csv if name == "original" else ds.realigned_csv).relative_to(a.root)),
            "delimiter": "TAB", "columns": cols, "n_rows": len(rows),
            "unique_VIDEO_ID": len({r["VIDEO_ID"] for r in rows}),
            "unique_VIDEO_NAME": len({r["VIDEO_NAME"] for r in rows}),
            "unique_SENTENCE_NAME": len({r["SENTENCE_NAME"] for r in rows}),
            "empty_cells": {c: sum(1 for r in rows if not r[c].strip()) for c in cols},
            "segment_duration_s": stats(durs),
            "non_positive_intervals": sum(1 for d in durs if d <= 0),
            "negative_start": sum(1 for r in rows if float(r[s_key]) < 0),
        }
    words = [len(r["SENTENCE"].split()) for r in orig]
    ann["sentence_words"] = stats(words)
    ann["vocabulary_size_whitespace_lowercase"] = len({w.lower() for r in orig for w in r["SENTENCE"].split()})
    ann["sentence_text_identical_across_files"] = all(
        o["SENTENCE"] == r["SENTENCE"] for o, r in zip(sorted(orig, key=lambda r: r["SENTENCE_NAME"]),
                                                       sorted(real, key=lambda r: r["SENTENCE_NAME"])))
    ann["rows_using_csv_quoting"] = quote_diff  # would be mis-parsed with QUOTE_NONE
    ann["gloss_columns_present"] = [c for c in ORIGINAL_COLUMNS + REALIGNED_COLUMNS if "GLOSS" in c.upper()]
    suffix = Counter(r["VIDEO_NAME"].rsplit("-rgb_front", 1)[0].rsplit("-", 1)[1] for r in orig)
    ann["VIDEO_NAME_numeric_suffix_counts"] = dict(sorted(suffix.items()))
    rmap = {r["SENTENCE_NAME"]: r for r in real}
    ious, inside, shifts = [], 0, []
    for o in orig:
        r = rmap[o["SENTENCE_NAME"]]
        oi = (float(o["START"]), float(o["END"]))
        ri = (float(r["START_REALIGNED"]), float(r["END_REALIGNED"]))
        ious.append(interval_iou(oi, ri))
        inside += oi[0] <= ri[0] and ri[1] <= oi[1]
        shifts.append(ri[0] - oi[0])
    ann["original_vs_realigned"] = {
        "iou": stats(ious), "rows_iou_zero": sum(1 for x in ious if x == 0),
        "rows_realigned_inside_original": inside, "start_shift_s(realigned-original)": stats(shifts)}
    rep["annotations"] = ann

    # 3. archive listings vs annotation ----------------------------------------------
    lst = {}
    for res in ("val_rgb_front_clips", "val_rgb_front_videos"):
        p = a.root / "_listings" / f"{res}.json"
        if p.exists():
            d = json.loads(p.read_text())
            lst[res] = {"archive_bytes": d["archive_bytes"], "n_members": len(d["members"]),
                        "member_uncompressed_MB": stats([m["size"] / 1e6 for m in d["members"]]),
                        "names": {Path(m["name"]).stem for m in d["members"]}}
    mapping = {}
    if "val_rgb_front_clips" in lst:
        names = {r["SENTENCE_NAME"] for r in orig}
        mapping["annotation_rows_without_clip_in_archive"] = sorted(names - lst["val_rgb_front_clips"]["names"])
        mapping["clips_in_archive_without_annotation"] = sorted(lst["val_rgb_front_clips"]["names"] - names)
    if "val_rgb_front_videos" in lst:
        vnames = {r["VIDEO_NAME"] for r in orig}
        mapping["annotation_videos_without_full_video"] = sorted(vnames - lst["val_rgb_front_videos"]["names"])
        mapping["full_videos_without_annotation"] = sorted(lst["val_rgb_front_videos"]["names"] - vnames)
    for v in lst.values():
        v.pop("names")
    rep["archives"] = lst
    rep["mapping"] = mapping

    # 4. downloaded clips: decode everything ------------------------------------------
    omap = {r["SENTENCE_NAME"]: r for r in orig}
    clips = sorted((a.root / "sentence_level" / a.split / "rgb_front" / "raw_videos").glob("*.mp4"))
    rows = []
    for c in clips:
        vi = probe_video(c, full_decode=True)
        o, r = omap.get(c.stem), rmap.get(c.stem)
        rows.append({
            "clip": c.stem, "bytes": c.stat().st_size, "fps": vi.fps, "frame_count": vi.frame_count,
            "decoded_frames": vi.decoded_frames, "corrupted_frames": vi.corrupted_frames,
            "resolution": f"{vi.width}x{vi.height}", "codec": vi.codec, "duration_s": round(vi.duration, 3),
            "has_annotation": o is not None,
            "clip_minus_original_interval_s": None if o is None else round(vi.duration - (float(o["END"]) - float(o["START"])), 3),
            "clip_minus_realigned_interval_s": None if r is None else round(vi.duration - (float(r["END_REALIGNED"]) - float(r["START_REALIGNED"])), 3),
        })
    rep["downloaded_clips"] = {
        "n": len(rows), "fps_counts": dict(Counter(str(r["fps"]) for r in rows)),
        "resolution_counts": dict(Counter(r["resolution"] for r in rows)),
        "codec_counts": dict(Counter(str(r["codec"]) for r in rows)),
        "frames": stats([r["decoded_frames"] for r in rows]), "duration_s": stats([r["duration_s"] for r in rows]),
        "clips_with_corrupted_frames": sum(1 for r in rows if r["corrupted_frames"]),
        "clips_reported_ne_decoded": sum(1 for r in rows if r["frame_count"] != r["decoded_frames"]),
        "abs_clip_minus_original_interval_s": stats([abs(r["clip_minus_original_interval_s"]) for r in rows if r["has_annotation"]]),
        "abs_clip_minus_realigned_interval_s": stats([abs(r["clip_minus_realigned_interval_s"]) for r in rows if r["has_annotation"]]),
        "per_clip": rows,
    }
    by_suffix: dict = {}
    for r in rows:
        if r["has_annotation"]:
            sfx = omap[r["clip"]]["VIDEO_NAME"].rsplit("-rgb_front", 1)[0].rsplit("-", 1)[1]
            by_suffix.setdefault(sfx, []).append(abs(r["clip_minus_original_interval_s"]))
    rep["downloaded_clips"]["abs_clip_minus_original_by_VIDEO_NAME_suffix"] = {
        k: {**stats(v), "n_over_0.25s": sum(1 for x in v if x > 0.25)} for k, v in sorted(by_suffix.items())}

    # 5. downloaded full videos + temporal alignment ----------------------------------
    fulls = sorted((a.root / "video_level" / a.split / "rgb_front" / "raw_videos").glob("*.mp4"))
    fv = []
    for v in fulls:
        vi = probe_video(v, full_decode=True)
        vrows = [r for r in orig if r["VIDEO_NAME"] == v.stem]
        loc = []
        for o in vrows:
            clip = a.root / "sentence_level" / a.split / "rgb_front" / "raw_videos" / f"{o['SENTENCE_NAME']}.mp4"
            r = rmap[o["SENTENCE_NAME"]]
            entry = {"sentence": o["SENTENCE_NAME"], "original": [float(o["START"]), float(o["END"])],
                     "realigned": [float(r["START_REALIGNED"]), float(r["END_REALIGNED"])],
                     "realigned_end_within_video": float(r["END_REALIGNED"]) <= vi.duration + 0.05,
                     "original_end_within_video": float(o["END"]) <= vi.duration + 0.05}
            if clip.exists():
                entry["clip_vs_original_start"] = locate_clip_in_video(clip, v, float(o["START"]), search_s=2.0)
                entry["clip_vs_realigned_start"] = locate_clip_in_video(clip, v, float(r["START_REALIGNED"]), search_s=2.0)
            loc.append(entry)
        fv.append({"video": v.stem, "fps": vi.fps, "frame_count": vi.frame_count, "decoded_frames": vi.decoded_frames,
                   "corrupted_frames": vi.corrupted_frames, "duration_s": round(vi.duration, 3),
                   "resolution": f"{vi.width}x{vi.height}", "n_sentences": len(vrows), "sentences": loc})
    rep["downloaded_full_videos"] = fv
    rep["runtime_s"] = round(time.perf_counter() - t0, 2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("downloaded_clips", "downloaded_full_videos")},
                     indent=1)[:6000])
    print("downloaded_clips:", json.dumps({k: v for k, v in rep["downloaded_clips"].items() if k != "per_clip"}, indent=1))
    for v in fv:
        print("full video", v["video"], {k: v[k] for k in ("fps", "frame_count", "decoded_frames", "duration_s", "n_sentences")})
        for s in v["sentences"]:
            co, cr = s.get("clip_vs_original_start", {}), s.get("clip_vs_realigned_start", {})
            print(f"  {s['sentence']:28s} orig={s['original']} re={s['realigned']} re_in_video={s['realigned_end_within_video']}"
                  f" | vs orig: off={co.get('offset_s')} mad={co.get('mad_best')}/{co.get('mad_at_expected')}"
                  f" | vs re: off={cr.get('offset_s')} mad={cr.get('mad_best')}/{cr.get('mad_at_expected')}")
    print(f"wrote {a.out} in {rep['runtime_s']}s")


if __name__ == "__main__":
    main()
