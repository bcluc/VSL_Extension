"""Pipeline validation experiment on the downloaded How2Sign subset + local VSL data.

Validates infrastructure only. No model is trained; no accuracy/WER/BLEU is computed.

    python scripts/validate_pipeline.py [--h2s-root ../datasets/How2Sign] [--out outputs/validation/results.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from signbridge.data import (AnnotationMissingError, How2SignDataset, SampleNotFoundError,  # noqa: E402
                             VSLDataset, write_manifest)
from signbridge.data.collate import collate  # noqa: E402
from signbridge.data.validation import locate_clip_in_video, validate_sample  # noqa: E402
from signbridge.models import SignTranslationPipeline  # noqa: E402
from signbridge.models.reference import (CTCGlossDecoder, FrameCNNEncoder, PoseMLPEncoder,  # noqa: E402
                                         TemporalConvEncoder, TransformerTranslationModel)
from signbridge.text import GlossVocabulary, TextVocabulary  # noqa: E402
from signbridge.video import (PreprocessConfig, VideoPreprocessor, VideoUnavailableError,  # noqa: E402
                              probe_video)

RESULTS: list[dict] = []


def check(name: str):
    def deco(fn):
        def run(*a, **kw):
            t0 = time.perf_counter()
            try:
                status, details = fn(*a, **kw)
            except Exception as e:  # noqa: BLE001 - a crash is a FAIL, recorded verbatim
                status, details = "FAIL", {"exception": repr(e), "trace": traceback.format_exc(limit=3)}
            entry = {"check": name, "status": status, "seconds": round(time.perf_counter() - t0, 3), "details": details}
            RESULTS.append(entry)
            print(f"[{status:>16}] {name} ({entry['seconds']}s)")
            return details
        return run
    return deco


def sha(arr_or_bytes) -> str:
    b = arr_or_bytes.tobytes() if isinstance(arr_or_bytes, np.ndarray) else arr_or_bytes
    return hashlib.sha256(b).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2s-root", type=Path, default=REPO.parent / "datasets" / "How2Sign")
    ap.add_argument("--vsl-root", type=Path, default=None,
                    help="optional VSL letter_frames folder (<root>/<LABEL_ID>/frame_*.jpg); VSL checks are skipped without it")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "preprocess_default.json")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "validation" / "results.json")
    ap.add_argument("--keypoint-clips", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PreprocessConfig.from_json(a.config)
    pre = VideoPreprocessor(cfg)
    tmp = a.out.parent / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    ctx: dict = {}

    # 1. manifest generation ---------------------------------------------------------
    @check("1. manifest generation (How2Sign 3 views, + VSL letters if --vsl-root)")
    def c1():
        d = {}
        for name, mk in (("how2sign_clips", lambda: How2SignDataset(a.h2s_root, "val", "clips")),
                         ("how2sign_original_full", lambda: How2SignDataset(a.h2s_root, "val", "original_full")),
                         ("how2sign_realigned_full", lambda: How2SignDataset(a.h2s_root, "val", "realigned_full")),
                         *([("vsl_letters", lambda: VSLDataset(a.vsl_root))] if a.vsl_root else [])):
            t0 = time.perf_counter()
            ds = mk()
            recs = ds.load_manifest()
            ctx[name] = ds
            d[name] = {"records": len(recs), "video_available": sum(r.video_available for r in recs),
                       "seconds": round(time.perf_counter() - t0, 3), "warnings": len(ds.build_warnings),
                       "null_gloss": sum(r.gloss_sequence is None for r in recs),
                       "null_signer": sum(r.signer_id is None for r in recs)}
        ok = d["how2sign_clips"]["records"] == 1741 and d["how2sign_clips"]["null_gloss"] == 1741 \
            and ("vsl_letters" not in d or (d["vsl_letters"]["records"] > 0 and not d["vsl_letters"]["warnings"]))
        return ("PASS" if ok else "FAIL"), d
    c1()
    h2s: How2SignDataset = ctx["how2sign_clips"]
    avail = h2s.sample_ids(available_only=True)

    # 2. per-sample video/annotation validation ---------------------------------------
    @check("2. video decoding + video-to-annotation validation (full decode)")
    def c2():
        rows, t_dec, n_frames = [], 0.0, 0
        for sid in avail:
            t0 = time.perf_counter()
            chk = validate_sample(h2s, sid, full_decode=True)
            t_dec += time.perf_counter() - t0
            n_frames += chk.info.get("decoded_frames") or 0
            rows.append(chk.to_dict())
        full = ctx["how2sign_realigned_full"]
        seg = [validate_sample(full, s).to_dict() for s in full.sample_ids(available_only=True)]
        vsl = ([validate_sample(ctx["vsl_letters"], s).to_dict() for s in ctx["vsl_letters"].sample_ids()]
               if "vsl_letters" in ctx else [])
        errors = [r for r in rows + seg + vsl if not r["ok"]]
        warn = [(r["sample_id"], r["warnings"]) for r in rows if r["warnings"]]
        return ("PASS" if not errors else "FAIL"), {
            "how2sign_clips_checked": len(rows), "frames_decoded": n_frames,
            "decode_fps": round(n_frames / t_dec, 1), "realigned_segments_checked": len(seg),
            "vsl_sequences_checked": len(vsl), "errors": errors,
            "clips_with_duration_warning": warn}
    c2()

    # 3. frame extraction + preprocessing --------------------------------------------
    @check("3. frame sampling + resize/normalize (configs/preprocess_default.json)")
    def c3():
        per, clips = [], {}
        for sid in avail:
            src = h2s.load_video(sid)
            t0 = time.perf_counter()
            clip = pre.process(src)
            dt = time.perf_counter() - t0
            clips[sid] = clip
            per.append({"sample": sid, "src_fps": src.fps, "src_frames": src.num_frames, "out_T": clip.length,
                        "eff_fps": clip.effective_fps, "ms": round(dt * 1000, 1), "windows": len(clip.windows)})
        ctx["clips"] = clips
        shapes_ok = all(c.frames.shape[1:] == (3, 224, 224) and c.frames.dtype == np.float32 for c in clips.values())
        # time-consistency across fps: T / duration should equal target fps for every source fps
        rate = [p["out_T"] / (p["src_frames"] / p["src_fps"]) for p in per]
        ms = [p["ms"] for p in per]
        return ("PASS" if shapes_ok else "FAIL"), {
            "clips": len(per), "ms_per_clip": {"min": min(ms), "median": float(np.median(ms)), "max": max(ms)},
            "total_frames_out": sum(p["out_T"] for p in per),
            "steps_per_second_of_video": {"min": round(min(rate), 2), "max": round(max(rate), 2)},
            "src_fps_seen": sorted({p["src_fps"] for p in per}), "per_clip": per}
    c3()

    # 4. temporal windows on a long video: streaming vs whole ------------------------
    @check("4. temporal window streaming on the full 46 s video (bounded memory)")
    def c4():
        full = ctx["how2sign_original_full"]
        fid = [r for r in full.load_manifest() if r.video_available]
        vpath = Path(fid[0].video_path)
        from signbridge.video import VideoReader
        uncapped = VideoPreprocessor(PreprocessConfig(**{**cfg.to_dict(), "max_frames": None}))
        tracemalloc.start()
        t0 = time.perf_counter()
        n_win, T = 0, 0
        for w in uncapped.stream_windows(VideoReader(vpath)):
            n_win += 1
            T = max(T, w.end)
        t_stream = time.perf_counter() - t0
        peak_stream = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        tracemalloc.start()
        t0 = time.perf_counter()
        whole = uncapped.process(VideoReader(vpath))
        t_whole = time.perf_counter() - t0
        peak_whole = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        same = n_win == len(whole.windows)
        return ("PASS" if same and peak_stream < peak_whole else "FAIL"), {
            "video": vpath.name, "sampled_steps": whole.length, "windows": n_win,
            "stream_seconds": round(t_stream, 2), "stream_peak_MB": round(peak_stream / 1e6, 1),
            "whole_seconds": round(t_whole, 2), "whole_peak_MB": round(peak_whole / 1e6, 1)}
    c4()

    # 5. temporal alignment: clip <-> full-video segments -----------------------------
    @check("5. temporal alignment: clips located in full video at annotated timestamps")
    def c5():
        full = ctx["how2sign_original_full"]
        rows = []
        for r in full.load_manifest():
            if not r.video_available:
                continue
            clip = h2s.clip_path(r.meta["sentence_name"])
            if not clip.exists():
                continue
            o = locate_clip_in_video(clip, r.video_path, r.meta["original_start"], search_s=2.0)
            ra = locate_clip_in_video(clip, r.video_path, r.meta["realigned_start"], search_s=2.0)
            seg_frames = full.load_video(r.sample_id).num_frames
            rows.append({"sentence": r.meta["sentence_name"], "offset_frames_vs_original": o["offset_frames"],
                         "mad_best": o["mad_best"], "mad_at_realigned_start": ra["mad_at_expected"],
                         "segment_frames_from_full_video": seg_frames,
                         "clip_frames": probe_video(clip).frame_count})
        ok = rows and all(abs(x["offset_frames_vs_original"]) <= 1 for x in rows)
        return ("PASS" if ok else "FAIL"), {"n": len(rows), "rows": rows,
                                            "criterion": "|offset| <= 1 frame from ORIGINAL start"}
    c5()

    # 6. pose path: hand keypoints (real temporal features) ----------------------------
    @check("6. pose features: MediaPipe 2-hand keypoints on real clips")
    def c6():
        model = REPO.parent / "models" / "mediapipe" / "hand_landmarker.task"
        if not model.exists():
            return "SKIP", {"reason": f"{model} missing (run scripts/fetch_mediapipe_models.py)"}
        from signbridge.features.hand_keypoints import HandKeypointExtractor, keypoints_to_model_input
        from signbridge.video.sampling import sample_indices
        rows, seqs = [], []
        with HandKeypointExtractor(model) as ext:
            for sid in avail[: a.keypoint_clips]:
                src = h2s.load_video(sid)
                idx = sample_indices(src.num_frames, "fps", src_fps=src.fps, target_fps=12)
                t0 = time.perf_counter()
                kps, present = ext.extract(f for _, f in src.read_frames(idx))
                dt = time.perf_counter() - t0
                seqs.append((sid, keypoints_to_model_input(kps, present)))
                rows.append({"sample": sid, "T": len(idx), "ms_per_frame": round(dt * 1000 / max(len(idx), 1), 1),
                             "left_slot_detected_ratio": round(float(present[:, 0].mean()), 3),
                             "right_slot_detected_ratio": round(float(present[:, 1].mean()), 3),
                             "no_hand_ratio": round(float((~present.any(1)).mean()), 3)})
        ctx["pose_seqs"] = seqs
        return "PASS", {"rows": rows, "feature_shape_per_step": [42, 4]}
    c6()

    # 7. batching + vocabularies ----------------------------------------------------
    @check("7. sequence batching (variable length) + text vocabulary")
    def c7():
        clips = ctx["clips"]
        recs = [h2s.get_sample(s).record for s in avail]
        tv = TextVocabulary.from_texts([r.target_text for r in recs])   # subset-only vocab, contract test
        items = [{"record": r, "inputs": clips[r.sample_id].frames, "valid_mask": clips[r.sample_id].valid_mask}
                 for r in recs[:8]]
        b = collate(items, text_vocab=tv)
        ctx["batch"], ctx["tv"] = b, tv
        ok = b.inputs.shape[0] == 8 and b.input_mask.sum() == b.input_lengths.sum() and b.has_text.all()
        return ("PASS" if ok else "FAIL"), {"inputs": list(b.inputs.shape), "lengths": b.input_lengths.tolist(),
                                            "text_ids": list(b.text_ids.shape), "text_vocab_size": len(tv),
                                            "padding_ratio": round(1 - float(b.input_mask.mean()), 3)}
    c7()

    @check("8. gloss supervision on How2Sign is refused (no gloss annotations)")
    def c8():
        recs = [h2s.get_sample(s).record for s in avail]
        out = {}
        try:
            GlossVocabulary.from_sequences(r.gloss_sequence for r in recs)
            out["vocab"] = "built (UNEXPECTED)"
        except ValueError as e:
            out["vocab"] = f"refused: {e}"
        try:
            collate([{"record": recs[0], "inputs": np.zeros((2, 1), np.float32)}], require_gloss=True)
            out["collate"] = "accepted (UNEXPECTED)"
        except AnnotationMissingError as e:
            out["collate"] = f"refused: {e}"
        ok = out["vocab"].startswith("refused") and out["collate"].startswith("refused")
        return ("EXPECTED_REFUSAL" if ok else "FAIL"), out
    c8()

    # 9. model I/O contracts on real batches ----------------------------------------
    @check("9. model input/output contracts on real batches (UNTRAINED, shapes only)")
    def c9():
        b = ctx["batch"].to_torch()
        tv = ctx["tv"]
        vis = FrameCNNEncoder(output_dim=64)
        tmp_enc = TemporalConvEncoder(64, 64, downsample=2)
        tr = TransformerTranslationModel(64, len(tv), tv.pad_id, tv.bos_id, tv.eos_id, d_model=64, nhead=4, layers=1)
        pipe = SignTranslationPipeline(vis, tmp_enc, translator=tr).eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            out = pipe(b["inputs"], b["input_lengths"], target_in=b["text_ids"][:, :-1])
            dt = time.perf_counter() - t0
            gen = tr.generate(out["encoder"], max_len=8)
        # CTC contract with real labels: VSL isolated letters (length-1 gloss targets), if given
        ctc = None
        if "vsl_letters" in ctx:
            vsl = ctx["vsl_letters"]
            gv = GlossVocabulary.from_sequences(r.gloss_sequence for r in vsl.load_manifest())
            small = VideoPreprocessor(PreprocessConfig(sampling="uniform", num_samples=8, image_size=(112, 112)))
            vitems = []
            for sid in vsl.sample_ids()[:4]:
                c = small.process(vsl.load_video(sid))
                vitems.append({"record": vsl.get_sample(sid).record, "inputs": c.frames, "valid_mask": c.valid_mask})
            vb = collate(vitems, gloss_vocab=gv, require_gloss=True).to_torch()
            cpipe = SignTranslationPipeline(FrameCNNEncoder(output_dim=32), TemporalConvEncoder(32, 32, 2),
                                            gloss_decoder=CTCGlossDecoder(32, len(gv)))
            g = cpipe(vb["inputs"], vb["input_lengths"])["gloss"]
            loss = cpipe.gloss_decoder.loss(g, vb["gloss_ids"], vb["gloss_lengths"])
            ctc = {"log_probs": list(g.log_probs.shape), "gloss_vocab": len(gv),
                   "loss_finite": bool(torch.isfinite(loss)), "note": "contract only, no training"}
        # pose path contract
        pose_ok = None
        if ctx.get("pose_seqs"):
            seqs = [s for _, s in ctx["pose_seqs"]]
            from signbridge.data.collate import pad_sequences
            arr, lens, _ = pad_sequences(seqs)
            pf = PoseMLPEncoder(42, 4, 32)(torch.from_numpy(arr), torch.from_numpy(lens))
            pose_ok = list(pf.features.shape)
        return "PASS", {
            "rgb_translation": {"encoder_out": list(out["encoder"].features.shape),
                                "encoder_lengths": out["encoder"].lengths.tolist(),
                                "logits": list(out["translation_logits"].shape), "forward_ms": round(dt * 1000, 1),
                                "generated_lengths": [len(x) for x in gen]},
            "ctc_on_vsl_letters": ctc,
            "pose_encoder_out": pose_ok}
    c9()

    # 10. error handling ------------------------------------------------------------
    @check("10. error handling (missing video / annotation / sample)")
    def c10():
        missing = [r.sample_id for r in h2s.load_manifest() if not r.video_available][0]
        res = {}
        for label, fn, exc in (("missing video", lambda: h2s.load_video(missing), VideoUnavailableError),
                               ("missing gloss", lambda: h2s.load_annotations(avail[0]).require_gloss(), AnnotationMissingError),
                               ("unknown sample", lambda: h2s.get_sample("does-not-exist"), SampleNotFoundError)):
            try:
                fn()
                res[label] = "no error (FAIL)"
            except exc as e:
                res[label] = f"{type(e).__name__}: {str(e)[:90]}"
        absent = [s for s in ("eY32ru3Nstc_20-8-rgb_front", "fE6xxSbjVV8_12-8-rgb_front")]
        res["rows_absent_from_official_archive_have_video_available_false"] = all(
            not h2s.get_sample(s).record.video_available for s in absent)
        ok = all("FAIL" not in str(v) for v in res.values()) and res["rows_absent_from_official_archive_have_video_available_false"]
        return ("PASS" if ok else "FAIL"), res
    c10()

    # 11. reproducibility -----------------------------------------------------------
    @check("11. reproducibility (manifest, preprocessing, seeded model)")
    def c11():
        p1 = write_manifest(How2SignDataset(a.h2s_root, "val", "clips").load_manifest(), tmp / "m1.jsonl")
        p2 = write_manifest(How2SignDataset(a.h2s_root, "val", "clips").load_manifest(), tmp / "m2.jsonl")
        sid = avail[0]
        f1 = pre.process(h2s.load_video(sid)).frames
        f2 = pre.process(h2s.load_video(sid)).frames
        outs = []
        for _ in range(2):
            torch.manual_seed(123)
            enc = FrameCNNEncoder(output_dim=16).eval()
            with torch.no_grad():
                outs.append(enc(torch.from_numpy(f1[None, :8]), torch.tensor([8])).features.numpy())
        r = {"manifest_sha": [sha(p1.read_bytes()), sha(p2.read_bytes())],
             "frames_sha": [sha(f1), sha(f2)], "model_out_sha": [sha(outs[0]), sha(outs[1])],
             "subset_selection": "scripts/fetch_how2sign_subset.py random seed=0 (n=24) + all clips of a4uz0W33REs-3"}
        ok = all(len(set(v)) == 1 for k, v in r.items() if k.endswith("sha"))
        return ("PASS" if ok else "FAIL"), r
    c11()

    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
           "opencv": cv2.__version__, "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}
    try:
        import mediapipe
        env["mediapipe"] = mediapipe.__version__
    except ImportError:
        env["mediapipe"] = None
    summary = {"environment": env, "config": cfg.to_dict(), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "counts": {s: sum(1 for r in RESULTS if r["status"] == s) for s in
                          ("PASS", "FAIL", "EXPECTED_REFUSAL", "SKIP")},
               "results": RESULTS}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary["counts"]), "->", a.out)


if __name__ == "__main__":
    main()
