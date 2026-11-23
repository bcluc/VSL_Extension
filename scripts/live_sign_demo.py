"""Live ASL greetings demo: webcam -> sign-by-sign recognition -> English/Vietnamese sentence -> speech.

Scope (shown on screen): 29 ASL signs for greetings/introductions, one sign at a time, hands
lowered between signs. Recognition model trained on ASL Citizen (Microsoft Research License,
non-commercial research). The sentence comes from a hand-written phrase table, not a learned
translation model.

    python -X utf8 scripts/live_sign_demo.py                    # webcam, speak English
    python -X utf8 scripts/live_sign_demo.py --speak vi         # speak the Vietnamese line (Piper)
    python -X utf8 scripts/live_sign_demo.py --video clip.mp4 --headless   # offline test

Keys: Q/ESC quit, SPACE end sentence now, C clear, L switch speech language, M mute.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.signs.features import LandmarkExtractor  # noqa: E402
from signbridge.signs.live import LiveConfig, SentenceEvent, SignEvent, SignSession, rest_line_y  # noqa: E402
from signbridge.signs.model import load_ensemble  # noqa: E402
from signbridge.live_io import HAND_EDGES, LatestFrame, text_panel  # noqa: E402,F401

MODELS = REPO.parent / "models"


class Speaker(threading.Thread):
    """Speaks sentences in the background so the video loop never blocks."""

    def __init__(self, warm: str | None = None):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.engines: dict = {}
        self.warm = warm

    def _engine(self, lang: str):
        if lang not in self.engines:
            from signbridge.tts import load_tts
            self.engines[lang] = load_tts("sapi") if lang == "en" else load_tts("piper")
        return self.engines[lang]

    def run(self):
        import pythoncom  # SAPI is COM: initialise it on this thread
        pythoncom.CoInitialize()
        if self.warm:  # load the voice (and run it once) now, so the first sentence is not delayed
            try:
                self._engine(self.warm).synthesize("Hello." if self.warm == "en" else "Xin chào.")
            except Exception as e:  # noqa: BLE001
                print(f"[speech] {e}", flush=True)
        while True:
            lang, text = self.q.get()
            try:
                self._engine(lang).synthesize(text).play()
            except Exception as e:  # noqa: BLE001
                print(f"[speech] {e}", flush=True)


PANEL_H = 190


def overlay(frame, hands, present, state: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    for s in range(2):
        if present[s]:
            pts = (hands[s, :, :2] * [w, h]).astype(int)
            for a, b in HAND_EDGES:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)
    view = cv2.flip(frame, 1)  # mirror only for display
    if state.get("rest_y") is not None and 0 < state["rest_y"] < 1:
        y = int(state["rest_y"] * h)
        col = (80, 80, 255) if state["in_sign"] else (120, 220, 120)
        cv2.line(view, (0, y), (w, y), col, 2)
        cv2.putText(view, "lower both wrists below this line between signs", (12, max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    elif state.get("rest_y") is not None:
        cv2.putText(view, "rest line is below the picture: move back or lower your hands out of view",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
    if state.get("warn_until", 0) > time.time():
        cv2.putText(view, "Sign cut at 4 s: your hands never rested. Lower them below the line.",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, "Sign cut at 4 s: your hands never rested. Lower them below the line.",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 160, 255), 1, cv2.LINE_AA)
    if state["in_sign"]:
        cv2.circle(view, (22, 22), 9, (80, 80, 255), -1)
        cv2.putText(view, "SIGNING", (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(view, "waiting for a sign (lower hands between signs)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, "waiting for a sign (lower hands between signs)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)
    info = (
        (f"last sign: {state['last']}", (255, 255, 255), 20),
        (f"signs: {' '.join(state['pending']) or '-'}", (120, 220, 255), 20),
        (f"EN: {state['en']}", (120, 255, 120), 24),
        (f"VI: {state['vi']}", (255, 220, 120), 24),
    )
    panel = text_panel(info, w, PANEL_H).copy()
    footer = (f"{state['fps']:.0f} fps | landmarks {state['ms']:.0f} ms | speech: {state['speak']} | "
              "29 ASL greeting signs, phrase table (not a translation model)")
    cv2.putText(panel, footer, (12, PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    return np.vstack([view, panel])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=Path, help="run on a video file instead of the webcam")
    ap.add_argument("--ckpt", type=Path, nargs="+", default=[MODELS / "signs" / "asl_greetings.pt"],
                    help="one checkpoint, or several to average (e.g. the 3 seed files)")
    ap.add_argument("--speak", choices=["en", "vi", "off"], default="en")
    ap.add_argument("--headless", action="store_true", help="no window (for tests with --video)")
    ap.add_argument("--min-confidence", type=float, default=LiveConfig.min_confidence)
    ap.add_argument("--log", type=Path, help="write all events to this JSON file")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds (smoke tests)")
    ap.add_argument("--mp-mode", choices=["image", "video"], default="image",
                    help="MediaPipe: 'image' = detect every frame (as the training features; default), "
                         "'video' = track across frames (faster, but keeps resting hands 'present' longer)")
    ap.add_argument("--rest-line", type=float, default=LiveConfig.rest_line,
                    help="how far below the shoulders (in shoulder widths) the hands count as resting")
    ap.add_argument("--save-landmarks", type=Path,
                    help="save the session's per-frame landmarks (no images) to this .npz, for diagnosis")
    args = ap.parse_args()

    ckpts = [c for c in args.ckpt if c.exists()]
    if not ckpts:
        print(f"no checkpoint found: {args.ckpt}")
        return 1
    if len(ckpts) < len(args.ckpt):
        print(f"[warning] using {len(ckpts)} of {len(args.ckpt)} checkpoints (missing files)")
    model, classes, t, meta = load_ensemble(ckpts)
    cfg = LiveConfig(min_confidence=args.min_confidence, rest_line=args.rest_line)
    session = SignSession(model, classes, t, cfg)
    ext = LandmarkExtractor(MODELS / "mediapipe" / "pose_landmarker_lite.task",
                            MODELS / "mediapipe" / "hand_landmarker.task", mode=args.mp_mode, parallel=True)
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("cannot open video source")
        return 1
    file_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else None
    grabber = None if args.video else LatestFrame(cap)  # a video file is read frame by frame, no drops
    if grabber:
        grabber.start()
    speaker = None
    speak = "off" if args.headless else args.speak
    if speak != "off":
        speaker = Speaker(warm=speak)
        speaker.start()
    state = {"in_sign": False, "last": "-", "pending": [], "en": "", "vi": "", "fps": 0.0, "ms": 0.0, "speak": speak,
             "rest_y": None, "warn_until": 0.0}
    rec: dict[str, list] = {"t": [], "pose": [], "hands": [], "present": [], "pose_ok": []}
    log: list[dict] = []
    proc_ms: list[float] = []
    loop_t: list[float] = []
    k, seq, t_start = 0, 0, time.time()
    while True:
        if grabber:
            frame, t_cap, seq = grabber.get(seq)
            ok = frame is not None
            now = t_cap - t_start
        else:
            ok, frame = cap.read()
            now = k / file_fps if file_fps else time.time() - t_start
        if not ok or (args.seconds and time.time() - t_start > args.seconds):
            break
        t0 = time.perf_counter()
        pose, hands, present, pose_ok = ext.frame(frame, int(now * 1000))
        events = session.push(now, pose, hands, present, pose_ok)
        if args.save_landmarks:
            for key_, val_ in (("t", now), ("pose", pose), ("hands", hands), ("present", present), ("pose_ok", pose_ok)):
                rec[key_].append(val_)
        state["rest_y"] = rest_line_y(pose, pose_ok, cfg)
        proc_ms.append((time.perf_counter() - t0) * 1000)
        loop_t.append(time.perf_counter())
        k += 1
        for ev in events:
            if isinstance(ev, SignEvent) and ev.gloss:
                state["last"] = (f"{ev.gloss} ({ev.confidence:.0%})" if ev.accepted
                                 else f"? (best {ev.gloss} {ev.confidence:.0%}, below threshold)")
                log.append({"type": "sign", "t": round(ev.t, 2), "gloss": ev.gloss, "conf": round(ev.confidence, 3),
                            "accepted": ev.accepted, "top3": [(g, round(p, 3)) for g, p in ev.top3],
                            "frames": ev.n_frames, "dur_s": round(ev.duration_s, 2), "ended_by": ev.ended_by})
                if ev.ended_by == "max":
                    state["warn_until"] = time.time() + 3.0
                print(f"[{now:6.2f}s] sign {state['last']}  top3={[(g, round(p, 2)) for g, p in ev.top3]}", flush=True)
            elif isinstance(ev, SentenceEvent):
                state["en"], state["vi"] = ev.composed.english, ev.composed.vietnamese
                log.append({"type": "sentence", "t": round(ev.t, 2), "glosses": ev.glosses,
                            "en": ev.composed.english, "vi": ev.composed.vietnamese})
                print(f"[{now:6.2f}s] SENTENCE {ev.glosses} -> EN: {ev.composed.english} | VI: {ev.composed.vietnamese}",
                      flush=True)
                if speaker and speak != "off":
                    speaker.q.put((speak, ev.composed.english if speak == "en" else ev.composed.vietnamese))
        state["in_sign"] = session.in_sign
        state["pending"] = session.pending_glosses
        state["ms"] = float(np.mean(proc_ms[-30:]))
        if len(loop_t) > 1:
            recent = loop_t[-31:]
            state["fps"] = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)
        if not args.headless:
            cv2.imshow("SignBridge - ASL greetings demo", overlay(frame, hands, present, state))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                for ev in session.flush(now):
                    if isinstance(ev, SentenceEvent):
                        state["en"], state["vi"] = ev.composed.english, ev.composed.vietnamese
                        if speaker and speak != "off":
                            speaker.q.put((speak, ev.composed.english if speak == "en" else ev.composed.vietnamese))
            elif key == ord("c"):
                session.flush(now)
                state.update(en="", vi="", last="-")
            elif key == ord("l"):
                speak = "vi" if speak == "en" else "en"
                state["speak"] = speak
            elif key == ord("m"):
                speak = "off" if speak != "off" else args.speak
                state["speak"] = speak
    if grabber:
        grabber.stop()
    if args.save_landmarks and rec["t"]:
        args.save_landmarks.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_landmarks, **{k_: np.array(v_) for k_, v_ in rec.items()})
    for ev in session.flush(k / file_fps if file_fps else time.time() - t_start):
        if isinstance(ev, SignEvent) and ev.gloss:
            log.append({"type": "sign", "t": round(ev.t, 2), "gloss": ev.gloss, "conf": round(ev.confidence, 3),
                        "accepted": ev.accepted})
            print(f"sign {ev.gloss} ({ev.confidence:.0%}) accepted={ev.accepted}")
        elif isinstance(ev, SentenceEvent):
            log.append({"type": "sentence", "glosses": ev.glosses, "en": ev.composed.english,
                        "vi": ev.composed.vietnamese})
            print(f"SENTENCE {ev.glosses} -> EN: {ev.composed.english} | VI: {ev.composed.vietnamese}")
    cap.release()
    cv2.destroyAllWindows()
    ext.close()
    loop_fps = (len(loop_t) - 1) / (loop_t[-1] - loop_t[0]) if len(loop_t) > 1 else 0.0
    print(f"frames {k}, mean landmark+model time {np.mean(proc_ms):.1f} ms/frame, loop {loop_fps:.1f} fps "
          f"(mediapipe mode: {args.mp_mode})")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps({"events": log, "frames": k, "mean_ms": float(np.mean(proc_ms)),
                                        "loop_fps": loop_fps, "mp_mode": args.mp_mode,
                                        "ckpt": [c.name for c in ckpts]},
                                       indent=1, ensure_ascii=False), encoding="utf-8")
    if speaker:
        time.sleep(0.5)
        while not speaker.q.empty():
            time.sleep(0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
