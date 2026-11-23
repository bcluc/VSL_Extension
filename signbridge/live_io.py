"""Shared helpers for the live webcam demos: newest-frame camera reader and cached text rendering."""
from __future__ import annotations

import functools
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

FONT = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class LatestFrame(threading.Thread):
    """Reads the camera continuously and keeps only the newest frame.

    Without this, ``cap.read()`` returns frames that waited in the driver's buffer while the
    previous frame was processed, so the picture (and the recognition) lags behind the signer.
    """

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap = cap
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None
        self.t = 0.0
        self.seq = 0
        self.running = True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            t = time.time()
            with self.cond:
                if not ok:
                    self.running = False
                else:
                    self.frame, self.t, self.seq = f, t, self.seq + 1
                self.cond.notify_all()

    def get(self, last_seq: int, timeout: float = 2.0) -> tuple[np.ndarray | None, float, int]:
        """The newest frame after ``last_seq`` (waits for it), or None if the camera stopped."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout=timeout)
            if self.seq == last_seq:
                return None, 0.0, last_seq
            return self.frame, self.t, self.seq

    def stop(self):
        self.running = False
        self.join(timeout=2.0)


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


@functools.lru_cache(maxsize=None)
def font(size: int):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()


@functools.lru_cache(maxsize=8)
def text_panel(lines: tuple[tuple[str, tuple[int, int, int], int], ...], w: int, h: int) -> np.ndarray:
    """Text block rendered with PIL (Vietnamese diacritics). Cached: re-rendered only when the text changes."""
    from PIL import Image, ImageDraw
    pil = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(pil)
    y = 8
    for text, rgb, size in lines:
        d.text((12, y), text, fill=rgb, font=font(size), stroke_width=2, stroke_fill=(0, 0, 0))
        y += size + 6
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_hand(view: np.ndarray, pts_xy: np.ndarray, color=(0, 200, 110)) -> None:
    """Draw 21 normalised (x, y) hand landmarks on ``view`` in place."""
    h, w = view.shape[:2]
    p = [(int(x * w), int(y * h)) for x, y in pts_xy]
    for a, b in HAND_EDGES:
        cv2.line(view, p[a], p[b], color, 2, cv2.LINE_AA)
    for q in p:
        cv2.circle(view, q, 3, (255, 255, 255), -1, cv2.LINE_AA)
