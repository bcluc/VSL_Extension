"""Per-frame transforms: color conversion, resize (letterbox or stretch), normalization."""
from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_frame(frame: np.ndarray, size: tuple[int, int], keep_aspect: bool = True,
                 pad_value: int = 0) -> np.ndarray:
    """Resize to ``size`` = (height, width). With ``keep_aspect`` the frame is letterboxed
    so signing hands near the frame edge are not cropped away."""
    th, tw = size
    h, w = frame.shape[:2]
    if not keep_aspect:
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    scale = min(th / h, tw / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((th, tw) + frame.shape[2:], pad_value, dtype=frame.dtype)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def to_model_input(frame_bgr: np.ndarray, size: tuple[int, int], keep_aspect: bool,
                   mean: tuple[float, ...], std: tuple[float, ...], color: str = "rgb") -> np.ndarray:
    """BGR uint8 HxWx3 -> normalized float32 CxHxW."""
    img = resize_frame(frame_bgr, size, keep_aspect)
    if color == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    elif color != "bgr":
        raise ValueError(f"unknown color mode {color!r}")
    x = img.astype(np.float32) / 255.0
    m = np.asarray(mean, dtype=np.float32)[: x.shape[-1]]
    s = np.asarray(std, dtype=np.float32)[: x.shape[-1]]
    x = (x - m) / s
    return np.ascontiguousarray(x.transpose(2, 0, 1))
