"""
VSL letters - Microsoft Teams tab (local demo)
==============================================
FastAPI server that:
  * serves the Teams tab page (static/index.html)
  * exposes POST /api/predict : JPEG frame (already mirrored) -> hand landmarks -> letter

Recognition uses ``signbridge.letters`` (the same code as scripts/live_letter_demo.py).
Train a model first: scripts/record_letters.py, then scripts/train_letter_classifier.py.

Run (from the project root):
    python teams_app/server.py                      # https://localhost:53000
    python teams_app/server.py --port 53000 --no-ssl   # plain http (browser test only)
"""

import argparse
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))

from signbridge.letters.labels import display  # noqa: E402
from signbridge.letters.recognizer import LetterRecognizer  # noqa: E402

MODELS = ROOT.parent / "models"
CHECKPOINT = Path(os.environ.get("SIGNBRIDGE_LETTER_CKPT", MODELS / "letters" / "vsl_letters.pt"))
HAND_MODEL = MODELS / "mediapipe" / "hand_landmarker.task"
MAX_UPLOAD = 4 * 1024 * 1024  # 4 MB per frame

# ── Load the model + MediaPipe once at startup ───────────────────────────────
try:
    recognizer = LetterRecognizer(CHECKPOINT, HAND_MODEL)
    MODEL_NAME = f"Landmark MLP ({len(recognizer.labels)} letters)"
    print(f"Loaded {CHECKPOINT.name}")
except FileNotFoundError as e:
    recognizer, MODEL_NAME = None, None
    print(f"[!] {e}\nThe page will load, but recognition is off until a model is trained.")
lock = threading.Lock()  # MediaPipe IMAGE-mode landmarker is not thread-safe

app = FastAPI(title="VSL letters Teams tab")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/config")
def config():
    """Teams tab configuration page (shown when the app is added to a meeting)."""
    return FileResponse(APP_DIR / "static" / "config.html")


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return ("<h1>Privacy</h1><p>Local research demo. Webcam frames are sent only to "
            "this computer's local server, processed in memory and never stored.</p>")


@app.get("/terms", response_class=HTMLResponse)
def terms():
    return "<h1>Terms</h1><p>Research demo, provided as-is, for evaluation only.</p>"


@app.get("/api/models")
def list_models():
    return {"ready": recognizer is not None, "name": MODEL_NAME,
            "labels": [display(lb) for lb in recognizer.labels] if recognizer else []}


@app.post("/api/predict")
async def predict(request: Request):
    if recognizer is None:
        raise HTTPException(503, "No trained letter model. Run scripts/record_letters.py, then "
                                 "scripts/train_letter_classifier.py, then restart the server.")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty body: send a JPEG/PNG frame as the request body.")
    if len(body) > MAX_UPLOAD:
        raise HTTPException(413, "Frame too large.")
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image.")
    with lock:
        pred = recognizer.predict(frame)
    out = {"hand": pred.hand is not None, "model": MODEL_NAME, "ms": round(pred.ms, 1)}
    if pred.hand is not None:
        out["top"] = [{"id": lb, "label": display(lb), "conf": round(p, 4)} for lb, p in pred.top(3)]
        out["landmarks"] = [[round(float(x), 4), round(float(y), 4)] for x, y in pred.hand.landmarks[:, :2]]
        out["bbox"] = list(pred.hand.bbox)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=53000)
    ap.add_argument("--cert", default=str(APP_DIR / "certs" / "localhost.crt"))
    ap.add_argument("--key",  default=str(APP_DIR / "certs" / "localhost.key"))
    ap.add_argument("--no-ssl", action="store_true", help="Serve plain HTTP (not usable inside Teams)")
    args = ap.parse_args()

    import uvicorn
    ssl = {}
    if not args.no_ssl:
        if not (Path(args.cert).exists() and Path(args.key).exists()):
            sys.exit("HTTPS certificate missing. Run: python teams_app/make_cert.py")
        ssl = {"ssl_certfile": args.cert, "ssl_keyfile": args.key}
    scheme = "http" if args.no_ssl else "https"
    print(f"\nVSL Teams tab running at {scheme}://localhost:{args.port}/\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
