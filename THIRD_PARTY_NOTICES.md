# Third-party notices

All code and documentation in this repository are covered by the MIT license in
[LICENSE](LICENSE). The repository contains no model files and no datasets. What SignBridge
downloads or installs keeps its own terms, listed below.

## 1. Models (downloaded to `../models/`, not in Git)

| Model | Source | Terms |
|---|---|---|
| `hand_landmarker.task`, `pose_landmarker_lite.task` | Google MediaPipe (`scripts/fetch_mediapipe_models.py`) | Apache License 2.0 |
| Piper voice `vi_VN-vais1000-medium` | trained on VAIS-1000 (`scripts/fetch_tts_voice.py`) | CC BY 4.0 (dataset) |
| ASL greetings classifier | trained here on ASL Citizen | non-commercial research only; **do not distribute** (see §3) |
| VSL letter classifier | trained here on your own recordings | yours |

## 2. Software dependencies (installed with pip, not bundled)

| Package | License | Note |
|---|---|---|
| `piper-tts` (Vietnamese TTS) | **GPL-3.0-or-later** | Fine for research. A product that ships with Piper takes on GPL obligations. Isolate Piper in its own process or replace it; see [docs/tts_implementation.md](docs/tts_implementation.md). |
| MediaPipe | Apache-2.0 | |
| PyTorch, torchvision | BSD-3-Clause | |
| OpenCV | Apache-2.0 | |
| NumPy | BSD-3-Clause | |
| Pillow | MIT-CMU (HPND) | |
| FastAPI | MIT | `teams_app/` |
| pywin32 | PSF | Windows English speech (SAPI) |
| Microsoft Teams JavaScript SDK | MIT | loaded from unpkg by `teams_app/static/index.html` |

## 3. Datasets (not in this repository)

All datasets stay in `../datasets/`, outside Git.

| Dataset | Terms | Use here |
|---|---|---|
| **ASL Citizen** (Microsoft Research) | Microsoft Research License: non-commercial research only, no redistribution, delete personal data at the end of the research | trains the 29-sign ASL greetings demo model. That model and the conversation test videos must not be shared. |
| **How2Sign** (Duarte et al., CVPR 2021) | CC BY-NC 4.0 | infrastructure checks only |
| **RWTH-PHOENIX-Weather 2014 T** | CC BY-NC-SA 3.0, research only | WER metric validation against the official sclite output |
| Own VSL letter recordings (`scripts/record_letters.py`) | recorded by the user; hand landmarks only, no images | trains the VSL letter classifier. Get consent from anyone you record. |
