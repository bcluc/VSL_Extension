"""Static VSL fingerspelling from hand landmarks: features, model, recognizer, recordings.

Pipeline: mirrored webcam frame -> MediaPipe HandLandmarker (21 points) -> 83-d hand-shape
features -> small MLP -> letter probabilities -> "hold to type" text. Training data is recorded
with ``scripts/record_letters.py`` (landmarks only, no images).
"""
from .features import FEATURE_DIM, augment, letter_features
from .labels import DEFAULT_LABELS, append_letter, display
from .model import LetterMLP, load_letter_model, save_letter_model

__all__ = ["DEFAULT_LABELS", "FEATURE_DIM", "LetterMLP", "append_letter", "augment", "display",
           "letter_features", "load_letter_model", "save_letter_model"]
