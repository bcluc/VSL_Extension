"""Label set for static VSL fingerspelling.

Each label has an ASCII id (used for folder names, checkpoints and APIs) and the text shown to
the user. The default set is the static handshapes of the Vietnamese manual alphabet plus the
two diacritic handshapes "mũ" (^, makes Â Ê Ô) and "râu" (horn, makes Ơ Ư). Letters that need
movement or tone marks are not covered. Record and train any other set with ``--labels``.
"""
from __future__ import annotations

DEFAULT_LABELS: list[tuple[str, str]] = [
    ("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("DD", "Đ"), ("E", "E"), ("G", "G"),
    ("H", "H"), ("I", "I"), ("K", "K"), ("L", "L"), ("M", "M"), ("N", "N"), ("O", "O"),
    ("P", "P"), ("Q", "Q"), ("R", "R"), ("S", "S"), ("T", "T"), ("U", "U"), ("V", "V"),
    ("X", "X"), ("Y", "Y"), ("MU", "Mũ"), ("RAU", "Râu"),
]
DISPLAY = dict(DEFAULT_LABELS)
MODIFIERS = {"MU": {"A": "Â", "E": "Ê", "O": "Ô"}, "RAU": {"O": "Ơ", "U": "Ư"}}


def display(label_id: str) -> str:
    return DISPLAY.get(label_id, label_id)


def label_from_dirname(name: str) -> str:
    """Folder name -> shown text (folders are named by label id)."""
    return DISPLAY.get(name, name)


def append_letter(text: str, label_id: str) -> str:
    """Add a recognised label to typed text; "MU" / "RAU" modify the previous letter."""
    if label_id in MODIFIERS:
        prev = text[-1:]
        new = MODIFIERS[label_id].get(prev.upper())
        if new is None:
            return text  # nothing to modify: ignore the diacritic
        return text[:-1] + (new if prev.isupper() else new.lower())
    return text + display(label_id)
