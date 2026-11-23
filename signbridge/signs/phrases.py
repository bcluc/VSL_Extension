"""Hand-written phrase table: recognised ASL gloss sequence -> English and Vietnamese sentence.

This is a deterministic lookup, not a learned translation model. It covers the greeting /
introduction use case only. Unknown sequences fall back to the bracketed gloss words, so the
output never pretends to be a fluent translation it is not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ASL Citizen has numbered lexical variants (WHAT1/WHAT2...) that mean the same word.
_VARIANT = re.compile(r"^([A-Z]+?)\d+$")

# (gloss sequence, English, Vietnamese). Longer entries are matched first.
PHRASES: list[tuple[tuple[str, ...], str, str]] = [
    (("NICE", "MEET", "YOU"), "Nice to meet you.", "Rất vui được gặp bạn."),
    (("NICE", "MEET"), "Nice to meet you.", "Rất vui được gặp bạn."),
    (("WHAT", "YOUR", "NAME"), "What is your name?", "Bạn tên là gì?"),
    (("YOUR", "NAME", "WHAT"), "What is your name?", "Bạn tên là gì?"),
    (("YOUR", "NAME"), "What is your name?", "Bạn tên là gì?"),
    (("MY", "NAME"), "My name is ...", "Tên tôi là ..."),
    (("ME", "LEARN", "SIGN"), "I am learning sign language.", "Tôi đang học ngôn ngữ ký hiệu."),
    (("LEARN", "SIGN"), "I am learning sign language.", "Tôi đang học ngôn ngữ ký hiệu."),
    (("SEE", "YOU", "LATER"), "See you later.", "Hẹn gặp lại bạn."),
    (("SEE", "LATER"), "See you later.", "Hẹn gặp lại."),
    (("HOW", "YOU"), "How are you?", "Bạn có khỏe không?"),
    (("YOU", "HOW"), "How are you?", "Bạn có khỏe không?"),
    (("ME", "FINE"), "I'm fine.", "Tôi khỏe."),
    (("FINE", "ME"), "I'm fine.", "Tôi khỏe."),
    (("ME", "GOOD"), "I'm good.", "Tôi ổn."),
    (("ME", "DEAF"), "I am Deaf.", "Tôi là người Điếc."),
    (("DEAF", "ME"), "I am Deaf.", "Tôi là người Điếc."),
    (("YOU", "DEAF"), "Are you Deaf?", "Bạn là người Điếc phải không?"),
    (("ME", "HEARING"), "I am hearing.", "Tôi là người nghe được."),
    (("YOU", "HEARING"), "Are you hearing?", "Bạn là người nghe được phải không?"),
    (("MY", "FRIEND"), "My friend.", "Bạn của tôi."),
    (("YOU", "FRIEND"), "You are my friend.", "Bạn là bạn của tôi."),
    (("HELLO",), "Hello.", "Xin chào."),
    (("THANKYOU",), "Thank you.", "Cảm ơn."),
    (("SORRY",), "Sorry.", "Xin lỗi."),
    (("PLEASE",), "Please.", "Làm ơn."),
    (("BYE",), "Goodbye.", "Tạm biệt."),
    (("YES",), "Yes.", "Vâng."),
    (("NO",), "No.", "Không."),
    (("FINE",), "Fine.", "Khỏe."),
    (("GOOD",), "Good.", "Tốt."),
    (("FRIEND",), "Friend.", "Bạn bè."),
]
PHRASES.sort(key=lambda p: -len(p[0]))


def base_gloss(gloss: str) -> str:
    """WHAT1 -> WHAT, DEAF2 -> DEAF; other glosses unchanged."""
    m = _VARIANT.match(gloss.upper())
    return m.group(1) if m else gloss.upper()


@dataclass(frozen=True)
class Composed:
    english: str
    vietnamese: str
    glosses: tuple[str, ...]
    matched: bool  # False if any part fell back to raw gloss words


def compose(glosses: list[str]) -> Composed:
    """Greedy longest-match over the phrase table; unmatched glosses are shown in brackets."""
    g = [base_gloss(x) for x in glosses]
    en, vi = [], []
    i, all_matched = 0, True
    while i < len(g):
        for seq, e, v in PHRASES:
            if tuple(g[i:i + len(seq)]) == seq:
                en.append(e)
                vi.append(v)
                i += len(seq)
                break
        else:
            en.append(f"[{g[i].lower()}]")
            vi.append(f"[{g[i].lower()}]")
            all_matched = False
            i += 1
    return Composed(" ".join(en), " ".join(vi), tuple(g), all_matched)
