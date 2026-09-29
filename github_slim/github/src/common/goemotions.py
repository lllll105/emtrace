"""GoEmotions label definitions and helpers."""

from __future__ import annotations


GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


LABEL_TO_ID = {label: idx for idx, label in enumerate(GOEMOTIONS_LABELS)}


def label_name(label_id: int) -> str:
    try:
        return GOEMOTIONS_LABELS[int(label_id)]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unknown GoEmotions label id: {label_id}") from exc
