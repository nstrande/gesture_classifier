from __future__ import annotations

from dataclasses import dataclass


@dataclass
class config:
    MODEL = "src.models.model_config.classification_lstm"
    FRAMES_PER_SECOND = 8
    SECONDS_PER_SEQUENCE = 1.5
    SEQUENCE_LENGTH = FRAMES_PER_SECOND * SECONDS_PER_SEQUENCE
    MAX_EXTRAPOLATION_FRAMES = 5
