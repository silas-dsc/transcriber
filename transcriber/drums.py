"""Drum transcription via onset detection and spectral band classification.

We detect onsets on the drum stem and classify each hit into kick / snare /
hi-hat by looking at where its spectral energy sits:

* **kick**   – energy concentrated in the low band (< ~150 Hz)
* **hi-hat** – energy concentrated in the high band (> ~6 kHz)
* **snare**  – broadband energy in the mid band, otherwise

This is a deliberately simple, dependency-light classifier.  It will not match
a neural drum transcriber but recovers a musically useful kick/snare/hat
pattern from a separated drum stem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

KICK = "kick"
SNARE = "snare"
HIHAT = "hihat"

# General MIDI percussion key map (channel 10) used when exporting to MIDI.
GM_DRUM_PITCH = {KICK: 36, SNARE: 38, HIHAT: 42}


@dataclass
class DrumHit:
    """A single detected drum stroke.

    Attributes:
        time: Onset time in seconds.
        kind: One of ``"kick"``, ``"snare"`` or ``"hihat"``.
        velocity: MIDI-style velocity (1-127) derived from onset strength.
    """

    time: float
    kind: str
    velocity: int = 90


def transcribe_drums(samples: np.ndarray, sr: int) -> list[DrumHit]:
    """Detect and classify drum hits in a percussive stem."""
    import librosa

    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=True
    )
    if len(onset_frames) == 0:
        return []

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Per-onset accent strength -> velocity.
    env_at_onset = onset_env[np.clip(onset_frames, 0, len(onset_env) - 1)]
    if env_at_onset.max() > 0:
        velocities = 40 + (env_at_onset / env_at_onset.max() * 87.0)
    else:
        velocities = np.full_like(env_at_onset, 90.0)

    # Short-time spectrum so we can read band energy around each onset.
    n_fft = 2048
    hop = 512
    S = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low = freqs < 150
    high = freqs > 6000
    mid = ~low & ~high

    hits: list[DrumHit] = []
    for t, vel in zip(onset_times, velocities):
        frame = int(round(t * sr / hop))
        frame = int(np.clip(frame, 0, S.shape[1] - 1))
        spectrum = S[:, frame]
        total = spectrum.sum() + 1e-9
        low_e = spectrum[low].sum() / total
        mid_e = spectrum[mid].sum() / total
        high_e = spectrum[high].sum() / total

        if low_e > 0.4 and low_e >= high_e:
            kind = KICK
        elif high_e > 0.5 and high_e > low_e:
            kind = HIHAT
        else:
            kind = SNARE if mid_e >= high_e else HIHAT

        hits.append(DrumHit(time=float(t), kind=kind, velocity=int(np.clip(vel, 1, 127))))

    logger.info("Detected %d drum hits", len(hits))
    return hits
