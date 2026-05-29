"""Shared pytest fixtures: synthetic audio with known tempo/content."""

from __future__ import annotations

import numpy as np
import pytest

SR = 22050
BPM = 120.0


def _seconds_per_beat() -> float:
    return 60.0 / BPM


@pytest.fixture(scope="session")
def sample_rate() -> int:
    return SR


@pytest.fixture(scope="session")
def tempo_bpm() -> float:
    return BPM


@pytest.fixture(scope="session")
def melody_audio() -> np.ndarray:
    """A clean monophonic C-major scale, one note per beat at 120 BPM."""
    spb = _seconds_per_beat()
    scale = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25]
    dur = len(scale) * spb
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    audio = np.zeros_like(t)
    for i, f in enumerate(scale):
        start, end = i * spb, i * spb + spb * 0.9
        mask = (t >= start) & (t < end)
        n = int(mask.sum())
        if n:
            audio[mask] += 0.5 * np.sin(2 * np.pi * f * t[mask]) * np.hanning(n)
    return audio.astype(np.float32)


@pytest.fixture(scope="session")
def drum_audio() -> np.ndarray:
    """Kick on each beat, hi-hat on each off-beat, at 120 BPM."""
    spb = _seconds_per_beat()
    n_beats = 8
    dur = n_beats * spb
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    audio = np.zeros_like(t)
    rng = np.random.default_rng(0)
    for b in range(n_beats):
        bt = b * spb
        k = (t >= bt) & (t < bt + 0.05)
        audio[k] += 0.8 * np.sin(2 * np.pi * 60 * t[k]) * np.exp(-30 * (t[k] - bt))
        ht = bt + spb / 2
        h = (t >= ht) & (t < ht + 0.03)
        if h.any():
            audio[h] += 0.3 * rng.standard_normal(int(h.sum())) * np.exp(-80 * (t[h] - ht))
    return audio.astype(np.float32)


@pytest.fixture(scope="session")
def mix_audio(melody_audio, drum_audio) -> np.ndarray:
    """A full mix combining melody and drums (lengths aligned by trimming)."""
    n = min(len(melody_audio), len(drum_audio))
    mix = melody_audio[:n] + drum_audio[:n]
    peak = np.max(np.abs(mix)) or 1.0
    return (0.9 * mix / peak).astype(np.float32)
