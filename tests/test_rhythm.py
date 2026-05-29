"""Tests for rhythm analysis and the seconds<->beats mapping."""

import numpy as np

from transcriber.rhythm import RhythmInfo, analyze_rhythm


def test_tempo_estimate_is_close_to_truth(melody_audio, sample_rate, tempo_bpm):
    rhythm = analyze_rhythm(melody_audio, sample_rate)
    # Beat trackers commonly land on a half/double-tempo octave; accept those.
    candidates = [tempo_bpm, tempo_bpm / 2, tempo_bpm * 2]
    assert min(abs(rhythm.tempo - c) for c in candidates) < 12.0


def test_time_to_beats_is_monotonic_and_anchored():
    beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # 120 BPM
    rhythm = RhythmInfo(tempo=120.0, beat_times=beat_times)
    beats = rhythm.time_to_beats(beat_times)
    assert np.allclose(beats, [0, 1, 2, 3, 4], atol=1e-6)
    # Monotonic increasing for an increasing time vector.
    dense = rhythm.time_to_beats(np.linspace(0, 2, 50))
    assert np.all(np.diff(dense) >= -1e-9)


def test_time_to_beats_extrapolates_past_grid():
    beat_times = np.array([0.0, 0.5, 1.0])
    rhythm = RhythmInfo(tempo=120.0, beat_times=beat_times)
    # 1.5s is one beat past the last grid point at 120 BPM -> beat 3.
    assert abs(float(rhythm.time_to_beats(1.5)) - 3.0) < 1e-6
    # Negative time before the grid extrapolates to a negative beat.
    assert float(rhythm.time_to_beats(-0.5)) < 0


def test_handles_too_few_beats_gracefully():
    rhythm = RhythmInfo(tempo=120.0, beat_times=np.array([0.3]))
    # Falls back to constant-tempo mapping from t=0.
    assert abs(float(rhythm.time_to_beats(0.5)) - 1.0) < 1e-6
