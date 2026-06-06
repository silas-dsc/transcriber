"""Rhythm analysis: tempo, beat grid and a seconds<->beats mapping.

The beat grid recovered here is the backbone of quantisation.  Every note
onset/offset (in seconds) is converted to a *beat position* by interpolating
against the detected beat times, which makes the subsequent note durations
robust to small tempo fluctuations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RhythmInfo:
    """Result of rhythm analysis.

    Attributes:
        tempo: Estimated tempo in beats per minute.
        beat_times: Times (seconds) of detected beats, ascending.
        beats_per_measure: Numerator of the guessed time signature.
        beat_unit: Denominator of the guessed time signature (4 == quarter).
    """

    tempo: float
    beat_times: np.ndarray
    beats_per_measure: int = 4
    beat_unit: int = 4
    _beat_index: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Fractional beat number for each detected beat time: 0, 1, 2, ...
        self._beat_index = np.arange(len(self.beat_times), dtype=float)

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo if self.tempo > 0 else 0.5

    def time_to_beats(self, t: float | np.ndarray) -> np.ndarray:
        """Map time(s) in seconds to fractional beat positions.

        Uses linear interpolation across the detected beat grid and a constant
        tempo extrapolation outside the grid, so onsets before the first or
        after the last detected beat are still placed sensibly.
        """
        t = np.asarray(t, dtype=float)
        if len(self.beat_times) >= 2:
            beats = np.interp(t, self.beat_times, self._beat_index)
            # np.interp clamps outside the range; extrapolate linearly instead.
            spb = self.seconds_per_beat
            before = t < self.beat_times[0]
            after = t > self.beat_times[-1]
            beats = np.where(before, (t - self.beat_times[0]) / spb, beats)
            beats = np.where(
                after,
                self._beat_index[-1] + (t - self.beat_times[-1]) / spb,
                beats,
            )
            return beats
        # Not enough beats detected: assume constant tempo from t=0.
        return t / self.seconds_per_beat


def analyze_rhythm(samples: np.ndarray, sr: int) -> RhythmInfo:
    """Estimate tempo and a beat grid for ``samples`` using librosa."""
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    if tempo <= 0 or not np.isfinite(tempo):
        tempo = 120.0

    beats_per_measure, beat_unit = _guess_time_signature(samples, sr, beat_times)
    return RhythmInfo(
        tempo=round(tempo, 2),
        beat_times=np.asarray(beat_times, dtype=float),
        beats_per_measure=beats_per_measure,
        beat_unit=beat_unit,
    )


def _guess_time_signature(
    samples: np.ndarray, sr: int, beat_times: np.ndarray
) -> tuple[int, int]:
    """Very small heuristic for the time-signature numerator.

    We compute onset-strength at each beat and look for the periodicity of
    accent peaks (downbeats).  We only distinguish the two overwhelmingly
    common cases, 4/4 and 3/4, and default to 4/4 when unsure.
    """
    import librosa

    if len(beat_times) < 6:
        return 4, 4

    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    beat_strength = np.interp(beat_times, times, onset_env)

    best_period, best_score = 4, -np.inf
    for period in (3, 4):
        # Sum accent strength on hypothesised downbeats for each phase and take
        # the strongest phase; the period that yields the strongest, most
        # consistent downbeat accent wins.
        phase_scores = [
            beat_strength[phase::period].mean()
            for phase in range(period)
            if len(beat_strength[phase::period]) > 0
        ]
        score = max(phase_scores) if phase_scores else 0.0
        if score > best_score:
            best_score, best_period = score, period

    return best_period, 4
