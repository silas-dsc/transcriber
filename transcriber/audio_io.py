"""Audio loading helpers.

Thin wrapper around :mod:`librosa`/:mod:`soundfile` so the rest of the code
base has a single, well-defined way of reading audio into a mono ``float32``
numpy array at a known sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 22.05 kHz is plenty for pitch/onset analysis and keeps everything fast.
# Source separation back-ends resample internally to whatever they need.
DEFAULT_SR = 22050


@dataclass
class Audio:
    """A loaded audio signal.

    Attributes:
        samples: 1-D mono ``float32`` array in the range ``[-1, 1]``.
        sr: Sample rate in Hz.
        path: Original file the audio was loaded from (if any).
    """

    samples: np.ndarray
    sr: int
    path: Path | None = None

    @property
    def duration(self) -> float:
        """Duration of the signal in seconds."""
        return len(self.samples) / float(self.sr)


def load_audio(path: str | Path, sr: int = DEFAULT_SR, mono: bool = True) -> Audio:
    """Load ``path`` into an :class:`Audio` instance.

    Decoding is delegated to ``librosa`` which uses ``soundfile`` for
    WAV/FLAC/OGG and ``audioread``/ffmpeg for compressed formats such as MP3.
    """
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    samples, file_sr = librosa.load(str(path), sr=sr, mono=mono)
    return Audio(samples=np.asarray(samples, dtype=np.float32), sr=int(file_sr), path=path)


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    """Write ``samples`` to ``path`` as a 16-bit PCM WAV file."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(samples, dtype=np.float32).T, sr, subtype="PCM_16")
    return path
