"""Pitch transcription: audio -> note events.

Primary back-end is `Spotify's basic-pitch
<https://github.com/spotify/basic-pitch>`_, a neural polyphonic note
estimator.  When it is not installed we fall back to ``librosa``'s pYIN
monophonic pitch tracker, which is appropriate for single-voice stems such as
bass or a solo vocal line.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Note:
    """A transcribed note event.

    Attributes:
        start: Onset time in seconds.
        end: Offset time in seconds.
        pitch: MIDI pitch number (0-127).
        velocity: MIDI velocity (1-127).
    """

    start: float
    end: float
    pitch: int
    velocity: int = 80

    @property
    def duration(self) -> float:
        return self.end - self.start


def transcribe_pitch(
    samples: np.ndarray,
    sr: int,
    backend: str = "auto",
    monophonic: bool = False,
) -> list[Note]:
    """Transcribe pitched ``samples`` into a list of :class:`Note`.

    Args:
        samples: Mono ``float32`` audio.
        sr: Sample rate.
        backend: ``"basic-pitch"``, ``"pyin"`` or ``"auto"``.
        monophonic: Hint that the stem is a single voice.  When using
            basic-pitch this collapses overlapping notes; with the pYIN
            fallback it is implied.

    Returns:
        Note events sorted by onset time.
    """
    if backend == "auto":
        backend = "basic-pitch" if _basic_pitch_available() else "pyin"

    if backend == "basic-pitch":
        notes = _transcribe_basic_pitch(samples, sr)
        return _make_monophonic(notes) if monophonic else notes
    if backend == "pyin":
        return _transcribe_pyin(samples, sr)
    raise ValueError(f"Unknown pitch backend: {backend!r}")


def _basic_pitch_available() -> bool:
    """True only if basic-pitch is installed *and* has a usable runtime.

    basic-pitch supports several inference back-ends (TensorFlow, CoreML,
    TFLite, ONNX) and picks whichever is importable.  If none is present its
    package import itself raises, so we probe by importing and confirming at
    least one back-end flag is set.  This lets ``auto`` fall back to pYIN
    instead of crashing on a half-installed basic-pitch.
    """
    import importlib.util

    if importlib.util.find_spec("basic_pitch") is None:
        return False
    try:
        import basic_pitch

        return any(
            getattr(basic_pitch, flag, False)
            for flag in ("TF_PRESENT", "CT_PRESENT", "TFLITE_PRESENT", "ONNX_PRESENT")
        )
    except Exception:
        logger.warning(
            "basic-pitch is installed but has no usable inference back-end "
            "(install one of: onnxruntime, coremltools, tensorflow). "
            "Falling back to pYIN."
        )
        return False


def _transcribe_basic_pitch(samples: np.ndarray, sr: int) -> list[Note]:
    """Transcribe using Spotify basic-pitch (requires the ``[full]`` extras)."""
    import soundfile as sf
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    logger.info("Transcribing pitch with basic-pitch")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, samples, sr, subtype="PCM_16")
        _model_output, midi_data, _note_events = predict(tmp.name, ICASSP_2022_MODEL_PATH)

    notes: list[Note] = []
    for instrument in midi_data.instruments:
        for n in instrument.notes:
            notes.append(
                Note(start=n.start, end=n.end, pitch=int(n.pitch), velocity=int(n.velocity))
            )
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _transcribe_pyin(
    samples: np.ndarray,
    sr: int,
    fmin: float = 65.0,
    fmax: float = 2093.0,
) -> list[Note]:
    """Monophonic transcription using librosa pYIN + note segmentation."""
    import librosa

    logger.info("Transcribing pitch with librosa pYIN fallback")
    hop_length = 256
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Convert frame-wise f0 to MIDI, rounding to the nearest semitone, then
    # merge consecutive frames of the same pitch into single note events.
    midi = librosa.hz_to_midi(f0)
    notes: list[Note] = []
    cur_pitch: int | None = None
    cur_start = 0.0

    def _flush(end_time: float) -> None:
        nonlocal cur_pitch
        if cur_pitch is not None and end_time - cur_start >= 0.05:
            notes.append(Note(start=cur_start, end=end_time, pitch=cur_pitch))
        cur_pitch = None

    for i, (t, m, voiced) in enumerate(zip(times, midi, voiced_flag)):
        pitch = int(round(m)) if voiced and np.isfinite(m) else None
        if pitch != cur_pitch:
            _flush(t)
            if pitch is not None:
                cur_pitch = pitch
                cur_start = t
    _flush(times[-1] if len(times) else 0.0)
    return notes


def _make_monophonic(notes: list[Note]) -> list[Note]:
    """Collapse overlapping notes to the highest-pitch note at each instant."""
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: n.start)
    result: list[Note] = []
    for n in notes:
        if result and n.start < result[-1].end:
            prev = result[-1]
            if n.pitch > prev.pitch:
                # Truncate the previous note where the louder/higher one starts.
                prev.end = n.start
                if prev.end <= prev.start:
                    result.pop()
                result.append(n)
            else:
                # Keep the previous note; drop the lower overlapping note.
                continue
        else:
            result.append(n)
    return [n for n in result if n.end > n.start]
