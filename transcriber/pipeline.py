"""End-to-end orchestration: audio file -> MusicXML score.

Wires together stem separation, rhythm analysis, pitch transcription and drum
transcription, then assembles the score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import drums as drums_mod
from . import pitch as pitch_mod
from . import stems as stems_mod
from .audio_io import load_audio
from .rhythm import analyze_rhythm
from .score import build_score, write_musicxml

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for :func:`transcribe`.

    Attributes:
        separation_backend: ``"auto"``, ``"demucs"`` or ``"hpss"``.
        demucs_model: Demucs model name when using the Demucs backend.
        pitch_backend: ``"auto"``, ``"basic-pitch"`` or ``"pyin"``.
        transcribe_drums: Whether to produce a drum part.
        sample_rate: Analysis sample rate.
        monophonic_stems: Stems to treat as a single voice (e.g. ``bass``).
        title: Score title.
    """

    separation_backend: str = "auto"
    demucs_model: str = "htdemucs"
    pitch_backend: str = "auto"
    transcribe_drums: bool = True
    sample_rate: int = 22050
    monophonic_stems: tuple[str, ...] = ("bass",)
    title: str = "Transcription"


@dataclass
class TranscriptionResult:
    """Outcome of a transcription run.

    Attributes:
        score: The assembled music21 score.
        output_path: Path to the written MusicXML file (if written).
        rhythm: Rhythm analysis used for the score.
        stem_names: Names of the stems that were separated.
    """

    score: object
    output_path: str | None
    rhythm: object
    stem_names: list[str] = field(default_factory=list)


def transcribe(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: TranscriptionConfig | None = None,
) -> TranscriptionResult:
    """Transcribe an audio recording to a MusicXML score.

    Args:
        input_path: Path to the input audio file (wav/flac/mp3/...).
        output_path: Where to write the MusicXML.  If ``None`` no file is
            written and only the in-memory score is returned.
        config: Optional :class:`TranscriptionConfig`.

    Returns:
        A :class:`TranscriptionResult`.
    """
    config = config or TranscriptionConfig()
    audio = load_audio(input_path, sr=config.sample_rate)
    logger.info("Loaded %.1fs of audio from %s", audio.duration, input_path)

    # Rhythm is analysed on the full mix for a stable global beat grid.
    rhythm = analyze_rhythm(audio.samples, audio.sr)
    logger.info(
        "Estimated tempo %.1f BPM, %d/%d",
        rhythm.tempo,
        rhythm.beats_per_measure,
        rhythm.beat_unit,
    )

    separated = stems_mod.separate(
        audio.samples,
        audio.sr,
        backend=config.separation_backend,
        model=config.demucs_model,
    )

    pitched_parts: dict[str, list] = {}
    all_drum_hits: list = []
    stem_names: list[str] = []

    for stem in separated:
        stem_names.append(stem.name)
        if stem.is_silent:
            logger.info("Skipping silent stem %s", stem.name)
            continue

        if stem.is_percussive:
            if config.transcribe_drums:
                hits = drums_mod.transcribe_drums(stem.samples, stem.sr)
                all_drum_hits.extend(hits)
            continue

        monophonic = stem.name in config.monophonic_stems
        notes = pitch_mod.transcribe_pitch(
            stem.samples,
            stem.sr,
            backend=config.pitch_backend,
            monophonic=monophonic,
        )
        logger.info("Stem %s -> %d notes", stem.name, len(notes))
        pitched_parts[stem.name] = notes

    score = build_score(
        rhythm,
        pitched_parts,
        drum_hits=all_drum_hits if config.transcribe_drums else None,
        title=config.title,
    )

    written: str | None = None
    if output_path is not None:
        written = write_musicxml(score, str(output_path))

    return TranscriptionResult(
        score=score,
        output_path=written,
        rhythm=rhythm,
        stem_names=stem_names,
    )
