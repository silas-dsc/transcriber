"""Assemble analysed parts into a quantised music21 score and export MusicXML.

The :func:`build_score` function takes the rhythm analysis plus the pitched
note events and drum hits for each stem, converts onset/offset *seconds* to
*beat positions* (via :class:`~transcriber.rhythm.RhythmInfo`), quantises to a
musical grid and lays everything out into measures with a tempo and time
signature.
"""

from __future__ import annotations

import logging

from music21 import (
    clef,
    instrument,
    meter,
    note,
    stream,
    tempo,
)

from .drums import HIHAT, KICK, SNARE, DrumHit
from .pitch import Note
from .rhythm import RhythmInfo

logger = logging.getLogger(__name__)

# Quantise to sixteenth notes (divisor 4) and triplets (divisor 3).
QUANTIZE_DIVISORS = (4, 3)

# General MIDI program numbers / music21 instruments for pitched stems.
STEM_INSTRUMENTS = {
    "bass": instrument.ElectricBass,
    "vocals": instrument.Vocalist,
    "other": instrument.Piano,
    "guitar": instrument.AcousticGuitar,
    "piano": instrument.Piano,
}

# Where each drum voice sits on the percussion staff (display pitch) following
# common percussion-notation convention.
DRUM_DISPLAY = {KICK: "F4", SNARE: "C5", HIHAT: "G5"}


def build_score(
    rhythm: RhythmInfo,
    pitched_parts: dict[str, list[Note]],
    drum_hits: list[DrumHit] | None = None,
    title: str = "Transcription",
) -> stream.Score:
    """Build a :class:`music21.stream.Score` from analysed parts.

    Args:
        rhythm: Tempo / beat-grid information used for quantisation.
        pitched_parts: Mapping of stem name -> note events.
        drum_hits: Optional list of drum strokes for a percussion part.
        title: Score title (work title metadata).

    Returns:
        A fully notated score ready to be written to MusicXML.
    """
    score = stream.Score()
    score.insert(0, _metadata(title))

    ts = meter.TimeSignature(f"{rhythm.beats_per_measure}/{rhythm.beat_unit}")
    mm = tempo.MetronomeMark(number=round(rhythm.tempo))

    added_any = False
    for stem_name, notes in pitched_parts.items():
        if not notes:
            continue
        part = _build_pitched_part(stem_name, notes, rhythm, ts, mm)
        if part is not None:
            score.insert(0, part)
            added_any = True

    if drum_hits:
        drum_part = _build_drum_part(drum_hits, rhythm, ts, mm)
        if drum_part is not None:
            score.insert(0, drum_part)
            added_any = True

    if not added_any:
        # Always return a structurally valid (if empty) score.
        empty = stream.Part()
        empty.insert(0, mm)
        empty.insert(0, ts)
        empty.append(note.Rest(quarterLength=float(rhythm.beats_per_measure)))
        score.insert(0, empty)

    score.makeNotation(inPlace=True)
    return score


def _metadata(title: str):
    from music21 import metadata

    md = metadata.Metadata()
    md.title = title
    md.composer = "Transcribed by transcriber"
    return md


def _build_pitched_part(
    stem_name: str,
    notes: list[Note],
    rhythm: RhythmInfo,
    ts: meter.TimeSignature,
    mm: tempo.MetronomeMark,
) -> stream.Part | None:
    part = stream.Part()
    part.partName = stem_name.capitalize()
    instr_cls = STEM_INSTRUMENTS.get(stem_name, instrument.Piano)
    part.insert(0, instr_cls())
    part.insert(0, mm)
    part.insert(0, ts)

    placed = 0
    for n in notes:
        offset_ql, length_ql = _beats_to_ql(n.start, n.end, rhythm)
        if length_ql <= 0:
            continue
        m21_note = note.Note(int(n.pitch))
        m21_note.quarterLength = length_ql
        m21_note.volume.velocity = int(n.velocity)
        part.insert(offset_ql, m21_note)
        placed += 1

    if placed == 0:
        return None

    _quantize(part)
    return part


def _build_drum_part(
    drum_hits: list[DrumHit],
    rhythm: RhythmInfo,
    ts: meter.TimeSignature,
    mm: tempo.MetronomeMark,
) -> stream.Part | None:
    part = stream.Part()
    part.partName = "Drums"
    part.insert(0, instrument.UnpitchedPercussion())
    part.insert(0, clef.PercussionClef())
    part.insert(0, mm)
    part.insert(0, ts)

    placed = 0
    for hit in drum_hits:
        offset_ql, _ = _beats_to_ql(hit.time, hit.time, rhythm)
        display = DRUM_DISPLAY.get(hit.kind, "C5")
        u = note.Unpitched(displayName=display)
        u.quarterLength = 0.5  # notate strokes as eighth notes by default
        u.volume.velocity = int(hit.velocity)
        u.stemDirection = "up" if hit.kind == HIHAT else "down"
        part.insert(offset_ql, u)
        placed += 1

    if placed == 0:
        return None

    _quantize(part)
    return part


def _beats_to_ql(start_s: float, end_s: float, rhythm: RhythmInfo) -> tuple[float, float]:
    """Convert a (start, end) pair in seconds to (offset_ql, length_ql).

    One beat is one quarter-note (quarterLength 1.0); the beat grid handles
    tempo, so beat position maps directly to quarterLength offset.
    """
    start_beat = float(rhythm.time_to_beats(start_s))
    end_beat = float(rhythm.time_to_beats(end_s))
    offset_ql = max(0.0, start_beat)
    length_ql = max(0.0, end_beat - start_beat)
    return offset_ql, length_ql


def _quantize(part: stream.Part) -> None:
    """Snap offsets and durations to the quantisation grid in place."""
    part.quantize(
        quarterLengthDivisors=QUANTIZE_DIVISORS,
        processOffsets=True,
        processDurations=True,
        inPlace=True,
    )


def write_musicxml(score: stream.Score, path: str) -> str:
    """Write ``score`` to ``path`` as MusicXML and return the path."""
    out = score.write("musicxml", fp=path)
    logger.info("Wrote MusicXML to %s", out)
    return str(out)
