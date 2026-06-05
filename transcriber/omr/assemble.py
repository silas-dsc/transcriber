"""Assemble a :class:`RecognizedScore` into a music21 score + MusicXML.

OMR note events already carry positions and durations in *quarter lengths*, so
(unlike the audio pipeline) there is no seconds-to-beats conversion -- we lay
notes straight onto a part timeline, one part per detected staff, and let
music21 bar the music up.

The MusicXML writer is shared with the audio pipeline
(:func:`transcriber.score.write_musicxml`) so both halves of the project emit
identical, MuseScore-clean output.
"""

from __future__ import annotations

import logging

from music21 import clef, instrument, meter, note, stream

from .types import OMRNote, RecognizedScore

logger = logging.getLogger(__name__)

_CLEF_OBJECTS = {
    "treble": clef.TrebleClef,
    "bass": clef.BassClef,
    "alto": clef.AltoClef,
}


def build_score(
    recognized: RecognizedScore,
    title: str | None = None,
    time_signature: str = "4/4",
    clefs: dict[int, str] | None = None,
) -> stream.Score:
    """Build a :class:`music21.stream.Score` from recognised notes.

    Args:
        recognized: The structured recognition result.
        title: Score title (defaults to ``recognized.title``).
        time_signature: Time signature to bar the music into.
        clefs: Optional per-staff clef names (1-based staff index).

    Returns:
        A fully notated, structurally valid score.
    """
    clefs = clefs or {}
    score = stream.Score()
    score.insert(0, _metadata(title or recognized.title))

    by_staff: dict[int, list[OMRNote]] = {}
    for n in recognized.notes:
        by_staff.setdefault(n.staff, []).append(n)

    if not by_staff:
        score.insert(0, _empty_part(time_signature))
        score.makeNotation(inPlace=True)
        _fix_voice_numbers(score)
        return score

    for staff_idx in sorted(by_staff):
        part = _build_part(
            by_staff[staff_idx],
            time_signature=time_signature,
            clef_name=clefs.get(staff_idx, "treble" if staff_idx == 1 else "bass"),
        )
        score.insert(0, part)

    score.makeNotation(inPlace=True)
    _fix_voice_numbers(score)
    return score


def _build_part(notes: list[OMRNote], time_signature: str, clef_name: str) -> stream.Part:
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, _CLEF_OBJECTS.get(clef_name, clef.TrebleClef)())
    part.insert(0, meter.TimeSignature(time_signature))

    placed = 0
    for n in sorted(notes, key=lambda x: (x.onset, -x.pitch)):
        if n.duration <= 0:
            continue
        m21 = note.Note(int(n.pitch))
        m21.quarterLength = float(n.duration)
        if n.accidental:
            m21.pitch.accidental = n.accidental
        part.insert(float(n.onset), m21)
        placed += 1

    if placed == 0:
        beats = int(time_signature.split("/")[0])
        part.append(note.Rest(quarterLength=float(beats)))
    return part


def _empty_part(time_signature: str) -> stream.Part:
    part = stream.Part()
    part.insert(0, meter.TimeSignature(time_signature))
    beats = int(time_signature.split("/")[0])
    part.append(note.Rest(quarterLength=float(beats)))
    return part


def _metadata(title: str):
    from music21 import metadata

    md = metadata.Metadata()
    md.title = title
    md.composer = "Recognised by transcriber-omr"
    return md


def _fix_voice_numbers(score: stream.Score) -> None:
    """Renumber voices so every ``<voice>`` is a positive integer.

    music21's ``makeNotation`` can emit 0-based voice ids, which MuseScore 4
    rejects as a corrupt file.  Shared rationale with the audio pipeline's
    :func:`transcriber.score._fix_voice_numbers`.
    """
    for part in score.parts:
        for measure in part.getElementsByClass(stream.Measure):
            for i, voice in enumerate(measure.getElementsByClass(stream.Voice), start=1):
                voice.id = str(i)
