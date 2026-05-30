"""Tests for score assembly and MusicXML export."""

import xml.dom.minidom as minidom

import numpy as np
from music21 import converter, stream

from transcriber.drums import DrumHit
from transcriber.pitch import Note
from transcriber.rhythm import RhythmInfo
from transcriber.score import build_score, write_musicxml


def _rhythm() -> RhythmInfo:
    # A clean 120 BPM grid spanning two 4/4 measures.
    beat_times = np.arange(0, 8) * 0.5
    return RhythmInfo(tempo=120.0, beat_times=beat_times, beats_per_measure=4, beat_unit=4)


def test_build_score_has_expected_parts():
    rhythm = _rhythm()
    notes = [Note(start=i * 0.5, end=i * 0.5 + 0.45, pitch=60 + i) for i in range(4)]
    hits = [DrumHit(time=i * 0.5, kind="kick") for i in range(4)]
    score = build_score(rhythm, {"bass": notes}, drum_hits=hits)
    part_names = [p.partName for p in score.parts]
    assert "Bass" in part_names
    assert "Drums" in part_names


def test_notes_quantize_onto_the_beat():
    rhythm = _rhythm()
    # Notes a little off the grid should snap to integer beat offsets.
    notes = [
        Note(start=0.02, end=0.47, pitch=60),
        Note(start=0.53, end=0.98, pitch=62),
    ]
    score = build_score(rhythm, {"other": notes})
    part = score.parts[0]
    offsets = sorted(n.offset for n in part.flatten().notes)
    for off in offsets:
        # Quantised to a 16th-note grid (multiples of 0.25 ql).
        assert abs((off / 0.25) - round(off / 0.25)) < 1e-6


def test_empty_input_still_produces_valid_score():
    score = build_score(_rhythm(), {})
    assert isinstance(score, stream.Score)
    assert len(score.parts) >= 1


def test_overlapping_notes_have_valid_voice_numbers(tmp_path):
    """Overlapping notes must not export <voice>0</voice> (MuseScore rejects it)."""
    import re

    rhythm = _rhythm()
    # Heavily overlapping notes force music21 to create multiple voices.
    notes = [
        Note(start=0.0, end=2.0, pitch=60),
        Note(start=0.5, end=2.0, pitch=64),
        Note(start=1.0, end=2.0, pitch=67),
        Note(start=1.5, end=2.0, pitch=72),
    ]
    score = build_score(rhythm, {"other": notes})
    out = tmp_path / "voices.musicxml"
    write_musicxml(score, str(out))
    text = out.read_text()

    voices = {int(v) for v in re.findall(r"<voice>(\d+)</voice>", text)}
    assert voices, "expected explicit voices for overlapping notes"
    assert min(voices) >= 1, f"found invalid voice number(s): {sorted(voices)}"
    assert "<voice>0</voice>" not in text


def test_musicxml_is_well_formed_and_reparseable(tmp_path):
    rhythm = _rhythm()
    notes = [Note(start=i * 0.5, end=i * 0.5 + 0.45, pitch=60 + i) for i in range(8)]
    hits = [DrumHit(time=i * 0.5, kind="hihat") for i in range(8)]
    score = build_score(rhythm, {"vocals": notes}, drum_hits=hits)

    out = tmp_path / "score.musicxml"
    path = write_musicxml(score, str(out))

    # Valid XML.
    minidom.parse(path)
    # music21 can round-trip it back in.
    reparsed = converter.parse(path)
    assert len(reparsed.parts) >= 2
    total_notes = sum(len(p.flatten().notes) for p in reparsed.parts)
    assert total_notes > 0
