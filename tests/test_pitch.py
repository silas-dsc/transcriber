"""Tests for pitch transcription (pYIN fallback) and monophonic collapsing."""

import librosa

from transcriber.pitch import Note, _make_monophonic, transcribe_pitch


def test_pyin_recovers_scale_pitches(melody_audio, sample_rate):
    notes = transcribe_pitch(melody_audio, sample_rate, backend="pyin")
    assert len(notes) >= 6  # most of the 8 scale notes
    pitches = [n.pitch for n in notes]
    # C major scale starts at C4 (MIDI 60); the first detected note should be
    # close to C4.
    expected_c4 = int(round(librosa.hz_to_midi(261.63)))
    assert abs(pitches[0] - expected_c4) <= 1
    # Pitches should be generally ascending across the scale.
    assert pitches[-1] > pitches[0]


def test_notes_have_positive_duration(melody_audio, sample_rate):
    notes = transcribe_pitch(melody_audio, sample_rate, backend="pyin")
    assert all(n.duration > 0 for n in notes)


def test_make_monophonic_resolves_overlaps():
    notes = [
        Note(start=0.0, end=1.0, pitch=60),
        Note(start=0.5, end=1.5, pitch=67),  # overlaps, higher -> wins
    ]
    mono = _make_monophonic(notes)
    # No two notes should overlap in time.
    for a, b in zip(mono, mono[1:]):
        assert a.end <= b.start + 1e-9
    assert any(n.pitch == 67 for n in mono)


def test_make_monophonic_empty():
    assert _make_monophonic([]) == []
