"""Tests for drum onset detection and classification."""

from transcriber.drums import HIHAT, KICK, transcribe_drums


def test_detects_drum_hits(drum_audio, sample_rate):
    hits = transcribe_drums(drum_audio, sample_rate)
    # 8 kicks + 8 hats expected; detector should find a good fraction.
    assert len(hits) >= 8


def test_classifies_kick_and_hihat(drum_audio, sample_rate):
    hits = transcribe_drums(drum_audio, sample_rate)
    kinds = {h.kind for h in hits}
    # Both a low (kick) and high (hi-hat) voice should be recognised.
    assert KICK in kinds
    assert HIHAT in kinds


def test_hits_sorted_in_time(drum_audio, sample_rate):
    hits = transcribe_drums(drum_audio, sample_rate)
    times = [h.time for h in hits]
    assert times == sorted(times)


def test_velocity_in_range(drum_audio, sample_rate):
    hits = transcribe_drums(drum_audio, sample_rate)
    assert all(1 <= h.velocity <= 127 for h in hits)
