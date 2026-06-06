"""End-to-end pipeline tests using the librosa fallback back-ends."""

import xml.dom.minidom as minidom

from transcriber import TranscriptionConfig
from transcriber.audio_io import write_wav
from transcriber.pipeline import transcribe
from transcriber.stems import separate


def test_separation_fallback_returns_drums_and_pitched(mix_audio, sample_rate):
    stems = separate(mix_audio, sample_rate, backend="hpss")
    names = {s.name for s in stems}
    assert "drums" in names
    assert any(not s.is_percussive for s in stems)
    assert any(s.is_percussive for s in stems)


def test_full_pipeline_writes_valid_musicxml(mix_audio, sample_rate, tmp_path):
    audio_path = tmp_path / "mix.wav"
    write_wav(audio_path, mix_audio, sample_rate)

    out = tmp_path / "out.musicxml"
    config = TranscriptionConfig(
        separation_backend="hpss",
        pitch_backend="pyin",
        sample_rate=sample_rate,
    )
    result = transcribe(audio_path, out, config=config)

    assert result.output_path is not None
    assert out.exists()
    minidom.parse(str(out))  # well-formed XML
    assert result.rhythm.tempo > 0
    assert "drums" in result.stem_names


def test_pipeline_without_output_path_returns_score(mix_audio, sample_rate, tmp_path):
    audio_path = tmp_path / "mix.wav"
    write_wav(audio_path, mix_audio, sample_rate)

    config = TranscriptionConfig(separation_backend="hpss", pitch_backend="pyin")
    result = transcribe(audio_path, None, config=config)
    assert result.output_path is None
    assert result.score is not None
