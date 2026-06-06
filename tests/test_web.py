"""Tests for the FastAPI web interface."""

import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from transcriber.audio_io import write_wav  # noqa: E402
from transcriber.web import _find_free_port, app  # noqa: E402

client = TestClient(app)


def test_index_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "transcriber" in resp.text
    assert "<form" in resp.text


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_rejects_unsupported_file_type():
    resp = client.post(
        "/transcribe",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert resp.status_code == 400


def test_transcribe_returns_musicxml(mix_audio, sample_rate, tmp_path):
    wav_path = tmp_path / "mix.wav"
    write_wav(wav_path, mix_audio, sample_rate)
    audio_bytes = wav_path.read_bytes()

    resp = client.post(
        "/transcribe",
        files={"file": ("mix.wav", io.BytesIO(audio_bytes), "audio/wav")},
        data={"separation": "hpss", "pitch": "pyin", "drums": "true"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.content
    assert body.startswith(b"<?xml")
    assert b"score-partwise" in body
    assert resp.headers["content-disposition"].endswith('filename="mix.musicxml"')


def test_find_free_port_returns_valid_port():
    port = _find_free_port()
    assert 1024 <= port <= 65535


def test_convert_accepts_sheet_music_image_and_returns_musicxml():
    # Render a known phrase to a PNG, upload it to /convert (OMR path).
    from transcriber.omr.eval.datasets import make_phrase
    from transcriber.omr.eval.render_ref import render_reference

    png = render_reference(make_phrase([60, 62, 64, 65, 67]), _png_tmp())
    resp = client.post(
        "/convert",
        files={"file": ("scale.png", io.BytesIO(png.read_bytes()), "image/png")},
        data={"engine": "primitive"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"<?xml")
    assert b"score-partwise" in resp.content
    assert resp.headers["content-disposition"].endswith('filename="scale.musicxml"')


def test_convert_rejects_unsupported_type():
    resp = client.post("/convert", files={"file": ("x.txt", b"nope", "text/plain")})
    assert resp.status_code == 400


def _png_tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp()) / "scale.png"
