# transcriber

Turn an audio recording into a **MusicXML score**.

Given a recording, `transcriber` runs an end-to-end pipeline:

1. **Stem separation** – split the mix into `drums` / `bass` / `vocals` / `other`.
2. **Rhythm & pitch analysis** – estimate tempo and a beat grid, detect note
   pitches for the melodic/harmonic stems and classify drum hits.
3. **Score assembly** – quantise everything onto the beat grid and export a
   MusicXML score you can open in MuseScore, Finale, Sibelius, Dorico, etc.

It is built by orchestrating best-in-class open-source projects and filling the
gaps between them (rhythm-aware quantisation, drum classification and MusicXML
assembly).

## How it works

| Stage | Primary back-end (open source) | Built-in fallback |
| ----- | ------------------------------ | ----------------- |
| Separation | [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`) | librosa harmonic/percussive separation (HPSS) |
| Pitch | [Spotify basic-pitch](https://github.com/spotify/basic-pitch) (polyphonic) | librosa [pYIN](https://librosa.org/doc/latest/generated/librosa.pyin.html) (monophonic) |
| Rhythm | [librosa](https://librosa.org) beat tracking | – |
| Drums | onset detection + spectral band classification (kick/snare/hi-hat) | – |
| Score | [music21](https://web.mit.edu/music21/) (quantise + MusicXML) | – |

The heavy machine-learning back-ends (Demucs, basic-pitch, which pull in
PyTorch/TensorFlow) are **optional**. With only the core dependencies installed
the pipeline still runs end-to-end using the librosa fallbacks, so you always
get a score — just at lower fidelity.

## Installation

Requires Python ≥ 3.9 and [ffmpeg](https://ffmpeg.org/) (for decoding MP3 and
other compressed formats).

```bash
# Core install — runs end-to-end with the librosa fallbacks.
pip install -e .

# Optional: high-quality ML back-ends (Demucs + basic-pitch + torch).
pip install -e ".[full]"
```

## Usage

### Command line

```bash
# Auto backends (uses Demucs/basic-pitch if installed, else fallbacks).
transcriber song.mp3 -o song.musicxml

# Force the high-quality back-ends.
transcriber song.wav --separation demucs --pitch basic-pitch -o song.musicxml

# Force the dependency-free fallbacks, skip drums, be verbose.
transcriber song.wav --separation hpss --pitch pyin --no-drums -v
```

Run `transcriber --help` for all options.

### Python API

```python
from transcriber import transcribe, TranscriptionConfig

result = transcribe(
    "song.wav",
    "song.musicxml",
    config=TranscriptionConfig(separation_backend="auto", pitch_backend="auto"),
)

print(result.rhythm.tempo, result.stem_names)
score = result.score          # a music21 Score you can manipulate further
```

### Web interface

A small browser app (upload audio → download MusicXML) is included:

```bash
pip install -e ".[web]"     # adds fastapi + uvicorn

transcriber-web             # auto-selects a free port and prints the URL
transcriber-web --port 8000  # or pick your own port
```

Then open the printed `http://127.0.0.1:<port>` URL, choose a file and the
back-ends, and the score downloads when it's ready.

### Try the bundled demo

```bash
transcriber examples/demo.wav -o demo.musicxml -v
```

## Project layout

```
transcriber/
  audio_io.py     # load/save audio
  stems.py        # source separation (Demucs / HPSS fallback)
  rhythm.py       # tempo + beat grid, seconds<->beats mapping
  pitch.py        # pitch transcription (basic-pitch / pYIN fallback)
  drums.py        # drum onset detection + classification
  score.py        # music21 score assembly + MusicXML export
  pipeline.py     # orchestration
  cli.py          # command-line interface
  web.py          # FastAPI browser upload UI
tests/            # pytest suite (runs on the core deps only)
examples/         # bundled synthetic demo recording
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite exercises the full pipeline using the librosa fallback
back-ends, so it runs without the large ML dependencies.

## Limitations

- The fallback separation/pitch back-ends are much cruder than Demucs and
  basic-pitch — install `.[full]` for serious transcriptions.
- Time-signature detection only distinguishes 4/4 and 3/4 and defaults to 4/4.
- Automatic music transcription is inherently approximate; expect to clean up
  the result in a score editor.

## License

MIT
