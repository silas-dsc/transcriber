# transcriber

Turn an audio recording **or a picture of sheet music** into a **MusicXML score**.

`transcriber` has two complementary front-ends that share the same MusicXML
assembly:

- **Audio → MusicXML** (`transcriber`): separate stems, analyse rhythm/pitch,
  assemble a score. Documented immediately below.
- **Sheet music image / PDF → MusicXML** (`omr`): optical music recognition.
  See **[Optical Music Recognition](#optical-music-recognition-imagepdf--musicxml)**.

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
```

### High-quality ML back-ends (Demucs + basic-pitch)

Use the helper script — it works on **all platforms including Apple Silicon
and Python 3.12+**:

```bash
bash scripts/install_ml.sh          # Demucs + basic-pitch (ONNX runtime)
```

> **Why a script?** `basic-pitch`'s packaging pins `tensorflow-macos`, which
> has no wheels on macOS + Python > 3.11, so a plain `pip install ".[full]"`
> fails to resolve there. The script installs basic-pitch with its bundled
> **ONNX** model + `onnxruntime` instead, avoiding TensorFlow entirely.

On Linux/Windows (or macOS with Python ≤ 3.11) the pip extras also work:

```bash
pip install -e ".[full]"            # or ".[separation]" / ".[pitch]"
```

basic-pitch auto-selects whatever runtime is installed (TensorFlow, CoreML,
TFLite or ONNX), so the code needs no changes either way. If no ML back-end is
present, the pipeline transparently falls back to the librosa implementations.

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

## Optical Music Recognition (image/PDF → MusicXML)

The `omr` sub-system is the *optical* counterpart to the audio pipeline: feed it
a scan, a phone photo, or a PDF of sheet music and it produces the same kind of
MusicXML score. It follows the same philosophy — orchestrate the best
open-source OMR engines, and always provide a built-in fallback so a score is
produced even with zero heavy dependencies.

### How it works

| Stage | Approach |
| ----- | -------- |
| Page rendering | [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) (PDF → bitmaps, no system Poppler) |
| Pre-processing | adaptive (Sauvola) binarisation, projection-profile **deskew**, speckle denoise |
| Recognition (primary) | [oemer](https://github.com/BreezeWhite/oemer) deep-learning OMR → MusicXML |
| Recognition (also supported) | [homr](https://github.com/liebharc/homr) (transformer / Polyphonic-TrOMR), [Audiveris](https://github.com/Audiveris/audiveris) (Java, on `PATH`) |
| Recognition (built-in fallback) | classical-CV recogniser: staff detection → staff-line removal → dual filled/hollow note-head detection → clef-aware pitch mapping |
| Ensembling | run several engines, keep the highest-confidence result |
| Post-processing | merge multi-page output, repair & normalise via [music21](https://web.mit.edu/music21/) |

Every external engine is **optional**. With only the core dependencies the
pipeline uses its built-in recogniser, which depends solely on numpy/scipy/Pillow.

### Cross-field techniques

Beyond wrapping existing engines, the pipeline borrows ideas from adjacent
fields to lift accuracy regardless of which engine runs:

- **Document-image analysis**: Sauvola adaptive thresholding (robust to the
  uneven lighting of phone photos) and a gradient-energy deskew that levels
  staves before recognition.
- **Morphological OMR**: a *dual* note-head detector — filled heads via
  morphological opening, hollow heads via enclosed-hole detection on the
  original mask (robust to staff lines bisecting a note head).
- **Model ensembling**: when several engines are installed, run them all and
  select the most confident transcription.

### Install

```bash
pip install -e .            # built-in recogniser works out of the box
pip install -e ".[omr]"     # + oemer (deep-learning OMR)
pip install -e ".[omr-eval]" # + verovio/cairosvg for the accuracy harness
```

### Command line

```bash
# Auto engine (best installed engine, else the built-in recogniser).
omr score.pdf -o score.musicxml

# Force a specific engine, or ensemble all available ones.
omr photo.jpg --engine oemer
omr scan.pdf  --engine ensemble -v

# What can run here?
omr --list-engines
```

### Python API

```python
from transcriber.omr import recognize, OMRConfig

result = recognize("score.pdf", "score.musicxml", OMRConfig(engine="auto"))
print(result.engine, result.page_count, result.confidence)
score = result.score          # a music21 Score you can manipulate further
```

### Measuring & refining accuracy

OMR is only as good as you can measure it. The `transcriber.omr.eval` harness
runs a closed **render → recognise → compare** loop with standard symbolic
metrics (note-level precision/recall/F1, Symbol Error Rate, and an MV2H-style
breakdown). Because it can render references with a built-in engraver, the
benchmark runs fully offline:

```bash
# Benchmark the built-in recogniser on synthetic phrases.
omr-eval --dataset synthetic --limit 12 --engine primitive

# Benchmark against real (offline) music from the music21 corpus.
omr-eval --dataset music21 --query bach --limit 5 --engine auto
```

On clean printed monophonic music the built-in recogniser scores **F1 ≈ 0.99**
(perfect recall, ~0.98 precision) in this loop — a useful floor that the
deep-learning engines exceed on real-world scores. For real PDF/MusicXML
corpora the harness supports [OpenScore Lieder](https://github.com/OpenScore/Lieder),
[PDMX](https://github.com/pnlong/PDMX), and the
[OMR-Datasets](https://apacha.github.io/OMR-Datasets/) collection (see
`transcriber/omr/eval/datasets.py`).

### Limitations

- The built-in recogniser targets clean printed monophonic / simple polyphonic
  music; install `.[omr]` (oemer) for handwritten, dense, or photographed
  scores.
- The built-in recogniser does not yet read accidental glyphs or key
  signatures, and a hollow note head sitting exactly on a staff line may be
  read as a filled (quarter) note. The deep-learning engines handle these.

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
  omr/            # optical music recognition (image/PDF -> MusicXML)
    rendering.py    # PDF/image -> page bitmaps (pypdfium2)
    preprocess.py   # binarise / deskew / denoise
    primitive.py    # built-in classical-CV recogniser (fallback)
    engines.py      # oemer / homr / Audiveris wrappers + fallback selection
    ensemble.py     # multi-engine selection
    postprocess.py  # page merge, repair, confidence
    assemble.py     # recognised notes -> music21 score
    pipeline.py     # orchestration
    cli.py          # `omr` command
    eval/           # accuracy harness: metrics, renderers, datasets
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
