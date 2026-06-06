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

A small browser app is included. It accepts **either an audio recording or a
picture of sheet music (PDF / image)** and returns MusicXML — the UI shows the
relevant options for whichever you upload, and the backend auto-routes audio to
the transcription pipeline and PDFs/images to the OMR pipeline.

```bash
pip install -e ".[web]"     # adds fastapi + uvicorn

transcriber-web             # auto-selects a free port and prints the URL
transcriber-web --port 8000  # or pick your own port
```

Open the printed `http://127.0.0.1:<port>` URL, choose a file, and the score
downloads when it's ready. The HTTP API is one endpoint — `POST /convert` with
a `file` field (audio or PDF/image) returns the MusicXML; CORS is enabled so a
separately-hosted static frontend can call it.

**Hosting it for free:** `frontend/index.html` is a standalone static page for
**GitHub Pages**, and the `Dockerfile` runs the backend on **Hugging Face
Spaces** / Render / Fly / Cloud Run. Step-by-step instructions (including
pushing to your own GitHub account) are in **[HOSTING.md](HOSTING.md)**.

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
| Recognition (built-in fallback) | classical-CV recogniser: staff detection → staff-line removal → **key-signature + inline-accidental detection** → dual filled/hollow note-head detection → key/accidental-aware pitch mapping |
| Ensembling | run several engines, keep the highest-confidence result |
| Post-processing | merge multi-page output, repair & normalise via [music21](https://web.mit.edu/music21/) |
| **Semantic checks** | musical sanity rules catch & repair likely errors (see below) |
| **LLM sense-check** (optional) | Claude reviews the symbolic result and proposes conservative corrections |

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

### Accuracy: semantic checks and LLM sense-checking

OMR errors are often invisible at the pixel level but obvious *musically*. Two
post-processing layers exploit that:

- **Semantic sanity checks** (`semantic.py`, on by default, deterministic and
  offline): infer the key; flag measures whose beats don't add up; detect notes
  an octave from their neighbours (a misread ledger line); flag pitches outside
  the clef's range; and merge exact duplicate detections. *Safe* repairs
  (duplicate merge, key-consistent re-spelling) are applied automatically;
  *risky* ones (octave correction) are applied only with `--aggressive`,
  otherwise just reported for review. On the clean-monophonic benchmark, the
  duplicate-merge step alone lifts the built-in recogniser from F1 ≈ 0.99 to
  **F1 ≈ 1.0**.
- **LLM sense-checking** (`llm_review.py`, optional, `--llm-review`): Claude
  reviews the *symbolic* note sequence (not the image — current models are not
  reliable at reading pitch from pixels) and proposes a closed set of
  conservative edits to notes that already exist (octave shift, duration fix,
  duplicate deletion). Every suggestion is validated against the score before
  being applied, so a bad suggestion is dropped, never trusted — the model can
  flag and fix, but never invent notes. Requires `pip install -e ".[llm]"` and
  `ANTHROPIC_API_KEY`.

### Confidence and human-in-the-loop review

No OMR system is perfect on real scans, so the honest route to a *correct*
score is to make the system **know where it is unsure** and surface only those
spots for a human to confirm. The pipeline fuses three independent confidence
signals (`confidence.py`) into a per-measure score and a ranked **review
queue** (`OMRResult.confidence_report`):

1. **Per-note recogniser confidence** — e.g. the built-in recogniser's
   head-position ambiguity (a head straddling two staff steps).
2. **Semantic checks** — the musical impossibilities from `semantic.py`.
3. **Multi-engine disagreement** — where oemer / homr / the built-in recogniser
   disagree on a note is exactly where to look. (Catches systematic errors the
   geometric signal misses, e.g. a skew-induced pitch shift.)

```bash
omr scan.pdf -o out.musicxml --mark-review   # annotate uncertain measures
```

`--mark-review` writes a MusicXML with a red `review?` mark on each flagged
measure and its low-confidence notes coloured, so it opens in any editor with
the spots to check already highlighted. On clean music the queue is empty; the
worse the input, the more it surfaces — which is how a human reaches a truly
correct score with minimal effort.

### Install

```bash
pip install -e .            # built-in recogniser + semantic checks, out of the box
pip install -e ".[omr]"     # + oemer (deep-learning OMR)
pip install -e ".[omr-eval]" # + verovio/cairosvg for the accuracy harness
pip install -e ".[llm]"     # + anthropic SDK for LLM sense-checking
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

Benchmark results for the built-in recogniser (offline, built-in renderer):

Note-level F1 on the offline render→recognise loop (built-in recogniser):

| Corpus | Pieces | F1 |
| --- | --- | --- |
| Synthetic monophonic | 12 | **1.00** |
| Bach chorales (top line) | 50 | **1.00** |
| Palestrina (top line) | 30 | **1.00** |
| Monteverdi madrigals (top line) | 48 | **1.00** |
| ryansMammoth fiddle tunes | 40 | **1.00** |
| O'Neill's 1850 Irish tunes | 40 | **1.00** |
| Essen folk songs | 80 | **1.00** |

The road to these numbers shows the methodology: measure errors, categorise
them, fix the systematic cause, re-measure. Reading the **key signature** off
the image lifted Bach from F1 0.79 → 0.97; **inline-accidental** detection (the
printed sharp/flat/natural glyphs on individual notes, including on high/low
ledger-line notes and distinguished from a key signature on the first note) and
a **simultaneous-note ordering** fix took every corpus to **note-perfect** pitch
recognition. These are a floor the deep-learning engines exceed on real-world
*scans* (which are harder — see the robustness note below). For real
PDF/MusicXML corpora the harness supports [OpenScore Lieder](https://github.com/OpenScore/Lieder),
[PDMX](https://github.com/pnlong/PDMX), and the
[OMR-Datasets](https://apacha.github.io/OMR-Datasets/) collection (see
`transcriber/omr/eval/datasets.py`).

### Limitations

- The built-in recogniser targets clean printed monophonic / simple polyphonic
  music; install `.[omr]` (oemer) for handwritten, dense, or photographed
  scores.
- The built-in recogniser reads key signatures and inline accidentals
  (sharp/flat/natural), but not double-accidentals, mid-staff clef changes, or
  multi-voice chords; a hollow note head sitting exactly on a staff line may be
  read as a filled (quarter) note; and beamed eighth-note durations are
  approximated. The deep-learning engines handle these. *(The accuracy numbers
  above are for the offline render→recognise loop; real-world camera scans are
  harder — see "the road to 100%" below.)*

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
  web.py          # FastAPI upload UI + API (audio OR pdf/image -> MusicXML)
  omr/            # optical music recognition (image/PDF -> MusicXML)
    rendering.py    # PDF/image -> page bitmaps (pypdfium2)
    preprocess.py   # binarise / deskew / denoise
    primitive.py    # built-in classical-CV recogniser (fallback)
    engines.py      # oemer / homr / Audiveris wrappers + fallback selection
    ensemble.py     # multi-engine selection
    postprocess.py  # page merge, repair, confidence
    semantic.py     # musical sanity checks & safe repairs
    llm_review.py   # optional Claude sense-check of the recognised notes
    assemble.py     # recognised notes -> music21 score
    pipeline.py     # orchestration
    cli.py          # `omr` command
    eval/           # accuracy harness: metrics, renderers, datasets
    confidence.py   # confidence scoring + human-in-the-loop review queue
frontend/         # static web UI for GitHub Pages (calls the backend API)
Dockerfile        # backend container (Hugging Face Spaces / Render / Fly / ...)
HOSTING.md        # free deployment guide (Pages frontend + Spaces backend)
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
