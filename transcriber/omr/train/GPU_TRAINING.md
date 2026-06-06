# Training jazz-font OMR on a rented GPU

End-to-end recipe to fine-tune an OMR model on the handwritten jazz style using
the corpus this repo generates. The goal is to fix what the measurements showed:
off-the-shelf engines are unreliable on MuseJazz (frequent crashes; lost chord
extensions). Training on the jazz style is the fix.

Model choice here: **Donut** (`naver-clova-ix/donut-base`) — a full-page
image→sequence transformer (Swin encoder + BART decoder). It fits our data
directly (a page image → a token string), handles 2D page layout (unlike TrOCR,
which is single-line), and fine-tunes with the HuggingFace `Seq2SeqTrainer`.
Alternative: fine-tune **homr/TrOMR** (already an engine in `engines.py`) if you
want an OMR-native model — less turnkey but purpose-built.

---

## 0. What you need

- A GPU box with **≥ 24 GB VRAM** (RTX 4090 / A10 / L4 is plenty for
  `donut-base`; A100-40GB if you scale the image resolution or batch size).
- The training corpus (image + token-sequence pairs) from this repo.
- ~a few hours of GPU time for a first useful model.

---

## 1. Generate the corpus (locally — you have MuseScore)

Generate on your Mac (MuseScore + the OMR extras already set up), then upload to
the GPU box. Mix fonts and augmentation so the model generalises instead of
overfitting one face.

```bash
cd transcriber
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib   # for verovio rasterise

# 1) Real engraving content (classical), rendered handwritten + scan-degraded
omr-build-corpus --dataset openscore-lieder --limit 1000 \
    --renderer musescore --style MuseJazz --augment photo -o data/lieder_jazz

# 2) Jazz CHORD-SYMBOL coverage (synthetic lead sheets: melody + chords)
omr-build-corpus --dataset leadsheet --limit 4000 --seed 1 \
    --renderer musescore --style MuseJazz --augment photo -o data/leadsheet_jazz

# 3) Font-invariance: same content in the engraved default (Leland) and clean
omr-build-corpus --dataset leadsheet --limit 2000 --seed 2 \
    --renderer musescore --style "" -o data/leadsheet_leland     # --style "" = Leland
omr-build-corpus --dataset leadsheet --limit 2000 --seed 3 \
    --renderer verovio --font Petaluma -o data/leadsheet_petaluma # another handwritten face
```

Each `data/*/` has `manifest.jsonl` (`{id, image, musicxml, tokens}`) + an
`images/` dir. Concatenate the manifests (fix the relative `image` paths or run
training with each dir as a root) and **hold out a split** for validation. Most
important: also assemble a small **hand-labelled set of real scanned fakebook
pages** — synthetic accuracy ≠ real accuracy; that is your true test set.

Tar it up and upload:

```bash
tar czf jazz_corpus.tgz data/
# scp / rsync jazz_corpus.tgz to the GPU box
```

> Generating on the GPU box instead? Install MuseScore + run it headless under
> `xvfb-run` (`apt-get install musescore xvfb` then `xvfb-run -a omr-build-corpus ...`).
> Rendering locally and uploading is simpler.

---

## 2. Rent + set up the GPU box

RunPod / Lambda / Vast.ai all work; pick a "PyTorch 2.x + CUDA 12" template.

```bash
tar xzf jazz_corpus.tgz
python -m venv .venv && . .venv/bin/activate
pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40" datasets accelerate pillow sentencepiece evaluate
pip install music21   # for token<->score reconstruction at eval time
```

---

## 3. Fine-tune

`train_donut.py` (in this directory) is a runnable scaffold. It reads one or
more `manifest.jsonl` files, treats `" ".join(tokens)` as the target string,
adds the music tokens as special tokens (so they are one-token-each, not
sub-worded), and fine-tunes Donut.

```bash
accelerate launch transcriber/omr/train/train_donut.py \
    --manifests data/leadsheet_jazz/manifest.jsonl data/lieder_jazz/manifest.jsonl \
                data/leadsheet_leland/manifest.jsonl data/leadsheet_petaluma/manifest.jsonl \
    --val-frac 0.05 \
    --image-size 1280 960 \
    --epochs 8 --batch-size 2 --grad-accum 8 --lr 5e-5 --fp16 \
    --out runs/donut-jazz
```

Notes:
- `--image-size H W`: Donut downsamples a lot; jazz lines are wide, so a tall
  enough height matters. Start ~1280×960; raise if small glyphs (the chord
  superscripts!) are lost, lower if you OOM.
- Effective batch = `batch-size × grad-accum` (here 16). Scale to your VRAM.
- Freeze the encoder for the first epoch (`--freeze-encoder-epochs 1`) for a
  more stable start, then let it adapt to the music glyphs.

---

## 4. Evaluate with the repo's metrics

The win condition is on **real** pages, scored symbolically, not training loss.

```bash
python transcriber/omr/train/eval_donut.py \
    --model runs/donut-jazz --manifest data/heldout_real/manifest.jsonl \
    --image-size 1280 960
```

`eval_donut.py` (scaffold here) generates a token string per image, parses it
back to a `music21` score, and scores against the ground truth with this repo's
`compare_scores` (notes) and `compare_chords` (chord symbols). Track note-F1 and
chord-F1 separately. Iterate: error-analyse the worst cases, synthesise more of
them (more of that font/aug/chord type), retrain.

---

## 5. Use the trained model

Wrap it as a new OMR engine alongside oemer/homr: a class in `engines.py` whose
`recognize()` runs Donut over each page image, parses the emitted tokens back to
a `music21` score (reuse the eval reconstruction), and returns it. Then it slots
into the existing pipeline, harness, and confidence/review queue for free.

---

## Cost & tips

- A first useful model: ~a few hours on one 24 GB GPU (≈ a few dollars).
- **Mix fonts + augmentation** — the single biggest lever for real-world
  generalisation. Jazz-heavy, but include engraved + clean so the model does not
  forget how to read print.
- **Chord superscripts** are the hardest target (the OCR pass loses them). Donut
  trained end-to-end on the rendered superscripts should do better than tesseract,
  but render at high enough resolution that they survive downsampling.
- **Always validate on real scans**, not just synthetic renders — the synthetic-
  to-real gap is the whole reason for the augmentation stage.
- Checkpoint every epoch and keep the best by **chord-F1 + note-F1**, not loss.
