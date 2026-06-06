# Jazz-font OMR training

## Why

Measured on this repo's harness, off-the-shelf engines handle engraved music
well (oemer ≈ 0.91 F1, reliable) but are **unstable on the handwritten MuseJazz
style** — oemer crashes during staffline/symbol extraction on a large,
run-dependent fraction of pages, and reads the rest at ≈ 0.4–0.9. Tesseract
likewise reads chord roots/qualities but loses superscript extensions. These
are limits of *pretrained* models on a style they never saw. The fix is to
train on the jazz style.

## What this provides

`corpus.py` generates the training data — (image, ground-truth) pairs — from
any MusicXML, rendered in the jazz font with optional scan augmentation:

```bash
# Handwritten MuseJazz lead sheets (melody + chord symbols), scan-degraded:
python -m transcriber.omr.train.corpus \
    --dataset leadsheet --limit 500 --augment photo -o data/jazz

# Or render real classical corpora in MuseJazz:
python -m transcriber.omr.train.corpus \
    --dataset music21 --query bach --limit 200 -o data/bach_jazz
```

Output:

```
data/jazz/
  manifest.jsonl            # {id, image, musicxml, tokens} per line
  images/<id>.png           # rendered (optionally degraded) page
  images/<id>.musicxml      # ground-truth score (universal label)
```

Scale it up by feeding larger MusicXML corpora (OpenScore Lead Sheets, PDMX,
Wikifonia) as `CorpusItem`s into `build_corpus`. Mix fonts (MuseJazz + Petaluma
+ Leland) and augmentation presets so the model generalises rather than
overfitting one face.

## Recommended training path: image → token seq2seq

The pairs above (`image` + `tokens`) suit a **sequence-to-sequence OMR model**
(encoder over the page image, decoder emitting the token stream) — e.g. TrOMR /
Polyphonic-TrOMR, the Sheet-Music-Transformer, or a TrOCR-style fine-tune. This
needs no pixel masks: the label is the token sequence we already emit. Swap the
placeholder tokeniser in `score_to_tokens` for **kern** or the PrIMuS
**agnostic** encoding for a production target.

Fine-tuning recipe (on a GPU box):
1. Generate a mixed-font corpus (jazz-heavy) with augmentation, here.
2. Start from a checkpoint pretrained on engraved music; fine-tune on the jazz
   corpus (optionally freeze the encoder first, then unfreeze).
3. Evaluate with this repo's metrics — `compare_scores` (notes) and
   `compare_chords` (chord symbols) — on a **held-out, hand-labelled** set of
   real scanned fakebook pages, not just synthetic renders.
4. Loop: error-analyse, synthesise more of the failure cases, repeat.

## Why not just fine-tune oemer here

oemer's recogniser is two **TensorFlow/Keras** segmentation U-Nets (shipped as
ONNX). Fine-tuning them needs **pixel-level masks** (stafflines, noteheads,
symbol classes) per image — a separate synthesis pipeline — plus a TF training
run on a **GPU**. This environment is deliberately ONNX-only (Python 3.13 +
numpy 2.x, no GPU; full TensorFlow doesn't install cleanly here), so model
training is out of scope for the repo. The seq2seq route above avoids the mask
problem and trains directly on what `corpus.py` emits.

A cheaper *interim* stabiliser (no training): normalise the MuseJazz render
before oemer (aggressive binarise / staffline thickening / rescale) to reduce
the extraction crashes — worth trying if you need usable output before a model
is trained.
