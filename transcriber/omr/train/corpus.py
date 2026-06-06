"""Build a jazz-font OMR training corpus from MusicXML.

For each source score this renders a page image in a handwritten font
(MuseJazz via MuseScore, or Petaluma via verovio), optionally degrades it like
a scan/photo, and writes:

* ``images/<id>.png``       -- the rendered (optionally degraded) page
* ``images/<id>.musicxml``  -- the ground-truth score (universal label)
* a line in ``manifest.jsonl`` with the id, paths and a linearised token
  sequence (a simple seq2seq target; swap in kern / the PrIMuS "agnostic"
  encoding for a production model)

The token sequence captures the jazz payload -- melody notes/rests **and**
chord symbols -- so a model can be trained to read both.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ..eval.augment import augment_preset
from ..eval.render_ref import render_reference

logger = logging.getLogger(__name__)


def score_to_tokens(score) -> list[str]:
    """Linearise a score to OMR target tokens (notes, rests, chord symbols).

    Order is by absolute offset.  Tokens: ``key_<sharps>``, ``time_<a/b>``,
    ``note_<NameOctave>_<ql>``, ``rest_<ql>``, ``chord_<figure>``.  Intended as
    a simple, swappable seq2seq target -- not a canonical OMR encoding.
    """
    from music21 import harmony, note

    toks: list[str] = []
    ks = score.recurse().getElementsByClass("KeySignature").first()
    if ks is not None and ks.sharps is not None:
        toks.append(f"key_{int(ks.sharps)}")
    ts = score.recurse().getElementsByClass("TimeSignature").first()
    if ts is not None:
        toks.append(f"time_{ts.ratioString}")

    elems = list(score.recurse().getElementsByClass((note.Note, note.Rest, harmony.ChordSymbol)))
    elems.sort(key=lambda e: e.getOffsetInHierarchy(score))
    for el in elems:
        if isinstance(el, harmony.ChordSymbol):
            toks.append(f"chord_{el.figure}")
        elif isinstance(el, note.Rest):
            toks.append(f"rest_{float(el.quarterLength)}")
        else:  # note.Note
            toks.append(f"note_{el.pitch.nameWithOctave}_{float(el.quarterLength)}")
    return toks


def build_corpus(
    items,
    out_dir: str | Path,
    renderer: str = "musescore",
    style: str | None = "MuseJazz",
    font: str | None = None,
    augment: str | None = None,
    dpi: int = 300,
) -> Path:
    """Render ``items`` (``CorpusItem``\\ s) into an OMR training corpus.

    Args:
        items: Source scores (e.g. from :mod:`transcriber.omr.eval.datasets`).
        out_dir: Destination directory (created).  Gets ``images/`` + a
            ``manifest.jsonl``.
        renderer / style / font / dpi: Passed to
            :func:`transcriber.omr.eval.render_ref.render_reference`.  Defaults
            render the handwritten **MuseJazz** font via MuseScore.
        augment: Optional degradation preset (e.g. ``"photo"``) applied to each
            page to emulate a scan; ``None`` keeps the clean render.

    Returns:
        Path to the written ``manifest.jsonl``.
    """
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    n_ok = 0
    with open(manifest, "w") as mf:
        for item in items:
            img_path = img_dir / f"{item.id}.png"
            try:
                render_reference(
                    item.score, img_path, renderer=renderer, style=style, font=font, dpi=dpi
                )
                if augment:
                    arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
                    arr = augment_preset(arr, augment)
                    Image.fromarray((arr * 255).clip(0, 255).astype("uint8")).save(img_path)
                mxl_path = img_dir / f"{item.id}.musicxml"
                item.score.write("musicxml", fp=str(mxl_path))
                record = {
                    "id": item.id,
                    "image": str(img_path.relative_to(out)),
                    "musicxml": str(mxl_path.relative_to(out)),
                    "tokens": score_to_tokens(item.score),
                }
            except Exception as exc:  # one bad score should not kill the corpus
                logger.warning("skipping %s: %s", item.id, exc)
                continue
            mf.write(json.dumps(record) + "\n")
            n_ok += 1
    logger.info("wrote %d training pairs to %s", n_ok, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..eval.datasets import (
        music21_corpus,
        openscore_lieder_corpus,
        synthetic_corpus,
        synthetic_lead_sheet_corpus,
    )

    p = argparse.ArgumentParser(
        prog="omr-build-corpus",
        description="Render a jazz-font OMR training corpus from MusicXML.",
    )
    p.add_argument(
        "--dataset",
        choices=["synthetic", "leadsheet", "music21", "openscore-lieder"],
        default="leadsheet",
        help="Source scores. 'openscore-lieder' fetches real CC0 scores from "
        "the OpenScore Lieder GitHub mirror (needs network + MuseScore CLI).",
    )
    p.add_argument("--query", default="bach", help="Composer/query for the music21 corpus.")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default=None, help="Download cache for openscore-lieder.")
    p.add_argument("--renderer", default="musescore", choices=["musescore", "verovio", "builtin"])
    p.add_argument("--style", default="MuseJazz", help="MuseScore style (e.g. MuseJazz).")
    p.add_argument("--font", default=None, help="verovio SMuFL font (e.g. Petaluma).")
    p.add_argument("--augment", default=None, help="degradation preset, e.g. photo.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("-o", "--out", default="omr_corpus")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dataset == "synthetic":
        items = synthetic_corpus(n_items=args.limit, seed=args.seed)
    elif args.dataset == "leadsheet":
        items = synthetic_lead_sheet_corpus(n_items=args.limit, seed=args.seed)
    elif args.dataset == "openscore-lieder":
        items = openscore_lieder_corpus(limit=args.limit, cache_dir=args.cache_dir)
    else:
        items = music21_corpus(query=args.query, limit=args.limit)

    manifest = build_corpus(
        items, args.out, renderer=args.renderer, style=args.style,
        font=args.font, augment=args.augment, dpi=args.dpi,
    )
    print(f"wrote corpus manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
