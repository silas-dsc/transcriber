"""CLI for the OMR accuracy harness.

Usage::

    python -m transcriber.omr.eval --dataset synthetic --engine primitive
    omr-eval --dataset music21 --query bach --limit 5 --engine auto -v
"""

from __future__ import annotations

import argparse
import logging
import sys

from .datasets import music21_corpus, synthetic_corpus
from .harness import evaluate_corpus, format_report
from .render_ref import VEROVIO_FONTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omr-eval",
        description="Benchmark OMR accuracy: render references, recognise, compare.",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "music21"],
        default="synthetic",
        help="Corpus to evaluate against (default: synthetic).",
    )
    parser.add_argument("--query", default="bach", help="Composer/query for the music21 corpus.")
    parser.add_argument("--limit", type=int, default=8, help="Max items to evaluate.")
    parser.add_argument("--notes", type=int, default=10, help="Notes per synthetic phrase.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic RNG seed.")
    parser.add_argument(
        "--engine",
        default="primitive",
        choices=["auto", "oemer", "homr", "audiveris", "primitive", "ensemble"],
        help="Engine under test (default: primitive).",
    )
    parser.add_argument(
        "--renderer",
        default="builtin",
        choices=["builtin", "verovio", "musescore", "auto"],
        help="Reference renderer (default: builtin).",
    )
    parser.add_argument(
        "--font",
        default=None,
        choices=list(VEROVIO_FONTS),
        help="SMuFL music font for the verovio renderer (implies --renderer "
        "verovio). Use 'Petaluma' for the handwritten / jazz face to measure "
        "the accuracy drop vs the default engraved font.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="MuseScore style for the musescore renderer (implies --renderer "
        "musescore): a .mss path, or 'MuseJazz' for the bundled handwritten "
        "jazz font + chord-symbol text. Closest match to a real jazz fakebook.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Raster resolution for the musescore renderer."
    )
    parser.add_argument(
        "--augment",
        default="clean",
        help="Image degradation preset to test robustness: clean, rotate, "
        "rotate_hard, noise, blur, warp, lighting, photo (default: clean). "
        "Pair a handwritten render with 'photo' to emulate a scanned fakebook.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.dataset == "synthetic":
        items = synthetic_corpus(n_items=args.limit, notes_per_item=args.notes, seed=args.seed)
    else:
        items = music21_corpus(query=args.query, limit=args.limit)

    if not items:
        print("error: no corpus items to evaluate", file=sys.stderr)
        return 1

    # A font/style only takes effect in its renderer; selecting one implies it.
    log = logging.getLogger(__name__)
    renderer = args.renderer
    if args.style and renderer == "builtin":
        log.info("--style set; switching renderer to musescore")
        renderer = "musescore"
    elif args.font and renderer == "builtin":
        log.info("--font set; switching renderer to verovio")
        renderer = "verovio"

    result = evaluate_corpus(
        items,
        engine=args.engine,
        renderer=renderer,
        augmentation=args.augment,
        font=args.font,
        style=args.style,
        dpi=args.dpi,
    )
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
