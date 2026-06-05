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
        choices=["builtin", "verovio", "auto"],
        help="Reference renderer (default: builtin).",
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

    result = evaluate_corpus(items, engine=args.engine, renderer=args.renderer)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
