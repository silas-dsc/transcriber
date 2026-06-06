"""Command-line interface for the OMR pipeline.

Usage::

    python -m transcriber.omr score.pdf -o score.musicxml
    omr photo.jpg --engine oemer
    omr scan.pdf --engine ensemble -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import engines as engines_mod
from .pipeline import OMRConfig, recognize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omr",
        description="Optical Music Recognition: turn a sheet-music PDF or image "
        "into a MusicXML score.",
    )
    parser.add_argument("input", type=Path, help="Input PDF or image (png/jpg/tiff/...).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output MusicXML path. Defaults to <input>.musicxml.",
    )
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "oemer", "homr", "audiveris", "primitive", "ensemble"],
        help="Recognition engine (default: auto -> best installed, else built-in).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render resolution for PDF inputs (default: 300).",
    )
    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="Time signature used to bar the music (default: 4/4).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Score title (defaults to the input file stem).",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip the musical sanity checks / safe repairs.",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Also apply risky semantic repairs (octave-outlier correction).",
    )
    parser.add_argument(
        "--llm-review",
        action="store_true",
        help="Cross-check the result with Claude (needs the 'llm' extra + API key).",
    )
    parser.add_argument(
        "--mark-review",
        action="store_true",
        help="Annotate the output MusicXML with 'review?' marks on uncertain measures.",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List the OMR engines available in this environment and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (INFO) logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_engines:
        available = engines_mod.available_engines()
        print("Available OMR engines (best first):")
        for name in available:
            print(f"  - {name}")
        return 0

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    output = args.output or args.input.with_suffix(".musicxml")
    title = args.title or args.input.stem

    config = OMRConfig(
        engine=args.engine,
        dpi=args.dpi,
        time_signature=args.time_signature,
        title=title,
        semantic_check=not args.no_semantic,
        semantic_aggressive=args.aggressive,
        llm_review=args.llm_review,
        mark_review=args.mark_review,
    )

    result = recognize(args.input, output, config=config)

    print(f"Engine:     {result.engine}")
    print(f"Pages:      {result.page_count}")
    print(f"Confidence: {result.confidence:.2f}")
    if len(result.engine_confidences) > 1:
        ranked = sorted(result.engine_confidences.items(), key=lambda kv: -kv[1])
        print("Ensemble:   " + ", ".join(f"{n}={c:.2f}" for n, c in ranked))
    if result.semantic_report is not None:
        print(f"Semantic:   {result.semantic_report.summary()}")
    if result.confidence_report is not None:
        cr = result.confidence_report
        print(f"Review:     {cr.summary()}")
        for item in cr.review_items[:10]:
            loc = f"part {item.part}" + (f", m.{item.measure}" if item.measure is not None else "")
            print(f"  [{item.confidence:.2f}] {loc}: {'; '.join(item.reasons)}")
    print(f"MusicXML written to: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
