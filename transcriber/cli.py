"""Command-line interface for the transcriber pipeline.

Usage::

    python -m transcriber input.wav -o score.musicxml
    transcribe input.mp3 --separation demucs --pitch basic-pitch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import TranscriptionConfig, transcribe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe an audio recording into a MusicXML score "
        "(stem separation -> rhythm/pitch analysis -> MusicXML).",
    )
    parser.add_argument("input", type=Path, help="Input audio file (wav/flac/mp3/...).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output MusicXML path. Defaults to <input>.musicxml.",
    )
    parser.add_argument(
        "--separation",
        choices=["auto", "demucs", "hpss"],
        default="auto",
        help="Stem-separation backend (default: auto).",
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs model name when using the Demucs backend.",
    )
    parser.add_argument(
        "--pitch",
        choices=["auto", "basic-pitch", "pyin"],
        default="auto",
        help="Pitch-transcription backend (default: auto).",
    )
    parser.add_argument(
        "--no-drums",
        action="store_true",
        help="Skip drum transcription.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Analysis sample rate in Hz (default: 22050).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Score title (defaults to the input file stem).",
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

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    output = args.output or args.input.with_suffix(".musicxml")
    title = args.title or args.input.stem

    config = TranscriptionConfig(
        separation_backend=args.separation,
        demucs_model=args.demucs_model,
        pitch_backend=args.pitch,
        transcribe_drums=not args.no_drums,
        sample_rate=args.sample_rate,
        title=title,
    )

    result = transcribe(args.input, output, config=config)

    print(f"Stems: {', '.join(result.stem_names) or '(none)'}")
    print(
        f"Tempo: {result.rhythm.tempo:.1f} BPM  "
        f"Time signature: {result.rhythm.beats_per_measure}/{result.rhythm.beat_unit}"
    )
    print(f"MusicXML written to: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
