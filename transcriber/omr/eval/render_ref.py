"""Render a reference score back to a page image for the accuracy loop.

To *measure* recognition we need (image, ground-truth) pairs.  Real corpora
ship MusicXML, so we render it to an image, recognise that, and compare.

Renderer preference:

1. **builtin** -- a dependency-free PIL engraver used by default.  It lays
   monophonic notes on a treble staff using the *same* diatonic coordinate
   system as :mod:`transcriber.omr.primitive`, which makes the round trip
   deterministic and ideal for unit tests and threshold tuning.
2. **verovio** -- if ``verovio`` (+ ``cairosvg``) are installed, used for
   high-fidelity, real-world-looking engraving of arbitrary scores (better for
   evaluating against full-page corpora).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
from music21 import converter, stream

logger = logging.getLogger(__name__)

# Geometry of the built-in engraver (pixels).  Matches primitive's treble-clef
# assumption: the bottom staff line is E4 (diatonic step 0).
_STAFF_SPACE = 12
_LINE_THICKNESS = 2
_LEFT_MARGIN = 60
_RIGHT_MARGIN = 50
_VMARGIN = 90
_NOTE_SPACING = 42
_LETTERS = "CDEFGAB"
_TREBLE_BOTTOM_DIATONIC = 4 * 7 + 2  # E4


def render_reference(
    score: stream.Score | str,
    out_path: str | Path,
    renderer: str = "builtin",
) -> Path:
    """Render ``score`` (or a MusicXML path) to an image at ``out_path``.

    Args:
        score: A music21 score or a path to MusicXML.
        out_path: Destination image path (``.png``).
        renderer: ``"builtin"``, ``"verovio"`` or ``"auto"`` (verovio if
            available, else builtin).

    Returns:
        The path the image was written to.
    """
    if isinstance(score, str):
        score = converter.parse(score)
    out_path = Path(out_path)

    if renderer == "auto":
        renderer = "verovio" if _verovio_available() else "builtin"

    if renderer == "verovio":
        rendered = _render_verovio(score, out_path)
        if rendered is not None:
            return rendered
        logger.warning("verovio unavailable; falling back to builtin renderer")

    return _render_builtin(score, out_path)


def render_reference_array(score: stream.Score | str, renderer: str = "builtin") -> np.ndarray:
    """Render to an in-memory grayscale ``float32`` array in ``[0, 1]``."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = render_reference(score, Path(tmp) / "ref.png", renderer=renderer)
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Built-in PIL engraver
# --------------------------------------------------------------------------- #
def _diatonic_step(pitch) -> int:
    diatonic = pitch.octave * 7 + _LETTERS.index(pitch.step)
    return diatonic - _TREBLE_BOTTOM_DIATONIC


def _render_builtin(score: stream.Score, out_path: Path) -> Path:
    from PIL import Image, ImageDraw

    notes = list(score.flatten().notes)
    # Expand chords to their lowest pitch for the monophonic engraver.
    events = []
    for n in notes:
        pitch = n.pitches[0] if n.isChord else n.pitch
        events.append((pitch, float(n.quarterLength)))

    n_notes = max(1, len(events))
    width = _LEFT_MARGIN + _RIGHT_MARGIN + n_notes * _NOTE_SPACING

    steps = [_diatonic_step(p) for p, _ in events] or [0]
    max_step, min_step = max(steps + [8]), min(steps + [0])
    top_pad = _VMARGIN + max(0, (max_step - 8)) * (_STAFF_SPACE // 2)
    bot_pad = _VMARGIN + max(0, (0 - min_step)) * (_STAFF_SPACE // 2)
    staff_height = 4 * _STAFF_SPACE
    height = top_pad + staff_height + bot_pad

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    top_line_y = top_pad
    bottom_line_y = top_line_y + staff_height
    line_ys = [top_line_y + k * _STAFF_SPACE for k in range(5)]
    for y in line_ys:
        draw.rectangle([_LEFT_MARGIN - 20, y - _LINE_THICKNESS // 2, width - 20, y + _LINE_THICKNESS // 2], fill=0)

    half = _STAFF_SPACE / 2.0
    nw, nh = int(round(1.4 * _STAFF_SPACE)), _STAFF_SPACE
    for idx, (pitch, ql) in enumerate(events):
        step = _diatonic_step(pitch)
        cx = _LEFT_MARGIN + idx * _NOTE_SPACING + _NOTE_SPACING // 2
        cy = bottom_line_y - step * half
        bbox = [cx - nw // 2, cy - nh // 2, cx + nw // 2, cy + nh // 2]

        filled = ql < 2.0
        if filled:
            draw.ellipse(bbox, fill=0)
        else:
            draw.ellipse(bbox, outline=0, width=2)

        _draw_ledger_lines(draw, step, cx, bottom_line_y, half, nw)

        if ql < 4.0:  # stems for everything but whole notes
            _draw_stem(draw, step, cx, cy, nw, nh)

    img.save(out_path)
    logger.info("Rendered reference (%d notes) to %s", len(events), out_path)
    return out_path


def _draw_ledger_lines(draw, step, cx, bottom_line_y, half, nw):
    extent = nw // 2 + 3
    if step < 0:
        for s in range(-2, step - 1, -2):
            y = bottom_line_y - s * half
            draw.rectangle([cx - extent, y - 1, cx + extent, y + 1], fill=0)
    elif step > 8:
        for s in range(10, step + 1, 2):
            y = bottom_line_y - s * half
            draw.rectangle([cx - extent, y - 1, cx + extent, y + 1], fill=0)


def _draw_stem(draw, step, cx, cy, nw, nh):
    stem_len = 3 * _STAFF_SPACE
    if step < 4:  # below middle line -> stem up on the right
        x = cx + nw // 2 - 1
        draw.rectangle([x - 1, cy - stem_len, x + 1, cy], fill=0)
    else:  # stem down on the left
        x = cx - nw // 2 + 1
        draw.rectangle([x - 1, cy, x + 1, cy + stem_len], fill=0)


# --------------------------------------------------------------------------- #
# verovio (optional, high fidelity)
# --------------------------------------------------------------------------- #
def _verovio_available() -> bool:
    return (
        importlib.util.find_spec("verovio") is not None
        and importlib.util.find_spec("cairosvg") is not None
    )


def _render_verovio(score: stream.Score, out_path: Path) -> Path | None:
    try:
        import cairosvg
        import verovio
    except ImportError:
        return None

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        mxl = Path(tmp) / "ref.musicxml"
        score.write("musicxml", fp=str(mxl))
        toolkit = verovio.toolkit()
        toolkit.loadFile(str(mxl))
        svg = toolkit.renderToSVG(1)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out_path), background_color="white")
    return out_path
