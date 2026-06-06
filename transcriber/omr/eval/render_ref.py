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
3. **musescore** -- if the MuseScore CLI is installed, the highest-fidelity
   option and the only one with the handwritten **MuseJazz** "jazz" font and
   chord-symbol text (pass ``style="MuseJazz"``).  Combine with an ``augment``
   preset (e.g. ``"photo"``) to approximate a scanned/photographed fakebook.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
from music21 import converter, stream

logger = logging.getLogger(__name__)

# SMuFL music fonts bundled with verovio.  "Leipzig" is verovio's default
# engraved face; "Petaluma" is the handwritten / "jazz" face -- rendering the
# corpus in Petaluma is how we measure how badly the engines degrade on the
# handwritten lead-sheet style used by jazz fakebooks.
VEROVIO_FONTS = ("Leipzig", "Bravura", "Petaluma", "Leland", "Gootville")

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

# Standard treble-clef staff steps (bottom line = 0) for the order of sharps
# (F C G D A E B) and flats (B E A D G C F).
_SHARP_STEPS = [8, 5, 9, 6, 3, 7, 4]
_FLAT_STEPS = [4, 7, 3, 6, 2, 5, 1]
_KEYSIG_SPACING = 11  # px between successive accidental glyphs


def render_reference(
    score: stream.Score | str,
    out_path: str | Path,
    renderer: str = "builtin",
    font: str | None = None,
    style: str | None = None,
    dpi: int = 300,
) -> Path:
    """Render ``score`` (or a MusicXML path) to an image at ``out_path``.

    Args:
        score: A music21 score or a path to MusicXML.
        out_path: Destination image path (``.png``).
        renderer: ``"builtin"``, ``"verovio"``, ``"musescore"`` or ``"auto"``
            (verovio if available, else builtin).
        font: SMuFL music font for the verovio renderer (see
            :data:`VEROVIO_FONTS`).  ``None`` keeps verovio's default
            (``Leipzig``).  Use ``"Petaluma"`` for the handwritten / "jazz"
            face.  Ignored by the built-in engraver, which draws raw glyphs.
        style: MuseScore style for the ``musescore`` renderer -- a ``.mss``
            path, or ``"MuseJazz"`` for the bundled handwritten jazz style
            (musical font + chord-symbol text).  Ignored by other renderers.
        dpi: Raster resolution for the ``musescore`` renderer.

    Returns:
        The path the image was written to.
    """
    if isinstance(score, str):
        score = converter.parse(score)
    out_path = Path(out_path)

    if renderer == "auto":
        renderer = "verovio" if _verovio_available() else "builtin"

    if renderer == "musescore":
        rendered = _render_musescore(score, out_path, style=style, dpi=dpi)
        if rendered is not None:
            return rendered
        logger.warning("MuseScore unavailable; falling back to builtin renderer")

    if renderer == "verovio":
        rendered = _render_verovio(score, out_path, font=font)
        if rendered is not None:
            return rendered
        logger.warning("verovio unavailable; falling back to builtin renderer")

    if font:
        logger.warning("built-in renderer cannot apply font %r; ignoring it", font)
    return _render_builtin(score, out_path)


def render_reference_array(
    score: stream.Score | str,
    renderer: str = "builtin",
    font: str | None = None,
    style: str | None = None,
    dpi: int = 300,
) -> np.ndarray:
    """Render to an in-memory grayscale ``float32`` array in ``[0, 1]``."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = render_reference(
            score, Path(tmp) / "ref.png", renderer=renderer, font=font, style=style, dpi=dpi
        )
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

    # Lay notes out in the same (onset, pitch) order the metric flattens them
    # in, so simultaneous notes (grace notes, ornaments, chord reductions) are
    # read back in a consistent order rather than registering as swaps.
    notes = sorted(
        score.flatten().notes,
        key=lambda n: (float(n.offset), (n.pitches[0] if n.isChord else n.pitch).midi),
    )
    events = []
    for n in notes:
        pitch = n.pitches[0] if n.isChord else n.pitch
        events.append((pitch, float(n.quarterLength)))

    sharps = _score_sharps(score)
    keysig_width = abs(sharps) * _KEYSIG_SPACING + (12 if sharps else 0)

    n_notes = max(1, len(events))
    width = _LEFT_MARGIN + keysig_width + _RIGHT_MARGIN + n_notes * _NOTE_SPACING

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
    _draw_key_signature(draw, sharps, bottom_line_y, half)

    key_alter = _key_alteration(sharps)
    nw, nh = int(round(1.4 * _STAFF_SPACE)), _STAFF_SPACE
    notes_x0 = _LEFT_MARGIN + keysig_width
    for idx, (pitch, ql) in enumerate(events):
        step = _diatonic_step(pitch)
        cx = notes_x0 + idx * _NOTE_SPACING + _NOTE_SPACING // 2
        cy = bottom_line_y - step * half
        bbox = [cx - nw // 2, cy - nh // 2, cx + nw // 2, cy + nh // 2]

        # Draw an inline accidental when the note departs from the key signature.
        actual_alter = pitch.accidental.alter if pitch.accidental else 0
        if actual_alter != key_alter.get(pitch.step, 0):
            _draw_accidental(draw, int(actual_alter), cx - nw // 2 - 13, cy)

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


def _score_sharps(score: stream.Score) -> int:
    """Signed key-signature size from an *explicit* key signature (0 if none).

    We deliberately do not fall back to key *analysis*: the notes carry their
    own accidental spelling, so inventing a key signature that disagrees with
    them would corrupt the render/recognise round trip.
    """
    ks = score.recurse().getElementsByClass("KeySignature").first()
    if ks is not None and ks.sharps is not None:
        return int(ks.sharps)
    return 0


def _draw_key_signature(draw, sharps: int, bottom_line_y: float, half: float) -> None:
    """Draw |sharps| accidental glyphs in standard order after the clef area."""
    if sharps == 0:
        return
    steps = (_SHARP_STEPS if sharps > 0 else _FLAT_STEPS)[: abs(sharps)]
    x = _LEFT_MARGIN - 6
    for step in steps:
        cy = bottom_line_y - step * half
        if sharps > 0:
            _draw_sharp(draw, x, cy)
        else:
            _draw_flat(draw, x, cy)
        x += _KEYSIG_SPACING


def _draw_sharp(draw, cx, cy):
    """A '#'-shaped glyph: two verticals crossed by two horizontals."""
    h = int(1.0 * _STAFF_SPACE)
    draw.rectangle([cx - 3, cy - h, cx - 3, cy + h], fill=0, width=1)
    draw.rectangle([cx + 3, cy - h, cx + 3, cy + h], fill=0, width=1)
    draw.rectangle([cx - 5, cy - 3, cx + 5, cy - 2], fill=0)
    draw.rectangle([cx - 5, cy + 2, cx + 5, cy + 3], fill=0)


def _draw_flat(draw, cx, cy):
    """A flat glyph: a tall ascender with a filled bowl at the bottom."""
    h = int(1.2 * _STAFF_SPACE)
    draw.rectangle([cx - 3, cy - h, cx - 2, cy + 4], fill=0)  # ascender
    # Filled bowl in the lower half (this is what distinguishes it from a sharp).
    draw.ellipse([cx - 2, cy - 2, cx + 4, cy + 5], fill=0)


def _draw_natural(draw, cx, cy):
    """A natural glyph: diagonally offset strokes (upper-left + lower-right)."""
    h = int(0.9 * _STAFF_SPACE)
    draw.rectangle([cx - 3, cy - h, cx - 3, cy + h // 3], fill=0, width=1)  # upper-left
    draw.rectangle([cx + 3, cy - h // 3, cx + 3, cy + h], fill=0, width=1)  # lower-right
    draw.rectangle([cx - 3, cy - 3, cx + 3, cy - 2], fill=0)
    draw.rectangle([cx - 3, cy + 2, cx + 3, cy + 3], fill=0)


def _draw_accidental(draw, alter: int, cx, cy) -> None:
    if alter > 0:
        _draw_sharp(draw, cx, cy)
    elif alter < 0:
        _draw_flat(draw, cx, cy)
    else:
        _draw_natural(draw, cx, cy)


def _key_alteration(sharps: int) -> dict:
    """Map a signed key signature to ``{step_letter: +1/-1}``."""
    order = "FCGDAEB" if sharps > 0 else "BEADGCF"
    alter = 1 if sharps > 0 else -1
    return {letter: alter for letter in order[: abs(sharps)]}


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
def _set_verovio_font(toolkit, font: str) -> bool:
    """Select ``font`` on a verovio toolkit, returning True on success.

    The pip ``verovio`` package takes a dict; older SWIG bindings took a JSON
    string.  ``setOptions`` returns False (and logs ``Cannot parse JSON``) for
    the wrong form, so try the dict first and fall back to the string.
    """
    import json

    try:
        if toolkit.setOptions({"font": font}):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(toolkit.setOptions(json.dumps({"font": font})))
    except (TypeError, ValueError):
        return False


def _verovio_available() -> bool:
    return (
        importlib.util.find_spec("verovio") is not None
        and importlib.util.find_spec("cairosvg") is not None
    )


def _render_verovio(score: stream.Score, out_path: Path, font: str | None = None) -> Path | None:
    try:
        import cairosvg  # cairocffi dlopens libcairo at import -> OSError if absent
        import verovio
    except (ImportError, OSError) as exc:
        logger.warning("verovio/cairosvg unavailable (%s); falling back to builtin", exc)
        return None

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        mxl = Path(tmp) / "ref.musicxml"
        score.write("musicxml", fp=str(mxl))
        toolkit = verovio.toolkit()
        if font and not _set_verovio_font(toolkit, font):
            logger.warning("verovio rejected font %r; rendering with its default", font)
        toolkit.loadFile(str(mxl))
        svg = toolkit.renderToSVG(1)
        try:
            cairosvg.svg2png(
                bytestring=svg.encode(), write_to=str(out_path), background_color="white"
            )
        except OSError as exc:
            # cairosvg imports fine but needs a system libcairo at call time;
            # without it, fall back to the built-in engraver rather than crash.
            logger.warning("cairosvg could not rasterise (libcairo missing?): %s", exc)
            return None
    return out_path


# --------------------------------------------------------------------------- #
# MuseScore (optional, highest fidelity -- the only renderer with the
# handwritten MuseJazz "jazz" font + chord-symbol text used by real lead sheets)
# --------------------------------------------------------------------------- #
MUSESCORE_CANDIDATES = (
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
    "mscore",
    "musescore",
    "MuseScore4",
    "mscore4portable",
)


def _find_musescore() -> str | None:
    """Locate a MuseScore CLI executable, or None if not installed."""
    import shutil

    for cand in MUSESCORE_CANDIDATES:
        if cand.startswith("/"):
            if Path(cand).exists():
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def _resolve_style(style: str | None, musescore: str) -> str | None:
    """Resolve a style argument to a ``.mss`` path.

    ``None`` -> no style (default engraving font).  An existing path -> used
    as-is.  ``"MuseJazz"`` -> the ``MuseJazz.mss`` bundled with the MuseScore
    app, so callers don't need to know the app-internal path.
    """
    if not style:
        return None
    if style.lower() != "musejazz":
        return style  # treat as an explicit style path
    if "/Contents/MacOS/" in musescore:
        cand = Path(musescore).parents[1] / "Resources" / "styles" / "MuseJazz.mss"
        if cand.exists():
            return str(cand)
    logger.warning("MuseJazz.mss not found near %s; rendering without a style", musescore)
    return None


def _run_musescore(musescore: str, args: list[str]) -> bool:
    """Run the MuseScore CLI; return True on success (logs + False on failure)."""
    import subprocess

    try:
        subprocess.run(
            [musescore, *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
            stdin=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("MuseScore command failed: %s", exc)
        return False


def _render_musescore(
    score: stream.Score, out_path: Path, style: str | None = None, dpi: int = 300
) -> Path | None:
    """Render ``score`` to ``out_path`` via the MuseScore CLI.

    A style (e.g. the handwritten MuseJazz font) is applied with a two-step
    convert -- ``musicxml + -S style -> .mscz -> .png`` -- because MuseScore 4
    only honours ``musicalSymbolFont`` from ``-S`` when going through a score
    file, not on a direct ``.musicxml -> .png`` export.  MuseScore writes one
    ``<stem>-N.png`` per page; pages are flattened onto white and stacked
    vertically into a single image so the rest of the pipeline sees one image.
    """
    import tempfile

    musescore = _find_musescore()
    if musescore is None:
        return None
    style_path = _resolve_style(style, musescore)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mxl = tmp / "ref.musicxml"
        score.write("musicxml", fp=str(mxl))

        src = mxl
        if style_path:
            mscz = tmp / "styled.mscz"
            if not _run_musescore(musescore, ["-S", style_path, str(mxl), "-o", str(mscz)]):
                return None
            src = mscz

        if not _run_musescore(musescore, ["-r", str(dpi), str(src), "-o", str(tmp / "page.png")]):
            return None

        pages = sorted(tmp.glob("page-*.png"))
        if not pages:
            logger.warning("MuseScore produced no PNG output for %s", out_path.name)
            return None
        _stack_pages_on_white(pages, out_path)
    return out_path


def _stack_pages_on_white(pages: list[Path], out_path: Path) -> None:
    """Flatten each (possibly RGBA) page onto white and stack vertically."""
    from PIL import Image

    imgs = []
    for p in pages:
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        imgs.append(Image.alpha_composite(bg, im).convert("RGB"))

    if len(imgs) == 1:
        imgs[0].save(out_path)
        return

    width = max(im.width for im in imgs)
    canvas = Image.new("RGB", (width, sum(im.height for im in imgs)), (255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out_path)
