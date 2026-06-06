"""Chord-symbol recognition for lead sheets -- the jazz-fakebook payload.

The note recognisers (oemer / homr / the built-in primitive) transcribe *notes*
but ignore the chord symbols printed above the staff, which are the whole point
of a jazz fake book.  This module reads that band of text and turns it into
chord symbols attachable to the score.

Status -- initial scaffold:

* :func:`jazz_text_to_figure` (jazz shorthand -> music21 figure) and
  :func:`attach_chords` (write symbols back into a score) are complete and
  tested -- they close the render/recognise/score loop.
* :func:`recognize_chords` provides the geometry (where to look) and a
  pluggable OCR step: it uses ``pytesseract`` or ``easyocr`` if installed and
  otherwise returns nothing (leaving the rest of the pipeline unaffected) with
  a clear log line.

Accuracy loop: render lead-sheet MusicXML with the MuseJazz font (which draws
chord text in MuseJazz Text), recognise, and score with
:func:`transcriber.omr.eval.metrics.compare_chords`.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .types import OMRChordSymbol, StaffSystem

logger = logging.getLogger(__name__)

# Jazz shorthand / unicode -> music21 figure fragments.
_JAZZ_UNICODE = {
    "♯": "#", "♭": "b", "△": "maj7", "Δ": "maj7",
    "ø": "m7b5", "Ø": "m7b5", "°": "dim", "–": "-", "—": "-",
}
_JTOK_RE = re.compile(r"^([A-G])([#b]?)(.*)$")


def jazz_text_to_figure(text: str) -> str | None:
    """Convert recognised jazz chord text into a music21 figure, or ``None``.

    Fakebook conventions: ``-`` = minor, ``△`` = major7, ``ø`` =
    half-diminished, ``°`` = diminished; a flat root spelled ``b`` becomes
    music21's ``-``.  Examples: ``"C-7"`` -> ``"Cm7"``, ``"Bb7"`` -> ``"B-7"``,
    ``"A-7b5"`` -> ``"Am7b5"``, ``"C△"`` -> ``"Cmaj7"``.
    """
    if not text:
        return None
    s = text.strip()
    for uni, ascii_ in _JAZZ_UNICODE.items():
        s = s.replace(uni, ascii_)
    m = _JTOK_RE.match(s)
    if not m:
        return None
    letter, acc, rest = m.groups()
    root = letter + ("-" if acc == "b" else acc)  # jazz flat 'b' -> music21 '-'
    quality = ("m" + rest[1:]) if rest.startswith("-") else rest  # leading '-' = minor
    return root + quality


def _valid_figure(figure: str) -> bool:
    from music21 import harmony

    try:
        harmony.ChordSymbol(figure)
        return True
    except Exception:
        return False


def attach_chords(score, omr_chords: list[OMRChordSymbol]):
    """Insert ``omr_chords`` into ``score``'s first part as music21 ChordSymbols.

    Skips figures music21 cannot parse.  Returns the same score for chaining;
    this is what lets a recognised result be scored by ``compare_chords``.
    """
    from music21 import harmony, stream

    part = score.parts[0] if score.parts else score
    for c in omr_chords:
        try:
            cs = harmony.ChordSymbol(c.figure)
        except Exception:
            logger.debug("dropping unparseable chord figure %r", c.figure)
            continue
        if isinstance(part, stream.Stream):
            part.insert(float(c.onset), cs)
    return score


def _ocr_backend() -> str | None:
    import importlib.util

    if importlib.util.find_spec("pytesseract") is not None:
        return "pytesseract"
    if importlib.util.find_spec("easyocr") is not None:
        return "easyocr"
    return None


def _crop_chord_band(image: np.ndarray, system: StaffSystem) -> np.ndarray | None:
    """Crop the band just above a staff, where chord symbols are printed."""
    space = system.staff_space or 10.0
    y0 = max(0, int(system.top_line_y - 3.0 * space))
    y1 = max(0, int(system.top_line_y - 0.3 * space))
    x0 = max(0, int(system.x_start))
    x1 = min(image.shape[1], int(system.x_end))
    if y1 <= y0 or x1 <= x0:
        return None
    band = image[y0:y1, x0:x1]
    if band.size == 0:
        return None
    if band.dtype != np.uint8:
        band = (np.clip(band, 0.0, 1.0) * 255).astype(np.uint8)
    return band


def _ocr_band(band: np.ndarray, backend: str) -> list[tuple[str, float]]:
    """OCR a cropped band -> ``[(text, x_center_fraction)]`` (0..1 across width)."""
    if band is None or band.size == 0:
        return []
    width = band.shape[1] or 1
    if backend == "pytesseract":
        import pytesseract
        from PIL import Image

        data = pytesseract.image_to_data(
            Image.fromarray(band), output_type=pytesseract.Output.DICT
        )
        toks = []
        for i, txt in enumerate(data["text"]):
            if txt.strip():
                xc = (data["left"][i] + data["width"][i] / 2.0) / width
                toks.append((txt.strip(), xc))
        return toks
    if backend == "easyocr":  # pragma: no cover - heavy optional dep
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        toks = []
        for box, txt, _conf in reader.readtext(band):
            xc = (sum(p[0] for p in box) / len(box)) / width
            toks.append((txt.strip(), xc))
        return toks
    return []


def recognize_chords(
    image: np.ndarray,
    systems: list[StaffSystem],
    beats_per_system: float = 4.0,
    confidence: float = 0.5,
) -> list[OMRChordSymbol]:
    """Recognise chord symbols above each staff system.

    The onset of each symbol is approximated from its horizontal position
    (``beats_per_system`` beats span each system); the coarse placement is why
    :func:`compare_chords` matches within a one-beat tolerance.  Returns an
    empty list (with a log line) when no OCR backend is installed, so callers
    can run unconditionally.
    """
    backend = _ocr_backend()
    if backend is None:
        logger.info(
            "no OCR backend installed (pip install pytesseract / easyocr); "
            "skipping chord-symbol recognition"
        )
        return []

    out: list[OMRChordSymbol] = []
    for staff_idx, system in enumerate(systems, start=1):
        band = _crop_chord_band(image, system)
        if band is None:
            continue
        for text, x_frac in _ocr_band(band, backend):
            figure = jazz_text_to_figure(text)
            if not figure or not _valid_figure(figure):
                continue
            onset = (staff_idx - 1) * beats_per_system + x_frac * beats_per_system
            out.append(
                OMRChordSymbol(figure=figure, onset=onset, staff=staff_idx, confidence=confidence)
            )
    logger.info("recognised %d chord symbol(s) via %s", len(out), backend)
    return out
