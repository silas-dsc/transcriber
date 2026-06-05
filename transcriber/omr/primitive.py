"""Built-in classical-computer-vision OMR recogniser (zero heavy deps).

This is the always-available fallback used when no deep-learning engine is
installed.  It implements the textbook OMR pipeline with numpy/scipy:

1. **Staff detection** -- horizontal run-length + row projection find the five
   lines of each staff and their spacing.
2. **Staff-line removal** -- clear the thin horizontal lines while keeping note
   heads and stems that cross them.
3. **Note-head detection** -- fill hollow heads, morphological opening to drop
   stems/beams, then connected-component filtering by size and shape.
4. **Pitch assignment** -- a head's vertical position on (or between) the staff
   lines gives its diatonic step; a clef maps that to a MIDI pitch.
5. **Duration & ordering** -- fill ratio separates filled (quarter) from hollow
   (half/whole) heads; left-to-right x-order gives the note sequence.

It is intentionally modest -- it targets clean printed monophonic and simple
polyphonic music -- but it produces a real, structurally valid score for any
input, and it is fully exercised by the accuracy harness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .preprocess import PreprocessConfig, preprocess
from .types import OMRNote, RecognizedScore, StaffSystem

logger = logging.getLogger(__name__)

# Diatonic letter -> semitone offset within an octave.
_LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Diatonic number (octave*7 + letter_index, C=0..B=6) of each clef's *bottom*
# staff line, i.e. the pitch at staff step 0.
_CLEF_BOTTOM_DIATONIC = {
    "treble": 4 * 7 + 2,  # E4
    "bass": 2 * 7 + 4,    # G2
    "alto": 3 * 7 + 0,    # C3 (middle line is C4; bottom line F3) -> see below
}
# Alto clef bottom line is F3; override explicitly to avoid confusion.
_CLEF_BOTTOM_DIATONIC["alto"] = 3 * 7 + 3  # F3


@dataclass
class PrimitiveConfig:
    """Configuration for the built-in recogniser.

    Attributes:
        preprocess: Pre-processing settings applied before detection.
        clefs: Clef to assume for each staff index (1-based), e.g.
            ``{1: "treble", 2: "bass"}``.  Staves without an entry use
            ``default_clef``.
        default_clef: Clef for staves not listed in ``clefs``.
        head_fill_threshold: Fill ratio below which a head is treated as hollow
            (half/whole) rather than filled (quarter).
    """

    preprocess: PreprocessConfig | None = None
    clefs: dict[int, str] | None = None
    default_clef: str = "treble"
    head_fill_threshold: float = 0.55


def recognize_image(image: np.ndarray, config: PrimitiveConfig | None = None) -> RecognizedScore:
    """Recognise a single grayscale page into a :class:`RecognizedScore`."""
    config = config or PrimitiveConfig()
    mask = preprocess(image, config.preprocess)

    staves = detect_staves(mask)
    if not staves:
        logger.info("No staves detected on page")
        return RecognizedScore(notes=[], systems=[], page_count=1)

    staff_space = float(np.median([s.staff_space for s in staves]))
    heads = _detect_heads(mask, staves, staff_space, config)

    notes = _heads_to_notes(heads, staves, config)
    logger.info("Primitive OMR: %d staves, %d notes", len(staves), len(notes))
    return RecognizedScore(notes=notes, systems=staves, page_count=1)


# --------------------------------------------------------------------------- #
# Staff detection
# --------------------------------------------------------------------------- #
def detect_staves(mask: np.ndarray) -> list[StaffSystem]:
    """Detect staff systems (groups of five lines) in an ink mask."""
    line_rows, thickness = _detect_staff_lines(mask)
    if len(line_rows) < 5:
        return []

    centers = np.array(line_rows, dtype=np.float64)
    diffs = np.diff(centers)
    if len(diffs) == 0:
        return []
    staff_space = float(np.median(diffs))
    if staff_space <= 0:
        return []

    # Group consecutive lines whose spacing is close to the staff spacing.
    groups: list[list[float]] = [[centers[0]]]
    for prev, cur in zip(centers[:-1], centers[1:]):
        if (cur - prev) <= 1.8 * staff_space:
            groups[-1].append(cur)
        else:
            groups.append([cur])

    staves: list[StaffSystem] = []
    for group in groups:
        for five in _chunks_of_five(group, staff_space):
            x0, x1 = _staff_extent(mask, five, thickness)
            staves.append(
                StaffSystem(
                    line_ys=[float(y) for y in five],
                    x_start=x0,
                    x_end=x1,
                    staff_space=float(np.mean(np.diff(five))) or staff_space,
                )
            )
    return staves


def _detect_staff_lines(mask: np.ndarray) -> tuple[list[float], int]:
    """Return the y-centres of staff lines and the typical line thickness."""
    proj = mask.sum(axis=1).astype(np.float64)
    if proj.max() <= 0:
        return [], 1
    width = mask.shape[1]
    # A staff line spans most of the staff width.  Threshold relative to the
    # strongest row but with an absolute floor so faint lines still register.
    thresh = max(0.25 * width, 0.45 * proj.max())
    line_rows = proj >= thresh

    centers: list[float] = []
    thicknesses: list[int] = []
    y = 0
    n = len(line_rows)
    while y < n:
        if line_rows[y]:
            y0 = y
            while y < n and line_rows[y]:
                y += 1
            centers.append((y0 + y - 1) / 2.0)
            thicknesses.append(y - y0)
        else:
            y += 1
    thickness = int(round(np.median(thicknesses))) if thicknesses else 1
    return centers, max(1, thickness)


def _chunks_of_five(group: list[float], staff_space: float):
    """Yield 5-line staves from a group of detected lines.

    Clean input yields exactly five lines per staff.  We tolerate a missing or
    extra line: extra lines are split into consecutive groups of five; a group
    of four is extrapolated to five using the spacing.
    """
    if len(group) >= 5:
        for i in range(0, len(group) - 4, 5):
            yield group[i : i + 5]
        return
    if len(group) == 4:
        # Extrapolate a fifth line below using the mean spacing.
        spacing = float(np.mean(np.diff(group))) if len(group) > 1 else staff_space
        yield [*group, group[-1] + spacing]
    # Groups of <=3 are too ambiguous to treat as a staff; skip.


def _staff_extent(mask: np.ndarray, line_ys: list[float], thickness: int) -> tuple[int, int]:
    """Horizontal [x_start, x_end] span of a staff, from its line rows."""
    rows = []
    for y in line_ys:
        lo = max(0, int(y) - thickness)
        hi = min(mask.shape[0], int(y) + thickness + 1)
        rows.append(mask[lo:hi, :])
    band = np.concatenate(rows, axis=0)
    cols = np.where(band.any(axis=0))[0]
    if len(cols) == 0:
        return 0, mask.shape[1] - 1
    return int(cols[0]), int(cols[-1])


# --------------------------------------------------------------------------- #
# Note-head detection
# --------------------------------------------------------------------------- #
@dataclass
class _Head:
    x: float
    y: float
    filled: bool
    staff_index: int


def _detect_heads(
    mask: np.ndarray,
    staves: list[StaffSystem],
    staff_space: float,
    config: PrimitiveConfig,
) -> list[_Head]:
    # Two complementary detectors catch the two kinds of note head:
    #
    #  * Filled heads (quarters/eighths) are solid blobs.  Removing the staff
    #    lines and opening with a head-sized disk isolates them from stems.
    #  * Hollow heads (halves/wholes) are rings.  A ring centred in a staff
    #    *space* has its top and bottom arcs lying exactly on the adjacent staff
    #    lines, so line removal erases the arcs and the ring falls apart.  But
    #    in the *original* mask the ring is still a closed loop enclosing a small
    #    white hole, so we find hollow heads as enclosed holes instead -- which
    #    is robust to whatever the staff lines do.
    heads = _detect_filled_heads(mask, staves, staff_space, config)
    heads += _detect_hollow_heads(mask, staves, staff_space, heads)
    heads.sort(key=lambda hd: (hd.staff_index, hd.x))
    return heads


def _detect_filled_heads(
    mask: np.ndarray,
    staves: list[StaffSystem],
    staff_space: float,
    config: PrimitiveConfig,
) -> list[_Head]:
    no_lines = _remove_staff_lines(mask, staves, staff_space)
    seal = _disk(max(2, int(round(staff_space * 0.16))))
    solid = ndimage.binary_fill_holes(ndimage.binary_closing(no_lines, structure=seal))
    radius = max(2, int(round(0.18 * staff_space)))
    blobs = ndimage.binary_opening(solid, structure=_disk(radius))

    labels, n = ndimage.label(blobs)
    if n == 0:
        return []

    heads: list[_Head] = []
    min_h, max_h = 0.55 * staff_space, 1.9 * staff_space
    min_w, max_w = 0.55 * staff_space, 2.4 * staff_space
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (min_h <= h <= max_h and min_w <= w <= max_w):
            continue
        component = labels[sl] == idx
        area = int(component.sum())
        if area < 0.25 * staff_space * staff_space:
            continue
        cy = (ys.start + ys.stop - 1) / 2.0
        cx = (xs.start + xs.stop - 1) / 2.0
        staff_idx = _nearest_staff(cy, cx, staves)
        if staff_idx is None:
            continue
        # Hollow vs filled: how much of the head region was actually inked
        # before hole-filling.  A filled head is ~100% ink; a hollow ring (e.g.
        # a half note sitting on a line) is well under half.
        fill_ratio = int(no_lines[sl][component].sum()) / area if area else 0.0
        heads.append(
            _Head(x=cx, y=cy, filled=fill_ratio >= config.head_fill_threshold, staff_index=staff_idx)
        )
    return heads


def _detect_hollow_heads(
    mask: np.ndarray,
    staves: list[StaffSystem],
    staff_space: float,
    existing: list[_Head],
) -> list[_Head]:
    """Find hollow heads as small enclosed holes in the original ink mask."""
    holes = ndimage.binary_fill_holes(mask) & ~mask
    labels, n = ndimage.label(holes)
    if n == 0:
        return []

    heads: list[_Head] = []
    min_h, max_h = 0.35 * staff_space, 1.1 * staff_space
    min_w, max_w = 0.35 * staff_space, 1.4 * staff_space
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (min_h <= h <= max_h and min_w <= w <= max_w):
            continue
        cy = (ys.start + ys.stop - 1) / 2.0
        cx = (xs.start + xs.stop - 1) / 2.0
        # Skip holes already explained by a detected (filled) head nearby.
        if any(abs(cx - e.x) < 0.8 * staff_space and abs(cy - e.y) < 0.8 * staff_space for e in existing):
            continue
        staff_idx = _nearest_staff(cy, cx, staves)
        if staff_idx is None:
            continue
        heads.append(_Head(x=cx, y=cy, filled=False, staff_index=staff_idx))
    return heads


def _remove_staff_lines(mask: np.ndarray, staves: list[StaffSystem], staff_space: float) -> np.ndarray:
    """Remove staff lines while preserving note heads and stems.

    Uses the standard morphological approach: a *horizontal opening* with a
    structuring element longer than a note head extracts the long horizontal
    runs (the staff lines, and any beams), which we then subtract from the ink.
    Because note heads span several rows, clipping the one or two rows where a
    line crosses them leaves the head essentially intact; a short vertical
    closing rebridges any hairline gap that does open up.
    """
    # Element longer than a note head (~1.3 staff spaces wide) so heads survive,
    # but far shorter than a staff line so the lines are fully captured.
    length = max(8, int(round(staff_space * 1.8)))
    horizontal = np.ones((1, length), dtype=bool)
    lines = ndimage.binary_opening(mask, structure=horizontal)
    out = mask & ~lines

    # Rebridge note heads/stems that the line crossing split vertically.
    bridge = max(3, int(round(staff_space * 0.3)) | 1)
    out = ndimage.binary_closing(out, structure=np.ones((bridge, 1), dtype=bool))
    return out


def _nearest_staff(cy: float, cx: float, staves: list[StaffSystem]) -> int | None:
    """Index of the staff a head belongs to (allowing a few ledger lines)."""
    best, best_dist = None, None
    for i, staff in enumerate(staves):
        if not (staff.x_start - staff.staff_space <= cx <= staff.x_end + staff.staff_space):
            continue
        # Vertical reach: the staff plus ~4 ledger positions either side.
        reach = staff.staff_space * 4
        if cy < staff.top_line_y - reach or cy > staff.bottom_line_y + reach:
            continue
        center = (staff.top_line_y + staff.bottom_line_y) / 2.0
        dist = abs(cy - center)
        if best_dist is None or dist < best_dist:
            best, best_dist = i, dist
    return best


# --------------------------------------------------------------------------- #
# Pitch & duration
# --------------------------------------------------------------------------- #
def _heads_to_notes(
    heads: list[_Head],
    staves: list[StaffSystem],
    config: PrimitiveConfig,
) -> list[OMRNote]:
    clefs = config.clefs or {}
    notes: list[OMRNote] = []
    onset_by_staff: dict[int, float] = {}

    for head in heads:
        staff = staves[head.staff_index]
        clef_name = clefs.get(head.staff_index + 1, config.default_clef)
        step = staff.step_at(head.y)
        pitch = _step_to_midi(step, clef_name)
        duration = 1.0 if head.filled else 2.0
        onset = onset_by_staff.get(head.staff_index, 0.0)
        notes.append(
            OMRNote(
                pitch=pitch,
                onset=onset,
                duration=duration,
                voice=1,
                staff=head.staff_index + 1,
            )
        )
        onset_by_staff[head.staff_index] = onset + duration

    return notes


def _step_to_midi(step: int, clef: str) -> int:
    """Map a diatonic staff step (0 = bottom line) to a MIDI pitch."""
    bottom = _CLEF_BOTTOM_DIATONIC.get(clef, _CLEF_BOTTOM_DIATONIC["treble"])
    diatonic = bottom + step
    octave, letter_idx = divmod(diatonic, 7)
    letter = "CDEFGAB"[letter_idx]
    midi = 12 * (octave + 1) + _LETTER_SEMITONE[letter]
    return int(np.clip(midi, 0, 127))


def _disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element of the given radius."""
    d = 2 * radius + 1
    yy, xx = np.ogrid[:d, :d]
    return (yy - radius) ** 2 + (xx - radius) ** 2 <= radius * radius
