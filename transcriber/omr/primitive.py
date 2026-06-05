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

# Order in which sharps / flats are added to a key signature (circle of fifths).
_SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
_FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]


def key_alteration_map(sharps: int) -> dict[str, int]:
    """Map a signed key-signature size to ``{letter: +1/-1}`` alterations."""
    if sharps > 0:
        return {letter: 1 for letter in _SHARP_ORDER[:sharps]}
    if sharps < 0:
        return {letter: -1 for letter in _FLAT_ORDER[: -sharps]}
    return {}


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
    key_sharps, keysig_x = detect_key_signature(mask, staves, staff_space)
    heads = _detect_heads(mask, staves, staff_space, config, min_x=keysig_x)

    notes = _heads_to_notes(heads, staves, config, key_sharps=key_sharps)
    logger.info(
        "Primitive OMR: %d staves, %d notes, key=%+d", len(staves), len(notes), key_sharps
    )
    return RecognizedScore(notes=notes, systems=staves, page_count=1, key_sharps=key_sharps)


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
    min_x: float = 0.0,
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
    heads = _detect_filled_heads(mask, staves, staff_space, config, min_x)
    heads += _detect_hollow_heads(mask, staves, staff_space, heads, min_x)
    heads.sort(key=lambda hd: (hd.staff_index, hd.x))
    return heads


def detect_key_signature(
    mask: np.ndarray, staves: list[StaffSystem], staff_space: float
) -> tuple[int, float]:
    """Detect the key signature from accidental glyphs after the clef.

    Returns ``(signed_count, region_right_x)``: a positive count for sharps, a
    negative count for flats, and the x past which note heads begin.  Key
    signatures use a fixed number of accidentals in a fixed circle-of-fifths
    order, so counting the glyphs and classifying them sharp-vs-flat is enough
    to reconstruct which pitches are altered -- their exact positions are not
    needed.
    """
    if not staves:
        return 0, 0.0
    staff = staves[0]
    no_lines = _remove_staff_lines(mask, staves, staff_space)
    labels, n = ndimage.label(no_lines)
    if n == 0:
        return 0, 0.0

    top = staff.top_line_y - 1.5 * staff_space
    bot = staff.bottom_line_y + 1.5 * staff_space
    glyphs: list[tuple[float, slice, int]] = []  # (cx, slice, label)
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        cy, cx = (ys.start + ys.stop - 1) / 2.0, (xs.start + xs.stop - 1) / 2.0
        # Accidental glyphs are tall and narrow, and sit on the staff.
        if not (1.1 * staff_space <= h <= 2.8 * staff_space and 0.2 * staff_space <= w <= 1.1 * staff_space):
            continue
        if (w / h) >= 0.8 or not (top <= cy <= bot):
            continue
        # A real sharp/flat has a tall vertical stroke; a staff-line-removal
        # edge stub is horizontal fragments with no tall column.  Require the
        # densest column to span most of the glyph height.
        component = labels[sl] == idx
        if int(component.sum(axis=0).max()) < 0.55 * h:
            continue
        glyphs.append((cx, sl, idx))

    glyphs.sort(key=lambda g: g[0])
    # Keep the leftmost contiguous cluster that begins right after the clef.
    cluster: list[tuple[float, slice, int]] = []
    for g in glyphs:
        if not cluster:
            if g[0] > staff.x_start + 3.0 * staff_space:
                break  # first glyph is too far right to be a key signature
            cluster.append(g)
        elif g[0] - cluster[-1][0] <= 1.8 * staff_space:
            cluster.append(g)
        else:
            break
    if not cluster:
        return 0, 0.0

    # Classify sharp vs flat by where the ink sits: a flat's filled bowl makes
    # the bottom third far heavier than the top (thin ascender); a sharp is
    # vertically balanced.  The bottom/top ink ratio separates them cleanly
    # (~1.1-1.8 for sharps vs ~3.4-4.3 for flats).
    ratios = []
    for _, sl, idx in cluster:
        comp = labels[sl] == idx
        h = comp.shape[0]
        top_ink = int(comp[: h // 3, :].sum())
        bot_ink = int(comp[2 * h // 3 :, :].sum())
        ratios.append(bot_ink / max(top_ink, 1))
    is_flat = float(np.median(ratios)) > 2.5

    count = len(cluster)
    region_right_x = cluster[-1][0] + 0.8 * staff_space
    return (-count if is_flat else count), region_right_x


def _is_head_shaped(h: float, w: float, staff_space: float) -> bool:
    """A note head is about as wide as it is tall; accidentals are tall+narrow."""
    if h <= 0 or w <= 0:
        return False
    # Reject tall-narrow glyphs (sharps/flats/naturals): width must be a decent
    # fraction of height.
    return (w / h) >= 0.8


def _detect_filled_heads(
    mask: np.ndarray,
    staves: list[StaffSystem],
    staff_space: float,
    config: PrimitiveConfig,
    min_x: float = 0.0,
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
        if not _is_head_shaped(h, w, staff_space):
            continue
        component = labels[sl] == idx
        area = int(component.sum())
        if area < 0.25 * staff_space * staff_space:
            continue
        cy = (ys.start + ys.stop - 1) / 2.0
        cx = (xs.start + xs.stop - 1) / 2.0
        if cx < min_x:  # inside the key-signature / clef region
            continue
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
    min_x: float = 0.0,
) -> list[_Head]:
    """Find hollow heads as small enclosed holes in the original ink mask."""
    holes = ndimage.binary_fill_holes(mask) & ~mask
    labels, n = ndimage.label(holes)
    if n == 0:
        return []

    heads: list[_Head] = []
    min_h, max_h = 0.35 * staff_space, 1.1 * staff_space
    min_w, max_w = 0.45 * staff_space, 1.4 * staff_space
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (min_h <= h <= max_h and min_w <= w <= max_w):
            continue
        if w < h:  # a hollow head's hole is wider than tall; sharp squares are not
            continue
        cy = (ys.start + ys.stop - 1) / 2.0
        cx = (xs.start + xs.stop - 1) / 2.0
        if cx < min_x:
            continue
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
    key_sharps: int = 0,
) -> list[OMRNote]:
    clefs = config.clefs or {}
    key_alter = key_alteration_map(key_sharps)
    notes: list[OMRNote] = []
    onset_by_staff: dict[int, float] = {}

    for head in heads:
        staff = staves[head.staff_index]
        clef_name = clefs.get(head.staff_index + 1, config.default_clef)
        step = staff.step_at(head.y)
        midi, letter = _step_to_pitch(step, clef_name)
        # A note head with no explicit accidental inherits the key signature.
        accidental = None
        alter = key_alter.get(letter, 0)
        if alter:
            midi += alter
            accidental = "sharp" if alter > 0 else "flat"
        duration = 1.0 if head.filled else 2.0
        onset = onset_by_staff.get(head.staff_index, 0.0)
        notes.append(
            OMRNote(
                pitch=int(np.clip(midi, 0, 127)),
                onset=onset,
                duration=duration,
                voice=1,
                staff=head.staff_index + 1,
                accidental=accidental,
            )
        )
        onset_by_staff[head.staff_index] = onset + duration

    return notes


def _step_to_pitch(step: int, clef: str) -> tuple[int, str]:
    """Map a diatonic staff step (0 = bottom line) to (natural MIDI, letter)."""
    bottom = _CLEF_BOTTOM_DIATONIC.get(clef, _CLEF_BOTTOM_DIATONIC["treble"])
    diatonic = bottom + step
    octave, letter_idx = divmod(diatonic, 7)
    letter = "CDEFGAB"[letter_idx]
    midi = 12 * (octave + 1) + _LETTER_SEMITONE[letter]
    return int(np.clip(midi, 0, 127)), letter


def _step_to_midi(step: int, clef: str) -> int:
    """Map a diatonic staff step (0 = bottom line) to a MIDI pitch."""
    return _step_to_pitch(step, clef)[0]


def _disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element of the given radius."""
    d = 2 * radius + 1
    yy, xx = np.ogrid[:d, :d]
    return (yy - radius) ** 2 + (xx - radius) ** 2 <= radius * radius
