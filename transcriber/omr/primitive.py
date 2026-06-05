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
    # Detect accidental glyphs first and erase *all* of them before head
    # detection, so neither a key signature nor an inline accidental is mistaken
    # for a note head.
    glyphs = _find_accidental_glyphs(mask, staves, staff_space)
    clean = _erase_glyphs(mask, glyphs)
    heads = _detect_heads(clean, staves, staff_space, config)

    # The key signature is the leftmost glyph cluster -- but only if it sits
    # clearly left of the first note head.  A lone accidental hugging the first
    # head is an inline accidental on that note, not a one-sharp/flat key.
    first_head_x = min((h.x for h in heads), default=None)
    key_sharps, keysig_x = _keysig_from_glyphs(glyphs, staves[0], staff_space, first_head_x)

    # Attach inline accidentals (printed sharps/flats/naturals that depart from
    # the key signature) to their note heads.
    _assign_inline_accidentals(glyphs, heads, staff_space, keysig_x)

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
    """Return the y-centres of staff lines and the typical line thickness.

    Project a *horizontally opened* mask so only long, full-width runs survive.
    This is essential on melodies that dwell on a single ledger-line pitch: the
    many short ledger segments would otherwise pile up into a phantom staff line
    and corrupt the staff geometry.
    """
    width = mask.shape[1]
    line_se = max(40, width // 6)  # far longer than a ledger line or note head
    lines_only = ndimage.binary_opening(mask, structure=np.ones((1, line_se), dtype=bool))
    proj = lines_only.sum(axis=1).astype(np.float64)
    if proj.max() <= 0:  # fall back to the raw projection (e.g. very narrow input)
        proj = mask.sum(axis=1).astype(np.float64)
        if proj.max() <= 0:
            return [], 1
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
    inline_alter: int | None = None  # explicit accidental on this note, if any


@dataclass
class _AccGlyph:
    """A detected accidental glyph (sharp / flat / natural)."""

    cx: float
    cy: float
    component: np.ndarray  # boolean bitmap of the glyph within ``bbox``
    bbox: tuple  # (slice, slice) location of the glyph in the image

    def alter(self) -> int:
        return _classify_accidental(self.component)


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


def _find_accidental_glyphs(
    mask: np.ndarray, staves: list[StaffSystem], staff_space: float
) -> list[_AccGlyph]:
    """Find every accidental-shaped glyph (sharp / flat / natural) on the staff.

    Accidentals are tall, narrow, sit on the staff, and have a tall vertical
    stroke (which distinguishes them from staff-line-removal edge stubs).
    """
    if not staves:
        return []
    staff = staves[0]
    no_lines = _remove_staff_lines(mask, staves, staff_space)
    labels, n = ndimage.label(no_lines)
    if n == 0:
        return []

    # An accidental sits at its note head's height, which for a ledger-line
    # note can be several staff spaces above/below the staff -- match the
    # note-head vertical reach so high/low accidentals are not missed.
    top = staff.top_line_y - 4.0 * staff_space
    bot = staff.bottom_line_y + 4.0 * staff_space
    glyphs: list[_AccGlyph] = []
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        cy, cx = (ys.start + ys.stop - 1) / 2.0, (xs.start + xs.stop - 1) / 2.0
        if not (1.1 * staff_space <= h <= 2.8 * staff_space and 0.2 * staff_space <= w <= 1.1 * staff_space):
            continue
        if (w / h) >= 0.8 or not (top <= cy <= bot):
            continue
        component = labels[sl] == idx
        if int(component.sum(axis=0).max()) < 0.55 * h:
            continue
        glyphs.append(_AccGlyph(cx=cx, cy=cy, component=component, bbox=sl))

    glyphs.sort(key=lambda g: g.cx)
    return glyphs


def _erase_glyphs(mask: np.ndarray, glyphs: list[_AccGlyph]) -> np.ndarray:
    """Return a copy of ``mask`` with the accidental glyphs removed.

    Detecting accidentals before note heads and clearing them prevents a flat's
    filled bowl from being mistaken for a small head, and stops sharps from
    disturbing an adjacent head.
    """
    out = mask.copy()
    for g in glyphs:
        region = out[g.bbox]
        region[g.component] = False
    return out


def _classify_accidental(comp: np.ndarray) -> int:
    """Classify an accidental glyph: ``+1`` sharp, ``-1`` flat, ``0`` natural."""
    h, w = comp.shape
    total = int(comp.sum()) or 1
    top_ink = int(comp[: h // 3, :].sum())
    bot_ink = int(comp[2 * h // 3 :, :].sum())
    # A flat's filled bowl makes the bottom far heavier than the (thin) top.
    if bot_ink / max(top_ink, 1) > 2.2:
        return -1
    # Sharp vs natural: a natural's strokes are diagonally offset (upper-left +
    # lower-right), a sharp is vertically symmetric.
    tl = int(comp[: h // 2, : w // 2].sum())
    tr = int(comp[: h // 2, w // 2 :].sum())
    bl = int(comp[h // 2 :, : w // 2].sum())
    br = int(comp[h // 2 :, w // 2 :].sum())
    diagonal = (tl + br) - (tr + bl)
    if diagonal / total > 0.12:
        return 0  # natural
    return 1  # sharp


def detect_key_signature(
    mask: np.ndarray, staves: list[StaffSystem], staff_space: float
) -> tuple[int, float]:
    """Detect the key signature from the accidental cluster after the clef.

    Returns ``(signed_count, region_right_x)``: positive for sharps, negative
    for flats.  Key signatures use a fixed number of accidentals in a fixed
    circle-of-fifths order, so the count plus a sharp-vs-flat decision is enough
    to reconstruct which pitches are altered.
    """
    if not staves:
        return 0, 0.0
    glyphs = _find_accidental_glyphs(mask, staves, staff_space)
    return _keysig_from_glyphs(glyphs, staves[0], staff_space)


def _keysig_from_glyphs(
    glyphs: list[_AccGlyph],
    staff: StaffSystem,
    staff_space: float,
    first_head_x: float | None = None,
) -> tuple[int, float]:
    # Keep the leftmost contiguous cluster that begins right after the clef.
    cluster: list[_AccGlyph] = []
    for g in glyphs:
        if not cluster:
            if g.cx > staff.x_start + 3.0 * staff_space:
                break  # first glyph too far right to be a key signature
            cluster.append(g)
        elif g.cx - cluster[-1].cx <= 1.8 * staff_space:
            cluster.append(g)
        else:
            break
    if not cluster:
        return 0, 0.0

    # A key signature sits clearly to the left of the first note (a clef's worth
    # of space); an accidental hugging the first head (~1.75 spaces, centre to
    # centre) is an inline accidental on that note, not a one-sharp/flat key.
    if first_head_x is not None and (first_head_x - cluster[-1].cx) < 2.5 * staff_space:
        return 0, 0.0

    alters = [g.alter() for g in cluster]
    # A key signature is all sharps or all flats; vote, and treat naturals (rare
    # here) as not part of a key signature.
    n_flat = sum(1 for a in alters if a < 0)
    n_sharp = sum(1 for a in alters if a > 0)
    count = len(cluster)
    region_right_x = cluster[-1].cx + 0.8 * staff_space
    if n_flat > n_sharp:
        return -count, region_right_x
    if n_sharp > 0:
        return count, region_right_x
    return 0, region_right_x


def _assign_inline_accidentals(
    glyphs: list[_AccGlyph],
    heads: list[_Head],
    staff_space: float,
    keysig_x: float,
) -> None:
    """Attach each non-key-signature accidental glyph to the note on its right."""
    for g in glyphs:
        if g.cx <= keysig_x:
            continue  # part of the key signature
        # The note an accidental modifies sits just to its right, same height.
        candidates = [
            hd
            for hd in heads
            if 0.1 * staff_space <= (hd.x - g.cx) <= 1.9 * staff_space
            and abs(hd.y - g.cy) <= 0.9 * staff_space
        ]
        if candidates:
            nearest = min(candidates, key=lambda hd: hd.x - g.cx)
            nearest.inline_alter = g.alter()


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
        # An explicit inline accidental overrides the key signature for this
        # note; otherwise the note inherits the key signature.
        accidental = None
        alter = head.inline_alter if head.inline_alter is not None else key_alter.get(letter, 0)
        if alter:
            midi += alter
            accidental = "sharp" if alter > 0 else "flat"
        elif head.inline_alter == 0:
            accidental = "natural"
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
                confidence=_pitch_confidence(head.y, staff),
            )
        )
        onset_by_staff[head.staff_index] = onset + duration

    return notes


# Head centroids carry a small (stem-direction-dependent) bias of up to ~0.3
# steps that does not change the rounded pitch.  Treat anything within this
# dead zone of a step as fully confident; only heads genuinely straddling the
# boundary between two steps are flagged.
_CONF_DEAD_ZONE = 0.3


def _pitch_confidence(y: float, staff: StaffSystem) -> float:
    """How unambiguous a head's pitch is, from how centred it is on a step.

    A head sitting on or near a step is unambiguous (1.0); one straddling the
    boundary between two steps could round either way (->0.0).  This is the
    recogniser's honest self-doubt signal, used to flag notes for human review.
    """
    half = staff.staff_space / 2.0
    if half <= 0:
        return 1.0
    frac = ((staff.bottom_line_y - y) / half) % 1.0
    dist_to_step = min(frac, 1.0 - frac)  # 0 on a step .. 0.5 between two steps
    if dist_to_step <= _CONF_DEAD_ZONE:
        return 1.0
    return float(max(0.0, 1.0 - (dist_to_step - _CONF_DEAD_ZONE) / (0.5 - _CONF_DEAD_ZONE)))


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
