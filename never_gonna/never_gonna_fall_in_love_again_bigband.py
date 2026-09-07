"""Never Gonna Fall in Love Again - 10-piece big band chart.

Built from a live recording (accordion, cello, acoustic guitar, percussion,
bass, voice).  The harmony, the form and the bass line are transcribed from
the recording; the horn writing, the piano part and the drum part are
arranged.  See README.md for how each number below was measured.

Scoring, 10 players:
    Voice
    Alto Sax, Tenor Sax, Baritone Sax
    Trumpet 1, Trumpet 2, Trombone
    Piano, Bass, Drums

The feel is a bolero: the recording's guitar, bass and percussion all group
the bar 3+3+2 in eighths, so the written figures sit on those accents.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

from music21 import (articulations, chord, clef, duration, dynamics, expressions,
                     harmony, instrument, key, layout, meter, note, pitch,
                     stream, tempo)

# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
# 124 bars.  Bar 93 is a 2/4 bar: the band cuts the last turnaround of the
# third verse short, which is what re-aligns everything after it (measured -
# see README).
TOTAL_BARS = 124
SHORT_BAR = 93                      # the one 2/4 bar
BEAT = 100                          # median measured tempo; the take is rubato


def bar_len(bar: int) -> float:
    """Length of a bar in quarter notes."""
    return 2.0 if bar == SHORT_BAR else 4.0


SECTIONS = [
    (1, 8, "Intro"),
    (9, 16, "Intro"),
    (17, 17, ""),
    (18, 32, "Verse 1"),
    (33, 48, "Chorus 1"),
    (49, 49, ""),
    (50, 64, "Verse 2"),
    (65, 80, "Chorus 2"),
    (81, 85, "Perc. break"),
    (86, 100, "Verse 3 (inst.)"),
    (101, 116, "Chorus 3"),
    (117, 124, "Outro"),
]
REHEARSAL = {1: "A", 9: "B", 18: "C", 33: "D", 50: "E", 65: "F",
             81: "G", 86: "H", 101: "I", 117: "J"}

# ---------------------------------------------------------------------------
# Harmony - one entry per bar; two entries where the chord changes mid-bar.
# Derived from chroma template matching against the guitar/accordion/cello
# stems with the transcribed bass supplying the root.  92.5% of the
# transcribed bass notes are chord tones of this chart.
# ---------------------------------------------------------------------------
CHART: list[list[str]] = [
    ['Am'], ['Am/G'], ['D7/F#'], ['D7/F#'],   # bars 1-4
    ['F'], ['E7'], ['Am'], ['E7'],   # bars 5-8
    ['Am'], ['Am/G'], ['D7/F#'], ['D7/F#'],   # bars 9-12
    ['F'], ['E7'], ['E7'], ['F'],   # bars 13-16
    ['E7'], ['Am'], ['Am/G'], ['D7/F#'],   # bars 17-20
    ['D7/F#'], ['F'], ['E7'], ['Am'],   # bars 21-24
    ['E7'], ['Am'], ['Am/G'], ['D7/F#'],   # bars 25-28
    ['D7/F#'], ['F'], ['E7'], ['Am'],   # bars 29-32
    ['C'], ['F'], ['G'], ['C', 'G/B'],   # bars 33-36
    ['Am', 'Am/G'], ['F'], ['G'], ['C'],   # bars 37-40
    ['C'], ['F'], ['G'], ['C', 'G/B'],   # bars 41-44
    ['Am', 'Am/G'], ['F'], ['G7'], ['C'],   # bars 45-48
    ['C'], ['Am'], ['Am/G'], ['D7/F#'],   # bars 49-52
    ['D7/F#'], ['F'], ['E7'], ['Am'],   # bars 53-56
    ['E7'], ['Am'], ['Am/G'], ['D7/F#'],   # bars 57-60
    ['D7/F#'], ['F'], ['E7'], ['Am'],   # bars 61-64
    ['C'], ['F'], ['G'], ['C', 'G/B'],   # bars 65-68
    ['Am', 'Am/G'], ['F'], ['G'], ['C'],   # bars 69-72
    ['C'], ['F'], ['G'], ['C', 'G/B'],   # bars 73-76
    ['Am', 'Am/G'], ['F'], ['G7'], ['C'],   # bars 77-80
    ['C'], ['NC'], ['NC'], ['NC'],   # bars 81-84
    ['NC'], ['Am'], ['Am/G'], ['D7/F#'],   # bars 85-88
    ['D7/F#'], ['F'], ['E7'], ['Am'],   # bars 89-92
    ['Am'], ['Am'], ['Am/G'], ['D7/F#'],   # bars 93-96
    ['D7/F#'], ['F'], ['E7'], ['Am'],   # bars 97-100
    ['C'], ['F'], ['G'], ['C', 'G/B'],   # bars 101-104
    ['Am', 'Am/G'], ['F'], ['G'], ['C'],   # bars 105-108
    ['C'], ['F'], ['G'], ['C', 'G/B'],   # bars 109-112
    ['Am', 'Am/G'], ['F'], ['G7'], ['C'],   # bars 113-116
    ['C'], ['C'], ['C'], ['C'],   # bars 117-120
    ['C'], ['C'], ['C'], ['C'],   # bars 121-124
]
assert len(CHART) == TOTAL_BARS

# ---------------------------------------------------------------------------
# Chords
# ---------------------------------------------------------------------------
# symbol -> (root pc, chord-tone pcs in voicing order, bass pc, ChordSymbol figure)
SPEC: dict[str, tuple[int, tuple[int, ...], int, str]] = {
    # symbol -> (root pc, voicing tones, bass pc, ChordSymbol figure)
    # The written symbol is kept as plain as the recording supports; the extra
    # voicing tone in each case is one the chroma measurements actually show
    # (the 7th on Am, the 6th on C, F and G), so the horns get some colour
    # without the chart claiming a chord the band does not play.
    "Am":    (9, (9, 0, 4, 7), 9, "Am"),
    "Am/G":  (9, (9, 0, 4, 7), 7, "Am/G"),
    "D7/F#": (2, (2, 6, 9, 0), 6, "D7/F#"),
    "F":     (5, (5, 9, 0, 2), 5, "F"),
    "E7":    (4, (4, 8, 11, 2), 4, "E7"),
    "C":     (0, (0, 4, 7, 9), 0, "C"),
    "G":     (7, (7, 11, 2, 4), 7, "G"),
    "G/B":   (7, (7, 11, 2, 4), 11, "G/B"),
    "G7":    (7, (7, 11, 2, 5), 7, "G7"),
    "NC":    (0, (), 0, ""),
}
# the plain triad, for figures that should not sound the 7th
TRIAD = {"Am": (9, 0, 4), "Am/G": (9, 0, 4), "D7/F#": (2, 6, 9), "F": (5, 9, 0),
         "E7": (4, 8, 11), "C": (0, 4, 7), "G": (7, 11, 2), "G/B": (7, 11, 2),
         "G7": (7, 11, 2), "NC": ()}


def chord_at(bar: int, off: float) -> str:
    """The chord sounding at ``off`` quarter notes into ``bar``."""
    cell = CHART[bar - 1]
    if len(cell) > 1 and off >= bar_len(bar) / 2:
        return cell[1]
    return cell[0]


def changes(bar: int) -> list[tuple[float, str]]:
    cell = CHART[bar - 1]
    if len(cell) == 1:
        return [(0.0, cell[0])]
    return [(0.0, cell[0]), (bar_len(bar) / 2, cell[1])]


def _ps(name: str) -> int:
    return int(pitch.Pitch(name).ps)


def nearest(pc: int, target: int) -> int:
    base = pc + 12 * ((target - pc) // 12)
    return min((base, base + 12), key=lambda x: abs(x - target))


# Practical written ranges expressed as concert MIDI.  Chosen conservatively:
# everything here sits inside the range a competent section player is happy to
# read at sight, with no extremes.
CONCERT_RANGE = {
    "alto":  (_ps("F3"), _ps("D5")),     # written D4  - B5
    "tenor": (_ps("C3"), _ps("A4")),     # written D4  - B5
    "bari":  (_ps("D-2"), _ps("C4")),    # written B-3 - A5
    "tpt1":  (_ps("A3"), _ps("F5")),     # written B3  - G5
    "tpt2":  (_ps("G3"), _ps("C5")),     # written A3  - D5
    "tbn":   (_ps("F2"), _ps("A4")),
}
REEDS = ("alto", "tenor", "bari")
BRASS = ("tpt1", "tpt2", "tbn")
ALL_HORNS = REEDS + BRASS


def fit(p: int, who: str) -> int:
    lo, hi = CONCERT_RANGE[who]
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def stack_below(lead: int, tones: tuple[int, ...], n: int) -> list[int]:
    """``n`` chord tones descending from ``lead`` - a close block voicing."""
    out = [lead]
    p = lead
    while len(out) < n + 1:
        cands = [q for q in (nearest(t, p - 6) for t in tones) if q < p]
        p = max(cands) if cands else p - 12
        out.append(p)
    return out


def section_voicing(lead: int, sym: str, slots: tuple[str, ...],
                    *, triad: bool = False) -> dict[str, int]:
    """Block-voice ``slots`` under ``lead``.

    The whole stack is octave-shifted together so the top voice lands in its
    own range; only then is each lower voice fitted.  Shifting as a block is
    what keeps a section sounding like one instrument instead of scattering
    the voicing across octaves.
    """
    tones = TRIAD[sym] if triad else SPEC[sym][1]
    if not tones:
        return {}
    stack = stack_below(lead, tones, len(slots) - 1)
    lo, hi = CONCERT_RANGE[slots[0]]
    shift = 0
    while stack[0] + shift > hi:
        shift -= 12
    while stack[0] + shift < lo:
        shift += 12
    stack = [p + shift for p in stack]
    out: dict[str, int] = {}
    prev: int | None = None
    for slot, p in zip(slots, stack):
        q = fit(p, slot)
        if prev is not None and q >= prev:          # never let voices cross
            q -= 12
            slo, _shi = CONCERT_RANGE[slot]
            if q < slo:
                q += 12
        out[slot] = q
        prev = q
    return out


# ---------------------------------------------------------------------------
# Transcribed material: (bar, eighth-position in bar, length in eighths, MIDI)
# BASS is the bass stem tracked with pYIN and quantised to the measured grid.
# VOX is the lead vocal, same method; 21 short chromatic glide artifacts were
# snapped to the nearest scale tone.
# ---------------------------------------------------------------------------
BASS = [
    (1,3,3,45), (1,6,3,52), (2,1,2,45), (2,3,8,43), (3,3,2,42), (3,6,4,42), (4,2,1,36), (4,3,7,42),
    (5,3,3,41), (5,6,3,48), (6,3,3,40), (6,6,5,47), (7,3,3,45), (7,6,3,52), (8,1,1,40), (8,3,6,45),
    (9,3,3,45), (9,6,3,52), (10,1,2,45), (10,3,8,43), (11,3,2,42), (11,6,4,42), (12,2,1,36),
    (12,3,4,42), (13,0,2,42), (13,3,3,41), (13,6,3,48), (14,3,2,40), (14,6,1,47), (15,0,1,47),
    (15,1,1,40), (15,3,2,45), (15,7,1,52), (16,3,3,41), (16,6,1,48), (16,7,6,40), (18,0,2,45),
    (18,5,2,45), (19,0,7,43), (19,7,2,42), (20,2,4,42), (20,6,1,36), (20,7,7,42), (21,7,3,41),
    (22,2,3,48), (22,5,1,41), (23,0,2,40), (23,3,4,47), (24,0,2,45), (24,4,1,52), (24,6,2,45),
    (25,0,2,41), (25,3,1,48), (25,4,3,40), (26,0,3,45), (26,3,3,52), (26,6,2,45), (27,1,7,43),
    (28,0,2,42), (28,3,4,42), (28,7,1,36), (29,0,8,42), (30,1,2,41), (30,3,3,48), (30,6,1,41),
    (31,0,3,40), (31,3,1,47), (32,0,1,45), (32,1,1,33), (32,3,5,52), (33,1,2,52), (33,3,5,48),
    (34,0,3,41), (34,6,2,41), (35,0,2,43), (35,3,3,43), (35,6,2,47), (36,0,4,48), (36,4,4,47),
    (37,0,4,45), (37,4,4,43), (38,0,3,41), (38,3,3,48), (38,6,1,41), (39,0,3,43), (39,3,3,50),
    (39,6,1,43), (40,0,1,36), (40,3,3,48), (40,7,1,43), (41,0,3,48), (41,3,3,52), (41,6,1,40),
    (42,0,3,41), (42,3,1,48), (42,6,1,41), (43,0,3,43), (43,6,1,43), (44,0,4,48), (44,4,3,47),
    (45,0,4,45), (45,4,4,43), (46,1,2,41), (46,3,3,48), (46,6,1,41), (47,0,5,43), (47,6,1,47),
    (48,0,2,48), (48,3,4,48), (48,7,1,43), (49,0,6,48), (49,6,2,47), (50,0,3,45), (50,4,1,33),
    (50,6,2,45), (51,0,8,43), (52,0,2,42), (52,3,3,42), (52,7,1,36), (53,0,2,42), (53,3,4,42),
    (54,0,3,41), (54,3,3,48), (54,6,1,41), (55,0,3,40), (55,3,2,47), (55,6,1,40), (56,0,6,45),
    (56,7,1,45), (57,0,3,41), (57,3,1,48), (57,4,3,40), (58,0,4,45), (58,6,2,45), (59,0,8,43),
    (60,0,2,42), (60,3,4,42), (60,7,1,36), (61,0,2,42), (61,3,4,42), (62,0,3,41), (62,3,3,48),
    (62,6,1,41), (63,0,3,40), (63,3,3,47), (63,6,1,40), (64,0,3,45), (64,6,1,45), (64,7,1,47),
    (65,0,7,48), (66,0,3,41), (66,6,2,41), (67,0,2,43), (67,3,2,43), (67,6,2,47), (68,0,3,48),
    (68,4,4,47), (69,0,4,45), (69,4,4,43), (70,0,3,41), (70,3,3,48), (70,6,1,41), (71,0,3,43),
    (71,6,1,43), (72,0,2,48), (72,3,3,48), (73,3,1,52), (73,6,1,43), (73,7,1,45), (74,0,3,41),
    (74,4,2,48), (74,6,1,41), (75,0,3,43), (75,6,1,43), (76,1,3,48), (76,5,3,47), (77,1,3,45),
    (77,5,4,43), (78,1,3,41), (78,4,3,48), (78,7,1,41), (79,1,2,43), (79,4,3,43), (79,7,2,47),
    (80,1,2,48), (80,4,4,48), (81,1,3,48), (84,6,1,32), (85,7,3,45), (86,2,3,52), (86,5,2,45),
    (86,7,2,43), (87,2,5,43), (87,7,2,42), (88,2,4,42), (88,6,1,36), (88,7,2,42), (89,2,4,42),
    (89,7,3,41), (90,2,3,48), (90,5,1,41), (90,7,3,40), (91,2,3,47), (91,5,1,40), (91,7,3,45),
    (92,5,2,45), (92,7,3,41), (93,2,1,48), (93,3,3,40), (94,3,3,45), (95,1,2,45), (95,3,2,43),
    (95,6,5,43), (96,3,2,42), (96,6,3,42), (97,3,2,42), (97,6,4,42), (98,3,2,41), (98,5,3,48),
    (99,0,1,41), (99,2,2,40), (99,6,1,52), (100,2,1,48), (100,3,1,47), (100,5,3,45), (101,3,4,48),
    (101,7,1,40), (102,1,3,41), (102,4,2,48), (102,7,1,41), (103,1,2,43), (103,4,3,43),
    (103,7,1,47), (104,1,3,48), (104,5,3,47), (105,1,4,45), (105,5,4,43), (106,1,3,41),
    (106,4,3,48), (107,1,3,43), (107,4,1,50), (107,7,1,43), (108,1,6,48), (109,0,1,43),
    (109,1,2,48), (109,4,2,48), (109,7,1,40), (110,1,3,41), (110,6,1,48), (111,1,3,43),
    (111,4,2,50), (111,7,1,43), (112,1,1,48), (112,3,2,48), (112,5,4,47), (113,1,4,45),
    (113,5,4,43), (114,2,1,41), (114,4,1,48), (114,7,1,41), (115,1,2,43), (115,4,3,43),
    (115,7,1,47), (116,5,1,48), (116,7,1,48), (117,0,1,43), (117,2,1,48), (117,4,3,48),
    (118,1,1,48), (118,5,1,52), (119,0,1,43), (119,1,2,48), (119,4,1,48), (120,7,1,43),
    (121,0,6,36), (121,7,1,43), (122,3,1,48), (122,5,2,48), (123,3,8,36)
]

VOX = [
    (9,7,1,57), (12,0,1,57), (13,0,1,59), (13,5,2,57), (14,4,2,59), (18,2,1,52), (18,3,1,53),
    (18,5,1,52), (18,6,2,59), (19,0,2,57), (19,2,1,57), (19,4,1,50), (19,5,1,48), (19,6,2,50),
    (20,2,1,50), (20,5,1,45), (20,6,1,48), (20,7,3,50), (21,2,2,48), (21,6,1,47), (22,1,2,50),
    (22,3,1,52), (22,4,2,50), (22,6,1,57), (22,7,1,59), (23,0,1,56), (23,1,1,55), (23,3,2,52),
    (23,6,1,50), (24,0,1,48), (26,3,1,52), (26,4,1,53), (26,5,2,52), (26,7,1,57), (27,0,2,59),
    (27,2,1,55), (27,3,1,50), (27,5,1,48), (27,7,2,50), (28,7,1,48), (29,0,3,50), (29,3,2,48),
    (29,7,1,47), (30,2,1,50), (30,4,1,52), (30,5,1,50), (30,7,1,57), (31,0,1,59), (31,1,1,60),
    (31,6,1,50), (31,7,1,48), (32,2,1,48), (33,5,1,52), (33,6,1,55), (34,1,1,57), (34,2,1,59),
    (34,4,4,57), (35,1,1,59), (35,3,1,47), (35,4,1,50), (35,6,1,53), (36,0,1,57), (36,1,2,55),
    (36,3,1,57), (36,4,2,55), (36,6,1,57), (37,2,2,45), (37,4,1,48), (37,5,2,52), (38,1,4,53),
    (39,2,1,50), (39,3,1,52), (39,4,5,50), (40,1,1,52), (40,3,2,48), (41,4,1,48), (41,5,1,52),
    (41,6,1,55), (42,0,1,59), (42,2,1,55), (42,3,3,57), (42,6,1,55), (42,7,1,57), (43,3,1,60),
    (43,5,1,60), (43,6,1,59), (43,7,1,55), (44,2,1,57), (44,4,1,55), (44,5,1,55), (44,6,1,57),
    (44,7,1,55), (45,2,1,45), (45,4,1,48), (45,6,2,52), (46,1,1,55), (46,2,1,55), (46,3,4,53),
    (47,3,5,50), (48,0,4,48), (50,3,1,52), (50,4,1,53), (50,6,1,52), (50,7,1,57), (51,0,2,59),
    (51,4,1,50), (51,5,1,50), (51,6,1,48), (51,7,2,50), (52,2,1,48), (53,0,3,50), (53,3,1,48),
    (53,7,1,47), (54,4,1,52), (54,5,1,50), (54,7,1,57), (55,0,1,59), (55,1,1,56), (55,3,2,52),
    (55,5,1,48), (55,7,2,50), (56,1,1,48), (58,2,2,52), (58,4,2,53), (58,7,1,57), (59,0,2,59),
    (59,2,3,60), (59,6,6,60), (60,5,1,60), (60,7,1,48), (61,0,1,52), (61,1,2,50), (61,3,2,48),
    (62,3,1,50), (62,4,1,52), (62,5,2,50), (62,7,2,59), (63,1,2,56), (63,3,1,52), (63,5,2,62),
    (63,7,5,60), (64,4,2,57), (65,4,1,52), (65,6,3,55), (66,1,1,59), (66,2,1,59), (66,6,2,56),
    (67,3,1,47), (67,4,1,50), (67,6,1,53), (68,0,1,55), (68,1,2,55), (68,3,2,57), (68,5,1,55),
    (68,6,1,57), (69,4,1,48), (69,5,4,52), (70,2,2,53), (70,5,2,53), (71,2,1,50), (71,3,1,52),
    (71,4,5,50), (72,1,1,52), (72,3,1,48), (73,2,2,48), (73,5,1,52), (73,6,1,55), (74,0,1,59),
    (74,2,1,55), (74,3,1,57), (74,5,1,57), (74,6,2,56), (75,3,1,60), (75,5,1,60), (75,6,1,59),
    (75,7,1,55), (76,0,1,57), (76,5,1,55), (76,7,1,55), (77,2,2,45), (77,4,2,48), (77,7,2,52),
    (78,2,2,55), (78,4,5,53), (79,5,1,50), (79,6,1,52), (79,7,2,50), (80,2,3,48), (94,7,1,57),
    (96,0,2,57), (96,5,2,57), (97,0,1,69), (98,2,1,74), (98,6,2,48), (99,3,1,56), (101,6,1,53),
    (101,7,2,55), (102,2,1,59), (102,3,1,59), (102,4,3,57), (102,7,1,55), (103,0,1,57),
    (103,4,1,47), (103,5,1,50), (103,7,1,53), (104,1,1,57), (104,2,2,55), (104,5,1,55),
    (104,6,2,57), (105,5,1,48), (105,6,1,52), (106,2,3,53), (106,6,1,57), (106,7,1,59),
    (107,5,5,50), (108,2,1,52), (108,3,2,48), (109,6,1,52), (109,7,2,55), (110,2,1,59),
    (110,3,1,55), (110,4,1,57), (110,7,1,55), (111,4,1,60), (111,6,1,60), (111,7,1,59),
    (112,0,1,55), (112,5,1,55), (112,7,2,55), (113,4,1,45), (113,6,1,48), (114,0,2,52),
    (114,2,1,53), (114,3,1,55), (114,4,2,53), (114,7,1,53), (115,3,1,50), (115,4,1,52),
    (115,5,5,50), (116,2,1,48), (116,5,1,48), (117,3,2,52), (117,6,2,48), (118,0,1,50),
    (118,4,1,48), (119,3,1,52), (119,5,1,50), (119,6,1,48), (119,7,2,50), (120,1,1,48),
    (120,4,1,48), (121,3,1,52), (121,5,2,48), (121,7,1,50)
]

# ---------------------------------------------------------------------------
# Rhythm section
# ---------------------------------------------------------------------------
# The bolero accent: eighth positions 0, 3 and 6 of the bar (3+3+2).  Measured
# in the recording's guitar, bass and percussion alike.
TRESILLO = (0.0, 1.5, 3.0)
TRESILLO_SHORT = (0.0, 1.5)          # the 2/4 bar


def tresillo(bar: int) -> tuple[float, ...]:
    return TRESILLO_SHORT if bar == SHORT_BAR else TRESILLO


def bass_events() -> dict[int, list[tuple[float, float, int]]]:
    """The transcribed bass, as {bar: [(offset, ql, midi)]}.

    Positions and lengths arrive in eighths; a note is clipped so it cannot
    run past its own bar, and anything left with no length is dropped.
    """
    out: dict[int, list[tuple[float, float, int]]] = {}
    for bar, off, ln, midi in BASS:
        if not 1 <= bar <= TOTAL_BARS:
            continue
        room = bar_len(bar) - off * 0.5
        if room <= 0:
            continue
        ql = min(ln * 0.5, room)
        if ql <= 0:
            continue
        out.setdefault(bar, []).append((off * 0.5, ql, midi))
    for bar, evs in out.items():
        evs.sort()
        # trim overlaps so the part is monophonic and readable
        for i in range(len(evs) - 1):
            o, q, m = evs[i]
            if o + q > evs[i + 1][0]:
                evs[i] = (o, evs[i + 1][0] - o, m)
        out[bar] = [e for e in evs if e[1] > 0]
    return out


def _fix_bass_octaves(bybar: dict[int, list]) -> dict[int, list]:
    """Pull isolated octave slips in the tracked bass back to their neighbours."""
    flat = [(b, o, q, m) for b, evs in bybar.items() for o, q, m in evs]
    flat.sort()
    for i in range(1, len(flat) - 1):
        b, o, q, m = flat[i]
        prv, nxt = flat[i - 1][3], flat[i + 1][3]
        if abs(m - prv) > 12 and abs(m - nxt) > 12:
            flat[i] = (b, o, q, min((m - 12, m, m + 12),
                                    key=lambda x: abs(x - prv) + abs(x - nxt)))
    out: dict[int, list] = {}
    for b, o, q, m in flat:
        out.setdefault(b, []).append((o, q, m))
    return out


BASS_BY_BAR = _fix_bass_octaves(bass_events())


def bass_for(bar: int) -> list[tuple[float, float, int]]:
    """Transcribed bass where it exists, otherwise an idiomatic bolero figure.

    Gaps are the drum break and the odd bar the tracker lost; filling them
    from the chart keeps the part continuous and playable.
    """
    got = BASS_BY_BAR.get(bar, [])
    if got:
        return got
    sym = chord_at(bar, 0.0)
    if sym == "NC":
        return []
    root, tones, bass_pc, _ = SPEC[sym]
    r = nearest(bass_pc, _ps("A2"))
    fifth = nearest((root + 7) % 12, r + 5)
    pat = tresillo(bar)
    if len(pat) == 2:
        return [(pat[0], 1.5, r), (pat[1], 0.5, fifth)]
    return [(pat[0], 1.5, r), (pat[1], 1.5, fifth), (pat[2], 1.0, r)]


# Rootless voicings, intervals in semitones above the root.  These are the
# shapes a pianist in the Bill Evans line actually puts under a singer: the
# root is left to the bass, minor and dominant chords are voiced from the
# third or the seventh with the 9th on top, and the plain major chords are
# taken as 6/9 rather than major-7 so the leading tone never fights the tune.
ROOTLESS: dict[str, tuple[int, ...]] = {
    "Am":    (3, 7, 10, 14),      # b3 5 b7 9
    "Am/G":  (3, 7, 10, 14),
    "D7/F#": (4, 9, 10, 14),      # 3 13 b7 9
    "F":     (4, 7, 9, 14),       # 3 5 6 9
    "E7":    (4, 7, 10, 13),      # 3 5 b7 b9  - the b9 belongs in A minor
    "C":     (4, 7, 9, 14),       # 6/9
    "G":     (4, 7, 9, 14),       # 6/9
    "G/B":   (4, 7, 9, 14),
    "G7":    (4, 9, 10, 14),      # 3 13 b7 9
}


def evans_voicing(sym: str, prev_top: int | None) -> list[int]:
    """A four-note rootless voicing, inverted to follow the previous one.

    Every inversion that fits inside a tenth is tried and the one whose top
    note moves least from the last chord wins, so the top voice walks rather
    than jumps and the inner voices come along with it.
    """
    if sym not in ROOTLESS:
        return []
    root = SPEC[sym][0]
    pcs = sorted({(root + i) % 12 for i in ROOTLESS[sym]})
    floor = _ps("B3")
    best: tuple[int, list[int]] | None = None
    for start in range(len(pcs)):
        seq: list[int] = []
        for k in range(len(pcs)):
            pc = pcs[(start + k) % len(pcs)]
            if not seq:
                p = pc + 12 * ((floor - pc + 11) // 12)
            else:
                p = pc + 12 * ((seq[-1] - pc) // 12 + 1)
            seq.append(p)
        if seq[-1] - seq[0] > 16 or seq[-1] > _ps("D5"):
            continue
        cost = abs(seq[-1] - (prev_top if prev_top is not None else _ps("A4")))
        if best is None or cost < best[0]:
            best = (cost, seq)
    return best[1] if best else []


def left_hand(sym: str, variant: int) -> list[int]:
    """Root, or a two-note shell.  Sometimes nothing - the bass has the root."""
    if sym not in SPEC or not SPEC[sym][1]:
        return []
    root, tones, bass_pc, _fig = SPEC[sym][:4]
    low = nearest(bass_pc, _ps("A2"))
    if variant == 0:
        return [low]
    if variant == 1:                       # root and seventh, the usual shell
        seventh = [t for t in tones if (t - root) % 12 in (10, 11)]
        if seventh:
            return sorted({low, nearest(seventh[0], low + 8)})
        return [low]
    if variant == 2:
        return sorted({low, nearest((root + 7) % 12, low + 7)})
    return []                              # lay out


# Comping rhythms as (offset, quarterLength).  The tune's own 3+3+2 is in
# here, but so are anticipations, late entries and bars of nothing: comping
# the same figure for 124 bars is what made the part sound mechanical.
COMP = [
    [(0.0, 1.5), (1.5, 1.5), (3.0, 1.0)],
    [(0.0, 2.5), (2.5, 1.5)],
    [(0.5, 1.0), (1.5, 1.5), (3.0, 1.0)],
    [(0.0, 4.0)],
    [(1.5, 1.5), (3.0, 1.0)],
    [(0.0, 1.0), (1.5, 0.5), (2.5, 1.5)],
    [(0.0, 1.5), (2.5, 1.5)],
    [(0.0, 2.0), (2.0, 1.0), (3.5, 0.5)],
    [],
    [(2.5, 1.5)],
]
# Which rhythms a section draws on, and how often the piano lays out.  The
# verses are sparser because the singer is exposed there.
COMP_PLAN = [
    (1, 8, (3, 0, 3, 6)),
    (9, 17, (0, 6, 1, 0)),
    (18, 32, (4, 8, 6, 9, 1, 8, 0, 4)),
    (33, 48, (0, 7, 2, 0, 5, 1, 0, 6)),
    (49, 64, (4, 8, 6, 9, 1, 8, 0, 4)),
    (65, 80, (0, 5, 2, 7, 0, 1, 5, 0)),
    (81, 85, (3,)),
    (86, 100, (1, 6, 0, 4, 2, 0, 6, 1)),
    (101, 116, (0, 5, 7, 0, 2, 1, 5, 0)),
    (117, 124, (3, 6, 3, 1, 3, 6, 3, 3)),
]


def comp_rhythm(bar: int) -> list[tuple[float, float]]:
    pat = (0,)
    for lo, hi, choices in COMP_PLAN:
        if lo <= bar <= hi:
            pat = choices
            break
    rows = COMP[pat[(bar - 1) % len(pat)]]
    total = bar_len(bar)
    out = []
    for off, ql in rows:
        if off >= total:
            continue
        out.append((off, min(ql, total - off)))
    return out


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------
# Where each piece of the kit sits on the percussion staff, and its notehead.
KIT = {
    # piece: (staff position, notehead, General MIDI drum number)
    # Standard kit only - kick, snare, toms, hi-hat, ride, crash.
    "crash":  ("A5", "x", 49),         # crash cymbal
    "hh":     ("G5", "x", 42),         # closed hi-hat, stick
    "hho":    ("G5", "circle-x", 46),  # open hi-hat
    "ride":   ("F5", "x", 51),         # ride cymbal
    "bell":   ("F5", "diamond", 53),   # ride bell
    "tom_hi": ("E5", None, 48),        # high tom
    "tom_md": ("D5", None, 47),        # mid tom
    "snare":  ("C5", None, 38),        # snare, struck
    "xstick": ("C5", "x", 37),         # snare, side stick
    "tom_lo": ("A4", None, 43),        # floor tom
    "bd":     ("F4", None, 36),        # bass drum
    "hhped":  ("D4", "x", 44),         # hi-hat, foot
}
# staff position + notehead identifies a piece uniquely, which is how the
# exported notes are matched back to their kit piece when the drum part list
# is written (music21 emits neither per-note instruments nor MIDI numbers).
BY_LOOK = {(pos, head): (name, midi) for name, (pos, head, midi) in KIT.items()}
# General MIDI percussion names, so a reader maps each line to the right piece
GM_NAME = {
    36: "Bass Drum 1", 37: "Side Stick", 38: "Acoustic Snare",
    42: "Closed Hi Hat", 43: "High Floor Tom", 44: "Pedal Hi-Hat",
    46: "Open Hi-Hat", 47: "Low-Mid Tom", 48: "Hi-Mid Tom",
    49: "Crash Cymbal 1", 51: "Ride Cymbal 1", 53: "Ride Bell",
}

# Grooves are (offset, [pieces], quarterLength).  Every one of them is one bar
# long, repeats without variation inside a section, and keeps the backbeat in
# the same place, so the part reads at a glance and stays out of the singer's
# way.  Texture changes between sections, not within them.
_EIGHTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def _ride(piece: str, hits: list[float]) -> list[tuple[float, list[str], float]]:
    return [(o, [piece], 0.5) for o in hits]


# Intro A - brushes feel: hi-hat pedal only, nothing on the head.
G_INTRO_A = [(1.0, ["hhped"], 1.0), (3.0, ["hhped"], 1.0),
             (0.0, ["bd"], 1.0)]
# Intro B - hi-hat eighths join, rim click marks 2 and 4.
G_INTRO_B = (_ride("hh", _EIGHTHS)
             + [(1.0, ["xstick"], 1.0), (3.0, ["xstick"], 1.0),
                (0.0, ["bd"], 1.0), (1.5, ["bd"], 0.5)])
# Verse - the core groove.  Rim click on 2 and 4, bass drum on the 3+3+2.
G_VERSE = (_ride("hh", _EIGHTHS)
           + [(1.0, ["xstick"], 1.0), (3.0, ["xstick"], 1.0),
              (0.0, ["bd"], 1.0), (1.5, ["bd"], 0.5)])
# Chorus - move to the ride and open the backbeat onto the snare.
G_CHORUS = (_ride("ride", _EIGHTHS)
            + [(1.0, ["snare"], 1.0), (3.0, ["snare"], 1.0),
               (0.0, ["bd"], 1.0), (1.5, ["bd"], 0.5), (3.0, ["bd"], 1.0)])
# Third verse is instrumental - same groove as the verse but on the ride.
G_VERSE3 = (_ride("ride", _EIGHTHS)
            + [(1.0, ["xstick"], 1.0), (3.0, ["xstick"], 1.0),
               (0.0, ["bd"], 1.0), (1.5, ["bd"], 0.5)])
# Last chorus - ride bell, and the bass drum doubles the horn accents.
G_CHORUS3 = (_ride("bell", _EIGHTHS)
             + [(1.0, ["snare"], 1.0), (3.0, ["snare"], 1.0),
                (0.0, ["bd"], 1.0), (1.5, ["bd"], 0.5), (3.0, ["bd"], 1.0)])
G_OUTRO = (_ride("ride", _EIGHTHS)
           + [(1.0, ["xstick"], 1.0), (3.0, ["xstick"], 1.0),
              (0.0, ["bd"], 1.0)])
# The percussion break: toms carry it, no cymbals, so the band can walk back in.
G_BREAK = [(0.0, ["tom_lo"], 0.5), (0.5, ["tom_lo"], 0.5), (1.0, ["snare"], 0.5),
           (1.5, ["tom_md"], 0.5), (2.0, ["tom_md"], 0.5), (2.5, ["snare"], 0.5),
           (3.0, ["tom_hi"], 0.5), (3.5, ["tom_hi"], 0.5),
           (0.0, ["bd"], 1.0), (2.0, ["bd"], 1.0)]

# Half-bar fill on beats 3-4, used to hand over between sections.
FILL_HALF = [(0.0, ["ride"], 1.0), (1.0, ["ride"], 1.0),
             (2.0, ["snare"], 0.5), (2.5, ["snare"], 0.5),
             (3.0, ["tom_md"], 0.5), (3.5, ["tom_lo"], 0.5),
             (0.0, ["bd"], 1.0)]
FILL_FULL = [(0.0, ["snare"], 0.5), (0.5, ["snare"], 0.5),
             (1.0, ["tom_hi"], 0.5), (1.5, ["tom_hi"], 0.5),
             (2.0, ["tom_md"], 0.5), (2.5, ["tom_md"], 0.5),
             (3.0, ["tom_lo"], 0.5), (3.5, ["tom_lo"], 0.5),
             (0.0, ["bd"], 1.0), (2.0, ["bd"], 1.0)]

GROOVE_PLAN = [
    (1, 8, G_INTRO_A), (9, 16, G_INTRO_B), (17, 17, G_INTRO_B),
    (18, 32, G_VERSE), (33, 48, G_CHORUS), (49, 49, G_VERSE),
    (50, 64, G_VERSE), (65, 80, G_CHORUS), (81, 81, G_CHORUS),
    (82, 85, G_BREAK), (86, 100, G_VERSE3), (101, 116, G_CHORUS3),
    (117, 124, G_OUTRO),
]
# Fills land in the last bar of a section, never inside one.
FILL_HALF_BARS = {8, 16, 32, 48, 57, 64, 80, 92, 100, 108, 116}
FILL_FULL_BARS = {85}
CRASH_BARS = {9, 18, 33, 50, 65, 86, 101, 117}


def groove_for(bar: int) -> list[tuple[float, list[str], float]]:
    base = G_VERSE
    for lo, hi, g in GROOVE_PLAN:
        if lo <= bar <= hi:
            base = g
            break
    if bar in FILL_FULL_BARS:
        base = FILL_FULL
    elif bar in FILL_HALF_BARS:
        base = FILL_HALF
    if bar == SHORT_BAR:                      # the 2/4 bar - keep the first half
        base = [e for e in base if e[0] < 2.0]
    evs = [(o, list(p), q) for o, p, q in base]
    if bar in CRASH_BARS:
        evs = [e for e in evs if not (e[0] == 0.0 and e[1][0] in ("ride", "hh", "bell"))]
        evs.append((0.0, ["crash"], 1.0))
    return evs

# ---------------------------------------------------------------------------
# Horns
# ---------------------------------------------------------------------------
# A written lead for the 8-bar verse, concert pitch.  It outlines the chord of
# each bar and lands on the 3+3+2, so the reeds can state it as a soli in the
# instrumental verse and the brass can punctuate the same shape.
# (offset, quarterLength, concert MIDI)
VERSE_LEAD: list[list[tuple[float, float, str]]] = [
    [(0.0, 1.5, "A4"), (1.5, 1.5, "C5"), (3.0, 1.0, "B4")],      # Am
    [(0.0, 1.5, "A4"), (1.5, 1.5, "G4"), (3.0, 1.0, "A4")],      # Am/G
    [(0.0, 1.5, "F#4"), (1.5, 1.5, "A4"), (3.0, 1.0, "C5")],     # D7/F#
    [(0.0, 2.0, "D5"), (2.0, 2.0, "C5")],                        # D7/F#
    [(0.0, 1.5, "C5"), (1.5, 1.5, "A4"), (3.0, 1.0, "F4")],      # F
    [(0.0, 1.5, "G#4"), (1.5, 1.5, "B4"), (3.0, 1.0, "E5")],     # E7
    [(0.0, 4.0, "A4")],                                          # Am
    [(0.0, 2.0, "B4"), (2.0, 2.0, "G#4")],                       # E7
]
# and for the chorus, a plainer one - the last chorus shouts it
CHORUS_LEAD: list[list[tuple[float, float, str]]] = [
    [(0.0, 1.5, "G4"), (1.5, 1.5, "C5"), (3.0, 1.0, "G4")],      # C
    [(0.0, 1.5, "A4"), (1.5, 1.5, "C5"), (3.0, 1.0, "A4")],      # F
    [(0.0, 1.5, "B4"), (1.5, 1.5, "D5"), (3.0, 1.0, "B4")],      # G
    [(0.0, 2.0, "C5"), (2.0, 2.0, "B4")],                        # C  G/B
    [(0.0, 2.0, "A4"), (2.0, 2.0, "G4")],                        # Am Am/G
    [(0.0, 1.5, "A4"), (1.5, 1.5, "C5"), (3.0, 1.0, "A4")],      # F
    [(0.0, 2.0, "B4"), (2.0, 2.0, "D5")],                        # G
    [(0.0, 4.0, "C5")],                                          # C
]


def voice_busy(bar: int) -> set[float]:
    """Offsets in ``bar`` where the lead vocal is sounding."""
    out: set[float] = set()
    for b, off, ln, _m in VOX:
        if b != bar:
            continue
        for k in range(max(1, ln)):
            out.add((off + k) * 0.5)
    return out


def _mk(off: float, ql: float, midi: int | None, artic: str | None = None):
    return (off, ql, midi, artic)


def _bar_is_free(bar: int) -> bool:
    """True when the singer is resting for most of the bar."""
    return len(voice_busy(bar)) <= 2


def add(plan: dict, bar: int, slot: str, events: list) -> None:
    plan.setdefault(bar, {}).setdefault(slot, []).extend(events)


def pad_bar(plan: dict, bar: int, slots: tuple[str, ...], lead_target: int,
            *, triad: bool = False, artic: str | None = None) -> int:
    """Sustained block voicing, re-struck when the chord changes mid-bar.

    ``lead_target`` is where the top voice wants to be; the nearest chord tone
    to it is taken, so passing a moving target gives the pad a shape."""
    last = lead_target
    for off, sym in changes(bar):
        if sym == "NC":
            continue
        end = bar_len(bar) if off == changes(bar)[-1][0] else bar_len(bar) / 2
        ql = end - off
        tones = TRIAD[sym] if triad else SPEC[sym][1]
        lead = min((nearest(t, last) for t in tones), key=lambda p: abs(p - last))
        voi = section_voicing(lead, sym, slots, triad=triad)
        for slot, p in voi.items():
            add(plan, bar, slot, [_mk(off, ql, p, artic)])
        last = lead
    return last


def hit_bar(plan: dict, bar: int, slots: tuple[str, ...], lead_target: int,
            *, ql: float = 0.5, artic: str = "accent") -> int:
    """Short accented chords on the 3+3+2."""
    last = lead_target
    for off in tresillo(bar):
        sym = chord_at(bar, off)
        if sym == "NC":
            continue
        tones = TRIAD[sym]
        lead = min((nearest(t, last) for t in tones), key=lambda p: abs(p - last))
        for slot, p in section_voicing(lead, sym, slots, triad=True).items():
            add(plan, bar, slot, [_mk(off, ql, p, artic)])
        last = lead
    return last


def line_bar(plan: dict, bar: int, slots: tuple[str, ...],
             line: list[tuple[float, float, str]], *, short: bool = False,
             artic: str | None = None) -> None:
    """Harmonise a written lead across ``slots`` (top voice takes the line).

    With ``short`` the same pitches come out as clipped accented punches - the
    section figure a big band plays on the 3+3+2 - instead of held notes.
    Punches take their contour from the written lead for a reason: choosing
    the nearest chord tone to the previous note instead, which is the obvious
    thing to do, minimises motion so completely that the lead trumpet ends up
    repeating one note for a whole chorus.
    """
    for off, ql, name in line:
        if bar == SHORT_BAR and off >= 2.0:
            continue
        ql = min(ql, bar_len(bar) - off)
        if short:
            ql = min(ql, 0.5)
        if ql <= 0:
            continue
        sym = chord_at(bar, off)
        if sym == "NC":
            continue
        lead = _ps(name)
        voi = section_voicing(lead, sym, slots, triad=short)
        for slot, p in voi.items():
            add(plan, bar, slot, [_mk(off, ql, p, artic)])


# ---------------------------------------------------------------------------
# The arrangement
# ---------------------------------------------------------------------------
VERSE_BLOCKS = [(1, 8), (9, 8), (18, 8), (26, 7), (50, 8), (58, 7),
                (86, 7), (94, 7)]
CHORUS_BLOCKS = [33, 41, 65, 73, 101, 109]


def verse_pos(bar: int) -> int | None:
    """Index 0-7 of ``bar`` within its 8-bar verse pattern, if it is in one."""
    for start, n in VERSE_BLOCKS:
        if start <= bar < start + n:
            return bar - start
    return None


def chorus_pos(bar: int) -> int | None:
    for start in CHORUS_BLOCKS:
        if start <= bar < start + 8:
            return bar - start
    return None


def build_horn_plan() -> dict[int, dict[str, list]]:
    """{bar: {slot: [(offset, ql, midi, artic)]}} for the six horns."""
    plan: dict[int, dict[str, list]] = {}

    # -- Intro B (9-16): reeds only, a quiet sustained pad under the melody
    lead = _ps("A4")
    for bar in range(9, 17):
        lead = pad_bar(plan, bar, REEDS, lead)

    # -- bar 17: brass set up the entry
    line_bar(plan, 17, BRASS, VERSE_LEAD[7], short=True, artic="accent")

    # -- Verse 1 (18-32): trombone and baritone hold the harmony underneath,
    #    alto and tenor answer only where the singer has stopped.
    lead = _ps("A4")
    for bar in range(18, 33):
        lead = pad_bar(plan, bar, ("tbn", "bari"), lead, triad=True)
    for bar in range(18, 33):
        pos = verse_pos(bar)
        if pos is None or not _bar_is_free(bar):
            continue
        if pos in (3, 6, 7):                   # the natural holes in the phrase
            line_bar(plan, bar, ("alto", "tenor"), VERSE_LEAD[pos])

    # -- Chorus 1 (33-48): brass punch the 3+3+2, reeds sustain over the top
    for bar in range(33, 49):
        pos = chorus_pos(bar)
        if pos is not None:
            line_bar(plan, bar, BRASS, CHORUS_LEAD[pos], short=True,
                     artic="accent")
    for bar in range(33, 49):
        pos = chorus_pos(bar)
        if pos is not None:
            pad_bar(plan, bar, REEDS, _ps(CHORUS_LEAD[pos][0][2]) + 5)

    # -- bar 49: reeds lead back into the verse
    line_bar(plan, 49, REEDS, [(2.0, 1.0, "C5"), (3.0, 1.0, "B4")])

    # -- Verse 2 (50-64): first trumpet takes a cup-muted counter-line in the
    #    gaps; the other five stay on a soft pad.
    lead = _ps("A4")
    for bar in range(50, 65):
        lead = pad_bar(plan, bar, ("tbn", "bari", "tenor"), lead, triad=True)
    for bar in range(50, 65):
        pos = verse_pos(bar)
        if pos is None or not _bar_is_free(bar):
            continue
        if pos in (3, 6, 7):
            line_bar(plan, bar, ("tpt1", "alto"), VERSE_LEAD[pos])

    # -- Chorus 2 (65-80): the whole band on the accents
    for bar in range(65, 81):
        pos = chorus_pos(bar)
        if pos is not None:
            line_bar(plan, bar, BRASS, CHORUS_LEAD[pos], short=True,
                     artic="accent")
            line_bar(plan, bar, REEDS, CHORUS_LEAD[pos], short=True,
                     artic="accent")

    # -- bar 81 one tutti chord, then the band drops out for the break
    pad_bar(plan, 81, ALL_HORNS, _ps("E5"), triad=True, artic="accent")

    # -- Verse 3 (86-100) is instrumental on the record: give it to the reeds
    #    as a soli, with the brass answering at the end of each phrase.
    for bar in range(86, 101):
        pos = verse_pos(bar)
        if pos is not None:
            line_bar(plan, bar, REEDS, VERSE_LEAD[pos])
    for bar in (92, 100):
        line_bar(plan, bar, BRASS, VERSE_LEAD[6], short=True, artic="accent")

    # -- Chorus 3 (101-116): shout chorus.  Brass carry the lead, reeds under.
    for bar in range(101, 117):
        pos = chorus_pos(bar)
        if pos is None:
            continue
        line_bar(plan, bar, BRASS, CHORUS_LEAD[pos], artic="accent")
        line_bar(plan, bar, REEDS, CHORUS_LEAD[pos])

    # -- Outro (117-124): sustained, thinning out, one last chord
    lead = _ps("E5")
    for bar in range(117, 123):
        lead = pad_bar(plan, bar, ALL_HORNS, lead, triad=True)
    pad_bar(plan, 123, ALL_HORNS, lead, triad=True)
    pad_bar(plan, 124, ALL_HORNS, lead, triad=True)
    smooth_octaves(plan)
    return plan


def smooth_octaves(plan: dict[int, dict[str, list]]) -> None:
    """Put every horn note in the octave nearest the one that part just played.

    Fitting each voice into its range independently can leave a part leaping
    two octaves between one chord and the next.  Every candidate here is the
    same pitch class, so the harmony is untouched; only the octave moves, and
    only within that instrument's range.
    """
    for slot in ALL_HORNS:
        lo, hi = CONCERT_RANGE[slot]
        prev: int | None = None
        for bar in sorted(plan):
            evs = plan[bar].get(slot)
            if not evs:
                continue
            evs.sort()
            for i, (off, ql, midi, artic) in enumerate(evs):
                if midi is None:
                    continue
                cands = [p for p in (midi - 24, midi - 12, midi, midi + 12,
                                     midi + 24) if lo <= p <= hi]
                if not cands:
                    cands = [midi]
                best = cands[0] if prev is None else min(
                    cands, key=lambda p: (abs(p - prev), abs(p - midi)))
                evs[i] = (off, ql, best, artic)
                prev = best


HORN_DYNAMICS = {9: "mp", 17: "mf", 18: "pp", 33: "mf", 50: "pp", 65: "f",
                 81: "ff", 86: "mf", 101: "ff", 117: "mf", 121: "mp"}
PART_TEXT = {17: None, 50: "cup mute (Tpt. 1)", 86: "soli", 101: "shout"}

# ---------------------------------------------------------------------------
# Score assembly
# ---------------------------------------------------------------------------
DRUM_UP = {"crash", "hh", "ride", "bell", "tom_hi", "tom_md", "snare",
           "xstick", "tom_lo"}
KEY_SHARPS = 0                       # A minor / C major


def _note(midi: int, ql: float, artic: str | None = None) -> note.Note:
    n = note.Note()
    n.pitch.ps = midi
    n.pitch.spellingIsInferred = True
    n.quarterLength = ql
    if artic:
        n.articulations = [{"accent": articulations.Accent(),
                            "staccato": articulations.Staccato(),
                            "marcato": articulations.StrongAccent()}[artic]]
    return n


def _respell(p: pitch.Pitch) -> pitch.Pitch:
    """Sharps for the leading tones this key actually uses, flats otherwise."""
    if p.accidental is not None and p.accidental.alter > 0 and p.name not in (
            "F#", "G#", "C#", "D#"):
        return p.getEnharmonic()
    return p


def _sharpen(p: pitch.Pitch) -> pitch.Pitch | None:
    """The sharp spelling of a flat-spelled pitch, or None to leave it alone."""
    if p.accidental is None or p.accidental.alter >= 0:
        return None
    alt = p.getEnharmonic()
    if alt.accidental is None or alt.accidental.alter in (0, 1):
        return alt
    return None


def new_part(name: str, abbrev: str, instr, *, treble: bool = True) -> stream.Part:
    p = stream.Part()
    p.id = name
    p.partName = name
    p.partAbbreviation = abbrev
    ins = instr()
    ins.partName = name
    ins.partAbbreviation = abbrev
    p.insert(0, ins)
    p.insert(0, clef.TrebleClef() if treble else clef.BassClef())
    return p


def new_measure(bar: int, *, first: bool, perc: bool = False,
                tempo_mark: bool = False) -> stream.Measure:
    m = stream.Measure(number=bar)
    if first:
        if not perc:
            m.insert(0.0, key.KeySignature(KEY_SHARPS))
        m.insert(0.0, meter.TimeSignature("4/4"))
        if tempo_mark:
            mk = tempo.MetronomeMark(number=BEAT, referent=duration.Duration(1.0))
            mk.placement = "above"
            m.insert(0.0, mk)
    if bar == SHORT_BAR:
        m.insert(0.0, meter.TimeSignature("2/4"))
    elif bar == SHORT_BAR + 1:
        m.insert(0.0, meter.TimeSignature("4/4"))
    return m


def _rest_run(m: stream.Measure, start: float, length: float) -> None:
    """Fill ``length`` from ``start`` with rests of legal, readable values."""
    cur, left = start, length
    while left > 1e-9:
        for val in (4.0, 2.0, 1.0, 0.5, 0.25):
            if val <= left + 1e-9 and abs(cur / val - round(cur / val)) < 1e-9:
                m.insert(cur, note.Rest(quarterLength=val))
                cur += val
                left -= val
                break
        else:
            m.insert(cur, note.Rest(quarterLength=left))
            left = 0.0


def finish(m: stream.Measure, bar: int) -> stream.Measure:
    """Pad a measure to its exact length.

    Done by hand rather than with ``makeRests``: at this point the measure is
    not yet inside a part, so it has no time signature to infer its length
    from, and a 2/4 bar sitting in a 4/4 part comes out wrong.
    """
    total = bar_len(bar)
    spans: list[list[float]] = []
    for n in list(m.notes):
        a = float(n.offset)
        b = a + float(n.quarterLength)
        if spans and a <= spans[-1][1] + 1e-9:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    spans.sort()
    if not spans:
        r = note.Rest(quarterLength=total)
        r.fullMeasure = True
        m.insert(0.0, r)
        return m
    cur = 0.0
    for a, b in spans:
        if a - cur > 1e-9:
            _rest_run(m, cur, a - cur)
        cur = max(cur, b)
    if total - cur > 1e-9:
        _rest_run(m, cur, total - cur)
    return m


def add_symbols(m: stream.Measure, bar: int) -> None:
    """Chord symbols, on the concert-pitch staves only."""
    for off, sym in changes(bar):
        if sym == "NC":
            cs = harmony.NoChord()
        else:
            cs = harmony.ChordSymbol(SPEC[sym][3])
        cs.writeAsChord = False
        cs.quarterLength = 0.0
        m.insert(off, cs)


def add_marks(m: stream.Measure, bar: int) -> None:
    if bar == 1:
        te = expressions.TextExpression("Bolero - rubato")
        te.placement = "above"
        te.style.fontStyle = "italic"
        m.insert(0.0, te)
    if bar in REHEARSAL:
        r = expressions.RehearsalMark(REHEARSAL[bar])
        m.insert(0.0, r)
    for lo, hi, label in SECTIONS:
        if bar == lo and label:
            te = expressions.TextExpression(label)
            te.placement = "above"
            m.insert(0.0, te)

# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
HORN_PARTS = [
    ("alto", "Alto Sax", "A. Sx.", instrument.AltoSaxophone, True),
    ("tenor", "Tenor Sax", "T. Sx.", instrument.TenorSaxophone, True),
    ("bari", "Baritone Sax", "B. Sx.", instrument.BaritoneSaxophone, True),
    ("tpt1", "Trumpet 1", "Tpt. 1", instrument.Trumpet, True),
    ("tpt2", "Trumpet 2", "Tpt. 2", instrument.Trumpet, True),
    ("tbn", "Trombone", "Tbn.", instrument.Trombone, False),
]


def build_voice() -> stream.Part:
    """The transcribed lead vocal.

    Isolated octave slips - pYIN reporting a note an octave off mid-phrase -
    are pulled back to the octave nearest their neighbours; nothing else about
    the pitch is changed.
    """
    p = new_part("Voice", "Vox", instrument.Vocalist)
    p.replace(p.getElementsByClass(clef.Clef)[0], clef.Treble8vbClef())
    by_bar: dict[int, list] = {}
    flat: list[tuple[int, float, float, int]] = []
    for bar, off, ln, midi in VOX:
        if not 1 <= bar <= TOTAL_BARS:
            continue
        room = bar_len(bar) - off * 0.5
        if room <= 0:
            continue
        flat.append((bar, off * 0.5, min(ln * 0.5, room), midi))
    flat.sort()
    for i, (bar, off, ql, midi) in enumerate(flat):
        if 0 < i < len(flat) - 1:
            prv, nxt = flat[i - 1][3], flat[i + 1][3]
            # only an isolated slip: a note stranded more than an octave from
            # both neighbours.  Correcting every note would flatten the tune.
            if abs(midi - prv) > 12 and abs(midi - nxt) > 12:
                midi = min((midi - 12, midi, midi + 12),
                           key=lambda q: abs(q - prv) + abs(q - nxt))
                flat[i] = (bar, off, ql, midi)
        by_bar.setdefault(bar, []).append([off, ql, midi])
    # pYIN drops out briefly inside a held note; close gaps of one eighth so a
    # sustained syllable reads as one note instead of two with a rest between
    for i in range(len(flat) - 1):
        b0, o0, q0, _m0 = flat[i]
        b1, o1, _q1, _m1 = flat[i + 1]
        if b0 == b1 and o1 - (o0 + q0) - 0.5 < 1e-9 and o1 > o0 + q0:
            for ev in by_bar[b0]:
                if abs(ev[0] - o0) < 1e-9:
                    ev[1] = o1 - o0
    for bar in range(1, TOTAL_BARS + 1):
        m = new_measure(bar, first=(bar == 1), tempo_mark=True)
        evs = sorted(by_bar.get(bar, []))
        for i, (off, ql, midi) in enumerate(evs):        # keep it monophonic
            if i + 1 < len(evs) and off + ql > evs[i + 1][0]:
                ql = evs[i + 1][0] - off
            if ql <= 0:
                continue
            n = _note(midi, ql)
            n.pitch = _respell(n.pitch)
            m.insert(off, n)
        add_marks(m, bar)
        p.append(finish(m, bar))
    return p


def build_horn(slot: str, name: str, abbrev: str, instr, treble: bool,
               plan: dict) -> stream.Part:
    p = new_part(name, abbrev, instr, treble=treble)
    for bar in range(1, TOTAL_BARS + 1):
        m = new_measure(bar, first=(bar == 1))
        evs = sorted(plan.get(bar, {}).get(slot, []))
        placed: list[tuple[float, float]] = []
        for off, ql, midi, artic in evs:
            if midi is None or ql <= 0:
                continue
            if any(off < b and off + ql > a for a, b in placed):
                continue
            ql = min(ql, bar_len(bar) - off)
            if ql <= 0:
                continue
            n = _note(midi, ql, artic)
            n.pitch = _respell(n.pitch)
            m.insert(off, n)
            placed.append((off, off + ql))
        if bar in HORN_DYNAMICS:
            m.insert(0.0, dynamics.Dynamic(HORN_DYNAMICS[bar]))
        p.append(finish(m, bar))
    return p


LH_VARIANT = (0, 1, 0, 2, 1, 0, 3, 1)
# a short right-hand line at the end of a section, instead of another chord
PIANO_FILLS = {16, 32, 48, 64, 80, 100, 116}


def piano_fill(sym: str, top: int | None) -> list[tuple[float, float, int]]:
    """Four descending eighths off the top of the voicing, into the next bar."""
    voi = evans_voicing(sym, top)
    if len(voi) < 4:
        return []
    line = sorted(voi, reverse=True)
    return [(2.0 + i * 0.5, 0.5, line[i]) for i in range(4)]


def build_piano() -> tuple[stream.PartStaff, stream.PartStaff, layout.StaffGroup]:
    rh = stream.PartStaff()
    rh.id = "PianoRH"
    lh = stream.PartStaff()
    lh.id = "PianoLH"
    ins = instrument.Piano()
    ins.partName = "Piano"
    ins.partAbbreviation = "Pno."
    rh.insert(0, ins)
    rh.insert(0, clef.TrebleClef())
    lh.insert(0, instrument.Piano())
    lh.insert(0, clef.BassClef())
    rh.partName = "Piano"
    rh.partAbbreviation = "Pno."
    top: int | None = None
    for bar in range(1, TOTAL_BARS + 1):
        mr = new_measure(bar, first=(bar == 1))
        ml = new_measure(bar, first=(bar == 1))
        add_symbols(mr, bar)
        chs = changes(bar)
        total = bar_len(bar)

        # right hand
        if bar in PIANO_FILLS:
            sym = chord_at(bar, 2.0)
            for off, ql, midi in piano_fill(sym, top):
                n = _note(midi, ql)
                n.pitch = _respell(n.pitch)
                mr.insert(off, n)
            first = chs[0][1]
            voi = evans_voicing(first, top)
            if voi:
                c = chord.Chord([_note(x, 2.0) for x in voi])
                c.quarterLength = 2.0
                c.pitches = tuple(_respell(x) for x in c.pitches)
                mr.insert(0.0, c)
                top = voi[-1]
        else:
            for off, ql in comp_rhythm(bar):
                sym = chord_at(bar, off)
                if sym == "NC":
                    continue
                nxt = [o for o, _s in chs if o > off]
                ql = min(ql, (nxt[0] if nxt else total) - off)
                voi = evans_voicing(sym, top)
                if ql <= 0 or not voi:
                    continue
                c = chord.Chord([_note(x, ql) for x in voi])
                c.quarterLength = ql
                c.pitches = tuple(_respell(x) for x in c.pitches)
                mr.insert(off, c)
                top = voi[-1]

        # left hand: one shell per chord, sustained, and sometimes tacet
        var = LH_VARIANT[(bar - 1) % len(LH_VARIANT)]
        for i, (off, sym) in enumerate(chs):
            if sym == "NC":
                continue
            end = chs[i + 1][0] if i + 1 < len(chs) else total
            notes = left_hand(sym, var)
            if not notes or end - off <= 0:
                continue
            c2 = chord.Chord([_note(x, end - off) for x in notes])
            c2.quarterLength = end - off
            c2.pitches = tuple(_respell(x) for x in c2.pitches)
            ml.insert(off, c2)
        rh.append(finish(mr, bar))
        lh.append(finish(ml, bar))
    grp = layout.StaffGroup([rh, lh], name="Piano", abbreviation="Pno.",
                            symbol="brace")
    grp.barTogether = True
    return rh, lh, grp


def build_bass() -> stream.Part:
    p = new_part("Bass", "Bass", instrument.AcousticBass, treble=False)
    for bar in range(1, TOTAL_BARS + 1):
        m = new_measure(bar, first=(bar == 1))
        add_symbols(m, bar)
        # tacet where the band drops out: the tracker still reports notes
        # through the percussion break, but that is decay and bleed - the bass
        # itself is 20-45 dB down there.
        evs = [] if chord_at(bar, 0.0) == "NC" else bass_for(bar)
        for off, ql, midi in evs:
            ql = min(ql, bar_len(bar) - off)
            if ql <= 0:
                continue
            n = _note(midi, ql)
            n.pitch = _respell(n.pitch)
            m.insert(off, n)
        p.append(finish(m, bar))
    return p


def build_drums() -> stream.Part:
    """Two voices: stems up for cymbals, snare and toms, stems down for feet.

    Everything landing on one offset inside a voice becomes a single
    percussion chord, and its length runs to the next attack in that voice -
    two notes of different lengths cannot share an offset in one voice, and
    MusicXML would otherwise serialise them one after the other.
    """
    from music21 import percussion
    p = stream.Part()
    p.id = "Drums"
    p.partName = "Drums"
    p.partAbbreviation = "Dr."
    ins = instrument.Percussion()
    ins.partName = "Drums"
    ins.partAbbreviation = "Dr."
    p.insert(0, ins)
    p.insert(0, clef.PercussionClef())
    for bar in range(1, TOTAL_BARS + 1):
        m = new_measure(bar, first=(bar == 1), perc=True)
        evs = groove_for(bar)
        total = bar_len(bar)
        for voice_no, up in ((1, True), (2, False)):
            at: dict[float, list[str]] = {}
            dur: dict[float, float] = {}
            for off, pieces, ql in evs:
                sel = [x for x in pieces if (x in DRUM_UP) == up]
                if not sel or off >= total:
                    continue
                for x in sel:
                    if x not in at.setdefault(off, []):
                        at[off].append(x)
                dur[off] = min(dur.get(off, ql), ql)
            if not at:
                continue
            v = stream.Voice()
            v.id = str(voice_no)
            offs = sorted(at)
            cursor = 0.0
            for i, off in enumerate(offs):
                nxt = offs[i + 1] if i + 1 < len(offs) else total
                # a drum is struck, not sustained: keep the written value the
                # groove asks for and fill the rest of the gap with rests,
                # rather than stretching a kick into a tied half note
                ql = min(dur.get(off, 0.5), nxt - off)
                if ql <= 0:
                    continue
                if off - cursor > 1e-9:
                    _rest_run(v, cursor, off - cursor)
                objs = []
                for piece in at[off]:
                    disp, head, _midi = KIT[piece]
                    u = note.Unpitched(displayName=disp)
                    u.storedInstrument = instrument.UnpitchedPercussion()
                    if head:
                        u.notehead = head
                    objs.append(u)
                ob = objs[0] if len(objs) == 1 else percussion.PercussionChord(objs)
                ob.quarterLength = ql
                ob.stemDirection = "up" if up else "down"
                v.insert(off, ob)
                cursor = off + ql
            if total - cursor > 1e-9:
                _rest_run(v, cursor, total - cursor)
            m.insert(0.0, v)
        if not len(m.voices):
            r = note.Rest(quarterLength=total)
            r.fullMeasure = True
            m.insert(0.0, r)
        p.append(m)
    return p


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
TITLE = "Never Gonna Fall in Love Again"
SUBTITLE = "10-piece big band"


def build_score() -> stream.Score:
    plan = build_horn_plan()
    sc = stream.Score()
    sc.insert(0, metadata_block())
    sc.insert(0, build_voice())
    for slot, name, abbrev, instr, treble in HORN_PARTS:
        sc.insert(0, build_horn(slot, name, abbrev, instr, treble, plan))
    rh, lh, grp = build_piano()
    sc.insert(0, rh)
    sc.insert(0, lh)
    sc.insert(0, grp)
    sc.insert(0, build_bass())
    sc.insert(0, build_drums())
    sc.atSoundingPitch = True
    for part in sc.parts:
        part.atSoundingPitch = True
    sc.toWrittenPitch(inPlace=True)
    for part in sc.parts:
        part.makeBeams(inPlace=True, failOnNoTimeSignature=False)
        # after transposition each part carries its own key signature (three
        # sharps for the E-flat horns, two for the B-flat ones), so accidentals
        # have to be resolved against that, not against the concert key
        ks = None
        for el in part.recurse().getElementsByClass(key.KeySignature):
            ks = el
            break
        # a part written in sharps should not show flats: transposing concert
        # F-sharp for an E-flat horn gives D-sharp, not E-flat
        if ks is not None and ks.sharps > 0:
            for n in list(part.recurse().notes):
                if isinstance(n, note.Note):
                    alt = _sharpen(n.pitch)
                    if alt is not None:
                        n.pitch = alt
                elif isinstance(n, chord.Chord):
                    n.pitches = tuple(_sharpen(x) or x for x in n.pitches)
        # Accidentals are resolved measure by measure, carrying the previous
        # bar's pitches forward.  Part.makeAccidentals only accepts a list of
        # altered pitches, and feeding it a sharp key signature's octave-less
        # pitches sends it into a runaway; Measure.makeAccidentals takes the
        # key signature itself and behaves.
        ks = ks if ks is not None else key.KeySignature(KEY_SHARPS)
        past: list[pitch.Pitch] = []
        for meas in part.getElementsByClass(stream.Measure):
            meas.makeAccidentals(useKeySignature=ks, pitchPastMeasure=past,
                                 inPlace=True, overrideStatus=True)
            past = [q for n in meas.recurse().notes
                    for q in (n.pitches if hasattr(n, "pitches") else [n.pitch])]
    return sc


def metadata_block():
    from music21 import metadata as md
    m = md.Metadata()
    m.title = TITLE
    m.movementName = TITLE
    m.composer = "arr. for 10-piece big band"
    return m


def _write_drumset(root: ET.Element) -> None:
    """Give the drum part a real drum set in the part list.

    music21 exports the staff position of each unpitched note correctly but
    declares only one instrument for the whole part, so a reader maps every
    note to that one sound and collapses the staff onto a single line.  Here
    the part gets one score-instrument per kit piece actually used, and every
    note gets an instrument reference, which is what makes the kick, snare,
    hi-hat and cymbals land on their own lines and play back as themselves.
    """
    pid = None
    for sp in root.findall("part-list/score-part"):
        nm = sp.find("part-name")
        if nm is not None and (nm.text or "") == "Drums":
            pid = sp.get("id")
            break
    if pid is None:
        return
    part = next((p for p in root.findall("part") if p.get("id") == pid), None)
    if part is None:
        return

    def look(n: ET.Element) -> tuple[str, int] | None:
        up = n.find("unpitched")
        if up is None:
            return None
        step = up.findtext("display-step")
        octv = up.findtext("display-octave")
        head = n.findtext("notehead")
        return BY_LOOK.get((f"{step}{octv}", head))

    used: dict[int, str] = {}
    for n in part.iter("note"):
        hit = look(n)
        if hit:
            used[hit[1]] = hit[0]

    sp = next(p for p in root.findall("part-list/score-part") if p.get("id") == pid)
    for child in list(sp):
        if child.tag in ("score-instrument", "midi-instrument"):
            sp.remove(child)
    ids = {}
    for midi in sorted(used):
        iid = f"{pid}-D{midi}"
        ids[midi] = iid
        si = ET.SubElement(sp, "score-instrument", {"id": iid})
        ET.SubElement(si, "instrument-name").text = GM_NAME.get(midi, used[midi])
    for midi in sorted(used):
        mi = ET.SubElement(sp, "midi-instrument", {"id": ids[midi]})
        ET.SubElement(mi, "midi-channel").text = "10"
        # MusicXML numbers unpitched sounds 1-128 against MIDI 0-127
        ET.SubElement(mi, "midi-unpitched").text = str(midi + 1)
        ET.SubElement(mi, "volume").text = "80"
        ET.SubElement(mi, "pan").text = "0"

    for n in part.iter("note"):
        hit = look(n)
        if not hit:
            continue
        for old in n.findall("instrument"):
            n.remove(old)
        ref = ET.Element("instrument", {"id": ids[hit[1]]})
        kids = list(n)
        at = 0
        for i, k in enumerate(kids):
            if k.tag in ("duration", "tie"):
                at = i + 1
        n.insert(at, ref)


def tidy_musicxml(path: str) -> None:
    """Post-export fixes music21 will not make itself.

    * a 2/4 bar exported inside a 4/4 part sometimes keeps a stale
      ``<divisions>`` sibling ordering, so re-sort ``<attributes>``
    * strip the duplicate time signature music21 writes on the staff that
      already carries one
    """
    ORDER = ["divisions", "key", "time", "staves", "part-symbol", "instruments",
             "clef", "staff-details", "transpose", "directive", "measure-style"]
    tree = ET.parse(path)
    root = tree.getroot()
    _write_drumset(root)
    for tag in ("root", "bass"):
        for el in root.iter(tag):
            for alt in list(el):
                if alt.tag.endswith("-alter") and alt.text in ("0", "0.0"):
                    el.remove(alt)
    for attrs in root.iter("attributes"):
        kids = list(attrs)
        kids.sort(key=lambda e: ORDER.index(e.tag) if e.tag in ORDER else 99)
        for k in list(attrs):
            attrs.remove(k)
        for k in kids:
            attrs.append(k)
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def write_mxl(xml_path: str, mxl_path: str) -> None:
    """Zip the finished MusicXML into a .mxl container.

    music21's own mxl writer consumes the .musicxml file and would package the
    version from before ``tidy_musicxml`` ran, so the archive is built here
    from the file that was actually corrected.
    """
    import os
    import zipfile

    name = os.path.basename(xml_path)
    container = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<container><rootfiles>"
        f'<rootfile full-path="{name}" '
        'media-type="application/vnd.recordare.musicxml+xml"/>'
        "</rootfiles></container>\n"
    )
    with zipfile.ZipFile(mxl_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/vnd.recordare.musicxml",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container)
        z.write(xml_path, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=f"{TITLE} - 10-piece Big Band.musicxml")
    args = ap.parse_args()
    sc = build_score()
    sc.write("musicxml", fp=args.out)
    tidy_musicxml(args.out)
    print(f"wrote {args.out}")
    mxl = args.out.rsplit(".", 1)[0] + ".mxl"
    write_mxl(args.out, mxl)
    print(f"wrote {mxl}")


if __name__ == "__main__":
    main()
