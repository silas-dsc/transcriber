#!/usr/bin/env python3
"""Every Little Thing - the piano part, transcribed from the recording.

A grand staff of what the pianist plays.  The recording is piano and voice
and nothing else, so the piano is the whole accompaniment.

**The tempo moves, so this is quantised to intent rather than to the clock.**
Notating a rubato performance to the millisecond produces rhythms nobody
played and nobody can read.  What is written here is the eighth-note grid the
performance sits on, with the grid itself following the tempo.

How it was pulled out:

* Demucs' six-source model separates the piano from the voice.  The split is
  clean: the piano stem is RMS 0.029 against 0.0001 for bass, drums and
  guitar, which is the noise floor - there are no other instruments to leak.
* Basic Pitch (Spotify's ICASSP-2022 polyphonic model) reads the stem as
  **1438 notes**, almost all of them between C2 and B4.
* Notes whose grid positions are within half an eighth are one event, so a
  spread chord is one chord and not five.  That matters here: the very first
  gesture is two notes 58 ms apart, which is a roll, not two beats.

**The metre and tempo were measured, not assumed.**  The autocorrelation of
the onset envelope has a clean duple hierarchy at 0.226, 0.447, 0.923 and
1.869 s, and the harmonic-change curve - how fast the chords turn over -
peaks at 0.906, 1.834, 3.738 and 7.338 s.  Those are a half bar, a bar, two
bars and a four-bar phrase of 4/4 with an eighth of about 0.23 s.  The key is
**G major**, which the Krumhansl-Schmuckler profile gives at 0.94 against
0.65 for the next candidate.

The grid is a piecewise-linear map from audio time to eighth-note position,
with a knot every 8 seconds, fitted by coordinate descent so that strikes
land on integers.  Fitted that way the tempo runs **122 to 136 BPM around a
median of 131**, and the drift is real: a single fixed tempo fits measurably
worse than the moving one.

**How exact this can honestly be.**  On the fitted grid, 66% of chords land
within a quarter of an eighth of their slot, against 50% for random times,
and locally - over any twenty-second window - it is about 70%.  That is a
real pulse but a loose one, and it is the reason this transcription stops at
eighth notes.  Sixteenths would be inventing detail the performance does not
support.  The score says *rubato* because the recording is.

**One thing that could not be measured: which beat is beat 1.**  Five
independent tests - bass-root positions modulo the bar, harmonic-change
energy against each candidate offset, and three others - all came back at
chance.  The reason is arithmetic: the residual is 0.19 of an eighth, about
44 ms, and an eighth here is only 229 ms, so nothing survives being counted
modulo eight.  The bar lines are therefore anchored musically, at the
opening, where the evidence is local and clean: the piece begins on a G, the
bass moves to A exactly eight eighths later and to C eight after that.  That
is a choice, defensible and stated, rather than a measurement.

    python every_little_thing/every_little_thing_piano.py -o Piano.musicxml
"""

from __future__ import annotations

import argparse
import os
import zipfile

from music21 import (bar, chord, clef, expressions, key, layout, metadata,
                     meter, note, pitch, stream, tempo, tie)

BARS = 109
EIGHTH = 0.5                      # a grid step is a written eighth note

# --------------------------------------------------------------------------
# (position in eighths from bar 1 beat 1, length in eighths, concert MIDI)
# --------------------------------------------------------------------------
LEFT: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (0, 3, (43,)), (3, 5, (43, 55)), (8, 3, (45,)),
    (11, 4, (45, 57)), (15, 4, (48, 52, 55, 57)), (19, 9, (36,)),
    (28, 3, (55,)), (31, 8, (43, 50, 55)), (39, 2, (45, 52)),
    (41, 1, (55,)), (42, 5, (52, 55)), (47, 2, (48, 52, 55)),
    (49, 9, (36,)), (58, 4, (36,)), (62, 1, (52,)),
    (63, 8, (43, 50, 55)), (71, 9, (45, 52, 55, 57)), (80, 1, (36,)),
    (81, 9, (48, 55)), (90, 5, (48, 55)), (95, 1, (50,)),
    (96, 1, (43, 50)), (97, 6, (55,)), (103, 1, (50,)),
    (104, 8, (45, 55, 57)), (112, 9, (36, 45, 48)), (121, 2, (55,)),
    (123, 4, (48, 55)), (127, 8, (36, 48, 55)), (135, 4, (52, 55)),
    (139, 2, (48, 52)), (141, 1, (55,)), (142, 2, (43, 55)),
    (144, 6, (50, 55)), (150, 1, (42,)), (151, 1, (50,)),
    (152, 1, (54,)), (153, 5, (50, 54)), (158, 1, (43,)),
    (159, 1, (50,)), (160, 2, (55,)), (162, 2, (55, 62)),
    (164, 1, (50, 55)), (165, 1, (42,)), (166, 1, (50,)),
    (167, 6, (54,)), (173, 1, (55,)), (174, 3, (52,)),
    (177, 2, (55, 59)), (179, 2, (52,)), (181, 3, (43, 50, 55)),
    (184, 3, (43, 50, 55, 59)), (187, 1, (55, 57)), (188, 5, (45, 57)),
    (193, 2, (45, 52)), (195, 5, (52, 57)), (200, 5, (45, 47, 57)),
    (205, 10, (38, 50, 55)), (215, 1, (55,)), (216, 1, (50,)),
    (217, 2, (55, 57)), (219, 1, (50, 57)), (220, 8, (43, 50, 55)),
    (228, 9, (45, 52, 55, 57)), (237, 1, (36, 43, 55)), (238, 6, (48, 55)),
    (244, 7, (43, 48, 55)), (251, 1, (50, 55)), (252, 6, (43,)),
    (258, 1, (55,)), (259, 1, (52, 55)), (260, 3, (45, 52)),
    (263, 1, (45, 55)), (264, 1, (52,)), (265, 1, (55, 57)),
    (266, 1, (52, 55)), (267, 11, (36, 43, 48, 52, 55)), (278, 1, (55,)),
    (279, 1, (48,)), (280, 1, (55,)), (281, 1, (48,)),
    (282, 1, (55,)), (283, 2, (36, 48)), (285, 5, (55,)),
    (290, 1, (48, 55)), (291, 5, (52, 55)), (296, 3, (48,)),
    (299, 1, (43,)), (300, 1, (50,)), (301, 1, (55,)),
    (302, 1, (50,)), (303, 4, (55,)), (307, 1, (42,)),
    (308, 1, (50,)), (309, 6, (54,)), (315, 2, (43,)),
    (317, 5, (50, 55)), (322, 2, (42, 50)), (324, 6, (54,)),
    (330, 1, (55,)), (331, 3, (52,)), (334, 5, (55,)),
    (339, 2, (50,)), (341, 4, (50, 55, 59)), (345, 9, (45, 52, 55)),
    (354, 2, (50,)), (356, 1, (45, 47, 50)), (357, 1, (50, 55)),
    (358, 1, (45,)), (359, 2, (45, 55)), (361, 2, (38, 55)),
    (363, 3, (50,)), (366, 2, (47,)), (368, 6, (50,)),
    (374, 2, (50, 55)), (376, 2, (50, 55, 57)), (378, 1, (36, 48)),
    (379, 3, (55,)), (382, 3, (48, 55)), (385, 5, (38, 55)),
    (390, 2, (50, 55)), (392, 8, (40, 52)), (400, 1, (55,)),
    (401, 8, (52, 55)), (409, 1, (38,)), (410, 7, (36, 48, 55)),
    (417, 1, (38, 50)), (418, 2, (55,)), (420, 5, (50, 55)),
    (425, 9, (43, 50, 55)), (434, 4, (43, 48, 52, 55)), (438, 4, (43, 50, 55)),
    (442, 1, (36, 55)), (443, 7, (48, 55)), (450, 1, (38, 55)),
    (451, 3, (50, 55)), (454, 3, (50, 55, 57)), (457, 1, (52,)),
    (458, 4, (40,)), (462, 4, (52,)), (466, 2, (50,)),
    (468, 2, (50, 57)), (470, 2, (50, 54)), (472, 2, (55,)),
    (474, 1, (36, 55)), (475, 1, (48,)), (476, 1, (55,)),
    (477, 1, (36,)), (478, 3, (36, 48, 55)), (481, 8, (38, 50, 55)),
    (489, 6, (40, 55)), (495, 2, (52,)), (497, 4, (40, 57)),
    (501, 4, (40, 52, 57)), (505, 1, (50,)), (506, 1, (55,)),
    (507, 1, (36, 55)), (508, 1, (48, 55)), (509, 2, (36, 48, 55)),
    (511, 7, (38, 50, 55)), (518, 6, (36, 48, 55)), (524, 6, (38, 50, 55, 57)),
    (530, 3, (40,)), (533, 4, (52,)), (537, 2, (42, 50, 54)),
    (539, 2, (57,)), (541, 3, (50, 54)), (544, 3, (43, 50, 55)),
    (547, 2, (45, 57)), (549, 2, (45, 47)), (551, 5, (48,)),
    (556, 4, (50,)), (560, 3, (52,)), (563, 4, (52, 59)),
    (567, 7, (54,)), (574, 2, (55,)), (576, 8, (55, 57, 59)),
    (584, 16, (48,)), (600, 7, (55,)), (607, 6, (43,)),
    (613, 1, (43, 50)), (614, 8, (43, 50, 55)), (622, 4, (45,)),
    (626, 1, (45, 57)), (627, 2, (50,)), (629, 1, (45,)),
    (630, 11, (36, 48, 52, 55)), (641, 5, (48, 55, 60)), (646, 8, (43, 50, 55)),
    (654, 2, (45, 52)), (656, 2, (50, 55)), (658, 4, (52, 55)),
    (662, 7, (48, 52, 55)), (669, 3, (43,)), (672, 1, (55,)),
    (673, 1, (48, 55, 60)), (674, 3, (52, 55)), (677, 1, (43,)),
    (678, 8, (43, 50, 55)), (686, 7, (45, 57)), (693, 1, (45, 48)),
    (694, 2, (36,)), (696, 2, (55,)), (698, 9, (36, 48, 55)),
    (707, 2, (48, 55)), (709, 1, (50,)), (710, 4, (43, 55)),
    (714, 2, (50, 55)), (716, 2, (43,)), (718, 7, (45, 52, 55, 57)),
    (725, 1, (48, 55)), (726, 1, (36,)), (727, 9, (55,)),
    (736, 1, (48, 55)), (737, 1, (48, 55, 60)), (738, 1, (43,)),
    (739, 1, (55,)), (740, 1, (48, 55)), (741, 1, (36, 50)),
    (742, 2, (43, 50, 55)), (744, 1, (31,)), (745, 1, (43,)),
    (746, 1, (50,)), (747, 1, (43, 55)), (748, 2, (57,)),
    (750, 1, (45, 55, 57)), (751, 1, (52,)), (752, 2, (57,)),
    (754, 3, (45, 52, 55, 57)), (757, 1, (48, 55)), (758, 10, (36, 43, 48, 52, 55)),
    (768, 1, (48,)), (769, 2, (48, 55)), (771, 1, (36, 55)),
    (772, 1, (48, 55)), (773, 2, (50, 55)), (775, 1, (43, 55)),
    (776, 7, (50,)), (783, 8, (45, 52, 55, 57)), (791, 1, (36, 43, 55)),
    (792, 2, (48, 55)), (794, 2, (43,)), (796, 1, (55,)),
    (797, 3, (43,)), (800, 1, (36, 48)), (801, 3, (55,)),
    (804, 2, (47,)), (806, 2, (48, 55)), (808, 9, (43, 50, 55)),
    (817, 1, (50,)), (818, 1, (43, 50)), (819, 2, (55, 57)),
    (821, 1, (43, 55)), (822, 1, (38,)), (823, 2, (54,)),
    (825, 1, (50,)), (826, 1, (38,)), (827, 2, (38, 43, 45, 47, 50)),
    (829, 6, (31,)), (835, 1, (50,)), (836, 1, (31, 38, 43, 47)),
    (837, 1, (55,)), (838, 1, (43,)), (839, 4, (55,)),
    (843, 1, (31,)), (844, 9, (43, 50)), (853, 4, (43,)),
    (857, 5, (31,)), (862, 4, (43,)), (866, 2, (50,)),
    (868, 4, (43, 50)),
)

RIGHT: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (0, 3, (78,)), (3, 6, (74, 78, 86)), (9, 1, (79,)),
    (10, 1, (78,)), (11, 4, (74,)), (15, 16, (60, 67)),
    (31, 1, (78,)), (32, 1, (79,)), (33, 1, (78,)),
    (34, 1, (62, 74)), (35, 5, (78,)), (40, 1, (79,)),
    (41, 1, (78,)), (42, 4, (74, 78)), (46, 11, (67,)),
    (57, 1, (86,)), (58, 5, (79,)), (63, 8, (59, 62)),
    (71, 9, (60,)), (80, 14, (64,)), (94, 2, (60, 64, 72)),
    (96, 1, (59,)), (97, 5, (62,)), (102, 1, (59,)),
    (103, 1, (62,)), (104, 8, (60,)), (112, 13, (60, 64)),
    (125, 1, (60,)), (126, 1, (60, 64)), (127, 1, (59,)),
    (128, 6, (64,)), (134, 5, (67, 71)), (139, 2, (60,)),
    (141, 4, (67, 71)), (145, 1, (62,)), (146, 1, (67,)),
    (147, 1, (62,)), (148, 2, (62, 71)), (150, 4, (62, 69)),
    (154, 2, (72,)), (156, 2, (69,)), (158, 3, (71,)),
    (161, 1, (62,)), (162, 1, (67, 71, 91)), (163, 1, (62,)),
    (164, 1, (62, 71)), (165, 2, (62, 69)), (167, 1, (66,)),
    (168, 2, (62, 66)), (170, 3, (72,)), (173, 4, (59, 67)),
    (177, 3, (64, 67, 71)), (180, 1, (62, 67, 71)), (181, 2, (59, 62)),
    (183, 1, (81,)), (184, 3, (62, 67, 71)), (187, 4, (59, 62)),
    (191, 2, (67,)), (193, 2, (60,)), (195, 4, (64,)),
    (199, 6, (64, 67)), (205, 13, (62,)), (218, 2, (64,)),
    (220, 8, (59, 62)), (228, 9, (60, 72)), (237, 9, (60, 64)),
    (246, 2, (67,)), (248, 2, (60, 62, 72)), (250, 2, (62, 67, 72)),
    (252, 6, (59,)), (258, 2, (62,)), (260, 7, (60,)),
    (267, 14, (60, 64, 72)), (281, 1, (60,)), (282, 1, (64,)),
    (283, 2, (59,)), (285, 5, (64,)), (290, 6, (64, 67, 71)),
    (296, 1, (60,)), (297, 1, (64,)), (298, 4, (71,)),
    (302, 1, (62,)), (303, 2, (62, 67)), (305, 2, (62, 67, 71)),
    (307, 3, (62, 69)), (310, 1, (62, 66)), (311, 2, (72,)),
    (313, 2, (62, 69)), (315, 2, (71,)), (317, 2, (62,)),
    (319, 1, (67,)), (320, 2, (62, 71)), (322, 3, (62, 69)),
    (325, 1, (66,)), (326, 4, (72,)), (330, 7, (59, 62, 67)),
    (337, 1, (62, 67, 71)), (338, 1, (59,)), (339, 2, (62, 67)),
    (341, 4, (62, 67, 71, 91)), (345, 14, (60,)), (359, 2, (67,)),
    (361, 13, (60, 64, 72)), (374, 2, (60, 72)), (376, 2, (62,)),
    (378, 6, (59,)), (384, 1, (62,)), (385, 2, (67,)),
    (387, 5, (62,)), (392, 9, (59, 66)), (401, 5, (59,)),
    (406, 4, (59, 67)), (410, 4, (62,)), (414, 9, (62, 67)),
    (423, 3, (67,)), (426, 3, (62,)), (429, 5, (59, 62)),
    (434, 4, (60, 78)), (438, 4, (59, 62)), (442, 3, (60,)),
    (445, 3, (60, 64)), (448, 3, (60, 72)), (451, 5, (62,)),
    (456, 2, (62, 74)), (458, 5, (59, 67)), (463, 3, (59, 67, 71)),
    (466, 1, (66, 69)), (467, 1, (62, 66)), (468, 5, (62, 66, 69)),
    (473, 1, (67,)), (474, 4, (62, 67)), (478, 3, (60, 67)),
    (481, 5, (62,)), (486, 9, (62, 67)), (495, 2, (59,)),
    (497, 4, (59, 62, 67)), (501, 6, (59, 62, 69)), (507, 2, (67,)),
    (509, 2, (62,)), (511, 7, (62, 66)), (518, 6, (62, 67)),
    (524, 5, (66,)), (529, 3, (62, 67)), (532, 1, (59,)),
    (533, 1, (67,)), (534, 2, (59,)), (536, 1, (67,)),
    (537, 4, (69,)), (541, 3, (62, 66)), (544, 1, (62, 67)),
    (545, 2, (67, 71)), (547, 2, (62, 67, 72)), (549, 2, (62, 74)),
    (551, 1, (60, 67, 76)), (552, 1, (72,)), (553, 1, (67, 72)),
    (554, 1, (72, 76)), (555, 1, (62, 67, 72)), (556, 1, (67, 78)),
    (557, 2, (67, 74)), (559, 1, (62, 67, 78)), (560, 1, (67, 79)),
    (561, 2, (59, 67, 71)), (563, 3, (64, 67, 71)), (566, 1, (67, 79)),
    (567, 1, (62, 66, 81)), (568, 2, (78,)), (570, 2, (62, 66, 74)),
    (572, 2, (66, 69, 71, 74)), (574, 2, (67, 83)), (576, 2, (62, 67, 69, 74, 84)),
    (578, 1, (62, 67, 74, 86)), (579, 1, (60, 79, 88)), (580, 1, (72,)),
    (581, 3, (84,)), (584, 1, (76,)), (585, 3, (72,)),
    (588, 2, (74,)), (590, 2, (76, 79)), (592, 1, (60, 72, 84)),
    (593, 2, (74, 76)), (595, 1, (79,)), (596, 4, (67,)),
    (600, 5, (60, 64)), (605, 2, (62,)), (607, 2, (64, 72)),
    (609, 1, (60, 62)), (610, 4, (79,)), (614, 3, (67, 78, 79)),
    (617, 2, (74,)), (619, 1, (67,)), (620, 1, (62, 67, 74)),
    (621, 1, (78,)), (622, 1, (74,)), (623, 1, (79,)),
    (624, 1, (78,)), (625, 1, (74,)), (626, 1, (62,)),
    (627, 1, (69,)), (628, 1, (74,)), (629, 1, (67,)),
    (630, 5, (74,)), (635, 1, (60, 62, 67)), (636, 4, (64,)),
    (640, 1, (60,)), (641, 3, (62, 64, 67, 72)), (644, 2, (60, 62, 67)),
    (646, 3, (67, 78)), (649, 2, (62, 74)), (651, 1, (67,)),
    (652, 1, (74,)), (653, 2, (78,)), (655, 1, (79,)),
    (656, 2, (78,)), (658, 2, (74,)), (660, 2, (67, 74, 86)),
    (662, 6, (79,)), (668, 1, (60, 62, 67)), (669, 4, (64,)),
    (673, 2, (62, 64, 67, 72)), (675, 4, (60, 67)), (679, 3, (59, 62)),
    (682, 1, (59, 62, 67)), (683, 1, (71,)), (684, 1, (59, 67)),
    (685, 10, (60,)), (695, 1, (72,)), (696, 8, (60,)),
    (704, 3, (60, 64, 72)), (707, 3, (60, 62)), (710, 8, (59, 62)),
    (718, 4, (60,)), (722, 3, (60, 64)), (725, 10, (69,)),
    (735, 2, (67, 74)), (737, 3, (62, 67, 72)), (740, 2, (60, 62, 67)),
    (742, 1, (59,)), (743, 2, (62,)), (745, 2, (59,)),
    (747, 3, (62,)), (750, 8, (60, 72)), (758, 11, (60, 64)),
    (769, 6, (60, 64, 72)), (775, 8, (59,)), (783, 8, (60,)),
    (791, 2, (60, 64)), (808, 2, (78,)), (810, 1, (67, 79)),
    (811, 1, (78,)), (812, 2, (74,)), (814, 1, (62, 69, 74)),
    (815, 2, (67,)), (817, 1, (66, 69)), (818, 3, (62, 67, 69)),
    (821, 4, (66, 69)), (825, 2, (62,)), (848, 4, (59,)),
)

SHARPS = 1                        # G major
NAMES = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]


def spell(midi: int) -> pitch.Pitch:
    """Spell a pitch for one sharp: F# rather than G-flat.

    displayStatus is left undecided so the exporter prints an accidental only
    where the key signature does not already give it - otherwise every F# in
    the piece carries a redundant sharp."""
    p = pitch.Pitch()
    p.midi = midi
    p.name = NAMES[midi % 12]
    p.octave = midi // 12 - 1
    if p.accidental is not None:
        p.accidental.displayStatus = None
    return p


def split_duration(at: float, length: float) -> list[float]:
    """Break a span into the largest note values that can be written at that
    point in the bar.

    A chord held for a whole bar should be a whole note, not eight tied
    eighths.  Values are tried longest first and only accepted where they
    start in a legal place: a semibreve only at the top of the bar, a minim
    only on beats 1 or 3, a dotted crotchet only on a beat."""
    out: list[float] = []
    eps = 1e-6
    while length > eps:
        for cand in (4.0, 2.0, 1.5, 1.0, 0.5):
            if cand > length + eps or at + cand > 4.0 + eps:
                continue
            if cand == 4.0 and abs(at) > eps:
                continue
            if cand == 2.0 and abs(at % 2.0) > eps:
                continue
            if cand in (1.5, 1.0) and abs(at % 1.0) > eps:
                continue
            out.append(cand)
            at += cand
            length -= cand
            break
        else:
            out.append(0.5)
            at += 0.5
            length -= 0.5
    return out


def fill(m: stream.Measure, at: float, length: float,
         pitches: tuple[int, ...] | None, tie_in: bool = False,
         tie_out: bool = False) -> None:
    """Put a rest or a (possibly tied) chord into a measure.

    Durations run to the next event in the same hand: the pedal holds, and a
    line of eighth rests between every chord is neither what was played nor
    something anyone wants to read."""
    parts = split_duration(at, length)
    for i, ql in enumerate(parts):
        if pitches is None:
            el: note.GeneralNote = note.Rest(quarterLength=ql)
        else:
            el = chord.Chord([spell(p) for p in pitches], quarterLength=ql)
            start = tie_in or i > 0
            stop = tie_out or i < len(parts) - 1
            if start and stop:
                el.tie = tie.Tie("continue")
            elif start:
                el.tie = tie.Tie("stop")
            elif stop:
                el.tie = tie.Tie("start")
        m.insert(at, el)
        at += ql


def build_staff(events, hand: str, clef_obj: clef.Clef) -> stream.PartStaff:
    p = stream.PartStaff()
    p.id = "Piano" + hand.upper()
    p.partName = "Piano" if hand == "rh" else ""
    p.insert(0, clef_obj)
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, key.KeySignature(SHARPS))

    # absolute quarter-note spans, so a held chord can cross a bar line
    spans = [(pos * EIGHTH, (pos + dur) * EIGHTH, pitches)
             for pos, dur, pitches in sorted(events)]
    for i in range(len(spans) - 1):            # never overlap the next attack
        s0, e0, ps = spans[i]
        spans[i] = (s0, min(e0, spans[i + 1][0]), ps)
    spans = [s for s in spans if s[1] > s[0]]

    idx = 0
    carry: tuple[tuple[int, ...], float] | None = None
    for b in range(BARS):
        m = stream.Measure(number=b + 1)
        bar_start, bar_end = 4.0 * b, 4.0 * (b + 1)
        cursor = bar_start
        if carry is not None:                  # a chord held in from the last bar
            pitches, end = carry
            stop = min(end, bar_end)
            fill(m, 0.0, stop - bar_start, pitches, tie_in=True,
                 tie_out=end > bar_end)
            cursor = stop
            carry = (pitches, end) if end > bar_end else None
        while idx < len(spans) and spans[idx][0] < bar_end:
            start, end, pitches = spans[idx]
            if start > cursor:
                fill(m, cursor - bar_start, start - cursor, None)
                cursor = start
            stop = min(end, bar_end)
            if stop > cursor:
                fill(m, cursor - bar_start, stop - cursor, pitches,
                     tie_out=end > bar_end)
                cursor = stop
            if end > bar_end:
                carry = (pitches, end)
            idx += 1
        if cursor < bar_end:
            fill(m, cursor - bar_start, bar_end - cursor, None)
        p.append(m)
    return p


def build_score() -> stream.Score:
    sc = stream.Score()
    sc.metadata = metadata.Metadata(
        title="Every Little Thing",
        subtitle="piano - transcribed from the recording",
    )
    rh = build_staff(RIGHT, "rh", clef.TrebleClef())
    lh = build_staff(LEFT, "lh", clef.BassClef())

    mm = tempo.MetronomeMark(number=131, referent=note.Note(type="quarter"))
    mm.placement = "above"
    rh.measure(1).insert(0.0, mm)
    for text in ("Rubato - the tempo moves between about 122 and 136",
                 "quantised to eighths: the performance does not support finer"):
        te = expressions.TextExpression(text)
        te.placement = "above"
        te.style.fontStyle = "italic"
        rh.measure(1).insert(0.0, te)

    for st in (rh, lh):
        st[-1].rightBarline = bar.Barline("final")
        st.makeAccidentals(inPlace=True,
                           alteredPitches=key.KeySignature(SHARPS).alteredPitches)
        sc.insert(0, st)
    sc.insert(0, layout.StaffGroup([rh, lh], symbol="brace",
                                   barTogether=True, name="Piano"))
    return sc


def pack_mxl(src: str, dst: str) -> None:
    name = os.path.basename(src)
    container = ('<?xml version="1.0" encoding="UTF-8"?>\n<container>'
                 '<rootfiles><rootfile full-path="%s" '
                 'media-type="application/vnd.recordare.musicxml+xml"/>'
                 '</rootfiles></container>\n' % name)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("META-INF/container.xml", container)
        z.write(src, name)


def main() -> None:
    ap = argparse.ArgumentParser(description="write the piano transcription")
    ap.add_argument("-o", "--out", default="Every_Little_Thing_Piano.musicxml")
    ap.add_argument("--mxl", help="also write a compressed .mxl here")
    args = ap.parse_args()

    sc = build_score()
    sc.write("musicxml", fp=args.out)
    n = sum(len(p) for _, _, p in LEFT) + sum(len(p) for _, _, p in RIGHT)
    print(f"wrote {args.out}  ({BARS} bars, {len(LEFT)} LH + {len(RIGHT)} RH "
          f"chords, {n} notes)")
    if args.mxl:
        pack_mxl(args.out, args.mxl)
        print(f"wrote {args.mxl}")


if __name__ == "__main__":
    main()
