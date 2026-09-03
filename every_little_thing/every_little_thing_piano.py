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
    (0, 1, (43,)), (1, 1, (43,)), (3, 1, (43, 55)),
    (4, 4, (43, 55)), (8, 1, (45,)), (9, 1, (45,)),
    (11, 3, (45, 57)), (14, 1, (45,)), (15, 4, (48, 52, 55, 57)),
    (19, 3, (36,)), (28, 1, (55,)), (29, 1, (55,)),
    (31, 1, (43, 50, 55)), (32, 1, (55,)), (33, 1, (50,)),
    (34, 1, (55,)), (35, 2, (43, 50, 55)), (37, 2, (55,)),
    (39, 2, (45, 52)), (41, 1, (55,)), (42, 3, (52, 55)),
    (45, 1, (55,)), (46, 1, (55,)), (47, 2, (48, 52, 55)),
    (49, 1, (36,)), (58, 1, (36,)), (62, 1, (52,)),
    (63, 1, (43, 50, 55)), (65, 4, (55,)), (69, 1, (43,)),
    (70, 1, (43,)), (71, 1, (45, 52, 55, 57)), (72, 1, (45, 57)),
    (74, 1, (45, 57)), (79, 1, (55,)), (80, 1, (36,)),
    (81, 3, (48, 55)), (85, 1, (55,)), (87, 1, (55,)),
    (88, 1, (48,)), (90, 1, (48, 55)), (91, 2, (48,)),
    (93, 1, (55,)), (94, 1, (48, 55)), (95, 1, (50,)),
    (96, 1, (43, 50)), (97, 2, (55,)), (101, 1, (55,)),
    (102, 1, (55,)), (103, 1, (50,)), (104, 1, (45, 55, 57)),
    (105, 1, (45, 55)), (107, 1, (45,)), (109, 1, (55,)),
    (110, 1, (55,)), (111, 1, (45, 55)), (112, 2, (36, 45, 48)),
    (118, 3, (48,)), (121, 1, (55,)), (123, 1, (48, 55)),
    (124, 1, (55,)), (125, 1, (48,)), (126, 1, (55,)),
    (127, 2, (36, 48, 55)), (132, 3, (48, 55)), (135, 1, (52, 55)),
    (137, 1, (52,)), (138, 1, (55,)), (139, 1, (48, 52)),
    (141, 1, (55,)), (142, 2, (43, 55)), (144, 2, (50, 55)),
    (146, 2, (55,)), (148, 2, (55,)), (150, 1, (42,)),
    (151, 1, (50,)), (152, 1, (54,)), (153, 2, (50, 54)),
    (155, 3, (50,)), (158, 1, (43,)), (159, 1, (50,)),
    (160, 2, (55,)), (162, 1, (55, 62)), (164, 1, (50, 55)),
    (165, 1, (42,)), (166, 1, (50,)), (167, 2, (54,)),
    (169, 1, (54,)), (170, 2, (54,)), (172, 1, (54,)),
    (173, 1, (55,)), (174, 2, (52,)), (177, 2, (55, 59)),
    (179, 1, (52,)), (181, 2, (43, 50, 55)), (184, 2, (43, 50, 55, 59)),
    (187, 1, (55, 57)), (188, 2, (45, 57)), (193, 1, (45, 52)),
    (195, 1, (52, 57)), (197, 3, (57,)), (200, 2, (45, 47, 57)),
    (205, 2, (38, 50, 55)), (207, 1, (55,)), (208, 1, (50, 55)),
    (209, 1, (38,)), (213, 2, (38, 50)), (215, 1, (55,)),
    (216, 1, (50,)), (217, 2, (55, 57)), (219, 1, (50, 57)),
    (220, 2, (43, 50, 55)), (223, 1, (55,)), (225, 1, (55,)),
    (226, 2, (43, 55)), (228, 1, (45, 52, 55, 57)), (229, 1, (45,)),
    (230, 3, (45,)), (237, 1, (36, 43, 55)), (238, 1, (48, 55)),
    (240, 2, (48, 55)), (244, 1, (43, 48, 55)), (245, 1, (43, 48, 55)),
    (246, 1, (55,)), (247, 1, (43, 55)), (248, 1, (43, 48, 55)),
    (249, 1, (43, 55)), (250, 1, (55,)), (251, 1, (50, 55)),
    (252, 1, (43,)), (257, 1, (43,)), (258, 1, (55,)),
    (259, 1, (52, 55)), (260, 1, (45, 52)), (261, 1, (45,)),
    (263, 1, (45, 55)), (264, 1, (52,)), (265, 1, (55, 57)),
    (266, 1, (52, 55)), (267, 1, (36, 43, 48, 52, 55)), (269, 1, (48,)),
    (270, 1, (43, 55)), (272, 1, (36, 55)), (273, 1, (48,)),
    (274, 3, (55,)), (278, 1, (55,)), (279, 1, (48,)),
    (280, 1, (55,)), (281, 1, (48,)), (282, 1, (55,)),
    (283, 2, (36, 48)), (285, 1, (55,)), (287, 1, (55,)),
    (289, 1, (55,)), (290, 1, (48, 55)), (291, 4, (52, 55)),
    (295, 1, (52, 55)), (296, 1, (48,)), (299, 1, (43,)),
    (300, 1, (50,)), (301, 1, (55,)), (302, 1, (50,)),
    (303, 1, (55,)), (307, 1, (42,)), (308, 1, (50,)),
    (309, 1, (54,)), (310, 1, (54,)), (312, 2, (54,)),
    (315, 1, (43,)), (316, 1, (43,)), (317, 1, (50, 55)),
    (318, 1, (50,)), (319, 2, (55,)), (322, 1, (42, 50)),
    (323, 1, (50,)), (324, 1, (54,)), (325, 3, (54,)),
    (330, 1, (55,)), (331, 2, (52,)), (334, 2, (55,)),
    (337, 1, (55,)), (338, 1, (55,)), (339, 2, (50,)),
    (341, 2, (50, 55, 59)), (343, 2, (50,)), (345, 4, (45, 52, 55)),
    (354, 1, (50,)), (356, 1, (45, 47, 50)), (357, 1, (50, 55)),
    (358, 1, (45,)), (359, 1, (45, 55)), (361, 2, (38, 55)),
    (363, 3, (50,)), (366, 1, (47,)), (368, 1, (50,)),
    (374, 1, (50, 55)), (376, 1, (50, 55, 57)), (377, 1, (55,)),
    (378, 1, (36, 48)), (379, 1, (55,)), (380, 1, (55,)),
    (382, 2, (48, 55)), (384, 1, (55,)), (385, 3, (38, 55)),
    (388, 2, (55,)), (390, 2, (50, 55)), (392, 1, (40, 52)),
    (398, 1, (52,)), (400, 1, (55,)), (401, 1, (52, 55)),
    (402, 2, (55,)), (404, 1, (55,)), (405, 1, (52,)),
    (406, 1, (52, 55)), (408, 1, (52, 55)), (409, 1, (38,)),
    (410, 1, (36, 48, 55)), (411, 2, (36,)), (414, 1, (36, 55)),
    (415, 1, (48,)), (416, 1, (36, 55)), (417, 1, (38, 50)),
    (418, 2, (55,)), (420, 1, (50, 55)), (421, 1, (50, 55)),
    (422, 1, (55,)), (423, 1, (55,)), (424, 1, (50,)),
    (425, 1, (43, 50, 55)), (426, 1, (55,)), (427, 2, (55,)),
    (429, 3, (43, 50, 55)), (432, 1, (50,)), (433, 1, (43, 55)),
    (434, 1, (43, 48, 52, 55)), (435, 3, (55,)), (438, 1, (43, 50, 55)),
    (439, 2, (55,)), (441, 1, (50, 55)), (442, 1, (36, 55)),
    (443, 2, (48, 55)), (445, 1, (48, 55)), (446, 1, (48,)),
    (447, 2, (55,)), (449, 1, (55,)), (450, 1, (38, 55)),
    (451, 2, (50, 55)), (454, 1, (50, 55, 57)), (455, 1, (55,)),
    (456, 1, (50, 55, 57)), (457, 1, (52,)), (458, 1, (40,)),
    (462, 1, (52,)), (464, 1, (52,)), (466, 1, (50,)),
    (468, 2, (50, 57)), (470, 1, (50, 54)), (471, 1, (54,)),
    (472, 1, (55,)), (473, 1, (55,)), (474, 1, (36, 55)),
    (475, 1, (48,)), (476, 1, (55,)), (477, 1, (36,)),
    (478, 1, (36, 48, 55)), (480, 1, (36, 55)), (481, 2, (38, 50, 55)),
    (483, 1, (50, 55)), (484, 2, (55,)), (486, 1, (50, 55)),
    (487, 1, (55,)), (488, 1, (50, 55)), (489, 3, (40, 55)),
    (495, 1, (52,)), (497, 2, (40, 57)), (500, 1, (57,)),
    (501, 1, (40, 52, 57)), (502, 2, (52, 57)), (504, 1, (52, 57)),
    (505, 1, (50,)), (506, 1, (55,)), (507, 1, (36, 55)),
    (508, 1, (48, 55)), (509, 1, (36, 48, 55)), (511, 1, (38, 50, 55)),
    (512, 1, (50, 55)), (513, 1, (38, 50, 55)), (515, 1, (38, 50)),
    (518, 1, (36, 48, 55)), (520, 1, (48, 55)), (521, 1, (36, 48, 55)),
    (524, 1, (38, 50, 55, 57)), (525, 1, (38, 50)), (526, 1, (38, 50, 55)),
    (527, 1, (38, 50, 55)), (528, 2, (38, 50, 55)), (530, 2, (40,)),
    (533, 1, (52,)), (534, 3, (52,)), (537, 2, (42, 50, 54)),
    (539, 2, (57,)), (541, 2, (50, 54)), (543, 1, (54,)),
    (544, 1, (43, 50, 55)), (545, 1, (43, 55)), (547, 2, (45, 57)),
    (549, 1, (45, 47)), (551, 2, (48,)), (556, 1, (50,)),
    (559, 1, (50,)), (560, 1, (52,)), (561, 2, (52,)),
    (563, 2, (52, 59)), (565, 1, (52,)), (567, 2, (54,)),
    (572, 1, (54,)), (574, 1, (55,)), (576, 2, (55, 57, 59)),
    (578, 1, (55, 59)), (584, 2, (48,)), (600, 4, (55,)),
    (604, 1, (55,)), (607, 4, (43,)), (613, 1, (43, 50)),
    (614, 2, (43, 50, 55)), (617, 1, (55,)), (618, 1, (43, 50, 55)),
    (619, 1, (55,)), (620, 1, (55,)), (621, 1, (43,)),
    (622, 2, (45,)), (626, 1, (45, 57)), (627, 1, (50,)),
    (629, 1, (45,)), (630, 4, (36, 48, 52, 55)), (636, 4, (48, 52, 55)),
    (641, 1, (48, 55, 60)), (642, 2, (48, 55, 60)), (644, 1, (55,)),
    (645, 1, (55,)), (646, 1, (43, 50, 55)), (647, 2, (50,)),
    (650, 1, (43, 50, 55)), (651, 1, (55,)), (652, 2, (55,)),
    (654, 2, (45, 52)), (656, 2, (50, 55)), (658, 3, (52, 55)),
    (661, 1, (55,)), (662, 6, (48, 52, 55)), (668, 1, (48, 52, 55)),
    (669, 1, (43,)), (672, 1, (55,)), (673, 1, (48, 55, 60)),
    (674, 3, (52, 55)), (677, 1, (43,)), (678, 4, (43, 50, 55)),
    (682, 1, (50, 55)), (683, 1, (43, 55)), (684, 1, (50,)),
    (686, 1, (45, 57)), (687, 1, (57,)), (688, 1, (57,)),
    (689, 1, (45,)), (690, 1, (45, 57)), (691, 1, (45, 57)),
    (692, 1, (45, 57)), (693, 1, (45, 48)), (694, 2, (36,)),
    (696, 1, (55,)), (697, 1, (55,)), (698, 2, (36, 48, 55)),
    (701, 1, (55,)), (702, 1, (36,)), (703, 1, (48,)),
    (704, 1, (48, 55)), (705, 1, (48,)), (706, 1, (36,)),
    (707, 2, (48, 55)), (709, 1, (50,)), (710, 3, (43, 55)),
    (714, 2, (50, 55)), (716, 1, (43,)), (718, 1, (45, 52, 55, 57)),
    (720, 2, (45,)), (722, 2, (45, 57)), (724, 1, (57,)),
    (725, 1, (48, 55)), (726, 1, (36,)), (727, 1, (55,)),
    (728, 1, (55,)), (729, 1, (55,)), (733, 2, (55,)),
    (735, 1, (55,)), (736, 1, (48, 55)), (737, 1, (48, 55, 60)),
    (738, 1, (43,)), (739, 1, (55,)), (740, 1, (48, 55)),
    (741, 1, (36, 50)), (742, 1, (43, 50, 55)), (743, 1, (55,)),
    (744, 1, (31,)), (745, 1, (43,)), (746, 1, (50,)),
    (747, 1, (43, 55)), (748, 1, (57,)), (750, 1, (45, 55, 57)),
    (751, 1, (52,)), (752, 1, (57,)), (754, 1, (45, 52, 55, 57)),
    (755, 1, (55, 57)), (756, 1, (57,)), (757, 1, (48, 55)),
    (758, 1, (36, 43, 48, 52, 55)), (759, 2, (55,)), (762, 1, (55,)),
    (763, 1, (48,)), (766, 1, (43,)), (768, 1, (48,)),
    (769, 1, (48, 55)), (770, 1, (48,)), (771, 1, (36, 55)),
    (772, 1, (48, 55)), (773, 2, (50, 55)), (775, 1, (43, 55)),
    (776, 6, (50,)), (783, 2, (45, 52, 55, 57)), (791, 1, (36, 43, 55)),
    (792, 2, (48, 55)), (794, 2, (43,)), (796, 1, (55,)),
    (797, 1, (43,)), (798, 1, (43,)), (800, 1, (36, 48)),
    (801, 1, (55,)), (802, 1, (55,)), (804, 2, (47,)),
    (806, 2, (48, 55)), (808, 2, (43, 50, 55)), (810, 1, (55,)),
    (811, 1, (43,)), (812, 2, (55,)), (814, 1, (50,)),
    (815, 2, (55,)), (817, 1, (50,)), (818, 1, (43, 50)),
    (819, 2, (55, 57)), (821, 1, (43, 55)), (822, 1, (38,)),
    (823, 2, (54,)), (825, 1, (50,)), (826, 1, (38,)),
    (827, 2, (38, 43, 45, 47, 50)), (829, 1, (31,)), (835, 1, (50,)),
    (836, 1, (31, 38, 43, 47)), (837, 1, (55,)), (838, 1, (43,)),
    (839, 1, (55,)), (843, 1, (31,)), (844, 9, (43, 50)),
    (853, 1, (43,)), (855, 1, (43,)), (856, 1, (43,)),
    (857, 5, (31,)), (862, 4, (43,)), (866, 2, (50,)),
    (868, 1, (43, 50)), (871, 1, (43,)),)

RIGHT: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (0, 1, (78,)), (1, 1, (78,)), (2, 1, (78,)),
    (3, 1, (74, 78, 86)), (5, 1, (78,)), (6, 1, (74,)),
    (7, 1, (74, 78)), (9, 1, (79,)), (10, 1, (78,)),
    (11, 3, (74,)), (14, 1, (74,)), (15, 11, (60, 67)),
    (31, 1, (78,)), (32, 1, (79,)), (33, 1, (78,)),
    (34, 1, (62, 74)), (35, 3, (78,)), (38, 2, (78,)),
    (40, 1, (79,)), (41, 1, (78,)), (42, 1, (74, 78)),
    (43, 1, (74, 78)), (45, 1, (74,)), (46, 2, (67,)),
    (54, 3, (67,)), (57, 1, (86,)), (58, 1, (79,)),
    (63, 1, (59, 62)), (64, 1, (59, 62)), (65, 1, (59,)),
    (66, 1, (59, 62)), (71, 1, (60,)), (72, 1, (60,)),
    (79, 1, (60,)), (80, 2, (64,)), (94, 1, (60, 64, 72)),
    (96, 1, (59,)), (97, 2, (62,)), (102, 1, (59,)),
    (103, 1, (62,)), (104, 1, (60,)), (105, 1, (60,)),
    (107, 1, (60,)), (112, 2, (60, 64)), (125, 1, (60,)),
    (126, 1, (60, 64)), (127, 1, (59,)), (128, 1, (64,)),
    (134, 2, (67, 71)), (137, 1, (67,)), (139, 1, (60,)),
    (141, 2, (67, 71)), (144, 1, (67,)), (145, 1, (62,)),
    (146, 1, (67,)), (147, 1, (62,)), (148, 1, (62, 71)),
    (149, 1, (62,)), (150, 1, (62, 69)), (153, 1, (62,)),
    (154, 2, (72,)), (156, 2, (69,)), (158, 3, (71,)),
    (161, 1, (62,)), (162, 1, (67, 71, 91)), (163, 1, (62,)),
    (164, 1, (62, 71)), (165, 2, (62, 69)), (167, 1, (66,)),
    (168, 1, (62, 66)), (170, 2, (72,)), (173, 2, (59, 67)),
    (177, 2, (64, 67, 71)), (179, 1, (64,)), (180, 1, (62, 67, 71)),
    (181, 2, (59, 62)), (183, 1, (81,)), (184, 2, (62, 67, 71)),
    (187, 1, (59, 62)), (190, 1, (62,)), (191, 1, (67,)),
    (193, 1, (60,)), (195, 1, (64,)), (199, 1, (64, 67)),
    (205, 2, (62,)), (218, 2, (64,)), (220, 2, (59, 62)),
    (223, 1, (59,)), (226, 2, (62,)), (228, 1, (60, 72)),
    (230, 3, (60,)), (237, 2, (60, 64)), (246, 1, (67,)),
    (248, 1, (60, 62, 72)), (250, 1, (62, 67, 72)), (251, 1, (62,)),
    (252, 1, (59,)), (258, 1, (62,)), (260, 1, (60,)),
    (263, 1, (60,)), (265, 1, (60,)), (267, 1, (60, 64, 72)),
    (281, 1, (60,)), (282, 1, (64,)), (283, 2, (59,)),
    (285, 1, (64,)), (287, 1, (64,)), (290, 1, (64, 67, 71)),
    (292, 2, (67,)), (296, 1, (60,)), (297, 1, (64,)),
    (298, 2, (71,)), (302, 1, (62,)), (303, 1, (62, 67)),
    (304, 1, (62,)), (305, 1, (62, 67, 71)), (306, 1, (62, 67)),
    (307, 1, (62, 69)), (310, 1, (62, 66)), (311, 2, (72,)),
    (313, 1, (62, 69)), (315, 1, (71,)), (317, 1, (62,)),
    (318, 1, (62,)), (319, 1, (67,)), (320, 1, (62, 71)),
    (321, 1, (62,)), (322, 1, (62, 69)), (324, 1, (62,)),
    (325, 1, (66,)), (326, 1, (72,)), (330, 4, (59, 62, 67)),
    (334, 2, (59, 62, 67)), (336, 1, (62,)), (337, 1, (62, 67, 71)),
    (338, 1, (59,)), (339, 2, (62, 67)), (341, 2, (62, 67, 71, 91)),
    (343, 2, (62,)), (345, 4, (60,)), (359, 1, (67,)),
    (361, 2, (60, 64, 72)), (374, 1, (60, 72)), (376, 1, (62,)),
    (377, 1, (62,)), (378, 1, (59,)), (384, 1, (62,)),
    (385, 2, (67,)), (387, 1, (62,)), (390, 2, (62,)),
    (392, 1, (59, 66)), (401, 2, (59,)), (406, 1, (59, 67)),
    (408, 1, (59,)), (410, 3, (62,)), (414, 1, (62, 67)),
    (416, 2, (62, 67)), (421, 1, (67,)), (423, 2, (67,)),
    (426, 1, (62,)), (427, 2, (62,)), (429, 3, (59, 62)),
    (432, 1, (62,)), (434, 1, (60, 78)), (438, 1, (59, 62)),
    (439, 3, (59, 62)), (442, 1, (60,)), (445, 1, (60, 64)),
    (448, 1, (60, 72)), (449, 1, (72,)), (451, 2, (62,)),
    (455, 1, (62,)), (456, 2, (62, 74)), (458, 1, (59, 67)),
    (461, 1, (59, 67)), (463, 1, (59, 67, 71)), (464, 1, (59, 67)),
    (466, 1, (66, 69)), (467, 1, (62, 66)), (468, 2, (62, 66, 69)),
    (471, 1, (62,)), (472, 1, (62,)), (473, 1, (67,)),
    (474, 2, (62, 67)), (476, 1, (67,)), (477, 1, (62, 67)),
    (478, 1, (60, 67)), (481, 2, (62,)), (483, 1, (62,)),
    (484, 2, (62,)), (486, 1, (62, 67)), (487, 1, (62, 67)),
    (488, 1, (62, 67)), (489, 3, (62, 67)), (492, 1, (62, 67)),
    (493, 1, (67,)), (495, 1, (59,)), (497, 2, (59, 62, 67)),
    (499, 1, (62,)), (500, 1, (59,)), (501, 1, (59, 62, 69)),
    (502, 2, (59, 62)), (506, 1, (62,)), (507, 1, (67,)),
    (509, 1, (62,)), (511, 1, (62, 66)), (513, 1, (62,)),
    (515, 1, (62, 66)), (518, 1, (62, 67)), (521, 1, (67,)),
    (524, 1, (66,)), (525, 1, (66,)), (526, 1, (66,)),
    (527, 1, (66,)), (529, 1, (62, 67)), (530, 2, (62, 67)),
    (532, 1, (59,)), (533, 1, (67,)), (534, 2, (59,)),
    (536, 1, (67,)), (537, 2, (69,)), (539, 2, (69,)),
    (541, 2, (62, 66)), (543, 1, (66,)), (544, 1, (62, 67)),
    (545, 1, (67, 71)), (547, 2, (62, 67, 72)), (549, 1, (62, 74)),
    (551, 1, (60, 67, 76)), (552, 1, (72,)), (553, 1, (67, 72)),
    (554, 1, (72, 76)), (555, 1, (62, 67, 72)), (556, 1, (67, 78)),
    (557, 1, (67, 74)), (558, 1, (67,)), (559, 1, (62, 67, 78)),
    (560, 1, (67, 79)), (561, 2, (59, 67, 71)), (563, 1, (64, 67, 71)),
    (564, 1, (67,)), (565, 1, (71,)), (566, 1, (67, 79)),
    (567, 1, (62, 66, 81)), (568, 2, (78,)), (570, 1, (62, 66, 74)),
    (572, 1, (66, 69, 71, 74)), (574, 1, (67, 83)), (576, 2, (62, 67, 69, 74, 84)),
    (578, 1, (62, 67, 74, 86)), (579, 1, (60, 79, 88)), (580, 1, (72,)),
    (581, 1, (84,)), (584, 1, (76,)), (585, 1, (72,)),
    (588, 2, (74,)), (590, 2, (76, 79)), (592, 1, (60, 72, 84)),
    (593, 2, (74, 76)), (595, 1, (79,)), (596, 4, (67,)),
    (600, 5, (60, 64)), (605, 2, (62,)), (607, 2, (64, 72)),
    (609, 1, (60, 62)), (610, 3, (79,)), (614, 2, (67, 78, 79)),
    (616, 1, (78, 79)), (617, 1, (74,)), (619, 1, (67,)),
    (620, 1, (62, 67, 74)), (621, 1, (78,)), (622, 1, (74,)),
    (623, 1, (79,)), (624, 1, (78,)), (625, 1, (74,)),
    (626, 1, (62,)), (627, 1, (69,)), (628, 1, (74,)),
    (629, 1, (67,)), (630, 3, (74,)), (633, 1, (74,)),
    (635, 1, (60, 62, 67)), (636, 4, (64,)), (640, 1, (60,)),
    (641, 1, (62, 64, 67, 72)), (642, 1, (62, 67, 72)), (643, 1, (62, 72)),
    (644, 1, (60, 62, 67)), (645, 1, (62,)), (646, 1, (67, 78)),
    (647, 2, (67, 78)), (649, 1, (62, 74)), (651, 1, (67,)),
    (652, 1, (74,)), (653, 1, (78,)), (655, 1, (79,)),
    (656, 2, (78,)), (658, 2, (74,)), (660, 1, (67, 74, 86)),
    (661, 1, (67,)), (662, 6, (79,)), (668, 1, (60, 62, 67)),
    (669, 1, (64,)), (673, 1, (62, 64, 67, 72)), (675, 1, (60, 67)),
    (679, 1, (59, 62)), (682, 1, (59, 62, 67)), (683, 1, (71,)),
    (684, 1, (59, 67)), (685, 1, (60,)), (689, 1, (60,)),
    (692, 1, (60,)), (693, 2, (60,)), (695, 1, (72,)),
    (696, 1, (60,)), (697, 1, (60,)), (704, 1, (60, 64, 72)),
    (705, 1, (60,)), (707, 3, (60, 62)), (710, 3, (59, 62)),
    (714, 2, (62,)), (716, 1, (62,)), (718, 1, (60,)),
    (722, 2, (60, 64)), (724, 1, (60,)), (725, 1, (69,)),
    (735, 1, (67, 74)), (736, 1, (74,)), (737, 2, (62, 67, 72)),
    (739, 1, (62, 67)), (740, 1, (60, 62, 67)), (742, 1, (59,)),
    (743, 2, (62,)), (745, 1, (59,)), (747, 1, (62,)),
    (749, 1, (62,)), (750, 1, (60, 72)), (751, 1, (60,)),
    (754, 1, (60,)), (755, 1, (60,)), (756, 2, (60, 72)),
    (758, 2, (60, 64)), (762, 1, (60,)), (769, 1, (60, 64, 72)),
    (775, 6, (59,)), (783, 2, (60,)), (785, 1, (60,)),
    (787, 1, (60,)), (788, 1, (60,)), (791, 1, (60, 64)),
    (794, 3, (60,)), (808, 2, (78,)), (810, 1, (67, 79)),
    (811, 1, (78,)), (812, 2, (74,)), (814, 1, (62, 69, 74)),
    (815, 2, (67,)), (817, 1, (66, 69)), (818, 1, (62, 67, 69)),
    (819, 2, (62,)), (821, 2, (66, 69)), (823, 2, (66,)),
    (825, 1, (62,)), (848, 1, (59,)), (856, 3, (59,)),)

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
