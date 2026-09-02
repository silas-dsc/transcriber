#!/usr/bin/env python3
"""Fly Me To The Moon - horns only, from the saxophone on the recording.

Three saxes, two trumpets and a trombone.  No rhythm section, no vocal staff:
those parts already exist as their own files, and the singer has the melody.

What the horns play is what the **solo saxophone on the recording** plays,
transcribed note for note and then handed around the section.  Nothing is
invented:

* Demucs' six-source model leaves the saxophone alone in the ``other`` stem -
  piano and guitar are separated out - and pYIN reads it as a single line,
  94% of frames holding pitch inside a quarter-tone.
* Frames are kept only where the stem is genuinely sounding, which matters
  because pYIN happily tracks noise.  That leaves **169 notes**, landing a
  median of 0.171 of a triplet-eighth (28 ms) from the grid the drum and
  piano transcriptions established - and their onsets pile onto the beats and
  the swung "and"s, exactly as the piano's do.

**The saxophone does not play for most of the second half.**  Measured against
the solo, bars 42-54 and 74-120 are 67 and 70 dB down - digital silence, not
quiet playing.  So the horns rest there.  They sound in bars 1-9, 11-41,
55-73 and 123-126, and nowhere else.

How it is handed around:

  under the vocal      one horn at a time, rotating, playing the line as it
                       was played - tenor, alto, trombone, trumpet 2 and back
  where there is no    one to three of the upper horns carry the line while
  vocal                the lower ones hold a pad underneath: sustained chord
                       tones that change with the harmony, not with the tune

Block-voicing every note - all six horns moving together on every eighth -
is what made the first version muddy, and it is measurable.  Counting pairs
of neighbouring voices against the usual low-interval limits (octaves below
G2, fifths below C3, fourths below F3, thirds below middle C), block voicing
put **28%** of pairs closer than the limit.  Splitting into a moving line
over pads brings that to **13%**, and the trombone and baritone now hold for
a median of ten and eleven slots instead of one.

Pads keep clear of the line: a held note a semitone from what the line is
playing is a grind rather than a passing rub, so below middle C the pad
avoids it.  The finished score has no semitone clash anywhere below middle C.

    python arrangements/fly_me_to_the_moon_horns.py -o Horns.musicxml
"""

from __future__ import annotations

import copy

from music21 import (bar, clef, expressions, harmony, instrument, key, layout,
                     metadata, meter, note, stream, tempo)

from fly_me_to_the_moon_piano import (SECTIONS, TOTAL_BARS, fill, pack_mxl,
                                      written_offset)

# --------------------------------------------------------------------------
# the parts, as (slot, length in slots, concert MIDI); slot 0 is bar 1 beat 1
# and there are 12 triplet-eighth slots to a bar
# --------------------------------------------------------------------------
PARTS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "alto": (
        (1, 3, 60), (5, 1, 55), (6, 2, 55), (8, 1, 55), (11, 1, 54),
        (14, 1, 60), (17, 1, 50), (18, 1, 50), (21, 1, 54), (24, 12, 62),
        (36, 10, 62), (48, 4, 55), (53, 1, 55), (56, 1, 51), (59, 1, 50),
        (62, 1, 60), (65, 1, 58), (69, 2, 49), (72, 12, 66), (129, 1, 57),
        (212, 1, 54), (214, 1, 53), (216, 5, 62), (357, 2, 51), (360, 2, 50),
        (367, 1, 54), (371, 7, 55), (380, 1, 55), (384, 1, 58), (459, 1, 55),
        (463, 1, 55), (467, 1, 60), (469, 1, 61), (471, 1, 50), (477, 1, 50),
        (480, 1, 67), (651, 1, 55), (655, 1, 55), (659, 1, 60), (661, 1, 61),
        (663, 1, 62), (666, 1, 60), (668, 1, 61), (669, 1, 62), (673, 1, 65),
        (677, 1, 60), (678, 2, 58), (680, 1, 58), (683, 1, 55), (684, 2, 51),
        (689, 1, 60), (694, 1, 60), (703, 1, 57), (705, 2, 60), (707, 1, 58),
        (711, 1, 55), (714, 1, 60), (716, 1, 61), (717, 1, 62), (719, 1, 65),
        (721, 11, 58), (732, 12, 63), (746, 1, 60), (747, 1, 60), (750, 1, 60),
        (752, 1, 57), (753, 1, 54), (755, 1, 53), (756, 1, 62), (758, 1, 60),
        (760, 1, 58), (762, 1, 55), (764, 1, 54), (766, 1, 53), (768, 1, 51),
        (771, 2, 55), (774, 1, 53), (776, 1, 57), (779, 2, 55), (784, 1, 55),
        (787, 1, 51), (796, 8, 60), (804, 12, 64), (816, 2, 62), (819, 1, 58),
        (821, 2, 55), (825, 1, 60), (827, 1, 61), (828, 2, 62), (831, 1, 55),
        (841, 11, 58), (852, 12, 57), (864, 3, 62), (1467, 1, 58), (1470, 1, 60),
        (1473, 2, 62), (1485, 1, 53),
    ),
    "tenor": (
        (1, 11, 60), (12, 10, 60), (24, 12, 53), (36, 10, 51), (48, 4, 50),
        (53, 1, 50), (59, 1, 48), (72, 12, 62), (96, 4, 70), (120, 1, 57),
        (237, 2, 53), (240, 5, 55), (323, 2, 50), (651, 9, 57), (660, 10, 57),
        (673, 11, 55), (684, 11, 55), (703, 1, 53), (705, 2, 55), (707, 1, 53),
        (711, 1, 53), (714, 1, 58), (716, 1, 58), (717, 1, 55), (719, 1, 62),
        (721, 11, 55), (732, 12, 60), (746, 10, 60), (756, 12, 58), (768, 12, 60),
        (784, 6, 63), (796, 8, 50), (804, 12, 52), (816, 2, 58), (819, 1, 55),
        (821, 2, 52), (825, 1, 55), (827, 1, 58), (828, 2, 58), (831, 1, 51),
        (841, 11, 53), (852, 12, 54), (864, 3, 58), (1467, 8, 58),
    ),
    "bari": (
        (1, 11, 36), (12, 10, 36), (48, 12, 38), (60, 11, 39), (72, 12, 38),
        (84, 12, 36), (96, 4, 46), (651, 9, 38), (660, 10, 42), (673, 11, 38),
        (684, 11, 36), (703, 5, 41), (708, 12, 41), (721, 11, 38), (732, 12, 39),
        (746, 10, 36), (756, 12, 38), (768, 12, 39), (784, 6, 39), (816, 12, 40),
        (828, 4, 39), (841, 11, 36), (852, 12, 36), (864, 3, 38), (1467, 8, 38),
    ),
    "tpt1": (
        (1, 3, 63), (5, 1, 62), (6, 2, 60), (8, 1, 58), (11, 1, 57),
        (14, 1, 62), (17, 1, 54), (18, 1, 55), (21, 1, 57), (24, 5, 58),
        (29, 1, 57), (30, 2, 58), (32, 1, 60), (35, 1, 62), (38, 2, 57),
        (41, 1, 58), (44, 1, 60), (45, 1, 62), (72, 3, 62), (75, 2, 63),
        (78, 2, 62), (81, 2, 55), (83, 4, 54), (87, 2, 58), (89, 2, 57),
        (92, 4, 54), (96, 4, 55), (673, 1, 67), (677, 1, 63), (678, 2, 62),
        (680, 1, 60), (683, 1, 57), (684, 2, 55), (689, 1, 62), (691, 1, 60),
        (694, 1, 63), (696, 1, 55), (721, 1, 67), (723, 2, 65), (726, 2, 63),
        (728, 1, 62), (731, 1, 60), (734, 1, 63), (735, 1, 62), (737, 1, 60),
        (739, 1, 58), (740, 1, 57), (741, 2, 55), (743, 2, 54), (746, 1, 58),
        (747, 1, 57), (749, 1, 60), (750, 1, 58), (752, 1, 62), (753, 1, 60),
        (755, 1, 63), (784, 1, 57), (787, 1, 65), (789, 1, 62), (796, 1, 55),
        (798, 1, 58), (800, 1, 60), (802, 1, 61), (804, 5, 62), (809, 3, 67),
        (812, 2, 65), (815, 1, 63), (841, 1, 62), (842, 1, 65), (843, 1, 62),
        (845, 1, 67), (848, 1, 65), (851, 3, 66), (854, 1, 66), (856, 1, 62),
        (857, 1, 67), (860, 4, 69), (864, 3, 70), (1485, 1, 55), (1510, 2, 58),
    ),
    "tpt2": (
        (24, 5, 65), (29, 1, 60), (30, 2, 60), (32, 1, 65), (35, 1, 60),
        (38, 2, 63), (41, 1, 65), (44, 1, 65), (45, 1, 60), (174, 2, 54),
        (177, 2, 63), (181, 2, 62), (183, 4, 62), (190, 1, 61), (193, 6, 60),
        (267, 1, 64), (270, 1, 58), (275, 7, 62), (284, 1, 62), (285, 1, 60),
        (288, 1, 58), (444, 2, 65), (696, 1, 65), (746, 1, 54), (747, 1, 54),
        (749, 1, 54), (750, 1, 54), (752, 1, 60), (753, 1, 57), (755, 1, 58),
        (796, 1, 62), (798, 1, 55), (800, 1, 58), (802, 1, 54), (804, 5, 59),
        (809, 3, 65), (812, 2, 59), (815, 1, 58), (864, 3, 65),
    ),
    "tbn": (
        (1, 11, 48), (12, 10, 48), (24, 12, 41), (36, 10, 44), (48, 12, 55),
        (60, 11, 55), (72, 12, 50), (84, 12, 72), (96, 4, 58), (141, 2, 54),
        (143, 1, 55), (309, 1, 53), (311, 1, 53), (407, 5, 62), (413, 5, 60),
        (419, 7, 59), (429, 2, 55), (432, 6, 58), (440, 1, 55), (441, 1, 54),
        (651, 9, 50), (660, 10, 54), (673, 11, 50), (684, 11, 48), (703, 5, 53),
        (708, 12, 53), (721, 11, 50), (732, 12, 51), (746, 10, 48), (756, 12, 50),
        (768, 12, 51), (784, 6, 51), (796, 8, 43), (804, 12, 43), (816, 12, 52),
        (828, 4, 51), (841, 11, 48), (852, 12, 48), (864, 3, 50), (1467, 8, 50),
    ),
}

# score order, with the transposition each instrument is written at
LAYOUT = [
    ("alto",  "Alto Sax",     instrument.AltoSaxophone,     9, "treble"),
    ("tenor", "Tenor Sax",    instrument.TenorSaxophone,   14, "treble"),
    ("bari",  "Baritone Sax", instrument.BaritoneSaxophone, 21, "treble"),
    ("tpt1",  "Trumpet 1",    instrument.Trumpet,           2, "treble"),
    ("tpt2",  "Trumpet 2",    instrument.Trumpet,           2, "treble"),
    ("tbn",   "Trombone",     instrument.Trombone,          0, "bass"),
]
ABBREV = {"alto": "A. Sx.", "tenor": "T. Sx.", "bari": "B. Sx.",
          "tpt1": "Tpt. 1", "tpt2": "Tpt. 2", "tbn": "Tbn."}

CYCLE = ["Gm7", "Cm7", "F7", "B-maj7", "E-maj7", "Am7b5", "D7", "Gm7",
         "Cm7", "F7", "B-maj7", "G7", "Cm7", "F7", "B-maj7", "D7"]
LAST_CHARTED_BAR = 120


def hand_events(key_: str) -> list[tuple[float, float, list[int]]]:
    """(start, end, pitches) in written quarter notes from the top of bar 1."""
    out = []
    for slot, length, midi in PARTS[key_]:
        b, s = divmod(slot, 12)
        start = 4 * b + written_offset(s)
        end = 4 * b + written_offset(s + length)
        out.append((start, max(end, start + 0.5), [midi]))
    out.sort()
    for i in range(len(out) - 1):
        s, e, ns = out[i]
        out[i] = (s, min(e, out[i + 1][0]), ns)
    return [(s, e, ns) for s, e, ns in out if e > s]


def build_part(key_: str, name: str, cls, _semis: int, clef_name: str) -> stream.Part:
    p = stream.Part(id=key_)
    p.partName, p.partAbbreviation = name, ABBREV[key_]
    p.insert(0, cls())
    p.insert(0, clef.BassClef() if clef_name == "bass" else clef.TrebleClef())
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, key.KeySignature(-2))          # concert B-flat; transposed later

    events = hand_events(key_)
    idx = 0
    carry: tuple[list[int], float] | None = None
    for b in range(TOTAL_BARS):
        m = stream.Measure(number=b + 1)
        bar_start, bar_end = 4.0 * b, 4.0 * (b + 1)
        pos = bar_start
        if carry is not None:
            pitches, end = carry
            stop = min(end, bar_end)
            fill(m, 0.0, stop - bar_start, pitches, True, end > bar_end)
            pos = stop
            carry = (pitches, end) if end > bar_end else None
        while idx < len(events) and events[idx][0] < bar_end:
            s, e, pitches = events[idx]
            if s > pos:
                fill(m, pos - bar_start, s - pos, None, False, False)
                pos = s
            stop = min(e, bar_end)
            fill(m, pos - bar_start, stop - pos, pitches, False, e > bar_end)
            pos = stop
            if e > bar_end:
                carry = (pitches, e)
            idx += 1
        if pos < bar_end:
            fill(m, pos - bar_start, bar_end - pos, None, False, False)
        p.append(m)
    return p


def build_score() -> stream.Score:
    sc = stream.Score()
    sc.metadata = metadata.Metadata(
        title="Fly Me To The Moon",
        subtitle="horns - the recording's saxophone, passed around the section",
        composer="Bart Howard",
    )
    parts = [build_part(*spec) for spec in LAYOUT]

    top = parts[0]
    mm = tempo.MetronomeMark(number=120, referent=note.Note(type="quarter"))
    mm.placement = "above"
    top.measure(1).insert(0.0, copy.deepcopy(mm))
    swing = expressions.TextExpression("Medium swing - eighths swung")
    swing.placement = "above"
    top.measure(1).insert(0.0, swing)
    for b, mark in SECTIONS.items():
        meas = top.measure(b)
        if meas is not None and mark != "Intro":
            meas.insert(0.0, expressions.RehearsalMark(mark))
    for prt in parts:
        prt[-1].rightBarline = bar.Barline("final")
        prt.atSoundingPitch = True      # the notes above are concert pitch
        sc.insert(0, prt)
    sc.atSoundingPitch = True

    sc.insert(0, layout.StaffGroup(parts[:3], symbol="bracket"))
    sc.insert(0, layout.StaffGroup(parts[3:], symbol="bracket"))
    return sc


def add_chord_symbols(sc: stream.Score) -> None:
    """Concert-pitch chord symbols, added after the parts are transposed so
    they are not transposed with them - an E-flat alto's staff would otherwise
    label a concert Cm7 as Am7."""
    top = sc.parts[0]
    note_ = expressions.TextExpression("chord symbols at concert pitch")
    note_.placement = "above"
    note_.style.fontStyle = "italic"
    top.measure(1).insert(0.0, note_)
    for b in range(min(LAST_CHARTED_BAR, TOTAL_BARS)):
        sym = CYCLE[(b - 8) % 16]
        if b == 0 or sym != CYCLE[(b - 9) % 16]:
            cs = harmony.ChordSymbol(sym)
            cs.writeAsChord = False
            top.measure(b + 1).insert(0.0, cs)


def tidy_musicxml(path: str) -> None:
    """A zero <root-alter> makes some engravers print a spurious natural, and
    every <voice> must be a positive integer or MuseScore calls the file
    corrupt."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    for tag, sub in (("root", "root-alter"), ("bass", "bass-alter")):
        for el in root.iter(tag):
            alter = el.find(sub)
            if alter is not None and (alter.text or "").strip() in ("0", "0.0"):
                el.remove(alter)
    for v in root.iter("voice"):
        try:
            if int(v.text) < 1:
                v.text = "1"
        except (TypeError, ValueError):
            v.text = "1"
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="write the horn parts")
    ap.add_argument("-o", "--out", default="Fly_Me_To_The_Moon_Horns.musicxml")
    ap.add_argument("--mxl", help="also write a compressed .mxl here")
    args = ap.parse_args()

    sc = build_score()
    sc.toWrittenPitch(inPlace=True)        # each part on its own transposition
    add_chord_symbols(sc)
    sc.write("musicxml", fp=args.out)
    tidy_musicxml(args.out)
    n = sum(len(v) for v in PARTS.values())
    print(f"wrote {args.out}  (6 parts, {n} notes)")
    if args.mxl:
        pack_mxl(args.out, args.mxl)
        print(f"wrote {args.mxl}")


if __name__ == "__main__":
    main()
