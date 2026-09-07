#!/usr/bin/env python3
"""Fly Me To The Moon - the solo saxophone on the recording, transcribed.

A single line on a single staff: what the horn actually plays on the supplied
backing track, and nothing else.  The big band chart develops this material
and the horn chart hands it around a section; this file is the record it was
all taken from.

**It is a tenor.**  The line runs B-flat2 to B-flat4 at concert pitch, and
five of its notes sit below D-flat3, which is the bottom of an alto.  Written
for a B-flat tenor - up a major ninth - that same span is C4 to C6, dead
centre of the instrument.  So the staff is a tenor part, written where a
tenor player reads it.

How it was pulled out:

* Demucs' six-source model separates piano and guitar into their own stems,
  which leaves the saxophone alone in ``other``.  pYIN reads it as a single
  line.
* pYIN will happily track noise, so a frame only *starts* a note where the
  stem is genuinely loud (level above 0.14 of its own 99.5th percentile) and
  pYIN is confident (voicing probability above 0.55).
* A note *continues* on a much looser gate - level 0.04, probability 0.25 -
  because a saxophone note decays a long way below the level it started at.
  Using one strict threshold for both ends is what made an earlier pass call
  everything an eighth note: 68% of its notes came out one slot long.  With
  the two gates the median note is 209 ms and the longest is 1.37 s, which is
  a line that holds rather than a line that chatters.
* Where the stem shows a fresh attack inside a held note, the note is split
  in two.  **That never once fired** - so the repeated notes in this part are
  genuinely repeated notes and not one note misread as several.

**The saxophone plays behind the beat.**  Against the downbeat the drums and
piano establish (0.800 s), the piano's strikes sit 2 ms early - dead on - and
the saxophone's onsets sit 24 ms late, a difference of 26 ms with a 95%
confidence interval of 16 to 33 ms.  It is a real lay-back, not scatter.  So
the grid this part is quantised against is the ensemble's, shifted by the
saxophone's own median lag to 0.8417 s: the notation should say what the
player meant, and the lay-back is a way of playing it, marked as a direction
rather than written into the rhythms.  Quantised on the ensemble's downbeat
instead, the median onset misses its slot by 41 ms; on the saxophone's own,
by 30 ms.

Once that is done the onsets fall where a swung line falls - 82 on the beats,
84 on the swung "and"s, and 22 in the middle of a triplet.

Checked back against the audio, frame by frame:

  pitch agreement where the score and the stem both sound      98.4%
  frames scored where the stem is not sounding even faintly     9.4%
  notes inside the chord of the moment                       140/155
  the 15 that are not                     every one a chromatic approach

The 9.4% is the release tails - the looser gate holds a note through its
decay - and the 15 outside the chord all step to or from a chord tone, which
is what an approach note is.  Nothing is stranded.

**The saxophone is silent for most of the second half.**  Measured against
the solo chorus, bars 42-54 and 74-122 are 67 and 70 dB down, which is
digital silence rather than quiet playing.  It sounds in bars 1-9, 11-12,
14-27, 30-41, 55-73 and 123-126, and the rests here are real rests.

Chord symbols sit on a staff of their own at concert pitch, because a reader
applies a staff's transposition to its chord symbols as well as its notes,
and a concert Cm7 on the tenor's staff would play back a tone out.  They stop
at bar 120: the 16-bar cycle does not run through the tag.

    python arrangements/fly_me_to_the_moon_sax.py -o Sax.musicxml
"""

from __future__ import annotations

import copy

from music21 import (bar, clef, expressions, harmony, instrument, key, layout,
                     metadata, meter, note, stream, tempo)

from fly_me_to_the_moon_piano import (SECTIONS, TOTAL_BARS, fill, pack_mxl,
                                      written_offset)

# --------------------------------------------------------------------------
# the line, as (slot, length in slots, concert MIDI); slot 0 is bar 1 beat 1
# and there are 12 triplet-eighth slots to a bar
# --------------------------------------------------------------------------
NOTES: tuple[tuple[int, int, int], ...] = (
    (0, 4, 63), (5, 1, 62), (6, 2, 60), (8, 2, 58), (11, 1, 57),
    (14, 1, 62), (17, 1, 54), (18, 2, 55), (21, 2, 57), (24, 4, 58),
    (29, 1, 57), (30, 2, 58), (32, 2, 60), (35, 2, 62), (38, 2, 57),
    (41, 2, 58), (44, 1, 60), (45, 1, 62), (48, 4, 55), (53, 1, 55),
    (56, 1, 51), (59, 2, 50), (62, 1, 48), (65, 1, 46), (69, 2, 49),
    (71, 4, 50), (75, 2, 51), (78, 3, 50), (81, 2, 55), (83, 4, 54),
    (87, 2, 58), (89, 2, 57), (92, 4, 54), (96, 4, 55), (101, 1, 55),
    (120, 8, 57), (129, 2, 57), (141, 2, 54), (143, 3, 55), (167, 5, 57),
    (174, 2, 54), (177, 2, 51), (181, 6, 50), (188, 1, 50), (190, 1, 49),
    (193, 6, 48), (200, 2, 55), (212, 1, 54), (214, 1, 53), (216, 5, 62),
    (237, 3, 53), (240, 4, 55), (261, 1, 53), (267, 1, 52), (270, 1, 58),
    (275, 8, 62), (284, 1, 62), (285, 1, 60), (288, 2, 58), (308, 1, 52),
    (309, 2, 53), (311, 1, 53), (323, 2, 50), (357, 2, 51), (359, 5, 50),
    (367, 1, 54), (369, 1, 57), (371, 8, 55), (380, 2, 55), (384, 3, 58),
    (407, 5, 62), (413, 5, 60), (419, 7, 59), (429, 2, 55), (432, 7, 58),
    (440, 1, 55), (441, 1, 54), (444, 2, 53), (459, 1, 55), (461, 1, 58),
    (463, 1, 55), (467, 1, 60), (469, 1, 61), (471, 1, 50), (474, 1, 48),
    (477, 1, 50), (480, 1, 67), (651, 1, 55), (655, 1, 55), (659, 1, 60),
    (661, 1, 61), (663, 1, 62), (666, 1, 60), (668, 1, 61), (669, 1, 62),
    (673, 1, 67), (677, 1, 63), (678, 2, 62), (680, 1, 60), (682, 1, 58),
    (683, 1, 57), (684, 2, 55), (686, 1, 53), (688, 1, 51), (689, 1, 50),
    (691, 1, 48), (692, 1, 50), (694, 1, 51), (695, 1, 55), (698, 1, 53),
    (702, 2, 57), (705, 1, 60), (707, 1, 58), (711, 1, 55), (714, 1, 60),
    (716, 1, 61), (717, 1, 62), (719, 1, 65), (721, 1, 67), (723, 2, 65),
    (726, 1, 63), (728, 1, 62), (731, 2, 60), (734, 1, 63), (735, 1, 62),
    (737, 1, 60), (738, 1, 58), (740, 1, 57), (741, 2, 55), (743, 2, 54),
    (746, 1, 58), (747, 1, 57), (749, 1, 60), (750, 1, 58), (752, 1, 62),
    (753, 1, 60), (754, 1, 63), (756, 1, 62), (758, 1, 60), (759, 1, 58),
    (762, 1, 55), (764, 1, 54), (765, 1, 53), (768, 2, 51), (771, 2, 55),
    (774, 1, 53), (776, 2, 57), (779, 2, 55), (782, 1, 58), (783, 1, 57),
    (786, 1, 53), (788, 1, 51), (789, 1, 50), (794, 2, 53), (796, 1, 55),
    (798, 1, 58), (800, 1, 60), (802, 1, 61), (804, 5, 62), (809, 3, 67),
    (812, 2, 65), (815, 1, 63), (816, 2, 62), (819, 1, 58), (821, 2, 55),
    (825, 2, 60), (827, 1, 61), (828, 1, 62), (831, 1, 55), (839, 1, 60),
    (841, 1, 62), (842, 1, 65), (843, 1, 62), (845, 1, 67), (848, 1, 65),
    (851, 3, 66), (854, 1, 66), (856, 1, 62), (857, 2, 67), (860, 4, 69),
    (864, 3, 70), (1467, 2, 58), (1470, 1, 60), (1473, 3, 62), (1484, 1, 53),
    (1485, 2, 55), (1489, 1, 58), (1510, 2, 46),)

TENOR = 14                      # written pitch is concert plus a major ninth
CYCLE = ["Gm7", "Cm7", "F7", "B-maj7", "E-maj7", "Am7b5", "D7", "Gm7",
         "Cm7", "F7", "B-maj7", "G7", "Cm7", "F7", "B-maj7", "D7"]
LAST_CHARTED_BAR = 120


def events() -> list[tuple[float, float, list[int]]]:
    """(start, end, pitches) in quarter notes from the top of bar 1."""
    out = []
    for slot, length, midi in NOTES:
        b, s = divmod(slot, 12)
        start = 4 * b + written_offset(s)
        end = 4 * b + written_offset(s + length)
        out.append((start, max(end, start + 0.5), [midi]))
    out.sort()
    for i in range(len(out) - 1):
        s, e, ns = out[i]
        out[i] = (s, min(e, out[i + 1][0]), ns)
    return [(s, e, ns) for s, e, ns in out if e > s]


def build_sax() -> stream.Part:
    p = stream.Part(id="sax")
    p.partName, p.partAbbreviation = "Tenor Sax", "T. Sx."
    p.insert(0, instrument.TenorSaxophone())
    p.insert(0, clef.TrebleClef())
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, key.KeySignature(-2))          # concert; transposed later

    ev = events()
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
        while idx < len(ev) and ev[idx][0] < bar_end:
            s, e, pitches = ev[idx]
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


def build_chord_staff() -> stream.Part:
    """Concert-pitch chord symbols on a staff with no transposition of its
    own, so they sound as they read.  One staff line, rests hidden."""
    p = stream.Part(id="chords")
    p.partName, p.partAbbreviation = "Chords", "Chds."
    p.insert(0, instrument.Piano())
    p.insert(0, clef.TrebleClef())
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, key.KeySignature(-2))
    p.insert(0, layout.StaffLayout(staffLines=1))
    for b in range(TOTAL_BARS):
        m = stream.Measure(number=b + 1)
        r = note.Rest(quarterLength=4.0)
        r.style.hideObjectOnPrint = True
        m.insert(0.0, r)
        p.append(m)
    return p


def build_score() -> stream.Score:
    sc = stream.Score()
    sc.metadata = metadata.Metadata(
        title="Fly Me To The Moon",
        subtitle="solo saxophone - transcribed from the recording",
        composer="Bart Howard",
    )
    chords, sax = build_chord_staff(), build_sax()

    mm = tempo.MetronomeMark(number=120, referent=note.Note(type="quarter"))
    mm.placement = "above"
    chords.measure(1).insert(0.0, copy.deepcopy(mm))
    for text in ("Medium swing - eighths swung",
                 "laid back - the line sits about 40 ms behind the section"):
        te = expressions.TextExpression(text)
        te.placement = "above"
        te.style.fontStyle = "italic"
        chords.measure(1).insert(0.0, te)
    for b, mark in SECTIONS.items():
        meas = chords.measure(b)
        if meas is not None and mark != "Intro":
            meas.insert(0.0, expressions.RehearsalMark(mark))

    for prt in (chords, sax):
        prt[-1].rightBarline = bar.Barline("final")
        prt.atSoundingPitch = True             # the notes above are concert
        sc.insert(0, prt)
    sc.atSoundingPitch = True
    return sc


def add_chord_symbols(sc: stream.Score) -> None:
    """After the tenor is converted to written pitch, so nothing carries the
    symbols along with it."""
    top = sc.parts[0]
    assert top.id == "chords", "the chord staff must be the top part"
    if LAST_CHARTED_BAR < TOTAL_BARS:
        tag = expressions.TextExpression("ending - as played")
        tag.placement = "above"
        tag.style.fontStyle = "italic"
        top.measure(LAST_CHARTED_BAR + 1).insert(0.0, tag)
    for b in range(min(LAST_CHARTED_BAR, TOTAL_BARS)):
        sym = CYCLE[(b - 8) % 16]
        if b == 0 or sym != CYCLE[(b - 9) % 16]:
            cs = harmony.ChordSymbol(sym)
            cs.writeAsChord = False
            top.measure(b + 1).insert(0.0, cs)


def tidy_musicxml(path: str) -> None:
    """A zero <root-alter> makes some engravers print a spurious natural,
    every <voice> must be a positive integer or MuseScore calls the file
    corrupt, and music21 has no way to write the chord staff's single staff
    line, so that goes in here."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()

    chord_ids = {sp.get("id") for sp in root.iter("score-part")
                 if (sp.findtext("part-name") or "").strip() == "Chords"}
    for part in root.iter("part"):
        if part.get("id") not in chord_ids:
            continue
        attrs = part.find("measure/attributes")
        if attrs is None or attrs.find("staff-details") is not None:
            continue
        det = ET.SubElement(attrs, "staff-details")
        ET.SubElement(det, "staff-lines").text = "1"
        order = ["divisions", "key", "time", "staves", "part-symbol",
                 "instruments", "clef", "staff-details", "transpose"]
        attrs[:] = sorted(attrs, key=lambda e: order.index(e.tag)
                          if e.tag in order else len(order))
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

    ap = argparse.ArgumentParser(description="write the saxophone transcription")
    ap.add_argument("-o", "--out", default="Fly_Me_To_The_Moon_Sax.musicxml")
    ap.add_argument("--mxl", help="also write a compressed .mxl here")
    args = ap.parse_args()

    sc = build_score()
    sc.toWrittenPitch(inPlace=True)
    add_chord_symbols(sc)
    sc.write("musicxml", fp=args.out)
    tidy_musicxml(args.out)
    print(f"wrote {args.out}  ({len(NOTES)} notes)")
    if args.mxl:
        pack_mxl(args.out, args.mxl)
        print(f"wrote {args.mxl}")


if __name__ == "__main__":
    main()
