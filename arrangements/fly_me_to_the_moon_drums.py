#!/usr/bin/env python3
"""Drum transcription of the Fly Me To The Moon backing track.

This is not an arrangement - it is what the drummer on the recording actually
plays, written out bar by bar.  The part was recovered from the audio like
this:

1.  Demucs (``htdemucs``) separated the drum kit from the rest of the mix.
2.  The beat grid was fitted to the kick pulse by least squares:
    120.002 BPM - near enough to a rock-steady 120 - with beat 1 of bar 1 at
    0.80 s.  The bar phase was settled by the bass line: taking the note the
    bass plays on each candidate downbeat, only one of the four phases makes
    the roots move in the descending fourths the tune is built from
    (Eb - A - D - G - C - F - Bb), and it does so 63% of the time against a
    22% baseline for the other three.
3.  Each slot of a triplet-eighth grid (three per beat, 12 per bar) was
    scored for onset strength in six frequency bands - 20-110 Hz, 110-260,
    260-900, 2.5-7k, 7-12.5k and 12.5-20k - and the bands' balance decides
    which piece of the kit was struck.  The hi-hat is unmistakable: on beats
    2 and 4 the strike is four times brighter (12.5-20 kHz against 7-12.5)
    and decays far faster than anything else on the record.

What that leaves is the whole part: bass drum feathering four to the bar,
hi-hat on 2 and 4, a snare that comps and fills, and seven cymbal crashes.
There is no ride pattern anywhere on the recording - the 7-12.5 kHz band
carries only 13 events that are not the hi-hat, where a swing ride would
have left something like 770.

Bars whose notation is an exact repeat of the bar before carry a ``%``.

    python arrangements/fly_me_to_the_moon_drums.py -o Drums.musicxml
"""

from __future__ import annotations

from music21 import (articulations, bar, clef, duration, expressions,
                     instrument, layout, metadata, meter, note, percussion,
                     stream, tempo)

# --------------------------------------------------------------------------
# the transcription
#
# One tuple per bar: (bass drum, snare, hi-hat, crash).  Each string is the
# twelve triplet-eighth slots of a 4/4 bar - slot 0 is beat 1, slot 3 beat 2,
# and so on; a capital S is an accented snare stroke.
# --------------------------------------------------------------------------
PICKUP = ("...........o", "........sss.", "............", "............")

BARS: list[tuple[str, str, str, str]] = [
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #   1
    ("o..o..o..o..", "........s..s", "...h.....h..", "............"),   #   2
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #   3
    ("o..o..o..o..", "........s..s", "...h.....h..", "............"),   #   4
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #   5
    ("o..o..o..o..", "........s..s", "...h.....h..", "............"),   #   6
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #   7
    ("o..o.......o", "............", "...h........", "............"),   #   8
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #   9
    ("o..o..o..o..", "........s..s", "...h.....h..", "............"),   #  10
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  11
    ("o..o..o..o..", "........s..s", "...h.....h..", "............"),   #  12
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  13
    ("o..o..o..o..", "........s..s", "...h.....h..", "............"),   #  14
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  15
    ("o..o..o..o.o", "........s..s", "...h.....h..", "............"),   #  16
    ("o..o..o..o.o", "............", "...h.....h..", "............"),   #  17
    ("o..o..o..o.o", "........s..s", "...h.....h..", "............"),   #  18
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  19
    ("o..o..o..o..", "........s..s", "...h.....h..", "............"),   #  20
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  21
    ("o..o..o..o..", "........s...", "...h.....h..", "............"),   #  22
    ("o..o..o..o..", "..s....ss.ss", "...h........", "............"),   #  23
    ("o..o.....o.o", "sss.s.S.....", "...h........", ".........c.."),   #  24
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  25
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  26
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  27
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  28
    ("o..o..o..o..", ".....S......", "...h.....h..", "............"),   #  29
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  30
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  31
    ("...o..o..o..", "............", "...h.....h..", "............"),   #  32
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  33
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  34
    ("o..o..o..o..", ".........S..", "...h........", "............"),   #  35
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  36
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  37
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  38
    ("o..o..o..o..", ".......s...S", "...h.....h..", "........c.c."),   #  39
    ("o..o..o..o..", "....s.......", "...h.....h..", "c..........."),   #  40
    ("oo.o..o..o..", "s...........", "...h.....h..", "............"),   #  41
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  42
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  43
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  44
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  45
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  46
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  47
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  48
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  49
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  50
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  51
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  52
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  53
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  54
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  55
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  56
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  57
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  58
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  59
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  60
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  61
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  62
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  63
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  64
    ("o..o..o..o..", ".........S..", "...h........", "............"),   #  65
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  66
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  67
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  68
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  69
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  70
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  71
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  72
    ("o..o..o..o..", ".....S......", "...h.....h..", "............"),   #  73
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  74
    ("o..o..o..o..", ".....S......", "...h.....h..", "............"),   #  75
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  76
    ("o..o..o..o..", ".....s...S..", "...h........", "............"),   #  77
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  78
    ("o..o..o..o..", ".....s...S..", "...h........", "............"),   #  79
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  80
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  81
    ("o..o.....o..", "............", "...h.....h..", "............"),   #  82
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  83
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  84
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  85
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  86
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  87
    ("ooooooo..o.o", ".ss.ssSs....", "...h.....h..", "c..........."),   #  88
    ("o..o..o..o..", "S...........", "...h.....h..", "............"),   #  89
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  90
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  91
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  92
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  93
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  94
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   #  95
    ("o..o.....o..", "....s...S..S", "...h.....h..", "............"),   #  96
    ("o..o..o..o..", "s...........", "...h.....h..", "............"),   #  97
    ("o..o..o..o..", "............", "...h.....h..", "............"),   #  98
    ("o..o..o..o..", "s...........", "...h.....h..", "............"),   #  99
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 100
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 101
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   # 102
    ("o..o..o..o..", "............", "...h........", ".........c.."),   # 103
    ("o.o...o.....", "S.SSS..SS.sS", ".........h..", ".....c......"),   # 104
    ("o..o..o..o..", "Ss...s...s..", "...h........", "............"),   # 105
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 106
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   # 107
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 108
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   # 109
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   # 110
    ("o..o..o..o..", ".....s......", "...h.....h..", "............"),   # 111
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 112
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 113
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 114
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 115
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 116
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 117
    ("o..o..o..o..", ".....s..S..S", "...h.....h..", "............"),   # 118
    ("o..o..o..o..", ".....S..s...", "...h.....h..", "............"),   # 119
    ("o..o..o..o..", "....sSSSSS..", "...h........", "............"),   # 120
    ("o..o..o..o..", "............", "...h.....h..", "............"),   # 121
    ("o..o..o..o..", "........s...", "...h.....h..", "............"),   # 122
    ("o..o..o..o..", "..S.........", "...h.....h..", "............"),   # 123
    ("o...........", "............", "............", "............"),   # 124
    ("............", "....s.S..S.s", "...h........", "............"),   # 125
    ("........oo..", "S..s.....S..", "............", "............"),   # 126
    ("............", "............", "............", "............"),   # 127
    ("............", "............", "............", "............"),   # 128
]

# where each piece sits on the staff, and the notehead it takes
KIT = {
    "c": ("A5", "x"),        # crash
    "h": ("G5", "x"),        # hi-hat, closed
    "s": ("C5", None),       # snare
    "S": ("C5", None),       # snare, accented
    "o": ("F4", None),       # bass drum
}
UP = "chsS"                  # cymbals and snare take the up-stem voice
DOWN = "o"                   # the bass drum takes the down-stem voice

# one <score-instrument> per piece, or a reader stacks the kit on one line
DRUMSET = {
    ("A", 5): ("Crash Cymbal", 49),
    ("G", 5): ("Closed Hi-Hat", 42),
    ("C", 5): ("Snare Drum", 38),
    ("F", 4): ("Bass Drum", 36),
}

# 8 bars of intro, then seven 16-bar choruses, then an 8-bar tag
SECTIONS = {1: "Intro", 9: "A", 25: "B", 41: "C", 57: "D",
            73: "E", 89: "F", 105: "G", 121: "Tag"}


def _slots(row: str, beat: int) -> dict[int, str]:
    """The struck slots of one beat, as {0,1,2} -> the character struck."""
    return {i: row[beat * 3 + i] for i in range(3) if row[beat * 3 + i] != "."}


def _hit(ch: str) -> note.Unpitched | percussion.PercussionChord:
    step, octv = KIT[ch][0][0], int(KIT[ch][0][1:])
    n = note.Unpitched()
    n.displayStep, n.displayOctave = step, octv
    if KIT[ch][1]:
        n.noteheadFill = None
        n.notehead = KIT[ch][1]
    if ch == "S":
        n.articulations.append(articulations.Accent())
    return n


def _stack(chars: str):
    """One or more pieces struck together."""
    hits = [_hit(c) for c in chars]
    return hits[0] if len(hits) == 1 else percussion.PercussionChord(hits)


def _beat(rows: dict[str, str], beat: int, chars: str, stem: str):
    """Notate one beat of one voice: a list of (offset, element)."""
    used: dict[int, str] = {}
    for ch in chars:
        for slot in _slots(rows[ch], beat):
            used.setdefault(slot, "")
            used[slot] += ch
    triplet = any(slot % 3 == 1 for slot in used) or _triplet_beat(rows, beat)

    out = []
    if triplet:
        for slot in range(3):
            el = _stack(used[slot]) if slot in used else note.Rest()
            el.duration = duration.Duration(1 / 3)
            el.duration.appendTuplet(duration.Tuplet(3, 2, "eighth"))
            out.append((slot / 3, el))
    elif not used:
        out.append((0.0, note.Rest(quarterLength=1.0)))
    elif set(used) == {0}:
        out.append((0.0, _stack(used[0])))
        out[-1][1].quarterLength = 1.0
    else:
        for slot, off in ((0, 0.0), (2, 0.5)):
            el = _stack(used[slot]) if slot in used else note.Rest()
            el.quarterLength = 0.5
            out.append((off, el))
    for _, el in out:
        if isinstance(el, note.Rest):
            continue
        el.stemDirection = stem
    return out


def _triplet_beat(rows: dict[str, str], beat: int) -> bool:
    """A beat is a triplet beat if *any* voice uses the middle partial, so the
    two voices stay vertically aligned."""
    return any(row[beat * 3 + 1] != "." for row in rows.values())


def _measure(data: tuple[str, str, str, str], number: int, pickup: bool = False):
    rows = {"o": data[0], "s": data[1], "h": data[2], "c": data[3]}
    # an accented snare lives in the same row, spelled with a capital
    rows["S"] = "".join(c if c == "S" else "." for c in data[1])
    rows["s"] = "".join(c if c == "s" else "." for c in data[1])

    m = stream.Measure(number=number)
    beats = range(2, 4) if pickup else range(4)
    if pickup:
        m.paddingLeft = 2.0
    if not any(c != "." for row in data for c in row):
        r = note.Rest(quarterLength=4.0)
        r.fullMeasure = True
        m.insert(0, r)
        return m
    for chars, stem, vid in ((UP, "up", 1), (DOWN, "down", 2)):
        v = stream.Voice(id=vid)
        for k, beat in enumerate(beats):
            for off, el in _beat(rows, beat, chars, stem):
                v.insert(k + off, el)
        m.insert(0, v)
    return m


def build_score() -> stream.Score:
    sc = stream.Score()
    sc.metadata = metadata.Metadata(
        title="Fly Me To The Moon",
        subtitle="drums, transcribed from the recording",
        composer="Bart Howard",
    )
    p = stream.Part(id="Drums")
    kit = instrument.Percussion()
    kit.instrumentName = kit.partName = "Drums"
    kit.partAbbreviation = "Dr."
    p.insert(0, kit)
    p.partName, p.partAbbreviation = "Drums", "Dr."

    m0 = _measure(PICKUP, 0, pickup=True)
    m0.insert(0, clef.PercussionClef())
    m0.insert(0, meter.TimeSignature("4/4"))
    mm = tempo.MetronomeMark(number=120, referent=note.Note(type="quarter"))
    mm.placement = "above"
    m0.insert(0, mm)
    swing = expressions.TextExpression("Medium swing - eighths swung")
    swing.placement = "above"
    m0.insert(0, swing)
    p.append(m0)

    repeats: list[int] = []
    for i, data in enumerate(BARS):
        m = _measure(data, i + 1)
        if (name := SECTIONS.get(i + 1)) is not None:
            m.insert(0, expressions.RehearsalMark(name))
        if i and data == BARS[i - 1] and any(c != "." for row in data for c in row):
            repeats.append(i + 1)
        p.append(m)
    p[-1].rightBarline = bar.Barline("final")
    p.repeatBarNumbers = repeats

    sc.insert(0, p)
    sc.insert(0, layout.StaffGroup([p], symbol="bracket"))
    return sc


# --------------------------------------------------------------------------
# MusicXML post-passes
# --------------------------------------------------------------------------
def build_drumset(path: str) -> None:
    """Give the part a real kit.

    music21 writes one ``<score-instrument>``, so every unpitched note points
    at the same drum and a reader stacks the whole kit on one line whatever
    ``display-step`` we asked for.  Replace it with one instrument per piece
    and tag each note with the right one.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    ids = {}
    part_id = None
    for sp in root.iter("score-part"):
        part_id = sp.get("id")
        for child in list(sp):
            if child.tag in ("score-instrument", "midi-instrument"):
                sp.remove(child)
        for (step, octv), (label, gm) in sorted(DRUMSET.items()):
            iid = f"{part_id}-{step}{octv}"
            ids[(step, str(octv))] = iid
            si = ET.SubElement(sp, "score-instrument")
            si.set("id", iid)
            ET.SubElement(si, "instrument-name").text = label
            mi = ET.SubElement(sp, "midi-instrument")
            mi.set("id", iid)
            ET.SubElement(mi, "midi-channel").text = "10"
            # MusicXML numbers MIDI notes from 1, so a GM note needs +1
            ET.SubElement(mi, "midi-unpitched").text = str(gm + 1)
        break

    for n in root.iter("note"):
        u = n.find("unpitched")
        if u is None:
            continue
        iid = ids.get((u.findtext("display-step"), u.findtext("display-octave")))
        if iid is None:
            continue
        el = ET.Element("instrument")
        el.set("id", iid)
        kids = list(n)
        at = len(kids)
        for i, child in enumerate(kids):
            if child.tag in ("voice", "type", "dot", "accidental", "stem",
                             "notehead", "beam", "notations", "time-modification"):
                at = i
                break
        n.insert(at, el)
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def mark_measure_repeats(path: str, bars: list[int]) -> None:
    """Turn runs of identical bars into ``%`` signs.

    music21 has no measure-repeat object, so the ``<measure-style>`` markers
    go in afterwards: one at the head of each run and one on the bar after it
    ends.
    """
    import xml.etree.ElementTree as ET

    if not bars:
        return
    runs, run = [], [bars[0]]
    for b in sorted(bars)[1:]:
        if b == run[-1] + 1:
            run.append(b)
        else:
            runs.append(run)
            run = [b]
    runs.append(run)

    tree = ET.parse(path)
    root = tree.getroot()
    for part in root.iter("part"):
        by_no = {int(m.get("number")): m for m in part.iter("measure")
                 if (m.get("number") or "").lstrip("-").isdigit()}
        for run in runs:
            for bar_no, kind in ((run[0], "start"), (run[-1] + 1, "stop")):
                meas = by_no.get(bar_no)
                if meas is None:
                    continue
                attrs = meas.find("attributes")
                if attrs is None:
                    attrs = ET.Element("attributes")
                    meas.insert(0, attrs)
                style = ET.SubElement(attrs, "measure-style")
                rep = ET.SubElement(style, "measure-repeat")
                rep.set("type", kind)
                if kind == "start":
                    rep.text = "1"
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def clear_repeated_measures(path: str, bars: list[int]) -> None:
    """A bar showing ``%`` repeats the previous one, so it must be otherwise
    empty - a reader that finds notes there draws them *and* the repeat sign.

    The measure still has to account for its full duration, so the notes are
    replaced by a ``<forward>``, which advances time without printing.
    """
    import xml.etree.ElementTree as ET

    if not bars:
        return
    wanted = set(bars)
    tree = ET.parse(path)
    root = tree.getroot()
    divisions = None
    for part in root.iter("part"):
        for meas in part.iter("measure"):
            d = meas.findtext("attributes/divisions")
            if d:
                divisions = int(d)
            no = meas.get("number")
            if not (no or "").lstrip("-").isdigit() or int(no) not in wanted:
                continue
            for child in list(meas):
                if child.tag in ("note", "backup", "forward"):
                    meas.remove(child)
            fwd = ET.SubElement(meas, "forward")
            ET.SubElement(fwd, "duration").text = str((divisions or 10080) * 4)
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def tidy_musicxml(path: str) -> None:
    """Every <voice> must be a positive integer or MuseScore reports the file
    as corrupt."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    for v in root.iter("voice"):
        try:
            if int(v.text) < 1:
                v.text = "1"
        except (TypeError, ValueError):
            v.text = "1"
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def pack_mxl(src: str, dst: str) -> None:
    """Zip the MusicXML into the compressed .mxl MuseScore opens directly."""
    import os
    import zipfile

    inner = os.path.basename(src)
    container = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<container><rootfiles><rootfile full-path="%s" '
                 'media-type="application/vnd.recordare.musicxml+xml"/>'
                 '</rootfiles></container>\n' % inner)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("META-INF/container.xml", container)
        z.write(src, inner)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="write the drum transcription")
    ap.add_argument("-o", "--out", default="Fly_Me_To_The_Moon_Drums.musicxml")
    ap.add_argument("--mxl", help="also write a compressed .mxl here")
    args = ap.parse_args()

    sc = build_score()
    repeats = sc.parts[0].repeatBarNumbers
    sc.write("musicxml", fp=args.out)
    tidy_musicxml(args.out)
    build_drumset(args.out)
    mark_measure_repeats(args.out, repeats)
    clear_repeated_measures(args.out, repeats)
    print(f"wrote {args.out}  ({len(BARS)} bars, {len(repeats)} repeat bars)")
    if args.mxl:
        pack_mxl(args.out, args.mxl)
        print(f"wrote {args.mxl}")


if __name__ == "__main__":
    main()
