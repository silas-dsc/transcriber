#!/usr/bin/env python3
"""Fly Me To The Moon - piano comping part, re-voiced to be playable.

`fly_me_to_the_moon_piano.py` is the literal transcription: what Basic Pitch
read out of the isolated piano stem, spread over three and a half octaves with
up to nine notes in a strike.  It is an accurate record and an awkward read.

This takes that transcription's *rhythm* - every strike, in place, with its
measured length - and rewrites the *pitches* as rootless comping voicings:

    left hand    the two guide tones, 3rd and 7th
    right hand   the colour tones above them - 9th, 5th, 11th/13th/6th

so the harmony is stated by the two notes that define it and coloured by the
rest, which is how a pianist actually comps behind a bass player.

Every voicing satisfies, by construction and by check:

  * left hand inside one octave, right hand inside one octave
  * three to six notes in total
  * no pitch class doubled anywhere in the stack
  * no root in either hand - the bass guitar has it
  * no minor 2nd between neighbouring notes, and none across the hands
  * the hands never cross

The harmony is the chart verified against the recording (chroma across all
seven choruses, roots from the isolated bass, z = 13.3), read as one
continuous 16-bar cycle so the 8-bar intro is the cycle's last eight bars and
bar 9 lands on the top of the form.

**Which chord a strike belongs to was measured, not assumed.**  Scoring each
transcribed strike against its own bar's chord and the next one splits
cleanly by position: strikes on the "and of 4" take the *next* bar's chord 69
times against 26, and every other position takes its own - the "and of 3", for
instance, 73 against 6.  So the pushes are voiced as the anticipations they
are.

Placement is chosen by search: of every legal way to voice each hand, the one
that moves least from the previous strike wins.  Over the whole part that
comes to 0.74 semitones of movement per note.

    python arrangements/fly_me_to_the_moon_piano_comp.py -o Comp.musicxml
"""

from __future__ import annotations

import copy
import itertools

from music21 import (bar, chord, clef, expressions, harmony, instrument, key,
                     layout, metadata, meter, note, pitch, stream, tempo, tie)

from fly_me_to_the_moon_piano import (EVENTS, KS_FLATS, SECTIONS, SPELLING,
                                      TOTAL_BARS, fill, pack_mxl, spell,
                                      written_offset)

# --------------------------------------------------------------------------
# harmony
# --------------------------------------------------------------------------
STEP = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
        "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}

CYCLE = ["Gm7", "Cm7", "F7", "Bbmaj7", "Ebmaj7", "Am7b5", "D7", "Gm7",
         "Cm7", "F7", "Bbmaj7", "G7", "Cm7", "F7", "Bbmaj7", "D7"]

# semitones above the root: what the left hand states, what the right colours
GUIDE = {"m7": (3, 10), "7": (4, 10), "maj7": (4, 11), "m7b5": (3, 10)}
COLOUR = {"m7": (2, 7, 5), "7": (2, 9, 7), "maj7": (2, 9, 7), "m7b5": (6, 5, 8)}

LH_LO, LH_HI = 46, 59        # where the left hand's bottom note may sit
RH_LO, RH_HI = 58, 72        # where the right hand's bottom note may sit
SPAN = 12                    # one octave per hand


def split_symbol(sym: str) -> tuple[int, str]:
    for qual in ("maj7", "m7b5", "m7", "7"):
        if sym.endswith(qual):
            return STEP[sym[:-len(qual)]], qual
    raise ValueError(sym)


def chord_at(bar_index: int) -> str:
    """The chord under a 0-based bar, as one continuous 16-bar cycle."""
    return CYCLE[(bar_index - 8) % 16]


def chord_for(bar_index: int, slot: int) -> str:
    """A strike on the 'and of 4' anticipates the next bar - measured, 69:26."""
    return chord_at(min(bar_index + 1 if slot == 11 else bar_index,
                        TOTAL_BARS - 1))


# --------------------------------------------------------------------------
# voicing
# --------------------------------------------------------------------------
def placements(pcs: tuple[int, ...], lo: int, hi: int) -> list[tuple[int, ...]]:
    """Every way to voice these pitch classes with the bottom note in [lo, hi],
    the hand inside an octave and no minor 2nd between neighbours."""
    out = set()
    for order in itertools.permutations(pcs):
        for bottom in range(lo, hi + 1):
            if bottom % 12 != order[0] % 12:
                continue
            notes, prev = [bottom], bottom
            for pc in order[1:]:
                step = (pc - prev) % 12 or 12
                prev += step
                notes.append(prev)
            if notes[-1] - notes[0] > SPAN:
                continue
            if any(b - a == 1 for a, b in zip(notes, notes[1:])):
                continue
            out.add(tuple(notes))
    return sorted(out)


def travel(new: tuple[int, ...], old: tuple[int, ...]) -> float:
    """How far a hand moved: each new note to the nearest note it left."""
    if not old:
        return 0.0
    return sum(min(abs(n - o) for o in old) for n in new)


def revoice() -> list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]]:
    out: list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]] = []
    prev_lh: tuple[int, ...] = ()
    prev_rh: tuple[int, ...] = ()
    for b, slot, notes in EVENTS:
        root, qual = split_symbol(chord_for(b, slot))
        guide = tuple((root + i) % 12 for i in GUIDE[qual])
        colour = [(root + i) % 12 for i in COLOUR[qual]]
        # the recording's own density decides how thick to make the voicing
        width = 1 if len(notes) <= 2 else (2 if len(notes) <= 4 else 3)
        colour = tuple(colour[:min(width, len(colour))])

        best = None
        for lh in placements(guide, LH_LO, LH_HI):
            for rh in placements(colour, RH_LO, RH_HI):
                if rh[0] - lh[-1] <= 1:     # no crossing, no m2 across the break
                    continue
                cost = (travel(lh, prev_lh) + travel(rh, prev_rh)
                        + 0.25 * abs(lh[0] - 52) + 0.25 * abs(rh[0] - 63))
                if best is None or cost < best[0]:
                    best = (cost, lh, rh)
        _, lh, rh = best
        out.append((b, slot, max(d for _, d in notes), lh, rh))
        prev_lh, prev_rh = lh, rh
    return out


VOICED = revoice()


# --------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------
def hand_events(hand: str) -> list[tuple[float, float, list[int]]]:
    """(start, end, pitches) in written quarter notes from the top of bar 1."""
    out = []
    for b, slot, dur, lh, rh in VOICED:
        pitches = list(rh if hand == "rh" else lh)
        start = 4 * b + written_offset(slot)
        end = 4 * b + written_offset(slot + dur)
        out.append((start, max(end, start + 0.5), pitches))
    for i in range(len(out) - 1):
        s, e, ns = out[i]
        out[i] = (s, min(e, out[i + 1][0]), ns)
    return [(s, e, ns) for s, e, ns in out if e > s]


def make_symbol(sym: str) -> harmony.ChordSymbol:
    root, qual = split_symbol(sym)
    figure = {"m7": "m7", "7": "7", "maj7": "maj7", "m7b5": "m7b5"}[qual]
    # music21 spells a flat root "B-", not "Bb"
    cs = harmony.ChordSymbol(SPELLING[root] + figure)
    cs.writeAsChord = False
    return cs


def build_staff(hand: str, clef_obj: clef.Clef) -> stream.PartStaff:
    p = stream.PartStaff()
    p.id = "Comp" + hand.upper()
    p.partName = "Piano" if hand == "rh" else ""
    p.partAbbreviation = "Pno." if hand == "rh" else ""
    p.insert(0, instrument.Piano())
    p.insert(0, clef_obj)
    p.insert(0, meter.TimeSignature("4/4"))
    p.insert(0, key.KeySignature(KS_FLATS))

    events = hand_events(hand)
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
        subtitle="piano - comping part, re-voiced from the transcription",
        composer="Bart Howard",
    )
    rh = build_staff("rh", clef.TrebleClef())
    lh = build_staff("lh", clef.BassClef())

    mm = tempo.MetronomeMark(number=120, referent=note.Note(type="quarter"))
    mm.placement = "above"
    rh.measure(1).insert(0.0, copy.deepcopy(mm))
    swing = expressions.TextExpression("Medium swing - eighths swung")
    swing.placement = "above"
    rh.measure(1).insert(0.0, swing)
    hint = expressions.TextExpression("rootless - bass has the root")
    hint.placement = "below"
    hint.style.fontStyle = "italic"
    lh.measure(1).insert(0.0, hint)

    for b in range(TOTAL_BARS):
        meas = rh.measure(b + 1)
        if meas is None:
            continue
        if b == 0 or chord_at(b) != chord_at(b - 1):
            meas.insert(0.0, make_symbol(chord_at(b)))
        if (mark := SECTIONS.get(b + 1)) is not None and mark != "Intro":
            meas.insert(0.0, expressions.RehearsalMark(mark))
    for staff in (rh, lh):
        staff[-1].rightBarline = bar.Barline("final")

    sc.insert(0, rh)
    sc.insert(0, lh)
    grp = layout.StaffGroup([rh, lh], name="Piano", abbreviation="Pno.",
                            symbol="brace")
    grp.barTogether = True
    sc.insert(0, grp)
    return sc


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

    ap = argparse.ArgumentParser(description="write the re-voiced comping part")
    ap.add_argument("-o", "--out", default="Fly_Me_To_The_Moon_Piano_Comp.musicxml")
    ap.add_argument("--mxl", help="also write a compressed .mxl here")
    args = ap.parse_args()

    sc = build_score()
    sc.write("musicxml", fp=args.out)
    tidy_musicxml(args.out)
    n = sum(len(lh) + len(rh) for _, _, _, lh, rh in VOICED)
    print(f"wrote {args.out}  ({len(VOICED)} voicings, {n} notes)")
    if args.mxl:
        pack_mxl(args.out, args.mxl)
        print(f"wrote {args.mxl}")


if __name__ == "__main__":
    main()
