#!/usr/bin/env python3
"""Fly Me To The Moon - the piano part, made playable.

`fly_me_to_the_moon_piano.py` is the literal transcription: what Basic Pitch
read out of the isolated piano stem.  It is an accurate record and an awkward
read - the same note turns up in two or three octaves at once, strikes run to
nine notes, and nothing says which hand takes what.

This follows that transcription closely and changes only two things:

1.  **No pitch class sounds in more than one octave.**  That is what removes
    most of the clutter: 442 of the transcription's 1450 notes are octave
    duplicates of something already in the chord.
2.  **Every chord is divided between the hands**, as evenly as the chord
    allows, at a real gap near middle C.

Everything else is left alone.  The pitches are the transcribed pitches, the
rhythm is the transcribed rhythm, and the wide spreads stay wide - so the
written-out moments survive, including the ending, whose last three chords
walk C-E flat-B flat, C sharp-E, D-F-B flat onto the tonic.

Choosing *which* octave a pitch class keeps is what preserves the shape.  The
top note is the line and always stays; the bottom note anchors the chord and
stays unless it duplicates the top; every other pitch class keeps whichever of
its transcribed octaves leaves the chord most evenly spread.  Without that
last rule a six-note spread collapses into a cluster with a hole in it.

Chord symbols run to bar 120, where the verified 16-bar cycle applies.  They
stop there because the tag does not follow it: the cycle predicts Gm7, and
what is played is a B-flat.

    python arrangements/fly_me_to_the_moon_piano_comp.py -o Comp.musicxml
"""

from __future__ import annotations

import copy

from music21 import (bar, clef, expressions, harmony, instrument, key, layout,
                     metadata, meter, note, stream, tempo)

from fly_me_to_the_moon_piano import (EVENTS, KS_FLATS, SECTIONS, SPELLING,
                                      TOTAL_BARS, fill, pack_mxl,
                                      written_offset)

MAX_NOTES = 6      # more than a hand can grab
MAX_SPAN = 14      # a hand reaches about a ninth
BREAK = 60         # middle C: where the hands naturally divide

# the verified 16-bar cycle, for the chord symbols over bars 1-120
CYCLE = ["Gm7", "Cm7", "F7", "B-maj7", "E-maj7", "Am7b5", "D7", "Gm7",
         "Cm7", "F7", "B-maj7", "G7", "Cm7", "F7", "B-maj7", "D7"]
LAST_CHARTED_BAR = 120


# --------------------------------------------------------------------------
# one note per pitch class, chosen to keep the chord's shape
# --------------------------------------------------------------------------
def dedupe(notes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    notes = sorted(notes)
    by_pc: dict[int, list[tuple[int, int]]] = {}
    for m, d in notes:
        by_pc.setdefault(m % 12, []).append((m, d))

    top, bottom = notes[-1], notes[0]
    keep, used = [top], {top[0] % 12}
    if bottom[0] % 12 not in used:
        keep.append(bottom)
        used.add(bottom[0] % 12)

    def widest_gap(ns):
        ms = sorted(n[0] for n in ns)
        return max((b - a for a, b in zip(ms, ms[1:])), default=0)

    # the pitch classes with the most octaves to choose from go first
    for pc in sorted(by_pc, key=lambda pc: -len(by_pc[pc])):
        if pc in used:
            continue
        keep.append(min(by_pc[pc], key=lambda c: widest_gap(keep + [c])))
        used.add(pc)
    return sorted(keep)


def trim(keep: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Thin from the bottom up: the bass guitar has the low end and the top
    note is the line."""
    return keep[len(keep) - MAX_NOTES:] if len(keep) > MAX_NOTES else keep


def split_hands(ps: list[int]) -> tuple[list[int], list[int]]:
    """Divide at a real gap near middle C, as evenly as the chord allows."""
    n = len(ps)
    if n == 1:
        return ([], ps) if ps[0] >= BREAK - 5 else (ps, [])
    best = None
    for k in range(n + 1):                        # k notes to the left hand
        lh, rh = ps[:k], ps[k:]
        if lh and lh[-1] - lh[0] > MAX_SPAN:
            continue
        if rh and rh[-1] - rh[0] > MAX_SPAN:
            continue
        if lh and lh[-1] > BREAK + 7:             # nothing clearly treble down there
            continue
        if rh and rh[0] < BREAK - 14:             # nothing clearly bass up there
            continue
        gap = (rh[0] - lh[-1]) if (lh and rh) else 0
        cost = (1.5 * abs(len(lh) - len(rh))      # aim for an even split
                - 0.7 * min(gap, 12)              # but break at a real gap
                + 0.15 * abs((lh[-1] if lh else BREAK) - BREAK))
        if best is None or cost < best[0]:
            best = (cost, lh, rh)
    if best is None:            # no legal split: drop the note that is in the way
        _, i = max((ps[i + 1] - ps[i], i) for i in range(n - 1))
        return split_hands(ps[i + 1:] if i < n // 2 else ps[:i + 1])
    return best[1], best[2]


def voiced() -> list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]]:
    out = []
    for b, slot, notes in EVENTS:
        keep = trim(dedupe(list(notes)))
        lh, rh = split_hands([m for m, _ in keep])
        out.append((b, slot, max(d for _, d in keep), tuple(lh), tuple(rh)))
    return out


VOICED = voiced()


# --------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------
def hand_events(hand: str) -> list[tuple[float, float, list[int]]]:
    out = []
    for b, slot, dur, lh, rh in VOICED:
        pitches = list(rh if hand == "rh" else lh)
        if not pitches:
            continue
        start = 4 * b + written_offset(slot)
        end = 4 * b + written_offset(slot + dur)
        out.append((start, max(end, start + 0.5), pitches))
    out.sort()
    for i in range(len(out) - 1):
        s, e, ns = out[i]
        out[i] = (s, min(e, out[i + 1][0]), ns)
    return [(s, e, ns) for s, e, ns in out if e > s]


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
        subtitle="piano - transcription, made playable",
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

    for b in range(min(LAST_CHARTED_BAR, TOTAL_BARS)):
        meas = rh.measure(b + 1)
        if meas is None:
            continue
        sym = CYCLE[(b - 8) % 16]
        if b == 0 or sym != CYCLE[(b - 9) % 16]:
            cs = harmony.ChordSymbol(sym)
            cs.writeAsChord = False
            meas.insert(0.0, cs)
    tag = rh.measure(LAST_CHARTED_BAR + 1)
    if tag is not None:
        te = expressions.TextExpression("ending - as played")
        te.placement = "above"
        te.style.fontStyle = "italic"
        tag.insert(0.0, te)
    for b, mark in SECTIONS.items():
        meas = rh.measure(b)
        if meas is not None and mark != "Intro":
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

    ap = argparse.ArgumentParser(description="write the playable piano part")
    ap.add_argument("-o", "--out", default="Fly_Me_To_The_Moon_Piano_Comp.musicxml")
    ap.add_argument("--mxl", help="also write a compressed .mxl here")
    args = ap.parse_args()

    sc = build_score()
    sc.write("musicxml", fp=args.out)
    tidy_musicxml(args.out)
    n = sum(len(l) + len(r) for _, _, _, l, r in VOICED)
    was = sum(len(ns) for _, _, ns in EVENTS)
    print(f"wrote {args.out}  ({len(VOICED)} strikes, {n} notes, was {was})")
    if args.mxl:
        pack_mxl(args.out, args.mxl)
        print(f"wrote {args.mxl}")


if __name__ == "__main__":
    main()
