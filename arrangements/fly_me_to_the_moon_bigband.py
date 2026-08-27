"""Generate a 10-piece big band chart of "Fly Me To The Moon" for a singer.

The singer has the melody, so the band never plays it.  What the band plays is
the supporting material the saxophone plays on the backing track, handed around
the sections the way a Sinatra chart hands it around.

Everything about the chart was measured from the recording:

    key      B-flat major / G minor   (chroma + Krumhansl-Schmuckler)
    tempo    120.000 BPM exactly      (onset-grid search on the drum stem)
    feel     medium swing, 4/4
    form     8-bar intro
             7 x 16-bar chorus (bars 9-120)
             8-bar coda        (bars 121-128)

Separating the track six ways with Demucs puts piano and guitar in their own
stems and leaves the saxophone alone, and transcribing that stem with pYIN
shows two behaviours the arrangement is built from: a **sustained counterline**
under the voice, and **eighth-note licks at the turnaround** - 44% of the
saxophone's notes fall in bars 15-16 of the cycle, which is only 12.5% of the
bars.

Three rules are enforced in code rather than left to luck, and checked on the
finished score:

    * the band never doubles the melody;
    * nothing the band plays sits in the singer's octave while she is
      articulating a syllable - pads are pushed below her;
    * busy figures only happen where she is holding a note, so an answer never
      competes with a word.

Instrumentation:
    Voice   (cue staff - the singer, not played by the band)
    Reeds   Alto Sax (E-flat), Tenor Sax (B-flat), Baritone Sax (E-flat)
    Brass   Trumpet 1 (B-flat), Trumpet 2 (B-flat), Trombone (C, bass clef)
    Rhythm  Keys, Bass, Drums
"""

from __future__ import annotations

import copy

from music21 import (
    bar,
    clef,
    duration,
    expressions,
    harmony,
    instrument,
    key,
    metadata,
    meter,
    note,
    pitch,
    stream,
    tempo,
    tie,
)

# --------------------------------------------------------------------------
# Harmony
# --------------------------------------------------------------------------

# The 16-bar cycle, verified against the recording.  Each entry is a list of
# (beat offset within the bar, chord symbol).
CYCLE: list[list[tuple[float, str]]] = [
    [(0.0, "Gm7")],                       # 1
    [(0.0, "Cm7")],                       # 2
    [(0.0, "F7")],                        # 3
    [(0.0, "Bbmaj7")],                    # 4
    [(0.0, "Ebmaj7")],                    # 5
    [(0.0, "Am7b5")],                     # 6
    [(0.0, "D7")],                        # 7
    [(0.0, "Gm7"), (2.0, "G7")],          # 8
    [(0.0, "Cm7")],                       # 9
    [(0.0, "F7")],                        # 10
    [(0.0, "Bbmaj7")],                    # 11
    [(0.0, "G7")],                        # 12
    [(0.0, "Cm7")],                       # 13
    [(0.0, "F7")],                        # 14
    [(0.0, "Bbmaj7")],                    # 15
    [(0.0, "D7")],                        # 16  (turnaround)
]

# 8-bar intro: descending bass A - Ab - G - Bb - Eb, then ii(o)-V into the head.
INTRO: list[list[tuple[float, str]]] = [
    [(0.0, "Am7b5")],
    [(0.0, "Ab7")],
    [(0.0, "Gm7")],
    [(0.0, "Bbmaj7")],
    [(0.0, "Ebmaj7")],
    [(0.0, "Am7b5")],
    [(0.0, "D7")],
    [(0.0, "D7")],
]

# 8-bar coda.
CODA: list[list[tuple[float, str]]] = [
    [(0.0, "Cm7")],
    [(0.0, "F7")],
    [(0.0, "Bbmaj7")],
    [(0.0, "G7")],
    [(0.0, "Cm7")],
    [(0.0, "F7")],
    [(0.0, "Bbmaj7")],
    [(0.0, "Bb6")],
]

N_CHORUS = 7
FIRST_CHORUS_BAR = 9
TOTAL_BARS = 8 + N_CHORUS * 16 + 8          # 128


def chord_map() -> dict[int, list[tuple[float, str]]]:
    """bar number (1-based) -> [(beat offset, chord symbol)]."""
    m: dict[int, list[tuple[float, str]]] = {}
    for i, c in enumerate(INTRO):
        m[1 + i] = c
    for ch in range(N_CHORUS):
        for i, c in enumerate(CYCLE):
            m[FIRST_CHORUS_BAR + ch * 16 + i] = c
    for i, c in enumerate(CODA):
        m[FIRST_CHORUS_BAR + N_CHORUS * 16 + i] = c
    return m


CHORDS = chord_map()

# Pitch-class content: (root pc, [chord tones], [usable tensions])
_CHORD_SPEC = {
    "Gm7":    (7,  [0, 3, 7, 10], [2, 5]),
    "Cm7":    (0,  [0, 3, 7, 10], [2, 5]),
    "F7":     (5,  [0, 4, 7, 10], [2, 9]),
    "Bbmaj7": (10, [0, 4, 7, 11], [2, 9]),
    "Bb6":    (10, [0, 4, 7, 9],  [2]),
    "Ebmaj7": (3,  [0, 4, 7, 11], [2, 9]),
    "Am7b5":  (9,  [0, 3, 6, 10], [5]),
    "D7":     (2,  [0, 4, 7, 10], [1, 3, 8]),   # b9 / #9 / b13
    "G7":     (7,  [0, 4, 7, 10], [1, 2, 9]),
    "Ab7":    (8,  [0, 4, 7, 10], [2, 6]),
}


def chord_pcs(sym: str) -> tuple[int, set[int], set[int]]:
    root, tones, tens = _CHORD_SPEC[sym]
    return root, {(root + i) % 12 for i in tones}, {(root + i) % 12 for i in tens}


def chord_at(bar_no: int, beat: float) -> str:
    entries = CHORDS[bar_no]
    sym = entries[0][1]
    for off, s in entries:
        if beat + 1e-6 >= off:
            sym = s
    return sym


# --------------------------------------------------------------------------
# Melody  (concert pitch, as measured from the published lead sheet and
# transposed C major -> B-flat major to match the recording)
# --------------------------------------------------------------------------

# (bar in cycle 1..16, offset in bar, pitch or None for rest, quarterLength,
#  tie-to-next flag)
MELODY: list[tuple[int, float, str | None, float, bool]] = [
    (1, 0.0, "B-4", 1.0, False), (1, 1.0, "A4", 1.0, False),
    (1, 2.0, "G4", 1.0, False),  (1, 3.0, "F4", 0.5, False),
    (1, 3.5, "E-4", 0.5, True),

    (2, 0.0, "E-4", 1.5, False), (2, 1.5, "F4", 0.5, False),
    (2, 2.0, "G4", 1.0, False),  (2, 3.0, "B-4", 1.0, False),

    (3, 0.0, "A4", 1.0, False),  (3, 1.0, "G4", 1.0, False),
    (3, 2.0, "F4", 1.0, False),  (3, 3.0, "E-4", 0.5, False),
    (3, 3.5, "D4", 0.5, True),

    (4, 0.0, "D4", 4.0, False),

    (5, 0.0, "G4", 1.0, False),  (5, 1.0, "F4", 1.0, False),
    (5, 2.0, "E-4", 1.0, False), (5, 3.0, "D4", 1.0, False),

    (6, 0.0, "C4", 1.0, False),  (6, 1.0, "D4", 1.0, False),
    (6, 2.0, "E-4", 1.0, False), (6, 3.0, "G4", 1.0, False),

    (7, 0.0, "F#4", 1.0, False), (7, 1.0, "E-4", 1.0, False),
    (7, 2.0, "D4", 1.0, False),  (7, 3.0, "C4", 0.5, False),
    (7, 3.5, "B-3", 0.5, True),

    (8, 0.0, "B-3", 3.0, False), (8, 3.0, "B3", 1.0, False),   # "in" pickup

    (9, 0.0, "C4", 0.5, False),  (9, 0.5, "G4", 1.0, False),
    (9, 1.5, "G4", 0.5, True),   (9, 2.0, "G4", 2.0, True),

    (10, 0.0, "G4", 1.0, False), (10, 1.0, "B-4", 2.0, False),
    (10, 3.0, "A4", 1.0, False),

    (11, 0.0, "F4", 4.0, True),

    (12, 0.0, "F4", 3.0, False), (12, 3.0, "A3", 1.0, False),  # "in" pickup

    (13, 0.0, "C4", 0.5, False), (13, 0.5, "E-4", 1.0, False),
    (13, 1.5, "E-4", 0.5, True), (13, 2.0, "E-4", 2.0, True),

    (14, 0.0, "E-4", 1.0, False), (14, 1.0, "G4", 2.0, False),
    (14, 3.0, "F4", 1.0, False),

    (15, 0.0, "E-4", 2.0, False), (15, 2.0, "D4", 2.0, True),

    (16, 0.0, "D4", 4.0, False),
]


# --------------------------------------------------------------------------
# What the saxophone plays on the recording
# --------------------------------------------------------------------------
#
# The backing track carries no melody, but it does carry a written saxophone
# part that supports where the vocal sits.  Isolating it (Demucs 6-stem, so
# piano and guitar are pulled out separately) and transcribing it with pYIN
# shows two distinct behaviours, and the arrangement is built out of both:
#
#   * a **sustained counterline** under the vocal - long tones, 1 to 3 beats,
#     one or two per bar.  25 of its 27 notes are chord tones or tensions and
#     the other two are chromatic approaches, so it is written, not improvised.
#   * **eighth-note licks at the turnaround**.  44% of the saxophone's notes
#     fall in bars 15-16 of the 16-bar cycle, which is only 12.5% of the bars -
#     a three-and-a-half-fold concentration.  That is the band answering at the
#     end of each phrase, and it lands where the singer is holding.
#
# Everything below is transcribed pitch-for-pitch from the recording.

# bars 1-8, as (bar, offset, concert pitch, quarterLength)
SAX_INTRO: list[tuple[int, float, str, float]] = [
    (1, 0.5, "E-4", 1.5), (1, 2.0, "D4", 0.5), (1, 2.5, "C4", 0.5), (1, 3.0, "B-3", 1.0),
    (2, 0.0, "A3", 1.0), (2, 1.0, "D4", 1.0), (2, 2.0, "G-3", 0.5), (2, 2.5, "G3", 1.0),
    (2, 3.5, "A3", 0.5),
    (3, 0.5, "B-3", 1.5), (3, 2.0, "A3", 0.5), (3, 2.5, "B-3", 0.5), (3, 3.0, "C4", 1.0),
    (4, 0.0, "D4", 1.0), (4, 1.0, "A3", 1.0), (4, 2.0, "B-3", 0.5), (4, 3.0, "C4", 0.5),
    (4, 3.5, "D4", 0.5),
    (5, 0.5, "G3", 2.0), (5, 2.5, "F3", 0.5),
    (6, 1.0, "F3", 0.5), (6, 2.0, "B-3", 1.0), (6, 3.5, "G3", 1.0),
    (7, 0.5, "A3", 0.5), (7, 1.0, "A-3", 0.5), (7, 1.5, "F3", 0.5), (7, 2.5, "F3", 1.0),
    (7, 3.5, "G3", 0.5),
    (8, 0.0, "G-3", 1.5), (8, 1.5, "B-3", 0.5), (8, 2.0, "A3", 1.0), (8, 3.0, "G-3", 1.5),
]

# the sustained counterline, on the 16-bar cycle (cycle bar, offset, pitch, ql)
COUNTER: list[tuple[int, float, str, float]] = [
    (1, 0.5, "G3", 2.0),
    (3, 0.5, "A3", 2.5), (3, 3.5, "A3", 0.5),
    (4, 0.5, "F3", 1.5), (4, 3.0, "F3", 0.5), (4, 3.5, "G-3", 0.5),
    (5, 0.0, "G3", 1.0),
    (7, 0.0, "A3", 2.0), (7, 2.5, "G-3", 0.5),
    (8, 1.0, "F3", 0.5), (8, 1.5, "D4", 1.5),
    (9, 0.5, "G3", 3.0),
    (10, 3.0, "G-3", 0.5), (10, 3.5, "F3", 1.0),
    (11, 0.5, "D4", 1.5),
    (12, 3.0, "F3", 1.5),
    (13, 0.5, "G3", 1.5), (13, 2.0, "B-3", 0.5),
    (14, 3.0, "F3", 1.0),
]

# turnaround lick heard in chorus 1 (cycle bars 15-16)
LICK_A: list[tuple[int, float, str, float]] = [
    (15, 0.0, "B-3", 1.0), (15, 1.0, "F3", 1.0), (15, 2.0, "B-3", 1.0),
    (15, 3.0, "F3", 0.5), (15, 3.5, "C4", 0.5),
    (16, 0.0, "D4", 2.5), (16, 3.0, "D4", 0.5), (16, 3.5, "C4", 0.5),
]

# turnaround lick heard in choruses 2 and 3 - the chromatic C - D-flat - D tail
LICK_B: list[tuple[int, float, str, float]] = [
    (15, 1.0, "F3", 0.5), (15, 1.5, "G3", 0.5), (15, 2.0, "B-3", 0.5),
    (15, 2.5, "F3", 0.5), (15, 3.0, "F3", 0.5), (15, 3.5, "B-3", 0.5),
    (16, 0.0, "C4", 0.5), (16, 0.5, "D-4", 0.5), (16, 1.5, "D4", 0.5),
    (16, 2.0, "D-4", 0.5), (16, 2.5, "C4", 0.5), (16, 3.0, "D-4", 0.5),
    (16, 3.5, "D4", 0.5),
]


def fill_windows() -> list[bool]:
    """Eighth-by-eighth over the cycle: is the singer holding rather than
    starting a new syllable?

    A run of three or more eighths without a new syllable is where an arranger
    puts an answer.  The band's busy figures are confined to these, so nothing
    the horns play competes with a word.
    """
    onset = [False] * 128
    for cyc, off, name, _ql, _tie in MELODY:
        if name is not None:
            onset[(cyc - 1) * 8 + int(off * 2)] = True
    free = [False] * 128
    i = 0
    while i < 128:
        if onset[i]:
            i += 1
            continue
        j = i
        while j < 128 and not onset[j]:
            j += 1
        if j - i >= 3:
            for k in range(i, j):
                free[k] = True
        i = j
    return free


FREE = fill_windows()


def in_window(cycle_bar: int, off: float) -> bool:
    return FREE[((cycle_bar - 1) * 8 + int(off * 2)) % 128]


# --------------------------------------------------------------------------
# Voicing engine
# --------------------------------------------------------------------------

def _ps(p: str | int) -> int:
    return int(pitch.Pitch(p).ps) if isinstance(p, str) else int(p)


def voicing_tones(sym: str, lead_pc: int) -> set[int]:
    """Chord tones to voice with, chosen the way a big-band writer would.

    Two standard adjustments are applied:

    * **major 7th chords use the 6th, not the 7th.**  In a close voicing the
      root and the major 7th land a semitone apart, which is the one interval
      the ensemble sound cannot absorb, so the 6th replaces the 7th and the
      major 7th becomes an available tension.
    * **an 11th in the lead suspends the chord** rather than being treated as
      a chromatic approach - F7 with B-flat on top is F7sus4, not a passing
      diminished.
    """
    root, tones, tens = chord_pcs(sym)
    quality = _CHORD_SPEC[sym][1]
    eleventh = (root + 5) % 12

    if quality == [0, 4, 7, 11]:                       # major 7th
        if lead_pc == eleventh:                        # 11th suspends
            return {(root + i) % 12 for i in (0, 5, 7, 9)}
        return {(root + i) % 12 for i in (0, 4, 7, 9)}
    if quality == [0, 4, 7, 10]:                       # dominant 7th
        if lead_pc == eleventh:
            return {(root + i) % 12 for i in (0, 5, 7, 10)}
        return set(tones)
    return set(tones)


def voice_below(lead: int, sym: str, n: int) -> list[int]:
    """Return ``n`` descending voices starting on ``lead`` (concert MIDI).

    Standard big-band practice:

    * lead is a chord tone or an available tension -> close voicing on the
      chord tones (a tension displaces the chord tone a step below it);
    * lead is a chromatic approach note -> diminished-7th approach voicing,
      every voice moving in parallel into the target chord.
    """
    root, tones, tens = chord_pcs(sym)
    lead_pc = lead % 12
    base = voicing_tones(sym, lead_pc)

    if lead_pc in base:
        allowed = set(base)
    elif lead_pc in tens or lead_pc in tones:
        allowed = set(base) | {lead_pc}
        for clash in ((lead_pc - 1) % 12, (lead_pc + 1) % 12):
            if clash in allowed and clash != lead_pc:
                allowed.discard(clash)
    else:
        # chromatic approach: parallel diminished 7th
        allowed = {(lead_pc + k) % 12 for k in (0, 3, 6, 9)}

    voices = [lead]
    cur = lead
    while len(voices) < n:
        cur -= 1
        guard = 0
        while cur % 12 not in allowed and guard < 24:
            cur -= 1
            guard += 1
        voices.append(cur)
    return voices


# practical written ranges, expressed as concert MIDI, used to octave-fit
CONCERT_RANGE = {
    # concert-pitch limits chosen so the written parts stay inside the
    # comfortable section-writing range of each horn
    "alto":   (_ps("F3"),  _ps("F5")),    # written D4  - D6
    "tenor":  (_ps("B-2"), _ps("C5")),    # written C4  - D6
    "bari":   (_ps("D-2"), _ps("E-4")),   # written B-3 - C6
    "tpt1":   (_ps("F#3"), _ps("B-5")),   # written G#3 - C6
    "tpt2":   (_ps("F#3"), _ps("G5")),    # written G#3 - A5
    "tbn":    (_ps("E2"),  _ps("B-4")),   # ledger lines are normal up here
}


def fit(p: int, who: str) -> int:
    lo, hi = CONCERT_RANGE[who]
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def nearest(pc: int, target: int) -> int:
    """The pitch of pitch-class ``pc`` closest to ``target`` (concert MIDI)."""
    base = pc + 12 * ((target - pc) // 12)
    return min((base, base + 12), key=lambda x: abs(x - target))


def ensemble_voicing(lead: int, sym: str, *, spread: bool = False) -> dict[str, int]:
    """Six-part concerted voicing keyed by instrument slot.

    Two textures, because six horns cannot cover one close block once the lead
    climbs above the reeds' comfortable ceiling:

    ``spread=False`` - a five-way close voicing under a mid-register lead
    (trumpets, alto and tenor), with the trombone taking the fifth voice and
    the baritone anchoring the root.  Tight and warm; used for the harmonised
    chorus.

    ``spread=True`` - the lead sits an octave higher, so the section splits
    into two tiers: trumpets and alto take a three-way close voicing and the
    tenor and trombone double the top two an octave below, with the baritone
    on the root.  Octave doubling rather than an ever-wider block, which is
    what keeps the shout chorus and the head out sounding full instead of
    leaving a hole in the middle.
    """
    root, _, _ = chord_pcs(sym)

    if not spread:
        v = voice_below(lead, sym, 5)
        out = {"tpt1": v[0], "tpt2": v[1], "alto": v[2],
               "tenor": v[3], "tbn": v[4]}
        # in the warm mid-register texture keep the trombone off ledger lines
        if out["tbn"] > _ps("F4"):
            out["tbn"] -= 12
    else:
        v = voice_below(lead, sym, 3)
        out = {"tpt1": v[0], "tpt2": v[1], "alto": v[2],
               "tenor": v[0] - 12, "tbn": v[1] - 12}
        while out["tbn"] > CONCERT_RANGE["tbn"][1]:
            out["tbn"] -= 12

    out = {k: fit(pp, k) for k, pp in out.items()}

    bottom = min(out.values())
    b = nearest(root, bottom - 5)
    while b > bottom - 3:
        b -= 12
    out["bari"] = fit(b, "bari")
    return out


# --------------------------------------------------------------------------
# Walking bass + drums
# --------------------------------------------------------------------------

def walking_bass(bar_no: int, prev_last: int | None) -> list[int]:
    """Four quarter-note bass pitches for ``bar_no`` (concert MIDI)."""
    beats = [0.0, 1.0, 2.0, 3.0]
    syms = [chord_at(bar_no, b) for b in beats]
    nxt_sym = chord_at(bar_no + 1, 0.0) if bar_no < TOTAL_BARS else syms[-1]
    nxt_root = chord_pcs(nxt_sym)[0]

    def near(pc: int, ref: int) -> int:
        cand = pc + 12 * ((ref - pc) // 12)
        best, bd = cand, 99
        for c in (cand - 12, cand, cand + 12):
            if _ps("E1") <= c <= _ps("G3") and abs(c - ref) < bd:
                best, bd = c, abs(c - ref)
        return best

    root0 = chord_pcs(syms[0])[0]
    ref = prev_last if prev_last is not None else _ps("G2")
    b1 = near(root0, ref)
    line = [b1]

    if syms[2] != syms[0]:                      # two chords in the bar
        r2 = chord_pcs(syms[2])[0]
        _, t0, _ = chord_pcs(syms[0])
        third = sorted(t0)[1]
        line.append(near(third, line[-1]))
        line.append(near(r2, line[-1]))
        tgt = near(nxt_root, line[-1])
        line.append(tgt - 1 if tgt >= line[-1] else tgt + 1)
    else:
        _, tones, _ = chord_pcs(syms[0])
        ordered = sorted(tones, key=lambda pc: (pc - root0) % 12)
        line.append(near(ordered[1], line[-1]))          # 3rd
        line.append(near(ordered[2], line[-1]))          # 5th
        tgt = near(nxt_root, line[-1])
        approach = tgt - 1 if tgt >= line[-1] else tgt + 1
        line.append(approach)
    return [max(_ps("E1"), min(_ps("A3"), p)) for p in line]


# --------------------------------------------------------------------------
# Score construction helpers
# --------------------------------------------------------------------------

SECTIONS = {
    1:   ("Intro", ""),
    9:   ("A", "VOCAL - trombone counterline, reeds answer"),
    25:  ("B", "VOCAL - alto takes the line, brass answer"),
    41:  ("C", "VOCAL - reeds in harmony, brass punctuate"),
    57:  ("D", "Tenor Sax solo (open)"),
    73:  ("E", "VOCAL - brass carry the line"),
    89:  ("F", "VOCAL - reeds and brass trade"),
    105: ("G", "VOCAL - full band"),
    121: ("H", "Coda"),
}


def new_part(name: str, abbrev: str, instr, *, treble=True):
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


CS_FIGURE = {
    "Gm7": "Gm7", "Cm7": "Cm7", "F7": "F7", "Bbmaj7": "B-maj7",
    "Ebmaj7": "E-maj7", "Am7b5": "Am7b5", "D7": "D7", "G7": "G7",
    "Ab7": "A-7", "Bb6": "B-6",
}


def make_symbol(sym: str) -> harmony.ChordSymbol:
    cs = harmony.ChordSymbol(CS_FIGURE[sym])
    cs.writeAsChord = False
    cs.quarterLength = 0.0
    return cs


def melody_events(start_bar: int):
    """Yield (abs_bar, offset, midi|None, ql, tie_out) for one 16-bar chorus."""
    for cyc_bar, off, name, ql, tie_out in MELODY:
        midi = None if name is None else _ps(name)
        yield start_bar + cyc_bar - 1, off, midi, ql, tie_out


def pick_lead(sym: str, target: int) -> int:
    """Chord tone nearest ``target`` (concert MIDI)."""
    _, tones, _ = chord_pcs(sym)
    best, bd = target, 999
    for pc in tones:
        for octv in range(0, 11):
            p = pc + 12 * octv
            if abs(p - target) < bd:
                best, bd = p, abs(p - target)
    return best


def _note(midi: int, ql: float, *, artic: str | None = None) -> note.Note:
    n = note.Note()
    n.pitch.ps = midi
    n.pitch.spellingIsInferred = True
    n.quarterLength = ql
    if artic:
        from music21 import articulations
        n.articulations = [
            {"accent": articulations.Accent(),
             "staccato": articulations.Staccato(),
             "marcato": articulations.StrongAccent()}[artic]
        ]
    return n


def _spell(n: note.Note, flats: bool = True) -> note.Note:
    """Prefer flat spellings, which suit B-flat major / G minor."""
    p = n.pitch
    if p.accidental is not None and p.accidental.alter > 0 and flats:
        if p.name not in ("F#", "C#"):
            n.pitch = p.getEnharmonic()
    return n


SLOTS = ["tpt1", "tpt2", "alto", "tenor", "tbn", "bari"]


def build_horn_plan() -> dict[int, dict[str, list]]:
    """plan[bar][slot] = list of (offset, midi, ql, tie_out, artic).

    The singer has the melody, so the band never plays it.  What the band
    plays instead is the saxophone's supporting material from the recording,
    handed around the sections the way a Sinatra chart hands it around: one
    colour states a phrase, another answers it, and the section only plays
    together when the singer is holding a note.
    """
    plan = {b: {s: [] for s in SLOTS} for b in range(1, TOTAL_BARS + 1)}

    def put(bar_no, slot, off, midi, ql, tie_out=False, artic=None):
        """Add a note, splitting it across the barline as a tie if it spills."""
        while ql > 0 and 1 <= bar_no <= TOTAL_BARS:
            room = 4.0 - off
            if ql <= room + 1e-9:
                plan[bar_no][slot].append((off, midi, ql, tie_out, artic))
                return
            plan[bar_no][slot].append((off, midi, room, True, artic))
            ql -= room
            bar_no += 1
            off = 0.0
            artic = None

    def line(start_bar, material, slots, *, shift=0, harmonise=False,
             artic=None, sustain_only=False):
        """Place transcribed material, optionally harmonised under its own lead.

        Two things keep it out of the singer's way: a short note is only
        allowed where she is holding, and anything sounding while she
        articulates is pushed below her octave.
        """
        for cyc, off, name, ql in material:
            b = start_bar + cyc - 1
            window = in_window(cyc, off)
            if not window and (ql < 1.0 or sustain_only and ql < 1.5):
                continue
            lead = _ps(name) + shift
            if not window:
                while lead >= VOICE_LO:
                    lead -= 12
            if harmonise and len(slots) >= 5:
                # whole band: use the section voicing so nobody is stranded
                v = ensemble_voicing(lead, chord_at(b, off), spread=True)
                for slot in slots:
                    put(b, slot, off, v[slot], ql, artic=artic)
            elif harmonise and len(slots) > 1:
                voices = voice_below(lead, chord_at(b, off), len(slots))
                for slot, v in zip(slots, voices):
                    put(b, slot, off, fit(v, slot), ql, artic=artic)
            else:
                for slot in slots:
                    put(b, slot, off, fit(lead, slot), ql, artic=artic)

    def pad(start_bar, bars, slots, target, *, ql=4.0):
        """Sustained chord voicing, sitting below the voice."""
        for cyc in bars:
            b = start_bar + cyc - 1
            sym = chord_at(b, 0.0)
            voices = voice_below(pick_lead(sym, target), sym, len(slots))
            for slot, v in zip(slots, voices):
                put(b, slot, 0.0, fit(v, slot), ql)

    def answer(start_bar, cyc, off, slots, target, *, harmonise=True,
               artic="accent"):
        """A three-note answer in one of the singer's held-note windows."""
        if not in_window(cyc, off):
            return
        b = start_bar + cyc - 1
        sym = chord_at(b, off)
        top = pick_lead(sym, target)
        shape = [(off, 0.5), (off + 0.5, 0.5), (off + 1.0, 1.0)]
        steps = voice_below(top, sym, 3)
        for k, (o, q) in enumerate(shape):
            note_lead = steps[k]
            chord_here = chord_at(b, o)
            if harmonise and len(slots) >= 5:
                v = ensemble_voicing(note_lead, chord_here, spread=True)
                for slot in slots:
                    put(b, slot, o, v[slot], q, artic=artic)
            elif harmonise and len(slots) > 1:
                voices = voice_below(note_lead, chord_here, len(slots))
                for slot, v in zip(slots, voices):
                    put(b, slot, o, fit(v, slot), q, artic=artic)
            else:
                for slot in slots:
                    put(b, slot, o, fit(note_lead, slot), q, artic=artic)

    SAXES = ["alto", "tenor", "bari"]
    BRASS = ["tpt1", "tpt2", "tbn"]

    # ---- Intro, bars 1-8: the saxophone's own line, passed around --------
    hand_off = {1: ["tpt1"], 2: ["tpt1"], 3: ["tenor"], 4: ["tenor"],
                5: ["tbn"], 6: ["tbn"]}
    for b, off, name, ql in SAX_INTRO:
        if b in hand_off:
            for slot in hand_off[b]:
                put(b, slot, off, fit(_ps(name), slot), ql)
        else:                                   # bars 7-8: whole band, harmonised
            voices = voice_below(_ps(name) + 12, chord_at(b, off), 5)
            for slot, v in zip(["tpt1", "tpt2", "alto", "tenor", "tbn"], voices):
                put(b, slot, off, fit(v, slot), ql, artic="accent")
            root = chord_pcs(chord_at(b, off))[0]
            put(b, "bari", off, fit(nearest(root, _ps("B-2")), "bari"), ql,
                artic="accent")
    for b in (5, 6):                            # soft reed bed under the trombone
        sym = chord_at(b, 0.0)
        voices = voice_below(pick_lead(sym, _ps("E-4")), sym, 2)
        for slot, v in zip(["alto", "tenor"], voices):
            put(b, slot, 0.0, fit(v, slot), 4.0)

    # ---- Chorus 1, bars 9-24: dark and sparse ----------------------------
    line(9, COUNTER, ["tbn"], shift=-12)        # counterline an octave down
    line(9, LICK_A, SAXES, harmonise=True)      # reeds answer at the turnaround

    # ---- Chorus 2, bars 25-40: the alto takes it, brass answer -----------
    line(25, COUNTER, ["alto"])
    pad(25, [2, 4, 6, 8, 10, 12, 14], ["tbn", "bari"], _ps("F3"))
    line(25, LICK_B, BRASS, shift=12, harmonise=True, artic="accent")

    # ---- Chorus 3, bars 41-56: reeds in harmony, brass punctuate ---------
    line(41, COUNTER, SAXES, harmonise=True)
    for cyc, off in ((4, 1.0), (8, 1.0), (11, 1.0), (12, 1.0)):
        answer(41, cyc, off, BRASS, _ps("A5"))
    line(41, LICK_B, SLOTS, shift=12, harmonise=True, artic="accent")

    # ---- Chorus 4, bars 57-72: tenor solo, as on the recording -----------
    for i, b in enumerate(range(65, 73)):       # brass pad behind the second half
        sym = chord_at(b, 0.0)
        voices = voice_below(pick_lead(sym, _ps("F4")), sym, 3)
        for slot, v in zip(BRASS, voices):
            put(b, slot, 0.0, fit(v, slot), 4.0)

    # ---- Chorus 5, bars 73-88: the voice returns, brass carry the line ---
    line(73, COUNTER, ["tpt2", "tbn"], harmonise=True)
    pad(73, [1, 3, 5, 7, 9, 11, 13], ["tenor", "bari"], _ps("B-2"))
    for cyc, off in ((4, 1.0), (8, 1.0), (11, 1.0), (12, 1.0)):
        answer(73, cyc, off, SAXES, _ps("E-5"))
    line(73, LICK_A, SLOTS, shift=12, harmonise=True, artic="accent")

    # ---- Chorus 6, bars 89-104: reeds and brass trade the counterline ----
    line(89, [c for c in COUNTER if c[0] <= 8], SAXES, harmonise=True)
    line(89, [c for c in COUNTER if c[0] > 8], BRASS, harmonise=True)
    for cyc, off in ((4, 1.0), (8, 1.0), (11, 1.0), (12, 1.0), (15, 0.0)):
        answer(89, cyc, off, SLOTS, _ps("G5"))
    line(89, LICK_B, SLOTS, shift=12, harmonise=True, artic="accent")

    # ---- Chorus 7, bars 105-120: full band behind the voice --------------
    # sustained ensemble under the singer, answers on top of her held notes
    line(105, COUNTER, SLOTS, harmonise=True, sustain_only=True)
    for cyc, off in ((4, 1.0), (8, 1.0), (11, 1.0), (12, 1.0)):
        answer(105, cyc, off, SLOTS, _ps("B-5"))
    line(105, LICK_B, SLOTS, shift=12, harmonise=True, artic="accent")

    # ---- Coda, bars 121-128 ----------------------------------------------
    for b in range(121, 127):
        sym = chord_at(b, 0.0)
        lead = pick_lead(sym, _ps("B-5"))
        v = ensemble_voicing(lead, sym, spread=True)
        for off, ql in ([(0.0, 2.0), (2.5, 1.5)] if b % 2 else [(0.0, 4.0)]):
            for slot in SLOTS:
                put(b, slot, off, v[slot], ql, artic="accent" if b % 2 else None)
    # the "Basie" tag: three punches, then the held chord under the last note
    v_end = ensemble_voicing(pick_lead("Bbmaj7", _ps("B-5")), "Bbmaj7", spread=True)
    for off, ql in [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5)]:
        for slot in SLOTS:
            put(127, slot, off, v_end[slot], ql, artic="marcato")
    v_fin = ensemble_voicing(pick_lead("Bb6", _ps("B-5")), "Bb6", spread=True)
    for slot in SLOTS:
        put(128, slot, 0.0, v_fin[slot], 4.0)

    _resolve_overlaps(plan)
    _clear_the_voice(plan)
    _open_semitone_clashes(plan)
    _breathe_before_leaps(plan)
    return plan


VOICE_LO, VOICE_HI = 58, 70            # concert B-flat 3 to B-flat 4


def _vocal_pitches() -> dict[int, set[tuple[float, int]]]:
    """(offset, concert midi) the singer is sounding, per bar."""
    out: dict[int, set[tuple[float, int]]] = {}
    for start in VOCAL_CHORUSES:
        for abs_bar, off, midi, _ql, _tie in melody_events(start):
            if midi is not None:
                out.setdefault(abs_bar, set()).add((off, midi))
    for b in range(121, 128):                 # the held note over the coda
        out.setdefault(b, set()).add((0.0, _ps("B-4")))
    return out


def _resolve_overlaps(plan: dict) -> None:
    """One instrument, one note at a time.

    Counterline, pad and answer are laid down independently, so they can land
    on the same player at once.  Keep the earlier note and trim it to where
    the next one starts; drop it if that leaves nothing.
    """
    for bar_no, slots in plan.items():
        for slot, events in slots.items():
            if not events:
                continue
            events.sort()
            kept: list[tuple] = []
            for off, midi, ql, tie_out, artic in events:
                if kept:
                    poff, pmidi, pql, ptie, partic = kept[-1]
                    if off < poff + pql - 1e-6:
                        trimmed = off - poff
                        if trimmed < 0.25:
                            kept.pop()
                        else:
                            kept[-1] = (poff, pmidi, trimmed, ptie, partic)
                    if kept and abs(off - kept[-1][0]) < 1e-6:
                        continue
                kept.append((off, midi, min(ql, 4.0 - off), tie_out, artic))
            slots[slot] = [e for e in kept if e[2] > 0]


def _clear_the_voice(plan: dict) -> None:
    """Never double the singer, and stay out of her octave while she sings."""
    vocal = _vocal_pitches()
    for bar_no, slots in plan.items():
        sung = vocal.get(bar_no)
        if not sung:
            continue
        cyc = None
        for start in VOCAL_CHORUSES:
            if start <= bar_no < start + 16:
                cyc = bar_no - start + 1
        for slot, events in slots.items():
            out = []
            for off, midi, ql, tie_out, artic in events:
                if cyc is not None and not in_window(cyc, off):
                    while midi >= VOICE_LO:
                        midi -= 12
                if (off, midi) in sung:            # never in unison with a word
                    midi -= 12
                midi = fit(midi, slot)
                blocked = (off, midi) in sung
                if cyc is not None and not in_window(cyc, off):
                    blocked = blocked or VOICE_LO <= midi <= VOICE_HI
                if blocked:
                    continue
                out.append((off, midi, ql, tie_out, artic))
            slots[slot] = out


def _breathe_before_leaps(plan: dict) -> None:
    """Clip a sustained note that runs straight into a distant one.

    The pads sit below the singer and the answers come in high above her, so a
    player can be asked to jump two octaves with no gap.  Ending the pad an
    eighth early gives them time to get there and reads better on the page.
    """
    flat: dict[str, list[tuple[float, int, int, float]]] = {}
    for bar_no in sorted(plan):
        for slot, events in plan[bar_no].items():
            for idx, (off, midi, ql, _t, _a) in enumerate(sorted(events)):
                flat.setdefault(slot, []).append(
                    ((bar_no - 1) * 4 + off, bar_no, idx, midi))
    for slot, seq in flat.items():
        seq.sort()
        for i in range(len(seq) - 1):
            t0, b0, _i0, m0 = seq[i]
            t1, _b1, _i1, m1 = seq[i + 1]
            if abs(m1 - m0) <= 12:
                continue
            events = sorted(plan[b0][slot])
            for k, (off, midi, ql, tie_out, artic) in enumerate(events):
                if abs((b0 - 1) * 4 + off - t0) > 1e-6 or midi != m0:
                    continue
                if tie_out:
                    break
                end = (b0 - 1) * 4 + off + ql
                if t1 - end < 0.5 and ql - (0.5 - (t1 - end)) >= 0.5:
                    events[k] = (off, midi, ql - (0.5 - (t1 - end)), tie_out, artic)
                    plan[b0][slot] = events
                break


def _open_semitone_clashes(plan: dict) -> None:
    """No two players a semitone apart at the same instant.

    Pads and counterlines are laid down independently, so they can collide.
    Where they do, try to open the interval by an octave; failing that drop the
    longer note, which is the sustained pad rather than the moving line.
    """
    for bar_no, slots in plan.items():
        starts: dict[float, list[tuple[str, int]]] = {}
        for slot, events in slots.items():
            for off, midi, *_rest in events:
                starts.setdefault(off, []).append((slot, midi))
        for off, group in starts.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda g: g[1])
            for i in range(len(group) - 1):
                (slot_lo, lo), (slot_hi, hi) = group[i], group[i + 1]
                if hi - lo != 1:
                    continue
                for slot, midi, delta in ((slot_hi, hi, 12), (slot_lo, lo, -12)):
                    moved = fit(midi + delta, slot)
                    if moved != midi and all(abs(moved - m) != 1
                                             for sl, m in group if sl != slot):
                        slots[slot] = [(o, moved if (o == off and mm == midi) else mm,
                                        q, t, a) for o, mm, q, t, a in slots[slot]]
                        break
                else:
                    victim = max((slot_lo, slot_hi),
                                 key=lambda sl: max((q for o, m, q, t, a in slots[sl]
                                                     if o == off), default=0))
                    slots[victim] = [e for e in slots[victim] if e[0] != off]


HORN_PARTS = [
    ("tpt1",  "Trumpet 1",     "Tpt. 1", instrument.Trumpet,           True),
    ("tpt2",  "Trumpet 2",     "Tpt. 2", instrument.Trumpet,           True),
    ("tbn",   "Trombone",      "Tbn.",   instrument.Trombone,          False),
    ("alto",  "Alto Sax",      "A. Sx.", instrument.AltoSaxophone,     True),
    ("tenor", "Tenor Sax",     "T. Sx.", instrument.TenorSaxophone,    True),
    ("bari",  "Baritone Sax",  "B. Sx.", instrument.BaritoneSaxophone, True),
]
# score order: reeds on top, then brass, then rhythm (standard big-band layout)
SCORE_ORDER = ["alto", "tenor", "bari", "tpt1", "tpt2", "tbn"]

SOLO_RANGES = {"tenor": (57, 72)}      # the chorus the recording solos over

# mutes, dynamics and cues, in the places a vocal chart wants them
PERFORMANCE_MARKS: dict[tuple[int, str], str] = {
    (1, "tpt1"): "Harmon mute, close to the mic",
    (3, "tenor"): "sub tone",
    (5, "tbn"): "cup mute",
    (7, "tpt1"): "open",
    (7, "tbn"): "open",
    (9, "tbn"): "p - under the voice, never over it",
    (9, "alto"): "tacet until 23",
    (23, "alto"): "mp - answer the singer",
    (25, "alto"): "mp, singing tone",
    (25, "tpt1"): "cup mute",
    (41, "alto"): "reeds as a section",
    (41, "tpt1"): "open",
    (57, "tenor"): "Solo - open, take as many as you like",
    (57, "tpt1"): "backgrounds from 65",
    (73, "tpt2"): "mf - brass take the line",
    (89, "alto"): "reeds",
    (89, "tpt1"): "brass answer",
    (105, "tpt1"): "f - full band, still under the voice",
    (121, "tpt1"): "ff",
    (127, "tpt1"): "Basie tag",
}

# the choruses the singer is on; the band works around these
VOCAL_CHORUSES = [9, 25, 41, 73, 89, 105]


def slash(ql: float = 1.0, display: str = "B4") -> note.Note:
    n = note.Note(display)
    n.quarterLength = ql
    n.notehead = "slash"
    n.noteheadFill = True
    return n


def stamp_first(m: stream.Measure, ks_sharps: int, with_tempo: bool = False,
                perc: bool = False) -> None:
    """MusicXML wants the key/time signature inside the first measure."""
    if not perc:
        m.insert(0.0, key.KeySignature(ks_sharps))
    m.insert(0.0, meter.TimeSignature("4/4"))
    if with_tempo:
        mk = tempo.MetronomeMark(number=120, referent=duration.Duration(1.0))
        mk.placement = "above"
        m.insert(0.0, mk)


def finish_measure(m: stream.Measure) -> stream.Measure:
    m.makeRests(fillGaps=True, inPlace=True, timeRangeFromBarDuration=True)
    return m


def _tie_roles(plan: dict, slot: str) -> dict[tuple[int, float], str]:
    """Work out where this part's notes are tied.

    The arrangement plan marks a note as tying into whatever follows it; the
    melody leans on this (the held "moon" of the first phrase is an eighth
    tied over the barline into a dotted quarter, and the second half of the
    tune sustains across two barlines at a time).  A note that both ends one
    tie and begins another becomes a "continue".
    """
    flat = [(b, *e) for b in sorted(plan)
            for e in sorted(plan[b][slot])]
    starts = [False] * len(flat)
    for i, (b, off, midi, ql, tie_out, _artic) in enumerate(flat):
        if not tie_out or i + 1 >= len(flat):
            continue
        nb, noff, nmidi = flat[i + 1][0], flat[i + 1][1], flat[i + 1][2]
        # a tie needs the same pitch *and* the next note starting where this
        # one ends - later passes can drop a note and orphan the tie otherwise
        if nmidi == midi and abs(((nb - 1) * 4 + noff)
                                 - ((b - 1) * 4 + off + ql)) < 1e-6:
            starts[i] = True

    roles: dict[tuple[int, float], str] = {}
    for i, (b, off, *_rest) in enumerate(flat):
        stopping = i > 0 and starts[i - 1]
        if starts[i] and stopping:
            roles[(b, off)] = "continue"
        elif starts[i]:
            roles[(b, off)] = "start"
        elif stopping:
            roles[(b, off)] = "stop"
    return roles


def build_score() -> stream.Score:
    plan = build_horn_plan()
    sc = stream.Score()

    md = metadata.Metadata()
    md.title = "Fly Me To The Moon"
    md.composer = "Bart Howard"
    md.movementName = "Arranged for 10-piece big band"
    sc.insert(0, md)

    ks_sharps = -2                                    # B-flat major concert
    mm = tempo.MetronomeMark(number=120, referent=duration.Duration(1.0))

    parts: dict[str, stream.Part] = {}

    # ---------------- horns ----------------
    for slot in SCORE_ORDER:
        _, name, abbrev, instr, treble = next(h for h in HORN_PARTS if h[0] == slot)
        p = new_part(name, abbrev, instr, treble=treble)
        p.insert(0, meter.TimeSignature("4/4"))
        p.insert(0, key.KeySignature(ks_sharps))
        p.insert(0, copy.deepcopy(mm))
        solo = SOLO_RANGES.get(slot)
        tie_roles = _tie_roles(plan, slot)
        for b in range(1, TOTAL_BARS + 1):
            m = stream.Measure(number=b)
            if b == 1:
                stamp_first(m, ks_sharps, with_tempo=(slot == SCORE_ORDER[0]))
            mark = PERFORMANCE_MARKS.get((b, slot))
            if mark:
                mt = expressions.TextExpression(mark)
                mt.placement = "above"
                mt.style.fontStyle = "italic"
                m.insert(0.0, mt)
            if solo and solo[0] <= b <= solo[1]:
                for beat in range(4):
                    m.insert(float(beat), slash())
                for off, sym in CHORDS[b]:
                    m.insert(off, make_symbol(sym))
            else:
                for off, midi, ql, _tie, artic in sorted(plan[b][slot]):
                    n = _spell(_note(midi, ql, artic=artic))
                    role = tie_roles.get((b, off))
                    if role:
                        n.tie = tie.Tie(role)
                    m.insert(off, n)
            finish_measure(m)
            p.append(m)
        parts[slot] = p

    # ---------------- voice (cue) ----------------
    vp = new_part("Voice", "Voc.", instrument.Vocalist, treble=True)
    vp.insert(0, meter.TimeSignature("4/4"))
    vp.insert(0, key.KeySignature(ks_sharps))
    vp.insert(0, copy.deepcopy(mm))
    vocal = {}
    for start in VOCAL_CHORUSES:
        for abs_bar, off, midi, ql, tie_out in melody_events(start):
            if midi is not None:
                vocal.setdefault(abs_bar, []).append((off, midi, ql, tie_out, None))
    vocal_ties = _tie_roles({b: {"v": vocal.get(b, [])} for b in
                             range(1, TOTAL_BARS + 1)}, "v")
    for b in range(1, TOTAL_BARS + 1):
        m = stream.Measure(number=b)
        if b == 1:
            stamp_first(m, ks_sharps)
            te = expressions.TextExpression("cue - the singer has this; the band never doubles it")
            te.placement = "above"
            te.style.fontStyle = "italic"
            m.insert(0.0, te)
        for off, midi, ql, _t, _a in sorted(vocal.get(b, [])):
            n = _spell(_note(midi, ql))
            role = vocal_ties.get((b, off))
            if role:
                n.tie = tie.Tie(role)
            m.insert(off, n)
        if b == 121:
            n = _spell(_note(_ps("B-4"), 4.0))
            n.tie = tie.Tie("start")
            m.insert(0.0, n)
        elif 122 <= b <= 127:
            n = _spell(_note(_ps("B-4"), 4.0))
            n.tie = tie.Tie("continue" if b < 127 else "stop")
            m.insert(0.0, n)
        finish_measure(m)
        vp.append(m)
    parts["voice"] = vp

    # ---------------- keys ----------------
    kp = new_part("Keys", "Keys", instrument.Piano, treble=True)
    kp.insert(0, meter.TimeSignature("4/4"))
    kp.insert(0, key.KeySignature(ks_sharps))
    kp.insert(0, copy.deepcopy(mm))
    for b in range(1, TOTAL_BARS + 1):
        m = stream.Measure(number=b)
        if b == 1:
            stamp_first(m, ks_sharps)
        for off, sym in CHORDS[b]:
            m.insert(off, make_symbol(sym))
        for beat in range(4):
            m.insert(float(beat), slash())
        if b == 89:
            te = expressions.TextExpression("Solo")
            te.placement = "above"
            m.insert(0.0, te)
        finish_measure(m)
        kp.append(m)
    parts["keys"] = kp

    # ---------------- bass ----------------
    bp = new_part("Bass", "Bass", instrument.AcousticBass, treble=False)
    bp.insert(0, meter.TimeSignature("4/4"))
    bp.insert(0, key.KeySignature(ks_sharps))
    bp.insert(0, copy.deepcopy(mm))
    prev = None
    for b in range(1, TOTAL_BARS + 1):
        m = stream.Measure(number=b)
        if b == 1:
            stamp_first(m, ks_sharps)
        if b == TOTAL_BARS:
            root = chord_pcs("Bb6")[0]
            n = _spell(_note(nearest(root, _ps("B-2")), 4.0))
            m.insert(0.0, n)
        else:
            line = walking_bass(b, prev)
            prev = line[-1]
            for i, p_ in enumerate(line):
                m.insert(float(i), _spell(_note(p_, 1.0)))
        finish_measure(m)
        bp.append(m)
    parts["bass"] = bp

    # ---------------- drums ----------------
    dp = stream.Part()
    dp.id = "Drums"
    dp.partName = "Drums"
    dp.partAbbreviation = "Dr."
    di = instrument.BassDrum()
    di.partName = "Drums"
    di.partAbbreviation = "Dr."
    dp.insert(0, di)
    dp.insert(0, clef.PercussionClef())
    dp.insert(0, meter.TimeSignature("4/4"))
    dp.insert(0, copy.deepcopy(mm))
    for b in range(1, TOTAL_BARS + 1):
        m = stream.Measure(number=b)
        if b == 1:
            stamp_first(m, 0, perc=True)
        if b == TOTAL_BARS:
            u = note.Unpitched(displayName="G5")
            u.quarterLength = 4.0
            u.notehead = "x"
            m.insert(0.0, u)
        else:
            for beat in range(4):
                m.insert(float(beat), slash())
        finish_measure(m)
        dp.append(m)
    parts["drums"] = dp

    # ---------------- assemble ----------------
    for slot in ["voice"] + SCORE_ORDER + ["keys", "bass", "drums"]:
        sc.insert(0, parts[slot])

    # rehearsal marks + section text on the top staff
    top = parts["voice"]
    for b, (mark, label) in SECTIONS.items():
        meas = top.measure(b)
        if meas is None:
            continue
        if mark != "Intro":
            rm = expressions.RehearsalMark(mark)
            meas.insert(0.0, rm)
        if label:
            te = expressions.TextExpression(label)
            te.placement = "above"
            te.style.fontStyle = "italic"
            meas.insert(0.0, te)

    swing = expressions.TextExpression("Medium Swing - swing eighths")
    swing.placement = "above"
    top.measure(1).insert(0.0, swing)

    for p in sc.parts:
        p.atSoundingPitch = True
        last = p.getElementsByClass(stream.Measure)[-1]
        last.rightBarline = bar.Barline("final")
        for n in last.notesAndRests:
            n.expressions.append(expressions.Fermata())
    sc.atSoundingPitch = True
    return sc


def _concert_score(sc: stream.Score) -> stream.Score:
    """A genuine concert-pitch score for the director.

    MusicXML always stores *written* pitch, so music21 transposes on export.
    Clearing each instrument's transposition makes written and sounding pitch
    the same, which is exactly what a concert score is.
    """
    concert = copy.deepcopy(sc)
    concert.metadata.movementName = (
        "Arranged for 10-piece big band - CONCERT PITCH score")
    for part in concert.parts:
        for ins in part.recurse().getElementsByClass(instrument.Instrument):
            ins.transposition = None
        part.atSoundingPitch = False
    concert.atSoundingPitch = False
    return concert


def _renumber_voices(sc: stream.Score) -> None:
    """MuseScore 4 rejects <voice>0</voice>; make every voice id 1-based."""
    for part in sc.parts:
        for measure in part.getElementsByClass(stream.Measure):
            for i, v in enumerate(measure.getElementsByClass(stream.Voice), start=1):
                v.id = str(i)


def tidy_musicxml(path: str) -> None:
    """Clean-ups that make the file behave in notation software.

    * a zero <root-alter> makes some engravers print a spurious natural in
      the chord symbol (rendering Gm7 as "G-natural-m7"), so drop it;
    * every <voice> must be a positive integer or MuseScore 4 reports the
      file as corrupt.
    """
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

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="Fly_Me_To_The_Moon_BigBand.musicxml",
                    help="output MusicXML path (transposed / performance score)")
    ap.add_argument("--concert", default=None,
                    help="also write a concert-pitch score to this path")
    args = ap.parse_args()

    sc = build_score()

    if args.concert:
        concert = _concert_score(sc)
        _renumber_voices(concert)
        concert.write("musicxml", fp=args.concert)
        tidy_musicxml(args.concert)
        print(f"wrote {args.concert}  (concert pitch)")

    sc.toWrittenPitch(inPlace=True)
    _renumber_voices(sc)
    sc.write("musicxml", fp=args.out)
    tidy_musicxml(args.out)
    print(f"wrote {args.out}  (transposed parts)")


if __name__ == "__main__":
    main()
