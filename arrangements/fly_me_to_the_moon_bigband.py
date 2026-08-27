"""Generate a 10-piece big band arrangement of "Fly Me To The Moon".

The arrangement is written to match a specific backing-track recording whose
musical parameters were measured directly from the audio:

    key      B-flat major / G minor   (chroma + Krumhansl-Schmuckler)
    tempo    120.000 BPM exactly      (onset-grid search on the drum stem)
    feel     medium swing, 4/4
    form     8-bar intro
             7 x 16-bar chorus (bars 9-120)
             8-bar coda        (bars 121-128)

The 16-bar cycle was recovered by averaging beat-synchronous chroma of the
Demucs "other" (comping) stem across all seven choruses and matching it
against chord templates, with bass roots taken from a pYIN transcription of
the isolated bass stem.

Instrumentation (as requested):
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


def fit_line(pitches: list[int], who: str) -> list[int]:
    """Octave-shift a whole phrase as a unit so its shape is never broken."""
    lo, hi = CONCERT_RANGE[who]
    best, best_cost = 0, None
    for shift in range(-3, 4):
        cand = [x + 12 * shift for x in pitches]
        cost = sum(max(0, lo - c) + max(0, c - hi) for c in cand)
        centre = abs(sum(cand) / len(cand) - (lo + hi) / 2)
        if best_cost is None or (cost, centre) < best_cost:
            best, best_cost = shift, (cost, centre)
    return [x + 12 * best for x in pitches]


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
    9:   ("A", "Head - melody"),
    25:  ("B", "Ensemble - harmonised"),
    41:  ("C", "Tenor Sax solo"),
    57:  ("D", "Trumpet solo"),
    73:  ("E", "Shout chorus"),
    89:  ("F", "Keys solo - backgrounds"),
    105: ("G", "Head out"),
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


# rhythmic figures for the shout chorus (offset, quarterLength)
SHOUT_A = [(0.0, 1.0), (1.5, 0.5), (2.0, 1.0), (3.5, 0.5)]
SHOUT_B = [(0.0, 1.5), (1.5, 0.5), (2.5, 0.5), (3.0, 1.0)]
SHOUT_C = [(0.0, 0.5), (1.0, 0.5), (1.5, 0.5), (2.5, 0.5), (3.0, 1.0)]
SHOUT_D = [(0.0, 2.0), (2.5, 0.5), (3.0, 1.0)]

# background figures behind the solos (offset, quarterLength)
BG_LONG = [(0.0, 4.0)]
BG_PUSH = [(0.0, 1.5), (1.5, 2.5)]
BG_HITS = [(1.5, 0.5), (2.5, 1.5)]


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
    """plan[bar][slot] = list of (offset, midi, ql, tie_out, artic)."""
    plan = {b: {s: [] for s in SLOTS} for b in range(1, TOTAL_BARS + 1)}

    def put(bar_no, slot, off, midi, ql, tie_out=False, artic=None):
        if 1 <= bar_no <= TOTAL_BARS:
            plan[bar_no][slot].append((off, midi, ql, tie_out, artic))

    # ---- Intro (1-8) -----------------------------------------------------
    for b in (5, 6):
        sym = chord_at(b, 0.0)
        v = ensemble_voicing(pick_lead(sym, _ps("C5")), sym)
        for slot in ("alto", "tenor", "bari"):
            put(b, slot, 0.0, v[slot], 4.0)
    # bar 7: brass punches; bar 8: unison lead-in to the head
    sym7 = chord_at(7, 0.0)
    v7 = ensemble_voicing(pick_lead(sym7, _ps("F5")), sym7)
    for slot in ("tpt1", "tpt2", "tbn"):
        put(7, slot, 0.0, v7[slot], 1.0, artic="accent")
        put(7, slot, 2.0, v7[slot], 1.0, artic="accent")
    lead_in_offsets = [2.5, 3.0, 3.5]
    lead_in_pitches = [_ps("F4"), _ps("G4"), _ps("A4")]
    for slot in SLOTS:
        base = [x + 12 for x in lead_in_pitches] if slot in ("tpt1", "tpt2") else list(lead_in_pitches)
        for off, p in zip(lead_in_offsets, fit_line(base, slot)):
            put(8, slot, off, p, 0.5, artic="accent")

    # ---- A: head, bars 9-24 ---------------------------------------------
    # bars 9-16: tenor sax states the melody alone over a low sustained pad;
    # bars 17-24: the band joins in octaves and the texture opens up
    for abs_bar, off, midi, ql, tie_out in melody_events(9):
        if midi is None:
            continue
        put(abs_bar, "tenor", off, fit(midi, "tenor"), ql, tie_out)
        if abs_bar > 16:
            put(abs_bar, "alto", off, fit(midi + 12, "alto"), ql, tie_out)
            put(abs_bar, "tpt1", off, fit(midi + 12, "tpt1"), ql, tie_out)
            put(abs_bar, "tbn", off, midi - 12, ql, tie_out)
            sym = chord_at(abs_bar, off)
            second = voice_below(midi + 12, sym, 2)[1]
            put(abs_bar, "tpt2", off, fit(second, "tpt2"), ql, tie_out)
    for b in range(9, 17):                     # soft low pad under the first eight
        sym = chord_at(b, 0.0)
        root, tones, _ = chord_pcs(sym)
        ordered = sorted(tones, key=lambda pc: (pc - root) % 12)
        put(b, "bari", 0.0, fit(nearest(root, _ps("B-2")), "bari"), 4.0)
        put(b, "tbn", 0.0, fit(nearest(ordered[2], _ps("F3")), "tbn"), 4.0)
    for b in range(17, 25):                    # baritone anchors the bottom
        sym = chord_at(b, 0.0)
        root = chord_pcs(sym)[0]
        put(b, "bari", 0.0, fit(root + 36, "bari"), 4.0)

    # ---- B: harmonised ensemble, bars 25-40 ------------------------------
    for abs_bar, off, midi, ql, tie_out in melody_events(25):
        if midi is None:
            continue
        sym = chord_at(abs_bar, off)
        v = ensemble_voicing(midi, sym)
        for slot in SLOTS:
            put(abs_bar, slot, off, v[slot], ql, tie_out)

    # ---- C: tenor solo 41-56, brass background from 49 -------------------
    for i, b in enumerate(range(49, 57)):
        sym = chord_at(b, 0.0)
        fig = BG_LONG if i % 4 in (0, 1) else BG_PUSH
        lead = pick_lead(sym, _ps("G5"))
        v = ensemble_voicing(lead, sym)
        for off, ql in fig:
            for slot in ("tpt1", "tpt2", "tbn"):
                put(b, slot, off, v[slot], ql, artic=None)

    # ---- D: trumpet solo 57-72, sax background from 65 -------------------
    for i, b in enumerate(range(65, 73)):
        sym = chord_at(b, 0.0)
        fig = BG_LONG if i % 4 in (0, 1) else BG_HITS
        lead = pick_lead(sym, _ps("E-5"))
        v = ensemble_voicing(lead, sym)
        for off, ql in fig:
            for slot in ("alto", "tenor", "bari"):
                put(b, slot, off, v[slot], ql)

    # ---- E: shout chorus 73-88 -------------------------------------------
    figs = [SHOUT_A, SHOUT_B, SHOUT_C, SHOUT_D]
    target = _ps("B-5")
    for i, b in enumerate(range(73, 89)):
        sym = chord_at(b, 0.0)
        fig = figs[i % 4]
        if i >= 14:
            fig = SHOUT_D
        lead = pick_lead(sym, target)
        target = lead
        v = ensemble_voicing(lead, sym, spread=True)
        for off, ql in fig:
            s2 = chord_at(b, off)
            vv = (ensemble_voicing(pick_lead(s2, lead), s2, spread=True)
                  if s2 != sym else v)
            for slot in SLOTS:
                put(b, slot, off, vv[slot], ql, artic="accent")

    # ---- F: keys solo 89-104, horn punches building from 101 -------------
    for b in range(101, 105):
        sym = chord_at(b, 0.0)
        v = ensemble_voicing(pick_lead(sym, _ps("A5")), sym, spread=True)
        for off, ql in [(1.5, 0.5), (3.0, 1.0)]:
            for slot in SLOTS:
                put(b, slot, off, v[slot], ql, artic="accent")

    # ---- G: head out 105-120 (spread voicing, full band) ------------------
    for abs_bar, off, midi, ql, tie_out in melody_events(105):
        if midi is None:
            continue
        sym = chord_at(abs_bar, off)
        v = ensemble_voicing(midi + 12, sym, spread=True)
        for slot in SLOTS:
            put(abs_bar, slot, off, v[slot], ql, tie_out)

    # ---- H: coda 121-128 --------------------------------------------------
    for b in range(121, 127):
        sym = chord_at(b, 0.0)
        lead = pick_lead(sym, _ps("B-5"))
        v = ensemble_voicing(lead, sym, spread=True)
        for off, ql in ([(0.0, 2.0), (2.5, 1.5)] if b % 2 else [(0.0, 4.0)]):
            for slot in SLOTS:
                put(b, slot, off, v[slot], ql, artic="accent" if b % 2 else None)
    # final two bars: the "Basie" tag - three punches then the held chord
    v_end = ensemble_voicing(pick_lead("Bbmaj7", _ps("B-5")), "Bbmaj7", spread=True)
    for off, ql in [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5)]:
        for slot in SLOTS:
            put(127, slot, off, v_end[slot], ql, artic="marcato")
    v_fin = ensemble_voicing(pick_lead("Bb6", _ps("B-5")), "Bb6", spread=True)
    for slot in SLOTS:
        put(128, slot, 0.0, v_fin[slot], 4.0)
    return plan


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

SOLO_RANGES = {"tenor": (41, 56), "tpt1": (57, 72)}


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
    for i, (_, _, midi, _, tie_out, _) in enumerate(flat):
        if tie_out and i + 1 < len(flat) and flat[i + 1][2] == midi:
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
            if solo and solo[0] <= b <= solo[1]:
                for beat in range(4):
                    m.insert(float(beat), slash())
                if b == solo[0]:
                    te = expressions.TextExpression("Solo (open)")
                    te.placement = "above"
                    m.insert(0.0, te)
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
    for slot in SCORE_ORDER + ["keys", "bass", "drums"]:
        sc.insert(0, parts[slot])

    # rehearsal marks + section text on the top staff
    top = parts[SCORE_ORDER[0]]
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
