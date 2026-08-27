# Fly Me To The Moon — 10-piece big band

An arrangement of the Bart Howard standard, written to match a specific
backing-track recording rather than to a generic lead sheet. Everything about
the chart — key, tempo, form, changes — was measured from the audio.

## Files

`fly_me_to_the_moon_bigband.py` is the generator. Running it writes two scores:

| File | What it is |
| ---- | ---------- |
| `Fly Me To The Moon - 10-piece Big Band.musicxml` | Performance score, **transposed parts** — open this one in MuseScore and use *Parts* to extract individual books |
| `Fly Me To The Moon - 10-piece Big Band (Concert Score).musicxml` | **Concert-pitch** score for the director |

Both are plain MusicXML and open directly in MuseScore, Sibelius, Finale or
Dorico. The scores themselves are not committed — they are deterministic
output of the generator, and the repository already ignores `*.musicxml` as
build output. Run the command at the bottom of this file to produce them.

## Instrumentation

```
Reeds    Alto Sax (E-flat) · Tenor Sax (B-flat) · Baritone Sax (E-flat)
Brass    Trumpet 1 (B-flat) · Trumpet 2 (B-flat) · Trombone
Rhythm   Keys · Bass · Drums
```

Note this is the instrumentation exactly as specified, which totals **nine**
players. Adding guitar to the rhythm section is the usual way to make it ten.

## What the recording turned out to be

The supplied MP3 is a **rhythm-section play-along track** — the Demucs vocal
stem is silent (RMS 0.0001), so there is no melody anywhere in the audio; it is
bass, drums and comping only. The melody in this chart therefore comes from the
published tune, transposed into the recording's key, and every note of it was
checked against the changes measured from the audio.

| Parameter | Value | How it was established |
| --------- | ----- | ---------------------- |
| Key | **B-flat major / G minor** | chroma + Krumhansl-Schmuckler (best major B-flat, r = 0.57); confirms the file's own `Gm` tag |
| Tempo | **120.000 BPM exactly** | onset-grid search over the isolated drum stem — an exact round number, so the track is sequenced |
| Feel | Medium swing, 4/4 | — |
| Downbeat | 0.600 s | 99 of 175 detected harmonic changes fall in one half-beat bin, with a second cluster exactly one half-bar later (chords change on beats 1 and 3) |
| Form | 8-bar intro · **7 × 16-bar chorus** (bars 9–120) · 8-bar coda = **128 bars** | 16-bar cycle template-matched at every offset; it starts at bar 9 regardless of alignment |

## The changes

Recovered by averaging beat-synchronous chroma of the Demucs comping stem
across all seven choruses (which cancels the improvisation) and matching
against chord templates, with roots taken from a pYIN transcription of the
isolated bass stem:

```
| Gm7    | Cm7 | F7  | B♭maj7 |
| E♭maj7 | Am7♭5 | D7 | Gm7  G7 |
| Cm7    | F7  | B♭maj7 | G7  |
| Cm7    | F7  | B♭maj7 | D7  |      ← turnaround
```

That is the standard progression of the tune, a whole tone below the common
key of C. Confidence was high on the diagnostic bars — Cm7 0.95, D7 0.91,
Gm7 0.88, Am7♭5 0.84.

The chart's changes match the recording bar by bar at **z = 13.3** against a
shuffled-alignment baseline (0.692 vs 0.550 ± 0.011).

## Road map

| Bars | Mark | Section |
| ---- | ---- | ------- |
| 1–8 | — | Intro: rhythm section, low sax pad from bar 5, brass punches in 7, full-band unison lead-in in 8 |
| 9–24 | **A** | Head. Tenor sax states the melody over a low trombone/bari pad; at 17 the band joins in octaves |
| 25–40 | **B** | Harmonised ensemble — six-part concerted, mid-register |
| 41–56 | **C** | Tenor sax solo (open); brass backgrounds from 49 |
| 57–72 | **D** | Trumpet solo (open); sax backgrounds from 65 |
| 73–88 | **E** | Shout chorus |
| 89–104 | **F** | Keys solo; horn punches build from 101 |
| 105–120 | **G** | Head out, full band |
| 121–128 | **H** | Coda — three-punch "Basie" tag into a held B♭6 |

Solo sections are written as slash notation with chord symbols, so they are
open — take as many choruses as you like, or play them as written over the
backing track.

## Two things the voicing engine does deliberately

Both are ordinary big-band practice, and both are enforced in code:

- **Major 7th chords are voiced with the 6th, not the 7th.** In a close
  voicing the root and major 7th collide a semitone apart. Substituting the
  6th removes every such clash (verified: 0 adjacent semitones anywhere in the
  score) and the major 7th stays available as a tension.
- **An 11th in the lead suspends the chord.** The melody sits on B-flat over
  F7 and on E-flat over B♭maj7; those are voiced as F7sus4 and B♭6sus4 rather
  than mistaken for chromatic approach notes. Genuine chromatic approaches do
  get the standard parallel diminished-7th treatment.

With only one trombone covering six-part writing, the section uses two
textures: a five-way close voicing for the mid-register harmonised chorus, and
octave-doubled tiers (trumpets + alto on top, tenor and trombone doubling them
an octave below) once the lead climbs for the shout chorus and head out. Every
written part was range-checked:

```
Trumpet 1 A4–C6 · Trumpet 2 A3–A5 · Alto A4–D6
Tenor A4–D6 · Trombone A2–B♭4 · Baritone A4–C6
```

## Regenerating

```bash
python fly_me_to_the_moon_bigband.py \
    -o "Fly Me To The Moon - 10-piece Big Band.musicxml" \
    --concert "Fly Me To The Moon - 10-piece Big Band (Concert Score).musicxml"
```

Only `music21` is needed.
