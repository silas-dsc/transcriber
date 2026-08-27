# Fly Me To The Moon — 10-piece big band, for a singer

A chart of the Bart Howard standard written to match a specific backing-track
recording. **The singer has the melody, so the band never plays it.** What the
band plays is the supporting material the saxophone plays on that recording,
handed around the sections the way a Sinatra chart hands it around.

## Files

`fly_me_to_the_moon_bigband.py` is the generator. Running it writes
`Fly Me To The Moon - 10-piece Big Band.musicxml` — plain MusicXML, opens
directly in MuseScore, Sibelius, Finale or Dorico. Use *Parts* to extract
individual books. The score is not committed: it is deterministic output of the
generator, and the repository already ignores `*.musicxml` as build output.

**There is deliberately no separate concert-pitch file.** MusicXML always
stores *written* pitch, so a "C score" made by stripping each instrument's
transposition is a trap — the notes become sounding pitches but a reader still
range-checks them as written pitch, and every transposing part appears far
below its staff in red (a baritone sax notated D2–D4 on a treble staff). Concert
pitch is a display mode: in MuseScore it is the **Concert Pitch** button, which
re-notates and re-clefs properly.

## Instrumentation

```
Voice    cue staff — the singer; the band never doubles it
Reeds    Alto Sax (E-flat) · Tenor Sax (B-flat) · Baritone Sax (E-flat)
Brass    Trumpet 1 (B-flat) · Trumpet 2 (B-flat) · Trombone
Rhythm   Keys (grand staff) · Bass · Drums
```

Nine players plus the singer. The Voice staff is a cue: it is there so the band
can see what they are working around, and so the fills land in the right place.

### Ranges and clefs

Every part is inside its instrument's practical range, so no part needs an
octave clef. MuseScore range-checks *sounding* pitch, so both are listed:

| Part | Written | Sounds | MuseScore range | |
| ---- | ------- | ------ | --------------- | - |
| Voice | A3–B♭4 | A3–B♭4 | — | |
| Alto Sax | D4–D6 | F3–F5 | 49–80 | clean |
| Tenor Sax | C4–C#6 | B♭2–B4 | 44–75 | clean |
| Baritone Sax | B3–B5 | D2–D4 | 36–67 | clean |
| Trumpet 1 | G#3–C6 | F#3–B♭5 | 54–82 | clean |
| Trumpet 2 | G#3–A5 | F#3–G5 | 54–82 | clean |
| Trombone | F2–B♭4 | F2–B♭4 | 40–72 | clean |
| Bass | A2–C4 | A1–C3 | 28–60 | clean |

The bass is the one part that needed fixing. It **sounds an octave below where
it is written**, which the score now declares (`<transpose><octave-change>-1`),
so the part sits in the normal bass-clef register on the page and still plays
back at the right pitch. An 8vb clef would say the same thing, but combined
with the transpose element some readers shift the octave twice, so the plain
bass clef plus the declaration is the safer encoding.

## What is on the recording

The backing track carries no melody — the isolated vocal stem is silent. It
does carry a written saxophone part, and that is the model for this chart.
Separating the track six ways with Demucs puts piano and guitar in their own
stems and leaves the saxophone alone; transcribing it with pYIN shows two
distinct behaviours:

- a **sustained counterline** under where the vocal sits — long tones, one or
  two a bar. 25 of its 27 notes are chord tones or tensions and the other two
  are chromatic approaches, so it is written, not improvised;
- **eighth-note licks at the turnaround**. 44% of the saxophone's notes fall in
  bars 15–16 of the 16-bar cycle, which is only 12.5% of the bars — a
  three-and-a-half-fold concentration. That is the band answering at the end of
  each phrase, and it lands where the singer is holding.

Both are transcribed pitch-for-pitch into the generator (`COUNTER`, `LICK_A`,
`LICK_B`, `SAX_INTRO`) and everything the band plays is built from them.

| Parameter | Value | How it was established |
| --------- | ----- | ---------------------- |
| Key | **B-flat major / G minor** | chroma + Krumhansl-Schmuckler; confirms the file's own `Gm` tag |
| Tempo | **120.000 BPM exactly** | onset-grid search over the isolated drum stem — an exact round number, so the track is sequenced |
| Downbeat | 0.600 s | 99 of 175 harmonic changes fall in one half-beat bin |
| Form | 8-bar intro · 7 × 16-bar chorus · 8-bar coda = **128 bars** | 16-bar cycle template-matched at every offset |

The changes, recovered by averaging chroma across all seven choruses with roots
from the isolated bass:

```
| Gm7    | Cm7   | F7     | B♭maj7 |
| E♭maj7 | Am7♭5 | D7     | Gm7 G7 |
| Cm7    | F7    | B♭maj7 | G7     |
| Cm7    | F7    | B♭maj7 | D7     |   ← turnaround
```

They match the recording bar by bar at **z = 13.3** against a shuffled-alignment
baseline (0.692 vs 0.550 ± 0.011).

## Road map

Each chorus gives the counterline to a different colour, and the answers come
from whoever is not carrying it:

| Bars | Mark | Who has the line | Who answers |
| ---- | ---- | ---------------- | ----------- |
| 1–8 | — | Intro: the saxophone's own line, passed Harmon trumpet → tenor → cup trombone, whole band for the last two bars | |
| 9–24 | **A** | Trombone, an octave down, *p* | Reeds, at the turnaround |
| 25–40 | **B** | Alto | Brass, harmonised |
| 41–56 | **C** | Reeds as a section | Brass punctuate |
| 57–72 | **D** | *Tenor sax solo (open)* — the chorus the recording solos over | Brass pad from 65 |
| 73–88 | **E** | Trumpet 2 and trombone | Reeds |
| 89–104 | **F** | Reeds take the first half, brass the second | Whole band |
| 105–120 | **G** | Whole band, sustained under the voice | Whole band, in every window |
| 121–128 | **H** | Coda — swell under the last held note into the "Basie" tag | |

## Rhythm section

**Keys** is a grand staff. It comps from chord symbols and slashes for most of
the chart — the slashes are a guide, not a rhythm — but the intro and the
ending are written out as rootless right-hand voicings over left-hand roots
(the bass already has the root, so the right hand takes the chord tones above
it plus the 9th), and there are written right-hand fills in five gaps where the
horns are thin enough to leave room.

**Drums** are a real part rather than 128 bars of slashes. Each section states
its groove in full for two bars and is then marked with repeat signs, so the
player reads a pattern:

- the basic swing ride is the usual "ding, ding-da-ding" — a quarter then two
  swung eighths, twice a bar — over hi-hat with the foot on 2 and 4;
- from **B** the groove gets busier: the snare comps with the ride on the second
  eighth of each half-bar and the bass drum starts punctuating;
- every eight bars the repeats break for a written-out fill — half a bar
  mid-section, a full bar to set up the next one — and the bar after a fill
  takes a crash.

A `%` bar repeats the previous one, so it is genuinely empty — the notes are
replaced by a `<forward>` that accounts for the bar's duration without printing
anything. Leaving the notes in means a reader draws them *and* the repeat sign.

The part also carries a real drumset: one `<score-instrument>` per kit piece
with its General MIDI note, and every stroke tagged with the one it belongs to.
Without that, a reader maps every unpitched note to the part's single
instrument and stacks the whole kit on one line, whatever `display-step` says.
Kick, snare, toms, ride, crash and hi-hat each sit on their own line or space.

## Three rules the code enforces

Staying out of a singer's way is the whole job, so it is checked rather than
hoped for. On the finished score:

| Rule | Result |
| ---- | ------ |
| the band never doubles the melody | **0** notes at the same beat and pitch as the voice |
| nothing sits in the singer's octave while she articulates | **0** of 762 band notes in vocal choruses |
| busy figures only where she is holding | **98%** of short notes fall inside a held-note window |

The windows come from the melody itself: a run of three or more eighths with no
new syllable is somewhere an arranger can answer. Pads that would land in the
singer's octave are pushed down an octave; anything that still collides is
dropped.

Also checked: 0 semitone clashes between simultaneous parts, 0 malformed bars,
0 dangling ties, and every part inside the ranges listed above.

Two adjustments the voicing engine makes, both ordinary big-band practice:
major 7th chords are voiced with the 6th so the root and 7th never collide, and
an 11th in the lead suspends the chord rather than being mistaken for a
chromatic approach.

## Regenerating

```bash
python fly_me_to_the_moon_bigband.py -o "Fly Me To The Moon - 10-piece Big Band.musicxml"
```

Only `music21` is needed.
