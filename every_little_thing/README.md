# Every Little Thing — piano transcription

The piano part from a piano-and-voice recording, on a grand staff.
`every_little_thing_piano.py` writes `Every Little Thing - Piano.musicxml`
(and `.mxl`). Only the piano is transcribed; the vocal is not part of this
score.

```bash
python every_little_thing_piano.py \
    -o "Every Little Thing - Piano.musicxml" \
    --mxl "Every Little Thing - Piano.mxl"
```

## What the recording is

| Stem | RMS | |
| ---- | --- | --- |
| piano | **0.0287** | the whole accompaniment |
| vocals | 0.0437 | lead and backing |
| bass, drums, guitar | 0.0001–0.0008 | noise floor — nothing there |

Demucs' six-source model separates it cleanly. There is no other instrument
to leak into the piano stem, which is the easy case for this kind of work.

## What was measured, and what was chosen

| | | How |
| --- | --- | --- |
| Key | **G major** | Krumhansl-Schmuckler, 0.94 against 0.65 for the next candidate. Every pitch class in the part is a note of the G major scale — there is not one chromatic note in the piece |
| Metre | **4/4** | the harmonic-change curve peaks at 0.906, 1.834, 3.738 and 7.338 s — a half bar, a bar, two bars, and a four-bar phrase |
| Tempo | **131 BPM, moving between 122 and 136** | a piecewise-linear time→beat map with a knot every 8 s, fitted by coordinate descent |
| Notes | **1438**, almost all C2–B4 | Basic Pitch (Spotify's ICASSP-2022 model) |
| Bar 1 beat 1 | **chosen, not measured** | see below |

## The tempo moves, so the grid moves with it

Quantising a rubato performance against a fixed clock produces rhythms nobody
played. The grid here is a map from audio time to eighth-note position whose
slope is free to change, fitted so that chords land on whole numbers. Fitted
that way the tempo runs 122–136 BPM, and the drift is real: a single fixed
tempo fits measurably worse.

**Eighths are as fine as this performance justifies.** On the fitted grid 66%
of chords land within a quarter of an eighth of their slot, against 50% for
random times; over any single twenty-second window it is about 70%. That is a
real pulse but a loose one. Sixteenths would be inventing detail the playing
does not contain, so the score stops at eighths and says *rubato*, because
the recording is.

Durations run to the next event in the same hand, up to two bars. The pedal
holds, and a line of eighth rests between every chord is neither what was
played nor anything anyone wants to read.

## The one thing that could not be measured

**Which beat is beat 1.** Five independent tests — bass-root positions modulo
the bar, harmonic-change energy against each of the eight candidate offsets,
and three others — all came back at chance (the best was 15% where chance is
12.5%). The reason is arithmetic rather than method: the residual is 0.19 of
an eighth, about 44 ms, and an eighth here is only 229 ms, so nothing
survives being counted modulo eight.

So the bar lines are anchored musically, at the opening, where the evidence
is local and free of accumulated drift: the piece begins on a G, the bass
moves to A exactly eight eighths later, and to C eight after that. That is a
defensible choice, stated as one. If the bar lines want moving by an eighth,
that is the number to change, and nothing else about the transcription
depends on it.
