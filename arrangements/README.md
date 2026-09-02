# Fly Me To The Moon — 10-piece big band, for a singer

A chart of the Bart Howard standard written to match a specific backing-track
recording. **The singer has the melody, so the band never plays it.** What the
band plays is the supporting material the saxophone plays on that recording,
handed around the sections the way a Sinatra chart hands it around.

## Files

`fly_me_to_the_moon_bigband.py` is the arrangement generator. Running it writes
`Fly Me To The Moon - 10-piece Big Band.musicxml` — plain MusicXML, opens
directly in MuseScore, Sibelius, Finale or Dorico. Use *Parts* to extract
individual books. The score is not committed: it is deterministic output of the
generator, and the repository already ignores `*.musicxml` as build output.

`fly_me_to_the_moon_drums.py` and `fly_me_to_the_moon_piano.py` are two more,
independent generators: the drum part and the piano part transcribed from the
recording, each on its own, as `Fly Me To The Moon - Drums.musicxml` and
`Fly Me To The Moon - Piano.musicxml` (both also `.mxl`). See
[The drum transcription](#the-drum-transcription) and
[The piano transcription](#the-piano-transcription).

`fly_me_to_the_moon_piano_comp.py` turns that transcription into something
playable — `Fly Me To The Moon - Piano Comp.musicxml`. See
[The playable piano part](#the-playable-piano-part).

`fly_me_to_the_moon_horns.py` is a horns-only chart — three saxes, two
trumpets and a trombone — built from the recording's solo saxophone:
`Fly Me To The Moon - Horns.musicxml`. See
[The horn chart](#the-horn-chart).

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
| Downbeat | **0.800 s** | least-squares fit of the kick pulse, then the bar phase from the bass line (see below) |
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

The downbeat in that table was originally read off harmonic-change bins as
0.600 s. Fitting the kick pulse for the drum transcription put it at 0.800 s
instead — the earlier figure was 0.4 of a beat early, because a chord change
smears across a window rather than landing on a frame. The chord sequence is
unaffected: bar-length chroma windows overlap 80% either way, and the roots
come out the same.

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

## The drum transcription

`fly_me_to_the_moon_drums.py` is a separate thing from the arrangement: it is
what the drummer on the recording actually plays, written out bar by bar, as a
one-staff drum part. Nothing about it is invented.

How it was recovered:

| Step | Result |
| ---- | ------ |
| Demucs (`htdemucs`) isolates the kit | a clean drum stem, 258.09 s |
| Least-squares fit of the kick pulse | 120.002 BPM, beat 1 of bar 1 at **0.800 s**, residual σ = 14 ms over 497 hits |
| Bar phase from the bass line | of the four candidate phases, only one makes the roots move in the descending fourths the tune is built from (E♭–A–D–G–C–F–B♭): **63%** against a 22% baseline for the other three |
| Onset strength per slot in six bands | 20–110 Hz, 110–260, 260–900, 2.5–7k, 7–12.5k, 12.5–20k, on a triplet-eighth grid (12 slots per bar × 128 bars) |
| Band balance names the piece struck | 499 bass drum, 236 hi-hat, 108 snare (34 accented), 7 crash |

The hi-hat is unmistakable: on beats 2 and 4 the strike is about four times
brighter in 12.5–20 kHz relative to 7–12.5 kHz than anything else on the
record, and decays far faster. It is there in 124 of the 128 bars.

**There is no ride pattern anywhere on the recording.** The 7–12.5 kHz band
carries only 13 events that are not the hi-hat; a swing ride at this tempo
would have left roughly 770. What the drummer plays is bass drum feathering
four to the bar, hi-hat on 2 and 4, and a snare that comps and fills — so
that is what the part says.

What the part does *not* claim:

* **No toms.** Every fill hit has snare snap in it; nothing in the low bands
  survives once the bass-drum bleed is accounted for. The fills are written as
  snare and bass drum, which is what the measurement supports.
* **Two dynamic levels only** — plain and accented. The accent threshold is a
  cut in snare-band onset strength, not a graded velocity.
* Ghost notes below the detection floor are not there.

Bars whose notation is an exact repeat of the bar before carry a `%` and are
otherwise empty, as before. There are 29 of them.

## The piano transcription

`fly_me_to_the_moon_piano.py` is the same idea as the drum part, for the
pianist: what is actually played, on a grand staff, D2–A♯5.

| Step | Result |
| ---- | ------ |
| Demucs' six-source model (`htdemucs_6s`) isolates the piano | real content, not bleed — RMS 0.023 and audibly active 72% of the track, where the guitar stem beside it is 0.0006 |
| Basic Pitch (Spotify's ICASSP-2022 polyphonic model) reads the stem | 1851 notes, A♯1–A♯5 |
| Note starts within 75 ms of each other are one strike | 637 strikes |
| Quantise each strike to the same triplet-eighth grid | the median strike lands **0.155 of a triplet-eighth** from a subdivision (28 ms); against a straight-sixteenth grid the median is 0.355 of a sixteenth, so the piano swings with the rest of the track |
| Check each strike back against the piano stem's own onset envelope | **336 strikes, 1450 notes** survive |

That last filter is the one worth trusting. It knows nothing about where beats
are — it only asks whether there is an attack in the audio under each strike —
and it removes every strike sitting on a middle triplet while keeping the
upbeats:

```
slot in bar   1   .   a   2   .   a   3   .   a   4   .   a
strikes      31   0  14  14   0  68  10   0  79  25   0  95
```

Which is the shape of the part: chords on beat 1 and on the upbeats of 2, 3
and 4, two or three a bar, left hand one to three notes, right hand two to
four, and 27% of the note-heads tied over a barline as pushes.

**Checked against the audio.** Turning the finished transcription back into a
chroma vector per bar and comparing it with the piano stem's own chroma gives
a mean agreement of **0.760**, against **0.566 ± 0.010** for the same bars
shuffled — z = 18.6 over 126 bars.

What the part does *not* claim: Basic Pitch is a general polyphonic model, not
a piano-specific one, so an inner voice here and there will be wrong even
where the chord is right — the 0.760 figure is the honest measure of how close
it is. Dynamics are not transcribed. The chord symbols are in the big band
chart, not here; this file is only the notes.

## The playable piano part

The transcription above is an accurate record and an awkward read: the same
note turns up in two or three octaves at once, strikes run to nine notes, and
nothing says which hand takes what.
`fly_me_to_the_moon_piano_comp.py` follows it closely and changes only two
things.

**No pitch class sounds in more than one octave.** That alone removes most of
the clutter — **442 of the transcription's 1450 notes** are octave duplicates
of something already in the chord, leaving 1008.

**Every chord is divided between the hands**, as evenly as the chord allows,
at a real gap near middle C. Of the 307 strikes that use both hands, **92%
are even or off by one note**. No hand spans more than **14 semitones** (mean
4.5), so everything is a grab.

Everything else is left alone — the pitches are the transcribed pitches, the
rhythm is the transcribed rhythm, and the wide spreads stay wide.

The part that took the most care is *which* octave a pitch class keeps, since
that is what preserves the shape:

* the top note is the line, and always stays;
* the bottom note anchors the chord, and stays unless it duplicates the top;
* every other pitch class keeps whichever of its transcribed octaves leaves
  the chord **most evenly spread**.

Without that last rule the written-out moments fall apart. The big B-flat
chord in bar 123 is spread over three octaves with the B-flat appearing three
times; keeping the lowest surviving octave of each pitch class turns it into
two notes at the bottom and one two octaves above, with a hole in the middle.
Choosing for even spread keeps it as F–D–B-flat across the two hands.

The ending survives intact, which was the point: the last three chords walk
C–E♭–B♭, C♯–E, D–F–B♭ onto the tonic, with the left hand out.

**The chord symbols stop at bar 120.** The verified 16-bar cycle holds through
the choruses, but the tag does not follow it — the cycle predicts Gm7 over the
last bars and what is actually played is a B-flat. Rather than print a chord
the recording contradicts, the tag is marked *ending — as played*.

## The horn chart

`fly_me_to_the_moon_horns.py` is six staves and nothing else: no rhythm
section, no vocal staff. What the horns play is what the **solo saxophone on
the recording** plays, transcribed note for note and then handed around the
section.

| Step | Result |
| ---- | ------ |
| Demucs' six-source model leaves the saxophone alone in `other` | piano and guitar are separated out |
| pYIN reads it as a single line | 94% of frames hold pitch inside a quarter-tone — it is monophonic, range E2–A4 |
| Keep only frames where the stem is genuinely sounding | **169 notes**; pYIN otherwise tracks noise and invents a line in silent bars |
| Quantise to the grid the drums and piano established | median onset **0.171 of a triplet-eighth (28 ms)** from a subdivision, and the onsets pile onto the beats and the swung "and"s exactly as the piano's do |

**The saxophone stops after the solo.** Measured against it, bars 42–54 and
74–120 are **67 and 70 dB down** — digital silence, not quiet playing. So the
horns rest there. They sound in bars 1–9, 11–41, 55–73 and 123–126, and
nowhere else. That leaves the last four choruses empty, which is what the
recording does; filling them would mean inventing material rather than
following the saxophone.

How it is handed around:

| Where | Treatment |
| ----- | --------- |
| under the vocal | one horn at a time, rotating — tenor, alto, trombone, trumpet 2 and back — playing the line as it was played |
| where there is no vocal | one to three of the upper horns carry the line while the lower ones **hold a pad** underneath: sustained chord tones that change with the harmony, not with the tune |

The no-vocal sections are the 8-bar intro, the solo chorus and the tag, and
which horns move and which hold rotates phrase by phrase.

**Why pads rather than block voicing.** The first version block-voiced
everything — all six horns moving together on every eighth — and it was
muddy. That is measurable. Counting pairs of neighbouring voices against the
usual low-interval limits (octaves below G2, fifths below C3, fourths below
F3, thirds below middle C):

| | pairs closer than the limit | trombone note length | baritone note length |
| --- | --- | --- | --- |
| block voicing | **28%** | 1 slot | 1 slot |
| line over pads | **13%** | 10 slots | 11 slots |

The low horns stopped moving on every note, which is the whole point — mud
comes from close intervals down low changing fast.

Pads also keep clear of the line: a held note a semitone from what the line
is playing is a grind rather than a passing rub, so below middle C the pad
avoids it. The finished score has **no semitone clash anywhere below middle
C**; the thirteen that remain are between moving upper voices, where a
second is ordinary. Every part is inside its range at sounding pitch.

325 notes in all — fewer than the 499 of the block version, because a pad is
one long note where the block voicing had a dozen short ones.

Each part is written at its own transposition, with the key signature to
match: alto and baritone in G, tenor and trumpets in C, trombone in B♭. The
**chord symbols are concert pitch** and are added after the parts are
transposed, so they are not transposed with them — an E♭ alto's staff would
otherwise label a concert Cm7 as Am7.

## Regenerating

```bash
python fly_me_to_the_moon_bigband.py -o "Fly Me To The Moon - 10-piece Big Band.musicxml"
python fly_me_to_the_moon_drums.py \
    -o "Fly Me To The Moon - Drums.musicxml" \
    --mxl "Fly Me To The Moon - Drums.mxl"
python fly_me_to_the_moon_piano.py \
    -o "Fly Me To The Moon - Piano.musicxml" \
    --mxl "Fly Me To The Moon - Piano.mxl"
python fly_me_to_the_moon_piano_comp.py \
    -o "Fly Me To The Moon - Piano Comp.musicxml" \
    --mxl "Fly Me To The Moon - Piano Comp.mxl"
python fly_me_to_the_moon_horns.py \
    -o "Fly Me To The Moon - Horns.musicxml" \
    --mxl "Fly Me To The Moon - Horns.mxl"
```

Only `music21` is needed to regenerate any of the five; the analysis behind
the transcriptions used Demucs, librosa, pYIN and Basic Pitch, and its results
are baked into the scripts.
