# Never Gonna Fall in Love Again — 10-piece big band

`never_gonna_fall_in_love_again_bigband.py` builds
`Never Gonna Fall in Love Again - 10-piece Big Band.musicxml` (and a `.mxl`
of the same content) from a live recording.

```
python never_gonna_fall_in_love_again_bigband.py
```

## Scoring

Voice, Alto Sax, Tenor Sax, Baritone Sax, Trumpet 1, Trumpet 2, Trombone,
Piano, Bass, Drums — nine players plus the singer.

Chord symbols sit on the **Piano** and **Bass** staves. Both are concert-pitch
parts, so the symbols read and play back at concert; nothing has to be
transposed to interpret them.

## What is transcribed and what is arranged

This matters, so it is stated plainly.

| Part | Source |
|---|---|
| Harmony, form, bar lengths | **transcribed** from the recording |
| Bass | **transcribed** (pYIN on the separated bass stem) |
| Voice | **transcribed**, guide quality — see caveats |
| Piano | **arranged** — there is no piano on the recording |
| Drums | **arranged** in the recording's own feel |
| All six horns | **arranged** — there are no horns on the recording |

The recording is accordion, cello, acoustic guitar, percussion, bass and
voice. Six-stem separation put the piano stem at −47.6 dB, i.e. empty, which
confirms it: the Keys part had to be written, not lifted.

## How the harmony and form were measured

Separation used Demucs `htdemucs_6s`. Everything below was measured, and each
figure is reported against what chance alone would give.

**Key** — C major / A minor, correlation 0.93 against Krumhansl-Schmuckler
profiles; the next candidate (A minor as a distinct centre) scored 0.69.

**Metre** — 4/4. Autocorrelating a harmonic-change function gave the 4/4 bar
length a score of **0.997** and the 3/4 (and 6/8) bar length **0.015**. The
two-bar and four-bar periods stayed strong (0.83, 0.75), so the grid holds
over whole phrases rather than just locally.

**Feel** — a bolero. The guitar, bass and percussion all accent eighth
positions 0, 3 and 6, i.e. **3+3+2**. It shows up three separate ways: as
autocorrelation peaks at 3 and 5 eighths inside an 8-eighth bar, as the
drum-accent profile across the bar, and in the note onsets of the transcribed
bass (57 / 56 / 45 attacks at positions 0 / 3 / 6, against 15–34 elsewhere).
The written figures sit on those accents.

**Tempo** — free, as you said. A single fixed tempo does not fit: onsets
against the best global constant grid give a mean error of 0.057 s versus
0.076 s for chance, which is barely better than guessing. A grid with knots
every 10 seconds tracks it properly and puts **87.4%** of onsets within a
quarter-eighth (chance 50%) and 76.5% within an eighth of an eighth (chance
25%). Local tempo runs 94–102 BPM around a median of 100. The score is
written at ♩=100 marked *rubato*: a moving click in the file would be a
fiction, and the band is playing to each other, not to a grid.

**Chords** — chroma template matching over the guitar/accordion/cello stems,
with the transcribed bass supplying each root, at half-bar resolution. The
cross-check that matters: **92.5% of the transcribed bass notes are chord
tones of the resulting chart**, and those two things were derived by
completely different methods (pitch tracking versus chroma templates), so
they corroborate each other rather than restate each other.

Two chords were decided rather than simply read off:

* **The second bar of the verse is Am/G, not G.** Am/G scores 1.88 against G
  major's 1.41, and the B that G major needs is nearly absent (0.278). That
  makes the verse the familiar line cliché **Am – Am/G – D7/F♯ – F – E7**
  over a descending A–G–F♯–F–E bass.
* **The F♯ bar is genuinely ambiguous.** D7/F♯ scores 2.318 and F♯m7♭5 2.248
  — a 3% gap. They differ only in D versus E and both notes sound. D7/F♯ is
  written because it is the more idiomatic reading of that descent and the
  simpler thing to hand a rhythm section, but this is a choice, not a
  measurement.

### Corrections after review

The chart was first built by fitting the two 8-bar templates to the whole
form. That was the wrong last step: where the band departs from the template,
the template won and the measurement lost. Five bars were corrected, three
reported by ear and two found by re-checking every bar against the audio:

| Bar | Was | Is | How |
|---|---|---|---|
| 8 | E7 | Am | measured — Am on beats 1–3 |
| 15 | E7 | Am | reported; measurement agreed (Am from beat 2) |
| 16 | F (4 beats) | F ⋅ E7 | reported; F measures as exactly 2 beats |
| 25 | E7 | F7 ⋅ E7 | measured — F on beats 1–2, margin 0.34 |
| 57 | E7 | F7 ⋅ E7 | reported; the raw measurement already said F ⋅ E7 |

Bars 25 and 57 are the same position in the form and measure almost
identically. The flat 7th is faint — E♭ reads 0.41 against 0.27 on the plain F
bars, about 1.5× — so F7 rests more on the report than on the measurement.

Every bar was then re-tested at beat resolution against the whole vocabulary.
Two things that looked like errors are not:

* **The Am/G bars.** The detector prefers "C/G" there, but Am7/G and C6/G are
  the same four notes — a labelling choice, not a disagreement. A (0.622)
  sits well above C (0.456), which is what makes Am/G the better name.
* **The second half of the chorus's fifth bar** (37, 45, 69, 77, 105, 113),
  where "G" scored up to 0.44 better than the written Am/G. That is the bass
  note inflating its own pitch class: in a bar that really is G, B measures
  0.398; in these bars B is 0.209 while A is 0.260 against 0.161. Am/G stands.

Everywhere else the symbols are kept plain (`Am`, `F`, `C`, `G`) as you asked.
Where a voicing carries an extra tone, it is one the chroma actually shows —
the 7th on Am, the 6th on C, F and G. `Fmaj7` was specifically *not* used:
plain F fits best (0.644), and of the four-note options F6 (0.603) beats
Fmaj7 (0.584).

## Form

124 bars. Verses in A minor, choruses in C.

| Bars | Section | | Bars | Section |
|---|---|---|---|---|
| 1–16 | Intro (verse changes ×2) | | 65–80 | Chorus 2 |
| 17 | Pickup | | 81–85 | **Percussion break** (5) |
| 18–25 / 26–32 | Verse 1 (8 + **7**) | | 86–92 + 93 / 94–100 | Verse 3, instrumental |
| 33–48 | Chorus 1 | | 101–116 | Chorus 3 |
| 49 | Pickup | | 117–124 | Outro |
| 50–57 / 58–64 | Verse 2 (8 + **7**) | | | |

Verse phrase: `Am | Am/G | D7/F♯ | D7/F♯ | F | E7 | Am | E7`
Chorus phrase: `C | F | G | C  G/B | Am  Am/G | F | G | C`

Sections were located by matching bar-level chroma blocks. The verse pattern
recurs at bars 1, 9, 18, 26, 50, 58, 86 and 94; the chorus at 33, 41, 65, 73,
101 and 109.

Three details of the form are irregular, and all three are real rather than
analysis artifacts:

* **The second phrase of every verse is 7 bars, not 8.** It happens at 26–32,
  58–64 and 94–100 — all three times, at the same point in the form. Grid
  drift would not reproduce the same elision three times in the same place.
* **The percussion break is 5 bars** (81–85). Everything drops 20–45 dB there
  except the drums; the band is out. Bass and piano are written tacet, marked
  `N.C.`
* **Bar 93 is a 2/4 bar.** The third verse's turnaround is cut short. This was
  found, not assumed: the final chorus matched the chorus template half a bar
  off (at "bar 100.5" and "108.5" on a uniform grid) while every earlier
  section matched on exact bar lines. Tracking it back located the missing
  half-bar at the end of the third verse's first phrase. Writing bar 93 as 2/4
  re-aligns everything after it, and chorus 3 then matches choruses 1 and 2
  bar for bar.

## The arrangement

Nat King Cole territory more than Sinatra, because the recording is already a
bolero and that is Cole's ground.

* **Intro (1–16)** — rhythm section alone for eight bars, then the reeds add a
  quiet sustained pad. Brass held back.
* **Verses 1 and 2** — the singer is exposed. Trombone and baritone hold the
  harmony underneath; alto and tenor answer *only in bars where the singer has
  stopped*, which is computed from the transcribed vocal rather than guessed.
  A cup-muted first trumpet takes the counter-line in verse 2.
* **Choruses 1 and 2** — brass punch the 3+3+2 while the reeds sustain over
  the top; chorus 2 opens out to the full ensemble.
* **Percussion break (81–85)** — one tutti chord on 81, then the drums alone.
* **Verse 3 (86–100)** — instrumental on the recording, so the reeds take it
  as a soli, with the brass answering at the end of each phrase.

**Melodic fills.** Where the singer stops and the accordion or cello carries a
line, that line goes to a horn rather than being replaced by an invented
figure, and the instrument changes from phrase to phrase — tenor, first
trumpet, alto, trombone, second trumpet, and round again. Eleven phrases, 50
notes. The part taking a phrase is cleared right across it so the line is
exposed instead of sounding against its own pad.

Getting those lines out took two attempts. Tracking the highest strong bin of
a CQT looked plausible and was worthless: run on the *vocal* stem, where a
melody certainly exists, it returned only 50–56% in-key notes against a 58%
chance rate, so the chromatic wandering it produced on the instrumental stems
said nothing at all. pYIN on the accordion/cello stem is sound — its notes are
87–92% in-key — so the extraction threshold was loosened until the in-key rate
started falling toward chance, which is what fixes it at 145 notes. One phrase
that came back entirely out-of-key was dropped rather than cleaned up, and
isolated octave slips inside a phrase are pulled back to their neighbours.

The guitar is not a source for this: it strums, and pYIN finds a confident
pitch in only 1.0% of its frames against 7.3% for the accordion and cello.
* **Chorus 3 (101–116)** — shout chorus, all six horns on the accents.
* **Outro (117–124)** — sustained, thinning, last chord.

**Drums** are deliberately plain: one groove per section, backbeat always in
the same place, texture changing *between* sections and not within them — rim
click and hi-hat under the verses, ride and snare in the choruses, ride bell
in the last one. Fills are half-bar and land only in the last bar of a section
(plus one full bar out of the break). Standard kit only — kick, snare, toms,
hi-hat, ride, crash — no auxiliary or Latin percussion, even though the
recording's own percussion is a bolero.

The part is written as a proper drum set, which takes some work in MusicXML:
music21 exports each note's staff position correctly but declares only one
instrument for the whole part, so a reader maps every note to that single
sound and collapses the staff onto one line. `_write_drumset` rebuilds the
part list with one score-instrument per kit piece actually used, with its
General MIDI number, and puts an instrument reference on every note. Kick,
snare, hi-hat and cymbals then sit on their own lines and play back as
themselves. Drums are also notated in two voices — stems up for hands, stems
down for feet — and struck rather than sustained: each note keeps the value
the groove asks for and the remainder of the bar is filled with rests, so a
kick reads as a quarter and an eighth rather than a tied half note.

**Piano** is comping, not a feature, but it is not the same bar 124 times.
Voicings are rootless in the Bill Evans sense: the bass keeps the root, minor
and dominant chords are voiced from the third or the seventh with the 9th on
top, plain major chords are taken as 6/9 rather than major-7 so the leading
tone never fights the tune, and E7 gets its flat 9, which is what belongs in
A minor. Every inversion that fits inside a tenth is tried and the one whose
top note moves least from the previous chord wins, so the top voice walks
instead of jumping. The left hand alternates between a bare root, a root-and-
seventh shell, a root and fifth, and nothing at all — letting the bass carry
it on its own is part of the vocabulary.

Rhythm is drawn from a library of ten comping figures rather than one. The
tune's own 3+3+2 is in there, along with anticipations, late entries,
sustained bars and bars of silence; each section draws on its own set, and the
verses are sparser because the singer is exposed. A short descending
right-hand line replaces the chord in the last bar of each section. The
result is 12 distinct rhythms across the chart, between 0 and 5 attacks per
bar, with the piano laying out entirely in 12 bars and the left hand resting
in 19.

## Playability

Checked mechanically, since you asked for it specifically:

* Every part is inside a conservative written range — no part touches an
  extreme. Alto E4–B5, Tenor E4–B5, Baritone B3–A5, Trumpet 1 B3–F♯5,
  Trumpet 2 A3–D5, Trombone G2–G4, Bass A1–G3.
* **No leap between consecutive notes exceeds an octave anywhere in the
  score.** The horns are much tighter than that: the trombone's largest leap
  is 5 semitones, the alto's 8. Section voicings are octave-shifted as a
  block, and a smoothing pass then puts every note in the octave nearest what
  that player just played.
* No part crosses another within a section voicing.
* Parts written in sharps contain no flat spellings (concert F♯ reaches the
  E♭ horns as D♯, not E♭).
* All 124 bars are the correct length in all 11 staves, including the 2/4 bar.

## Caveats

* **The vocal staff is a guide, not an edition.** It is pitch-tracked, and
  tracking a voice inside a live mix is the least reliable thing here. Isolated
  octave slips are corrected and single-eighth tracking dropouts inside held
  notes are closed, but the rhythms are approximate and ornaments and slides
  are not represented. Use it for entries and contour, not for phrasing.
* **The horn parts are invention.** They are consistent with the recording's
  harmony and feel, but nothing in the recording plays them.
* The tempo map is a measurement of a rubato performance. Bar lines are in
  musically sensible places, but a player following a click will not match the
  record.
* `N.C.` is written with MusicXML's `kind text="N.C."`. MuseScore renders it;
  some lighter-weight renderers show the root instead.
