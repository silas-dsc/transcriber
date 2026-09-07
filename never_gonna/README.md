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
* **Chorus 3 (101–116)** — shout chorus, all six horns on the accents.
* **Outro (117–124)** — sustained, thinning, last chord.

Drums are deliberately plain: one groove per section, backbeat always in the
same place, texture changing *between* sections and not within them — rim
click and hi-hat under the verses, ride and snare in the choruses, ride bell
in the last one. Fills are half-bar and land only in the last bar of a
section (plus one full bar out of the break).

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
