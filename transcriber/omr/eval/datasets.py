"""Corpora for testing and refining OMR accuracy.

Three sources, in increasing realism:

* :func:`synthetic_corpus` -- random monophonic phrases generated on the fly.
  No network, fully deterministic; ideal for unit tests and threshold tuning.
* :func:`music21_corpus` -- real music shipped *offline* with music21 (Bach
  chorales, folk songs, ...).  Reduced to a monophonic top line so it round
  trips through the built-in engraver.
* :func:`download_musicxml` / :data:`DATASET_URLS` -- fetch real PDF/MusicXML
  corpora (OpenScore Lieder, PDMX, Mutopia) when network access is available.

Each corpus yields ``CorpusItem`` records of ``(id, score)`` that the harness
renders, recognises and compares.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from music21 import chord, clef, key, meter, note, stream

logger = logging.getLogger(__name__)


@dataclass
class CorpusItem:
    """One reference example: an id and its ground-truth score."""

    id: str
    score: stream.Score


# Documented sources of real PDF/MusicXML data (used by :func:`download_musicxml`).
DATASET_URLS: dict[str, str] = {
    # OpenScore Lieder: 1300+ songs; the canonical corpus mirror.
    "openscore_lieder": "https://github.com/OpenScore/Lieder",
    # PDMX: 250k+ public-domain MusicXML scores (Zenodo).
    "pdmx": "https://zenodo.org/records/13763756",
    # Mutopia: LilyPond + PDF classical scores.
    "mutopia": "https://www.mutopiaproject.org/",
    # Aggregator of OMR datasets (DeepScores, MUSCIMA++, Camera-PrIMuS, ...).
    "omr_datasets": "https://apacha.github.io/OMR-Datasets/",
}


# Semitone classes of the "white keys" (C major / no accidentals).
_NATURAL_CLASSES = {0, 2, 4, 5, 7, 9, 11}


def synthetic_corpus(
    n_items: int = 8,
    notes_per_item: int = 10,
    seed: int = 0,
    durations: tuple[float, ...] = (1.0, 2.0),
    pitch_range: tuple[int, int] = (60, 81),
    diatonic_only: bool = True,
) -> list[CorpusItem]:
    """Generate random monophonic phrases as reference scores.

    Args:
        diatonic_only: Draw only natural (white-key) pitches.  The built-in
            engraver and primitive recogniser do not yet handle accidental
            glyphs, so chromatic notes cannot round-trip; restricting to
            naturals makes the benchmark a faithful measure of *geometric*
            detection accuracy rather than penalising unsupported accidentals.
    """
    rng = random.Random(seed)
    lo, hi = pitch_range
    pool = list(range(lo, hi + 1))
    if diatonic_only:
        pool = [p for p in pool if p % 12 in _NATURAL_CLASSES]

    items: list[CorpusItem] = []
    for i in range(n_items):
        items.append(
            CorpusItem(
                id=f"synthetic_{i:03d}",
                score=make_phrase(
                    [rng.choice(pool) for _ in range(notes_per_item)],
                    [rng.choice(durations) for _ in range(notes_per_item)],
                ),
            )
        )
    return items


def make_phrase(pitches: list[int], durations: list[float] | None = None) -> stream.Score:
    """Build a single-staff treble-clef score from MIDI pitches + durations."""
    durations = durations or [1.0] * len(pitches)
    score = stream.Score()
    part = stream.Part()
    part.insert(0, clef.TrebleClef())
    part.insert(0, meter.TimeSignature("4/4"))
    offset = 0.0
    for p, d in zip(pitches, durations):
        n = note.Note(int(p))
        n.quarterLength = float(d)
        part.insert(offset, n)
        offset += float(d)
    score.insert(0, part)
    score.makeNotation(inPlace=True)
    return score


def music21_corpus(query: str = "bach", limit: int = 8) -> list[CorpusItem]:
    """Load real scores from music21's bundled offline corpus.

    Each score is reduced to its monophonic top line so it round-trips through
    the built-in engraver.  Requires no network.
    """
    from music21 import corpus

    # getComposer indexes the well-known composers; for bundled *collections*
    # (ryansMammoth, essenFolksong, oneills1850, ...) fall back to matching the
    # query against the core corpus paths.
    paths = corpus.getComposer(query)
    if not paths:
        q = query.lower()
        paths = [p for p in corpus.getCorePaths() if q in str(p).lower()]

    items: list[CorpusItem] = []
    for path in paths:
        if len(items) >= limit:
            break
        try:
            parsed = corpus.parse(path)
        except Exception as exc:  # pragma: no cover - corpus parse edge cases
            logger.warning("Skipping %s: %s", path, exc)
            continue
        stem = Path(str(path)).stem
        # An ".abc" file can be an Opus holding several tunes; expand it.
        scores = list(parsed.scores) if isinstance(parsed, stream.Opus) else [parsed]
        for i, sc in enumerate(scores):
            if len(items) >= limit:
                break
            try:
                mono = monophonic_top_line(sc)
            except Exception as exc:  # pragma: no cover
                logger.warning("Skipping %s[%d]: %s", stem, i, exc)
                continue
            items.append(CorpusItem(id=f"{stem}_{i}" if len(scores) > 1 else stem, score=mono))
    return items


def monophonic_top_line(score: stream.Score) -> stream.Score:
    """Reduce a score to a single treble staff carrying its highest notes."""
    top = score.parts[0] if score.parts else score
    out = stream.Score()
    part = stream.Part()
    part.insert(0, clef.TrebleClef())
    # Preserve the source key signature so accidentals round-trip correctly.
    ks = top.recurse().getElementsByClass("KeySignature").first()
    if ks is not None and ks.sharps:
        part.insert(0, key.KeySignature(int(ks.sharps)))
    part.insert(0, meter.TimeSignature("4/4"))
    for n in top.flatten().notes:
        pitch = max(n.pitches, key=lambda p: p.midi) if isinstance(n, chord.Chord) else n.pitch
        new = note.Note(pitch.midi)
        new.quarterLength = float(n.quarterLength) or 1.0
        part.insert(float(n.offset), new)
    score_out = out
    score_out.insert(0, part)
    score_out.makeNotation(inPlace=True)
    return score_out


def download_musicxml(urls: list[str], dest: str | Path, limit: int | None = None) -> list[CorpusItem]:
    """Download MusicXML files from explicit URLs into ``dest`` and parse them.

    A thin, network-dependent helper for building corpora from the public
    sources in :data:`DATASET_URLS`.  Files that fail to download or parse are
    skipped with a warning.
    """
    import urllib.request

    from music21 import converter

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    items: list[CorpusItem] = []
    for i, url in enumerate(urls[: limit or len(urls)]):
        try:
            local = dest / f"{i:04d}_{Path(url).name}"
            urllib.request.urlretrieve(url, local)  # noqa: S310 - explicit user URLs
            items.append(CorpusItem(id=local.stem, score=converter.parse(str(local))))
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to fetch/parse %s: %s", url, exc)
    return items


# --------------------------------------------------------------------------- #
# Lead sheets with chord symbols (the jazz-fakebook payload)
# --------------------------------------------------------------------------- #
# A small pool of jazz chord qualities (music21 figure syntax) to draw from.
# music21 writes flats as "B-"/"E-" (not "Bb"); the renderer/OCR see proper
# flat glyphs and the metric normalises either spelling.
_JAZZ_QUALITIES = ("", "m7", "7", "maj7", "m7b5", "6", "m6", "9", "dim7")
_JAZZ_ROOTS = ("C", "D", "E", "F", "G", "A", "B-", "E-")


def make_lead_sheet(
    pitches: list[int],
    durations: list[float] | None = None,
    chords: list[tuple[float, str]] | None = None,
) -> stream.Score:
    """Build a single-staff treble lead sheet: a melody plus chord symbols.

    Args:
        pitches: Melody MIDI pitches.
        durations: Per-note quarter lengths (defaults to all quarter notes).
        chords: ``(offset_ql, figure)`` chord symbols in music21 figure syntax,
            e.g. ``[(0.0, "Cmaj7"), (4.0, "Am7"), (8.0, "Dm7"), (12.0, "G7")]``.
    """
    from music21 import harmony

    durations = durations or [1.0] * len(pitches)
    score = stream.Score()
    part = stream.Part()
    part.insert(0, clef.TrebleClef())
    part.insert(0, meter.TimeSignature("4/4"))
    offset = 0.0
    for p, d in zip(pitches, durations):
        n = note.Note(int(p))
        n.quarterLength = float(d)
        part.insert(offset, n)
        offset += float(d)
    for c_off, figure in chords or []:
        part.insert(float(c_off), harmony.ChordSymbol(figure))
    score.insert(0, part)
    score.makeNotation(inPlace=True)
    return score


def synthetic_lead_sheet_corpus(
    n_items: int = 4, bars: int = 8, seed: int = 0
) -> list[CorpusItem]:
    """Generate random diatonic melodies with one chord symbol per bar.

    Deterministic and offline -- ideal for exercising the chord-symbol
    render/recognise/compare loop without a network corpus.
    """
    rng = random.Random(seed)
    pool = [p for p in range(60, 82) if p % 12 in _NATURAL_CLASSES]
    items: list[CorpusItem] = []
    for i in range(n_items):
        pitches = [rng.choice(pool) for _ in range(bars * 4)]  # 4 quarter notes / bar
        chords = [
            (float(b * 4), rng.choice(_JAZZ_ROOTS) + rng.choice(_JAZZ_QUALITIES))
            for b in range(bars)
        ]
        items.append(
            CorpusItem(id=f"leadsheet_{i:03d}", score=make_lead_sheet(pitches, chords=chords))
        )
    return items
