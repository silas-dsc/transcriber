"""MusicXML post-processing: merge pages, repair, and score confidence.

These run on whatever a back-end produces, so they improve *every* engine's
output uniformly: stitch multi-page recognitions into one score, normalise it
through music21 so it is structurally valid, and compute a confidence heuristic
the ensemble uses to pick between engines.
"""

from __future__ import annotations

import logging

from music21 import note, stream

logger = logging.getLogger(__name__)


def merge_page_scores(scores: list[stream.Score]) -> stream.Score:
    """Concatenate per-page scores into one, joining each part across pages.

    Parts are matched by index (part 0 of page 2 continues part 0 of page 1).
    Measures are appended in page order; music21 renumbers them on export.
    """
    scores = [s for s in scores if s is not None]
    if not scores:
        return stream.Score()
    if len(scores) == 1:
        return scores[0]

    merged = stream.Score()
    # Carry metadata from the first page.
    if scores[0].metadata is not None:
        merged.insert(0, scores[0].metadata)

    max_parts = max(len(s.parts) for s in scores)
    for part_idx in range(max_parts):
        out_part = stream.Part()
        for page in scores:
            if part_idx >= len(page.parts):
                continue
            src = page.parts[part_idx]
            measures = list(src.getElementsByClass(stream.Measure))
            if measures:
                for measure in measures:
                    out_part.append(measure)
            else:
                # No measures yet (un-barred); copy notes/rests through.
                for el in src.notesAndRests:
                    out_part.append(el)
        merged.insert(0, out_part)

    merged.makeNotation(inPlace=True)
    return merged


def repair_score(score: stream.Score) -> stream.Score:
    """Normalise a score so it is valid and clean MusicXML.

    Idempotent and defensive: callable on any engine's output.  Ensures the
    score is barred and renumbers voices to positive integers (MuseScore 4
    rejects ``<voice>0</voice>``).
    """
    try:
        score.makeNotation(inPlace=True)
    except Exception as exc:  # pragma: no cover - music21 edge cases
        logger.warning("makeNotation failed during repair: %s", exc)
    for part in score.parts:
        for measure in part.getElementsByClass(stream.Measure):
            for i, voice in enumerate(measure.getElementsByClass(stream.Voice), start=1):
                voice.id = str(i)
    return score


def score_confidence(score: stream.Score) -> float:
    """Heuristic recognition confidence in ``[0, 1]``.

    No ground truth is available at inference time, so we use proxies for "this
    looks like a real transcription": it contains notes, and notes dominate
    rests.  A score that is mostly rests (or empty) is almost certainly a
    failed recognition and scores low.
    """
    notes = list(score.recurse().getElementsByClass(note.NotRest))
    rests = list(score.recurse().getElementsByClass(note.Rest))
    n_notes = len(notes)
    if n_notes == 0:
        return 0.0
    note_ratio = n_notes / (n_notes + len(rests) + 1e-9)
    # Saturating reward for having a reasonable number of notes.
    volume = min(1.0, n_notes / 16.0)
    return float(0.5 * note_ratio + 0.5 * volume)
