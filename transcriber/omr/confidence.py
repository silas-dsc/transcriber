"""Confidence scoring and a human-in-the-loop review queue.

OMR cannot be perfect on real scans, so the honest path to a correct score is
to make the system *know where it is unsure* and surface only those spots for a
human to confirm.  This module fuses three independent confidence signals into
a per-measure score and a ranked review queue:

1. **Per-note recogniser confidence** -- e.g. the built-in recogniser's
   head-position ambiguity (a head halfway between two staff steps).
2. **Semantic checks** -- musical impossibilities flagged by
   :mod:`transcriber.omr.semantic` (a measure that does not add up, an
   out-of-range note, a low-confidence key).
3. **Multi-engine disagreement** -- where oemer / homr / the built-in
   recogniser disagree on a note, that note is exactly where to look.

The output is a :class:`ConfidenceReport` whose ``review_items`` are the
measures worth a human's attention, lowest-confidence first.  The fewer and
better the engines/checks, the shorter that queue -- which is how a human gets
a real score to 100% with minimal effort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from music21 import stream

from .eval.metrics import _align


@dataclass
class ReviewItem:
    """A spot flagged for human review."""

    part: int
    measure: int | None
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ConfidenceReport:
    """Confidence assessment of a recognised score."""

    overall: float
    review_items: list[ReviewItem] = field(default_factory=list)
    n_measures: int = 0

    @property
    def n_review(self) -> int:
        return len(self.review_items)

    def summary(self) -> str:
        return (
            f"confidence {self.overall:.2f}; {self.n_review} of {self.n_measures} "
            f"measure(s) flagged for review"
        )


def _note_pitches(score: stream.Score) -> list[int]:
    out: list[int] = []
    for n in score.flatten().notes:
        out.append(int((n.pitches[0] if n.isChord else n.pitch).midi))
    return out


def engine_agreement(primary: stream.Score, others: list[stream.Score]) -> dict[int, float]:
    """Per-note agreement of ``primary`` with the ``others`` engines.

    Returns ``{id(note): fraction_of_engines_that_agree}`` for the primary
    score's notes.  Notes all engines agree on score 1.0; contested notes score
    lower and bubble to the top of the review queue.
    """
    prim_notes = [n for n in primary.flatten().notes]
    if not others:
        return {id(n): 1.0 for n in prim_notes}

    prim_pitches = [int((n.pitches[0] if n.isChord else n.pitch).midi) for n in prim_notes]
    agree = [0] * len(prim_notes)
    for other in others:
        _, matches = _align(prim_pitches, _note_pitches(other))
        for i, _j in matches:
            agree[i] += 1
    return {id(prim_notes[i]): agree[i] / len(others) for i in range(len(prim_notes))}


# Confidence ceilings imposed by a semantic issue of each severity.
_SEVERITY_CAP = {"error": 0.3, "warning": 0.6, "info": 0.85}


def build_confidence(
    score: stream.Score,
    semantic_report=None,
    engine_scores: dict[str, stream.Score] | None = None,
    review_threshold: float = 0.85,
) -> ConfidenceReport:
    """Fuse the confidence signals into a per-measure report + review queue.

    Args:
        score: The recognised score (the one being returned).
        semantic_report: Optional :class:`~transcriber.omr.semantic.SemanticReport`.
        engine_scores: Optional ``{engine_name: score}`` for all engines that
            ran (used for disagreement; the primary score should be included).
        review_threshold: Measures below this confidence are queued for review.
    """
    # Per-note agreement across engines (primary vs the others).
    others = []
    if engine_scores and len(engine_scores) > 1:
        others = [s for s in engine_scores.values() if s is not score]
    agreement = engine_agreement(score, others)

    # Index semantic issues by (part, measure).
    sem_by_measure: dict[tuple[int | None, int | None], list] = {}
    if semantic_report is not None:
        for issue in semantic_report.issues:
            sem_by_measure.setdefault((issue.part, issue.measure), []).append(issue)

    items: list[ReviewItem] = []
    confidences: list[float] = []

    for pi, part in enumerate(score.parts):
        measures = list(part.getElementsByClass(stream.Measure)) or [part]
        for measure in measures:
            mnum = getattr(measure, "number", None)
            notes = list(measure.flatten().notes) if measure is not part else list(part.flatten().notes)
            reasons: list[str] = []
            conf = 1.0

            # 1. Per-note recogniser confidence (worst note in the measure).
            note_confs = [float(getattr(n, "omr_confidence", 1.0)) for n in notes]
            if note_confs and min(note_confs) < review_threshold:
                conf = min(conf, min(note_confs))
                reasons.append(f"ambiguous note position ({min(note_confs):.2f})")

            # 2. Multi-engine disagreement.
            if others and notes:
                ag = [agreement.get(id(n), 1.0) for n in notes]
                disagreed = sum(1 for a in ag if a < 0.999)
                if disagreed:
                    conf = min(conf, min(ag))
                    reasons.append(f"{disagreed} note(s) engines disagree on")

            # 3. Semantic issues touching this measure (and part-wide issues).
            for key in ((pi, mnum), (pi, None), (None, None)):
                for issue in sem_by_measure.get(key, []):
                    cap = _SEVERITY_CAP.get(issue.severity, 0.85)
                    if cap < conf:
                        conf = cap
                        reasons.append(issue.message)

            confidences.append(conf)
            if conf < review_threshold:
                items.append(ReviewItem(part=pi, measure=mnum, confidence=conf, reasons=reasons))

    items.sort(key=lambda it: it.confidence)
    overall = sum(confidences) / len(confidences) if confidences else 1.0
    return ConfidenceReport(overall=overall, review_items=items, n_measures=len(confidences))


def annotate_review(score: stream.Score, report: ConfidenceReport, color: str = "red") -> stream.Score:
    """Mark flagged measures/notes in the score for a human reviewer.

    Adds a ``review?`` text mark at the start of each flagged measure and
    colours its low-confidence notes, so the exported MusicXML opens in any
    editor with the uncertain spots highlighted.
    """
    from music21 import expressions

    flagged = {(it.part, it.measure) for it in report.review_items}
    for pi, part in enumerate(score.parts):
        for measure in part.getElementsByClass(stream.Measure):
            if (pi, getattr(measure, "number", None)) not in flagged:
                continue
            te = expressions.TextExpression("review?")
            te.style.color = color
            measure.insert(0.0, te)
            for n in measure.flatten().notes:
                if float(getattr(n, "omr_confidence", 1.0)) < 0.85:
                    n.style.color = color
    return score
