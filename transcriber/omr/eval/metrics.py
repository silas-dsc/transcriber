"""Symbolic OMR accuracy metrics computed from MusicXML / music21 scores.

The headline numbers follow standard OMR practice:

* **Note-level precision / recall / F1** -- align the predicted and reference
  note sequences and count pitch matches.  This is robust to absolute-timing
  errors (the recogniser may misjudge rhythm yet get every pitch right).
* **Symbol Error Rate (SER)** -- normalised edit distance over a tokenised note
  stream, the most common end-to-end OMR figure of merit.
* **MV2H-lite** -- an MV2H-inspired breakdown (multi-pitch, note value, and
  onset structure) giving a single 0-1 quality score.

Everything is derived from two :class:`music21.stream.Score` objects (or
MusicXML files), so it works against any engine and any reference corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from music21 import converter, stream


@dataclass
class NoteEvent:
    """A flattened note for comparison: ``(onset, duration, midi)`` in QL."""

    onset: float
    duration: float
    midi: int


@dataclass
class ScoreComparison:
    """Result of comparing a predicted score against a reference.

    Attributes:
        precision / recall / f1: Pitch-sequence note-level scores in ``[0, 1]``.
        symbol_error_rate: Pitch-token edit distance / reference length.
        duration_accuracy: Fraction of matched notes with the correct duration.
        onset_accuracy: Fraction of matched notes whose onset is within
            tolerance of the reference.
        mv2h_lite: Combined MV2H-style quality score in ``[0, 1]``.
        n_reference / n_predicted: Note counts.
        n_matched: Number of aligned pitch matches.
    """

    precision: float
    recall: float
    f1: float
    symbol_error_rate: float
    duration_accuracy: float
    onset_accuracy: float
    mv2h_lite: float
    n_reference: int
    n_predicted: int
    n_matched: int


def score_to_events(score: stream.Score | str) -> list[NoteEvent]:
    """Flatten a score (or a MusicXML path) into ordered :class:`NoteEvent`."""
    if isinstance(score, str):
        score = converter.parse(score)
    flat = score.flatten()
    events: list[NoteEvent] = []
    for n in flat.notes:
        onset = float(n.offset)
        dur = float(n.quarterLength)
        if n.isChord:
            for p in n.pitches:
                events.append(NoteEvent(onset, dur, int(p.midi)))
        else:
            events.append(NoteEvent(onset, dur, int(n.pitch.midi)))
    events.sort(key=lambda e: (e.onset, e.midi))
    return events


def _align(ref: list[int], pred: list[int]) -> tuple[int, list[tuple[int, int]]]:
    """Levenshtein alignment of two integer sequences.

    Returns ``(edit_distance, matches)`` where ``matches`` are the index pairs
    ``(i, j)`` aligned as *equal* (a substitution is not a match).
    """
    n, m = len(ref), len(pred)
    # dp[i][j] = edit distance between ref[:i] and pred[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost,  # match / substitution
            )

    # Backtrace, preferring diagonal moves so equal pairs are recorded.
    matches: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if ref[i - 1] == pred[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            if cost == 0:
                matches.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return dp[n][m], matches


def symbol_error_rate(reference: stream.Score | str, predicted: stream.Score | str) -> float:
    """Pitch-token Symbol Error Rate (edit distance / reference length)."""
    ref = [e.midi for e in score_to_events(reference)]
    pred = [e.midi for e in score_to_events(predicted)]
    if not ref:
        return 0.0 if not pred else 1.0
    distance, _ = _align(ref, pred)
    return distance / len(ref)


def compare_scores(
    reference: stream.Score | str,
    predicted: stream.Score | str,
    onset_tolerance: float = 0.25,
) -> ScoreComparison:
    """Compare a predicted score against a reference and return all metrics.

    Args:
        reference: Ground-truth score or MusicXML path.
        predicted: Recognised score or MusicXML path.
        onset_tolerance: Onset match window in quarter lengths.
    """
    ref_events = score_to_events(reference)
    pred_events = score_to_events(predicted)
    ref_pitches = [e.midi for e in ref_events]
    pred_pitches = [e.midi for e in pred_events]

    n_ref, n_pred = len(ref_events), len(pred_events)
    distance, matches = _align(ref_pitches, pred_pitches)
    n_matched = len(matches)

    precision = n_matched / n_pred if n_pred else (1.0 if n_ref == 0 else 0.0)
    recall = n_matched / n_ref if n_ref else (1.0 if n_pred == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    ser = distance / n_ref if n_ref else (0.0 if n_pred == 0 else 1.0)

    # Duration / onset accuracy on matched (same-pitch) pairs.
    dur_ok = onset_ok = 0
    for i, j in matches:
        if abs(ref_events[i].duration - pred_events[j].duration) < 1e-3:
            dur_ok += 1
        if abs(ref_events[i].onset - pred_events[j].onset) <= onset_tolerance:
            onset_ok += 1
    duration_accuracy = dur_ok / n_matched if n_matched else 0.0
    onset_accuracy = onset_ok / n_matched if n_matched else 0.0

    # MV2H-lite: pitch F1 dominates, refined by value (duration) correctness.
    mv2h_lite = f1 * (0.6 + 0.25 * duration_accuracy + 0.15 * onset_accuracy)

    return ScoreComparison(
        precision=precision,
        recall=recall,
        f1=f1,
        symbol_error_rate=ser,
        duration_accuracy=duration_accuracy,
        onset_accuracy=onset_accuracy,
        mv2h_lite=min(1.0, mv2h_lite),
        n_reference=n_ref,
        n_predicted=n_pred,
        n_matched=n_matched,
    )
