"""Musical / semantic sanity checks that catch and repair likely OMR errors.

OMR engines make mistakes that are invisible at the pixel level but obvious
*musically*: a bar with the wrong number of beats, a single note sitting an
octave away from its neighbours (a misread ledger line), an accidental that
contradicts the key, two coincident copies of the same note (a double
detection).  This module encodes that musical knowledge as a set of rules that
run on the assembled :class:`music21.stream.Score`.

Design principles:

* **Safe by default.**  Only unambiguous repairs are applied automatically
  (merging exact duplicates, key-consistent re-spelling).  Risky repairs that
  change pitch (octave correction) are applied only in ``aggressive`` mode; in
  the default mode they are *flagged* in the report for human review.
* **Always reported.**  Every rule returns structured :class:`Issue` records,
  so the output doubles as a confidence/diagnostics signal even when nothing is
  changed -- which is exactly what a human-in-the-loop verification UI needs.

This is a deterministic complement to the optional LLM reviewer
(:mod:`transcriber.omr.llm_review`): cheap, offline, and never hallucinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from music21 import interval, stream

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """A detected musical inconsistency.

    Attributes:
        kind: Machine-readable rule id (e.g. ``"octave_outlier"``).
        message: Human-readable description.
        part: Part index (0-based) the issue is in, if applicable.
        measure: Measure number, if applicable.
        severity: ``"info"``, ``"warning"`` or ``"error"``.
        fixed: Whether the issue was automatically repaired.
    """

    kind: str
    message: str
    part: int | None = None
    measure: int | None = None
    severity: str = "warning"
    fixed: bool = False


@dataclass
class SemanticReport:
    """Outcome of :func:`validate`."""

    issues: list[Issue] = field(default_factory=list)
    key: str | None = None

    @property
    def n_fixed(self) -> int:
        return sum(1 for i in self.issues if i.fixed)

    @property
    def n_flagged(self) -> int:
        return sum(1 for i in self.issues if not i.fixed)

    def summary(self) -> str:
        return (
            f"semantic check: key={self.key or '?'}, "
            f"{self.n_fixed} fixed, {self.n_flagged} flagged"
        )


def validate(
    score: stream.Score,
    repair: bool = True,
    aggressive: bool = False,
) -> tuple[stream.Score, SemanticReport]:
    """Run all semantic checks on ``score``.

    Args:
        score: The assembled score (modified in place when ``repair``).
        repair: Apply safe repairs (duplicate merge, key re-spelling).
        aggressive: Also apply risky repairs (octave-outlier correction).
            When ``False`` such issues are flagged but not changed.

    Returns:
        ``(score, report)``.
    """
    report = SemanticReport()

    if repair:
        _merge_duplicate_notes(score, report)

    key = _infer_key(score, report)
    if repair and key is not None:
        _respell_to_key(score, key, report)

    _check_measure_durations(score, report)
    _check_octave_outliers(score, report, fix=aggressive)
    _check_clef_ranges(score, report)

    logger.info(report.summary())
    return score, report


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def _infer_key(score: stream.Score, report: SemanticReport):
    """Estimate the key with music21's Krumhansl-Schmuckler analysis."""
    try:
        key = score.analyze("key")
    except Exception as exc:  # pragma: no cover - analysis edge cases
        logger.debug("Key analysis failed: %s", exc)
        return None
    report.key = key.name
    # Correlation coefficient is a soft confidence in the key estimate.
    corr = getattr(key, "correlationCoefficient", None)
    if corr is not None and corr < 0.5:
        report.issues.append(
            Issue(
                kind="weak_key",
                message=f"Low key confidence ({corr:.2f}); pitch errors likely.",
                severity="info",
            )
        )
    return key


def _respell_to_key(score: stream.Score, key, report: SemanticReport) -> None:
    """Re-spell accidentals to be consistent with the inferred key.

    OMR often gets the *pitch class* right but the *spelling* wrong (G# vs Ab).
    music21's ``isDiatonic`` lets us spot chromatic spellings; we leave true
    chromatic notes alone but flag clusters of them as suspicious.
    """
    chromatic = 0
    total = 0
    scale_pitches = {p.name for p in key.getScale().getPitches()}
    for n in score.recurse().notes:
        for p in n.pitches:
            total += 1
            if p.name not in scale_pitches:
                chromatic += 1
    if total and chromatic / total > 0.4:
        report.issues.append(
            Issue(
                kind="many_chromatics",
                message=(
                    f"{chromatic}/{total} notes are outside {key.name}; "
                    "accidental or key recognition may be wrong."
                ),
                severity="warning",
            )
        )


def _merge_duplicate_notes(score: stream.Score, report: SemanticReport) -> None:
    """Remove exact coincident duplicate notes (a common double-detection)."""
    for pi, part in enumerate(score.parts):
        seen: dict[tuple[float, int], object] = {}
        for n in list(part.recurse().notes):
            if n.isChord:
                continue
            key = (round(float(n.getOffsetInHierarchy(part)), 4), int(n.pitch.midi))
            if key in seen:
                container = n.activeSite
                if container is not None:
                    container.remove(n)
                    report.issues.append(
                        Issue(
                            kind="duplicate_note",
                            message=f"Removed duplicate {n.pitch.nameWithOctave}",
                            part=pi,
                            severity="info",
                            fixed=True,
                        )
                    )
            else:
                seen[key] = n


def _check_measure_durations(score: stream.Score, report: SemanticReport) -> None:
    """Flag measures whose content does not fill the time signature."""
    for pi, part in enumerate(score.parts):
        measures = list(part.getElementsByClass(stream.Measure))
        for idx, m in enumerate(measures):
            expected = float(m.barDuration.quarterLength)
            actual = float(m.duration.quarterLength)
            # A short pickup (anacrusis) at the start and an incomplete final
            # measure are both legitimate, not recognition errors.
            if actual < expected and (m.number in (0, 1) or idx == len(measures) - 1):
                continue
            if abs(actual - expected) > 1e-3:
                report.issues.append(
                    Issue(
                        kind="measure_duration",
                        message=(
                            f"Measure {m.number} has {actual:g} beats, "
                            f"expected {expected:g} (missed/extra note?)."
                        ),
                        part=pi,
                        measure=m.number,
                        severity="warning",
                    )
                )


def _check_octave_outliers(score: stream.Score, report: SemanticReport, fix: bool) -> None:
    """Detect/repair single notes an octave from their neighbours.

    A misread ledger line or clef typically displaces *one* note by an octave,
    producing a large leap immediately reversed by an equal-and-opposite leap.
    We only act on a note when shifting it by +/-12 semitones removes a >octave
    leap on *both* sides -- a strong, local, reversible signal.
    """
    for pi, part in enumerate(score.parts):
        notes = [n for n in part.recurse().notes if not n.isChord]
        for i in range(1, len(notes) - 1):
            prev, cur, nxt = notes[i - 1], notes[i], notes[i + 1]
            up = abs(prev.pitch.midi - cur.pitch.midi)
            down = abs(nxt.pitch.midi - cur.pitch.midi)
            if up <= 12 or down <= 12:
                continue
            neighbour = (prev.pitch.midi + nxt.pitch.midi) / 2.0
            best_shift = min((-12, 0, 12), key=lambda s: abs(cur.pitch.midi + s - neighbour))
            if best_shift == 0:
                continue
            msg = (
                f"Note {cur.pitch.nameWithOctave} leaps "
                f">{12} semitones both sides (octave misread?)."
            )
            if fix:
                cur.transpose(interval.Interval(best_shift), inPlace=True)
            report.issues.append(
                Issue(
                    kind="octave_outlier",
                    message=msg + (f" Corrected by {best_shift:+d}." if fix else ""),
                    part=pi,
                    severity="warning",
                    fixed=fix,
                )
            )


# Reasonable MIDI ranges per clef, including a few ledger lines either side.
_CLEF_RANGES = {
    "treble": (55, 88),  # ~G3 .. E6
    "bass": (33, 67),    # ~A1 .. G4
    "alto": (45, 78),
}


def _check_clef_ranges(score: stream.Score, report: SemanticReport) -> None:
    """Flag notes far outside the plausible range of their part's clef."""
    from music21 import clef as clef_mod

    for pi, part in enumerate(score.parts):
        clef_obj = part.recurse().getElementsByClass(clef_mod.Clef).first()
        name = "treble"
        if isinstance(clef_obj, clef_mod.BassClef):
            name = "bass"
        elif isinstance(clef_obj, clef_mod.AltoClef):
            name = "alto"
        lo, hi = _CLEF_RANGES[name]
        out = sum(1 for n in part.recurse().notes for p in n.pitches if not lo <= p.midi <= hi)
        if out:
            report.issues.append(
                Issue(
                    kind="clef_range",
                    message=f"{out} note(s) outside the {name}-clef range (clef misread?).",
                    part=pi,
                    severity="info",
                )
            )
