"""Optional LLM-based sense-check of a recognised score (Claude).

**Where an LLM helps, and where it does not.** Current models are *not*
reliable at reading exact pitches off sheet-music pixels -- that is the OMR
engine's job, and asking a model to "re-read the image" tends to hallucinate
notes.  What an LLM *is* good at is judging the **musical plausibility** of the
*symbolic* result the engine already produced: spotting a lone note an octave
from its neighbours, a measure that does not add up, a phrase ending that
contradicts the key.  So this module treats Claude as a careful reviewer of the
note sequence, not as a recogniser.

Safety model: the reviewer may only propose a fixed, closed set of
*conservative* edits to notes that **already exist** (octave shifts, duration
fixes, deleting an exact duplicate).  It can never add notes.  Every proposed
edit is validated against the score before being applied, so a bad suggestion
is dropped rather than trusted.

This is optional and off by default.  It requires the ``anthropic`` package
and an API key (``pip install -e ".[llm]"`` and ``ANTHROPIC_API_KEY``).  For
testing and for callers that already hold a client, a client can be injected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from music21 import interval, stream

logger = logging.getLogger(__name__)

# Default to the most capable model; callers can override for cost/latency.
DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM_PROMPT = """You are an expert musician reviewing the OUTPUT of an \
optical music recognition (OMR) system. You are given the recognised note \
sequence as text -- you CANNOT see the original image, and you must NOT try to \
guess what the image said. Your only job is to flag notes whose pitch or \
duration is musically implausible *given the surrounding notes and the key*, \
and which are therefore likely OMR errors.

Typical OMR errors you can catch:
- a single note displaced by exactly one octave (misread ledger line / clef),
  recognisable as a large leap immediately reversed;
- an exact duplicate of the previous note at the same position (double \
detection);
- a duration that makes a measure not add up.

Rules:
- ONLY propose edits to notes that already exist. NEVER invent or add notes.
- Be conservative: if a note is unusual but musically defensible, leave it.
- Allowed actions: "octave_up" (+12), "octave_down" (-12), "delete" (remove a \
duplicate/spurious note), "set_duration" (provide quarter_length).
- Refer to notes by their integer index in the provided list."""

# Strict JSON schema for the model's response (structured outputs).
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "action": {
                        "type": "string",
                        "enum": ["octave_up", "octave_down", "delete", "set_duration"],
                    },
                    "quarter_length": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "action", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["corrections"],
    "additionalProperties": False,
}


@dataclass
class LLMReviewReport:
    """Outcome of an LLM review."""

    corrections: list[dict] = field(default_factory=list)
    applied: int = 0
    skipped: int = 0
    model: str = DEFAULT_MODEL
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"llm review: skipped ({self.error})"
        return f"llm review ({self.model}): {self.applied} applied, {self.skipped} rejected"


def review_score(
    score: stream.Score,
    *,
    key_hint: str | None = None,
    time_signature: str = "4/4",
    apply: bool = True,
    model: str = DEFAULT_MODEL,
    client=None,
    max_corrections: int = 16,
) -> tuple[stream.Score, LLMReviewReport]:
    """Have Claude review the recognised notes and optionally apply safe fixes.

    Args:
        score: The recognised score (modified in place when ``apply``).
        key_hint: Inferred key name to give the reviewer context.
        time_signature: Time signature for measure-arithmetic context.
        apply: Apply validated corrections (else just report them).
        model: Claude model id.
        client: An ``anthropic.Anthropic``-compatible client.  If ``None`` one
            is constructed from the environment; injectable for testing.
        max_corrections: Ignore responses proposing more edits than this (a
            runaway response is a red flag, not a fix).

    Returns:
        ``(score, report)``.
    """
    notes = [n for n in score.flatten().notes if not n.isChord]
    if not notes:
        return score, LLMReviewReport(model=model, error="no notes")

    if client is None:
        client = _make_client()
        if client is None:
            return score, LLMReviewReport(model=model, error="anthropic SDK / API key unavailable")

    prompt = _build_prompt(notes, key_hint=key_hint, time_signature=time_signature)
    try:
        raw = _call_claude(client, model, prompt)
        corrections = json.loads(raw).get("corrections", [])
    except Exception as exc:  # pragma: no cover - network/SDK dependent
        logger.warning("LLM review failed: %s", exc)
        return score, LLMReviewReport(model=model, error=str(exc))

    report = LLMReviewReport(corrections=corrections, model=model)
    if len(corrections) > max_corrections:
        report.error = f"too many corrections ({len(corrections)})"
        return score, report

    for corr in corrections:
        if _apply_correction(notes, corr, apply=apply):
            report.applied += 1
        else:
            report.skipped += 1

    logger.info(report.summary())
    return score, report


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _make_client():
    try:
        import anthropic
    except ImportError:
        return None
    try:
        return anthropic.Anthropic()
    except Exception:  # pragma: no cover - missing credentials
        return None


def _build_prompt(notes, key_hint: str | None, time_signature: str) -> str:
    listing = "\n".join(
        f"{i}: {n.pitch.nameWithOctave} (midi {n.pitch.midi}) "
        f"dur={float(n.quarterLength):g} at beat {float(n.offset):g}"
        for i, n in enumerate(notes)
    )
    header = f"Key (estimated): {key_hint or 'unknown'}. Time signature: {time_signature}."
    return (
        f"{header}\n\nRecognised notes ({len(notes)} total):\n{listing}\n\n"
        "Return the corrections you are confident about."
    )


def _call_claude(client, model: str, prompt: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _RESPONSE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


def _apply_correction(notes, corr: dict, apply: bool) -> bool:
    """Validate a single correction and (optionally) apply it. Returns success."""
    try:
        index = int(corr["index"])
        action = str(corr["action"])
    except (KeyError, TypeError, ValueError):
        return False
    if not 0 <= index < len(notes):
        return False
    note_obj = notes[index]

    if action in ("octave_up", "octave_down"):
        if not apply:
            return True
        semitones = 12 if action == "octave_up" else -12
        note_obj.transpose(interval.Interval(semitones), inPlace=True)
        return True

    if action == "set_duration":
        ql = corr.get("quarter_length")
        if not isinstance(ql, (int, float)) or not 0 < ql <= 16:
            return False
        if apply:
            note_obj.quarterLength = float(ql)
        return True

    if action == "delete":
        if apply:
            site = note_obj.activeSite
            if site is None:
                return False
            site.remove(note_obj)
        return True

    return False
