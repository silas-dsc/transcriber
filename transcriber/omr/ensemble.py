"""Multi-engine ensembling.

Different OMR engines fail in different ways (oemer is strong on photos, homr
on dense polyphony, the primitive recogniser on clean monophonic scans).  When
more than one is available we can run several and keep the best result rather
than betting on a single engine.

Selection uses :func:`transcriber.omr.postprocess.score_confidence` when no
reference is available.  The harness can instead pass a scoring callback (e.g.
note-level F1 against a ground-truth score) to pick the empirically best
engine, which is how the system is tuned offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from music21 import stream

from .engines import OMREngine
from .postprocess import score_confidence

logger = logging.getLogger(__name__)


@dataclass
class EngineOutcome:
    """Result of running one engine within an ensemble."""

    name: str
    score: stream.Score | None
    confidence: float
    error: str | None = None


def recognize_ensemble(
    engines: list[OMREngine],
    page_paths: list[Path],
    page_images: list[np.ndarray],
    workdir: Path,
    scorer: Callable[[stream.Score], float] | None = None,
    **kwargs,
) -> tuple[stream.Score, str, list[EngineOutcome]]:
    """Run several engines and return the best score, its engine, and all outcomes.

    Args:
        engines: Engines to try (already filtered to available ones).
        page_paths / page_images: The pages to recognise.
        workdir: Scratch directory for engine outputs.
        scorer: Optional scoring function; defaults to confidence heuristic.

    Returns:
        ``(best_score, best_engine_name, outcomes)``.
    """
    scorer = scorer or score_confidence
    outcomes: list[EngineOutcome] = []

    for engine in engines:
        try:
            score = engine.recognize(page_paths, page_images, workdir, **kwargs)
            conf = float(scorer(score))
            outcomes.append(EngineOutcome(engine.name, score, conf))
            logger.info("Engine %s -> score %.3f", engine.name, conf)
        except Exception as exc:  # pragma: no cover - engine/env dependent
            logger.warning("Engine %s failed: %s", engine.name, exc)
            outcomes.append(EngineOutcome(engine.name, None, 0.0, error=str(exc)))

    valid = [o for o in outcomes if o.score is not None]
    if not valid:
        raise RuntimeError(
            "All OMR engines failed: "
            + "; ".join(f"{o.name}: {o.error}" for o in outcomes)
        )

    best = max(valid, key=lambda o: o.confidence)
    logger.info("Ensemble selected %s (%.3f)", best.name, best.confidence)
    return best.score, best.name, outcomes
