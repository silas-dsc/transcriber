"""End-to-end orchestration: sheet-music image / PDF -> MusicXML score.

Wires together page rendering, engine selection (single or ensemble),
recognition, and MusicXML post-processing -- the optical mirror of
:func:`transcriber.transcribe`.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from music21 import stream

from . import engines as engines_mod
from .confidence import ConfidenceReport, annotate_review, build_confidence
from .ensemble import recognize_ensemble
from .postprocess import repair_score, score_confidence
from .primitive import PrimitiveConfig
from .rendering import load_pages
from .semantic import SemanticReport
from .semantic import validate as semantic_validate

logger = logging.getLogger(__name__)


@dataclass
class OMRConfig:
    """Configuration for :func:`recognize`.

    Attributes:
        engine: ``"auto"``, ``"oemer"``, ``"homr"``, ``"audiveris"``,
            ``"primitive"`` or ``"ensemble"`` (run all available, keep best).
        dpi: Render resolution for PDF inputs.
        time_signature: Time signature assumed when barring the music.
        clefs: Optional per-staff clef names (1-based) for the primitive engine.
        title: Score title (defaults to the input file stem).
        primitive_config: Fine-grained settings for the built-in recogniser.
        semantic_check: Run musical sanity checks and safe repairs (duplicate
            merge, key-consistent re-spelling) on the recognised score.
        semantic_aggressive: Also apply risky semantic repairs (octave-outlier
            correction).  When ``False`` such issues are flagged, not changed.
        llm_review: Cross-check the recognised notes with Claude and apply
            validated, conservative corrections (octave / duration / duplicate).
            Off by default; requires the ``anthropic`` package and an API key.
        llm_model: Claude model id used when ``llm_review`` is enabled.
        review: Compute a confidence report + human-review queue (fuses per-note
            confidence, semantic checks and multi-engine disagreement).
        review_threshold: Measures below this confidence are flagged for review.
        mark_review: Annotate the output MusicXML with review markers/colours.
    """

    engine: str = "auto"
    dpi: int = 300
    time_signature: str = "4/4"
    clefs: dict[int, str] | None = None
    title: str = "Optical transcription"
    primitive_config: PrimitiveConfig | None = None
    semantic_check: bool = True
    semantic_aggressive: bool = False
    llm_review: bool = False
    llm_model: str = "claude-opus-4-8"
    review: bool = True
    review_threshold: float = 0.85
    mark_review: bool = False


@dataclass
class OMRResult:
    """Outcome of a recognition run.

    Attributes:
        score: The assembled music21 score.
        output_path: Path to the written MusicXML file (if written).
        engine: Name of the engine whose result was used.
        page_count: Number of pages processed.
        confidence: Confidence of the selected result in ``[0, 1]``.
        engine_confidences: Per-engine confidence (populated for ensembles).
        semantic_report: Musical sanity-check report, if semantic checks ran.
        confidence_report: Per-measure confidence + human-review queue.
    """

    score: stream.Score
    output_path: str | None
    engine: str
    page_count: int
    confidence: float
    engine_confidences: dict[str, float] = field(default_factory=dict)
    semantic_report: SemanticReport | None = None
    confidence_report: ConfidenceReport | None = None


def recognize(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: OMRConfig | None = None,
) -> OMRResult:
    """Recognise sheet music into a MusicXML score.

    Args:
        input_path: A PDF or image of sheet music.
        output_path: Where to write MusicXML.  If ``None`` only the in-memory
            score is returned.
        config: Optional :class:`OMRConfig`.

    Returns:
        An :class:`OMRResult`.
    """
    config = config or OMRConfig()
    input_path = Path(input_path)

    pages = load_pages(input_path, dpi=config.dpi)
    logger.info("Loaded %d page(s) from %s", len(pages), input_path)

    kwargs = dict(
        title=config.title,
        time_signature=config.time_signature,
        primitive_config=config.primitive_config,
    )

    with tempfile.TemporaryDirectory(prefix="omr_") as tmp:
        workdir = Path(tmp)
        page_paths = _write_page_pngs(pages, workdir)

        other_engine_scores: dict[str, stream.Score] = {}
        if config.engine == "ensemble":
            available = [engines_mod.get_engine(n) for n in engines_mod.available_engines()]
            score, engine_name, outcomes = recognize_ensemble(
                available, page_paths, pages, workdir, **kwargs
            )
            engine_confidences = {o.name: o.confidence for o in outcomes}
            confidence = engine_confidences.get(engine_name, 0.0)
            other_engine_scores = {
                o.name: o.score for o in outcomes if o.name != engine_name and o.score is not None
            }
        else:
            engine = engines_mod.get_engine(config.engine)
            score = engine.recognize(page_paths, pages, workdir, **kwargs)
            engine_name = engine.name
            confidence = score_confidence(score)
            engine_confidences = {engine_name: confidence}

    score = repair_score(score)

    semantic_report = None
    if config.semantic_check:
        score, semantic_report = semantic_validate(
            score, repair=True, aggressive=config.semantic_aggressive
        )

    if config.llm_review:
        from .llm_review import review_score

        key_hint = semantic_report.key if semantic_report else None
        score, _ = review_score(
            score,
            key_hint=key_hint,
            time_signature=config.time_signature,
            model=config.llm_model,
        )

    confidence_report = None
    if config.review:
        # Include the final score (as the primary) plus the other engines' raw
        # transcriptions, so disagreement is measured against the returned score.
        engine_scores = {**other_engine_scores, engine_name: score}
        confidence_report = build_confidence(
            score, semantic_report, engine_scores, review_threshold=config.review_threshold
        )
        if config.mark_review:
            annotate_review(score, confidence_report)

    written: str | None = None
    if output_path is not None:
        written = str(score.write("musicxml", fp=str(output_path)))
        logger.info("Wrote MusicXML to %s", written)

    return OMRResult(
        score=score,
        output_path=written,
        engine=engine_name,
        page_count=len(pages),
        confidence=confidence,
        engine_confidences=engine_confidences,
        semantic_report=semantic_report,
        confidence_report=confidence_report,
    )


def _write_page_pngs(pages, workdir: Path) -> list[Path]:
    """Persist page bitmaps as PNGs so external CLI engines can read them."""
    paths: list[Path] = []
    for i, page in enumerate(pages):
        arr = (page * 255).clip(0, 255).astype("uint8")
        path = workdir / f"page_{i:03d}.png"
        Image.fromarray(arr, mode="L").save(path)
        paths.append(path)
    return paths
