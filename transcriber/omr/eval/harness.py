"""Run an OMR engine over a corpus and report aggregate accuracy.

The loop per item is: render the reference score to an image, recognise that
image, and compare the recognition against the reference.  Because rendering
and comparison are self-contained, the whole benchmark runs offline with the
built-in renderer + primitive engine, and scales up to real engines/corpora by
swapping arguments.
"""

from __future__ import annotations

import logging
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..pipeline import OMRConfig, recognize
from .augment import augment_preset
from .datasets import CorpusItem
from .metrics import ScoreComparison, compare_scores
from .render_ref import render_reference

logger = logging.getLogger(__name__)


@dataclass
class HarnessResult:
    """Aggregate benchmark outcome."""

    per_item: list[tuple[str, ScoreComparison]] = field(default_factory=list)
    engine: str = "auto"

    @property
    def n_items(self) -> int:
        return len(self.per_item)

    def _mean(self, attr: str) -> float:
        vals = [getattr(c, attr) for _, c in self.per_item]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def mean_f1(self) -> float:
        return self._mean("f1")

    @property
    def mean_precision(self) -> float:
        return self._mean("precision")

    @property
    def mean_recall(self) -> float:
        return self._mean("recall")

    @property
    def mean_ser(self) -> float:
        return self._mean("symbol_error_rate")

    @property
    def mean_duration_accuracy(self) -> float:
        return self._mean("duration_accuracy")

    @property
    def mean_mv2h_lite(self) -> float:
        return self._mean("mv2h_lite")


def evaluate_corpus(
    items: list[CorpusItem],
    engine: str = "auto",
    renderer: str = "builtin",
    dpi: int = 300,
    time_signature: str = "4/4",
    augmentation: str = "clean",
) -> HarnessResult:
    """Render, recognise and score every item in ``items``.

    Args:
        augmentation: Named degradation preset applied to the rendered image
            before recognition (``"clean"`` = none).  Exercises the
            pre-processing pipeline against skew / warp / noise / blur.
    """
    result = HarnessResult(engine=engine)
    with tempfile.TemporaryDirectory(prefix="omr_eval_") as tmp:
        tmpdir = Path(tmp)
        for item in items:
            image = render_reference(item.score, tmpdir / f"{item.id}.png", renderer=renderer)
            if augmentation != "clean":
                image = _augment_file(image, augmentation)
            config = OMRConfig(engine=engine, dpi=dpi, time_signature=time_signature)
            recognized = recognize(image, output_path=None, config=config)
            comparison = compare_scores(item.score, recognized.score)
            result.per_item.append((item.id, comparison))
            logger.info(
                "%s: F1=%.3f SER=%.3f (ref=%d pred=%d)",
                item.id,
                comparison.f1,
                comparison.symbol_error_rate,
                comparison.n_reference,
                comparison.n_predicted,
            )
    return result


def _augment_file(image_path: Path, preset: str) -> Path:
    """Load, degrade, and re-save a rendered image; return the new path."""
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    degraded = augment_preset(arr, preset)
    out = image_path.with_name(f"{image_path.stem}_{preset}.png")
    Image.fromarray((degraded * 255).clip(0, 255).astype("uint8"), mode="L").save(out)
    return out


def format_report(result: HarnessResult) -> str:
    """Render a :class:`HarnessResult` as a human-readable text report."""
    lines = [
        f"OMR accuracy report  (engine={result.engine}, {result.n_items} items)",
        "=" * 60,
        f"{'item':<28}{'F1':>7}{'SER':>7}{'dur':>7}{'MV2H':>7}",
        "-" * 60,
    ]
    for item_id, c in result.per_item:
        name = item_id if len(item_id) <= 27 else item_id[:24] + "..."
        lines.append(
            f"{name:<28}{c.f1:>7.3f}{c.symbol_error_rate:>7.3f}"
            f"{c.duration_accuracy:>7.3f}{c.mv2h_lite:>7.3f}"
        )
    lines += [
        "-" * 60,
        f"{'MEAN':<28}{result.mean_f1:>7.3f}{result.mean_ser:>7.3f}"
        f"{result.mean_duration_accuracy:>7.3f}{result.mean_mv2h_lite:>7.3f}",
        "",
        f"Precision {result.mean_precision:.3f}   Recall {result.mean_recall:.3f}   "
        f"F1 {result.mean_f1:.3f}   SER {result.mean_ser:.3f}",
    ]
    return "\n".join(lines)
