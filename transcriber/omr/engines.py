"""OMR engine back-ends with graceful fallback.

Every engine implements :class:`OMREngine`: a name, an availability probe, and
a ``recognize`` method that turns a list of page-image file paths into a
:class:`music21.stream.Score`.  This mirrors the audio pipeline's
"auto backend with fallback" pattern.

Order of preference for ``auto``:

1. :class:`OemerEngine`  -- deep-learning OMR (``pip install oemer``).
2. :class:`HomrEngine`   -- transformer OMR (``pip install homr``).
3. :class:`AudiverisEngine` -- the Java engine, if ``audiveris`` is on ``PATH``.
4. :class:`PrimitiveEngine` -- always available, pure numpy/scipy/Pillow.

The deep-learning engines emit MusicXML which we parse into a music21 score;
the primitive engine builds the score directly.  Multi-page inputs are
recognised page-by-page and concatenated.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from music21 import converter, stream

from .assemble import build_score
from .postprocess import merge_page_scores
from .primitive import PrimitiveConfig, recognize_image

logger = logging.getLogger(__name__)

# Generous per-page timeout for the external engines (model load + inference).
_ENGINE_TIMEOUT_S = 600


class OMREngine(ABC):
    """Common interface for an OMR back-end."""

    name: str = "base"

    @abstractmethod
    def available(self) -> bool:
        """True if this engine can run in the current environment."""

    @abstractmethod
    def recognize(
        self,
        page_paths: list[Path],
        page_images: list[np.ndarray],
        workdir: Path,
        **kwargs,
    ) -> stream.Score:
        """Recognise pages into a single concatenated score."""


def get_engine(name: str) -> OMREngine:
    """Return an engine by name, resolving ``"auto"`` to the best available."""
    if name == "auto":
        return _select_auto()
    engine = _ENGINES.get(name)
    if engine is None:
        raise ValueError(
            f"Unknown OMR engine {name!r}. Choices: {', '.join(['auto', *_ENGINES])}."
        )
    return engine()


def _select_auto() -> OMREngine:
    for cls in (OemerEngine, HomrEngine, AudiverisEngine):
        engine = cls()
        if engine.available():
            logger.info("Auto-selected OMR engine: %s", engine.name)
            return engine
    logger.info("No external OMR engine available; using built-in recogniser")
    return PrimitiveEngine()


# --------------------------------------------------------------------------- #
# Deep-learning engines (subprocess CLIs that emit MusicXML)
# --------------------------------------------------------------------------- #
class _SubprocessEngine(OMREngine):
    """Shared machinery for engines invoked as a CLI that writes MusicXML."""

    executable: str = ""
    module: str = ""

    def available(self) -> bool:
        if self.executable and shutil.which(self.executable):
            return True
        return bool(self.module) and importlib.util.find_spec(self.module) is not None

    def _run(self, cmd: list[str], cwd: Path) -> None:
        logger.info("Running: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
                timeout=_ENGINE_TIMEOUT_S,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                f"{self.name} failed (exit {exc.returncode}): {exc.stderr.strip()[:500]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - env dependent
            raise RuntimeError(f"{self.name} timed out after {_ENGINE_TIMEOUT_S}s") from exc

    def _command(self, image: Path, outdir: Path) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def _find_output(self, image: Path, outdir: Path) -> Path | None:  # pragma: no cover
        candidates = sorted([*outdir.glob("*.musicxml"), *outdir.glob("*.mxl"), *outdir.glob("*.xml")])
        return candidates[0] if candidates else None

    def recognize(self, page_paths, page_images, workdir, **kwargs) -> stream.Score:
        page_scores: list[stream.Score] = []
        for i, image in enumerate(page_paths):
            outdir = workdir / f"{self.name}_page{i}"
            outdir.mkdir(parents=True, exist_ok=True)
            self._run(self._command(image, outdir), cwd=outdir)
            out = self._find_output(image, outdir)
            if out is None:  # pragma: no cover - env dependent
                raise RuntimeError(f"{self.name} produced no MusicXML for {image.name}")
            page_scores.append(converter.parse(str(out)))
        return merge_page_scores(page_scores)


class OemerEngine(_SubprocessEngine):
    """`oemer <https://github.com/BreezeWhite/oemer>`_ -- end-to-end DL OMR."""

    name = "oemer"
    executable = "oemer"
    module = "oemer"

    def _command(self, image: Path, outdir: Path) -> list[str]:
        # `oemer IMAGE -o OUTDIR` writes <stem>.musicxml into OUTDIR.
        if shutil.which("oemer"):
            return ["oemer", str(image), "-o", str(outdir)]
        return ["python", "-m", "oemer", str(image), "-o", str(outdir)]


class HomrEngine(_SubprocessEngine):
    """`homr <https://github.com/liebharc/homr>`_ -- transformer polyphonic OMR."""

    name = "homr"
    executable = "homr"
    module = "homr"

    def _command(self, image: Path, outdir: Path) -> list[str]:
        # homr writes <image>.musicxml next to the input; copy the input in.
        local = outdir / image.name
        if not local.exists():
            shutil.copy(image, local)
        if shutil.which("homr"):
            return ["homr", str(local)]
        return ["python", "-m", "homr", str(local)]


class AudiverisEngine(_SubprocessEngine):
    """`Audiveris <https://github.com/Audiveris/audiveris>`_ -- mature Java OMR."""

    name = "audiveris"
    executable = "audiveris"
    module = ""

    def available(self) -> bool:
        return shutil.which("audiveris") is not None

    def _command(self, image: Path, outdir: Path) -> list[str]:
        return [
            "audiveris",
            "-batch",
            "-export",
            "-output",
            str(outdir),
            "--",
            str(image),
        ]


# --------------------------------------------------------------------------- #
# Built-in fallback
# --------------------------------------------------------------------------- #
class PrimitiveEngine(OMREngine):
    """Always-available classical-CV recogniser (:mod:`primitive`)."""

    name = "primitive"

    def available(self) -> bool:
        return True

    def recognize(self, page_paths, page_images, workdir, **kwargs) -> stream.Score:
        primitive_config: PrimitiveConfig | None = kwargs.get("primitive_config")
        title = kwargs.get("title", "Optical transcription")
        time_signature = kwargs.get("time_signature", "4/4")

        page_scores: list[stream.Score] = []
        for image in page_images:
            recognized = recognize_image(image, primitive_config)
            recognized.title = title
            page_scores.append(build_score(recognized, title=title, time_signature=time_signature))
        return merge_page_scores(page_scores)


_ENGINES: dict[str, type[OMREngine]] = {
    "oemer": OemerEngine,
    "homr": HomrEngine,
    "audiveris": AudiverisEngine,
    "primitive": PrimitiveEngine,
}


def available_engines() -> list[str]:
    """Names of engines that can currently run, best-first."""
    out = []
    for key in ("oemer", "homr", "audiveris", "primitive"):
        if _ENGINES[key]().available():
            out.append(key)
    return out
