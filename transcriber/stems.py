"""Source separation into instrument stems.

Primary back-end is `Demucs <https://github.com/facebookresearch/demucs>`_
(``htdemucs``), which produces high quality ``drums``/``bass``/``vocals``/
``other`` stems.  When Demucs (and its torch dependency) are not installed we
fall back to ``librosa`` harmonic-percussive source separation (HPSS): the
percussive component becomes the ``drums`` stem and the harmonic component
becomes the ``other`` (pitched) stem.  HPSS is obviously far cruder than
Demucs but lets the full pipeline run with no machine-learning dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Canonical role names used throughout the pipeline.
DRUMS = "drums"
BASS = "bass"
VOCALS = "vocals"
OTHER = "other"

# Stems whose content is pitched (melodic/harmonic) and therefore handled by
# the pitch transcriber.  ``drums`` is handled separately by the drum module.
PITCHED_STEMS = (BASS, VOCALS, OTHER)


@dataclass
class Stem:
    """One separated source.

    Attributes:
        name: Role of the stem, e.g. ``"drums"`` or ``"vocals"``.
        samples: Mono ``float32`` audio for the stem.
        sr: Sample rate in Hz.
        is_percussive: Whether the stem should be transcribed as drums.
    """

    name: str
    samples: np.ndarray
    sr: int
    is_percussive: bool = False

    @property
    def is_silent(self, threshold: float = 1e-4) -> bool:
        """True if the stem carries essentially no energy."""
        return float(np.sqrt(np.mean(np.square(self.samples)))) < threshold


def separate(
    samples: np.ndarray,
    sr: int,
    backend: str = "auto",
    model: str = "htdemucs",
) -> list[Stem]:
    """Separate ``samples`` into a list of :class:`Stem`.

    Args:
        samples: Mono ``float32`` audio.
        sr: Sample rate.
        backend: ``"demucs"``, ``"hpss"`` or ``"auto"``.  ``"auto"`` uses
            Demucs when importable and otherwise falls back to HPSS.
        model: Demucs model name (ignored for HPSS).

    Returns:
        A list of stems, each marked as percussive or pitched.
    """
    if backend == "auto":
        backend = "demucs" if _demucs_available() else "hpss"

    if backend == "demucs":
        return _separate_demucs(samples, sr, model=model)
    if backend == "hpss":
        return _separate_hpss(samples, sr)
    raise ValueError(f"Unknown separation backend: {backend!r}")


def _demucs_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("demucs") is not None


def _separate_demucs(samples: np.ndarray, sr: int, model: str) -> list[Stem]:
    """Separate using Demucs.  Requires the ``[full]`` extras (torch+demucs)."""
    import torch  # noqa: F401  (imported for side effects / availability check)
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    logger.info("Separating with Demucs model %s", model)
    demucs_model = get_model(model)
    demucs_model.eval()
    model_sr = demucs_model.samplerate

    # Demucs expects shape (batch, channels, samples) at the model's sample
    # rate.  Feed it stereo by duplicating the mono signal.
    wav = _resample(samples, sr, model_sr)
    tensor = torch.from_numpy(np.stack([wav, wav])).unsqueeze(0).float()

    with torch.no_grad():
        sources = apply_model(demucs_model, tensor, split=True, overlap=0.25)[0]

    stems: list[Stem] = []
    for name, source in zip(demucs_model.sources, sources):
        mono = source.mean(dim=0).cpu().numpy().astype(np.float32)
        mono = _resample(mono, model_sr, sr)
        stems.append(Stem(name=name, samples=mono, sr=sr, is_percussive=(name == DRUMS)))
    return stems


def _separate_hpss(samples: np.ndarray, sr: int) -> list[Stem]:
    """Fallback separation using librosa harmonic-percussive separation."""
    import librosa

    logger.info("Separating with librosa HPSS fallback (install '[full]' for Demucs)")
    # ``margin`` > 1 gives a cleaner, more aggressive split between the
    # sustained (harmonic) and transient (percussive) components.
    harmonic, percussive = librosa.effects.hpss(samples, margin=(2.0, 2.0))
    return [
        Stem(name=DRUMS, samples=percussive.astype(np.float32), sr=sr, is_percussive=True),
        Stem(name=OTHER, samples=harmonic.astype(np.float32), sr=sr, is_percussive=False),
    ]


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    import librosa

    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
