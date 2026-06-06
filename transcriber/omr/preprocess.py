"""Image pre-processing that lifts OMR accuracy regardless of the engine.

Borrowed from document-image analysis (the "techniques from other fields"
brief): adaptive binarisation, deskew by projection-profile maximisation, and
speckle denoising.  Clean, level, high-contrast input is the single biggest
lever on downstream recognition quality.

All functions are pure numpy/scipy so they run with zero heavy dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Knobs for :func:`preprocess`.

    Attributes:
        binarize: ``"sauvola"`` (adaptive, best for uneven lighting / photos),
            ``"otsu"`` (global, fast, best for clean scans) or ``"none"``.
        deskew: Whether to estimate and correct small rotations.
        max_skew_deg: Search range for deskew (degrees, +/-).
        prefilter: Median-filter the grayscale before binarising to suppress
            sensor / salt-and-pepper noise (preserves thin staff lines).
        denoise: Whether to drop tiny connected components (speckles).
        min_component_area: Components smaller than this (pixels) are removed.
    """

    binarize: str = "sauvola"
    deskew: bool = True
    max_skew_deg: float = 8.0
    prefilter: bool = True
    denoise: bool = True
    min_component_area: int = 6


def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Turn a grayscale page into a clean boolean ink mask (``True`` = ink).

    Args:
        image: Grayscale ``float32`` array in ``[0, 1]`` (0 = black).
        config: Optional :class:`PreprocessConfig`.

    Returns:
        A boolean array (same shape) where ``True`` marks foreground ink.
    """
    config = config or PreprocessConfig()

    if config.prefilter:
        image = _denoise_gray(image)

    if config.deskew:
        angle = estimate_skew(image, max_deg=config.max_skew_deg)
        if abs(angle) > 0.05:
            image = _rotate_gray(image, angle)
            logger.info("Deskewed by %.2f degrees", angle)

    mask = binarize(image, method=config.binarize)

    if config.denoise:
        mask = _drop_small_components(mask, config.min_component_area)

    return mask


def _denoise_gray(image: np.ndarray) -> np.ndarray:
    """Suppress noise in the grayscale image without harming thin staff lines.

    * **Salt-and-pepper**: an *adaptive* median replaces only pixels that are
      extreme outliers vs their 3x3 median; staff lines (which match their
      neighbourhood) and clean images are left untouched.
    Crucially this is *gated*: it only acts when the fraction of outlier pixels
    indicates real noise.  Clean line art has a few naturally-isolated pixels
    (thin stems, line ends) that should NOT be touched -- filtering them nicks
    staff lines and breaks detection -- so on clean input the image passes
    through unchanged.
    """
    med = ndimage.median_filter(image, size=3)
    extreme = np.abs(image - med) > 0.4  # salt-and-pepper pixels
    # Clean renders have ~0.15% naturally-isolated pixels (thin stems, line
    # ends); salt-and-pepper has many times that.  Only act when clearly above
    # that baseline, and replace *only* the extreme pixels -- so clean input and
    # staff lines are untouched.  Moderate Gaussian noise produces few extreme
    # pixels, so it does not trigger here and is left to Sauvola thresholding
    # (which handles it well; blurring would nick the thin staff lines).
    # Clean scores -- even those full of thin sharp/flat strokes -- stay well
    # under this threshold (~0.003); salt-and-pepper at 2%+ is several times it.
    if extreme.mean() < 0.006:
        return image
    return np.where(extreme, med, image)


def binarize(image: np.ndarray, method: str = "sauvola") -> np.ndarray:
    """Binarise a grayscale image into an ink mask (``True`` = ink)."""
    if method == "none":
        return image < 0.5
    if method == "otsu":
        thresh = _otsu_threshold(image)
        return image < thresh
    if method == "sauvola":
        return _sauvola(image)
    raise ValueError(f"Unknown binarize method: {method!r}")


def _otsu_threshold(image: np.ndarray, bins: int = 256) -> float:
    """Classic Otsu global threshold on a ``[0, 1]`` image."""
    hist, edges = np.histogram(image, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cumsum_mean = np.cumsum(hist * centers)
    total_mean = cumsum_mean[-1]
    # Guard against divide-by-zero at the extremes.
    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.where(weight_bg > 0, cumsum_mean / np.maximum(weight_bg, 1), 0)
    mean_fg = np.where(
        weight_fg > 0, (total_mean - cumsum_mean) / np.maximum(weight_fg, 1), 0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = 0
    return float(centers[int(np.argmax(between))])


def _sauvola(image: np.ndarray, window: int = 31, k: float = 0.2, r: float = 0.5) -> np.ndarray:
    """Sauvola adaptive thresholding.

    ``T(x,y) = mean * (1 + k * (std / r - 1))``.  Robust to uneven illumination
    such as phone photos and aged scans.  Implemented with uniform filters so
    it is O(N) regardless of window size.
    """
    window = max(3, window | 1)  # force odd
    mean = ndimage.uniform_filter(image, size=window, mode="reflect")
    mean_sq = ndimage.uniform_filter(image * image, size=window, mode="reflect")
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
    thresh = mean * (1 + k * (std / r - 1))
    return image < thresh


def estimate_skew(image: np.ndarray, max_deg: float = 8.0, step: float = 0.5) -> float:
    """Estimate page skew (degrees) from the horizontal projection profile.

    Sheet music is dominated by long horizontal staff lines.  When the page is
    level those lines collapse into a few rows, so the row-projection profile
    has sharp peaks and a large *gradient energy* (sum of squared row-to-row
    differences).  Skewing smears the lines across rows and flattens the
    profile.  We therefore pick the rotation that maximises gradient energy --
    a much sharper, more reliable criterion than raw projection variance.
    """
    # Work on a lightly downscaled mask for speed; linear interpolation during
    # rotation avoids the aliasing that makes thin lines "stack" at odd angles.
    small = image[::2, ::2]
    mask = (small < _otsu_threshold(small)).astype(np.float32)
    if mask.sum() == 0:
        return 0.0

    def score_at(angle: float) -> float:
        rotated = ndimage.rotate(mask, angle, reshape=False, order=1, mode="constant")
        diff = np.diff(rotated.sum(axis=1))
        return float(np.dot(diff, diff))

    # Coarse search over the full range, then refine around the best angle.  On
    # very wide staves a residual of even 0.25 deg drifts note heads by several
    # pixels across the page, so sub-0.5-deg precision matters.
    coarse = np.arange(-max_deg, max_deg + step, step)
    best_angle = max(coarse, key=score_at)
    fine = np.arange(best_angle - step, best_angle + step + 1e-9, 0.1)
    best_angle = max(fine, key=score_at)
    return float(best_angle)


def _rotate_gray(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a grayscale image, filling new corners with white (1.0)."""
    return ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=1.0)


def _drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected ink components smaller than ``min_area`` pixels."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_area])  # label 0 = background
    return keep[labels]
