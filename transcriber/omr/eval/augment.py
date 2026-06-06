"""Image degradations for robustness testing.

The render -> recognise loop with a *clean* render never exercises the
pre-processing pipeline (deskew / binarise / denoise).  These augmentations
simulate the conditions of a real scan or phone photo -- rotation, perspective
warp, sensor noise, blur, uneven lighting -- so the harness can measure how
accuracy degrades and prove the pre-processing actually earns its keep.

All operations are pure numpy/scipy and act on a grayscale ``float32`` image in
``[0, 1]`` (1 = white), returning the same.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Named presets used by the harness / CLI.  Each maps to a kwargs dict for
# :func:`augment`.
PRESETS: dict[str, dict] = {
    "clean": {},
    "rotate": {"rotate_deg": 4.0},
    "rotate_hard": {"rotate_deg": 7.0},
    "noise": {"gaussian_sigma": 0.08, "salt_pepper": 0.02},
    "blur": {"blur_sigma": 1.0},
    "warp": {"warp": 0.0025},
    "lighting": {"gradient": 0.5},
    # A realistic "phone photo": mild skew + perspective + noise + blur + light.
    "photo": {
        "rotate_deg": 3.0,
        "warp": 0.0015,
        "gaussian_sigma": 0.05,
        "blur_sigma": 0.8,
        "gradient": 0.4,
    },
}


def augment(
    image: np.ndarray,
    *,
    rotate_deg: float = 0.0,
    warp: float = 0.0,
    gaussian_sigma: float = 0.0,
    salt_pepper: float = 0.0,
    blur_sigma: float = 0.0,
    gradient: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Apply a combination of degradations to ``image``.

    Args:
        rotate_deg: Rotation in degrees (simulates a skewed scan).
        warp: Perspective-warp strength (0 = none, ~0.002 = mild).
        gaussian_sigma: Std-dev of additive Gaussian sensor noise.
        salt_pepper: Fraction of pixels flipped to pure black/white.
        blur_sigma: Gaussian blur std-dev (out-of-focus / low resolution).
        gradient: Strength of a smooth brightness gradient (uneven lighting).
        seed: RNG seed for reproducible noise.
    """
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32)

    if warp:
        out = _perspective_warp(out, warp, rng)
    if rotate_deg:
        out = ndimage.rotate(out, rotate_deg, reshape=True, order=1, mode="constant", cval=1.0)
    if blur_sigma:
        out = ndimage.gaussian_filter(out, sigma=blur_sigma)
    if gradient:
        out = _lighting_gradient(out, gradient, rng)
    if gaussian_sigma:
        out = out + rng.normal(0.0, gaussian_sigma, size=out.shape).astype(np.float32)
    if salt_pepper:
        mask = rng.random(out.shape)
        out = np.where(mask < salt_pepper / 2, 0.0, out)
        out = np.where(mask > 1 - salt_pepper / 2, 1.0, out)

    return np.clip(out, 0.0, 1.0)


def augment_preset(image: np.ndarray, name: str, seed: int = 0) -> np.ndarray:
    """Apply a named preset from :data:`PRESETS`."""
    if name not in PRESETS:
        raise ValueError(f"Unknown augmentation preset {name!r}. Choices: {list(PRESETS)}")
    return augment(image, seed=seed, **PRESETS[name])


def _lighting_gradient(image: np.ndarray, strength: float, rng) -> np.ndarray:
    """Multiply by a smooth random linear brightness gradient."""
    h, w = image.shape
    gy, gx = np.mgrid[0:h, 0:w]
    ang = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * gx / w + np.sin(ang) * gy / h)
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) or 1.0)  # 0..1
    factor = 1.0 - strength * ramp  # darken toward one corner
    return image * factor


def _perspective_warp(image: np.ndarray, strength: float, rng) -> np.ndarray:
    """Apply a mild quadratic perspective-like warp via coordinate remapping."""
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Small x-shear that varies with y, and y-bow that varies with x.
    sx = strength * rng.uniform(-1, 1)
    sy = strength * rng.uniform(-1, 1)
    src_x = xx + sx * (yy - h / 2) * (xx - w / 2) / w
    src_y = yy + sy * (xx - w / 2) * (yy - h / 2) / h
    return ndimage.map_coordinates(
        image, [src_y, src_x], order=1, mode="constant", cval=1.0
    ).reshape(h, w)
