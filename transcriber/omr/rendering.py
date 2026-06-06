"""Load the OMR *input* (PDF / image files) into page bitmaps.

The product use-case is "feed me a PDF or a photo".  We normalise both into a
list of grayscale ``float32`` page images in ``[0, 1]`` (0 = black, 1 = white)
that the rest of the pipeline consumes.

PDF rendering prefers `pypdfium2 <https://github.com/pypdfium2-team/pypdfium2>`_
because it ships self-contained wheels (no system Poppler/Ghostscript needed).
If that is unavailable we fall back to ``pdf2image`` (Poppler) and finally to a
clear error.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def load_pages(path: str | Path, dpi: int = 300) -> list[np.ndarray]:
    """Load ``path`` into a list of grayscale page images.

    Args:
        path: A PDF file or a single raster image.
        dpi: Render resolution for PDF pages.  300 DPI is a good default for
            OMR: high enough to resolve staff spacing, low enough to stay fast.

    Returns:
        One ``float32`` array per page, shape ``(H, W)``, values in ``[0, 1]``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OMR input not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, dpi=dpi)
    if suffix in IMAGE_SUFFIXES:
        return [_load_image(path)]
    raise ValueError(
        f"Unsupported OMR input '{suffix}'. Expected a PDF or image "
        f"({', '.join(sorted(IMAGE_SUFFIXES))})."
    )


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        gray = im.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    logger.info("Loaded image %s (%dx%d)", path.name, arr.shape[1], arr.shape[0])
    return arr


def _load_pdf(path: Path, dpi: int) -> list[np.ndarray]:
    pages = _load_pdf_pdfium(path, dpi) or _load_pdf_pdf2image(path, dpi)
    if pages is None:
        raise RuntimeError(
            "Could not render PDF: install 'pypdfium2' (recommended) or "
            "'pdf2image' + Poppler. Try: pip install pypdfium2"
        )
    logger.info("Rendered %d page(s) from %s at %d DPI", len(pages), path.name, dpi)
    return pages


def _load_pdf_pdfium(path: Path, dpi: int) -> list[np.ndarray] | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None

    scale = dpi / 72.0  # PDF user space is 72 DPI.
    pages: list[np.ndarray] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for page in pdf:
            bitmap = page.render(scale=scale, grayscale=True)
            pil = bitmap.to_pil().convert("L")
            pages.append(np.asarray(pil, dtype=np.float32) / 255.0)
            page.close()
    finally:
        pdf.close()
    return pages


def _load_pdf_pdf2image(path: Path, dpi: int) -> list[np.ndarray] | None:
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return None

    images = convert_from_path(str(path), dpi=dpi, grayscale=True)
    return [np.asarray(im.convert("L"), dtype=np.float32) / 255.0 for im in images]
