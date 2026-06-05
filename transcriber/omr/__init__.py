"""Optical Music Recognition: sheet-music image / PDF -> MusicXML score.

This is the *optical* counterpart to the top-level audio pipeline.  Where
:func:`transcriber.transcribe` turns a recording into a score, :func:`recognize`
turns a **picture of sheet music** (a scan, a phone photo, or a PDF) into the
same kind of :class:`music21.stream.Score` and MusicXML output.

It is built in the same spirit as the rest of the project: orchestrate the
best-in-class open-source OMR engines and fill the gaps between them.

Engines (best first, all optional):

* `oemer <https://github.com/BreezeWhite/oemer>`_ -- end-to-end deep-learning
  OMR (ONNX), robust to phone photos and skew.
* `homr <https://github.com/liebharc/homr>`_ -- transformer-based polyphonic
  OMR (Polyphonic-TrOMR) with per-staff dewarping.
* `Audiveris <https://github.com/Audiveris/audiveris>`_ -- the mature Java OMR
  engine, detected on ``PATH`` if installed.

When none of those are installed the pipeline transparently falls back to a
**built-in classical-computer-vision recogniser** (:mod:`transcriber.omr.primitive`)
that depends only on numpy/scipy/Pillow, so a score is *always* produced.

On top of the engines the pipeline adds cross-cutting improvements that lift
accuracy regardless of which engine runs:

* image pre-processing -- adaptive binarisation, deskew, denoise
  (:mod:`transcriber.omr.preprocess`);
* PDF / multi-page rendering (:mod:`transcriber.omr.rendering`);
* multi-engine ensembling (:mod:`transcriber.omr.ensemble`);
* MusicXML repair / normalisation via music21
  (:mod:`transcriber.omr.postprocess`).

The :mod:`transcriber.omr.eval` sub-package provides a self-contained accuracy
harness (render MusicXML -> image -> recognise -> compare) plus dataset
fetchers, so the system can be measured and refined against real
PDF/MusicXML corpora.
"""

from .pipeline import OMRConfig, OMRResult, recognize
from .types import OMRNote, RecognizedScore, StaffSystem

__all__ = [
    "OMRConfig",
    "OMRResult",
    "recognize",
    "OMRNote",
    "RecognizedScore",
    "StaffSystem",
]
