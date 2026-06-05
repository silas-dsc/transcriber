"""Accuracy evaluation for the OMR pipeline.

A self-contained loop to *measure and refine* recognition quality:

* :mod:`metrics` -- symbolic metrics computed from MusicXML pairs (note-level
  precision/recall/F1, Symbol Error Rate, and an MV2H-style breakdown).
* :mod:`render_ref` -- render a reference MusicXML score back to a page image
  (verovio / MuseScore / LilyPond if present, else a built-in renderer) so the
  recognise-and-compare loop needs no external tools.
* :mod:`datasets` -- fetch real PDF/MusicXML corpora (OpenScore Lieder, PDMX)
  and generate synthetic phrases for fast offline testing.
* :mod:`harness` -- run an engine over a corpus and report aggregate accuracy.
"""

from .metrics import ScoreComparison, compare_scores, symbol_error_rate

__all__ = ["ScoreComparison", "compare_scores", "symbol_error_rate"]
