"""Tests for the optical music recognition (OMR) pipeline.

The backbone is a self-contained round trip: render a known score to an image
with the built-in engraver, recognise it with the built-in (always-available)
recogniser, and assert we get the music back.  This exercises preprocessing,
staff/head detection, pitch mapping, assembly, the metrics and the harness
without any heavy dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest
from music21 import converter

from transcriber.omr import OMRConfig, recognize
from transcriber.omr.eval.datasets import make_phrase, synthetic_corpus
from transcriber.omr.eval.harness import evaluate_corpus, format_report
from transcriber.omr.eval.metrics import compare_scores, symbol_error_rate
from transcriber.omr.eval.render_ref import render_reference, render_reference_array
from transcriber.omr.engines import available_engines, get_engine
from transcriber.omr.preprocess import PreprocessConfig, binarize, estimate_skew, preprocess
from transcriber.omr.primitive import detect_staves, recognize_image

C_MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71, 72]


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def test_binarize_methods_agree_on_clean_image():
    img = render_reference_array(make_phrase(C_MAJOR_SCALE))
    otsu = binarize(img, "otsu")
    sauvola = binarize(img, "sauvola")
    # Both should mark a small minority of pixels as ink on a sparse score.
    assert 0.0 < otsu.mean() < 0.3
    assert 0.0 < sauvola.mean() < 0.3


def test_estimate_skew_is_zero_for_level_image():
    img = render_reference_array(make_phrase(C_MAJOR_SCALE))
    assert abs(estimate_skew(img)) < 1.0


def test_deskew_recovers_a_known_rotation():
    from scipy import ndimage

    img = render_reference_array(make_phrase(C_MAJOR_SCALE))
    rotated = ndimage.rotate(img, 4.0, reshape=False, order=1, mode="constant", cval=1.0)
    # The estimator should detect a rotation that compensates the +4 deg skew.
    assert estimate_skew(rotated) < -1.0


def test_preprocess_returns_boolean_ink_mask():
    img = render_reference_array(make_phrase(C_MAJOR_SCALE))
    mask = preprocess(img, PreprocessConfig(deskew=False))
    assert mask.dtype == bool
    assert mask.shape == img.shape
    assert mask.any()


# --------------------------------------------------------------------------- #
# Staff & note-head detection
# --------------------------------------------------------------------------- #
def test_detects_a_single_five_line_staff():
    img = render_reference_array(make_phrase(C_MAJOR_SCALE))
    staves = detect_staves(preprocess(img, PreprocessConfig(deskew=False)))
    assert len(staves) == 1
    assert len(staves[0].line_ys) == 5
    assert staves[0].staff_space > 0


def test_primitive_recovers_a_c_major_scale_exactly():
    img = render_reference_array(make_phrase(C_MAJOR_SCALE))
    recognized = recognize_image(img)
    assert [n.pitch for n in recognized.notes] == C_MAJOR_SCALE


def test_primitive_distinguishes_filled_and_hollow_durations():
    # A quarter (filled) then a half (hollow) on the same pitch.  C5 sits in a
    # staff *space*, the case where hollow-head detection is most reliable
    # (a hollow head centred *on* a line is harder and may read as filled).
    score = make_phrase([72, 72], [1.0, 2.0])
    recognized = recognize_image(render_reference_array(score))
    durations = [n.duration for n in recognized.notes]
    assert durations == [1.0, 2.0]


def test_blank_page_yields_no_notes():
    blank = np.ones((200, 300), dtype=np.float32)
    recognized = recognize_image(blank)
    assert recognized.notes == []


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_identical_scores_score_perfectly():
    score = make_phrase(C_MAJOR_SCALE)
    comparison = compare_scores(score, score)
    assert comparison.f1 == pytest.approx(1.0)
    assert comparison.symbol_error_rate == pytest.approx(0.0)
    assert comparison.n_matched == len(C_MAJOR_SCALE)


def test_metrics_penalise_a_wrong_note():
    reference = make_phrase(C_MAJOR_SCALE)
    wrong = make_phrase([60, 62, 99, 65, 67, 69, 71, 72])  # one pitch changed
    comparison = compare_scores(reference, wrong)
    assert comparison.f1 < 1.0
    assert comparison.symbol_error_rate > 0.0


def test_symbol_error_rate_counts_insertions():
    reference = make_phrase([60, 62, 64])
    longer = make_phrase([60, 62, 64, 65, 67])  # two extra notes
    assert symbol_error_rate(reference, longer) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_render_reference_writes_an_image(tmp_path):
    out = render_reference(make_phrase(C_MAJOR_SCALE), tmp_path / "ref.png")
    assert out.exists()
    arr = render_reference_array(make_phrase(C_MAJOR_SCALE))
    assert arr.ndim == 2 and arr.min() >= 0.0 and arr.max() <= 1.0


# --------------------------------------------------------------------------- #
# Engines & pipeline
# --------------------------------------------------------------------------- #
def test_primitive_engine_is_always_available():
    engines = available_engines()
    assert "primitive" in engines


def test_auto_engine_falls_back_to_primitive_without_external_engines():
    # In a bare environment (no oemer/homr/audiveris) auto resolves to primitive.
    if available_engines() == ["primitive"]:
        assert get_engine("auto").name == "primitive"


def test_recognize_writes_valid_musicxml(tmp_path):
    image = render_reference(make_phrase(C_MAJOR_SCALE), tmp_path / "sheet.png")
    out = tmp_path / "out.musicxml"
    result = recognize(image, out, OMRConfig(engine="primitive", title="Scale"))
    assert result.output_path is not None
    assert out.exists()
    reparsed = converter.parse(str(out))  # must be valid MusicXML
    assert [int(n.pitch.midi) for n in reparsed.flatten().notes] == C_MAJOR_SCALE


def test_ensemble_selects_an_available_engine(tmp_path):
    image = render_reference(make_phrase(C_MAJOR_SCALE), tmp_path / "sheet.png")
    result = recognize(image, None, OMRConfig(engine="ensemble"))
    assert result.engine in available_engines()
    assert result.confidence > 0.0


def test_recognize_multipage_pdf(tmp_path):
    from PIL import Image

    p1 = render_reference(make_phrase([60, 62, 64, 65]), tmp_path / "p1.png")
    p2 = render_reference(make_phrase([67, 69, 71, 72]), tmp_path / "p2.png")
    pdf = tmp_path / "sheets.pdf"
    Image.open(p1).convert("1").save(
        pdf, save_all=True, append_images=[Image.open(p2).convert("1")]
    )
    result = recognize(pdf, None, OMRConfig(dpi=200))
    assert result.page_count == 2
    # Both pages' notes should appear in one merged part.
    assert len(result.score.flatten().notes) == 8


# --------------------------------------------------------------------------- #
# Accuracy harness (regression guard)
# --------------------------------------------------------------------------- #
def test_synthetic_benchmark_meets_accuracy_floor():
    items = synthetic_corpus(n_items=6, notes_per_item=8, seed=1)
    result = evaluate_corpus(items, engine="primitive")
    # The built-in recogniser must clear a quality floor on clean printed music.
    assert result.mean_f1 >= 0.8
    assert result.mean_precision >= 0.9
    assert "MEAN" in format_report(result)
