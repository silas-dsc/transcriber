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


def _keyed_phrase(midis, sharps):
    """Build a single-staff phrase with an explicit key signature."""
    from music21 import clef, key, meter, note, stream

    score = stream.Score()
    part = stream.Part()
    part.insert(0, clef.TrebleClef())
    part.insert(0, key.KeySignature(sharps))
    part.insert(0, meter.TimeSignature("4/4"))
    for i, m in enumerate(midis):
        n = note.Note(m)
        n.quarterLength = 1.0
        part.insert(float(i), n)
    score.insert(0, part)
    score.makeNotation(inPlace=True)
    return score


def test_detects_key_signature_and_applies_sharps():
    # D major (2 sharps: F#, C#). Notes on the F and C staff positions must be
    # read as F# / C# via the detected key signature.
    score = _keyed_phrase([66, 69, 73, 74, 73, 69, 66, 62], sharps=2)  # F#4..D5 line
    recognized = recognize_image(render_reference_array(score))
    assert recognized.key_sharps == 2
    assert [n.pitch for n in recognized.notes] == [66, 69, 73, 74, 73, 69, 66, 62]


def test_detects_flat_key_signature():
    # B-flat major (2 flats: B, E).
    score = _keyed_phrase([65, 67, 70, 72, 70, 67, 65, 63], sharps=-2)
    recognized = recognize_image(render_reference_array(score))
    assert recognized.key_sharps == -2
    assert [n.pitch for n in recognized.notes] == [65, 67, 70, 72, 70, 67, 65, 63]


def test_no_false_key_signature_on_natural_phrase():
    # A natural-only phrase has no key signature and must not gain one.
    recognized = recognize_image(render_reference_array(make_phrase(C_MAJOR_SCALE)))
    assert recognized.key_sharps == 0
    assert [n.pitch for n in recognized.notes] == C_MAJOR_SCALE


def _phrase_named(names):
    from music21 import clef, meter, note, stream

    score = stream.Score()
    part = stream.Part()
    part.insert(0, clef.TrebleClef())
    part.insert(0, meter.TimeSignature("4/4"))
    for i, name in enumerate(names):
        n = note.Note(name)
        n.quarterLength = 1.0
        part.insert(float(i), n)
    score.insert(0, part)
    score.makeNotation(inPlace=True)
    return score


def test_recognizes_inline_sharps_flats_and_naturals():
    # No key signature: every accidental here is an inline glyph on the note.
    score = _phrase_named(["C5", "C#5", "D5", "D-5", "E5", "F5", "G5", "G#5"])
    recognized = recognize_image(render_reference_array(score))
    expected = [n.pitch.midi for n in score.flatten().notes]
    assert [n.pitch for n in recognized.notes] == expected


def test_clean_preprocessing_is_lossless_on_clean_input():
    # The noise gating must leave clean line art untouched (no nicked staff
    # lines): a clean phrase still round-trips perfectly.
    recognized = recognize_image(render_reference_array(make_phrase(C_MAJOR_SCALE)))
    assert [n.pitch for n in recognized.notes] == C_MAJOR_SCALE


def test_robust_to_lighting_and_warp():
    from transcriber.omr.eval.augment import augment_preset

    score = make_phrase(C_MAJOR_SCALE)
    clean = render_reference_array(score)
    for preset in ("lighting", "warp"):
        degraded = augment_preset(clean, preset)
        recognized = recognize_image(degraded)
        assert [n.pitch for n in recognized.notes] == C_MAJOR_SCALE, preset


def test_salt_pepper_despeckling_helps():
    from transcriber.omr.eval.augment import augment
    from transcriber.omr.eval.metrics import compare_scores
    from transcriber.omr.assemble import build_score
    from transcriber.omr.primitive import PrimitiveConfig
    from transcriber.omr.preprocess import PreprocessConfig

    score = make_phrase(C_MAJOR_SCALE)
    noisy = augment(render_reference_array(score), salt_pepper=0.04, seed=1)
    off = PrimitiveConfig(preprocess=PreprocessConfig(prefilter=False))
    f1_off = compare_scores(score, build_score(recognize_image(noisy, off))).f1
    f1_on = compare_scores(score, build_score(recognize_image(noisy))).f1
    # The de-speckling prefilter must substantially beat raw thresholding.
    assert f1_on > f1_off + 0.2


def test_inline_accidental_on_high_ledger_note():
    # An inline sharp on a high ledger-line note (C#6) must be detected and
    # applied -- the accidental glyph sits several spaces above the staff.
    score = _phrase_named(["C#6", "C#6", "C#6", "C#6"])
    recognized = recognize_image(render_reference_array(score))
    assert [n.pitch for n in recognized.notes] == [85, 85, 85, 85]


def test_first_note_accidental_not_mistaken_for_key_signature():
    # A lone inline accidental on the first note must not be read as a one-sharp
    # key signature (which would also shift later notes).
    score = _phrase_named(["G#4", "A4", "B4", "C5"])
    recognized = recognize_image(render_reference_array(score))
    assert recognized.key_sharps == 0
    assert [n.pitch for n in recognized.notes] == [68, 69, 71, 72]


def test_inline_natural_cancels_key_signature():
    # D major (F#, C#) with an explicit F-natural: must read F natural (65),
    # not the key-signature F# (66).
    score = _keyed_phrase([66, 69, 71, 74, 65, 69, 66, 62], sharps=2)
    score.recurse().notes[4].pitch.accidental = "natural"  # the F4
    recognized = recognize_image(render_reference_array(score))
    assert recognized.notes[4].pitch == 65


# --------------------------------------------------------------------------- #
# Confidence & human-in-the-loop review (Target 2)
# --------------------------------------------------------------------------- #
def test_clean_score_has_high_confidence_and_no_review(tmp_path):
    image = render_reference(make_phrase(C_MAJOR_SCALE), tmp_path / "clean.png")
    result = recognize(image, None, OMRConfig(engine="primitive"))
    assert result.confidence_report is not None
    assert result.confidence_report.overall >= 0.9
    assert result.confidence_report.n_review == 0


def test_degraded_score_is_flagged_for_review(tmp_path):
    from PIL import Image
    from transcriber.omr.eval.augment import augment_preset

    clean = render_reference_array(make_phrase(C_MAJOR_SCALE))
    degraded = augment_preset(clean, "blur")
    path = tmp_path / "blur.png"
    Image.fromarray((degraded * 255).clip(0, 255).astype("uint8")).save(path)
    result = recognize(path, None, OMRConfig(engine="primitive"))
    # A blurred page is uncertain -> at least one measure should be queued.
    assert result.confidence_report.n_review >= 1


def test_multi_engine_disagreement_flags_contested_notes():
    from transcriber.omr.confidence import build_confidence

    primary = make_phrase([60, 62, 64, 65])
    primary.makeNotation(inPlace=True)
    agree = make_phrase([60, 62, 64, 65])
    agree.makeNotation(inPlace=True)
    disagree = make_phrase([60, 62, 99, 65])  # differs on the 3rd note
    disagree.makeNotation(inPlace=True)

    report = build_confidence(
        primary, None, {"primitive": primary, "oemer": disagree, "homr": agree}
    )
    assert report.n_review >= 1
    assert any("disagree" in r for item in report.review_items for r in item.reasons)


def test_annotate_review_marks_flagged_measures(tmp_path):
    from PIL import Image
    from music21 import expressions
    from transcriber.omr.eval.augment import augment_preset

    degraded = augment_preset(render_reference_array(make_phrase(C_MAJOR_SCALE)), "blur")
    path = tmp_path / "blur.png"
    Image.fromarray((degraded * 255).clip(0, 255).astype("uint8")).save(path)
    result = recognize(path, None, OMRConfig(engine="primitive", mark_review=True))
    marks = list(result.score.recurse().getElementsByClass(expressions.TextExpression))
    assert any(m.content == "review?" for m in marks)


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
    # The built-in recogniser + semantic post-processing must clear a high
    # quality floor on clean printed monophonic music.
    assert result.mean_f1 >= 0.9
    assert result.mean_precision >= 0.95
    assert "MEAN" in format_report(result)


# --------------------------------------------------------------------------- #
# Semantic sanity checks
# --------------------------------------------------------------------------- #
def test_semantic_infers_a_key():
    from transcriber.omr.semantic import validate

    # A bare C-major scale is tonally ambiguous (music21 may report the
    # relative A minor); we only assert that key inference produced something.
    _, report = validate(make_phrase(C_MAJOR_SCALE))
    assert report.key is not None
    assert report.key.split()[0] in {"C", "A"}


def test_semantic_merges_exact_duplicate_notes():
    from music21 import note as m21note

    from transcriber.omr.semantic import validate

    score = make_phrase([60, 62, 64], [1, 1, 1])
    dup = m21note.Note(62)
    dup.quarterLength = 1.0
    score.parts[0].insert(1.0, dup)  # duplicate of the second note
    before = len(score.flatten().notes)
    fixed, report = validate(score, repair=True)
    assert len(fixed.flatten().notes) == before - 1
    assert report.n_fixed >= 1


def test_semantic_corrects_octave_outlier_when_aggressive():
    from transcriber.omr.semantic import validate

    # 84 (C6) is an octave-plus leap away from its neighbours on both sides.
    score = make_phrase([60, 62, 84, 64, 67, 69, 71, 72], [1] * 8)
    fixed, report = validate(score, repair=True, aggressive=True)
    pitches = [n.pitch.midi for n in fixed.flatten().notes]
    assert 84 not in pitches  # the outlier was pulled back an octave
    assert any(i.kind == "octave_outlier" and i.fixed for i in report.issues)


def test_semantic_flags_but_does_not_fix_octave_outlier_by_default():
    from transcriber.omr.semantic import validate

    score = make_phrase([60, 62, 84, 64, 67, 69, 71, 72], [1] * 8)
    fixed, report = validate(score, repair=True, aggressive=False)
    pitches = [n.pitch.midi for n in fixed.flatten().notes]
    assert 84 in pitches  # left unchanged
    assert any(i.kind == "octave_outlier" and not i.fixed for i in report.issues)


# --------------------------------------------------------------------------- #
# LLM review (offline, with an injected fake client)
# --------------------------------------------------------------------------- #
class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeMessages:
    def __init__(self, payload):
        self._payload = payload

    def create(self, **kwargs):
        import json
        import types

        return types.SimpleNamespace(content=[_FakeBlock(json.dumps(self._payload))])


class _FakeClient:
    def __init__(self, payload):
        self.messages = _FakeMessages(payload)


def test_llm_review_applies_validated_octave_correction():
    from transcriber.omr.llm_review import review_score

    score = make_phrase([60, 62, 84, 64], [1, 1, 1, 1])
    client = _FakeClient({"corrections": [{"index": 2, "action": "octave_down", "reason": "outlier"}]})
    fixed, report = review_score(score, client=client, apply=True)
    assert report.applied == 1
    assert 72 in [n.pitch.midi for n in fixed.flatten().notes]  # 84 -> 72


def test_llm_review_rejects_out_of_range_index():
    from transcriber.omr.llm_review import review_score

    score = make_phrase([60, 62, 64], [1, 1, 1])
    client = _FakeClient({"corrections": [{"index": 99, "action": "octave_up", "reason": "bad"}]})
    _, report = review_score(score, client=client, apply=True)
    assert report.applied == 0 and report.skipped == 1


def test_llm_review_never_runs_without_client_or_sdk(monkeypatch):
    from transcriber.omr import llm_review

    # Simulate no SDK / no credentials: _make_client returns None.
    monkeypatch.setattr(llm_review, "_make_client", lambda: None)
    score = make_phrase(C_MAJOR_SCALE)
    _, report = llm_review.review_score(score, client=None)
    assert report.error is not None and report.applied == 0


# --------------------------------------------------------------------------- #
# Jazz / handwritten font rendering (verovio only)
# --------------------------------------------------------------------------- #
def test_petaluma_font_renders_and_differs_from_default(tmp_path):
    """Rendering with the handwritten 'Petaluma' face must work and differ
    from the default engraved font -- the precondition for measuring how much
    accuracy the jazz/handwritten glyphs cost."""
    pytest.importorskip("verovio")
    # cairocffi dlopens libcairo at *import* time, raising OSError (not
    # ImportError) when the system lib is absent -- so importorskip won't catch
    # it.  Skip cleanly in both cases.
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("cairosvg/libcairo unavailable (try: brew install cairo)")
    from PIL import Image

    score = make_phrase(C_MAJOR_SCALE)
    default_png = render_reference(score, tmp_path / "default.png", renderer="verovio")
    jazz_png = render_reference(score, tmp_path / "jazz.png", renderer="verovio", font="Petaluma")

    default_arr = np.asarray(Image.open(default_png).convert("L"))
    jazz_arr = np.asarray(Image.open(jazz_png).convert("L"))
    assert jazz_arr.min() < 128  # actually drew ink, not a blank page
    # Different fonts => different pixels (allowing for identical canvas sizes).
    assert default_arr.shape != jazz_arr.shape or not np.array_equal(default_arr, jazz_arr)


def test_builtin_renderer_ignores_font_without_crashing(tmp_path):
    """A font on the built-in engraver is a no-op, not an error."""
    out = render_reference(make_phrase(C_MAJOR_SCALE), tmp_path / "b.png",
                           renderer="builtin", font="Petaluma")
    assert out.exists()


def test_musescore_musejazz_renders_when_available(tmp_path):
    """If MuseScore is installed, the musescore renderer applies the MuseJazz
    handwritten style and produces a non-blank page (precondition for the
    jazz-font corpus)."""
    from transcriber.omr.eval.render_ref import _find_musescore, render_reference

    if _find_musescore() is None:
        pytest.skip("MuseScore CLI not installed")
    from PIL import Image

    out = render_reference(
        make_phrase(C_MAJOR_SCALE), tmp_path / "mj.png",
        renderer="musescore", style="MuseJazz", dpi=150,
    )
    assert out.exists()
    assert np.asarray(Image.open(out).convert("L")).min() < 128  # drew ink


# --------------------------------------------------------------------------- #
# Chord-symbol recognition (jazz lead-sheet payload)
# --------------------------------------------------------------------------- #
def test_jazz_text_to_figure_handles_fakebook_shorthand():
    from transcriber.omr.chords import jazz_text_to_figure

    assert jazz_text_to_figure("C-7") == "Cm7"      # '-' = minor
    assert jazz_text_to_figure("Bb7") == "B-7"       # jazz flat -> music21 flat
    assert jazz_text_to_figure("A-7b5") == "Am7b5"   # half-dim spelling
    assert jazz_text_to_figure("C△") == "Cmaj7"  # triangle = major7
    assert jazz_text_to_figure("F#m") == "F#m"
    assert jazz_text_to_figure("") is None


def test_normalize_chord_figure_canonicalises_music21_spelling():
    from transcriber.omr.eval.metrics import normalize_chord_figure

    assert normalize_chord_figure("B-m6") == "Bbm6"   # music21 '-' = flat
    assert normalize_chord_figure("Cmin7") == "Cm7"
    assert normalize_chord_figure("C△") == "Cmaj7"
    assert normalize_chord_figure("Cmaj7") == "Cmaj7"


def test_make_lead_sheet_carries_chord_symbols():
    from transcriber.omr.eval.datasets import make_lead_sheet
    from transcriber.omr.eval.metrics import score_to_chords

    score = make_lead_sheet([60, 62, 64, 65], chords=[(0.0, "Cm7"), (2.0, "F7")])
    chords = score_to_chords(score)
    assert chords == [(0.0, "Cm7"), (2.0, "F7")]


def test_compare_chords_perfect_and_penalises_misses():
    from transcriber.omr.eval.datasets import make_lead_sheet
    from transcriber.omr.eval.metrics import compare_chords

    ref = make_lead_sheet([60, 62, 64, 65], chords=[(0.0, "Cm7"), (2.0, "F7")])
    assert compare_chords(ref, ref).f1 == 1.0
    partial = make_lead_sheet([60, 62, 64, 65], chords=[(0.0, "Cm7")])
    c = compare_chords(ref, partial)
    assert c.recall == 0.5 and c.precision == 1.0


def test_attach_chords_round_trips_through_compare():
    from music21 import stream

    from transcriber.omr.chords import attach_chords
    from transcriber.omr.types import OMRChordSymbol
    from transcriber.omr.eval.datasets import make_lead_sheet
    from transcriber.omr.eval.metrics import compare_chords

    ref = make_lead_sheet([60, 62, 64, 65], chords=[(0.0, "Cm7"), (2.0, "F7")])
    pred = stream.Score()
    pred.insert(0, stream.Part())
    # attach_chords feeds figures verbatim to music21, so OMRChordSymbol.figure
    # is already music21 syntax (recognize_chords converts jazz text first).
    attach_chords(pred, [OMRChordSymbol("Cm7", 0.0), OMRChordSymbol("F7", 2.0)])
    assert compare_chords(ref, pred).f1 == 1.0


def test_recognize_chords_without_ocr_returns_empty():
    from transcriber.omr.chords import recognize_chords, _ocr_backend

    if _ocr_backend() is not None:
        pytest.skip("an OCR backend is installed; this guards the no-OCR path")
    assert recognize_chords(np.ones((80, 200), dtype=np.float32), []) == []


def test_chord_ocr_reads_musejazz_roots_when_available(tmp_path):
    """With tesseract + MuseScore present, the chord OCR recognises at least
    one chord root off a MuseJazz lead-sheet render. Extensions (superscript 7)
    are not asserted -- off-the-shelf OCR loses most of them."""
    pytest.importorskip("pytesseract")
    import numpy as np
    from PIL import Image

    from transcriber.omr.chords import recognize_chords, _ocr_backend
    from transcriber.omr.eval.render_ref import _find_musescore, render_reference
    from transcriber.omr.eval.datasets import make_lead_sheet
    from transcriber.omr.preprocess import preprocess, PreprocessConfig
    from transcriber.omr.primitive import detect_staves

    if _ocr_backend() is None or _find_musescore() is None:
        pytest.skip("needs a tesseract OCR backend + the MuseScore CLI")
    ref = make_lead_sheet([60, 62, 64, 65, 67, 69, 71, 72],
                          chords=[(0.0, "Cmaj7"), (4.0, "Am7")])
    p = render_reference(ref, tmp_path / "ls.png", renderer="musescore", style="MuseJazz", dpi=300)
    arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    systems = detect_staves(preprocess(arr, PreprocessConfig(deskew=False)))
    chords = recognize_chords(arr, systems, beats_per_system=8.0)
    roots = {c.figure[:1] for c in chords}
    assert roots & {"C", "A"}, f"no expected chord root recognised; got {roots}"
