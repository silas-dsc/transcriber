"""Training-data generation for jazz-font OMR.

The measured instability of off-the-shelf engines on the handwritten MuseJazz
style (frequent staffline/symbol-extraction crashes; see the package README) is
a *pretraining* gap -- the fix is to train on the jazz style.  This subpackage
produces the inputs for that: (image, ground-truth) pairs rendered in the jazz
font with scan-style augmentation, from any MusicXML corpus.

Actual model training (a GPU job) lives outside this repo; see
``transcriber/omr/train/README.md`` for the recommended path and why oemer's
own segmentation fine-tune is the harder option.
"""

from .corpus import build_corpus, score_to_tokens, tokens_to_score

__all__ = ["build_corpus", "score_to_tokens", "tokens_to_score"]
