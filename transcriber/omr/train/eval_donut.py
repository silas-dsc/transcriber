"""Evaluate a fine-tuned Donut OMR model with this repo's symbolic metrics.

SCAFFOLD (GPU box; needs torch + transformers). For each manifest item: render
the page through the model, decode the token string, rebuild a music21 score
with `tokens_to_score`, and grade it against the ground truth with
`compare_scores` (notes) and `compare_chords` (chord symbols).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    from transcriber.omr.eval.metrics import compare_chords, compare_scores
    from transcriber.omr.train.corpus import tokens_to_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-size", nargs=2, type=int, default=[1280, 960])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()
    H, W = args.image_size

    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    processor.image_processor.size = {"height": H, "width": W}

    root = Path(args.manifest).parent
    note_f1, chord_f1 = [], []
    for line in open(args.manifest):
        rec = json.loads(line)
        ref = __import__("music21").converter.parse(str(root / rec["musicxml"]))
        img = Image.open(root / rec["image"]).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=args.max_new_tokens)
        text = processor.tokenizer.batch_decode(ids, skip_special_tokens=False)[0]
        toks = [t for t in text.replace("<s>", "").replace("</s>", "").split() if "_" in t]
        pred = tokens_to_score(toks)
        note_f1.append(compare_scores(ref, pred).f1)
        chord_f1.append(compare_chords(ref, pred).f1)

    print(f"items={len(note_f1)}  note_F1={statistics.fmean(note_f1):.3f}  "
          f"chord_F1={statistics.fmean(chord_f1):.3f}")


if __name__ == "__main__":
    main()
