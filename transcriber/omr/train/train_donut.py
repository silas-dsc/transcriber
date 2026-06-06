"""Fine-tune Donut on a jazz-font OMR corpus (image -> token string).

SCAFFOLD: runs on a GPU box with `torch`, `transformers`, `datasets`,
`accelerate` installed (not part of the repo's deps).  See GPU_TRAINING.md.
Adapt freely -- this is a faithful starting point, not a turnkey SOTA recipe.

Reads one or more manifest.jsonl files produced by `omr-build-corpus`
(`{id, image, musicxml, tokens}`), targets `" ".join(tokens)`, registers the
music tokens as atomic special tokens, and fine-tunes naver-clova-ix/donut-base.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(manifests: list[str]) -> list[dict]:
    recs = []
    for m in manifests:
        root = Path(m).parent
        for line in open(m):
            r = json.loads(line)
            r["image_path"] = str((root / r["image"]).resolve())
            r["text"] = " ".join(r["tokens"])
            recs.append(r)
    return recs


def main() -> None:
    import torch
    from datasets import Dataset
    from PIL import Image
    from transformers import (
        DonutProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--image-size", nargs=2, type=int, default=[1280, 960])  # H W
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--freeze-encoder-epochs", type=int, default=1)
    ap.add_argument("--base", default="naver-clova-ix/donut-base")
    ap.add_argument("--out", default="runs/donut-jazz")
    args = ap.parse_args()
    H, W = args.image_size

    recs = load_records(args.manifests)
    vocab = sorted({t for r in recs for t in r["tokens"]})

    processor = DonutProcessor.from_pretrained(args.base)
    model = VisionEncoderDecoderModel.from_pretrained(args.base)

    # Music tokens become atomic (one id each), not sub-worded.
    processor.tokenizer.add_special_tokens({"additional_special_tokens": vocab})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    processor.image_processor.size = {"height": H, "width": W}
    processor.image_processor.do_align_long_axis = False
    model.config.encoder.image_size = [H, W]
    model.config.decoder.max_length = 1024
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or 0

    def encode(batch):
        img = Image.open(batch["image_path"]).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values[0]
        labels = processor.tokenizer(
            batch["text"], add_special_tokens=True, max_length=1024,
            padding="max_length", truncation=True,
        ).input_ids
        labels = [t if t != processor.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixel_values, "labels": labels}

    ds = Dataset.from_list(recs).train_test_split(test_size=args.val_frac, seed=0)
    ds = ds.map(encode, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch", columns=["pixel_values", "labels"])

    if args.freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False  # unfreeze after a warmup epoch (see callback note in docs)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["test"],
    )
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"saved fine-tuned Donut to {args.out}")


if __name__ == "__main__":
    main()
