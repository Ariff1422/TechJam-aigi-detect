"""Required deliverable (PLAN.md Section 0 / Section 4).

Takes a directory of images, produces a JSON file with `image_path` and
`pred` (a confidence score in [0,1], not a thresholded label) for each image.
`pred` is P(fake) / P(AI-generated) — label convention 1=fake, 0=real,
locked per PLAN.md Section 0. `image_path` is relative to --image-dir.
"""
import argparse
import json
import os

import torch

from src.checkpoint import load_model_from_checkpoint
from src.features import extract_embeddings

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir: str):
    paths = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.join(image_dir, fname))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory of images to classify")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained head checkpoint (.pt)")
    parser.add_argument("--output", default="outputs/predictions.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess, head, embedding_dim = load_model_from_checkpoint(args.checkpoint, device)

    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    embeddings = extract_embeddings(image_paths, model, preprocess, device).to(device)
    with torch.no_grad():
        logits = head(embeddings)
        probs = torch.sigmoid(logits).cpu().tolist()

    results = [
        {"image_path": os.path.relpath(path, args.image_dir), "pred": float(prob)}
        for path, prob in zip(image_paths, probs)
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
