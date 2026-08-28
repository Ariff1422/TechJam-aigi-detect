"""Cheap, training-free image-quality heuristics (Phase A go/no-go gate
for degradation-aware calibration — see PLAN.md changelog).

Three signals, each meant to correlate with one of the distortion types
in PLAN.md Section 3.1, computed directly from pixels with no model
forward pass:

- blur_score: Laplacian variance. Low variance = blurry (few sharp
  edges); high variance = sharp.
- noise_score: high-frequency residual energy — image minus a
  heavily-blurred version of itself, then the residual's variance. High
  residual energy = noisy/high-frequency content beyond what a blurred
  version would explain.
- jpeg_block_score: 8x8 block-edge energy — the mean absolute pixel
  difference across block boundaries, which JPEG's block-based DCT
  compression tends to produce as visible discontinuities.

None of these involve the classifier or any trained model — this is a
Phase A diagnostic to check whether pixel-level heuristics separate
clean from degraded images at all, before building anything that
routes on them (Phase B).
"""
import cv2
import numpy as np
from PIL import Image


def blur_score(img: Image.Image) -> float:
    """Laplacian variance. Lower = blurrier."""
    gray = np.array(img.convert("L"), dtype=np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def noise_score(img: Image.Image, blur_ksize: int = 5) -> float:
    """High-frequency residual energy. Higher = noisier."""
    gray = np.array(img.convert("L"), dtype=np.float64)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    residual = gray - blurred
    return float(residual.var())


def jpeg_block_score(img: Image.Image, block_size: int = 8) -> float:
    """8x8 block-edge energy — mean abs pixel difference across block
    boundaries. Higher = more visible JPEG blocking artifacts."""
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    diffs = []
    for x in range(block_size, w, block_size):
        if x >= w:
            break
        diffs.append(np.abs(gray[:, x] - gray[:, x - 1]))
    for y in range(block_size, h, block_size):
        if y >= h:
            break
        diffs.append(np.abs(gray[y, :] - gray[y - 1, :]))

    if not diffs:
        return 0.0
    return float(np.mean([d.mean() for d in diffs]))


def compute_all(img: Image.Image) -> dict:
    return {
        "blur_score": blur_score(img),
        "noise_score": noise_score(img),
        "jpeg_block_score": jpeg_block_score(img),
    }
