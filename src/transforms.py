"""The 6 robustness transforms from PLAN.md Section 3.1.

Each transform is a pure function: PIL.Image in, severity in, PIL.Image out.
They take no other state, so they compose/chain trivially — PLAN.md
Section 3.4 needs to run e.g. resize -> blur -> JPEG in sequence for the
stacked worst-case tests, so each function returns a new PIL.Image that
can be fed straight into the next one.

Used for BOTH training-time augmentation (Section 3.2/3.3) and the
evaluation harness (Section 3.1/Phase 2/5) — implement once, reuse
everywhere, per PLAN.md Section 4.
"""
import io

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# The exact transform grid from PLAN.md Section 3.1, reused everywhere so
# eval/train/robustness_eval all draw from one source of truth.
TRANSFORM_GRID = {
    "jpeg": [90, 70, 50, 30],
    "gaussian_blur": [0.5, 1.0, 2.0],
    "resize": [0.5, 0.25],
    "gaussian_noise": [0.02, 0.05, 0.10],
    "color_jitter": [0.20],
    "center_crop": [0.80],
}

HELD_IN_TRANSFORMS = ["jpeg", "gaussian_blur", "color_jitter", "center_crop"]
HELD_OUT_TRANSFORMS = ["resize", "gaussian_noise"]


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at the given quality (Section 3.1: 90/70/50/30)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur with the given sigma (Section 3.1: 0.5/1.0/2.0)."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize(image: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale` then upscale back to original size (Section 3.1: 0.5x/0.25x)."""
    w, h = image.size
    small_w, small_h = max(1, round(w * scale)), max(1, round(h * scale))
    downsized = image.resize((small_w, small_h), Image.BILINEAR)
    return downsized.resize((w, h), Image.BILINEAR)


def apply_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian noise, sigma in [0,1] pixel-value units (Section 3.1: 0.02/0.05/0.10)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8))


def apply_color_jitter(image: Image.Image, strength: float) -> Image.Image:
    """Brightness/contrast/saturation each jittered by +-strength (Section 3.1: +-20%)."""
    img = image.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + np.random.uniform(-strength, strength)
        img = enhancer_cls(img).enhance(factor)
    return img


def apply_center_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Center-crop to `fraction` of width/height, then resize back to original size (Section 3.1: 80%)."""
    w, h = image.size
    crop_w, crop_h = round(w * fraction), round(h * fraction)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.BILINEAR)


TRANSFORM_FNS = {
    "jpeg": apply_jpeg,
    "gaussian_blur": apply_gaussian_blur,
    "resize": apply_resize,
    "gaussian_noise": apply_gaussian_noise,
    "color_jitter": apply_color_jitter,
    "center_crop": apply_center_crop,
}


def apply_transform(image: Image.Image, name: str, severity) -> Image.Image:
    """Dispatch to the named transform. Single entry point for both training
    augmentation and eval, and the building block for chaining (Section 3.4)."""
    if name not in TRANSFORM_FNS:
        raise ValueError(f"Unknown transform: {name}")
    return TRANSFORM_FNS[name](image, severity)


def apply_chain(image: Image.Image, steps) -> Image.Image:
    """Apply a sequence of (name, severity) pairs in order (Section 3.4,
    e.g. resize 0.25x -> blur sigma=2.0 -> JPEG q=30). Each step's output
    feeds directly into the next, since every transform above is a pure
    PIL.Image -> PIL.Image function."""
    for name, severity in steps:
        image = apply_transform(image, name, severity)
    return image
