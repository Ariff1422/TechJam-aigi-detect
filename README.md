# AI-Generated Image Detection (TikTok TechJam 2026, Track 5)

Binary classifier — authentic photograph vs. fully AI-generated image — built on a frozen CLIP vision encoder with a linear probe. Full build log, evidence, and every decision (including the ones we reversed) is in [`PLAN.md`](PLAN.md); this README is the short version.

## Overview

**Model:** frozen CLIP ViT-B/32 (OpenAI pretrained, `open_clip`) + a single trainable linear layer on the 512-dim embedding. **151,277,826 parameters total** — 151,277,313 frozen (backbone) + 513 trainable (head). No fine-tuning of the backbone in the delivered model.

**Training data:** [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) (real photos from OpenImages V7, synthetic images from FLUX), tampered/locally-edited images excluded by design — this is a fully-generated-vs-real detector, not a manipulation-localization one. 17,945 images subsampled, deduplicated (pHash), split 70/15/15 train/val/test (12,561 / 2,692 / 2,692).

**Robustness strategy:** trained with JPEG, Gaussian blur, color jitter, and center crop applied to every training image (`distortion_prob=1.0`); Resize and Gaussian Noise held out entirely from training so the reported robustness numbers reflect genuine generalization, not memorized augmentation.

**Why this architecture:** a frozen CLIP linear probe was the only method in a published cross-generator comparison without a "blind spot" generator family (consistent ~93% mAP on both GANs and diffusion models, vs. large family-specific drops for a plain CNN or a reconstruction-based detector). See PLAN.md Section 1 for the full evidence and the DINOv2 counter-evidence we weighed against it.

## Results

**In-distribution (SID_Set — same real-photo source and generator as training):**

| Metric | Split | Value |
|---|---|---|
| Clean accuracy / AUC | validation split | 98.85% / 0.9996 (threshold=0.5) |
| Clean accuracy / AUC | test split | 99.15% / 0.9997 |
| Held-in transforms (JPEG, blur, color jitter, crop), all severities | test split | within ~0.3pt of clean |
| Held-out Resize, never trained on | test split | 99.18% / 0.9997 (0.5x), 98.70% / 0.9995 (0.25x) |
| Held-out Gaussian Noise σ=0.10, never trained on | test split | 82.99% / 0.9976 |
| Worst stacked distortion (resize→noise→JPEG, beyond-spec) | test split | 71.51% / 0.9955 |

Full transform × severity grid: `outputs/robustness_table_phase4.csv`.

**Out-of-distribution (CIFAKE — a different dataset entirely: Stable Diffusion instead of FLUX, and CIFAR-10-sourced real photos instead of OpenImages, i.e. both the generator *and* the real-image source differ from training):**

| Metric | Value |
|---|---|
| Accuracy / AUC (threshold=0.5) | 52.3% / 0.8214 |

This number should not be read at face value as "cross-generator generalization" on its own — see [Limitations](#limitations-and-what-wed-improve-with-more-time) below for a decomposition showing roughly half of this drop is a resolution artifact shared with in-distribution data, not something specific to the generator change.

Full reasoning and every intermediate result: `PLAN.md`'s changelog.

## Setup

```bash
pip install -r requirements.txt
```

Built and tested on Python 3.12.10; no lower bound has been verified, so don't treat that as a floor. No GPU required for inference or for the delivered linear-probe training (backbone is frozen); a GPU would meaningfully speed up the feature-caching step and make the DINOv2/LoRA experiments (see Limitations) practical — this project was built entirely on CPU.

## Running inference (the required deliverable)

```bash
python -m src.infer --image-dir /path/to/images --checkpoint outputs/checkpoints/phase4_model.pt --output outputs/predictions.json
```

Produces JSON: `[{"image_path": "relative/path.jpg", "pred": 0.0234}, ...]` — `pred` is P(fake) / P(AI-generated), a raw confidence score in [0,1], not a thresholded label. Threshold used for accuracy reporting throughout this project is 0.5 (stated explicitly, not left implicit — see PLAN.md Section 6).

## Reproducing from scratch

Every step below is seeded (`seed=42`, in `configs/default.yaml`) for reproducibility.

1. **Get the data.** Download SID_Set shards and extract real/synthetic images (drops tampered), check for a native-resolution shortcut, normalize resolution, dedupe, and split:
   ```bash
   python -m src.prepare_data --step extract --shard-dir <path to downloaded SID_Set parquet shards>
   python -m src.prepare_data --step check_resolution
   python -m src.prepare_data --step normalize_resolution
   python -m src.prepare_data --step split
   ```
   ⚠️ **The `split` step overwrites `data/splits/{train,val,test}.txt` in place** — including the ones already checked into this repo, which are what the shipped checkpoint (`outputs/checkpoints/phase4_model.pt`) and its reported results were produced from. If you just want to reproduce/inspect the shipped model, skip this whole step and use the checked-in splits as-is; only re-run `extract`/`split` if you're rebuilding the dataset from scratch (e.g. with a different `subsample_size` in `configs/default.yaml`), and know that doing so invalidates the shipped checkpoint's numbers until you retrain on the new splits.
2. **Cache CLIP embeddings** (frozen backbone, one-time cost — this is the expensive step, ~1.5-2 hours on CPU for the full train split with augmentation):
   ```bash
   python -m src.cache_embeddings --split train
   python -m src.cache_embeddings --split val
   python -m src.cache_embeddings --split test
   python -m src.cache_quick_robustness
   ```
3. **Train the linear head** (seconds, since it trains on cached embeddings, not raw images):
   ```bash
   python -m src.train
   ```
4. **Evaluate:**
   ```bash
   python -m src.robustness_eval --labeled-dir <folder with REAL/ and FAKE/ subfolders> --checkpoint outputs/checkpoints/phase4_model.pt
   ```

`src/toy_train.py` is a separate, much smaller pipeline-plumbing check against CIFAKE — useful for verifying the environment works end-to-end in minutes, not part of the real training path.

## Limitations and what we'd improve with more time

These are drawn from concrete diagnostics run during this project, not generic caveats — see `outputs/error_analysis.md` and `PLAN.md`'s changelog for the full evidence behind each one.

**The model is systematically underconfident under distortion and distribution shift, not symmetrically wrong.** Every degraded/out-of-distribution condition we tested shows the same pattern: AUC barely moves while fixed-threshold (0.5) accuracy drops substantially. On the worst stacked-distortion test-split cell, all 139 misclassifications (out of a 500-image sample) were false negatives — the model never mistakes a degraded real photo for fake, it only fails to catch degraded fakes. We tested whether this was simply a poorly-placed cutoff: swept the threshold on the validation split and found 0.64 maximizes accuracy there (99.07% vs. 98.85% at 0.5), then re-ran the full robustness and CIFAKE evaluations at 0.64. The result rejects the simple fix — clean/validation improved only trivially (+0.07 to +0.22pt), while every degraded and out-of-distribution condition got *worse*, with damage scaling with severity: held-out Gaussian noise at the highest severity dropped 4.2pt, the worst stacked chain dropped 3.8pt, CIFAKE dropped 1.0pt. This confirms the underlying issue is the whole confidence distribution shifting under distortion, not a poorly-placed cutoff — a single *global* threshold can't fix that, whichever direction you move it. We also tried temperature scaling (fitting a single scalar to divide the logits by, T=0.8143 on validation): zero accuracy change on every condition, expected by construction — a pure scale can never flip a prediction across a fixed threshold, it only rescales confidence, it doesn't shift it.

**So we tested the natural next question: is the gap fixable at all with a threshold, just not a single global one?** For each condition, we split its images in half, found the accuracy-maximizing threshold on one half (an oracle — using knowledge a real inference call doesn't have), and measured accuracy on the *other* half at that threshold, to avoid tuning and testing on the same data. The answer is yes, decisively: oracle per-condition thresholds recover most of the lost accuracy on every hard condition — held-out Gaussian noise at the highest severity goes from 81.8% to 97.3% (+15.5pt), the worst stacked chain from 70.9% to 94.1% (+23.2pt), and even CIFAKE (out-of-distribution) from 51.3% to 72.0% (+20.7pt). This confirms the leading hypothesis: the gap really is substantially about calibration, not discrimination ability — it just isn't fixable with one number for all conditions.

**Important caveat, stated plainly:** this is an upper-bound diagnostic, not something we can ship. `infer.py` has no way to know at inference time which distortion (or none) a given image has been through, so it can't select the matching per-condition threshold. **What we'd do:** build an actual degradation-aware calibration mechanism — e.g. a lightweight auxiliary classifier that estimates distortion type/severity from the image and selects (or interpolates) a threshold accordingly — which is real future work, not a change we could make today.

**A meaningful chunk of the cross-generator gap is actually a resolution artifact, not a generator-generalization failure.** We measured 52.3% accuracy / 0.8214 AUC on CIFAKE (Stable Diffusion, different real-photo source than training). To find out how much of that was really about the different generator, we took SID_Set test images the model gets 100% right and degraded them with the same resolution round-trip CIFAKE images went through natively (no generator change) — accuracy on those same images dropped to 72.3% from resolution alone. So roughly 28 of the ~48 accuracy points lost on CIFAKE are a resolution-sensitivity issue the model also has *in-distribution*, and only the remaining ~20 points are attributable to the generator/source shift itself. **What we'd do:** add the harsher resolution-degradation path (native → 32×32 → upscale, not just the milder 0.5x/0.25x Resize already in the training augmentation) to training augmentation directly.

**We evaluated the CIFAKE cross-generator check as a stand-in for the brief's own WildFake COCO-vs-DALL·E demonstration subset**, which we didn't have access to in this environment — CIFAKE differs from that intended benchmark on both axes the check is meant to test (Stable Diffusion vs. DALL·E, CIFAR-10 vs. COCO), so it's a weaker, narrower signal than the brief's own subset would give. Worth re-running with the real subset if it becomes available.

**We compute-constrained our way out of two planned experiments, both closed out with quantified reasons rather than left silently undone:**
- *DINOv2 backbone comparison* (motivated by published evidence it may generalize better than CLIP to messy, real-world images): DINOv2 ViT-L/14 measured ~63x slower per image than CLIP on our CPU-only hardware (2.835 sec/image vs. 0.045), driven by its 518×518 input (~1370 tokens/image) vs. CLIP's 224×224 (~50 tokens) — a property of the model family, not something a smaller DINOv2 variant would fix. Even a reduced-scope comparison projected to ~2.2 hours, over our budget for this hackathon.
- *LoRA fine-tuning* (motivated by published +12.72% accuracy gains over frozen linear probes elsewhere): fully scoped and implemented, including verified checkpoint-and-resume for long unattended runs, but not run — the diagnostics above showed in-distribution robustness already near-ceiling (nothing to gain from more capacity there) and the cross-generator gap looking substantially like a calibration/resolution issue rather than a representation-capacity one, which LoRA wasn't a clearly targeted fix for. Adapting further into SID_Set's single-generator (FLUX) training distribution risked narrowing generalization further rather than fixing the actual problem.

Both would be reasonable next experiments with GPU access.

**Training data covers a single generator (FLUX).** SID_Set's synthetic side is entirely FLUX-generated. The cross-generator check (above) is the honest test of whether this matters, and the answer is: partially — some real generalization exists, but it's weaker and less confident than in-distribution performance suggests.

**This model does not detect local edits/tampering, and we checked that empirically rather than just assuming it from the literature.** Ran the unmodified checkpoint on 300 tampered SID_Set images (never trained or evaluated on): only ~2% scored above the 0.5 threshold, and the score distribution (median P(fake) ≈ 0.0009) sits in the same near-zero range as genuine real photos, nowhere close to the ~1.0 typical of fully-generated fakes. This confirms the Section 0 scoping decision to exclude tampered images rather than mix two different problems — the model genuinely can't do this, not just "wasn't tested for it."

## Team contributions

Solo project — Ariff Muhammed designed, built, and ran the full pipeline: data preparation, model architecture and training, the robustness evaluation harness, error analysis, and this documentation.
