# Error Analysis Note

Model: frozen CLIP ViT-B/32 + linear probe (`outputs/checkpoints/phase4_model.pt`), trained on SID_Set (FLUX-generated / OpenImages real). Threshold: 0.5. All numbers below are from the real held-out SID_Set test split (2,692 images) unless noted otherwise.

## 1. Clean performance is essentially perfect, in-distribution

On a random 500-image sample from the test split, only **1 misclassification** occurred at threshold 0.5: a real photo (`data/raw/real/5438c4ffef84ea15.jpg`) scored 0.598 — just over the boundary. There is no systematic clean-condition failure mode to report; errors only appear once distortion is introduced.

## 2. The dominant failure mode: false negatives under distortion, never false positives

Across every degraded condition tested — held-out Gaussian Noise, and especially the worst-case stacked chain (resize 0.5x → noise σ=0.10 → JPEG q=50) — **misclassifications are overwhelmingly or exclusively false negatives** (fake images predicted as real), not false positives.

On the stacked worst-case chain (500-image sample): **139/500 misclassified, and every single one was a false negative (139 FN, 0 FP)**. The model never mistakes a degraded real photo for AI-generated; it only fails to catch degraded fakes.

**Hypothesis:** the linear probe's decision boundary sits on features that correlate with "clean, high-fidelity synthesis artifacts." Heavy distortion — especially the compound noise+resize+JPEG chain — degrades or removes exactly the signal the model relies on to call an image fake, pushing its confidence toward the "default" real-leaning region rather than toward false confidence in the wrong direction. This is a conservative failure mode (never confidently wrong about real photos) but it means recall on fakes degrades faster than precision does under real-world image degradation — a meaningfully different risk profile than a model that fails symmetrically.

Concrete examples (stacked chain, all true label = fake, predicted as real):
- `data/raw/synthetic/full_synthetic_066625.jpg` — pred_prob 0.034 (also failed at noise σ=0.10 alone: 0.075)
- `data/raw/synthetic/full_synthetic_012517.jpg` — pred_prob 0.012 (also failed at noise σ=0.10 alone: 0.033)
- `data/raw/synthetic/full_synthetic_001818.jpg` — pred_prob 0.261
- `data/raw/synthetic/full_synthetic_010962.jpg` — pred_prob 0.078

The first two images fail under noise alone *and* under the full stacked chain, suggesting these specific synthetic images sit close to the decision boundary even before compounding distortions — likely images with unusually photorealistic, low-artifact generations to begin with.

## 3. AUC holds up far better than accuracy under distortion — a calibration/confidence issue, not a ranking failure

This pattern recurs at every scale tested in this project:

| Condition | Accuracy | AUC |
|---|---|---|
| Clean (test split) | 99.15% | 0.9997 |
| Held-out Gaussian Noise, σ=0.10 (test split) | 82.99% | 0.9976 |
| Stacked worst-case chain (test split) | 71.51% | 0.9955 |
| Same-generator 32×32 resolution round-trip (289 high-confidence images) | 72.3% | 0.9903 |
| CIFAKE cross-generator (different generator + real source) | 52.3% | 0.8214 |

In every case, AUC degrades far more gently than accuracy at the fixed 0.5 threshold. This means the model's **relative ranking of real vs. fake survives distortion and distribution shift much better than its absolute confidence does** — errors are a calibration/confidence problem at least as much as a discrimination problem. A single global threshold of 0.5, calibrated on clean SID_Set data, is a poor fit for degraded or out-of-distribution inputs; a lower or adaptive threshold would likely recover meaningful accuracy without touching the underlying representation (see Section 6 discussion below).

**We tested this directly, and the simple version of the fix doesn't work.** Swept the classification threshold on the validation split: 0.64 maximizes accuracy there (99.07% vs. 98.85% at 0.5). Re-running the full robustness sweep and the CIFAKE check at 0.64 shows clean/validation improve only trivially (+0.07 to +0.22pt), while every degraded and out-of-distribution condition gets *worse*, with the damage scaling directly with severity — held-out Gaussian Noise at σ=0.10 drops 4.2pt, the worst stacked chain drops 3.8pt, CIFAKE drops 1.0pt. A single global threshold, in either direction, cannot fix this: the finding above isn't "0.5 happens to be in the wrong place," it's that the whole confidence distribution compresses toward zero under distortion, so raising the bar only makes already-underconfident true fakes harder to catch, with nothing to gain on the real-image side that wasn't already correctly classified. Per-condition calibration or temperature scaling — which reshape the distribution itself rather than move one fixed line through it — remains the more promising direction.

## 4. The CIFAKE cross-generator gap decomposes into two separate, additive effects — not one

This is the most important finding for understanding *why* the model doesn't generalize as well as in-distribution numbers suggest, and it required a targeted diagnostic to see clearly (a naive read of the raw 52.3%/0.82 CIFAKE numbers alone would overstate how much of the gap is really about generator differences).

**The decomposition:** 289 SID_Set test images the model classifies with 100% accuracy and >0.9 confidence were degraded with the same resolution round-trip CIFAKE images natively went through (downscale to 32×32, upscale back), with everything else held fixed — same generator (FLUX), same real-photo source (OpenImages), only resolution changed.

| Condition | Accuracy | AUC | Mean P(fake) on true fakes |
|---|---|---|---|
| Clean (same 289 images) | 100.0% | 1.000 | — |
| Resolution round-trip only (same images, same generator) | 72.3% | 0.9903 | 0.434 |
| CIFAKE (different generator, different real source) | 52.3% | 0.8214 | 0.098 |

**Roughly 28 points of the clean→CIFAKE accuracy gap is a resolution/quality-degradation artifact that has nothing to do with cross-generator generalization** — the same model, same generator, same training distribution, loses that much accuracy purely from a resolution round-trip. A further ~20 points is attributable to genuine cross-generator/cross-real-source effects once resolution is controlled for.

**Implication for reporting:** the raw CIFAKE number should never be presented as a pure measurement of "how well does this model generalize to a new generator" — a meaningful fraction of that number reflects a resolution-sensitivity issue the model also has *in-distribution* (native-resolution imbalance between SID_Set's real and synthetic images was flagged and mitigated at the raw-data level in Phase 3, but the model's *sensitivity* to resolution changes at inference time, as opposed to memorizing raw native resolution as a shortcut, is evidently still present and separate).

## 5. What this suggests for "what you'd improve with more time"

1. **Threshold calibration**, not further fine-tuning, is the most promising near-term lever: since AUC survives distortion/distribution-shift far better than the fixed-threshold accuracy does, a distortion-aware or per-condition threshold (or a calibration layer, e.g. temperature scaling on the logits) could likely recover a meaningful fraction of the accuracy lost under noise, the stacked chain, and CIFAKE — without touching the frozen backbone or retraining the head.
2. **Resolution-robustness training** specifically (not just the held-out Resize transform already tested, which the model handles well) — since the round-trip degradation used here (native → 32×32 → upscale) is a harsher, more information-destructive operation than the Resize transform in Section 3.1 (which only goes to 0.5x/0.25x before upscaling back), adding this specific harsher degradation to training augmentation could plausibly close part of the resolution-driven portion of the CIFAKE gap directly.
3. **LoRA fine-tuning (Phase 4b) was considered and deliberately not pursued** given these findings — in-distribution robustness is already near-ceiling (nothing here needs more capacity), and LoRA fine-tuning further into the single-generator SID_Set distribution is not an obviously targeted fix for either the calibration issue (item 1) or the resolution-sensitivity issue (item 2), both of which look more like data/calibration problems than representation-capacity problems.
4. **DINOv2 backbone comparison (Section 1.1b) was attempted and dropped for a quantified, hardware-specific reason** (~63x slower per image than CLIP on CPU-only hardware, driven by DINOv2's much higher input resolution/token count — not something a smaller variant fixes) rather than a finding against DINOv2 itself. This remains a legitimate open question for anyone with GPU access, and PROBE's published result (DINOv2 substantially outperforming CLIP on in-the-wild benchmarks specifically) is suggestive enough that it would be a natural next experiment if compute allows.
