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

In every case, AUC degrades far more gently than accuracy at the fixed 0.5 threshold. This means the model's **relative ranking of real vs. fake survives distortion and distribution shift much better than its absolute confidence does** — errors are a calibration/confidence problem at least as much as a discrimination problem. We tested this claim directly, in three stages, immediately below — the result confirms it, with an important caveat about what's actually deployable.

**We tested this directly, in three increasingly targeted ways, and the results together are the headline finding of this whole error analysis.**

*Global threshold shift:* swept the classification threshold on the validation split — 0.64 maximizes accuracy there (99.07% vs. 98.85% at 0.5). Re-running the full robustness sweep and the CIFAKE check at 0.64 shows clean/validation improve only trivially (+0.07 to +0.22pt), while every degraded and out-of-distribution condition gets *worse*, with the damage scaling directly with severity — held-out Gaussian Noise at σ=0.10 drops 4.2pt, the worst stacked chain drops 3.8pt, CIFAKE drops 1.0pt. A single global threshold, in either direction, cannot fix this.

*Temperature scaling:* fit T=0.8143 on validation logits by minimizing NLL, then re-scored every cached robustness condition and CIFAKE at the fixed 0.5 threshold — **zero accuracy change on every single condition, without exception.** Mathematically expected: a pure scale on logits can't change which side of a fixed threshold (logit=0, prob=0.5) any prediction falls on, so it can't move accuracy at a fixed cutoff no matter how well it improves calibration in the strict NLL sense (which it did, marginally: 0.0289→0.0279).

*Oracle per-condition thresholds — the decisive test.* For each condition, split its images in half (fixed seed), found the accuracy-maximizing threshold on one half, evaluated on the *other* half at that threshold (avoiding the circularity of tuning and testing on the same data). **Result: oracle per-condition thresholds recover most of the lost accuracy on every hard condition:**

| Condition | acc@0.5 (global) | acc@oracle (per-condition) | Gain |
|---|---|---|---|
| Resize 0.5x (held-out) | 99.03% | 98.96% | -0.07pt |
| Resize 0.25x (held-out) | 99.03% | 98.51% | -0.52pt |
| Gaussian Noise σ=0.02 (held-out) | 95.91% | 98.96% | +3.05pt |
| Gaussian Noise σ=0.05 (held-out) | 88.93% | 98.14% | +9.21pt |
| Gaussian Noise σ=0.10 (held-out) | 81.80% | 97.25% | +15.45pt |
| Stacked chain (resize/blur/jpeg) | 88.86% | 97.25% | +8.40pt |
| Stacked chain (resize/noise/jpeg, worst) | 70.88% | 94.06% | **+23.18pt** |
| CIFAKE (out-of-distribution) | 51.33% | 72.00% | **+20.67pt** |

**This confirms the leading hypothesis decisively: the gap really is substantially a calibration problem, not a discrimination-ability problem — it just isn't fixable with one global number.** On easy conditions (Resize, already near-ceiling), the oracle threshold makes no meaningful difference, as expected. On every hard condition, including out-of-distribution CIFAKE, it recovers the large majority of the lost accuracy.

**Critical caveat, stated plainly: this is an oracle upper-bound diagnostic, not a deployable fix.** `infer.py` has no way to know at inference time which distortion (or none) an incoming image has been through, so it cannot select the matching per-condition threshold — the numbers above represent a ceiling on what calibration alone could achieve, not something achievable today. What this motivates is a real degradation-aware calibration mechanism (e.g. a lightweight auxiliary classifier estimating distortion type/severity, feeding into an adaptive threshold or calibration layer) — genuine future work, not a change made in this project.

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

1. **A degradation-aware calibration mechanism is the most promising near-term lever, and we now have direct evidence it would work, not just a hypothesis.** Section 3 tested the calibration hypothesis three ways: a global threshold shift (rejected — damage scales with severity), temperature scaling (mathematically inert at a fixed threshold, confirmed with zero accuracy change on every condition), and oracle per-condition thresholds (recovers +15 to +23pt on the hardest conditions, including CIFAKE). The oracle result is the key one: it proves the accuracy loss under distortion is substantially recoverable through calibration alone, without touching the frozen backbone or retraining the head — but only if the right per-condition threshold can be selected at inference time, which nothing in the current pipeline does. The concrete next step is a small auxiliary model that estimates distortion type/severity from an image and selects (or interpolates) an appropriate threshold or calibration adjustment accordingly — not a single global scale or shift, which are now both ruled out empirically. **We also tried a cheap, training-free version of that auxiliary model as a fast path:** a 2-bin router built from three pixel heuristics (Laplacian variance for blur, high-frequency residual energy for noise, block-edge energy for JPEG), each showing real, monotonic separation between clean and degraded images in isolation. On a realistic mixed-condition test set (true distortion unknown to the router, n=400, non-degenerate routing verified at real scale) it produced no net accuracy gain — 98.00% vs. 98.00% for fixed-0.5. This doesn't contradict the oracle finding; it shows a simple heuristic router isn't sufficient to reach the ceiling the oracle test found, sharpening rather than weakening the case that closing this gap needs an actually-trained degradation classifier, not hand-built pixel signals.
2. **Resolution-robustness training** specifically (not just the held-out Resize transform already tested, which the model handles well) — since the round-trip degradation used here (native → 32×32 → upscale) is a harsher, more information-destructive operation than the Resize transform in Section 3.1 (which only goes to 0.5x/0.25x before upscaling back), adding this specific harsher degradation to training augmentation could plausibly close part of the resolution-driven portion of the CIFAKE gap directly.
3. **LoRA fine-tuning (Phase 4b) was considered and deliberately not pursued** given these findings — in-distribution robustness is already near-ceiling (nothing here needs more capacity), and LoRA fine-tuning further into the single-generator SID_Set distribution is not an obviously targeted fix for either the calibration issue (item 1) or the resolution-sensitivity issue (item 2), both of which look more like data/calibration problems than representation-capacity problems.
4. **DINOv2 backbone comparison (Section 1.1b) was attempted and dropped for a quantified, hardware-specific reason** (~63x slower per image than CLIP on CPU-only hardware, driven by DINOv2's much higher input resolution/token count — not something a smaller variant fixes) rather than a finding against DINOv2 itself. This remains a legitimate open question for anyone with GPU access, and PROBE's published result (DINOv2 substantially outperforming CLIP on in-the-wild benchmarks specifically) is suggestive enough that it would be a natural next experiment if compute allows.
5. **A 2-layer MLP head (`embedding_dim -> 256 -> 1`, ReLU, dropout 0.3) was trained and compared against the shipped linear probe, on the same cached embeddings.** Result: small but consistent gains exactly where the linear probe was weakest (held-out Gaussian Noise sigma=0.10: +0.90pt, worst stacked chain: +2.05pt) and negligible everywhere it was already strong — but a small CIFAKE regression (-1.50pt). We kept the linear probe as the shipped model: the MLP's gains are real but narrowly in-distribution and don't touch the actual harder problem (the cross-generator gap), and a checkpoint swap this close to final packaging wasn't worth the risk for a marginal, scope-limited improvement. Worth revisiting with more time, ideally alongside the shift-term calibration approaches above rather than instead of them.

## 6. The scope boundary (excluding tampered/local-manipulation images) was checked empirically, not just asserted

Section 0 excludes tampered/locally-edited images from this model's scope by design, citing published evidence that CLIP-linear-probe methods drop from ~93% AP on fully-generated images to ~73% AP on local manipulations (Smeu et al., WACV 2025). We verified this holds for our actual checkpoint rather than taking it on faith: ran the unmodified Phase 4 model on 300 tampered SID_Set images (extracted from the already-downloaded shards, never trained or evaluated on elsewhere). Result: only ~2% scored above the 0.5 threshold, and the score distribution (median P(fake) ≈ 0.0009, mean ≈ 0.048) sits in the same near-zero range as genuine real photos (median ≈ 0.0001, mean ≈ 0.019) — nowhere close to the fully-generated fake distribution (median ≈ 0.9996, mean ≈ 0.986). The model doesn't misclassify tampered images inconsistently; it consistently treats them as real. This confirms the scoping decision was correct, not just conveniently assumed, and rules out "maybe it picks up local edits a little by accident" as a hidden capability worth claiming credit for.
