## Inspiration

Track 5 asks for robust detection of AI-generated images — robust specifically to the kind of post-processing real images undergo once they're shared online (compression, resizing, cropping, noise). A detector that only works on pristine images isn't solving the actual problem users face on social platforms.

## What it does

A binary classifier that scores any image with a confidence value for "AI-generated" — takes a directory of images, outputs `{"image_path", "pred"}` JSON with `pred` as a raw P(fake) score in [0,1]. I deliberately scoped this to fully-generated-image detection, explicitly excluding locally-edited/tampered images as a separate, harder problem — published evidence shows the same class of detector (CLIP-linear-probe) scores ~93% AP on fully-generated images but only ~73% AP on local manipulations (Smeu et al., WACV 2025); mixing the two problems would have hurt both.

**Results:** 99.15% accuracy / 0.9997 AUC on a held-out SID_Set test split (clean images). Robustness holds up almost perfectly across every trained-on transform and even the held-out Resize transform (never seen during training) — accuracy barely moves. The one real robustness weak point is held-out Gaussian Noise at high severity (accuracy drops to 83% at σ=0.10, though AUC stays at 0.998) and stacked worst-case distortions (a beyond-spec test I added myself, chaining resize→noise→JPEG: accuracy drops to 71.5%). I also went further than required and tested cross-generator generalization on CIFAKE (a different generator, Stable Diffusion, and a different real-photo source, CIFAR-10): 52.3% accuracy, 0.82 AUC — then isolated *why*, finding roughly half of that accuracy drop is a resolution-degradation artifact shared with in-distribution data, not specific to the generator change.

## How I built it

**Architecture:** a frozen CLIP ViT-B/32 vision encoder (`open_clip`, OpenAI-pretrained) with a single trainable linear layer on top — 151.3M parameters total, only 513 of them trainable. I chose this over training a CNN from scratch based on a published cross-generator comparison: CLIP+linear-probe was the only method tested without a "blind spot" generator family (consistent ~93% mAP on both GAN and diffusion-generated images), while a plain CNN and a reconstruction-based detector (DIRE) each had large family-specific failure modes (Ojha et al., CVPR 2023; MoLE, arXiv 2404.04883).

**Training data:** [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) — real photos from OpenImages V7, synthetic images from FLUX, photo-realistic and social-media-framed. 17,945 images subsampled and deduplicated via perceptual hashing (to prevent near-duplicate leakage across splits), split 70/15/15.

**Robustness strategy:** I trained with JPEG compression, Gaussian blur, color jitter, and center crop applied to *every* training image, at every severity level in the brief's grid — but deliberately held out Resize and Gaussian Noise entirely from training. This means my reported robustness numbers on those two transforms are genuine generalization evidence, not memorized augmentation.

**A real bug I caught before it mattered:** while building the raw dataset, I measured native image resolution across the real/synthetic classes before doing any further processing — every synthetic image turned out to be identically 1024×1024 (zero variance), while real images varied across 817 distinct resolutions. Left unfixed, the model could have learned "exactly 1024×1024 → synthetic" as a free shortcut that would only surface as a false robustness failure once the held-out Resize transform was tested. I normalized every image, both classes, to a common resolution before feature extraction to close this off.

**Feature caching:** since the backbone is frozen, I cache CLIP embeddings once (130,994 total: 10 versions per training image — 1 clean + 9 augmented combinations — plus clean versions of val/test) rather than re-running the backbone every training epoch. This turned a CPU-only training run into a seconds-per-epoch problem instead of an hours-per-epoch one.

## Challenges I ran into

- **CPU-only compute** shaped several decisions honestly rather than silently: I benchmarked a DINOv2 backbone comparison (motivated by published evidence it may generalize better to real-world images) and measured it at ~63x slower per image than CLIP, driven by its much higher input resolution — closed that experiment out with the actual numbers rather than leaving it as a vague "didn't have time."
- **A LoRA fine-tuning experiment** (motivated by published gains over frozen linear probes) was fully built and verified — including checkpoint-and-resume for long unattended training, tested with an actual mid-run kill-and-resume — but I chose not to run it once my diagnostics showed in-distribution robustness was already near-ceiling and the actual generalization gap looked like a calibration issue, not a capacity one. LoRA wasn't a clearly targeted fix for the problem I'd actually found.
- **Unattended background jobs got interrupted** by the environment mid-run more than once during feature caching. I built the caching pipeline with shard-based, resumable checkpointing (atomic writes, skip-what's-done-on-restart) specifically so these interruptions cost minutes, not hours of lost compute.
- **The calibration gap took three rounds of testing to actually understand.** A single global threshold shift made every degraded/out-of-distribution condition worse, and temperature scaling turned out to be mathematically incapable of changing accuracy at a fixed threshold at all — both dead ends, chased to their real conclusion instead of stopped at the first negative result.

## Accomplishments that I'm proud of

**The oracle per-condition threshold result — the strongest finding of the whole project, and one I verified twice before trusting it.** Instead of one global cutoff, I tested what happens if the classification threshold is tuned separately *for each specific distortion condition* — fit on one half of that condition's images, evaluated on the other half, so there's no circularity between tuning and testing. The result: oracle per-condition thresholds recover most of the accuracy lost on every hard condition — +15.45pt on held-out Gaussian Noise at max severity, +23.18pt on the worst stacked distortion chain, and +20.67pt on CIFAKE, the out-of-distribution cross-generator test. On the easy conditions (already near-ceiling), the effect is within noise, exactly as expected. This decisively confirms that the accuracy loss under distortion is substantially a calibration problem, not a discrimination-ability failure in the underlying representation.

I didn't just report this once — I later rebuilt the check as a saved, reusable script and reran it independently on a different night. All 17 robustness-condition numbers reproduced *exactly* (down to the same +23.18pt and +15.45pt figures), which is the kind of result I trust enough to lead a submission with.

I also held myself to a strict honesty standard throughout: every one of these results is framed explicitly as what it is — the oracle result is stated plainly as an upper-bound diagnostic, not a deployable fix, since `infer.py` has no way to know a test image's true distortion condition at inference time. When a follow-up attempt to actually reach that ceiling with cheap image heuristics came back as a flat null on a realistic test, I reported that as plainly as the positive results.

## What I learned

The calibration investigation taught me to distinguish three genuinely different questions that are easy to conflate: *is the threshold wrong* (tested with a global shift — no, moving it in either direction just trades one error type for another), *is the model's confidence miscalibrated* (tested with temperature scaling — provably not fixable this way, since a pure logit scale can never flip a prediction across a fixed threshold), and *is there recoverable signal the model already has but a single fixed rule can't access* (tested with oracle per-condition thresholds — yes, substantially). Only the third framing turned out to be right, and I would not have found that without ruling out the first two on real evidence rather than intuition.

I also learned that a "negative" result and an "inconclusive" result look identical if you don't check your own methodology's assumptions — I caught this directly when a follow-up experiment (a trained classifier meant to replace hand-built heuristics as the calibration router) initially reported an accuracy regression, but inspecting the actual fitted thresholds showed a sample-size bug had produced degenerate values, not a real finding. Building a sanity check that refuses to report a result on top of an unverified assumption — and then honoring that check even when I wanted a clean number — was worth more than any single accuracy figure.

## What's next

- **A degradation-aware calibration mechanism** — the oracle result establishes there's a real ceiling worth reaching; the next step is something that can estimate an incoming image's likely distortion condition and route to an appropriate threshold or calibration adjustment, rather than a single fixed rule. An early attempt at this with cheap, training-free pixel heuristics (blur/noise/JPEG-artifact estimators) didn't move the needle on a realistic test, so the more promising direction is something actually trained on labeled degradation data.
- **The DINOv2 backbone comparison**, benchmarked but not completed on CPU-only hardware — a legitimate open question with GPU access, motivated by published evidence it may generalize better than CLIP to messy, real-world images.
- **The brief's own WildFake COCO-vs-DALL·E cross-generator subset**, substituted with CIFAKE in this environment due to not having access to the brief's provided file — worth re-running the cross-generator check against the real intended benchmark if it becomes accessible.
- **Resolution-robustness training** specifically targeting the harsher native-resolution degradation isolated in the CIFAKE error analysis, which the current Resize augmentation (only down to 0.25x before upscaling back) doesn't fully cover.

## Tools, models, libraries, and datasets used

- **Model:** CLIP ViT-B/32 (`open_clip`, OpenAI pretrained weights)
- **Libraries:** PyTorch, `open_clip_torch`, `timm` (for the DINOv2 comparison attempt), `peft` (for the LoRA scaffolding), scikit-learn, `imagehash` (pHash deduplication), Pillow, pandas
- **Primary dataset:** SID_Set (Huang et al., CVPR 2025)
- **Secondary/diagnostic dataset:** CIFAKE (Stable Diffusion-based, used only for Phase-1 pipeline validation and the cross-generator generalization check — never trained on)
- **Reference/demo-only dataset:** the brief's own WildFake COCO-vs-DALL·E subset — intended as my primary cross-generator check per the brief's framing, substituted with CIFAKE in this environment due to not having access to the brief's provided file; flagged explicitly as a substitute of convenience, not treated as equivalent
- **Development tools used:** Python 3.12.10, VS Code, Claude Code as the primary dev environment, on a CPU-only local machine (no GPU access during this project — see "What's next" for what that ruled out)
