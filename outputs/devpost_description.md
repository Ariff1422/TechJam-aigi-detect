# Devpost Written Project Description (Draft)

_Trimmed and edited from `PLAN.md`'s changelog, per the brief's Section 5.5 requirements. Fill in team name/inspiration framing as needed — the technical content below is complete and sourced from real results, not placeholder text._

## Inspiration / Problem

Track 5 asks for robust detection of AI-generated images — robust specifically to the kind of post-processing real images undergo once they're shared online (compression, resizing, cropping, noise). A detector that only works on pristine images isn't solving the actual problem users face on social platforms.

## What it does

A binary classifier that scores any image with a confidence value for "AI-generated" — takes a directory of images, outputs `{"image_path", "pred"}` JSON with `pred` as a raw P(fake) score in [0,1]. We deliberately scoped this to fully-generated-image detection, explicitly excluding locally-edited/tampered images as a separate, harder problem — published evidence shows the same class of detector (CLIP-linear-probe) scores ~93% AP on fully-generated images but only ~73% AP on local manipulations (Smeu et al., WACV 2025); mixing the two problems would have hurt both.

## How we built it

**Architecture:** a frozen CLIP ViT-B/32 vision encoder (`open_clip`, OpenAI-pretrained) with a single trainable linear layer on top — 151.3M parameters total, only 513 of them trainable. We chose this over training a CNN from scratch based on a published cross-generator comparison: CLIP+linear-probe was the only method tested without a "blind spot" generator family (consistent ~93% mAP on both GAN and diffusion-generated images), while a plain CNN and a reconstruction-based detector (DIRE) each had large family-specific failure modes (Ojha et al., CVPR 2023; MoLE, arXiv 2404.04883).

**Training data:** [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) — real photos from OpenImages V7, synthetic images from FLUX, photo-realistic and social-media-framed. 17,945 images subsampled and deduplicated via perceptual hashing (to prevent near-duplicate leakage across splits), split 70/15/15.

**Robustness strategy:** we trained with JPEG compression, Gaussian blur, color jitter, and center crop applied to *every* training image, at every severity level in the brief's grid — but deliberately held out Resize and Gaussian Noise entirely from training. This means our reported robustness numbers on those two transforms are genuine generalization evidence, not memorized augmentation.

**A real bug we caught before it mattered:** while building the raw dataset, we measured native image resolution across the real/synthetic classes before doing any further processing — every synthetic image turned out to be identically 1024×1024 (zero variance), while real images varied across 817 distinct resolutions. Left unfixed, the model could have learned "exactly 1024×1024 → synthetic" as a free shortcut that would only surface as a false robustness failure once the held-out Resize transform was tested. We normalized every image, both classes, to a common resolution before feature extraction to close this off.

**Feature caching:** since the backbone is frozen, we cache CLIP embeddings once (130,994 total: 10 versions per training image — 1 clean + 9 augmented combinations — plus clean versions of val/test) rather than re-running the backbone every training epoch. This turned a CPU-only training run into a seconds-per-epoch problem instead of an hours-per-epoch one.

## Results

- **99.15% accuracy / 0.9997 AUC** on a held-out SID_Set test split, clean images.
- Robustness holds up almost perfectly across every trained-on transform and even the held-out Resize transform (never seen during training) — accuracy barely moves.
- The one real robustness weak point: held-out Gaussian Noise at high severity (accuracy drops to 83% at σ=0.10, though AUC stays at 0.998) and stacked worst-case distortions (a beyond-spec test we added ourselves, chaining resize→noise→JPEG: accuracy drops to 71.5%).
- We went further than required and tested cross-generator generalization on CIFAKE (a different generator, Stable Diffusion, and a different real-photo source, CIFAR-10): 52.3% accuracy, 0.82 AUC. We then isolated *why* — a targeted diagnostic showed roughly half of that accuracy drop is a resolution-degradation artifact shared with in-distribution data, not specific to the generator change, and the AUC-vs-accuracy gap throughout points to a calibration problem (the model's relative ranking of real-vs-fake holds up far better than its absolute confidence does under distortion).

## Challenges we ran into

- **CPU-only compute** shaped several decisions honestly rather than silently: we benchmarked a DINOv2 backbone comparison (motivated by published evidence it may generalize better to real-world images) and measured it at ~63x slower per image than CLIP, driven by its much higher input resolution — closed that experiment out with the actual numbers rather than leaving it as a vague "didn't have time."
- **A LoRA fine-tuning experiment** (motivated by published gains over frozen linear probes) was fully built and verified — including checkpoint-and-resume for long unattended training, tested with an actual mid-run kill-and-resume — but we chose not to run it once our diagnostics showed in-distribution robustness was already near-ceiling and the actual generalization gap looked like a calibration issue, not a capacity one. LoRA wasn't a clearly targeted fix for the problem we'd actually found.
- **Unattended background jobs got interrupted** by the environment mid-run more than once during feature caching. We built the caching pipeline with shard-based, resumable checkpointing (atomic writes, skip-what's-done-on-restart) specifically so these interruptions cost minutes, not hours of lost compute.

## Tools, models, libraries, and datasets used

- **Model:** CLIP ViT-B/32 (`open_clip`, OpenAI pretrained weights)
- **Libraries:** PyTorch, `open_clip_torch`, `timm` (for the DINOv2 comparison attempt), `peft` (for the LoRA scaffolding), scikit-learn, `imagehash` (pHash deduplication), Pillow, pandas
- **Primary dataset:** SID_Set (Huang et al., CVPR 2025)
- **Secondary/diagnostic dataset:** CIFAKE (Stable Diffusion-based, used only for Phase-1 pipeline validation and the cross-generator generalization check — never trained on)
- **Reference/demo-only dataset:** the brief's own WildFake COCO-vs-DALL·E subset — intended as our primary cross-generator check per the brief's framing, substituted with CIFAKE in this environment due to not having access to the brief's provided file; flagged explicitly as a substitute of convenience, not treated as equivalent
