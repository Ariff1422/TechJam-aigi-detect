# Demo Script Outline

Target total length: **~3-4 minutes**. Record last, once the pipeline is stable (per PLAN.md Phase 6) — this outline assumes everything currently in the repo works as-is.

---

## 1. Problem framing (30s)

**Say:** "We built a detector for fully AI-generated images that's specifically robust to the kind of degradation real images go through once they're shared online — compression, resizing, cropping, noise. A detector that only works on pristine images doesn't solve the real problem."

**Show:** nothing yet, or a title slide / the README's opening paragraph.

## 2. Architecture, briefly (30s)

**Say:** "Frozen CLIP vision encoder, one trainable linear layer on top — 151 million parameters total, but only 513 of them actually trainable. We picked this because published results show it's the one architecture that doesn't have a blind spot across different generator families, unlike a CNN trained from scratch."

**Show:** README's Overview section, or `src/model.py`'s `LinearHead` class (it's genuinely just one `nn.Linear` — visually makes the "513 trainable params" claim credible in about 2 seconds).

## 3. Live inference demo (60-90s) — the core of the demo

**Do this live, don't pre-record it:**

```bash
python -m src.infer --image-dir <a small folder with a few real + a few fake images> --checkpoint outputs/checkpoints/phase4_model.pt --output outputs/predictions.json
cat outputs/predictions.json
```

**Prep beforehand:** create a small demo folder (5-10 images) mixing a few real photos and a few `data/raw/synthetic/full_synthetic_*.jpg` images, so the output JSON visibly shows low `pred` scores for real images and high `pred` scores for fake ones. Pick images you've spot-checked get classified correctly, so the live demo doesn't accidentally show a misclassification without you being ready to explain it (if it does — that's fine too, see step 5).

**Say while the command runs:** "This is the actual required deliverable — point it at any folder, get back a confidence score per image, not just a label."

**Show:** the terminal output and the resulting JSON — call out one clearly-real and one clearly-fake image's score explicitly.

## 4. Robustness under distortion (45-60s)

**Say:** "We didn't just test on clean images. We trained with four kinds of distortion — JPEG compression, blur, color jitter, cropping — applied to every training image. Then we tested on two *more* distortions the model never saw during training at all, to check it actually generalizes rather than just memorizing our augmentation recipe."

**Show:** `outputs/robustness_table_phase4.csv` opened in a spreadsheet/viewer, or the README's Results table. Point specifically at the held-out Resize row (barely moves from clean) as the headline result, then briefly acknowledge the held-out Gaussian Noise row (does degrade at high severity) as an honest limitation, not something to hide.

## 5. The honest part — error analysis (45-60s)

**Say:** "We went further than the brief requires and tested on a completely different dataset — different generator, different source of real photos. Accuracy dropped a lot. But instead of just reporting that number, we dug into *why*."

**Show:** the two-row comparison from `outputs/error_analysis.md` section 4 — same SID_Set images, clean vs. resolution-degraded (100% → 72.3%) vs. the actual cross-generator result (52.3%). "About half of that drop turned out to be a resolution artifact that has nothing to do with the different generator — we isolated that with a targeted test, not just eyeballed it."

**This section is the strongest evidence of genuine rigor** — most hackathon submissions report a number; showing you decomposed *why* the number is what it is is the differentiator. Don't rush it.

## 6. What we'd do next (15-20s)

**Say:** "Given more time or GPU access: calibrate the decision threshold per-condition rather than use one fixed cutoff — the model's ranking ability holds up much better than its fixed-threshold accuracy does under distortion. We also fully built and tested a LoRA fine-tuning path, including crash-safe checkpointing, but chose not to run it once we found the actual problem was calibration, not model capacity."

**Show:** nothing new — talking head or the README's Limitations section as a backdrop.

---

## Timing summary

| Section | Time | Cumulative |
|---|---|---|
| Problem framing | 0:30 | 0:30 |
| Architecture | 0:30 | 1:00 |
| Live inference | 1:00 | 2:00 |
| Robustness table | 0:50 | 2:50 |
| Error analysis | 0:50 | 3:40 |
| What's next | 0:20 | 4:00 |

If you need to cut for time, cut section 6 first (it's a natural closing line even if spoken quickly) and section 2 second (the architecture claim can be folded into section 1's opening line instead). Don't cut section 5 — it's the part that shows genuine investigation rather than a leaderboard number.
