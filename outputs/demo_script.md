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

## 6. What we'd do next (~30s)

**Say:** "We tested three ways to fix the calibration gap. A global threshold made it worse. Temperature scaling literally can't help — that's provable, not just observed. An oracle test — reproduced exactly on a second run — proved a real ceiling exists: up to 23 points recoverable. But our first attempt to actually reach that ceiling with cheap image heuristics, on a realistic unknown-distortion test, came back flat. We're reporting that honestly, not spinning it."

**Show:** nothing new — talking head or the README's Limitations section as a backdrop. If you want a visual, the oracle-vs-global-vs-heuristic-routed numbers table in PLAN.md's changelog or `error_analysis.md` Section 3 works, but this section is mostly a spoken summary.

*(This is the primary script for this section now — genuinely close to its ~30s budget and much lower-risk to deliver cleanly on a live take than the fuller version. It still lands the core point: three fixes tried, one confirmed a real ceiling twice over (the "reproduced exactly on a second run" detail is kept in this short version — it's cheap to say and it's the strongest evidence in the whole section), one honest null result. The LoRA mention is the one piece cut for time — it's in README/`error_analysis.md`/PLAN.md if you want to reference it in Q&A or show notes, it just doesn't need to be said out loud. If you have room to breathe and want the fuller version instead, here it is — it's the same material with more detail spoken rather than implied:*

*"We chased the calibration question as far as it goes. A single global threshold change made things worse, not better, once you get past clean data. Temperature scaling turned out to be mathematically incapable of fixing this at all — it rescales confidence but can't flip a prediction across a fixed cutoff. Then we tested the oracle case — if you knew exactly what distortion an image had gone through and picked the perfect threshold for it, you'd recover 15 to 23 points of accuracy on the hardest conditions, including on the cross-generator test. And this wasn't a one-off number — we reproduced it exactly from a second, independent script on a different night, so we trust it. That confirmed the gap really is about calibration, not discrimination ability. So we tried to actually build that — cheap pixel-level heuristics that estimate blur, noise, and JPEG artifacts from the image itself, no model needed, and route to a different threshold based on what they see. The heuristics themselves work — they cleanly separate clean from degraded images. But on a realistic test where the distortion is unknown, like real deployment, it didn't move the needle: same accuracy as just using 0.5. We're reporting that honestly rather than dressing it up — the ceiling is real, we just haven't built the mechanism that reaches it yet. Also fully built and tested a LoRA fine-tuning path, including crash-safe checkpointing, but chose not to run it once we confirmed the problem was calibration, not model capacity." (~45-50s at natural pace, not the primary plan.))*

---

## Timing summary

| Section | Time | Cumulative |
|---|---|---|
| Problem framing | 0:30 | 0:30 |
| Architecture | 0:30 | 1:00 |
| Live inference | 1:00 | 2:00 |
| Robustness table | 0:50 | 2:50 |
| Error analysis | 0:50 | 3:40 |
| What's next | 0:30 | 4:10 |

With section 6 now scripted to its ~30s shortened version as the primary read, the total lands around **4:10** — sections 1-5 alone already sum to 3:40, so 4:10 is the honest total once section 6 is added, not 3:40-3:50; getting under 4:00 needs a cut somewhere in sections 1-5, not just a shorter section 6. If you need to land under 4:00, cut section 2 first (fold the architecture claim into section 1's opening line, saving ~30s) — that gets you to ~3:40. Don't cut section 5; it's still the part that shows genuine investigation rather than a leaderboard number. Section 6 itself is already at its shortest reasonable form as scripted above; only reach for the fuller version in the note if you have *extra* time to spend, not as a way to save it.

## What's new since the last script revision (context for whoever records this)

Verified fresh against PLAN.md before recording: checkpoint (`phase4_model.pt`), threshold (0.5), and the robustness table are all unchanged since the calibration investigation described in section 6 above — nothing here is stale.

The calibration investigation itself: global threshold shift (rejected), temperature scaling (mathematically dead end, confirmed), and oracle per-condition thresholds (large recoverable ceiling confirmed — the headline finding, and independently reproduced exactly on a second run using a different script, on a different night, which is why section 6 now says so explicitly). A real attempt to reach that ceiling with training-free pixel heuristics came back a flat null, reported honestly. All of this is reflected in README, `error_analysis.md`, and `devpost_description.md`, and in this script.

**Deliberately NOT added to this script**, on top of the general "already-covered material stays covered" scoping: a follow-up session tried a heuristic router (98.00%, no net gain, matching the earlier null), a trained degradation-type classifier (74% held-in accuracy — a real result, but with an unresolved confusion pattern), and an attempted classifier-routed calibration (aborted mid-run by its own sanity check, no clean accuracy number produced). None of that is clean enough to present confidently on camera; it's logged in PLAN.md for anyone continuing the investigation, not in any of the three reader-facing deliverables. Also not added: a tampered-image check (confirmed the model correctly doesn't detect local edits, as scoped from the start) and an MLP-head comparison (small real gains, not adopted as the shipped model) — both already covered in README/`error_analysis.md` but not called out in this script, which stays scoped to the strongest material rather than a full history. Nothing about the shipped model, checkpoint, or its headline numbers (99.15% clean test accuracy, 0.9997 AUC) has changed.
