# Model Card — Instagram Opioid Classifier

A two-stage text-then-image classifier that labels Instagram posts as `opioid_related` or `non_opioid_related`. Used to triage a Sprinklr-collected corpus for downstream thematic analysis.

For running instructions see [`RUNBOOK.md`](RUNBOOK.md). For retraining the probe see [`TRAINING.md`](TRAINING.md).

---

## Intended use

- **Primary**: Bulk triage of Instagram posts already pre-filtered by the keyword "opioid" via Sprinklr, into two buckets for downstream human review and thematic coding.
- **Secondary**: Filtering out posts whose captions merely mention opioids in passing (news, policy, unrelated personal stories), so researchers can focus on visually or substantively opioid-related content.

### Out of scope

- **General opioid detection on arbitrary social media.** The model has only ever seen posts that already matched a Sprinklr keyword filter. Distribution shift on unfiltered data is unknown and likely severe.
- **Clinical, legal, or enforcement decisions.** This is a research triage tool, not a diagnostic or investigative system. No individual post label should be acted on without human review.
- **Fine-grained thematic classification** (e.g., "overdose prevention" vs "pill sales" vs "recovery"). Only binary opioid-vs-non-opioid is supported. Thematic classifiers are planned but not built.
- **Non-English captions.** Text classification uses distilbart-MNLI, which is English-only. Non-English captions will fall through to the image-only path.

---

## Inputs

A post directory containing:

- One image file (`.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp`)
- A `.txt` caption file with the same stem as the image (optional)
- A metadata file that contains number of likes, post-id, comment count etc. 

Hashtags (`#\w+`) are stripped from captions before text classification. Captions shorter than 3 characters after stripping are treated as missing.

## Outputs

Per post:

| Field | Values |
|---|---|
| `category` | `opioid_related` \| `non_opioid_related` \| `error` |
| `classification_method` | `text` \| `image` \| `error` |
| `status` | `processed` \| `processed_with_errors` \| `skipped` \| `skipped_exist` \| `error` |

Posts are uploaded to the matching Google Drive folder (`OPIOID_RELATED/`, `NON_OPIOID_CONTENT/`, `ERRORS/`) with all original files preserved.

---

## Pipeline architecture

```
┌────────────────────┐
│  Caption (.txt)    │──┐
└────────────────────┘  │                   TEXT STAGE
                        ▼
              ┌─────────────────────────┐
              │ distilbart-mnli-12-3    │
              │ zero-shot: {opioid,     │
              │          non_opioid}    │
              └─────────────┬───────────┘
                            │ confidence = |p1 − p2|
                            │
          ┌─────────────────┼─────────────────┐
          │                                   │
     conf ≥ 0.9                           conf < 0.9
          │                                   │
          ▼                                   ▼
   ACCEPT TEXT LABEL                     IMAGE STAGE
                                              │
┌────────────────────┐                        │
│  Image (.jpg/…)    │────────────────────────┤
└────────────────────┘                        ▼
                               ┌──────────────────────────┐
                               │  CLIP ViT-L/14 encoder   │
                               │   768-dim L2-normalized  │
                               └──────────────┬───────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              probe loaded?       zero-shot fallback
                                    │                   │
                                    ▼                   ▼
                         ┌──────────────────┐   mean-pooled per-category
                         │ Linear(768 → 1)  │   logits @ T=100 → softmax
                         │ sigmoid → thresh │   (~95 hand-crafted prompts)
                         └────────┬─────────┘          │
                                  │                    │
                                  └─────────┬──────────┘
                                            ▼
                                      FINAL LABEL
```

### Stage 1 — Text

- Model: `valhalla/distilbart-mnli-12-3` (Hugging Face zero-shot-classification pipeline).
- Candidate labels: `["opioid_related", "non_opioid_related"]`.
- Confidence = gap between top-1 and top-2 probabilities.
- If confidence ≥ `TEXT_CONFIDENCE_THRESHOLD` (default **0.6**), the text label wins and the image is never loaded.

Rationale: captions with strong signals ("naloxone administration training tonight" / "pasta recipe") are cheap wins. Ambiguous captions fall through. The text stage **cannot be removed**: it filters posts where opioids are mentioned only in passing, which the image stage would otherwise see as ambiguous.

### Stage 2 — Image

**Encoder**: OpenAI CLIP ViT-L/14, frozen. Produces 768-dim L2-normalized embeddings.

**Head**: One of two, depending on whether `src/probe_weights.pt` is present.

1. **Linear probe (preferred)** — `torch.nn.Linear(768, 1)` trained with class-weighted BCE on the 300-image fine-tuning set. Threshold optimized to maximize specificity subject to recall ≥ 0.90. See [`TRAINING.md`](TRAINING.md).
2. **Zero-shot prompt matching (fallback)** — ~44 opioid prompts + ~51 non-opioid prompts, encoded by CLIP's text encoder. Per-category mean logit, temperature scaling (T=100), softmax. If top gap < `CLIP_CONFIDENCE_THRESHOLD` the post is forced to `non_opioid_related`. **This fallback has near-zero specificity** — only use it when you are debugging CLIP features directly.

### Error handling

- Corrupted image (`UnidentifiedImageError`), missing file, or unhandled exception → `("error", "error")`.
- No image in the post directory → `"error"` category.
- Failed Drive upload → the item is still classified; upload failures are counted in the result dict.

---

## Training data

| Dataset | Labels | Size | Class split |
|---|---|---|---|
| Fine-tuning  | `400_images_labeling.csv` | 300 | ~21:1 opioid:non-opioid (severe skew) |
| Test (good labels)| `100_testing_images_labels.csv` | 100 | 61 opioid, 39 non-opioid |
| Test (legacy, noisy labels)  | `100_testing_images_labels(noisy_do_not_use).csv` | 100 | do not use for final metrics |

All posts were pre-filtered by Sprinklr using the keyword `"opioid"`. **This creates selection bias**: `non_opioid_related` in this corpus means "visually/topically non-opioid despite mentioning opioids in the caption," not "arbitrary Instagram content." Do not generalize metrics beyond this distribution.

Labels in `400_images_labeling.csv`: `Post-id`, `Relevance` (0/1), `Valence` (1/2), `Theme` (free text), `Confidence` (1–10).

---

## Metrics

Numbers below are from the text + CLIP + linear probe configuration shipped in `src/probe_weights.pt`. They are reproducible via `python src/evaluate_opioid_classifier_accuracy.py`.

### Full 400-image set (300 fine-tune + 100 test)

| Metric | Value |
|---|---|
| Precision | 98.2% |
| Recall | 85.9% |
| Specificity | 66.7% (12/18 non-opioid posts correctly identified) |

### 100-image held-out test set (good labels)

| Metric | Value |
|---|---|
| Precision | 98.9% |
| Recall | 89.9% |
| TP / FP / FN / TN | 89 / 1 / 10 / 0 |

> Note: TN=0 on the 100-image run suggests the evaluator was fed a subset that contained no non-opioid posts (or all non-opioid posts fell into `ERRORS/`). Investigate before quoting this as specificity evidence. The 400-image number is the trustworthy one for specificity.



The probe traded essentially nothing on recall to recover specificity from near-useless to usable.

### Method mix (400-image run, baseline thresholds)

- Classified by **text**: 84 / 400 (21%)
- Fell back to **image**: 316 / 400 (79%)

Text-path share grows as `TEXT_CONFIDENCE_THRESHOLD` drops.

---

## Known biases and limitations

1. **Sprinklr pre-filter bias.** Every input already contains the word "opioid" somewhere. The model is only calibrated on this distribution. Running it against unfiltered Instagram would be off-label use.
2. **Severe class imbalance in training data** (21:1 positive:negative). The linear probe handles this with class-weighted loss, but only 18 `non_opioid_related` examples means the decision boundary is fragile — a single mislabeled negative meaningfully shifts validation metrics.
3. **Text classifier is English-only.** Non-English captions always fall through to the image path.
4. **Zero-shot fallback is not safe for production.** If `probe_weights.pt` is missing, the pipeline silently downgrades to the prompt-matching path whose specificity is ~5%. Monitor startup logs for `Linear probe loaded` on every run.
5. **No calibration guarantees.** Sigmoid probabilities from the probe are not calibrated to real class posteriors. Treat scores as rank signals, not probabilities.
6. **Threshold brittleness.** The 0.6 text threshold and the probe's sigmoid threshold are tuned on this specific corpus. New corpora need re-tuning.
7. **OCR / visual text.** CLIP sometimes classifies memes and text-heavy infographics based on visible captions, so an image containing the word "opioid" in large text may land as `opioid_related` even when the underlying photo is visually unrelated.
8. **No image-level explanations.** The pipeline returns a category and a scalar confidence — nothing about *why*.

## Failure modes to watch for

- Posts that are genuinely opioid-related but have a visually unrelated image (e.g., portrait shot of a speaker giving a talk about overdose policy) → classified `non_opioid_related` unless the text stage catches it.
- Posts of pills / pharmaceutical packaging that are *not* opioids (vitamins, antibiotics) → the probe was not trained to distinguish, so it may classify as `opioid_related`.
- Memes with "opioid crisis" overlay text on unrelated imagery → usually classified `opioid_related` via the text stage; treat as correct triage, review at the thematic stage.
- Corrupted images → classified `error` and uploaded to `ERRORS/`. Always scan that folder before reporting totals.

---

## Versioning

| Component | Version |
|---|---|
| CLIP | `ViT-L/14` (OpenAI) |
| Text classifier | `valhalla/distilbart-mnli-12-3` |
| Probe | `src/probe_weights.pt` — state dict of `Linear(768, 1)` + scalar `threshold` |
| Training script | `src/train_probe.py` |

Probe weights are not currently version-tagged. If you retrain, record the date, seed (currently `42`), fine-tuning set size, CV metrics, and optimal threshold in commit messages.
