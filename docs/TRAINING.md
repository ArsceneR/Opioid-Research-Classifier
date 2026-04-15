# Training the Linear Probe

How to (re)train `src/probe_weights.pt`, the linear probe on frozen CLIP ViT-L/14 embeddings that powers the image stage of the classifier.

For *how the pipeline uses the probe*, see [`MODEL_CARD.md`](MODEL_CARD.md). For *running the classifier itself*, see [`RUNBOOK.md`](RUNBOOK.md).

---

## When to retrain

Retrain when any of the following changes:

- The fine-tuning dataset (new labels, new posts, relabeled posts).
- The CLIP backbone (e.g., switching from `ViT-L/14` to `ViT-L/14@336px` or a SigLIP variant).
- The recall floor used for threshold selection (currently `0.90` in `train_probe.py:235`).
- The severe class imbalance shifts meaningfully (current is ~21:1 positive:negative).

Do **not** retrain just because the classifier misclassified a few posts — the probe has 769 parameters and will happily memorize noise.

---

## Dataset requirements

The training job in `src/train_probe.py` reads two inputs:

1. **Image directory** — `300_images_(fine_tuning)/`, one subfolder per post. Each subfolder needs one image (`.jpg`, `.jpeg`, or `.png`). The subfolder name **must** be the post shortcode.
2. **Labels CSV** — `400_images_labeling.csv` at the repo root. Required columns:

   | Column | Type | Notes |
   |---|---|---|
   | `Post-id` | string | Must match a subfolder name exactly (whitespace is stripped) |
   | `Relevance` | `0` or `1` | 1 = opioid-related, 0 = non-opioid |

   Other columns (`Valence`, `Theme`, `Confidence`) are ignored by the probe but should still be present for `evaluate_opioid_classifier_accuracy.py`.

Subfolders without a matching CSV row are skipped with a warning. CSV rows without a matching subfolder are silently ignored.

> If you are training on a different dataset, update `FINETUNING_DIR` and `CSV_PATH` at the top of `src/train_probe.py`. Both are resolved relative to the repo root (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`), so the defaults work wherever the repo is checked out as long as the fine-tuning folder and labels CSV keep their current names.

---

## Prerequisites

Same Modal + service-account setup as `RUNBOOK.md §2`, except the training job does **not** touch Google Drive, so `GDRIVE_PARENT_FOLDER_ID` and the Drive secret are not required. You only need:

- Modal CLI authenticated (`modal token new`).
- `requirements.txt` installed locally.
- The dataset directory and labels CSV in place.

Cost: one L40S container, 1800 s timeout (30 min), typical runs finish well under 5 minutes including CLIP cold start.

---

## Running the job

```bash
modal run src/train_probe.py
```

Output: weights are written to `src/probe_weights.pt` locally via `main()` (`train_probe.py:322`). The file is a dict:

```python
{
    "weights": OrderedDict(...),  # state_dict of torch.nn.Linear(768, 1)
    "threshold": float,            # optimal sigmoid threshold from sweep
    "cv_metrics": {                # average across 5 folds
        "precision": ...,
        "recall": ...,
        "specificity": ...,
        "f1": ...,
        "threshold": ...,
    },
}
```

On the next `modal run src/classifier.py`, `classifier.py:73` detects the file and adds it to the container image. Nothing else to do.

---

## What the script does

Steps map to comments in `src/train_probe.py`:

1. **Load labels** (§B). Reads `400_images_labeling.csv`, strips header whitespace (the CSV has trailing spaces in headers), builds `post_id → relevance` map.
2. **Pair images with labels**. Walks `300_images_(fine_tuning)/`, finds the first `.jpg`/`.jpeg`/`.png` in each subfolder, skips subfolders without a matching CSV row.
3. **Extract embeddings** (§C). Loads CLIP `ViT-L/14`, encodes each image with `preprocess` + `encode_image`, L2-normalizes, stacks into `X ∈ ℝ^{N×768}`. CLIP is freed after this step to leave GPU memory for the probe.
4. **Stratified 5-fold split** (§D). Shuffles positives and negatives independently with `seed=42`, then deals round-robin into 5 folds so every fold gets 3–4 non-opioid samples.
5. **CV loop** (§E). For each fold: train on 4 folds, sweep threshold on the training set, evaluate on the held-out fold. Logs per-fold TP/FP/TN/FN + metrics. Averages across folds at the end.
6. **Final training** (§F). Trains one more probe on *all 300* images (no held-out set), sweeps threshold on the full data, picks the one that maximizes specificity with recall ≥ 0.90.
7. **Serialize and return**. Weights + threshold + CV metrics are pickled to bytes, returned from the Modal function, and written locally by `main()`.

### Hyperparameters

All defined inline in `train_model()` (`train_probe.py:150`). None are exposed as environment variables — edit the script to change them.

| Param | Value | Notes |
|---|---|---|
| Optimizer | Adam | `train_probe.py:154` |
| Learning rate | `1e-3` | |
| Weight decay | `1e-4` | L2 regularization |
| Epochs | `300` | Hard cap |
| Early stopping patience | `50` | Patience measured against best val specificity |
| Eval cadence | every 10 epochs | Early stopping signal is computed here |
| Seed | `42` | Controls fold shuffling + torch RNG |
| Folds | `5` | Stratified by class |
| Threshold sweep range | `0.05 → 0.90` step `0.05` | 18 candidates |
| Threshold selection rule | argmax specificity s.t. recall ≥ 0.90 | `train_probe.py:235` |

### Class weighting

Rather than `pos_weight` (which would upweight the *majority* class here), `train_model` builds a per-sample weight tensor that upweights the minority (`non_opioid_related`, label=0) class by the ratio $N_{pos}/N_{neg} \approx 21$. Loss is then a weighted mean of per-sample BCE:

```
sample_weights[y == 0] = n_pos / n_neg   # ~21
sample_weights[y == 1] = 1.0
loss = (BCE_raw * sample_weights).mean()
```

If you switch to a dataset with a different imbalance ratio this math is still correct — the ratio is recomputed from the training split. If your classes become balanced, the weights collapse to 1.0 and this is equivalent to standard BCE.

### Threshold selection

The rule is deliberately asymmetric: specificity is maximized *subject to* a recall floor, not a plain F1 optimum. This encodes the domain priority — missing an opioid post is worse than a false alarm, but the baseline's near-zero specificity made the model useless for triage. If your priorities change, edit `sweep_threshold` in `train_probe.py:217` (the `rec >= 0.90` line).

---

## Validating a new probe

After `src/probe_weights.pt` is updated, do all of the following before shipping:

1. **Sanity-check CV metrics** in the training logs. A healthy run shows per-fold specificity > 0.4 and recall > 0.85; wildly varying folds (e.g., one fold at 0.0 specificity) usually mean the split put all non-opioid samples into one fold and should be investigated.
2. **Run the full evaluator**:
   ```bash
   python src/evaluate_opioid_classifier_accuracy.py
   ```
   Confirm precision / recall / specificity are in the same ballpark as the model card. A >5 pp drop anywhere is a red flag — investigate before merging weights.
3. **Spot-check the false positives and negatives**. `evaluate_opioid_classifier_accuracy.py` prints FP/FN post IDs; open a few in the Drive folder and verify the human label was correct. Label noise is the most common cause of regression at this dataset size.
4. **End-to-end run** on a small subset via `modal run src/classifier.py` with a scratch `LOCAL_DOWNLOADS_DIR`, confirm the startup log shows `Linear probe loaded (threshold=...)` with the new threshold value.
5. **Record the retrain** in the commit message: date, dataset size, CV metrics, optimal threshold, seed if changed. The probe file itself has no metadata once loaded into the classifier.

---

## Gotchas

- **Path defaults assume the shipped layout.** `FINETUNING_DIR` and `CSV_PATH` are derived from `PROJECT_ROOT` and expect `300_images_(fine_tuning)/` and `400_images_labeling.csv` at the repo root. If either is renamed or moved, update the two constants at the top of `train_probe.py`.
- **CSV header whitespace.** The labels CSV has trailing spaces in header names (e.g., `"Post-id "`). `train_probe.py:65` strips them — do not "fix" the CSV without checking this.
- **Serialization round-trip.** Weights are serialized to bytes inside the Modal container and deserialized locally (`train_probe.py:317`). `torch.load(..., weights_only=False)` is required because the payload contains a Python dict. If PyTorch changes the default, this will break noisily — not silently.
- **Determinism.** The seed controls fold shuffling and torch RNG but *not* CLIP's forward pass (irrelevant here, since it's deterministic in eval mode) or CUDA nondeterminism. Expect tiny metric differences across runs even with the same seed.
- **Training on the full dataset at the end.** The final probe (step 6) trains on all 300 images including the ones used for threshold sweeping. CV metrics are an honest estimate; the shipped probe will usually be slightly better than CV on the training set and is *not* evaluated on held-out data inside the training script. Use `evaluate_opioid_classifier_accuracy.py` against the 100-image test set for the honest shipped number.
