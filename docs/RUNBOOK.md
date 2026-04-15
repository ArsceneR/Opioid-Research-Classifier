# Runbook

Operational guide for running the Instagram opioid classification pipeline. If something breaks or you are running it for the first time, start here.

For *what the models do and how well*, see [`MODEL_CARD.md`](MODEL_CARD.md). For *how to retrain the linear probe*, see [`TRAINING.md`](TRAINING.md).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Must match `MODAL_PYTHON_VERSION` (default `3.10`) |
| [Modal](https://modal.com) account + CLI | `pip install modal && modal token new` |
| Google Cloud project w/ Drive API enabled | Service account + JSON key (see §2.1) |
| Local image corpus | One subfolder per post, each containing a `.jpg` and `.txt` |
| `requirements.txt` installed | `pip install -r requirements.txt` |

---

## 2. First-time setup

### 2.1 Google Drive service account

1. Create a service account in Google Cloud Console, enable the **Drive API**, and download the JSON key.
2. Share your target Drive parent folder with the service account's email address (it needs Editor access).
3. Note the **Drive folder ID** from the URL — this becomes `GDRIVE_PARENT_FOLDER_ID`.
4. Register the JSON as a Modal secret:
   ```bash
   modal secret create google_drive_secret \
     SERVICE_ACCOUNT_JSON="$(cat path/to/service-account.json)"
   ```
   The secret name must match `MODAL_SECRET_NAME` (default `google_drive_secret`).

### 2.2 Environment file

```bash
cp .env.sample .env
```

Fill in at minimum:

| Variable | What it is |
|---|---|
| `GDRIVE_PARENT_FOLDER_ID` | Drive folder where `Classified_Posts/` will live |
| `LOCAL_DOWNLOADS_DIR` | Absolute path to your post subfolders |
| `SERVICE_ACCOUNT_JSON_PATH` | Local path to the JSON key (used by `setup_drive_folders` which runs locally) |

Leave the `*_FOLDER_ID` fields blank — `setup_drive_folders()` populates them on first run and appends them to `.env`.

### 2.3 Linear probe weights

The classifier looks for `src/probe_weights.pt`. If present it is added to the container image at build time and used for image classification. If missing, the pipeline falls back to zero-shot prompt matching (worse specificity — see the model card).

- To use the shipped weights: they should already be at `src/probe_weights.pt`.
- To train fresh weights: follow [`TRAINING.md`](TRAINING.md).

---

## 3. Running the classifier

```bash
modal run src/classifier.py
```

What happens:

1. **Local phase** — `setup_drive_folders()` ensures `Classified_Posts/`, `OPIOID_RELATED/`, `NON_OPIOID_CONTENT/`, and `ERRORS/` exist in Drive. Folder IDs get appended to `.env`.
2. **Enumerate** — `main()` lists every subdirectory under `LOCAL_DOWNLOADS_DIR`.
3. **Fan-out** — `Classifier.process_item.map(...)` dispatches items across up to 10 L40S(or another gpu of choice such as h100) containers (`min_containers=2`).
4. **Per item** — text → image classification (see model card), then `create_drive_folder` + per-file upload.
5. **Summary** — `main()` prints totals: processed / skipped / errors / text-vs-image breakdown.

Expected throughput: ~2–4 posts/sec steady-state once containers are warm. Cold start dominates for small batches. Modals seems to cache / persist images on re-run 

### Optional overrides

```bash
# Override the Drive parent folder for this run:
modal run src/classifier.py --drive-parent-id "1AbCdEfGhIjKlMnOp"
```

Anything else (thresholds, GPU type, model IDs) is changed via `.env`. Key knobs:

| Variable | Default | Effect |
|---|---|---|
| `TEXT_CONFIDENCE_THRESHOLD` | `0.6` | How confident distilbart must be to resolve a post without looking at the image. Raise → more image fallbacks. |
| `CLIP_CONFIDENCE_THRESHOLD` | `0.0` | Only used in the **zero-shot** image path. If the top–second gap is below this, the post is forced to `non_opioid_related`. No effect when the linear probe is active. |
| `USE_LINEAR_PROBE` | `true` | Set `false` to force the zero-shot prompt-matching path (for debugging). |
| `CLIP_TEMPERATURE` | `100.0` | Softmax temperature for the zero-shot path. Ignored by the probe. |
| `MODAL_GPU_CONFIG` | `L40S` | Any Modal GPU string (`A10G`, `A100`, etc.). |

---

## 4. Running the evaluator

```bash
python src/evaluate_opioid_classifier_accuracy.py
```

This compares classifier outputs (pulled from Drive or whichever source the script is pointed at) against the human labels in `400_images_labeling.csv` / `100_testing_images_labels.csv`. Writes `classifier_evaluation.xlsx` at the repo root and logs precision / recall / specificity / F1 plus the false-positive and false-negative post IDs.

Run after every probe retrain and after any threshold change.

---

## 5. Retraining the probe

See [`TRAINING.md`](TRAINING.md). Short version:

```bash
modal run src/train_probe.py
```

The trained weights land at `src/probe_weights.pt`. On the next `modal run src/classifier.py`, they are baked into the container image automatically (see `classifier.py:73`).

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `GDRIVE_PARENT_FOLDER_ID environment variable is not set` | `.env` not loaded or empty | `cp .env.sample .env` and fill in the parent folder ID. |
| `SERVICE_ACCOUNT_JSON not found` | Modal secret missing or misnamed | Re-run `modal secret create google_drive_secret ...`; confirm `MODAL_SECRET_NAME` matches. |
| `CLASSIFICATION_FOLDER_ID environment variable not set in container` | Setup step never ran or `.env` wasn't reloaded before dispatch | Delete category folder IDs from `.env` and rerun — `main()` re-invokes `setup_drive_folders.local()`. |
| `No subdirectories found in ...` | `LOCAL_DOWNLOADS_DIR` points at an empty or wrong directory | Fix the path in `.env`. Each post must be its own subdirectory. |
| `No image found in <subfolder>` in logs | Subfolder has only the `.txt` file | Expected for malformed downloads; those items end up in `Processing Errors/`. |
| All items classified as `opioid_related` / nothing in `NON_OPIOID_CONTENT/` | Probe not loaded → fell back to zero-shot | Check startup logs for `Linear probe loaded` vs `Probe weights not found`. If the former is missing, confirm `src/probe_weights.pt` exists *before* `modal run`. |
| Drive folder `Classified_Posts/` created but posts never appear | Service account lacks access to the parent folder | Re-share the parent folder with the service account email as Editor. |
| `Subfolder already exists in Google Drive` — items marked `skipped_exist` | Rerunning against the same corpus | Expected; dedupe is by post shortcode. Delete the Drive subfolder if you want to reclassify. |
| Cold start > 3 min per container | CLIP + probe + distilbart loading | Normal first time. Modal caches the image layer across runs — subsequent runs are much faster. |

### Logs

- `LOG_LEVEL=DEBUG` in `.env` for verbose per-item diagnostics (including text-classifier scores and CLIP per-category probs).
- Modal container logs: visible in the CLI during `modal run`, and in the Modal dashboard afterwards.

---

## 7. Cost and rate-limit notes

- **Modal**: L40S ~= $1.95/hr list (check Modal pricing). A full 400-image pass takes ~10–15 min of container time with `min_containers=2`, so a single full run is well under a dollar.
- **Google Drive API**: default per-user quota is generous but not infinite. If you run against tens of thousands of posts, expect sporadic 403/429s — `upload_to_drive` logs them and moves on; the item is still recorded with upload errors.
- **Hugging Face**: distilbart and CLIP are downloaded on container cold start. If HF rate-limits you, Modal image caching will mask it after the first run.

---

## 8. Handoff checklist

Before handing the project off, verify in order:

- [ ] `.env.sample` is current and mirrors the real `.env` variables (no secrets, no local paths).
- [ ] `src/probe_weights.pt` is checked in (or documented as a build artifact to regenerate).
- [ ] `modal run src/classifier.py` completes end-to-end against a small test corpus.
- [ ] `python src/evaluate_opioid_classifier_accuracy.py` reproduces the numbers in the model card within noise.
- [ ] This runbook, the model card, and the training doc are linked from the README.
