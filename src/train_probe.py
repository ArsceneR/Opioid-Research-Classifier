"""
Train a linear probe on frozen CLIP ViT-L/14 embeddings for opioid image classification.

Uses the 300-image fine-tuning set with human labels. Runs on Modal GPU.
Performs stratified 5-fold cross-validation, threshold sweep, then final training.
Saves probe weights + optimal threshold to src/probe_weight.pt locally.
"""

import csv
import random
import logging
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNING_DIR = PROJECT_ROOT / "300_images_(fine_tuning)"
CSV_PATH = PROJECT_ROOT / "400_images_labeling.csv"
OUTPUT_PATH = PROJECT_ROOT / "src" / "probe_weights.pt"

CONTAINER_DATA_DIR = Path("/data/images")
CONTAINER_CSV_PATH = Path("/data/labels.csv")

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "ftfy",
        "regex",
        "tqdm",
        "Pillow",
        "git+https://github.com/openai/CLIP.git",
    )
    .add_local_dir(str(FINETUNING_DIR), str(CONTAINER_DATA_DIR))
    .add_local_file(str(CSV_PATH), str(CONTAINER_CSV_PATH))
)

app = modal.App("train-opioid-probe", image=image)


@app.function(gpu="L40S", timeout=1800)
def train_probe():
    """Extract embeddings, cross-validate, sweep threshold, train final model."""
    import torch
    import torch.nn.functional as F
    import clip
    from PIL import Image
    import io

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- B. Data loading ---
    labels_map = {}
    with open(CONTAINER_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Strip whitespace from header names (CSV has trailing spaces like "Post-id ")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            post_id = row["Post-id"].strip()
            relevance = int(row["Relevance"].strip())
            labels_map[post_id] = relevance

    logger.info(f"Loaded {len(labels_map)} labels from CSV")

    # Pair each subfolder's .jpg with its label
    image_paths = []
    image_labels = []
    missing = 0
    for subfolder in sorted(CONTAINER_DATA_DIR.iterdir()):
        if not subfolder.is_dir():
            continue
        post_id = subfolder.name
        if post_id not in labels_map:
            missing += 1
            continue
        # Find .jpg file
        jpg_files = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.jpeg")) + list(subfolder.glob("*.png"))
        if not jpg_files:
            logger.warning(f"No image found in {subfolder}")
            continue
        image_paths.append(jpg_files[0])
        image_labels.append(labels_map[post_id])

    n_pos = sum(image_labels)
    n_neg = len(image_labels) - n_pos
    logger.info(f"Paired {len(image_paths)} images: {n_pos} positive (opioid), {n_neg} negative (neutral)")
    if missing > 0:
        logger.warning(f"{missing} subfolders had no matching CSV entry")

    # --- C. Embedding extraction ---
    logger.info("Loading CLIP ViT-L/14...")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    embeddings = []
    valid_labels = []
    for i, (img_path, label) in enumerate(zip(image_paths, image_labels)):
        try:
            img = Image.open(img_path).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(img_input)
                features /= features.norm(dim=-1, keepdim=True)  # L2 normalize
            embeddings.append(features.cpu().float())
            valid_labels.append(label)
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
        if (i + 1) % 50 == 0:
            logger.info(f"  Extracted {i + 1}/{len(image_paths)} embeddings")

    X = torch.cat(embeddings, dim=0)  # (N, 768)
    y = torch.tensor(valid_labels, dtype=torch.float32)  # (N,)
    logger.info(f"Embeddings shape: {X.shape}, Labels shape: {y.shape}")

    # Free CLIP model memory
    del model, preprocess
    torch.cuda.empty_cache()

    # --- D & E. Stratified 5-fold cross-validation ---
    SEED = 42
    N_FOLDS = 5
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Split indices by class for stratification
    pos_indices = [i for i in range(len(y)) if y[i] == 1]
    neg_indices = [i for i in range(len(y)) if y[i] == 0]
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)

    # Distribute into folds
    pos_folds = [[] for _ in range(N_FOLDS)]
    neg_folds = [[] for _ in range(N_FOLDS)]
    for i, idx in enumerate(pos_indices):
        pos_folds[i % N_FOLDS].append(idx)
    for i, idx in enumerate(neg_indices):
        neg_folds[i % N_FOLDS].append(idx)

    fold_indices = [pos_folds[k] + neg_folds[k] for k in range(N_FOLDS)]

    # Training function
    def train_model(X_train, y_train, X_val, y_val, n_epochs=300, lr=1e-3, wd=1e-4, patience=50):
        """Train linear probe with class-weighted BCE loss. Returns model, best val metrics."""
        embed_dim = X_train.shape[1]
        probe = torch.nn.Linear(embed_dim, 1).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)

        X_tr = X_train.to(device)
        y_tr = y_train.to(device)
        X_v = X_val.to(device)
        y_v = y_val.to(device)

        # Per-sample weights: upweight minority class (neutral, label=0)
        n_p = (y_tr == 1).sum().item()
        n_n = (y_tr == 0).sum().item()
        weight_ratio = n_p / max(n_n, 1)
        sample_weights = torch.ones_like(y_tr)
        sample_weights[y_tr == 0] = weight_ratio
        logger.info(f"  Class weights: neutral={weight_ratio:.1f}, opioid=1.0 (pos={n_p}, neg={n_n})")

        best_spec = -1.0
        best_epoch = 0
        best_state = None

        for epoch in range(n_epochs):
            probe.train()
            logits = probe(X_tr).squeeze()
            loss_raw = F.binary_cross_entropy_with_logits(logits, y_tr, reduction='none')
            loss = (loss_raw * sample_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate
            if (epoch + 1) % 10 == 0 or epoch == 0:
                probe.eval()
                with torch.no_grad():
                    val_logits = probe(X_v).squeeze()
                    val_probs = torch.sigmoid(val_logits)
                    val_preds = (val_probs >= 0.5).float()

                    tp = ((val_preds == 1) & (y_v == 1)).sum().item()
                    fp = ((val_preds == 1) & (y_v == 0)).sum().item()
                    tn = ((val_preds == 0) & (y_v == 0)).sum().item()
                    fn = ((val_preds == 0) & (y_v == 1)).sum().item()

                    spec = tn / max(tn + fp, 1)
                    rec = tp / max(tp + fn, 1)

                if spec > best_spec:
                    best_spec = spec
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in probe.state_dict().items()}

                if (epoch + 1) % 50 == 0:
                    logger.info(f"    Epoch {epoch+1}: loss={loss.item():.4f} spec={spec:.3f} rec={rec:.3f}")

            # Early stopping
            if epoch - best_epoch >= patience and epoch > 50:
                logger.info(f"    Early stopping at epoch {epoch+1} (best spec at epoch {best_epoch+1})")
                break

        if best_state:
            probe.load_state_dict(best_state)
        return probe

    # Threshold sweep function
    def sweep_threshold(probe, X_data, y_data, thresholds):
        """Find threshold maximizing specificity while maintaining recall >= 0.90."""
        probe.eval()
        X_d = X_data.to(device)
        y_d = y_data.to(device)
        with torch.no_grad():
            probs = torch.sigmoid(probe(X_d).squeeze())

        best_thresh = 0.5
        best_spec = 0.0
        for t in thresholds:
            preds = (probs >= t).float()
            tp = ((preds == 1) & (y_d == 1)).sum().item()
            fp = ((preds == 1) & (y_d == 0)).sum().item()
            tn = ((preds == 0) & (y_d == 0)).sum().item()
            fn = ((preds == 0) & (y_d == 1)).sum().item()
            rec = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            if rec >= 0.90 and spec > best_spec:
                best_spec = spec
                best_thresh = t
        return best_thresh, best_spec

    thresholds = [i / 20.0 for i in range(1, 19)]  # 0.05 to 0.90 in 0.05 steps

    # Cross-validation loop
    fold_metrics = []
    fold_thresholds = []
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {N_FOLDS}-fold stratified cross-validation")
    logger.info(f"{'='*60}")

    for fold in range(N_FOLDS):
        val_idx = fold_indices[fold]
        train_idx = [i for k in range(N_FOLDS) if k != fold for i in fold_indices[k]]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        n_val_pos = (y_val == 1).sum().item()
        n_val_neg = (y_val == 0).sum().item()
        logger.info(f"\nFold {fold+1}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)} (pos={n_val_pos}, neg={n_val_neg})")

        probe = train_model(X_train, y_train, X_val, y_val)

        # Sweep threshold on training data
        best_thresh, _ = sweep_threshold(probe, X_train, y_train, thresholds)
        fold_thresholds.append(best_thresh)

        # Evaluate on val set with best threshold
        probe.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(probe(X_val.to(device)).squeeze())
            val_preds = (val_probs >= best_thresh).float()
            y_v = y_val.to(device)

            tp = ((val_preds == 1) & (y_v == 1)).sum().item()
            fp = ((val_preds == 1) & (y_v == 0)).sum().item()
            tn = ((val_preds == 0) & (y_v == 0)).sum().item()
            fn = ((val_preds == 0) & (y_v == 1)).sum().item()

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        fold_metrics.append({"precision": prec, "recall": rec, "specificity": spec, "f1": f1, "threshold": best_thresh})

        logger.info(f"  Fold {fold+1} results (threshold={best_thresh:.2f}):")
        logger.info(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
        logger.info(f"    Precision={prec:.3f} Recall={rec:.3f} Specificity={spec:.3f} F1={f1:.3f}")

    # Average metrics
    avg = {k: sum(m[k] for m in fold_metrics) / N_FOLDS for k in fold_metrics[0]}
    logger.info(f"\n{'='*60}")
    logger.info("Average cross-validation metrics:")
    logger.info(f"  Precision:   {avg['precision']:.3f}")
    logger.info(f"  Recall:      {avg['recall']:.3f}")
    logger.info(f"  Specificity: {avg['specificity']:.3f}")
    logger.info(f"  F1:          {avg['f1']:.3f}")
    logger.info(f"  Threshold:   {avg['threshold']:.3f}")
    logger.info(f"{'='*60}")

    # --- F. Final training on all data ---
    logger.info("\nTraining final model on all data...")
    final_probe = train_model(X, y, X, y, n_epochs=300, patience=50)

    # Sweep threshold on full dataset
    optimal_threshold, full_spec = sweep_threshold(final_probe, X, y, thresholds)
    logger.info(f"Optimal threshold (full data): {optimal_threshold:.2f} (specificity={full_spec:.3f})")

    # Serialize and return
    final_probe.cpu()
    result = {
        "weights": final_probe.state_dict(),
        "threshold": optimal_threshold,
        "cv_metrics": avg,
    }

    # Serialize to bytes for transfer
    buffer = io.BytesIO()
    torch.save(result, buffer)
    return buffer.getvalue()


@app.local_entrypoint()
def main():
    """Run training on Modal and save weights locally."""
    import torch
    import io

    logger.info("Launching training on Modal...")
    result_bytes = train_probe.remote()

    result = torch.load(io.BytesIO(result_bytes), weights_only=False)
    logger.info(f"Received probe weights. Threshold: {result['threshold']:.3f}")
    logger.info(f"CV metrics: {result['cv_metrics']}")

    # Save locally
    torch.save(result, OUTPUT_PATH)
    logger.info(f"Saved probe weights to {OUTPUT_PATH}")
