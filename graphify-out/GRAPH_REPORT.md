# Graph Report - .  (2026-05-01)

## Corpus Check
- 22 files · ~14,634 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 136 nodes · 133 edges · 23 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Modal Classifier Pipeline|Modal Classifier Pipeline]]
- [[_COMMUNITY_Evaluation & Methodology|Evaluation & Methodology]]
- [[_COMMUNITY_Two-Stage Classification Logic|Two-Stage Classification Logic]]
- [[_COMMUNITY_Excel Comment Augmentation|Excel Comment Augmentation]]
- [[_COMMUNITY_Download Analysis Utilities|Download Analysis Utilities]]
- [[_COMMUNITY_ModalDrive Orchestration|Modal/Drive Orchestration]]
- [[_COMMUNITY_CLIP Linear Probe Foundations|CLIP Linear Probe Foundations]]
- [[_COMMUNITY_Module-Level Function Surface|Module-Level Function Surface]]
- [[_COMMUNITY_Linear Probe Training|Linear Probe Training]]
- [[_COMMUNITY_Instaloader Download Layer|Instaloader Download Layer]]
- [[_COMMUNITY_Accuracy Evaluation Script|Accuracy Evaluation Script]]
- [[_COMMUNITY_TrainTest Set Reshuffler|Train/Test Set Reshuffler]]
- [[_COMMUNITY_Class Imbalance Handling|Class Imbalance Handling]]
- [[_COMMUNITY_Metadata Audit Helper|Metadata Audit Helper]]
- [[_COMMUNITY_Random Image Sampler|Random Image Sampler]]
- [[_COMMUNITY_Rate-Limited Scraping|Rate-Limited Scraping]]
- [[_COMMUNITY_Duplicate Download Cleanup|Duplicate Download Cleanup]]
- [[_COMMUNITY_Per-Item Processing|Per-Item Processing]]
- [[_COMMUNITY_clean_data rename function|clean_data rename function]]
- [[_COMMUNITY_Folder Structure Reformat|Folder Structure Reformat]]
- [[_COMMUNITY_Caption Length Stats|Caption Length Stats]]
- [[_COMMUNITY_Random-Picture Helper|Random-Picture Helper]]
- [[_COMMUNITY_Semiotic Divergence Reference|Semiotic Divergence Reference]]

## God Nodes (most connected - your core abstractions)
1. `train_probe.train_probe` - 9 edges
2. `Classifier` - 5 edges
3. `classifier.main (local entrypoint)` - 5 edges
4. `reshuffle_set.reshuffle` - 5 edges
5. `create_drive_folder()` - 4 edges
6. `process_item()` - 4 edges
7. `count_comments()` - 4 edges
8. `Classifier._analyze_image` - 4 edges
9. `Classifier.process_item` - 4 edges
10. `distilbart-mnli zero-shot text classifier` - 4 edges

## Surprising Connections (you probably didn't know these)
- `train_probe.train_probe` --shares_data_with--> `400_images_labeling.csv (labels)`  [EXTRACTED]
  src/train_probe.py → labels/400_images_labeling.csv
- `Zero-shot CLIP prompt-matching fallback` --implements--> `Zero-shot CLIP baseline (rationale: pre-filter Sprinklr means visual signal must discriminate; baseline gave 0% specificity)`  [EXTRACTED]
  src/classifier.py → research_methodology.txt
- `downloader.batch_post_downloads` --cites--> `Abdukhamidov et al. 2022 (Instaloader in research)`  [INFERRED]
  src/modules/downloader.py → research_methodology.txt
- `train_probe.train_model (inner)` --implements--> `Class-imbalance reweighting (rationale: 32:1 imbalance, unweighted BCE collapses to majority; per-sample inverse class-frequency weights equalize gradient contribution)`  [EXTRACTED]
  src/train_probe.py → research_methodology.txt
- `train_probe.sweep_threshold (inner)` --implements--> `Threshold selection rule (rationale: default 0.5 inherits zero-specificity; F1-optimal symmetric, mismatches cost structure; sweep maximizes specificity s.t. recall>=0.90)`  [EXTRACTED]
  src/train_probe.py → research_methodology.txt

## Hyperedges (group relationships)
- **Two-stage classification pipeline (text -> image fallback -> probe-or-zero-shot)** — classifier_analyze_image, classifier_classify_text, classifier_linear_probe_loader, classifier_zero_shot_image_fallback [EXTRACTED 1.00]
- **Probe training flow: CV + threshold sweep + final fit serialized to probe_weights.pt consumed by classifier** — train_probe_train_probe, train_probe_stratified_5_fold_cv, train_probe_threshold_sweep, train_probe_probe_weights_pt, classifier_linear_probe_loader [EXTRACTED 1.00]
- **Dataset balancing + evaluation flow over 400 labeled posts** — reshuffle_set_reshuffle, evaluate_opioid_classifier_accuracy_evaluate, labels_csv_400, fine_tuning_300_dataset, testing_100_dataset [EXTRACTED 1.00]

## Communities

### Community 0 - "Modal Classifier Pipeline"
Cohesion: 0.17
Nodes (12): Classifier, create_drive_folder(), main(), process_item(), Reads caption text from corresponding .txt file and preprocesses it., Classifies caption text using zero-shot classification.                  Args:, Analyzes a single image using fallback logic: text classification if confident,, Creates the main classification folder in Google Drive if it doesn't exist. (+4 more)

### Community 1 - "Evaluation & Methodology"
Cohesion: 0.15
Nodes (15): evaluate_opioid_classifier_accuracy.evaluate, 300_images_(fine_tuning) dataset, 400_images_labeling.csv (labels), src/main.py orchestrator, Stratified 5-fold CV (rationale: tiny minority class makes single held-out split unreliable; stratification ensures every fold has neutrals), Threshold selection rule (rationale: default 0.5 inherits zero-specificity; F1-optimal symmetric, mismatches cost structure; sweep maximizes specificity s.t. recall>=0.90), reshuffle_set._count_neutrals, reshuffle_set.get_human_labels (+7 more)

### Community 2 - "Two-Stage Classification Logic"
Cohesion: 0.17
Nodes (13): Classifier._analyze_image, CATEGORIES prompt sets (opioid/non-opioid), Classifier._classify_text, distilbart-mnli zero-shot text classifier, Linear probe load (probe_weights.pt), Classifier._read_caption, Classifier.start (modal.enter), Zero-shot CLIP prompt-matching fallback (+5 more)

### Community 3 - "Excel Comment Augmentation"
Cohesion: 0.18
Nodes (7): add_comments_to_excel(), Append comment counts to the existing Excel files based on metadata in .xz files, find_failed_urls(), Find and log failed(not downloaded) URLs that are in Excel files but missing fro, count_comments(), Count comments for Instagram posts based on metadata in .xz files.      Args:, get_column_data()

### Community 4 - "Download Analysis Utilities"
Cohesion: 0.22
Nodes (7): find_duplicate_downloads(), find_empty_folders(), get_img_types(), Get a set of unique image file extensions in the download directory.      Args:, Find and log empty folders in the download directory., Find and log duplicate downloads by looking for multiple files with the same Ins, remove_duplicates()

### Community 5 - "Modal/Drive Orchestration"
Cohesion: 0.28
Nodes (6): Classifier (Modal class), classifier.create_drive_folder, classifier.main (local entrypoint), Classifier.process_item, classifier.setup_drive_folders, classifier.upload_to_drive

### Community 6 - "CLIP Linear Probe Foundations"
Cohesion: 0.25
Nodes (8): Alain & Bengio 2017 (linear probing), Radford et al. 2021 (CLIP), Kingma & Ba 2015 (Adam), Kornblith et al. 2019 (transfer benchmarks), Linear probe design (rationale: 300 samples too small for end-to-end finetune of 304M params; linear probe sample-efficient and preserves CLIP features; MLP rejected as nine-sample minority class cannot constrain extra parameters), ViT-L/14 backbone choice (rationale: outperforms smaller CLIP backbones on linear-probe transfer; encoder frozen so cost amortized), Zero-shot CLIP baseline (rationale: pre-filter Sprinklr means visual signal must discriminate; baseline gave 0% specificity), CLIP ViT-L/14 encoder

### Community 7 - "Module-Level Function Surface"
Cohesion: 0.29
Nodes (7): add_comments_to_excel, analyze_downloads.find_empty_folders, analyze_downloads.find_failed_urls, count_comments.count_comments, data_reader.get_column_data, data_reader.read_file, find_files_without_metadata

### Community 8 - "Linear Probe Training"
Cohesion: 0.33
Nodes (5): main(), Train a linear probe on frozen CLIP ViT-L/14 embeddings for opioid image classif, Run training on Modal and save weights locally., Extract embeddings, cross-validate, sweep threshold, train final model., train_probe()

### Community 9 - "Instaloader Download Layer"
Cohesion: 0.4
Nodes (2): batch_post_downloads(), MyRateController

### Community 10 - "Accuracy Evaluation Script"
Cohesion: 0.5
Nodes (3): evaluate(), Evaluate classifier accuracy against human labels.  Compares classifier predicti, _summary_      Args:         excel_path (str): Path to CSV file with human label

### Community 11 - "Train/Test Set Reshuffler"
Cohesion: 0.83
Nodes (3): _count_neutrals(), get_human_labels(), reshuffle()

### Community 12 - "Class Imbalance Handling"
Cohesion: 0.5
Nodes (4): Class-imbalance reweighting (rationale: 32:1 imbalance, unweighted BCE collapses to majority; per-sample inverse class-frequency weights equalize gradient contribution), Cui et al. 2019 (class-balanced reweighting), Class-weighted BCE loss, train_probe.train_model (inner)

### Community 13 - "Metadata Audit Helper"
Cohesion: 0.67
Nodes (2): find_files_without_metadata(), Find directories that don't have .xz metadata files.

### Community 14 - "Random Image Sampler"
Cohesion: 0.67
Nodes (2): copy_random_images_with_captions(), Select up to n random images (and their caption files if present) from source an

### Community 15 - "Rate-Limited Scraping"
Cohesion: 0.67
Nodes (3): downloader.batch_post_downloads, MyRateController, Abdukhamidov et al. 2022 (Instaloader in research)

### Community 17 - "Duplicate Download Cleanup"
Cohesion: 1.0
Nodes (2): analyze_downloads.find_duplicate_downloads, analyze_downloads.remove_duplicates

### Community 18 - "Per-Item Processing"
Cohesion: 1.0
Nodes (1): Processes a single item directory: finds image, classifies, uploads all files.

### Community 22 - "clean_data rename function"
Cohesion: 1.0
Nodes (1): clean_data.rename_files

### Community 23 - "Folder Structure Reformat"
Cohesion: 1.0
Nodes (1): analyze_downloads.reformat_download_structure

### Community 24 - "Caption Length Stats"
Cohesion: 1.0
Nodes (1): analyze_downloads.get_caption_lengths

### Community 25 - "Random-Picture Helper"
Cohesion: 1.0
Nodes (1): copy_random_images_with_captions

### Community 26 - "Semiotic Divergence Reference"
Cohesion: 1.0
Nodes (1): Xiao et al. 2025 (semiotic divergence)

## Knowledge Gaps
- **50 isolated node(s):** `Train a linear probe on frozen CLIP ViT-L/14 embeddings for opioid image classif`, `Extract embeddings, cross-validate, sweep threshold, train final model.`, `Run training on Modal and save weights locally.`, `Creates a new folder in Google Drive or returns its ID if it already exists.`, `Reads caption text from corresponding .txt file and preprocesses it.` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Instaloader Download Layer`** (5 nodes): `batch_post_downloads()`, `downloader.py`, `MyRateController`, `.sleep()`, `rate_controller.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Metadata Audit Helper`** (3 nodes): `find_files_without_metadata()`, `find_files_without_metadata.py`, `Find directories that don't have .xz metadata files.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Random Image Sampler`** (3 nodes): `copy_random_images_with_captions()`, `random_picture.py`, `Select up to n random images (and their caption files if present) from source an`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Duplicate Download Cleanup`** (2 nodes): `analyze_downloads.find_duplicate_downloads`, `analyze_downloads.remove_duplicates`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Per-Item Processing`** (1 nodes): `Processes a single item directory: finds image, classifies, uploads all files.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `clean_data rename function`** (1 nodes): `clean_data.rename_files`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Folder Structure Reformat`** (1 nodes): `analyze_downloads.reformat_download_structure`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Caption Length Stats`** (1 nodes): `analyze_downloads.get_caption_lengths`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Random-Picture Helper`** (1 nodes): `copy_random_images_with_captions`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Semiotic Divergence Reference`** (1 nodes): `Xiao et al. 2025 (semiotic divergence)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_probe.train_probe` connect `Evaluation & Methodology` to `Two-Stage Classification Logic`, `Class Imbalance Handling`, `Modal/Drive Orchestration`, `CLIP Linear Probe Foundations`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `probe_weights.pt artifact` connect `Two-Stage Classification Logic` to `Evaluation & Methodology`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **What connects `Train a linear probe on frozen CLIP ViT-L/14 embeddings for opioid image classif`, `Extract embeddings, cross-validate, sweep threshold, train final model.`, `Run training on Modal and save weights locally.` to the rest of the system?**
  _50 weakly-connected nodes found - possible documentation gaps or missing edges._