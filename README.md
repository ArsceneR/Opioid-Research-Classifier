# Instagram Scrape and Store

A tool for downloading, classifying, and organizing Instagram posts related to opioid content using CLIP image embeddings and zero-shot text classification, running on Modal cloud GPUs.

## Overview

This project classifies Instagram posts as **opioid-related** or **neutral** using a two-stage pipeline:

1. **Text classification**: Captions are run through a zero-shot NLI model (distilbart-mnli). If confidence is high enough, the text result is used directly.
2. **Image classification (fallback)**: When text confidence is low, the image is encoded via CLIP ViT-L/14. A trained **linear probe** on the 768-dim embedding classifies the post. A zero-shot prompt-matching fallback exists but is largely superseded by the probe.

Classified posts are uploaded to Google Drive, organized into category folders.

## Project Structure

```
src/
├── classifier.py                        # Main classification pipeline (runs on Modal)
├── train_probe.py                       # Train linear probe on CLIP embeddings
├── evaluate_opioid_classifier_accuracy.py # Evaluate predictions vs human labels
├── main.py                              # Entry point
├── probe_weights.pt                     # Trained probe weights + threshold
├── helpers/
│   └── random_picture.py                # Random image sampling utility
└── modules/
    ├── downloader.py                    # Instagram post downloading
    ├── analyze_downloads.py             # Post analysis tools
    ├── data_reader.py                   # Excel data processing
    ├── count_comments.py                # Comment analysis
    ├── clean_data.py                    # Data cleaning utilities
    ├── add_comments_to_excel.py         # Excel integration
    ├── find_files_without_metadata.py   # Find dirs missing metadata
    └── rate_controller.py               # Rate limiting control
```

## Prerequisites

- Python 3.10+
- [Modal](https://modal.com) account and CLI (`pip install modal`)
- Google Cloud service account with Drive API enabled
- Required packages: `pip install -r requirements.txt`

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the sample env file and fill in your values:
   ```bash
   cp .env.sample .env
   ```
4. Set up Modal secrets:
   ```bash
   modal secret create google_drive_secret SERVICE_ACCOUNT_JSON='<your-service-account-json>'
   ```
5. Place source images in your `LOCAL_DOWNLOADS_DIR` (each post in its own subfolder with a `.jpg` and `.txt` caption file)

## Configuration

Key `.env` variables:

| Variable | Description | Default |
|---|---|---|
| `LOCAL_DOWNLOADS_DIR` | Path to source images | `~/Downloads/All_Downloads` |
| `GDRIVE_PARENT_FOLDER_ID` | Google Drive parent folder ID | (required) |
| `CLIP_CONFIDENCE_THRESHOLD` | Image classification confidence threshold | `0.9` |
| `TEXT_CONFIDENCE_THRESHOLD` | Text classification confidence threshold | `0.9` |
| `TEXT_CLASSIFIER_MODEL` | Zero-shot text model | `valhalla/distilbart-mnli-12-3` |
| `MODAL_GPU_CONFIG` | GPU type for Modal containers | `L40S` |
| `USE_LINEAR_PROBE` | Enable/disable linear probe | `true` |

## Usage

### Run the classifier

```bash
modal run src/classifier.py
```

This will:
1. Create Google Drive folder structure (`setup_drive_folders()`)
2. Scan `LOCAL_DOWNLOADS_DIR` for post directories
3. Classify each post in parallel via `process_item.map()`
4. Upload classified posts to the appropriate Drive folder

### Train the linear probe

```bash
modal run src/train_probe.py
```

Trains a linear probe on frozen CLIP ViT-L/14 embeddings using the fine-tuning dataset with human labels. Performs stratified 5-fold cross-validation, threshold sweep (maximizing specificity while maintaining recall >= 0.90), then saves `probe_weights.pt`.

### Evaluate accuracy

```bash
python src/evaluate_opioid_classifier_accuracy.py
```

Compares classifier predictions (downloaded from Drive) against human labels. Outputs confusion matrix, precision, recall, F1, specificity, and writes results to `classifier_evaluation.xlsx`.

## Classification Pipeline

```
Input Post (image + caption)
│
├─ Caption exists? ──→ Text Classifier (distilbart zero-shot)
│                         │
│                         ├─ Confidence ≥ 0.9 ──→ Use text result
│                         │
│                         └─ Confidence < 0.9 ──→ Fallback to image ─┐
│                                                                     │
└─ No caption ──────────────────────────────────────────────────────→─┤
                                                                      │
                                                          CLIP encode_image()
                                                          768-dim embedding
                                                                │
                                                    ┌───────────┴───────────┐
                                                    │                       │
                                              Probe loaded?           No probe
                                                    │                       │
                                             Linear(768,1)         Zero-shot prompts
                                             sigmoid → threshold   ~100 templates
                                                    │               softmax
                                                    └───────┬───────┘
                                                            │
                                                      Category Result
                                                 (opioid_related | neutral_content | error)
                                                            │
                                                    Upload to Google Drive
```

## Data

### Testing set (100 images)
- **Folder**: `100_images_(testing_good_labeled)/`
- **Labels**: `Coding Sheet_n=100(Sheet1).csv`

### Fine-tuning set (300 images)
- **Folder**: `300_images_(fine_tuning)/`
- **Labels**: `Excel_With_IDs(Sheet1).csv`

Each subfolder contains a `.jpg` image and a `.txt` caption file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
