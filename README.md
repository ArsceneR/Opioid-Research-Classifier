# Instagram Scrape and Store

A comprehensive tool for downloading, analyzing, and classifying Instagram posts with advanced image processing capabilities.

## Features

- **Post Downloading**: Batch download Instagram posts from specified sources
- **Data Analysis**: Analyze downloaded content including caption lengths and engagement metrics
- **Image Classification**: Advanced image processing and classification using Modal's GPU infrastructure
- **Google Drive Integration**: Automatic organization and storage of classified content
- **Data Management**: Tools for cleaning, organizing, and analyzing downloaded content
- **Comment Analysis**: Track and analyze comment counts and engagement

## Project Structure

```
src/
├── main.py                 # Main entry point
├── classifier.py           # Image classification and processing
└── modules/
    ├── downloader.py       # Instagram post downloading
    ├── analyze_downloads.py # Post analysis tools
    ├── data_reader.py      # Excel data processing
    ├── count_comments.py   # Comment analysis
    ├── clean_data.py       # Data cleaning utilities
    ├── add_comments_to_excel.py # Excel integration
    └── rate_controller.py  # Rate limiting control custom class 
```

## Prerequisites

- Python 3.10+
- Modal account and credentials
- Google Drive API access
- Required Python packages (see requirements.txt)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file with the following variables:
     ```
     LOCAL_DOWNLOADS_DIR=path/to/downloads
     CONTAINER_DOWNLOADS_DIR=/Downloads/
     MODAL_SECRET_NAME=your_modal_secret
     GDRIVE_ROOT_FOLDER_NAME=Classified_Posts
     MODAL_GPU_CONFIG=L40S
     GDRIVE_PARENT_FOLDER_ID=your_folder_id
     ```

## Usage

1. Configure your download sources in `main.py`
2. Run the main script:
   ```bash
   python src/main.py
   ```

The system will:
- Download specified Instagram posts
- Analyze content and captions
- Classify images using AI
- Organize and store content in Google Drive

## Features in Detail

### Post Downloading
- Batch download capabilities
- Automatic metadata extraction
- Rate limiting to prevent API blocks

### Image Classification
- GPU-accelerated processing
- Automatic categorization
- Google Drive integration for storage

### Data Analysis
- Caption length analysis
- Comment tracking
- Excel integration for data management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

