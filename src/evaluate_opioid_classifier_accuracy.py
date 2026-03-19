"""
Simple accuracy test: Compare CLIP's opioid_related predictions with human labels.

The folders in the test directory are posts CLIP classified as opioid_related.
We compare these with human classifications from Excel.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    excel_path = Path("/Users/arscenerubayita/Documents/Personal_Programming/Instagram_Scrape_and_Store/Coding Sheet_n=100(Sheet1).csv") #human labeled source of truth 
    test_dir = Path("/Users/arscenerubayita/Downloads/Opioid_Related_100") #machine classified opioid_related posts (folders named by post ID)
    
    # Load CSV with id and relevance columns
    df = pd.read_csv(excel_path)
    logger.info(f"CSV columns: {list(df.columns)}")
    print(f"number of rows in CSV: {len(df)}")
    # Find id and relevance columns
    id_col = None
    relevance_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'id' in col_lower and id_col is None:
            id_col = col
        if 'relevance' in col_lower and relevance_col is None:
            relevance_col = col
    
    if not id_col or not relevance_col:
        raise ValueError(f"Need 'id' and 'relevance' columns. Found: {list(df.columns)}")
    
    logger.info(f"Using ID column: '{id_col}', Relevance column: '{relevance_col}'")
    
    # Create mapping: id -> human relevance
    human_labels = {}
    for _, row in df.iterrows():
        post_id = (row[id_col])
        relevance = int(row[relevance_col])
        human_labels[post_id] = relevance
    
    logger.info(f"Loaded {len(human_labels)} human labels from Excel")
    
    # Get folders in test directory (these are CLIP's opioid_related predictions)
    clip_opioid_folders = [d.name for d in test_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(clip_opioid_folders)} folders (CLIP classified as opioid_related)")
    
    # Compare: for each folder CLIP said was opioid_related, check human label
    matches = 0
    mismatches = 0
    total = 0
    
    for folder_id in clip_opioid_folders:
        total += 1
        post_id = folder_id
        if post_id in human_labels:
            human_label = human_labels[post_id]
            if human_label == 1:
                matches += 1
            elif human_label == 0:
                mismatches += 1
        
    if total == 0:
        logger.error("No matching folders found!")
        return
    
    accuracy = (matches / total) * 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Matches(clip said opioid_related, human said also): {matches}")
    print(f"Mismatches(clip said opioid_related, human said not): {mismatches}")
    print(f"Total: {total}")


if __name__ == '__main__':
    main()
