from collections import defaultdict
import os
import json
import lzma
import shutil
import logging
from typing import Dict, List
from modules.data_reader import get_column_data

def find_failed_urls(file_paths: List[str], download_dir: str) -> List[str]:
    """Find and log failed(not downloaded) URLs that are in Excel files but missing from downloaded posts."""
    
    # Get all URLs from the Excel files
    urls_from_excel = set(get_column_data(file_paths))
    logging.info(f"Number of unique urls: {len(urls_from_excel)}")
    

    # Walk through download directory
    for root, _, files in os.walk(os.path.expanduser(download_dir)):
        if "Post" in root:
            for file in files:
                if file.endswith(".xz"):
                    file_path = os.path.join(root, file)
                    try:
                        with lzma.open(file_path, "rt", encoding="utf-8") as f:
                            data = json.load(f)
                            shortcode = data.get("node", {}).get("shortcode")
                            if shortcode:
                                url = f"https://www.instagram.com/p/{shortcode}/"
                                urls_from_excel.discard(url)  # remove if exists
                    except (lzma.LZMAError, json.JSONDecodeError, OSError) as e:
                        logging.error(f"Error processing {file_path}: {e}")

    # Log the number of failed URLs
    logging.info(f"Number of failed URLs: {len(urls_from_excel)}")

    # Write failed URLs to file
    if urls_from_excel:
        with open("failed_urls.txt", "w", encoding="utf-8") as file:
            for url in urls_from_excel:
                file.write(url + "\n")

    # Log each failed URL and its index
     # Log each failed URL and its index
    for i, url in enumerate(urls_from_excel):
        logging.info(f"Failed URL: {url} at index: {i}")

    return list(urls_from_excel)

def find_empty_folders(file_paths: List[str], download_dir: str) -> None:
    """Find and log empty folders in the download directory."""
   
    empty_dirs = []
    for root, _, files in os.walk(os.path.expanduser(download_dir)): 
        if "Post" in root and len(files) == 0:
            empty_dirs.append(root)
    
    empty_dirs.sort(key=lambda x:int(x.split("-")[-1]))
    
    for directory in empty_dirs:
        with open("empty_folders.txt", "a", encoding="utf-8") as file:
            file.write(directory + "\n")
        logging.info(f"Empty folder: {directory}")


def find_duplicate_downloads(download_dir: str) -> Dict[str, List[str]]:
    """
    Find and log duplicate downloads by looking for multiple files with the same Instagram URL.
    
    Args:
        download_dir: Directory containing downloaded posts
        
    Returns:
        Dictionary mapping duplicated URLs to list of file paths where they were found
    """
    url_bucket = defaultdict(list)
    
    for root, _, files in os.walk(os.path.expanduser(download_dir)):
        
        for file in files:
            if file.endswith(".xz"):
                file_path = os.path.join(root, file)
                try:
                    with lzma.open(file_path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                        shortcode = data["node"]["shortcode"]
                        url = f"https://www.instagram.com/p/{shortcode}/"
                        url_bucket[url].append(root)
                except (lzma.LZMAError, json.JSONDecodeError) as e:
                    logging.error(f"Error processing {root}: {e}")
    

    duplicates = {url: paths for url, paths in url_bucket.items() if len(paths) > 1}
    logging.info(f"Number of duplicate URLs: {len(duplicates)}")
    
    with open("duplicates.txt", "w", encoding="utf-8") as f:
        for url, paths in duplicates.items():
            f.write(f"{url} -> {paths}\n------------------------\n")
            #print(f"{url} -> {paths}\n\n")
    return duplicates

def remove_duplicates(download_dir: str) -> None: 
    dups = find_duplicate_downloads(download_dir)
    
    for url, paths in dups.items(): 
        # Sort to ensure consistent behavior when deciding which to keep
        sorted_paths = sorted(paths)
        
        for path in sorted_paths[1:]:  # Keep the first, remove the rest
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    logging.info(f"Removed duplicate folder: {path}")
                except OSError as e:
                    logging.error(f"Error removing folder {path}: {e}")
            else:
                logging.warning(f"Path does not exist: {path}")
    
def reformat_download_structure(download_dir: str, destination_dir: str) -> None:
    destination_root = os.path.expanduser(destination_dir)
    suffix = 0

    for root, _, files in os.walk(os.path.expanduser(download_dir), topdown=False):
        grouped = defaultdict(list)

        # Group files by timestamp prefix
        for file in files:
            if 'json' in file:
                prefix = "_".join(os.path.splitext(os.path.splitext(file)[0])[0].split("_")[:4])
                grouped[prefix].append(file)
            else:
                prefix = "_".join(os.path.splitext(file)[0].split("_")[:4])
                grouped[prefix].append(file)
            
        for prefix, matched_files in grouped.items():
            required_exts = {".jpg", ".txt"}
            found_exts = {os.path.splitext(f)[1] for f in matched_files}

            if found_exts >= required_exts:
                # Create folder with unique suffix if needed
                new_folder_path = os.path.join(destination_root, f"{prefix}_{suffix}")
                suffix += 1  # Increment for next group

                os.makedirs(new_folder_path, exist_ok=True)

                for f in matched_files:
                    src = os.path.join(root, f)
                    dst = os.path.join(new_folder_path, f)

                    if os.path.exists(dst):
                        print(f"File {dst} already exists, skipping.")
                    else:
                        shutil.move(src, dst)

                print(f"Moved {prefix} group to {new_folder_path}")
                
    print(f"Moved {suffix} files")
 
    for url, files in grouped.items():
        if len(files) == 3:  # Check if all three required files are present
            logging.info(f"URL {url} has all three required files.")
            
    total_with_all_three = sum(1 for file_count in grouped.values() if len(file_count) == 3)
    logging.info(f"{total_with_all_three} groups have all three required files out of {len(grouped)}")
    
def get_img_types(download_dir: str) -> set:
    """
    Get a set of unique image file extensions in the download directory.

    Args:
        download_dir: Directory containing downloaded posts

    Returns:
        A set of unique image file extensions
    """
    img_types = set()

    for root, _, files in os.walk(os.path.expanduser(download_dir), topdown=False):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in {".jpg", ".jpeg", ".png", ".gif"}:  # Add more extensions if needed
                img_types.add(ext.lower())

    logging.info(f"Found image types: {img_types}")
    return img_types

    
def get_caption_lengths(download_dir: str) -> None:
    
    caption_lengths = defaultdict(int)

    for root, _, files in os.walk(os.path.expanduser(download_dir), topdown=False):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in {".txt"}:  
                file_path = os.path.join(root, file)
                with open(file_path, "r") as  f:
                   text=  f.read().rstrip()
                   caption_lengths[len(text)]+=1
                    
    caption_lengths = sorted(caption_lengths.items(), key=lambda x: x[1])
    count_greater_than_248 = sum(freq   for len, freq in caption_lengths  if len >248)
    count_greater_than_77 = sum(freq   for len, freq in caption_lengths  if len >77)

    print(f"{count_greater_than_248} posts exceed Long CLIP max 248 context length")
    print(f"{count_greater_than_77} posts exceed  CLIP max 77 context length")

    

