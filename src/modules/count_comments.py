import os
import json
import lzma
from typing import Dict, List
from modules.data_reader import get_column_data


def count_comments(file_paths: List[str], download_dir: str) -> Dict[str, int]:
    """
    Count comments for Instagram posts based on metadata in .xz files.

    Args:
        file_paths (List[str]): List of Excel file paths containing URLs.
        download_dir (str): Directory containing downloaded posts.

    Returns:
        Dict[str, int]: A dictionary mapping post URLs to their total comment counts.
    """
    # Initialize a dictionary with URLs from the Excel files
    excel_urls = {url: -1 for url in get_column_data(file_paths)}

    # Walk through the directory to process .xz files
    for root, _, files in os.walk(os.path.expanduser(download_dir)):
        for file in files:
            if file.endswith(".xz"):
                file_path = os.path.join(root, file)
                try:
                    # Uncompress the .xz file using lzma
                    with lzma.open(file_path, "rt", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract the shortcode and comment count
                    shortcode = data["node"]["shortcode"]
                    url = f"https://www.instagram.com/p/{shortcode}/"
                    comment_count = data["node"]["edge_media_to_parent_comment"]["count"]

                    # Update the comment count for the URL if it exists in the Excel data
                    if url in excel_urls:
                        if excel_urls[url] == -1:
                            excel_urls[url] = comment_count
                        else:
                            excel_urls[url] += comment_count

                except (KeyError, json.JSONDecodeError, FileNotFoundError) as e:
                    # Log or handle errors gracefully
                    print(f"Error processing file {file_path}: {e}")

    return excel_urls