import os
import json
import lzma
from typing import List


def find_files_without_metadata(download_dir: str) -> List[str]:
    """Find directories that don't have .xz metadata files."""
    dirs_without_metadata = []
    
    for root, _, files in os.walk(os.path.expanduser(download_dir)):
        has_metadata = False
        for file in files:
            if file.endswith(".xz"):
                try:
                    file_path = os.path.join(root, file)
                    with lzma.open(file_path, "rt", encoding="utf-8") as f:
                        json.load(f)  # Verify it's a valid JSON file
                        has_metadata = True
                        break
                except (lzma.LZMAError, json.JSONDecodeError) as e:
                    print(f"Error reading metadata from {file_path}: {e}")
                    continue
        
        if not has_metadata:
            dirs_without_metadata.append(root)
    
    return dirs_without_metadata



