import os
import lzma
import json
import logging

# Precondition: each directory has one .xz file and no duplicate shortcodes
def rename_files(download_dir) -> None:
    for root, _, files in os.walk(os.path.expanduser(download_dir), topdown=False):
        for file in files:
            if file.endswith(".xz"):
                file_path = os.path.join(root, file)
                try:
                    with lzma.open(file_path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                        shortcode = data["node"]["shortcode"]
                        new_dir_name = shortcode
                        new_root = os.path.join(os.path.dirname(root), new_dir_name)
                        if root != new_root:
                            os.rename(root, new_root)
                            logging.info(f"Renamed {root} → {new_root}")
                except (lzma.LZMAError, json.JSONDecodeError, KeyError, OSError) as e:
                    logging.error(f"Error processing {file_path}: {e}")
                break  # only one .xz file per directory, so we stop after renaming
