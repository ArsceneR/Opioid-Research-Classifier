import os
import random
import shutil
from typing import List


def copy_random_images_with_captions(source_dir_path: str, n: int, destination_dir_path: str) -> List[str]:
    """Select up to n random images (and their caption files if present) from source and copy each into
    a new numbered subfolder under destination (1,2,3,...). Returns list of created folder paths.

    - Images and captions are expected to live in the same folder in `source_dir_path`.
    - Captions are any files that share the image filename stem (filename without extension). Multiple
      caption extensions are supported (.txt, .json, .xz, ...).
    - .xz caption files are copied as-is; no decompression occurs.
    """
    source_dir = os.path.abspath(os.path.expanduser(source_dir_path))
    dest_dir = os.path.abspath(os.path.expanduser(destination_dir_path))

    if not os.path.isdir(source_dir):
        raise ValueError(f"The directory {source_dir!r} does not exist or is not a directory.")

    os.makedirs(dest_dir, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    # caption extensions to look for (including compressed .xz)
    caption_exts = {".txt", ".json", ".srt", ".vtt", ".md", ".caption", ".xz"}

    # Gather candidate image file paths (images may be nested in subfolders)
    count = 0
    candidates = []
    for root, _, filenames in os.walk(source_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in image_exts:
                candidates.append(os.path.join(root, fname))
        count+=1
    if not candidates:
        raise ValueError(f"No image files found in the directory {source_dir}.")

    # Randomly select up to n distinct images
    selected = random.sample(candidates, min(n, len(candidates)))

    created_folders: List[str] = []

    for idx, src_image in enumerate(selected, start=1):
        folder_name = str(idx)
        target_folder = os.path.join(dest_dir, folder_name)
        
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
            
        os.makedirs(target_folder)

        # Copy image
        image_basename = os.path.basename(src_image)
        target_image_path = os.path.join(target_folder, image_basename)
        shutil.copy2(src_image, target_image_path)

        # Find and copy caption files with same stem
        stem = os.path.splitext(image_basename)[0]
        src_dir = os.path.dirname(src_image)
        for ext in caption_exts:
            caption_name = stem + ext
            caption_src = os.path.join(src_dir, caption_name)
            if os.path.exists(caption_src) and os.path.isfile(caption_src):
                caption_dst = os.path.join(target_folder, caption_name)
                shutil.copy2(caption_src, caption_dst)

        created_folders.append(target_folder)

    return created_folders
