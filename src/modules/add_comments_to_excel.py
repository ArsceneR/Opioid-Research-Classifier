from modules.count_comments import count_comments
import pandas as pd

def add_comments_to_excel(file_paths, download_dir):
    """
    Append comment counts to the existing Excel files based on metadata in .xz files.

    Args:
        file_paths (List[str]): List of Excel file paths containing a 'Permalink' column.
        download_dir (str): Directory containing downloaded posts.
    """
    # Get comment counts in the format {url: comment_count}
    comment_counts = count_comments(file_paths, download_dir)
    

    for file_path in file_paths:
        # Load Excel into DataFrame
        df = pd.read_excel(file_path)

        # Map 'Permalink' URLs to their comment counts
        df['Comment Count'] = df['Permalink'].map(comment_counts).fillna(-1)

        # Save the updated DataFrame back to the file
        df.to_excel(file_path, index=False)
