import logging
from modules.downloader import batch_post_downloads
from modules.data_reader import get_column_data
from modules.count_comments import count_comments
from modules.add_comments_to_excel import add_comments_to_excel
import modules.analyze_downloads as download_analysis
from modules.clean_data import rename_files

file_paths = [
    './ConversationStreamDistribution/ConversationStreamDistribution_3d42a086-f00d-490c-86c6-39c6b783c1b0_2.xlsx',
    './ConversationStreamDistribution/ConversationStreamDistribution_ac4cea66-b9fb-4b10-8023-d032dc646d1f_1.xlsx'
]

my_download_dir = "~/Downloads/All_Downloads" #directory containing downloaded posts

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    download_analysis.get_caption_lengths(download_dir=my_download_dir)
            

