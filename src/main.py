import logging
from modules.downloader import batch_post_downloads
from modules.data_reader import get_column_data
from modules.count_comments import count_comments
from modules.add_comments_to_excel import add_comments_to_excel
import modules.analyze_downloads as download_analysis
from modules.clean_data import rename_files
from helpers.random_picture import copy_random_images_with_captions
import random

# file_paths = [
#     './ConversationStreamDistribution/ConversationStreamDistribution_3d42a086-f00d-490c-86c6-39c6b783c1b0_2.xlsx',
#     './ConversationStreamDistribution/ConversationStreamDistribution_ac4cea66-b9fb-4b10-8023-d032dc646d1f_1.xlsx'
# ]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # you can call the modules here, for 

    # random.seed()
    
    # download_analysis.get_caption_lengths(download_dir=my_download_dir)
    
    # selected_file_paths = copy_random_images_with_captions(source_dir_path=opioid_downlaod_dir, n=400, destination_dir_path=dest_dir_path)
   
            

