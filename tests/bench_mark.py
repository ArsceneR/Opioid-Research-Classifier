import timeit
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ConversationStreamDistribution"))
print(base_dir)
file_paths = [
    os.path.join(base_dir, "ConversationStreamDistribution_3d42a086-f00d-490c-86c6-39c6b783c1b0_2.xlsx"),
    os.path.join(base_dir, "ConversationStreamDistribution_ac4cea66-b9fb-4b10-8023-d032dc646d1f_1.xlsx")
]

# Measure the new version using data_reader
new_duration = timeit.timeit(
    stmt="get_column_data(file_paths)",
    setup=f"from modules.data_reader import get_column_data; file_paths = {file_paths}",
    number=1
)
print(f"New Version Average Duration: {new_duration / 20:.4f} seconds")