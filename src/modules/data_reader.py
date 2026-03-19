from typing import List
import polars as pl
from concurrent.futures import ThreadPoolExecutor

def read_file(file_path: str) -> pl.DataFrame:
    try:
        return pl.read_excel(file_path, columns=["Permalink"])
    except FileNotFoundError:
        return pl.DataFrame()

def get_column_data(file_paths: List[str]) -> List[str]:
    with ThreadPoolExecutor() as executor:
        dataframes = [df for df in executor.map(read_file, file_paths) if not df.is_empty()]
    
    return pl.concat(dataframes)["Permalink"].to_list() if dataframes else []


