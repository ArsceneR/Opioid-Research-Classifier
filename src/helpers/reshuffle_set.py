
from pathlib import Path

import pandas as pd 
import shutil


NUM_FINE_TUNING = 300
NUM_TESTING = 100

def _count_neutrals(root:Path, labels):                                                                                                           
      return sum(1 for p in root.iterdir()
                 if p.is_dir() and labels.get(p.name) == 0)                                                                                    
   
def reshuffle(fine_tuning_path:Path, testing_path:Path, label_path:Path):                                                                                   
    labels = get_human_labels(pd.read_csv(label_path))
    n_ft = _count_neutrals(fine_tuning_path, labels)                                                                                         
    n_te = _count_neutrals(testing_path, labels)                                                                                             
    if n_ft < n_te:
        return reshuffle(testing_path, fine_tuning_path, label_path)                                                                         
                                                                                                                                            
    diff = n_ft - n_te
    if diff == 0:                                                                                                                            
        return  

    # move `diff` neutrals: fine_tuning -> testing                                                                                           
    moved = 0
    for p in list(fine_tuning_path.iterdir()):                                                                                               
        if moved == (diff)//2:                                                                                                                    
            break
        if p.is_dir() and labels.get(p.name) == 0:                                                                                           
            shutil.move(str(p), str(testing_path / p.name))                                                                                  
            moved += 1
    assert moved == diff//2                                                                                                                    
                
    # move `diff` positives back: testing -> fine_tuning (preserve set sizes)                                                                
    moved = 0
    for p in list(testing_path.iterdir()):                                                                                                   
        if moved == (diff)//2:                                                                                                                    
            break
        if p.is_dir() and labels.get(p.name) == 1:                                                                                           
            shutil.move(str(p), str(fine_tuning_path / p.name))
            moved += 1                                                                                                                       
    assert moved == diff//2




def get_human_labels(df:pd.DataFrame)->dict:
    id_col = None
    relevance_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'id' in col_lower and id_col is None:
            id_col = col
        if 'relevance' in col_lower and relevance_col is None:
            relevance_col = col

    if not id_col or not relevance_col:
        raise ValueError(f"Need 'id' and 'relevance' columns. Found: {list(df.columns)}")


    # Create mapping: id -> human relevance (1=opioid, 0=neutral)
    human_labels = {}
    for _, row in df.iterrows():
        post_id = str(row[id_col]).strip()
        relevance = int(row[relevance_col])
        human_labels[post_id] = relevance
    
    return human_labels
