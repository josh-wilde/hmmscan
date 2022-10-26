from typing import List
import os
import sys
from tqdm import tqdm
import pandas as pd

# Take in directory from projects
input_dirname: str = sys.argv[1]
output_dirname: str = sys.argv[2]

# List all of the file names
fname_list: List[str] = os.listdir(input_dirname)

# Rename all files
for fname in tqdm(fname_list):
    df: pd.DataFrame = pd.read_csv(os.path.join(input_dirname, fname))
    df['sequence_name'] = df['sequence_name'].str.replace('enbrel_amgen_us_', '')
    df['generating_model_spec'] = df['generating_model_spec'].str.replace('enbrel_amgen_us_', '')
    df.to_csv(os.path.join(output_dirname, fname.replace('enbrel_amgen_us_', '')))
