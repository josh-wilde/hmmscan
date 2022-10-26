from typing import List
import os
import glob
import pandas as pd
from tqdm import tqdm
from hmmscan.utils.load_data_utils import get_ae_path


def process(fname):
    return pd.read_csv(fname, index_col=0)


dirpath: str = os.path.join(get_ae_path('validation'), 'use_case', 'fits')
print('---Gathering paths---')
fits_fpaths: List[str] = sorted(glob.glob(os.path.join(dirpath, "*sample*.csv")))
print('---Reading CSVs---')
dfs: List[pd.DataFrame] = [
    process(fname)
    for fname in tqdm(fits_fpaths)
    if os.path.getsize(fname) > 0
]
print('---Concating dfs---')
output: pd.DataFrame = pd.concat(dfs, axis=0)
print('---Saving output---')
output.to_csv(os.path.join(get_ae_path('validation'), 'use_case', 'fits', 'all_fits.csv'))
