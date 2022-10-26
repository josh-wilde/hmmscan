from typing import List, Dict
import os
import pandas as pd
from tqdm import tqdm
from hmmscan.utils.load_data_utils import get_ae_path

dirpath: str = os.path.join(get_ae_path('validation'), 'simulation', 'fits')
print('---Gathering paths---')
eid_fpaths: Dict = {
    's1': [f"eid_{i:04d}" for i in range(1, 31)],
    's2c1': [f"eid_{i:04d}" for i in range(31, 481)],
    's2c2': [f"eid_{i:04d}" for i in range(481, 1831)],
    's3': [f"eid_{i:04d}" for i in range(1831, 2911)],
    's4': [f"eid_{i:04d}" for i in range(2911, 3031)]
}

print('---Reading CSVs---')
for structure in eid_fpaths:
    print(f'-----Reading {structure}')
    eids: List[str] = eid_fpaths[structure]
    dfs: List[pd.DataFrame] = [
        pd.read_csv(os.path.join(dirpath, f"{eid}.csv"), index_col=0)
        for eid in tqdm(eids)
        if (
            (os.path.exists(os.path.join(dirpath, f"{eid}.csv")))
            and (os.path.getsize(os.path.join(dirpath, f"{eid}.csv")) > 0)
        )
    ]
    if len(dfs) > 0:
        print('-----Concating dfs')
        output: pd.DataFrame = pd.concat(dfs, axis=0)
        print('-----Saving output')
        output.to_csv(os.path.join(get_ae_path('validation'), 'simulation', 'fits', f'{structure}.csv'))
        del dfs
        del output
    else:
        print('-----No dfs.')
