import pandas as pd
import os
from hmmscan.utils.load_data_utils import get_ae_path

# Directory to scan
dirname: str = os.path.join(get_ae_path('validation'), 'simulation', 'fits')

# Get all files from directory
for fname in sorted(os.listdir(dirname)):
    if os.path.getsize(os.path.join(dirname, fname)) > 0:
        df: pd.DataFrame = pd.read_csv(os.path.join(dirname, fname))
        if len(df) not in [1100, 1200, 1300]:
            print(f"{fname}: {len(df)}")
