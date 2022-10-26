from typing import List
import os
import sys
from hmmscan.utils.load_data_utils import get_ae_path

# Max experiment
ae_subdir: str = sys.argv[1]
max_eid: int = int(sys.argv[2])

# Directory to scan
dirpath: str = os.path.join(get_ae_path(), ae_subdir)

file_list: List[str] = os.listdir(dirpath)
eid_list: List[int] = [int(f.replace('.csv', '').replace('eid_', '')) for f in file_list]
missing_list: List[int] = list(set(range(1, max_eid + 1)) - set(eid_list))

print(f"These eids are missing: {sorted(missing_list)}")
print(f"Number of missing eids: {len(missing_list)}")
