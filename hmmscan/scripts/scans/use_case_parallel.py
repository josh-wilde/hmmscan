import pandas as pd
import sys
from typing import List
import os

from hmmscan.Scanner import Scanner
from hmmscan.utils.load_data_utils import write_result_csv


def main():
    # There is going to be an index that maps to a point in the (n_components, seed_starter) grid in grid_rseed
    # Assumed that the grid is (1,...,9) components and (0,1,...,49) seed starters
    # This can be used by Slurm array index

    # Other option is to just specify the seed_starter directly

    # Accept command line arguments
    run_type: str = sys.argv[1]  # ['seed_starter', 'grid_component_seed_starter']
    sequence_name: str = sys.argv[2]
    ae_type: str = sys.argv[3]
    n_states: int = int(sys.argv[4])
    grid_index: int = int(sys.argv[5])
    output_subdir: str = sys.argv[6]  # 'by_date_ex_iqr' for paper results

    if run_type == 'seed_starter':
        rseeds: List[int] = [100 * grid_index + i for i in range(10)]
        component_list: List[int] = list(range(1, 10))
        output_fname: str = f"{sequence_name}_{ae_type}_s{str(n_states)}_{str(grid_index)}"
    else:
        # Recover the n_components and seed_starter from the grid_index
        component_list: List[int] = [grid_index // 50 + 1]
        seed_starter: int = grid_index % 50
        rseeds: List[int] = [100 * seed_starter + i for i in range(10)]

        output_fname: str = f"{sequence_name}_{ae_type}_s{str(n_states)}c{component_list[0]}_{str(seed_starter)}"

    # Initialize the scanner
    scanner: Scanner = Scanner(
        states=[n_states],
        components=component_list,
        fixed_ns=[100000],
        fixed_ps=[],
        directory='use_case',
        sequence_names=[sequence_name],
        ae_types=[ae_type],
        gaps=[1],
        init_types=['random'],
        rseeds=rseeds,
        rand_init_ubs=[0.02]
    )

    # Run the scanner
    scanner_output: pd.DataFrame = scanner.run(verbose=False)

    # Save the output
    write_result_csv(
        scanner_output, os.path.join('scans', 'use_case', 'random_initializations', output_subdir), output_fname
    )


if __name__ == '__main__':
    main()
