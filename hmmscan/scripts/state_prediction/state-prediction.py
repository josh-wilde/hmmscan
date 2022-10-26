from typing import List
import pandas as pd
import sys
import os

from hmmscan.utils.load_data_utils import get_ae_path
from hmmscan.utils.model_initialization_utils import initialize_model

def main():
    print('--Parsing cmd line args--')
    initialization_stubfpath: str = sys.argv[1]  # like 'scans/use_case/best_initializations/by_date_ex_iqr.csv'
    dirname: str = sys.argv[2]  # like 'use_case'
    sequence_name: str = sys.argv[3]  # like 'dfc_by_date_ex_iqr_outliers'
    ae_type: str = sys.argv[4]  # like 'serious_std'
    n_states: int = int(sys.argv[5])
    n_mix_comps: int = int(sys.argv[6])

    print('--Getting model initialization data--')

    # Get paths for high level subdirs
    results_dir_path = get_ae_path('results')
    data_dir_path = get_ae_path('data')

    # Pull in the initialization we want to predict on
    # Obtain the initialization row
    init_df: pd.DataFrame = pd.read_csv(os.path.join(results_dir_path, initialization_stubfpath))
    init_row_df: pd.DataFrame = init_df.loc[
        (init_df['dir'] == dirname)
        & (init_df['sequence_name'] == sequence_name)
        & (init_df['ae_type'] == ae_type)
        & (init_df['n_states'] == n_states)
        & (init_df['n_mix_comps'] == n_mix_comps),
        :
    ]
    if init_row_df.shape[0] > 1:
        raise ValueError(
            f"Dataframe filters on {initialization_stubfpath} ",
            f"specify {init_row_df.shape[0]} rows, should be a single row."
        )
    init_row_num: int = init_row_df.index[0]
    init_series: pd.Series = init_row_df.squeeze()

    print('--Initializing model--')
    m = initialize_model(
        init_series,
        'fully_specified',
        init_file_row=init_row_num,
        init_file_name=initialization_stubfpath
    )

    print('--Get prediction sequence--')

    # Pull in the sequence that we need
    prediction_sequence_fpath: str = os.path.join(data_dir_path, dirname, sequence_name + '.csv')
    pred_seq_df: pd.DataFrame = pd.read_csv(prediction_sequence_fpath)
    pred_sequence: pd.Series = pred_seq_df[ae_type]

    print('--Making predictions--')

    # Run viterbi algorithm
    # Returns the integer list of states without the initialization
    viterbi_no_init: List[int] = m.predict_states(pred_sequence.to_numpy())
    output_df: pd.DataFrame = pd.concat([
        pd.DataFrame({'viterbi': viterbi_no_init}), pred_sequence.to_frame()
    ], axis=1)

    print('--Writing output--')

    # Save the predictions (and aes?) in a user friendly place
    model_structure_str: str = 's' + str(n_states) + 'c' + str(n_mix_comps)
    output_dir: str = os.path.join(results_dir_path, 'state_prediction', dirname)
    os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(
        os.path.join(output_dir, f"{sequence_name}_{ae_type}_{model_structure_str}.csv"),
        index=False
    )


if __name__ == '__main__':
    main()
