from typing import List, Dict
import argparse
import os
import pandas as pd
import numpy as np
import sys

from hmmscan.Scanner import Scanner
from hmmscan.utils.load_data_utils import get_ae_path


def run_single_exp_sim_evaluator(
        eid: int,
        models_ref_fstub: str,
        fixed_n: int,
        min_states: int,
        max_states: int,
        min_mix_comps: int,
        max_mix_comps: int,
        sample_index_min: int,
        sample_index_max: int
):
    # For a single experiment, need to get the list of models to fit to the sample
    # Also need to get a list of the samples that need to be fitted
    # This needs to only be the stuff that doesn't already exist
    # Create a dataframe for each experiment that is: eid, sample_id, n_states, n_mix_comps, log_prob, bic

    # Figure out which samples, states, components need to end up in the output file
    # Start with the models reference file
    model_ref: pd.DataFrame = pd.read_csv(
        os.path.join(get_ae_path('validation'), models_ref_fstub)
    )
    models_to_fit: pd.DataFrame = model_ref.loc[
        (eid >= model_ref['eid_min']) & (eid <= model_ref['eid_max']), ['n_states', 'n_mix_comps']
    ]
    # Limit to the model states and components that I want to fit
    models_to_fit: pd.DataFrame = models_to_fit.loc[
        (model_ref['n_states'] >= min_states) & (model_ref['n_states'] <= max_states)
        & (model_ref['n_mix_comps'] >= min_mix_comps) & (model_ref['n_mix_comps'] <= max_mix_comps),
        ['n_states', 'n_mix_comps']
    ]
    # Get the samples file
    experiment_samples: pd.DataFrame = pd.read_csv(
        os.path.join(
            get_ae_path('validation'), 'simulation', 'samples', f"eid_{eid:04d}.csv"
        ),
        index_col=0
    )
    experiment_samples = experiment_samples.loc[
                         (experiment_samples.index >= sample_index_min)
                         & (experiment_samples.index <= sample_index_max), :]
    experiment_sample_ids: pd.DataFrame = experiment_samples.index.to_frame(name='sample_id')

    # Combine the samples and the models to fit
    work_to_do: pd.DataFrame = models_to_fit.join(experiment_sample_ids, how='cross')
    work_to_do['eid'] = eid

    # Pull in any existing output file
    output_fpath: str = os.path.join(get_ae_path('validation'), 'simulation', 'fits', f"eid_{eid:04d}.csv")
    if os.path.exists(output_fpath):
        existing_fits_df: pd.DataFrame = pd.read_csv(output_fpath, index_col=0)
        existing_fits: pd.DataFrame = existing_fits_df[['eid', 'sample_id', 'n_states', 'n_mix_comps']]

        # Get rid of the stuff that already exists
        work_to_do = (
            pd.merge(
                work_to_do, existing_fits,
                on=['eid', 'sample_id', 'n_states', 'n_mix_comps'],
                indicator=True, how='left'
            )
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
        )
    else:
        existing_fits_df: pd.DataFrame = pd.DataFrame()

    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)

    # Loop through the samples and the models
    fits_output_series_dict: Dict = {}

    i = 1
    total_rows = len(work_to_do)
    for row_idx, row in work_to_do.iterrows():
        print(f"Fitting {i} of {total_rows}")
        ae_sequence: np.ndarray = experiment_samples[experiment_samples.index == row['sample_id']].values.reshape(-1)
        scanner: Scanner = Scanner(
            states=[row['n_states']],
            components=[row['n_mix_comps']],
            fixed_ns=[fixed_n],
            fixed_ps=[],
            directory='validation/simulation',
            sequence_names=[f"eid_{eid:04d}_sample{row['sample_id']}"],
            ae_types=['sample'],
            gaps=[1],
            init_types=['random'],
            rseeds=[123*i for i in range(1, 26)],
            rand_init_ubs=[0.20],
            ae_sequence=ae_sequence
        )
        scanner_output: pd.DataFrame = scanner.run(predict=True, verbose=False, max_iter=1000)
        best_bic_init: pd.Series = scanner_output.sort_values(by='bic', ascending=True).iloc[0]
        fits_output: pd.Series = pd.concat([row.drop(['n_states', 'n_mix_comps']), best_bic_init])

        fits_output_series_dict[row_idx] = fits_output

        # Save on every iteration
        output: pd.DataFrame = pd.DataFrame(fits_output_series_dict).T
        output.columns = output.columns.map(str)
        output = pd.concat([existing_fits_df, output], axis=0).reset_index(drop=True)
        output.to_csv(output_fpath)

        i += 1


def run_simulation_evaluator(
        init_file_path: str,
        exp_ids: List[int],
        exp_id_min: int,
        exp_id_max: int,
        models_ref_fstub: str,
        min_states: int,
        max_states: int,
        min_mix_comps: int,
        max_mix_comps: int,
        sample_index_min: int,
        sample_index_max: int
):
    # Create the experiment list
    if len(exp_ids) > 0:
        exp_list: List[int] = exp_ids
    elif -1 < exp_id_min <= exp_id_max and exp_id_max > -1:
        exp_list: List[int] = list(range(exp_id_min, exp_id_max+1))
    else:
        raise ValueError(f'Problem with experiments. exp_ids: {exp_ids}. min, max: {exp_id_min}, {exp_id_max}.')

    for eid in exp_list:
        init_df: pd.DataFrame = pd.read_csv(os.path.join(get_ae_path('validation'), init_file_path))
        fixed_n: int = init_df.loc[init_df['eid'] == eid, 'fixed_n'].values[0]

        run_single_exp_sim_evaluator(
            eid,
            models_ref_fstub,
            fixed_n,
            min_states,
            max_states,
            min_mix_comps,
            max_mix_comps,
            sample_index_min,
            sample_index_max
        )


def run_use_case_evaluator(
        init_file_path: str,
        sequence_name: str,
        ae_type: str,
        eval_n_states: int,
        eval_n_mix_comps: int,
        gen_n_states: int,
        gen_n_mix_comps: int,
        max_states: int,
        max_mix_comps: int,
        sample_index: int
):
    # List of model structures to evaluate
    models_to_fit: pd.DataFrame = (
        pd.DataFrame({'n_states': range(1, max_states + 1)})
        .join(pd.DataFrame({'n_mix_comps': range(1, max_mix_comps + 1)}), how='cross')
    )
    models_to_fit['gen_n_states'] = gen_n_states
    models_to_fit['gen_n_mix_comps'] = gen_n_mix_comps
    models_to_fit['eval_n_states'] = eval_n_states
    models_to_fit['eval_n_mix_comps'] = eval_n_mix_comps

    # Get the samples file
    generating_model_spec: str = f"{sequence_name}_{ae_type}_s{gen_n_states}_c{gen_n_mix_comps}"
    experiment_samples: pd.DataFrame = pd.read_csv(
        os.path.join(get_ae_path('validation'), 'use_case', 'samples', f"{generating_model_spec}.csv"),
        index_col=0
    )
    experiment_samples = experiment_samples.loc[experiment_samples.index == sample_index, :]
    experiment_sample_ids: pd.DataFrame = experiment_samples.index.to_frame(name='sample_id')

    # Get the fixed_n from the initializing model file
    init_df: pd.DataFrame = pd.read_csv(os.path.join(get_ae_path('results'), init_file_path))
    fixed_n: int = (
        init_df.loc[
            (init_df['sequence_name'] == sequence_name)
            & (init_df['ae_type'] == ae_type)
            & (init_df['n_states'] == gen_n_states)
            & (init_df['n_mix_comps'] == gen_n_mix_comps),
            'fixed_n'
        ].values[0]
    )

    # Combine the samples and the models to fit
    work_to_do: pd.DataFrame = models_to_fit.join(experiment_sample_ids, how='cross')
    work_to_do['generating_model_spec'] = generating_model_spec

    # Pull in any existing output file
    output_fpath: str = os.path.join(
        get_ae_path('validation'), 'use_case', 'fits',
        f"{generating_model_spec}_evals{eval_n_states}c{eval_n_mix_comps}_sample{sample_index}.csv"
    )
    if os.path.exists(output_fpath):
        existing_fits_df: pd.DataFrame = pd.read_csv(output_fpath, index_col=0)
        existing_fits: pd.DataFrame = existing_fits_df[['n_states', 'n_mix_comps']]

        # See if the eval structure has been fit and any other model has a lower BIC
        eval_df: pd.DataFrame = (
            existing_fits_df.loc[
                (existing_fits_df['n_states'] == eval_n_states) & (existing_fits_df['n_mix_comps'] == eval_n_mix_comps)
                , :
            ]
        )
        if len(eval_df) > 0:
            eval_bic: float = eval_df['bic'].values[0]
        else:
            eval_bic: float = 0

        min_bic: float = existing_fits_df['bic'].min()

        if eval_bic > min_bic:
            work_to_do: pd.DataFrame = pd.DataFrame()
        else:
            # Get rid of the stuff that already exists
            work_to_do = (
                pd.merge(
                    work_to_do, existing_fits,
                    on=['n_states', 'n_mix_comps'],
                    indicator=True, how='left'
                )
                .query('_merge=="left_only"')
                .drop('_merge', axis=1)
            )
    else:
        existing_fits_df: pd.DataFrame = pd.DataFrame()
        eval_bic: float = 0

    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)

    # Sort so that the evaluation structure is first
    if len(work_to_do) > 0:
        work_to_do['temp_is_eval'] = (
                (work_to_do['n_states'] == eval_n_states) & (work_to_do['n_mix_comps'] == eval_n_mix_comps)
        )
        work_to_do.sort_values(
            by=['temp_is_eval', 'n_states', 'n_mix_comps'], inplace=True, ascending=[False, True, True]
        )
        work_to_do.drop('temp_is_eval', axis=1, inplace=True)

    # Loop through the samples and the models
    fits_output_series_dict: Dict = {}

    i = 1
    total_rows = len(work_to_do)
    for row_idx, row in work_to_do.iterrows():
        print(f"Fitting {i} of {total_rows}")
        ae_sequence: np.ndarray = experiment_samples[experiment_samples.index == row['sample_id']].values.reshape(-1)
        scanner: Scanner = Scanner(
            states=[row['n_states']],
            components=[row['n_mix_comps']],
            fixed_ns=[fixed_n],
            fixed_ps=[],
            directory='validation/use_case',
            sequence_names=[f"{generating_model_spec}_sample{row['sample_id']}"],
            ae_types=['sample'],
            gaps=[1],
            init_types=['random'],
            rseeds=[123*i for i in range(1, 26)],
            rand_init_ubs=[0.02],
            ae_sequence=ae_sequence
        )
        scanner_output: pd.DataFrame = scanner.run(predict=False, verbose=False, max_iter=1000)
        best_bic_init: pd.Series = scanner_output.sort_values(by='bic', ascending=True).iloc[0]
        best_bic_init['sample_id'] = row['sample_id']
        best_bic_init['generating_model_spec'] = generating_model_spec
        best_bic_init['gen_n_states'] = gen_n_states
        best_bic_init['gen_n_mix_comps'] = gen_n_mix_comps
        best_bic_init['eval_n_states'] = eval_n_states
        best_bic_init['eval_n_mix_comps'] = eval_n_mix_comps
        fits_output_series_dict[row_idx] = best_bic_init

        # Save on every iteration
        output: pd.DataFrame = pd.DataFrame(fits_output_series_dict).T
        output.columns = output.columns.map(str)
        output = pd.concat([existing_fits_df, output], axis=0).reset_index(drop=True)
        output.to_csv(output_fpath)

        # Check to see if the last model added has BIC less than eval
        if (best_bic_init['n_states'] == eval_n_states) and (best_bic_init['n_mix_comps'] == eval_n_mix_comps):
            eval_bic = best_bic_init['bic']
        min_bic = output['bic'].min()
        if eval_bic > min_bic:
            break

        print(f"min bic: {min_bic}, eval_bic: {eval_bic}")

        i += 1


def run_ci_evaluator(
    init_file_path: str,
    generating_sequence_name: str,
    generating_ae_type: str,
    generating_n_states: int,
    generating_n_mix_comps: int,
    sample_index_min: int,
    sample_index_max: int
):
    # For a single experiment, need to get the list of models to fit to the sample
    # Also need to get a list of the samples that need to be fitted
    # This needs to only be the stuff that doesn't already exist
    # Create a dataframe for each experiment that is:
    # generating_model_spec, sample_id, n_states, n_mix_comps, log_prob, bic

    # Figure out which samples need to end up in the output file
    # Get the samples file
    generating_model_spec: str = (
        f"{generating_sequence_name}_{generating_ae_type}_s{generating_n_states}_c{generating_n_mix_comps}"
    )
    experiment_samples: pd.DataFrame = pd.read_csv(
        os.path.join(get_ae_path('validation'), 'use_case', 'samples', f"{generating_model_spec}.csv"),
        index_col=0
    )
    experiment_samples = experiment_samples.loc[
                         (experiment_samples.index >= sample_index_min)
                         & (experiment_samples.index <= sample_index_max), :]
    experiment_sample_ids: pd.DataFrame = experiment_samples.index.to_frame(name='sample_id')

    # Get the fixed_n from the initializing model file
    init_df: pd.DataFrame = pd.read_csv(os.path.join(get_ae_path('results'), init_file_path))
    init_row_df: pd.DataFrame = (
        init_df.loc[
            (init_df['sequence_name'] == generating_sequence_name)
            & (init_df['ae_type'] == generating_ae_type)
            & (init_df['n_states'] == generating_n_states)
            & (init_df['n_mix_comps'] == generating_n_mix_comps), :]
    )
    if len(init_row_df) > 1:
        raise ValueError(f"init_row_df has {len(init_row_df)} rows, and it should have 1.")
    init_file_row: int = init_row_df.index[0]
    fixed_n: int = init_row_df['fixed_n'].values[0]

    # Combine the samples and the models to fit
    work_to_do: pd.DataFrame = experiment_sample_ids.copy()
    work_to_do['n_states'] = generating_n_states
    work_to_do['n_mix_comps'] = generating_n_mix_comps
    work_to_do['generating_model_spec'] = generating_model_spec

    # Pull in any existing output file
    output_fpath: str = os.path.join(get_ae_path('validation'), 'ci', 'fits', f"{generating_model_spec}.csv")
    if os.path.exists(output_fpath):
        existing_fits_df: pd.DataFrame = pd.read_csv(output_fpath, index_col=0)
        existing_fits: pd.DataFrame = (
            existing_fits_df[
                ['generating_model_spec', 'sample_id', 'n_states', 'n_mix_comps']
            ]
        )

        # Get rid of the stuff that already exists
        work_to_do = (
            pd.merge(
                work_to_do, existing_fits,
                on=['generating_model_spec', 'sample_id', 'n_states', 'n_mix_comps'],
                indicator=True, how='left'
            )
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
        )
    else:
        existing_fits_df: pd.DataFrame = pd.DataFrame()

    # Loop through the samples and the models
    fits_output_series_dict: Dict = {}

    i = 1
    total_rows = len(work_to_do)
    for row_idx, row in work_to_do.iterrows():
        print(f"Fitting {i} of {total_rows}")
        ae_sequence: np.ndarray = experiment_samples[experiment_samples.index == row['sample_id']].values.reshape(-1)
        scanner: Scanner = Scanner(
            states=[row['n_states']],
            components=[row['n_mix_comps']],
            fixed_ns=[fixed_n],
            fixed_ps=[],
            directory='validation/ci',
            sequence_names=[f"{generating_model_spec}_sample{row['sample_id']}"],
            ae_types=['sample'],
            gaps=[1],
            init_types=['random'],
            rseeds=[123*i for i in range(1, 26)],
            rand_init_ubs=[0.02],
            ae_sequence=ae_sequence
        )
        scanner_output: pd.DataFrame = scanner.run(predict=True, verbose=False, max_iter=1000)
        best_bic_init: pd.Series = scanner_output.sort_values(by='bic', ascending=True).iloc[0]
        best_bic_init['sample_id'] = row['sample_id']
        best_bic_init['generating_model_spec'] = generating_model_spec
        fits_output_series_dict[row_idx] = best_bic_init

        # Save on every iteration
        output: pd.DataFrame = pd.DataFrame(fits_output_series_dict).T
        output.columns = output.columns.map(str)
        output = pd.concat([existing_fits_df, output], axis=0).reset_index(drop=True)
        output.to_csv(output_fpath)

        i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_type", type=str, default='simulation')
    parser.add_argument("--init_file_path", type=str, default='simulation/full_simulation_input.csv')
    parser.add_argument("--min_states", type=int, default=1)
    parser.add_argument("--max_states", type=int, default=4)
    parser.add_argument("--min_mix_comps", type=int, default=1)
    parser.add_argument("--max_mix_comps", type=int, default=9)
    parser.add_argument("--sample_index_min", type=int, default=0)
    parser.add_argument("--sample_index_max", type=int, default=99)

    # Arguments for simulation evaluation
    parser.add_argument('--exp_ids', type=int, nargs="*", default=[])
    parser.add_argument('--exp_id_min', type=int, default=-1)
    parser.add_argument('--exp_id_max', type=int, default=-1)
    parser.add_argument("--models_ref_fstub", type=str, default='simulation/comparison_models.csv')

    # Arguments for ci evaluation
    parser.add_argument('--generating_sequence_name', type=str, default='dfa_by_date_ex_iqr_outliers')
    parser.add_argument('--generating_ae_type', type=str, default='serious_std')
    parser.add_argument('--generating_n_states', type=int, default=0)
    parser.add_argument('--generating_n_mix_comps', type=int, default=0)

    # Arguments for use_case evaluation
    parser.add_argument('--sequence_name', type=str, default='dfa_by_date_ex_iqr_outliers')
    parser.add_argument('--ae_type', type=str, default='serious_std')
    parser.add_argument('--eval_n_states', type=int, default=0)
    parser.add_argument('--eval_n_mix_comps', type=int, default=0)
    parser.add_argument('--grid_index', type=int, default=0)
    parser.add_argument('--sample_index', type=int, default=0)

    args = parser.parse_args()

    if args.validation_type == 'simulation':
        run_simulation_evaluator(
            args.init_file_path,
            args.exp_ids,
            args.exp_id_min,
            args.exp_id_max,
            args.models_ref_fstub,
            args.min_states,
            args.max_states,
            args.min_mix_comps,
            args.max_mix_comps,
            args.sample_index_min,
            args.sample_index_max
        )
    elif args.validation_type == 'use_case':
        gen_n_states: int = args.grid_index // 9 + 1
        gen_n_mix_comps: int = args.grid_index % 9 + 1
        run_use_case_evaluator(
            args.init_file_path,
            args.sequence_name,
            args.ae_type,
            args.eval_n_states,
            args.eval_n_mix_comps,
            gen_n_states,
            gen_n_mix_comps,
            args.max_states,
            args.max_mix_comps,
            args.sample_index
        )
    elif args.validation_type == 'ci':
        run_ci_evaluator(
            args.init_file_path,
            args.generating_sequence_name,
            args.generating_ae_type,
            args.generating_n_states,
            args.generating_n_mix_comps,
            args.sample_index_min,
            args.sample_index_max
        )
    else:
        raise ValueError(f"Validation type must be simulation, use_case, or ci, not {args.validation_type}.")


if __name__ == '__main__':
    main()
