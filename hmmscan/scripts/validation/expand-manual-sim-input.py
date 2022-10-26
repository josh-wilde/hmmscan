from typing import List, Dict, Any
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize, OptimizeResult

from hmmscan.utils.load_data_utils import get_ae_path


def get_transition_probs(params: Dict[str, Any]) -> Dict[str, float]:
    if params.get('n_states', -1) == 1:
        return {
            's0_stat': 1.0,
            's0_s0': 1.0
        }
    elif params.get('n_states', -1) == 2:
        return get_twostate_transition_probs(params)
    elif params.get('n_states', -1) == 3:
        return get_threestate_transition_probs(params)
    elif params.get('n_states', -1) == 4:
        return get_fourstate_transition_probs(params)
    else:
        raise ValueError(f"Can only handle 1-4 state models, not {params.get('n_states', -1)}")


def get_twostate_transition_probs(params: Dict[str, Any]) -> Dict[str, float]:
    s0_stat: float = params.get('input_s0_stat', -1)
    s1_mean_sojourn: float = params.get('mean_sojourn', -1)

    if s0_stat < 0 or s1_mean_sojourn < 1:
        raise ValueError(f"Parameter error. s0_stat = {s0_stat}, s1_soj = {s1_mean_sojourn}.")

    s1_stat: float = 1 - s0_stat
    p11: float = (s1_mean_sojourn - 1) / s1_mean_sojourn  # s1_soj = mean of geometric RV with param 1/(1-p11)
    p10: float = 1 - p11
    p00: float = (s0_stat - p10 * s1_stat) / s0_stat
    p01: float = 1 - p00

    return {
        's0_stat': s0_stat,
        's1_stat': s1_stat,
        's0_s0': p00,
        's0_s1': p01,
        's1_s0': p10,
        's1_s1': p11
    }


def get_threestate_transition_probs(params: Dict[str, Any]) -> Dict[str, float]:
    if np.isnan(params.get('threestate_id', 0)):
        tm_sid: int = 0
    else:
        tm_sid: int = params.get('threestate_id', 0)
    if tm_sid in [1, 2]:
        s0_stat: float = params['input_s0_stat']
        s1_stat: float = (1 - s0_stat) / 2.0
        s2_stat: float = s1_stat
        p10: float = 1 - (params['mean_sojourn'] - 1) / params['mean_sojourn']
        p20: float = p10
        p00: float = (s0_stat - s1_stat*p10 - s2_stat*p20) / s0_stat
        p12: float = params['condp_12'] * (1 - p10)
        p11: float = 1 - p10 - p12

        return {
            's0_stat': s0_stat, 's1_stat': s1_stat, 's2_stat': s1_stat,
            's0_s0': p00, 's0_s1': (1 - p00) / 2, 's0_s2': (1 - p00) / 2,
            's1_s0': p10, 's1_s1': p11, 's1_s2': p12,
            's2_s0': p20, 's2_s1': p12, 's2_s2': p11,
        }
    elif tm_sid in [3, 4]:
        s2_stat: float = params['input_s2_stat']
        s0_stat: float = (1 - s2_stat) / 2.0
        s1_stat: float = s0_stat
        p22: float = (params['mean_sojourn'] - 1) / params['mean_sojourn']
        p02: float = s2_stat * (1 - p22) / (s0_stat + s1_stat)
        p20: float = (1 - p22) / 2.0
        p21: float = (1 - p22) / 2.0
        p01: float = params['condp_01'] * (1 - p02)
        p00: float = 1 - p01 - p02

        return {
            's0_stat': s0_stat, 's1_stat': s1_stat, 's2_stat': s2_stat,
            's0_s0': p00, 's0_s1': p01, 's0_s2': p02,
            's1_s0': p01, 's1_s1': p00, 's1_s2': p02,
            's2_s0': p20, 's2_s1': p21, 's2_s2': p22,
        }
    elif tm_sid == 5:
        s0_stat: float = params['input_s0_stat']
        s1_stat: float = params['input_s1_condstat'] * (1 - s0_stat)
        s2_stat: float = 1 - s0_stat - s1_stat
        p00: float = params['input_s0_s0']
        p01: float = params['condp_01'] * (1 - p00)
        p02: float = 1 - p00 - p01
        p20: float = params['input_s2_s0']
        p21: float = params['condp_21'] * (1 - p20)
        p10: float = (s0_stat - s0_stat * p00 - s2_stat * p20) / s1_stat
        p11: float = (s1_stat - s0_stat * p01 - s2_stat * p21) / s1_stat

        return {
            's0_stat': s0_stat, 's1_stat': s1_stat, 's2_stat': s2_stat,
            's0_s0': p00, 's0_s1': p01, 's0_s2': p02,
            's1_s0': p10, 's1_s1': p11, 's1_s2': 1 - p10 - p11,
            's2_s0': p20, 's2_s1': p21, 's2_s2': 1 - p20 - p21,
        }
    elif tm_sid == 6:
        s0_stat: float = params['input_s0_stat']
        s1_stat: float = params['input_s1_stat']
        s2_stat: float = 1 - s0_stat - s1_stat
        p00: float = (params['mean_sojourn'] - 1) / params['mean_sojourn']
        p01: float = params['condp_01'] * (1 - p00)
        p02: float = 1 - p00 - p01
        p11: float = p00
        p22: float = p00
        p12: float = (s2_stat - s0_stat * p02 - s2_stat * p22) / s1_stat
        p21: float = (s1_stat - s0_stat * p01 - s1_stat * p11) / s2_stat

        return {
            's0_stat': s0_stat, 's1_stat': s1_stat, 's2_stat': s2_stat,
            's0_s0': p00, 's0_s1': p01, 's0_s2': p02,
            's1_s0': 1 - p11 - p12, 's1_s1': p11, 's1_s2': p12,
            's2_s0': 1 - p21 - p22, 's2_s1': p21, 's2_s2': p22,
        }
    elif tm_sid == 7:
        s0_stat: float = params['input_s0_stat']
        s1_stat: float = params['input_s1_stat']
        s2_stat: float = 1 - s0_stat - s1_stat
        p00: float = (params['mean_sojourn'] - 1) / params['mean_sojourn']
        p02: float = params['condp_02'] * (1 - p00)
        p01: float = 1 - p00 - p02
        p11: float = p00
        p22: float = p00
        p12: float = (s2_stat - s0_stat * p02 - s2_stat * p22) / s1_stat
        p21: float = (s1_stat - s0_stat * p01 - s1_stat * p11) / s2_stat

        return {
            's0_stat': s0_stat, 's1_stat': s1_stat, 's2_stat': s2_stat,
            's0_s0': p00, 's0_s1': p01, 's0_s2': p02,
            's1_s0': 1 - p11 - p12, 's1_s1': p11, 's1_s2': p12,
            's2_s0': 1 - p21 - p22, 's2_s1': p21, 's2_s2': p22,
        }
    elif tm_sid == 8:
        p00: float = params['input_s0_s0']
        p01: float = params['condp_01'] * (1 - p00)
        p11: float = params['input_s1_s1']
        p12: float = params['condp_12'] * (1 - p11)
        p20: float = params['input_s2_s0']
        p22: float = params['condp_22'] * (1 - p20)

        return {
            's0_s0': p00, 's0_s1': p01, 's0_s2': 1 - p00 - p01,
            's1_s0': 1 - p11 - p12, 's1_s1': p11, 's1_s2': p12,
            's2_s0': p20, 's2_s1': 1 - p20 - p22, 's2_s2': p22,
        }
    elif tm_sid == 9:
        p00: float = params['input_s0_s0']
        p02: float = params['condp_02'] * (1 - p00)
        p11: float = params['input_s1_s1']
        p10: float = params['condp_10'] * (1 - p11)
        p22: float = params['input_s2_s2']
        p21: float = params['condp_21'] * (1 - p22)

        return {
            's0_s0': p00, 's0_s1': 1 - p00 - p02, 's0_s2': p02,
            's1_s0': p10, 's1_s1': p11, 's1_s2': 1 - p10 - p11,
            's2_s0': 1 - p21 - p22, 's2_s1': p21, 's2_s2': p22,
        }
    elif tm_sid == 10:
        p00: float = params['input_s0_s0']
        p01: float = params['condp_01'] * (1 - p00)
        p11: float = params['input_s1_s1']
        p10: float = params['condp_10'] * (1 - p11)
        p22: float = params['input_s2_s2']
        p21: float = params['condp_21'] * (1 - p22)

        return {
            's0_s0': p00, 's0_s1': p01, 's0_s2': 1 - p00 - p01,
            's1_s0': p10, 's1_s1': p11, 's1_s2': 1 - p10 - p11,
            's2_s0': 1 - p21 - p22, 's2_s1': p21, 's2_s2': p22,
        }
    else:
        raise ValueError(f"Bad tm_sid: {tm_sid}")


def get_fourstate_transition_probs(params: Dict[str, Any]) -> Dict[str, float]:
    s0_stat: float = params['input_s0_stat']
    s1_stat: float = params['input_s1_stat']
    s2_stat: float = params['input_s2_stat']

    output: Dict[str, float] = {
        's0_stat': s0_stat,
        's1_stat': s1_stat,
        's2_stat': s2_stat,
        's3_stat': 1 - s0_stat - s1_stat - s2_stat,
    }

    p00: float = (params['mean_sojourn'] - 1) / params['mean_sojourn']
    p01: float = 0.45 * (1 - p00)
    p03: float = 0.45 * (1 - p00)
    p11: float = p00
    p10: float = 0.45 * (1 - p11)
    p12: float = 0.45 * (1 - p11)
    p22: float = p00
    p21: float = 0.45 * (1 - p22)
    p23: float = 0.45 * (1 - p22)
    p33: float = p00
    p30: float = 0.45 * (1 - p33)
    p32: float = 0.45 * (1 - p33)

    output = (
            output |
            {
                's0_s0': p00, 's0_s1': p01, 's0_s2': 1 - p00 - p01 - p03, 's0_s3': p03,
                's1_s0': p10, 's1_s1': p11, 's1_s2': p12, 's1_s3': 1 - p11 - p10 - p12,
                's2_s0': 1 - p22 - p21 - p23, 's2_s1': p21, 's2_s2': p22, 's2_s3': p23,
                's3_s0': p30, 's3_s1': 1 - p33 - p30 - p32, 's3_s2': p32, 's3_s3': p33,
            }
    )

    return output


def get_ssd_params(params: Dict[str, Any]) -> Dict[str, float]:
    if 2 <= params.get('n_states', -1) <= 4:
        return get_multiplestate_ssd_params(
            n=int(params.get('fixed_n', -1)),
            n_states=int(params.get('n_states', -1)),
            s0_0_param=params.get('s0_0_param', -1),
            ub_add=params.get('ub_add', -1),
            ovl=params.get('ovl', -1),
            ovl_intra=params.get('ovl_intra', -1),
            n_mix_comps=int(params.get('n_mix_comps', -1)),
        )
    elif params.get('n_states', -1) == 1:
        return get_onestate_ssd_params(
            n=int(params.get('fixed_n', -1)),
            s0_0_param=params.get('s0_0_param', -1),
            ub_add=params.get('ub_add', -1),
            ovl_intra=params.get('ovl_intra', -1),
            n_mix_comps=int(params.get('n_mix_comps', -1)),
        )
    else:
        raise ValueError(f"Only implemented for n_states in [1,4], not {params.get('n_states', -1)}")


def get_onestate_ssd_params(
        n: int, s0_0_param: float, ub_add: float, ovl_intra: float, n_mix_comps: int
) -> Dict[str, float]:
    if n_mix_comps != 2:
        raise ValueError(f"Only implementing for n_mix_comps = 2, not {n_mix_comps}")
    if n < 0 or s0_0_param < 0 or ub_add < 0 or ovl_intra < 0:
        raise ValueError(f"Parameter error. n = {n}, s0_0_param = {s0_0_param}, ovl_intra = {ovl_intra}.")

    ovl_results: Dict[str, float] = solve_ovl(
        n=n, p_low=s0_0_param, p_comp_diff=0, ub=s0_0_param + ub_add, ovl=ovl_intra
    )

    return {
        's0_1_param': ovl_results['x'],
        'ovl_intra_actual': ovl_results['ovl_actual'],
        's0_0_wt': 0.5,
        's0_1_wt': 0.5,
    }


def get_multiplestate_ssd_params(
        n: int, n_states: int, s0_0_param: float, ub_add: float, ovl: float, ovl_intra: float, n_mix_comps: int
) -> Dict[str, float]:
    if n < 0 or s0_0_param < 0 or ub_add < 0 or ovl < 0:
        raise ValueError(f"Parameter error. n = {n}, s0_0_param = {s0_0_param}, ovl = {ovl}.")

    output: Dict[str, float]
    if n_mix_comps == 1:
        # Figure out the difference between single component states
        ovl_inter_results: Dict[str, float] = solve_ovl(
            n=n, p_low=s0_0_param, p_comp_diff=0, ub=s0_0_param + ub_add, ovl=ovl
        )
        ovl_inter_diff: float = ovl_inter_results['x'] - s0_0_param  # this is diff in p_high and p_low that induces ovl

        output = {
            'ovl_actual': ovl_inter_results['ovl_actual'],
            's0_0_wt': 1.0,
        }

        for state in range(1, n_states):
            prior_state_param: float = s0_0_param if state == 1 else output[f"s{state-1}_0_param"]
            output[f"s{state}_0_param"] = prior_state_param + ovl_inter_diff
            output[f"s{state}_0_wt"] = 1.0

    elif n_mix_comps == 2:
        # Figure out the intrastate component difference
        ovl_intra_results: Dict[str, float] = solve_ovl(
            n=n, p_low=s0_0_param, p_comp_diff=0, ub=s0_0_param + ub_add, ovl=ovl_intra
        )
        ovl_intra_diff: float = ovl_intra_results['x'] - s0_0_param  # diff in p_high and p_low that induces ovl_intra

        # Then plug this in to get the difference between the two mixtures interstate
        ovl_inter_results: Dict[str, float] = solve_ovl(
            n=n, p_low=s0_0_param, p_comp_diff=ovl_intra_diff, ub=s0_0_param + ovl_intra_diff + ub_add, ovl=ovl
        )

        # Difference between the middle two components, s01 and s10 for example
        ovl_inter_diff: float = ovl_inter_results['x'] - s0_0_param - ovl_intra_diff

        output = {
            'ovl_actual': ovl_inter_results['ovl_actual'],
            'ovl_intra_actual': ovl_intra_results['ovl_actual'],
            's0_1_param': s0_0_param + ovl_intra_diff,
            's0_0_wt': 0.5,
            's0_1_wt': 0.5,
        }
        for state in range(1, n_states):
            output[f"s{state}_0_param"] = output[f"s{state-1}_1_param"] + ovl_inter_diff
            output[f"s{state}_1_param"] = output[f"s{state}_0_param"] + ovl_intra_diff
            output[f"s{state}_0_wt"] = 0.5
            output[f"s{state}_1_wt"] = 0.5
    else:
        raise ValueError(f"Only implemented for 1 and 2 mixture components, not {n_mix_comps}")

    return output


def ovl_error(x: np.ndarray, n: int, p_low: float, ovl: float) -> float:
    return (ovl -
            sum(min(
                binom.pmf(k=k, n=n, p=p_low), binom.pmf(k=k, n=n, p=x[0])
            ) for k in range(n + 1))) ** 2


def ovl_error_twocomp(x: np.ndarray, n: int, p_low: float, p_comp_diff: float, ovl: float) -> float:
    return (ovl -
            sum(min(
                0.5 * binom.pmf(k=k, n=n, p=p_low) + 0.5 * binom.pmf(k=k, n=n, p=p_low+p_comp_diff),
                0.5 * binom.pmf(k=k, n=n, p=x[0]) + 0.5 * binom.pmf(k=k, n=n, p=x[0]+p_comp_diff),
                ) for k in range(n + 1))) ** 2


def solve_ovl(n: float, p_low: float, p_comp_diff: float, ub: float, ovl: float) -> Dict[str, float]:
    if p_comp_diff == 0:
        opt_result: OptimizeResult = minimize(
            fun=ovl_error,
            x0=np.array([p_low]),
            bounds=[(p_low, ub)],
            args=(n, p_low, ovl)
        )
    else:
        opt_result: OptimizeResult = minimize(
            fun=ovl_error_twocomp,
            x0=np.array([p_low+p_comp_diff]),
            bounds=[(p_low+p_comp_diff, ub)],
            args=(n, p_low, p_comp_diff, ovl)
        )
    return {'x': opt_result.x[0], 'ovl_actual': ovl - opt_result.fun ** 2}


def test_twostate_params():
    params: Dict[str, Any] = {
        'n_states': 2,
        'fixed_n': 1000,
        'ovl': 0.05,
        's0_0_param': 0.01,
        'ub_add': 0.05,
        'input_s0_stat': 0.9,
        'mean_sojourn': 2,
        'n_mix_comps': 1
    }

    transition_stuff: Dict[str, Any] = get_transition_probs(params)
    print(transition_stuff)

    ssd_stuff: Dict[str, Any] = get_ssd_params(params)
    print(ssd_stuff)

    params: Dict[str, Any] = {
        'n_states': 2,
        'fixed_n': 1000,
        'ovl': 0.05,
        'ovl_intra': 0.05,
        's0_0_param': 0.01,
        'ub_add': 0.05,
        'input_s0_stat': 0.9,
        'mean_sojourn': 2,
        'n_mix_comps': 2
    }

    transition_stuff = get_transition_probs(params)
    print(transition_stuff)

    ssd_stuff = get_ssd_params(params)
    print(ssd_stuff)


def test_onestate_params():
    params: Dict[str, Any] = {
        'n_states': 1,
        'fixed_n': 1000,
        'ovl': np.nan,
        'ovl_intra': 0.05,
        'n_mix_comps': 2,
        's0_0_param': 0.01,
        'ub_add': 0.03,
        'tm_sid': 'tm12'
    }

    transition_stuff: Dict[str, Any] = get_transition_probs(params)
    print(transition_stuff)

    ssd_stuff: Dict[str, Any] = get_ssd_params(params)
    print(ssd_stuff)


def test_multstate_tms():
    params: Dict[str, Any] = {
        'n_states': 3,
        'threestate_id': 1,
        'tm_sid': 'tm31_01_11',
        'input_s0_stat': 0.9,
        'input_s1_stat': np.nan,
        'input_s2_stat': np.nan,
        'input_s1_condstat': np.nan,
        'mean_sojourn': 2,
        'ovl': 0.05,
        'ovl_intra': 0.05,
        'n_mix_comps': 1,
        'fixed_n': 1000,
        's0_0_param': 0.01,
        'ub_add': 0.05,
        'condp_01': 0.5,
        'condp_02': np.nan,
        'condp_10': np.nan,
        'condp_11': np.nan,
        'condp_12': 0.05,
        'condp_20': np.nan,
        'condp_21': np.nan,
        'condp_22': np.nan,
        'input_s0_s0': np.nan,
        'input_s2_s0': np.nan,
        'input_s1_s1': np.nan,
        'input_s2_s2': np.nan,
    }

    transition_stuff: Dict[str, Any] = get_transition_probs(params)
    print(transition_stuff)


def test_twocomp_params():
    params: Dict[str, Any] = {
        'n_states': 2,
        'fixed_n': 1000,
        'ovl': np.nan,
        'ovl_intra': 0.05,
        'n_mix_comps': 2,
        's0_0_param': 0.01,
        'ub_add': 0.05,
        'tm_sid': 'tm12'
    }

    ssd_stuff: Dict[str, Any] = get_ssd_params(params)
    print(ssd_stuff)


def get_simulation_inputs(sim_dirpath: str, input_fpath: str, output_fpath: str) -> pd.DataFrame:
    # Get the raw inputs dataframe and the outputs
    raw_inputs: pd.DataFrame = pd.read_csv(input_fpath)
    if os.path.exists(output_fpath):
        existing_output: pd.DataFrame = pd.read_csv(output_fpath, index_col=-1)
        raw_inputs = (
            pd.merge(raw_inputs, existing_output['eid'], on=['eid'], indicator=True, how='left')
            .query('_merge=="left_only"')
            .drop('_merge', axis=1)
        )
    else:
        existing_output: pd.DataFrame = pd.DataFrame()

    # Create the new output
    tm_inputs: pd.DataFrame = raw_inputs[
        ['n_states', 'threestate_id', 'mean_sojourn',
         'input_s0_stat', 'input_s1_stat', 'input_s2_stat', 'input_s1_condstat',
         'condp_01', 'condp_02', 'condp_10', 'condp_11', 'condp_12', 'condp_20', 'condp_21', 'condp_22',
         'input_s0_s0', 'input_s2_s0', 'input_s1_s1', 'input_s2_s2'
         ]
    ].drop_duplicates()
    ssd_inputs: pd.DataFrame = raw_inputs[
        ['n_states', 'n_mix_comps', 'fixed_n', 's0_0_param', 'ub_add', 'ovl', 'ovl_intra']
    ].drop_duplicates()

    tm_output_rows: List[pd.Series] = []
    i: int = 1
    for idx, row in tm_inputs.iterrows():
        tm_info: Dict[str, float] = get_transition_probs(dict(row))
        tm_output_rows.append(row.append(pd.Series(tm_info)))
        print(f"Finished {i} of {len(tm_inputs)} tm rows.")
        i += 1
    tm_output_unique: pd.DataFrame = pd.DataFrame(tm_output_rows)

    ssd_output_rows: List[pd.Series] = []
    i: int = 1
    for idx, row in ssd_inputs.iterrows():
        ssd_info: Dict[str, float] = get_ssd_params(dict(row))
        output_row: pd.Series = pd.concat([row, pd.Series(ssd_info)])
        output_row.name = idx
        ssd_output_rows.append(output_row)
        print(f"Finished {i} of {len(ssd_inputs)} ssd rows.")
        i += 1
    ssd_output_unique: pd.DataFrame = pd.DataFrame(ssd_output_rows)
    print('Finished ssd')

    new_output: pd.DataFrame = (
        raw_inputs
        .merge(
            tm_output_unique,
            how='left',
            on=['n_states', 'threestate_id', 'mean_sojourn',
                'input_s0_stat', 'input_s1_stat', 'input_s2_stat', 'input_s1_condstat',
                'condp_01', 'condp_02', 'condp_10', 'condp_11', 'condp_12', 'condp_20', 'condp_21', 'condp_22',
                'input_s0_s0', 'input_s2_s0', 'input_s1_s1', 'input_s2_s2'
                ]
        )
        .merge(
            ssd_output_unique,
            how='left',
            on=['n_states', 'n_mix_comps', 'fixed_n', 's0_0_param', 'ub_add', 'ovl', 'ovl_intra']
        )
    )

    # Add some additional columns
    new_output['dir'] = sim_dirpath
    new_output['sequence_name'] = 'sim'
    new_output['ae_type'] = 'sim'
    new_output['fixed_p'] = np.nan
    new_output['fit_p'] = True

    # Merge the existing output back in during return

    return pd.concat([existing_output, new_output], axis=0)


def main():
    input_fname: str = sys.argv[1]  # 'manual_input.csv'
    output_fname: str = sys.argv[2]  # 'full_simulation_input.csv'

    sim_dirpath: str = os.path.join(get_ae_path('validation'), 'simulation')

    # Get sim input and output file paths
    input_fpath: str = os.path.join(sim_dirpath, input_fname)
    output_fpath: str = os.path.join(sim_dirpath, output_fname)

    sim_inputs: pd.DataFrame = get_simulation_inputs(sim_dirpath, input_fpath, output_fpath)
    sim_inputs.to_csv(output_fpath, index=False)


if __name__ == '__main__':
    main()
