from typing import Union, Dict

import numpy as np
import pandas as pd

import json
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import HiddenMarkovModel

from hmmscan.BinomialModel.distribution import BinomialDistribution


def initialize_bin(info):
    info = info.dropna()
    fit_p = bool(info['fit_p'])

    if fit_p:
        n = float(info['fixed_n'])
    else:
        p = float(info['fixed_p'])

    try_param = 's0_0_param'
    if try_param in info:
        if fit_p:
            p = float(info[try_param])
        else:
            n = float(info[try_param])
    else:
        if fit_p:
            p = 0.0
        else:
            n = 0.0

    return BinomialDistribution(n, p, fit_p)


def initialize_mix(info):
    info = info.dropna()

    n_comps = int(info['n_mix_comps'])
    fit_p = bool(info['fit_p'])

    if fit_p:
        n = float(info['fixed_n'])
    else:
        p = float(info['fixed_p'])

    # Parameters for emission distribution
    components = []
    weights = []
    for mix_comp in range(n_comps):
        try_param = 's0_' + str(mix_comp) + '_param'
        try_weight = 's0_' + str(mix_comp) + '_wt'
        if try_param in info:
            if fit_p:
                p = float(info[try_param])
            else:
                n = float(info[try_param])
            wt = float(info[try_weight])
        else:
            if fit_p:
                p = 0.0
            else:
                n = 0.0
            wt = 0.0

        components = components + [BinomialDistribution(n, p, fit_p)]
        weights = weights + [wt]

    return GeneralMixtureModel(components, weights=weights)


def initialize_hmm(info):
    info = info.dropna()

    n_states = int(info['n_states'])
    fit_p = bool(info['fit_p'])

    if fit_p:
        n = float(info['fixed_n'])
    else:
        p = float(info['fixed_p'])

    dists = []  # list of binomial distributions
    trans_mat = np.zeros((n_states, n_states))
    starts = np.zeros(n_states)

    for state in range(n_states):

        # Parameters for emission distribution
        try_param = 's' + str(state) + '_0_param'
        if try_param in info:
            if fit_p:
                p = float(info[try_param])
            else:
                n = float(info[try_param])
        else:
            if fit_p:
                p = 0.0
            else:
                n = 0.0

        dists = dists + [BinomialDistribution(n, p, fit_p)]

        # Transition matrix
        for to_state in range(n_states):
            try_trans_entry = 's' + str(state) + '_s' + str(to_state)
            if try_trans_entry in info:
                trans_mat[state, to_state] = float(info[try_trans_entry])

        # Stationary distribution
        try_stat_entry = 's' + str(state) + '_stat'
        if try_stat_entry in info:
            starts[state] = float(info[try_stat_entry])

    return HiddenMarkovModel.from_matrix(trans_mat, dists, starts)


def initialize_mix_hmm(info):
    info = info.dropna()

    n_states = int(info['n_states'])
    n_comps = int(info['n_mix_comps'])
    fit_p = bool(info['fit_p'])

    if fit_p:
        n = float(info['fixed_n'])
    else:
        p = float(info['fixed_p'])

    dists = []  # list of mixture distributions
    trans_mat = np.zeros((n_states, n_states))
    starts = np.zeros(n_states)

    for state in range(n_states):

        # Parameters for emission distribution
        components = []
        weights = []
        for mix_comp in range(n_comps):
            try_param = 's' + str(state) + '_' + str(mix_comp) + '_param'
            try_weight = 's' + str(state) + '_' + str(mix_comp) + '_wt'
            if try_param in info:
                if fit_p:
                    p = float(info[try_param])
                else:
                    n = float(info[try_param])
                wt = float(info[try_weight])
            else:
                if fit_p:
                    p = 0.0
                else:
                    n = 0.0
                wt = 0.0

            components = components + [BinomialDistribution(n, p, fit_p)]
            weights = weights + [wt]

        dists = dists + [GeneralMixtureModel(components, weights=weights)]

        # Transition matrix
        for to_state in range(n_states):
            try_trans_entry = 's' + str(state) + '_s' + str(to_state)
            if try_trans_entry in info:
                trans_mat[state, to_state] = float(info[try_trans_entry])

        # Stationary distribution
        try_stat_entry = 's' + str(state) + '_stat'
        if try_stat_entry in info:
            starts[state] = float(info[try_stat_entry])

    return HiddenMarkovModel.from_matrix(trans_mat, dists, starts)


def _initialize_full_data_model(
        init_data: pd.Series, init_file_row=None, init_file_name=None, save_json_fpath: str = None
):
    n = float(init_data['fixed_n'])
    n_states = int(init_data['n_states'])
    n_comps = int(init_data['n_mix_comps'])

    if n_states == 1 and n_comps == 1:
        mdl = initialize_bin(init_data)
        if save_json_fpath is not None:
            with open(save_json_fpath, 'w') as outfile:
                json.dump(mdl.to_json(), outfile)  # , sort_keys=True, indent=4
        from hmmscan.BinomialModel.wrappers.SingleBinomialModel import SingleBinomialModel
        return SingleBinomialModel(
            n=n, json_string=json.loads(mdl.to_json()), init_file_row=init_file_row, init_file_name=init_file_name
        )
    elif n_states > 1 and n_comps == 1:
        mdl = initialize_hmm(init_data)
        if save_json_fpath is not None:
            with open(save_json_fpath, 'w') as outfile:
                json.dump(mdl.to_json(), outfile)  # , sort_keys=True, indent=4
        from hmmscan.BinomialModel.wrappers.SingleBinomialHMM import SingleBinomialHMM
        return SingleBinomialHMM(
            n=n, n_states=n_states, json_string=json.loads(mdl.to_json()),
            init_file_row=init_file_row, init_file_name=init_file_name
        )
    elif n_states == 1 and n_comps > 1:
        mdl = initialize_mix(init_data)
        if save_json_fpath is not None:
            with open(save_json_fpath, 'w') as outfile:
                json.dump(mdl.to_json(), outfile)  # , sort_keys=True, indent=4
        from hmmscan.BinomialModel.wrappers.BinomialMixtureModel import BinomialMixtureModel
        return BinomialMixtureModel(
            n=n, n_comps=n_comps, json_string=json.loads(mdl.to_json()),
            init_file_row=init_file_row, init_file_name=init_file_name
        )
    else:
        mdl = initialize_mix_hmm(init_data)
        if save_json_fpath is not None:
            with open(save_json_fpath, 'w') as outfile:
                json.dump(mdl.to_json(), outfile)  # , sort_keys=True, indent=4
        from hmmscan.BinomialModel.wrappers.BinomialMixtureHMM import BinomialMixtureHMM
        return BinomialMixtureHMM(
            n=n, n_states=n_states, n_comps=n_comps, json_string=json.loads(mdl.to_json()),
            init_file_row=init_file_row, init_file_name=init_file_name
        )


def _initialize_minimal_data_model(init_data: Dict):
    n_states: int = init_data['n_states']
    n_mix_comps: int = init_data['n_mix_comps']
    n: int = init_data['n']
    p: int = init_data['p']
    fit_p: bool = init_data['fit_p']
    init_type: str = init_data['init_type']
    rseed: int = init_data['rseed']
    rand_init_ub: int = init_data['rand_init_ub']

    if n_states == 1 and n_mix_comps == 1:
        # Initialize a single binomial
        from hmmscan.BinomialModel.wrappers.SingleBinomialModel import SingleBinomialModel
        return SingleBinomialModel(
            n=n, p=p, fit_p=fit_p, init_type=init_type, rseed=rseed, rand_init_ub=rand_init_ub
        )
    elif n_states == 1 and n_mix_comps > 1:
        # Initialize a binomial mixture
        from hmmscan.BinomialModel.wrappers.BinomialMixtureModel import BinomialMixtureModel
        return BinomialMixtureModel(
            n_comps=n_mix_comps, n=n, p=p, fit_p=fit_p, init_type=init_type, rseed=rseed, rand_init_ub=rand_init_ub
        )
    elif n_states > 1 and n_mix_comps == 1:
        # Initialize a binomial HMM
        from hmmscan.BinomialModel.wrappers.SingleBinomialHMM import SingleBinomialHMM
        return SingleBinomialHMM(
            n_states=n_states, n=n, p=p, fit_p=fit_p, init_type=init_type, rseed=rseed, rand_init_ub=rand_init_ub
        )
    else:
        # Initialize a binomial mixture HMM
        from hmmscan.BinomialModel.wrappers.BinomialMixtureHMM import BinomialMixtureHMM
        return BinomialMixtureHMM(
            n_states=n_states, n_comps=n_mix_comps,
            n=n, p=p, fit_p=fit_p, init_type=init_type, rseed=rseed, rand_init_ub=rand_init_ub
        )


def initialize_model(
        init_data: Union[Dict, pd.Series],
        init_data_type: str,
        init_file_row=None,
        init_file_name=None,
        save_json_fpath: str = None
):

    # Can either initialize from a pandas Series that has all of the individual parameters
    # Or minimal init that does either random or deterministic initialization
    if init_data_type == 'fully_specified':
        fit_p = bool(init_data['fit_p'])
        if not fit_p:
            raise ValueError("Not implemented for fitting n yet.")

        return _initialize_full_data_model(init_data, init_file_row, init_file_name, save_json_fpath)
    else:
        return _initialize_minimal_data_model(init_data)



