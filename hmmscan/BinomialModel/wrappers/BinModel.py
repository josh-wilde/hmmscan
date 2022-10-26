from typing import List
from math import log
import numpy as np


class BinModel:
    def __init__(self):
        self.model = None  # trained on all data
        self.metadata = {}

    def __repr__(self):
        return self.model.to_json()

    def _get_comp_params(self, threshold=1e-5):
        raise NotImplementedError

    def get_comp_params(self, threshold=1e-5):
        if not self.metadata['is_fit']:
            print('Fit model with all data before evaluating')
            return None
        else:
            return self._get_comp_params(threshold)

    def _get_stat_dist(self):
        raise NotImplementedError

    def get_stat_dist(self):
        if not self.metadata['is_fit']:
            print('Fit model with all data before evaluating')
            return None
        else:
            return self._get_stat_dist()

    def _get_trans_mat(self):
        raise NotImplementedError

    def get_trans_mat(self):
        if not self.metadata['is_fit']:
            print('Fit model with all data before evaluating')
            return None
        else:
            return self._get_trans_mat()

    def get_eval(self, dl, evaluation=None):
        if self.metadata['is_fit']:
            if evaluation == 'log_prob':
                return self.get_log_prob(dl.sequences)
            elif evaluation == 'bic':
                return self.get_bic(dl.sequences)
            else:
                return self.get_mmdl(dl.sequences)
        else:
            print('Fit model with all data before evaluating')
            return np.nan

    def get_log_prob(self, sequences):
        raise NotImplementedError

    def get_log_prob_array(self, sequences):
        raise NotImplementedError

    def get_bic(self, sequences):
        # Reference: https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
        # k = ((parameters for comp dist + mix)*n_mix) - 1
        k = (1+1) * self.metadata['n_mix_comps'] - 1
        p = self.metadata['n_states']**2 + self.metadata['n_states']*k - 1
        t = sum([len(s) for s in sequences])

        log_prob = self.get_log_prob(sequences)

        return -2*log_prob + p*log(t)

    def get_bic_array(self, sequences):
        # Reference: https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
        # k = ((parameters for comp dist + mix)*n_mix) - 1
        k = (1+1)*self.metadata['n_mix_comps'] - 1
        p = self.metadata['n_states']**2 + self.metadata['n_states']*k - 1
        t = np.array([len(s) for s in sequences])

        log_probs = self.get_log_prob_array(sequences)

        return -2*log_probs + p*np.log(t)

    def get_mmdl(self, sequences):
        # Reference: Bicego 2003, p. 1399
        trans_mat_params = self.metadata['n_states'] * (self.metadata['n_states'] - 1)
        init_dist_params = self.metadata['n_states'] - 1
        comp_params = 2*self.metadata['n_mix_comps'] - 1
        t = sum([len(s) for s in sequences])
        stat_dist_x_t = t * self.get_stat_dist()
        stat_dist_x_t_adj = stat_dist_x_t[stat_dist_x_t >= 1]

        log_prob = self.get_log_prob(sequences)

        state_penalty = (trans_mat_params + init_dist_params) * log(t)
        comp_penalty = comp_params*np.nansum(np.log(stat_dist_x_t_adj))

        return -2 * log_prob + state_penalty + comp_penalty

    def get_samples(self, n=1, length=1, path=False, random_state=0):
        # Sampling procedure depends on whether it is HMM or not

        if self.metadata['n_states'] == 1:
            samples = [self.model.sample(n=length, random_state=random_state + i) for i in range(n)]
            if path:
                paths = [np.array([1] * length) for _ in range(n)]
                return samples, paths
            else:
                return samples
        else:
            # This gives back a tuple (sample, path) if path=True
            # Where sample is
            # samples_and_paths
            raw_samples = self.model.sample(n=n, length=length, path=path, random_state=random_state)

            if path:
                samples = []
                paths = []

                for sample_package in raw_samples:
                    samples.append(np.array(sample_package[0]))
                    path = np.array([int(state_obj.name[1:]) + 1 for state_obj in sample_package[1][1:]])
                    paths.append(path)

                return samples, paths
            else:
                return raw_samples
