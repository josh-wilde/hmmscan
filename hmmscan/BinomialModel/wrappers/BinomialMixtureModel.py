from typing import List, Union
import numpy as np
from pomegranate.gmm import GeneralMixtureModel
from hmmscan.BinomialModel.distribution import BinomialDistribution
from .BinModel import BinModel


class BinomialMixtureModel(BinModel):
    def __init__(
            self, n_comps=None, n=None, p=None, fit_p=True,
            json_string=None, init_type=None, rseed=1, rand_init_ub=0.01, init_file_row=None, init_file_name=None
    ):
        super().__init__()
        self.metadata = self.initialize_metadata(
            n_comps, n, p, fit_p, json_string, init_type, rseed, rand_init_ub, init_file_row, init_file_name
        )

        # Initialize model
        self.model = self.initialize_model(json_string)

    @staticmethod
    def initialize_metadata(
            n_comps, n, p, fit_p, json_string, init_type, rseed, rand_init_ub, init_file_row, init_file_name
    ):
        metadata = {
            'model_type': 'BinomialMixtureModel',
            'n_states': 1,
            'init_file_row': init_file_row,
            'init_file_name': init_file_name
        }

        if json_string is None:
            metadata['n_mix_comps'] = n_comps
            metadata['fit_p'] = fit_p
            metadata['init_type'] = init_type
            metadata['rseed'] = rseed
            metadata['rand_init_ub'] = rand_init_ub
            metadata['is_fit'] = False

            if fit_p:
                metadata['fixed_n'] = n
                metadata['fixed_p'] = None
            else:
                metadata['fixed_n'] = None
                metadata['fixed_p'] = p
        else:
            d = json_string
            example_component = d['distributions'][0]
            metadata['n_mix_comps'] = len(d['distributions'])
            metadata['fit_p'] = example_component['fitting p?']
            metadata['init_type'] = 'from_json'
            metadata['rseed'] = -1
            metadata['rand_init_ub'] = -1
            metadata['is_fit'] = False

            if metadata['fit_p']:
                metadata['fixed_n'] = example_component['parameters'][0]
                metadata['fixed_p'] = None
            else:
                metadata['fixed_n'] = None
                metadata['fixed_p'] = example_component['parameters'][1]

        return metadata

    def initialize_p(self):
        # Output a list of p's
        n = self.metadata['fixed_n']
        n_comps = self.metadata['n_mix_comps']
        rand_ub = self.metadata['rand_init_ub']

        if self.metadata['init_type'] == 'deterministic':
            init_p = [(.1*1 + .01 * (m+1)) * 500.0 / n for m in range(n_comps)]
        else:
            rng: np.random.Generator = np.random.default_rng(self.metadata['rseed'])
            init_p = [rng.uniform(0, rand_ub) * 500.0 / n for _ in range(n_comps)]

        return init_p

    def initialize_n(self):
        # Output a list of n's
        p = self.metadata['fixed_p']
        n_comps = self.metadata['n_mix_comps']
        rand_ub = self.metadata['rand_init_ub']

        if self.metadata['init_type'] == 'deterministic':
            init_n = [(.1*1 + .01 * (m+1)) * 500.0 / p for m in range(n_comps)]
        else:
            rng: np.random.Generator = np.random.default_rng(self.metadata['rseed'])
            init_n = [rng.uniform(0, rand_ub) * 500.0 / p for _ in range(n_comps)]

        return init_n

    def initialize_model(self, json_string):
        if json_string is None:
            return self.initialize_gmm()
        else:
            return self.init_from_json(json_string)

    def initialize_gmm(self):
        comps = []

        if self.metadata['fit_p']:
            ps = self.initialize_p()
            for p in ps:
                comps = comps + [BinomialDistribution(n=self.metadata['fixed_n'], p=p, fit_p=self.metadata['fit_p'])]
        else:
            ns = self.initialize_n()
            for n in ns:
                comps = comps + [BinomialDistribution(n=n, p=self.metadata['fixed_p'], fit_p=self.metadata['fit_p'])]

        return GeneralMixtureModel(comps)

    @staticmethod
    def init_from_json(s):
        d = s

        comps = []
        for c in d['distributions']:
            n, p = c['parameters']
            fit_p = c['fitting p?']
            comps = comps + [BinomialDistribution(n, p, fit_p)]
        mix_wts = d['weights']

        return GeneralMixtureModel(comps, weights=mix_wts)

    def fit(self, dl, **fit_args):

        # Translate sequences into a single np.array
        if fit_args.get('use_single', False):
            # print('Using singles for training')
            train_data = np.concatenate(dl.sequences_single)
        else:
            # print('No singles for training')
            train_data = np.concatenate(dl.sequences)

        # Fit model
        self.model.fit(
            train_data,
            verbose=fit_args.get('verbose', True),
            max_iterations=fit_args.get('max_iter', 1e8),
            stop_threshold=fit_args.get('stop_thresh', 1e-9)
        )

        # Flip the is_fit flag
        self.metadata['is_fit'] = True

    def get_log_prob(self, sequences):
        """
        Input: sequences to evaluate and model to eval on
        Output: log probability of the sequences
        """
        # Deficiency of my BinomialDistribution class
        # Only accepts 2d numpy arrays for log_probability evaluation
        if len(sequences) > 0:
            data = np.concatenate(sequences).reshape(-1, 1)
            return sum(self.model.log_probability(data))
        else:
            return np.nan

    def get_log_prob_array(self, sequences):
        """
        Input: sequences to evaluate and model to eval on
        Output: log probability of each of the sequences
        """
        # Deficiency of my CBinomialDistribution class
        # Only accepts 2d numpy arrays for log_probability evaluation
        if len(sequences) > 0:
            return np.array([sum(self.model.log_probability(seq.reshape(-1, 1))) for seq in sequences])
        else:
            return np.nan

    def _get_comp_params(self, threshold=1e-7):
        weights = np.exp(self.model.weights)

        if self.metadata['fit_p']:
            params = np.array([d.p for d in self.model.distributions])
        else:
            params = np.array([d.n for d in self.model.distributions])

        return {'s0': np.where(weights > threshold, params, np.nan)}, {'s0': weights}

    def _get_trans_mat(self):
        # HMM transition matrix is in the form of a 2D numpy array
        return np.array([[1.0]])

    def _get_stat_dist(self):
        return np.array([1.0])

    def sample(self, n=1, length=1):
        # model.sample(n, random_state=integer)
        pass

    @staticmethod
    def predict_states(sequences: Union[np.ndarray, List[np.ndarray]]):
        # for each sequence, get the predicted states and then concatenate together (without initial states)
        if type(sequences) == np.ndarray:
            sequences: List[np.ndarray] = [sequences]

        # Concatenate all sequences
        predictions: List[int] = []
        for sequence in sequences:
            predictions += [0] * len(sequence)

        # return predictions without the initial state
        return predictions
