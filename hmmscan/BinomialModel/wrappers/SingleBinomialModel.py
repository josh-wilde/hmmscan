from typing import List, Union
import numpy as np
from hmmscan.BinomialModel.distribution import BinomialDistribution
from hmmscan.BinomialModel.wrappers import BinModel


class SingleBinomialModel(BinModel):
    def __init__(
            self, n=None, p=None, fit_p=True, json_string=None,
            init_type=None, rseed=1, rand_init_ub=0.01, init_file_row=None, init_file_name=None
    ):
        super().__init__()
        self.metadata = self.initialize_metadata(
            n, p, fit_p, json_string, init_type, rseed, rand_init_ub, init_file_row, init_file_name
        )

        # Initialize model
        self.model = self.initialize_model(json_string)

    @staticmethod
    def initialize_metadata(
            n, p, fit_p, json_string, init_type, rseed, rand_init_ub, init_file_row, init_file_name
    ):
        metadata = {
            'model_type': 'SingleBinomialModel',
            'n_states': 1,
            'n_mix_comps': 1,
            'is_fit': False,
            'init_file_row': init_file_row,
            'init_file_name': init_file_name
        }

        if json_string is None:
            metadata['fit_p'] = fit_p
            metadata['init_type'] = init_type
            metadata['rseed'] = rseed
            metadata['rand_init_ub'] = rand_init_ub

            if fit_p:
                metadata['fixed_n'] = n
                metadata['fixed_p'] = None
            else:
                metadata['fixed_n'] = None
                metadata['fixed_p'] = p
        else:
            d = json_string
            metadata['fit_p'] = d['fitting p?']
            metadata['init_type'] = 'from_json'
            metadata['rseed'] = -1
            metadata['rand_init_ub'] = -1

            if metadata['fit_p']:
                metadata['fixed_n'] = d['parameters'][0]
                metadata['fixed_p'] = None
            else:
                metadata['fixed_n'] = None
                metadata['fixed_p'] = d['parameters'][1]

        return metadata

    def initialize_model(self, json_string):
        if json_string is None:
            if self.metadata['fit_p']:
                p = self.initialize_p()
                return BinomialDistribution(n=self.metadata['fixed_n'], p=p, fit_p=self.metadata['fit_p'])
            else:
                n = self.initialize_n()
                return BinomialDistribution(n=n, p=self.metadata['fixed_p'], fit_p=self.metadata['fit_p'])
        else:
            d = json_string
            n, p = d['parameters']
            fit_p = d['fitting p?']
            return BinomialDistribution(n, p, fit_p)

    def initialize_p(self):
        n = self.metadata['fixed_n']
        rand_ub = self.metadata['rand_init_ub']

        if self.metadata['init_type'] == 'deterministic':
            return (.1*1 + .01*(0+1)) * 500.0 / n
        else:
            rng: np.random.Generator = np.random.default_rng(self.metadata['rseed'])
            return rng.uniform(0, rand_ub) * 500.0 / n

    def initialize_n(self):
        p = self.metadata['fixed_p']
        rand_ub = self.metadata['rand_init_ub']

        if self.metadata['init_type'] == 'deterministic':
            return (.1*1 + .01*(0+1)) * 500.0 / p
        else:
            rng: np.random.Generator = np.random.default_rng(self.metadata['rseed'])
            return rng.uniform(0, rand_ub) * 500.0 / p

    def fit(self, dl, **fit_args):

        # Translate sequences into a single np.array, with single lots
        if fit_args.get('use_single', False):
            train_data = np.concatenate(dl.sequences_single)
        else:
            train_data = np.concatenate(dl.sequences)

        # Fit model
        self.model.fit(train_data)

        # Flip the is_fit flag
        self.metadata['is_fit'] = True

    def get_log_prob(self, sequences):
        """
        Input: sequences to evaluate and model to eval on
        Output: log probability of the sequences
        """
        # Deficiency of my CBinomialDistribution class
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
        if self.metadata['fit_p']:
            return {'s0': np.array([self.model.p])}, {'s0': np.array([1.0])}
        else:
            return {'s0': np.array([self.model.n])}, {'s0': np.array([1.0])}

    def _get_trans_mat(self):
        # HMM transition matrix is in the form of a 2D numpy array
        return np.array([[1.0]])

    def _get_stat_dist(self):
        return np.array([1.0])

    def sample(self, n=1, length=1):
        # Same as the BinModel wrapper
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
