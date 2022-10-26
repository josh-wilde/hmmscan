from typing import List, Union
import numpy as np
from pomegranate.hmm import HiddenMarkovModel
from hmmscan.BinomialModel.distribution import BinomialDistribution
from hmmscan.BinomialModel.wrappers import BinModel
from hmmscan.utils.bin_model_utils import get_stat_from_trans


class SingleBinomialHMM(BinModel):
    def __init__(
            self, n_states=None, n=None, p=None, fit_p=True,
            json_string=None, init_type=None, rseed=1, rand_init_ub=0.01, init_file_row=None, init_file_name=None
    ):
        super().__init__()
        self.metadata = self.initialize_metadata(
            n_states, n, p, fit_p, json_string, init_type, rseed, rand_init_ub, init_file_row, init_file_name
        )

        # Initialize model
        self.model = self.initialize_model(json_string)

    @staticmethod
    def initialize_metadata(
            n_states, n, p, fit_p, json_string, init_type, rseed, rand_init_ub, init_file_row, init_file_name
    ):
        metadata = {
            'model_type': 'SingleBinomialHMM',
            'n_mix_comps': 1,
            'init_file_row': init_file_row,
            'init_file_name': init_file_name
        }

        if json_string is None:
            metadata['n_states'] = n_states

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
            example_component = d['states'][0]['distribution']
            metadata['n_states'] = len(d['states']) - 2  # accounts for start and end states
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
        n = self.metadata['fixed_n']
        n_states = self.metadata['n_states']
        rand_ub = self.metadata['rand_init_ub']

        if self.metadata['init_type'] == 'deterministic':
            init_p = {'s'+str(s): (.1*s + .01) * 500.0 / n for s in range(n_states)}
        else:
            rng: np.random.Generator = np.random.default_rng(self.metadata['rseed'])
            init_p = {'s'+str(s): rng.uniform(0, rand_ub) * 500.0 / n for s in range(n_states)}

        return init_p

    def initialize_n(self):
        p = self.metadata['fixed_p']
        n_states = self.metadata['n_states']
        rand_ub = self.metadata['rand_init_ub']

        if self.metadata['init_type'] == 'deterministic':
            init_n = {'s'+str(s): (.1*s + .01) * 500.0 / p for s in range(n_states)}
        else:
            rng: np.random.Generator = np.random.default_rng(self.metadata['rseed'])
            init_n = {'s'+str(s): rng.uniform(0, rand_ub) * 500.0 / p for s in range(n_states)}

        return init_n

    def initialize_model(self, json_string):
        if json_string is None:
            if self.metadata['fit_p']:
                ps = self.initialize_p()
                ns = None
            else:
                ps = None
                ns = self.initialize_n()

            # Initialize base model
            return self.initialize_hmm(ns, ps)
        else:
            return self.init_from_json(json_string)

    def initialize_hmm(self, ns, ps):
        n_states = self.metadata['n_states']
        fit_p = self.metadata['fit_p']

        dists = []  # list of mixture distributions

        if fit_p:
            n = self.metadata['fixed_n']
            for state in ps:
                p = ps[state]
                dists = dists + [BinomialDistribution(n, p, fit_p)]
        else:
            p = self.metadata['fixed_p']
            for state in ns:
                n = ns[state]
                dists = dists + [BinomialDistribution(n, p, fit_p)]

        # Puts 0.6 in diagonals and distributes the rest equally
        trans_mat = (1-0.6)/(n_states-1)*np.ones((n_states, n_states)) + (.6-(1-0.6)/(n_states-1))*np.eye(n_states)

        starts = np.array([float(1/n_states) for _ in range(n_states)])

        return HiddenMarkovModel.from_matrix(trans_mat, dists, starts)

    def init_from_json(self, d):
        dists = []  # list of mixture distributions

        for state in d['states']:
            if state['distribution'] is not None:
                n, p = state['distribution']['parameters']
                fit_p = state['distribution']['fitting p?']
                dists = dists + [BinomialDistribution(n, p, fit_p)]

        trans_mat = self.mat_and_starts_from_edges(d['edges'])
        starts = get_stat_from_trans(trans_mat)

        return HiddenMarkovModel.from_matrix(trans_mat, dists, starts)

    def mat_and_starts_from_edges(self, edges):
        # edges is the edges section from json
        n_states = self.metadata['n_states']

        trans_mat = np.zeros((n_states, n_states))
        for e in edges:
            if e[0] < n_states and e[1] < n_states:
                trans_mat[e[0], e[1]] = e[2]

        return trans_mat

    def fit(self, dl, **fit_args):
        # Fit model
        self.model.fit(
            dl.sequences,
            algorithm="baum-welch",
            verbose=fit_args.get('verbose', True),
            max_iterations=fit_args.get('max_iter', 1e8),
            stop_threshold=fit_args.get('stop_thresh', 1e-9)
        )

        # Flip the is_fit flag
        self.metadata['is_fit'] = True

    def get_log_prob(self, sequences):
        """
        Input: fitted HMM model and list of sequences for fitting
        Output: log probability of the sequences
        """
        # No need to translate sequences
        return sum([self.model.log_probability(x) for x in sequences])

    def get_log_prob_array(self, sequences):
        """
        Input: fitted HMM model and list of sequences to evaluate
        Output: list of log probabilities of each sequence
        """
        # No need to translate sequences
        return np.array([self.model.log_probability(x) for x in sequences])

    def _get_comp_params(self, threshold=1e-7):
        params = {}
        weights = {}

        # Get the transition matrix
        # NA the parameters for the states that are never reached
        trans_mat = self.get_trans_mat()

        for s in self.model.states:
            if s.name[:4] != 'None':
                d = s.distribution
                weights[s.name] = np.array([1.0])

                if self.metadata['fit_p']:
                    raw_param = np.array([d.p])
                else:
                    raw_param = np.array([d.n])

                state_num = int(s.name[-1:])
                params[s.name] = np.where(sum(trans_mat[:, state_num]) > 0, raw_param, np.nan)

        return params, weights

    def _get_trans_mat(self):
        n_states = self.metadata['n_states']
        return self.model.dense_transition_matrix()[:n_states, :n_states]

    def _get_stat_dist(self):
        p = self.get_trans_mat()

        # Get the eigenvectors and eigenvalues
        eig = np.linalg.eig(p.T)

        # Index of eigenvector corresponding to eigenvalue 1
        eig_vec_idx = np.where(np.isclose(eig[0], 1))[0][0]

        # Normalized eigenvector
        stat_dist = np.real(eig[1][:, eig_vec_idx]/sum(eig[1][:, eig_vec_idx]))

        return stat_dist

    def sample(self, n=1, length=1):
        # Same as the BinModel wrapper
        pass

    def predict_states(self, sequences: Union[np.ndarray, List[np.ndarray]]):
        # for each sequence, get the predicted states and then concatenate together (without initial states)
        if type(sequences) == np.ndarray:
            sequences: List[np.ndarray] = [sequences]

        # predict() gives viterbi predictions including the initial state
        # Need to take it off and then concatenate
        viterbi_predictions: List[int] = []
        for sequence in sequences:
            viterbi_predictions += self.model.predict(sequence, algorithm='viterbi')[1:]

        # return predictions without the initial state
        return viterbi_predictions
