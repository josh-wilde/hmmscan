import numpy as np
import numba as nb
import numbers
import json
from math import exp, log, lgamma

NEGINF = float("-inf")


# Copied from the pomegranate source code
def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance.
    This function will check to see whether the input seed is a valid seed
    for generating random numbers. This is a slightly modified version of
    the code from sklearn.utils.validation.
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)


@nb.jit(nopython=True)
def fast_pmf(x, n, p):
    """
    Input:
    x: probability of this argument
    n: lot size
    p: prob that an AE is reported
    q: probability that a dose generates an AE
    """
    return exp(fast_log_pmf(x, n, p))


@nb.jit(nopython=True)
def fast_log_pmf(x, n, p):
    """
    Input:
    x: probability of this argument
    n: lot size
    p: probability that a dose generates a reported AE
    """
    return lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) + x * log(p) + (n - x) * log(1 - p)


@nb.jit(nopython=True)
def fast_array_pmf(X, n, p):
    len_X = X.shape[0]
    logp_array = np.zeros(len_X, np.float64)

    for i in range(len_X):
        x = X[i][0]
        if np.isnan(x):
            logp_array[i] = 1.0
        elif x < 0 or p == 0:
            logp_array[i] = 0.0
        else:
            logp_array[i] = fast_pmf(x, n, p)

    return logp_array


@nb.jit(nopython=True)
def fast_array_log_pmf(X, n, p):
    len_X = X.shape[0]
    logp_array = np.zeros(len_X, np.float64)

    for i in range(len_X):
        x = X[i][0]
        if np.isnan(x):
            logp_array[i] = 0.0
        elif x < 0 or p == 0:
            logp_array[i] = NEGINF
        else:
            logp_array[i] = fast_log_pmf(x, n, p)

    return logp_array


@nb.jit(nopython=True)
def fast_summarize(X, weights, column_idx, init_wt_sum, init_x_sum):
    w_sum = 0.0
    x_sum = 0.0
    new_summaries = np.zeros(2)

    for i in range(X.shape[0]):
        if np.isnan(X[i]):
            continue

        w_sum = w_sum + weights[i]
        x_sum = x_sum + X[i] * weights[i]

    new_summaries[0] = init_wt_sum + w_sum
    new_summaries[1] = init_x_sum + x_sum

    return new_summaries


class BinomialDistribution:
    def __init__(self, n, p, fit_p=True, frozen=False):
        self.name = 'BinomialDistribution'
        self.n = n
        self.p = p
        self.d = 1  # dimension - univariate
        self.fit_p = fit_p  # says whether to fit p or fit n
        self.parameters = (self.n, self.p)
        self.summaries = [0, 0]
        self.frozen = frozen

    def __repr__(self):
        return self.to_json()

    def probability(self, X):
        return fast_array_pmf(X, self.n, self.p)

    def log_probability(self, X):
        return fast_array_log_pmf(X, self.n, self.p)

    def sample(self, n=None, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.binomial(self.n, self.p, n)

    def fit(self, items, weights=None, inertia=0.0, column_idx=0):
        """
        Set the parameters of this Distribution to maximize the likelihood of
        the given sample. Items holds some sort of sequence. If weights is
        specified, it holds a sequence of value to weight each item by.
        """

        if self.frozen:
            return

        self.summarize(items, weights, column_idx)
        self.from_summaries(inertia)

    def summarize(self, items, weights=None, column_idx=0):
        # Assumes one dimensional!!!
        X = np.array(items).ravel()

        if weights is None:
            weights = np.ones(X.shape[0])

        new_summaries = fast_summarize(X, weights, column_idx, self.summaries[0], self.summaries[1])

        self.summaries[0] = new_summaries[0]
        self.summaries[1] = new_summaries[1]

    def from_summaries(self, inertia=0.0):

        # If the distribution is frozen, don't bother with any calculation
        if self.frozen is True or self.summaries[0] < 1e-7:
            return

        w_sum, x_sum = self.summaries
        mean_count = x_sum / w_sum
        if self.fit_p:
            p = mean_count / self.n
            self.p = p * (1 - inertia) + self.p * inertia
        else:
            n = mean_count / self.p
            self.n = n * (1 - inertia) + self.n * inertia

        self.parameters = (self.n, self.p)
        self.clear_summaries()

    def clear_summaries(self, inertia=0.0):
        self.summaries = [0, 0]

    def to_json(self, separators=(',', ' :'), indent=4):
        model = {
            'class': 'Distribution',
            'name': 'BinomialDistribution',
            'parameters': self.parameters,
            'fitting p?': self.fit_p,
            'frozen': self.frozen
        }
        return json.dumps(model, separators=separators, indent=indent)

    def to_dict(self):
        return {
            'class': 'Distribution',
            'name': 'BinomialDistribution',
            'parameters': self.parameters,
            'fitting p?': self.fit_p,
            'frozen': self.frozen
        }

    @classmethod
    def from_samples(cls, X, param=-1, fit_p=True, weights=None):
        if fit_p:
            if param == -1:
                param = 100  # param is n if fitting p
            d = BinomialDistribution(param, 0)
        else:
            if param == -1:
                param = 0.05  # param is p if fitting n
        d = BinomialDistribution(0, param, fit_p=False)
        d.summarize(X, weights)
        d.from_summaries()
        return d

    @classmethod
    def blank(cls):
        return BinomialDistribution(0, 0)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)

        return BinomialDistribution(
            n=d['parameters'][0], p=d['parameters'][1], fit_p=d['fitting p?'], frozen=d['frozen']
        )
