import numpy as np
from hmmscan.BinomialModel.distribution import BinomialDistribution
from hmmscan.BinomialModel.wrappers import (
    SingleBinomialModel, SingleBinomialHMM, BinomialMixtureModel, BinomialMixtureHMM
)
from hmmscan.DataLoader.ManualLoader import ManualLoader

bd: BinomialDistribution = BinomialDistribution(n=100, p=0.01)
assert bd.parameters == (100, 0.01)

single_binomial: SingleBinomialModel = SingleBinomialModel(n=100)
single_binomial_hmm: SingleBinomialHMM = SingleBinomialHMM(n_states=2, n=100)
binomial_mixture_model: BinomialMixtureModel = BinomialMixtureModel(n_comps=2, n=100)
binomial_mixture_hmm: BinomialMixtureHMM = BinomialMixtureHMM(n_states=2, n_comps=2, n=1000)

# Create ManualDataLoaders to hold some fake data
manual_dl_11: ManualLoader = ManualLoader(
    sequence=np.array([4, 5, 6, 6, 7, 8, np.nan])
)
manual_dl_12: ManualLoader = ManualLoader(
    sequence=np.array([4, 5, 6, 6, 7, 8, 79, 80, 81, 81, 82, 83, np.nan])
)
manual_dl_21: ManualLoader = ManualLoader(
    sequence=np.array(4 * [4, 5, 6, 6, 7, 8, 79, 80, 81, 81, 82, 83])
)
manual_dl_22: ManualLoader = ManualLoader(
    sequence=np.array(
        ([5, 7, 80, 82] + [600, 602, 800, 802] + [80, 5, 82, 7] + [800, 802, 600, 602]) * 30
    )
)

# single_binomial.fit(manual_dl_11)
# print(single_binomial.get_comp_params())

# binomial_mixture_model.fit(manual_dl_12)
# print(binomial_mixture_model.get_comp_params())

# single_binomial_hmm.fit(manual_dl_21)
# print(single_binomial_hmm.get_comp_params())
# print(np.allclose(np.exp(binomial_mixture_model.model.weights), [0.5, 0.5], atol=0.01))

binomial_mixture_hmm.fit(manual_dl_22, verbose=False)
print(binomial_mixture_hmm.get_comp_params())
