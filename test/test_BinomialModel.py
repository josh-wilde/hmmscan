from typing import Dict, Tuple
import numpy as np
from hmmscan.BinomialModel.distribution import BinomialDistribution
from hmmscan.BinomialModel.wrappers import (
    SingleBinomialModel, SingleBinomialHMM, BinomialMixtureModel, BinomialMixtureHMM
)
from hmmscan.DataLoader.ManualLoader import ManualLoader


class TestBinomialDistribution:
    def test_bd_init(self):
        """Just tests to see if binomial distribution initializes correctly"""
        bd: BinomialDistribution = BinomialDistribution(n=500, p=0.01)
        assert bd.parameters == (500, 0.01)


class TestWrappers:
    def test_wrapper_inits(self):
        """Just tests to see if binomial models initialize at all"""
        binomial_mixture_hmm: BinomialMixtureHMM = BinomialMixtureHMM(n_states=2, n_comps=3, n=100)

    def test_11_fit(self):
        """Test to see if the models are fitting correctly"""

        # single binomial model
        manual_dl_11: ManualLoader = ManualLoader(
            sequence=np.array([4, 5, 6, 6, 7, 8, np.nan])
        )
        m: SingleBinomialModel = SingleBinomialModel(n=100)
        m.fit(manual_dl_11)
        component_params: Tuple[Dict[str, np.ndarray]] = m.get_comp_params()

        assert component_params[0]['s0'][0] == 0.06

    def test_12_fit(self):
        # single state binomial mixture
        manual_dl_12: ManualLoader = ManualLoader(
            sequence=np.array([4, 5, 6, 6, 7, 8, 79, 80, 81, 81, 82, 83, np.nan])
        )
        m: BinomialMixtureModel = BinomialMixtureModel(n_comps=2, n=100)
        m.fit(manual_dl_12)
        component_params: Tuple[Dict[str, np.ndarray]] = m.get_comp_params()

        assert np.allclose(component_params[0]['s0'], [0.06, 0.81], atol=0.0001)
        assert np.allclose(component_params[1]['s0'], [0.5, 0.5], atol=0.01)

    def test_21_fit(self):
        # two state single binomial HMM
        manual_dl_21: ManualLoader = ManualLoader(
            sequence=np.array(4 * [4, 5, 6, 6, 7, 8, 79, 80, 81, 81, 82, 83])
        )
        m: SingleBinomialHMM = SingleBinomialHMM(n_states=2, n=100)
        m.fit(manual_dl_21)
        component_params: Tuple[Dict[str, np.ndarray]] = m.get_comp_params()

        assert np.allclose(component_params[0]['s0'][0], [0.06], atol=0.0001)
        assert np.allclose(component_params[0]['s1'][0], [0.81], atol=0.0001)




