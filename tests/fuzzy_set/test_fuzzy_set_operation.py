import sys
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzy_operation import ContinuousFuzzyCombination, ContinuousFuzzyNegation, ContinuousFuzzyOWA, DiscreteFuzzyCombination, DiscreteFuzzyNegation, DiscreteFuzzyOWA
from softpy.fuzzy.operations import minimum
from softpy.fuzzy.fuzzyset import ContinuousFuzzySet, DiscreteFuzzySet, GaussianFuzzySet, TriangularFuzzySet
from tests.fuzzy_set.configuration import generate_plot, not_raises 

class TestContinuousFuzzyOWA:
    @pytest.mark.parametrize(
            "fuzzysets,weights,exception_expected", 
            [
                ([], [], ValueError),
                ([GaussianFuzzySet(0, 1), GaussianFuzzySet(2, 3)], [0.5, 0.5], None),
                ([GaussianFuzzySet(0, 1), GaussianFuzzySet(2, 3)], [0.6, 0.5], ValueError),
                ([GaussianFuzzySet(0, 1)], [0.6, 0.5], ValueError),
                ([GaussianFuzzySet(0, 1), GaussianFuzzySet(2, 3)], [2, -1], TypeError),
                ([DiscreteFuzzySet(['a', 'b'], [0, 1]), GaussianFuzzySet(2, 3)], [0.5, 0.5], TypeError),
            ])
    def test_creation(self,
                      fuzzysets: list[ContinuousFuzzySet] | np.ndarray | tuple[np.number], 
                      weights: list[np.number] | np.ndarray | tuple[np.number],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = ContinuousFuzzyOWA(fuzzysets, weights)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = ContinuousFuzzyOWA(fuzzysets, weights)

    @pytest.mark.parametrize(
            "owa,arg,exception_expected", 
            [
                (ContinuousFuzzyOWA([GaussianFuzzySet(0, 1), GaussianFuzzySet(2, 3)],
                                    [0.5, 0.5]), 0.5, None),
                (ContinuousFuzzyOWA([GaussianFuzzySet(0, 1), GaussianFuzzySet(2, 3)],
                                    [0.5, 0.5]), [0.5, 1], None),
                (ContinuousFuzzyOWA([GaussianFuzzySet(0, 1), GaussianFuzzySet(2, 3)],
                                    [0.5, 0.5]), [0.5], ValueError),
            ])
    def test_memberships(self,
                         owa: ContinuousFuzzyOWA, 
                         arg: np.number | list[np.number] | tuple[np.number] | np.ndarray,
                         exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = owa(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = owa(arg)

class TestDiscreteFuzzyOWA:
    @pytest.mark.parametrize(
            "fuzzysets,weights,exception_expected", 
            [
                ([], [], ValueError),
                ([DiscreteFuzzySet(['a', 'b'], [0.5, 1]), DiscreteFuzzySet(['c', 'd'], [0.6, 1])], [0.5, 0.5], None),
                ([DiscreteFuzzySet(['a', 'b'], [0, 1]), DiscreteFuzzySet(['c', 'd'], [0, 1])], [0.6, 0.5], ValueError),
                ([DiscreteFuzzySet(['a', 'b'], [0, 1]), DiscreteFuzzySet(['c', 'd'], [0, 1])], [2, -1], TypeError),
                ([DiscreteFuzzySet(['a', 'b'], [0, 1]), GaussianFuzzySet(2, 3)], [0.5, 0.5], TypeError),
                ([DiscreteFuzzySet(['a', 'b'], [0, 1]), DiscreteFuzzySet(['c', 'd'], [0, 1])], [0.5], ValueError),
                ('a', [0.5, 0.5], TypeError),
                ([DiscreteFuzzySet(['a', 'b'], [0, 1]), DiscreteFuzzySet(['c', 'd'], [0, 1])], 'a', TypeError),
            ])
    def test_creation(self,
                      fuzzysets: list[DiscreteFuzzySet] | np.ndarray | tuple[np.number], 
                      weights: list[np.number] | np.ndarray | tuple[np.number],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = DiscreteFuzzyOWA(fuzzysets, weights)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = DiscreteFuzzyOWA(fuzzysets, weights)
    @pytest.mark.parametrize(
            "owa,arg,exception_expected", 
            [
                (DiscreteFuzzyOWA([DiscreteFuzzySet(['a', 'b'], [0, 1]), DiscreteFuzzySet(['c', 'd'], [0, 1])], 
                                  [0.5, 0.5]), 0.5, None),
            ])
    def test_memberships(self,
                         owa: ContinuousFuzzyOWA, 
                         arg: np.number | list[np.number] | tuple[np.number] | np.ndarray,
                         exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = owa(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = owa(arg)

class TestNegation:
    __PATH: str = "./img/plots_continuous_negation/"

    @pytest.mark.parametrize(
            "fuzzy_set,exception_expected", 
            [
                (DiscreteFuzzySet(['a','b'], [0.5, 1]), None),
                ('a', TypeError),
            ])
    def test_discrete(self,
                      fuzzy_set: DiscreteFuzzySet, 
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                neg_fuzzy = DiscreteFuzzyNegation(fuzzy_set)
                for n, f in zip(neg_fuzzy.memberships, fuzzy_set.memberships):
                    assert n == 1 - f
        else:
            with pytest.raises(exception_expected) as e_info:
                neg_fuzzy = DiscreteFuzzyNegation(fuzzy_set)

    @pytest.mark.parametrize(
            "fuzzy_set,name_file,exception_expected", 
            [
                (GaussianFuzzySet(3, 1), 'gaussian', None),
                ('a', None, TypeError),
            ])
    def test_continuous(self,
                        fuzzy_set: ContinuousFuzzySet, 
                        name_file: str,
                        exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                neg_fuzzy = ContinuousFuzzyNegation(fuzzy_set)
                generate_plot(neg_fuzzy.memberships_function, 
                              [], 
                              self.__PATH, 
                              name_file,
                              {'normal': fuzzy_set.memberships_function})
        else:
            with pytest.raises(exception_expected) as e_info:
                neg_fuzzy = ContinuousFuzzyNegation(fuzzy_set)

class TestTNorm:
    __PATH: str = "./img/plots_continuous_t_norm/"

    @pytest.mark.parametrize(
            "fuzzy_set,exception_expected", 
            [
                ([DiscreteFuzzySet(['a','b'], [0.5, 1])], None),
                ([DiscreteFuzzySet(['a','b'], [0.5, 1]), DiscreteFuzzySet(['a','b'], [0.3, 0.2])], None),
                ([DiscreteFuzzySet(['a','b'], [0.5, 1]), DiscreteFuzzySet(['a','b', 'c'], [0.3, 0.2, 0.7]), DiscreteFuzzySet(['a','c'], [0.8, 0.2])], None),
                ('a', TypeError),
                (['a'], TypeError),
            ])
    def test_discrete(self,
                      fuzzy_set: list[DiscreteFuzzySet], 
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                t_fuzzy = DiscreteFuzzyCombination(fuzzy_set, minimum)
                common_keys = set(fuzzy_set[0].set_items.keys())
                if len(fuzzy_set) > 1:
                    for f in fuzzy_set[1:]:
                        common_keys = set.intersection(common_keys, f.set_items.keys())
                
                print(common_keys)
                m = []
                for k in common_keys:
                    for f in fuzzy_set:
                        m.append(f.memberships[f.set_items[k]]) 

                    assert t_fuzzy.memberships[t_fuzzy.set_items[k]] == np.min(m)
                    m = []

        else:
            with pytest.raises(exception_expected) as e_info:
                neg_fuzzy = DiscreteFuzzyCombination(fuzzy_set, minimum)

    @pytest.mark.parametrize(
            "fuzzy_set,name_file,exception_expected", 
            [
                ([TriangularFuzzySet(1, 2, 3)], 'triangular-1', None),
                ([TriangularFuzzySet(1, 2, 3), TriangularFuzzySet(2, 3, 4)], 'triangular-2', None),
                ([TriangularFuzzySet(1, 2, 3), TriangularFuzzySet(2, 3, 4), TriangularFuzzySet(3, 4, 5)], 'triangular-3', None),
                ('a', None, TypeError),
                (['a'], None, TypeError),
            ])
    def test_continuous(self,
                        fuzzy_set: list[ContinuousFuzzySet], 
                        name_file: str,
                        exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                f = ContinuousFuzzyCombination(fuzzy_set, minimum)
                generate_plot(f.memberships_function, 
                              [], 
                              self.__PATH, 
                              name_file,
                              {str(i): fuzzy_set[i].memberships_function for i in range(len(fuzzy_set))})
        else:
            with pytest.raises(exception_expected) as e_info:
                f = ContinuousFuzzyCombination(fuzzy_set, minimum)

                