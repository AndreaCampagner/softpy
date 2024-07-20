import sys
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.operations import ContinuousFuzzyCombination, ContinuousFuzzyOWA, DiscreteFuzzyOWA, minimum, negation
from softpy.fuzzy.fuzzyset import ContinuousFuzzySet, DiscreteFuzzySet, FuzzySet, GaussianFuzzySet, TriangularFuzzySet
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
    __PATH: str = "./plots_continuous_negation/"

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
                neg_fuzzy = negation(fuzzy_set)
                for n, f in zip(neg_fuzzy.memberships, fuzzy_set.memberships):
                    assert n == 1 - f
        else:
            with pytest.raises(exception_expected) as e_info:
                neg_fuzzy = negation(fuzzy_set)

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
                neg_fuzzy = negation(fuzzy_set)
                generate_plot(neg_fuzzy.memberships_function, [], self.__PATH, name_file)
        else:
            with pytest.raises(exception_expected) as e_info:
                neg_fuzzy = negation(fuzzy_set)

class TestTNorm:
    __PATH: str = "./plots_continuous_t_norm/"

    @pytest.mark.parametrize(
            "left_fuzzy_set,right_fuzzy_set,exception_expected", 
            [
                (DiscreteFuzzySet(['a','b'], [0.5, 1]), DiscreteFuzzySet(['a','b'], [0.3, 0.2]), None),
                ('a', DiscreteFuzzySet(['a','b'], [0.5, 1]), TypeError),
                (DiscreteFuzzySet(['a','b'], [0.5, 1]), 'a', TypeError),
            ])
    def test_discrete(self,
                      left_fuzzy_set: DiscreteFuzzySet, 
                      right_fuzzy_set: DiscreteFuzzySet, 
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                t_fuzzy = minimum(left_fuzzy_set, right_fuzzy_set)
                common_keys = set.intersection(set(t_fuzzy.set_items.keys()),
                                               set(left_fuzzy_set.set_items.keys()),
                                               set(right_fuzzy_set.set_items.keys()))
                for k in common_keys:
                    l = left_fuzzy_set.memberships[left_fuzzy_set.set_items[k]]
                    r = right_fuzzy_set.memberships[right_fuzzy_set.set_items[k]]
                    assert t_fuzzy.memberships[t_fuzzy.set_items[k]] == np.min([l,r])
        else:
            with pytest.raises(exception_expected) as e_info:
                neg_fuzzy = minimum(left_fuzzy_set, right_fuzzy_set)

    @pytest.mark.parametrize(
            "left_fuzzy_set,right_fuzzy_set,name_file,exception_expected", 
            [
                (TriangularFuzzySet(1, 2, 3), TriangularFuzzySet(2, 3, 4), 'triangular', None),
                ('a', TriangularFuzzySet(2, 3, 4), None, TypeError),
                (TriangularFuzzySet(2, 3, 4), 'a', None, TypeError),
            ])
    def test_continuous(self,
                        left_fuzzy_set: ContinuousFuzzySet, 
                        right_fuzzy_set: ContinuousFuzzySet, 
                        name_file: str,
                        exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                neg_fuzzy = minimum(left_fuzzy_set, right_fuzzy_set)
                generate_plot(neg_fuzzy.memberships_function, [], self.__PATH, name_file)
        else:
            with pytest.raises(exception_expected) as e_info:
                neg_fuzzy = minimum(left_fuzzy_set, right_fuzzy_set)