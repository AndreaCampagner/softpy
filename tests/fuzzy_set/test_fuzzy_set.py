import sys
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/..")

from softpy.fuzzy.fuzzyset import ContinuousFuzzySet
from softpy.fuzzy.memberships_function import gaussian

@contextmanager
def not_raises():
    try:
        yield
        
    except Exception as err:
        raise AssertionError(
            # "Did raise exception {0} when it should not!".format(
                
            # )
            repr(err)
        )

class TestContinuousFuzzySet:
    @pytest.mark.parametrize(
            "memberships,epsilon,bound,exception_expected", 
            [
                (lambda x: 1 if x == 0 else 0, 1e-3, (0, 0), None), #check membership function
                (1, 1e-3, (0, 0), TypeError),
                (lambda x: 2 if x == 0 else 0, 1e-3, (0, 0), None), 
                (lambda x: 1 if x == 0 else -1, 1e-3, (0, 0), None),
                (gaussian, 1e-3, (0, 1), None),
                (gaussian, (0,1), (0, 1), TypeError), #check epsilon
                (gaussian, 2, (0, 1), ValueError),
                (gaussian, -1e-3, (0, 1), ValueError),
                (gaussian, 1e-3, 2, TypeError), #check bound
                (gaussian, 1e-3, (1, 0), ValueError)
            ])
    def test_creation(self,
                      memberships: Callable, 
                      epsilon: np.number, 
                      bound: tuple, 
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = ContinuousFuzzySet(memberships, epsilon, bound)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = ContinuousFuzzySet(memberships, epsilon, bound)

    @pytest.mark.parametrize(
            "fuzzy_set,alpha_cut,exception_expected", 
            [
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.1, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.2, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.3, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.4, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.5, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.6, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.7, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.7, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.8, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.9, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 0.1, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 2, ValueError),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 'a', TypeError),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), 2.2, ValueError),
                (ContinuousFuzzySet(gaussian, 1e-3, (0, 1)), -1.1, ValueError),
            ])
    def test_alpha_cut(self, 
                       fuzzy_set: ContinuousFuzzySet, 
                       alpha_cut: np.number,
                       exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                step = int((fuzzy_set.bound[1] - fuzzy_set.bound[0] + 1)/fuzzy_set.epsilon)

                x_values = np.linspace(fuzzy_set.bound[0], 
                                       fuzzy_set.bound[1], 
                                       step)
                
                discr_memb_func = np.array([fuzzy_set.memberships_function(x) for x in  x_values])
                alpha_cut_set = np.array([v if v >= alpha_cut else np.nan for v in discr_memb_func])
                assert np.array_equal(alpha_cut_set, fuzzy_set[alpha_cut], equal_nan=True)
        else:
            with pytest.raises(exception_expected) as e_info:
                alpha_cut_set = fuzzy_set[alpha_cut]


    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (ContinuousFuzzySet(gaussian, 1e-3), 'a', TypeError),
                (ContinuousFuzzySet(gaussian, 1e-3), 3, None),
                (ContinuousFuzzySet(gaussian, 1e-3), 0, None),
                (ContinuousFuzzySet(gaussian, 1e-3), 1, None),
                (ContinuousFuzzySet(gaussian, 1e-3), 1.2, None),
                (ContinuousFuzzySet(gaussian, 1e-3), 1.5, None),
                (ContinuousFuzzySet(gaussian, 1e-3), 1.7, None),
                (ContinuousFuzzySet(gaussian, 1e-3, (1, 2)), 2, None),
                (ContinuousFuzzySet(lambda x: 2 if x == 2 else 0, 1e-3, (1, 2)), 2, None),
                (ContinuousFuzzySet(lambda x: 1 if x == 2 else -1, 1e-3, (1, 2)), 2, None),
            ]
            )
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: tuple,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert (fuzzy_set(arg) - (fuzzy_set.memberships_function)(arg)) <= fuzzy_set.epsilon
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

