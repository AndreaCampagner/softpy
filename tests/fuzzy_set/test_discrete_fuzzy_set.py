import sys

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzyset import DiscreteFuzzySet
from tests.fuzzy_set.configuration import not_raises 

class TestDiscreteFuzzySet:
    @pytest.mark.parametrize(
            "items,memberships,dynamic,exception_expected", 
            [
                ([0, 1, 2], [0, 1, 0.5], True, None),
                ([0, 1, 2], [0, 1], True, ValueError),
                ([0, 1, 2], [0, 1, 0], False, None),
                (['a', 1, 2], [0, 1, 0], False, None),
                (['a', 1, 2], ['a', 1, 0], False, TypeError),
                (['a', 1, 2], [2, 1, 0], False, ValueError),
                (['a', 1, 2], [1, 1, 0], 'a', TypeError),
                ('a', [1, 1, 0], False, TypeError),
                (['a', 1, 2], 'a', False, TypeError),
            ])
    def test_creation(self,
                      items: list | np.ndarray, 
                      memberships: list | np.ndarray, 
                      dynamic: bool, 
                      exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = DiscreteFuzzySet(items = items, memberships = memberships, dynamic = dynamic)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = DiscreteFuzzySet(items = items, memberships = memberships, dynamic = dynamic)
    
    @pytest.mark.parametrize(
            "fuzzy_set,alpha,result_expected,exception_expected", 
            [
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 0.5, [1,2], None),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 1, [1], None),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 0, [0, 1, 2], None),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), -5, [1], ValueError),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 5, [1], ValueError),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 'a', [1], TypeError),
            ])
    def test_alpha_cut(self,
                       fuzzy_set: DiscreteFuzzySet, 
                       alpha: np.number,
                       result_expected: list,
                       exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                alpha_cut = fuzzy_set[alpha]
                assert np.array_equal(alpha_cut, np.array(result_expected))
        else:
            with pytest.raises(exception_expected) as e_info:
                alpha_cut = fuzzy_set[alpha]

    @pytest.mark.parametrize(
            "fuzzy_set,element,memberships_expected,exception_expected", 
            [
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 0, 0, None),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 1, 1, None),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 2, 0.5, None),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5]), 10, 0, None),
                (DiscreteFuzzySet([0, 1, 2], [0, 1, 0.5], False), 10, 0, ValueError),
            ])
    def test_memberships(self,
                       fuzzy_set: DiscreteFuzzySet, 
                       element: object,
                       memberships_expected: np.number,
                       exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(element) == memberships_expected
        else:
            with pytest.raises(exception_expected) as e_info:
                memb = fuzzy_set(element)
