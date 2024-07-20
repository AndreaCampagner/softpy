import sys
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzy_control import MamdaniRule
from softpy.fuzzy.fuzzyset import FuzzySet, GaussianFuzzySet, TrapezoidalFuzzySet
from softpy.fuzzy.operations import ContinuousFuzzyCombination, DiscreteFuzzyCombination, minimum
from tests.fuzzy_set.configuration import not_raises, generate_plot

class TestMandamiRule:

    __PATH = './plots_mamdani_rule_aggregation/'

    @pytest.mark.parametrize(
            "premises,tnorm_operation,exception_expected", 
            [
                ([], minimum, ValueError),
                ([GaussianFuzzySet(0, 1)], minimum, ValueError),
                ([GaussianFuzzySet(0, 1), GaussianFuzzySet(1, 1)], minimum, None),
                ([GaussianFuzzySet(0, 1), GaussianFuzzySet(1, 1)], minimum, None),
                ([GaussianFuzzySet(0, 1), GaussianFuzzySet(1, 1), GaussianFuzzySet(1, 1)], minimum, None),
                ('a', minimum, TypeError),
                ([GaussianFuzzySet(0, 1), GaussianFuzzySet(1, 1)], 'a', TypeError),
            ])
    def test_creation(self,
                      premises : list[FuzzySet], 
                      tnorm_operation: Callable[[FuzzySet], ContinuousFuzzyCombination | DiscreteFuzzyCombination],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = MamdaniRule(premises, tnorm_operation)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = MamdaniRule(premises, tnorm_operation)

    @pytest.mark.parametrize(
            "mandami_rule,name_file,exception_expected", 
            [
                (MamdaniRule([GaussianFuzzySet(0, 1), 
                              GaussianFuzzySet(1, 1)]), 'gaussian', None),
                (MamdaniRule([GaussianFuzzySet(0, 4), 
                              TrapezoidalFuzzySet(0, 1, 2, 3)]), 'gaussian-trapezoidal', None),
            ])
    def test_evaluate(self,
                      mandami_rule: MamdaniRule, 
                      name_file: str,
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                generate_plot(mandami_rule.rule.memberships_function, [], self.__PATH, name_file)
        else:
            with pytest.raises(exception_expected) as e_info:
                NotImplemented