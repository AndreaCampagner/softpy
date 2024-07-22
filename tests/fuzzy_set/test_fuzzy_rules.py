import sys
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzy_control import DNFRule, MamdaniRule, TSKRule
from softpy.fuzzy.fuzzyset import FuzzySet, GaussianFuzzySet, TrapezoidalFuzzySet
from softpy.fuzzy.operations import ContinuousFuzzyCombination, DiscreteFuzzyCombination, maximum, minimum
from tests.fuzzy_set.configuration import not_raises, generate_plot

class TestMandamiRule:

    __PATH = './plots_mamdani_rule_aggregation/'

    @pytest.mark.parametrize(
            "premises,name_conseguence,tnorm_operation,exception_expected", 
            [
                ({'temp1': GaussianFuzzySet(0, 1), 
                  'temp2': GaussianFuzzySet(1, 1)}, 'cons', minimum, None),
                ({'temp1': GaussianFuzzySet(0, 1), 
                  'temp2': GaussianFuzzySet(1, 1), 
                  'temp3': GaussianFuzzySet(2, 1)}, 'cons', minimum, None),
                ('a', 'cons', minimum, TypeError),
                ({}, 'cons', minimum, ValueError),
                ({'temp': GaussianFuzzySet(0, 1)}, 'cons', minimum, ValueError),
                ({'temp1': GaussianFuzzySet(0, 1), 
                  'temp2': GaussianFuzzySet(1, 1)}, 1, minimum, TypeError),
                ({'temp1': GaussianFuzzySet(0, 1), 
                  'temp2': GaussianFuzzySet(1, 1)}, '', minimum, ValueError),
                ({'temp1': GaussianFuzzySet(0, 1), 
                  'temp2': GaussianFuzzySet(1, 1)}, 'cons', 'a', TypeError),
                ({1: GaussianFuzzySet(0, 1), 
                  'temp2': GaussianFuzzySet(1, 1)}, 'cons', minimum, TypeError),
                ({'temp1': 'a', 
                  'temp2': GaussianFuzzySet(1, 1)}, 'cons', minimum, TypeError),
            ])
    def test_creation(self,
                      premises : dict[FuzzySet], 
                      name_conseguence : str, 
                      tnorm_operation: Callable[[FuzzySet, FuzzySet], ContinuousFuzzyCombination | DiscreteFuzzyCombination],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = MamdaniRule(premises, name_conseguence, tnorm_operation)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = MamdaniRule(premises, name_conseguence, tnorm_operation)

    @pytest.mark.parametrize(
            "mandami_rule,name_file", 
            [
                (MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 
                              'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1'), 'gaussian'),
                (MamdaniRule({'Gaussian(0,4)': GaussianFuzzySet(0, 4), 
                              'Trapezoidal(0, 1, 2, 3)': TrapezoidalFuzzySet(0, 1, 2, 3)}, 'cons2'), 'gaussian-trapezoidal'),
            ])
    def test_evaluate(self,
                      mandami_rule: MamdaniRule, 
                      name_file: str):
        d = {}
        for name, fuzzy in mandami_rule.premises.items():
            d[name] = fuzzy.memberships_function
        generate_plot(mandami_rule.rule.memberships_function, [], self.__PATH, name_file, additional_call=d)

class TestDNFRule:

    __PATH = './plots_dnf_rule_aggregation/'

    @pytest.mark.parametrize(
            "premises,name_conseguence,tnorm_operation,tconorm_operation,exception_expected", 
            [
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 'cons', minimum, maximum, None),
                ('a', 'cons', minimum, maximum, TypeError),
                ([], 'cons', minimum, maximum, ValueError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 'cons', 'a', maximum, TypeError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 'cons', minimum, 'a', TypeError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 1, minimum, maximum, TypeError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], '', minimum, maximum, ValueError),
            ])
    def test_creation(self,
                      premises : dict[FuzzySet], 
                      name_conseguence : str, 
                      tnorm_operation: Callable[[FuzzySet, FuzzySet], ContinuousFuzzyCombination | DiscreteFuzzyCombination],
                      tconorm_operation: Callable[[FuzzySet, FuzzySet], ContinuousFuzzyCombination | DiscreteFuzzyCombination],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = DNFRule(premises, name_conseguence, tnorm_operation, tconorm_operation)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = DNFRule(premises, name_conseguence, tnorm_operation, tconorm_operation)

    @pytest.mark.parametrize(
            "dnf_rule,name_file", 
            [
                (DNFRule([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp2': TrapezoidalFuzzySet(1, 2, 3, 4), 'temp3': TrapezoidalFuzzySet(3, 4, 5, 6)}
                    ], 'cons1'), 'gaussian-trapezoidal'),
            ])
    def test_evaluate(self,
                      dnf_rule: DNFRule, 
                      name_file: str):
        d = {}
        i = 0
        for fuzzy in dnf_rule.or_clausule_premises:
            d['clausole' + str(i)] = fuzzy.rule.memberships_function
            i = i + 1
        generate_plot(dnf_rule.rule.memberships_function, [], self.__PATH, name_file, additional_call=d)

class TestTSKRule:
    __PATH = './plots_tsk_rule_aggregation/'
    
    @pytest.mark.parametrize(
            "premises,weights,name_conseguence,exception_expected", 
            [
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0, 0.5, 0.5], 'cons', None),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0, 0.7, 0.3], 'cons', None),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0.2, 0.4, 0.4], 'cons', None),
                ('a', [0.2, 0.4, 0.4], 'cons', TypeError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1)}, [0.2, 0.4, 0.4], 'cons', ValueError),
                ({1: GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0.2, 0.4, 0.4], 'cons', TypeError),
                ({'Gaussian(0, 1)': 'a', 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0.2, 0.4, 0.4], 'cons', TypeError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, 'a', 'cons', TypeError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0.5, 0.5], 'cons', ValueError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0, 1, 0], 'cons', None),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0, 2, 0], 'cons', ValueError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0, -2, 0], 'cons', ValueError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [-0.5, 0.5, 1], 'cons', ValueError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0.2, 0.4, 0.4], '', ValueError),
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0.2, 0.4, 0.4], 1, TypeError),
            ])
    def test_creation(self,
                      premises : dict[FuzzySet], 
                      weights: list[np.number],
                      name_conseguence : str, 
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = TSKRule(premises, weights, name_conseguence)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = TSKRule(premises, weights, name_conseguence)

    @pytest.mark.parametrize(
            "tsk_rule,name_file", 
            [
                (TSKRule({'Gaussian(0, 1)': GaussianFuzzySet(2, 1), 
                          'Gaussian(2, 1)': GaussianFuzzySet(1, 1)}, [0, 0.5, 0.5], 'cons'), 'gaussian-trapezoidal'),
            ])
    def test_evaluate(self,
                      tsk_rule: TSKRule,
                      name_file: str):

        d = {}
        for k, fuzzy in tsk_rule.premises.items():
            d[k] = fuzzy.memberships_function
        generate_plot(tsk_rule.memberships_function, [], self.__PATH, name_file, additional_call=d)