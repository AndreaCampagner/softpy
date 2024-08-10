import sys
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzy_rule import DNFRule, MamdaniRule
from softpy.fuzzy.fuzzy_operation import ContinuousFuzzyCombination, DiscreteFuzzyCombination
from softpy.fuzzy.fuzzyset import FuzzySet, GaussianFuzzySet, LinearSFuzzySet, LinearZFuzzySet, TrapezoidalFuzzySet, TriangularFuzzySet
from softpy.fuzzy.operations import maximum, minimum
from tests.fuzzy_set.configuration import not_raises, generate_plot

class TestMandamiRule:

    __PATH = './plots_mamdani_rule_aggregation/'

    @pytest.mark.parametrize(
            "premises,name_conseguence,conseguence,tnorm_operation,exception_expected", 
            [
                ({'temp1': [GaussianFuzzySet(0, 1)], 
                  'temp2': [GaussianFuzzySet(1, 1)]}, 
                  'cons', 
                  TrapezoidalFuzzySet(0, 1, 2, 3), 
                  minimum, 
                  None),
                ({'temp1': [GaussianFuzzySet(0, 1)], 
                  'temp2': [GaussianFuzzySet(1, 1)], 
                  'temp3': [GaussianFuzzySet(2, 1)]}, 
                  'cons', 
                  TrapezoidalFuzzySet(0, 1, 2, 3), 
                  minimum, 
                  None),
                ('a', 'cons', TrapezoidalFuzzySet(0, 1, 2, 3), minimum, TypeError),
                ({}, 'cons', TrapezoidalFuzzySet(0, 1, 2, 3), minimum, ValueError),
                ({'temp1': GaussianFuzzySet(0, 1)}, 
                 'cons', 
                 TrapezoidalFuzzySet(0, 1, 2, 3), 
                 minimum,
                 TypeError),
                ({'temp1': ['a']}, 
                 'cons', 
                 TrapezoidalFuzzySet(0, 1, 2, 3), 
                 minimum,
                 TypeError),
                ({'temp': [GaussianFuzzySet(0, 1)]}, 
                 'cons', 
                 TrapezoidalFuzzySet(0, 1, 2, 3),
                 minimum, 
                 None),
                ({'temp1': [GaussianFuzzySet(0, 1)], 
                  'temp2': [GaussianFuzzySet(1, 1)]}, 
                  1, 
                  TrapezoidalFuzzySet(0, 1, 2, 3),
                  minimum, 
                  TypeError),
                ({'temp1': [GaussianFuzzySet(0, 1)], 
                  'temp2': [GaussianFuzzySet(1, 1)]}, 
                  '', 
                  TrapezoidalFuzzySet(0, 1, 2, 3),
                  minimum,
                  ValueError),
                ({'temp1': [GaussianFuzzySet(0, 1)], 
                  'temp2': [GaussianFuzzySet(1, 1)]}, 
                  'cons', 
                  'a',
                  minimum, 
                  TypeError),
                ({'temp1': [GaussianFuzzySet(0, 1)], 
                  'temp2': [GaussianFuzzySet(1, 1)]}, 'cons', 
                  TrapezoidalFuzzySet(0, 1, 2, 3),
                  'a', 
                  TypeError),
                ({1: [GaussianFuzzySet(0, 1)], 
                  'temp2': [GaussianFuzzySet(1, 1)]}, 
                  'cons', 
                  TrapezoidalFuzzySet(0, 1, 2, 3),
                  minimum, 
                  TypeError),
                ({'temp1': ['a'], 
                  'temp2': [GaussianFuzzySet(1, 1)]}, 
                  'cons', 
                  TrapezoidalFuzzySet(0, 1, 2, 3),
                  minimum, 
                  TypeError),
            ])
    def test_creation(self,
                      premises : dict[str, list[FuzzySet]], 
                      name_conseguence : str, 
                      conseguence : FuzzySet, 
                      tnorm_operation: Callable[[FuzzySet, FuzzySet], ContinuousFuzzyCombination | DiscreteFuzzyCombination],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = MamdaniRule(premises, name_conseguence, conseguence, tnorm_operation)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = MamdaniRule(premises, name_conseguence, conseguence, tnorm_operation)

    @pytest.mark.parametrize(
            "mandami_rule,input_rule,name_file", 
            [
                (MamdaniRule({'Gaussian(0,1)': [GaussianFuzzySet(0, 1)], 
                              'Gaussian(1,1)': [GaussianFuzzySet(1, 1)]}, 
                              'cons1',
                              TriangularFuzzySet(0, 1.5, 3)),
                              {'Gaussian(0,1)': 0.5, 'Gaussian(1,1)': 1},
                              'gaussian'),
                (MamdaniRule({'Gaussian(0,4)': [GaussianFuzzySet(0, 4)], 
                              'Trapezoidal(0, 1, 2, 3)': [TrapezoidalFuzzySet(0, 1, 2, 3)]}, 
                              'cons2',
                              TriangularFuzzySet(0, 1.5, 3)), 
                              {'Gaussian(0,4)': 3, 'Trapezoidal(0, 1, 2, 3)': 0.5},
                              'gaussian-trapezoidal'),

                (MamdaniRule({'temperature': [LinearZFuzzySet(12, 18, bound=(0, 40))]},
                              'controller', 
                              LinearZFuzzySet(1,2, bound=(-40, 40))),
                              {'temperature': 17.5},
                              'first-rule'),
                (MamdaniRule({'temperature': [LinearZFuzzySet(12, 18, bound=(0, 40)),
                                              TrapezoidalFuzzySet(18, 20, 22, 24, bound=(0, 40))]},
                              'controller', 
                              TrapezoidalFuzzySet(0, 0.5, 1, 1.5, bound=(-40, 40))),
                              {'temperature': 17.5},
                              'second-rule'),
                (MamdaniRule({'temperature': [TrapezoidalFuzzySet(18, 20, 22, 24, bound=(0, 40))]},
                              'controller', 
                              TriangularFuzzySet(-0.1, 0, 0.1, bound=(-40, 40))),
                              {'temperature': 17.5},
                              'third-rule'),
                (MamdaniRule({'temperature': [TrapezoidalFuzzySet(18, 20, 22, 24, bound=(0, 40)),
                                              LinearSFuzzySet(26, 40, bound=(0, 40))]},
                              'controller', 
                              TrapezoidalFuzzySet(-1.5, -1, -0.5, 0, bound=(-40, 40))),
                              {'temperature': 17.5},
                              'fourth-rule'),
                (MamdaniRule({'temperature': [LinearSFuzzySet(26, 40, bound=(0, 40))]},
                              'controller', 
                              LinearSFuzzySet(-2,-1, bound=(-40, 40))),
                              {'temperature': 17.5},
                              'fifth-rule')
            ])
    def test_evaluate(self,
                      mandami_rule: MamdaniRule, 
                      input_rule: dict[str,np.number], 
                      name_file: str):
        d = {}
        minim = []
        maxim = []
        i = 0
        print(mandami_rule.premises)
        print(mandami_rule.get_premises_fuzzy_set())
        for fuzzy in mandami_rule.get_premises_fuzzy_set():
            d[str(i)] = fuzzy.memberships_function
            minim.append(fuzzy.bound[0])
            maxim.append(fuzzy.bound[1])
            i = i + 1
        d['cons'] = mandami_rule.conseguence.memberships_function
        minim.append(mandami_rule.conseguence.bound[0])
        maxim.append(mandami_rule.conseguence.bound[1])
        ris = mandami_rule.evaluate(input_rule)
        generate_plot(ris.memberships_function, [], self.__PATH, name_file, additional_call=d, start=np.min(minim), end=np.max(maxim))

class TestDNFRule:

    __PATH = './plots_dnf_rule_aggregation/'

    @pytest.mark.parametrize(
            "premises,name_conseguence,conseguence,tnorm_operation,tconorm_operation,exception_expected", 
            [
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 
                    'cons', 
                    TriangularFuzzySet(0, 1, 2),
                    minimum, 
                    maximum, 
                    None),
                ('a', 
                 'cons', 
                  TriangularFuzzySet(0, 1, 2),
                  minimum, 
                  maximum, 
                  TypeError),
                ([], 
                 'cons', 
                 TriangularFuzzySet(0, 1, 2),
                 minimum, 
                 maximum, 
                 ValueError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ],
                    'cons', 
                    'a',
                    minimum, 
                    maximum,
                    TypeError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 
                    'cons', 
                    TriangularFuzzySet(0, 1, 2),
                    'a', 
                    maximum, 
                    TypeError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 
                    'cons', 
                    TriangularFuzzySet(0, 1, 2),
                    minimum, 
                    'a', 
                    TypeError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 
                    1, 
                    TriangularFuzzySet(0, 1, 2),
                    minimum, 
                    maximum,
                    TypeError),
                ([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': GaussianFuzzySet(1, 1), 'temp4': GaussianFuzzySet(2, 1)},
                    ], 
                    '', 
                    TriangularFuzzySet(0, 1, 2),
                    minimum, 
                    maximum, 
                    ValueError),
            ])
    def test_creation(self,
                      premises : dict[FuzzySet], 
                      name_conseguence : str, 
                      conseguence : FuzzySet, 
                      tnorm_operation: Callable[[FuzzySet, FuzzySet], ContinuousFuzzyCombination | DiscreteFuzzyCombination],
                      tconorm_operation: Callable[[FuzzySet, FuzzySet], ContinuousFuzzyCombination | DiscreteFuzzyCombination],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = DNFRule(premises, name_conseguence, conseguence, tnorm_operation, tconorm_operation)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = DNFRule(premises, name_conseguence, conseguence, tnorm_operation, tconorm_operation)

    @pytest.mark.parametrize(
            "dnf_rule,input_rules,name_file", 
            [
                (DNFRule([
                    {'temp1': GaussianFuzzySet(0, 1), 'temp2': GaussianFuzzySet(1, 1)},
                    {'temp3': TrapezoidalFuzzySet(1, 2, 3, 4), 'temp4': TrapezoidalFuzzySet(3, 4, 5, 6)}
                    ], 
                    'cons1',
                    TriangularFuzzySet(1, 2, 4),
                    ), 
                    {'temp1': 0.5, 
                     'temp2': 2,
                     'temp3': 2, 
                     'temp4': 3.5},
                    'gaussian-trapezoidal'),
            ])
    def test_evaluate(self,
                      dnf_rule: DNFRule, 
                      input_rules: dict[str, np.number],
                      name_file: str):
        d = {}
        i = 0
        for fuzzy in dnf_rule.or_clausule_premises:
            d['clausole' + str(i)] = fuzzy.memberships_function
            i = i + 1
        d['cons'] = dnf_rule.conseguence.memberships_function
        
        ris = dnf_rule.evaluate(input_rules)

        generate_plot(ris.memberships_function, [], self.__PATH, name_file, additional_call=d)

'''
class TestTSKRule:
    __PATH = './plots_tsk_rule_aggregation/'
    
    @pytest.mark.parametrize(
            "premises,weights,name_conseguence,conseguence,exception_expected", 
            [
                ({'Gaussian(0, 1)': GaussianFuzzySet(0, 1), 
                  'Gaussian(1, 1)': GaussianFuzzySet(1, 1)}, [0, 0.5, 0.5], 
                  'cons', 

                  None),
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
        generate_plot(tsk_rule.memberships_function, [], self.__PATH, name_file, additional_call=d)'''