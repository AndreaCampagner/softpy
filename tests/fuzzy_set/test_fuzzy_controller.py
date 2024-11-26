'''
import sys
from typing import Callable

import numpy as np
import pytest
from softpy.fuzzy.operations import maximum

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzy_control import AggregationType, FuzzyControlSystem, FuzzyRule, MamdaniRule, center_of_gravity
from softpy.fuzzy.fuzzyset import GaussianFuzzySet
from tests.fuzzy_set.configuration import not_raises 

class TestFuzzyControlSystem:
    @pytest.mark.parametrize(
            "rules,aggregation_type,tconorm_aggregation,defuzzification_function,exception_expected", 
            [
                ([MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1'),
                  MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1')],
                  AggregationType.FITA, 
                  maximum, 
                  center_of_gravity, 
                  None),
                ([MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1'),
                  MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1')],
                  'a', 
                  maximum, 
                  center_of_gravity, 
                  TypeError),
                ('a',
                 AggregationType.FITA, 
                 maximum, 
                 center_of_gravity, 
                 TypeError),
                (['a'],
                 AggregationType.FITA, 
                 maximum, 
                 center_of_gravity, 
                 TypeError),
                ([MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1'),
                  MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1')],
                  AggregationType.FITA, 
                  'a', 
                  center_of_gravity, 
                  TypeError),
                ([MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1'),
                  MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1')],
                  AggregationType.FITA, 
                  maximum, 
                  'a', 
                  TypeError), 
                ([MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1'),
                  MamdaniRule({'Gaussian(0,1)': GaussianFuzzySet(0, 1), 'Gaussian(1,1)': GaussianFuzzySet(1, 1)}, 'cons1')],
                  AggregationType.FITA, 
                  maximum, 
                  'a', 
                  TypeError), 
            ])
    def test_creation(self,
                      rules: list[FuzzyRule],
                      aggregation_type: AggregationType,
                      tconorm_aggregation: Callable, 
                      defuzzification_function: Callable,
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = FuzzyControlSystem(rules=rules, 
                                         aggregation_type=aggregation_type, 
                                         tconorm_aggregation=tconorm_aggregation,
                                         defuzzification_function=defuzzification_function)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = FuzzyControlSystem(rules=rules, 
                                         aggregation_type=aggregation_type, 
                                         tconorm_aggregation=tconorm_aggregation,
                                         defuzzification_function=defuzzification_function)
    
    '''