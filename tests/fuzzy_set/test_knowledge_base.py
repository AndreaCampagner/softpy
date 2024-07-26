import sys
from typing import Callable

import numpy as np
import pytest
from softpy.fuzzy.knowledge_base import DNFRule, KnowledgeBase
import warnings

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.knowledge_base import AggregationType, FuzzyRule, MamdaniRule, center_of_gravity
from softpy.fuzzy.fuzzyset import TriangularFuzzySet, ZShapedFuzzySet, LinearZFuzzySet, GaussianFuzzySet, SShapedFuzzySet, LinearSFuzzySet
from softpy.fuzzy.operations import maximum
from tests.fuzzy_set.configuration import not_raises 

class TestKnowledgeBase:
    @pytest.mark.parametrize(
            "rules_list,aggregation_type,tconorm_aggregation,defuzzification_function,exception_expected", 
            [
                ([
                    MamdaniRule({'service': ZShapedFuzzySet(0, 5, bound=(0,10)), 
                                 'food': LinearZFuzzySet(2, 4, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                ],
                AggregationType.FATI, 
                maximum, 
                center_of_gravity,
                None),
                ('a',
                AggregationType.FATI, 
                maximum, 
                center_of_gravity,
                TypeError),
                ([
                    'a',
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                ],
                AggregationType.FATI, 
                maximum, 
                center_of_gravity,
                TypeError),
                ([
                    MamdaniRule({'service': ZShapedFuzzySet(0, 5, bound=(0,10)), 
                                 'food': LinearZFuzzySet(2, 4, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                ],
                'a', 
                maximum, 
                center_of_gravity,
                TypeError),
                ([
                    MamdaniRule({'service': ZShapedFuzzySet(0, 5, bound=(0,10)), 
                                 'food': LinearZFuzzySet(2, 4, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                ],
                AggregationType.FATI,
                'a', 
                center_of_gravity,
                TypeError),
                ([
                    MamdaniRule({'service': ZShapedFuzzySet(0, 5, bound=(0,10)), 
                                 'food': LinearZFuzzySet(2, 4, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                ],
                AggregationType.FATI,
                maximum, 
                'a',
                TypeError),
                ([
                    DNFRule([{'service': ZShapedFuzzySet(0, 5, bound=(0,10)),
                             'food': LinearZFuzzySet(2, 4, bound=(0,10))}], 
                             'tip',
                             TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                ],
                AggregationType.FATI,
                maximum, 
                center_of_gravity,
                None)
            ])
    def test_creation(self,
                      rules_list: list[FuzzyRule],
                      aggregation_type: AggregationType,
                      tconorm_aggregation: Callable, 
                      defuzzification_function: Callable,
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = KnowledgeBase(rules=rules_list, 
                                    aggregation_type=aggregation_type,
                                    tconorm_aggregation=tconorm_aggregation,
                                    defuzzification_function=defuzzification_function)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = KnowledgeBase(rules=rules_list, 
                                    aggregation_type=aggregation_type,
                                    tconorm_aggregation=tconorm_aggregation,
                                    defuzzification_function=defuzzification_function)
    
    @pytest.mark.parametrize(
            "kb,params", 
            [
                (KnowledgeBase([
                    MamdaniRule({'service': ZShapedFuzzySet(0, 5, bound=(0,10)), 
                                 'food': LinearZFuzzySet(2, 4, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                    ]),
                    {
                        'service': 3,
                        'food': 2
                    }),
                (KnowledgeBase([
                    MamdaniRule({'service': ZShapedFuzzySet(0, 5, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'food': LinearZFuzzySet(2, 4, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30))),
                    MamdaniRule({'food': LinearSFuzzySet(6, 8, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                    ], aggregation_type=AggregationType.FITA),
                    {
                        'service': 3,
                        'food': 2
                    }),
            ])
    def test_inference(self,
                      kb: KnowledgeBase,
                      params: dict[str, np.number]):
        print(kb.infer(params))
            