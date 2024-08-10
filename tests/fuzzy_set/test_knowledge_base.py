import sys
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzy_rule import FuzzyRule, MamdaniRule,DNFRule, TSKRule
from softpy.fuzzy.knowledge_base import AggregationType, TSKKnowledgeBase, center_of_gravity, MamdaniKnowledgeBase
from softpy.fuzzy.fuzzyset import TriangularFuzzySet, ZShapedFuzzySet, LinearZFuzzySet, GaussianFuzzySet, SShapedFuzzySet, LinearSFuzzySet
from softpy.fuzzy.operations import maximum
from tests.fuzzy_set.configuration import generate_plot, not_raises 

class TestMamdaniKnowledgeBase:
    __PATH = 'plots_knowledge_base_aggregation/'

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
                lfs = MamdaniKnowledgeBase(rules=rules_list,
                                           aggregation_type=aggregation_type,
                                           tconorm_aggregation=tconorm_aggregation,
                                           defuzzification_function=defuzzification_function)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = MamdaniKnowledgeBase(rules=rules_list, 
                                           aggregation_type=aggregation_type,
                                           tconorm_aggregation=tconorm_aggregation,
                                           defuzzification_function=defuzzification_function)
    
    @pytest.mark.parametrize(
            "kb,params", 
            [
                (MamdaniKnowledgeBase([
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
                (MamdaniKnowledgeBase([
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
                (MamdaniKnowledgeBase([
                    DNFRule([{'service': ZShapedFuzzySet(0, 5, bound=(0,10)),
                              'food': LinearZFuzzySet(2, 4, bound=(0,10))}], 
                             'tip',
                             TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    DNFRule([{'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                              'food': LinearSFuzzySet(6, 8, bound=(0,10))}], 
                              'tip',
                              TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                    ], aggregation_type=AggregationType.FITA),
                    {
                        'service': 3,
                        'food': 2
                    }),
            ])
    def test_inference(self,
                      kb: MamdaniKnowledgeBase,
                      params: dict[str, np.number]):
        print(kb.infer(params))

    @pytest.mark.parametrize(
            "kb,params,file_name", 
            [
                (MamdaniKnowledgeBase([
                    MamdaniRule({'service': ZShapedFuzzySet(0, 5, bound=(0,10)), 
                                 'food': LinearZFuzzySet(2, 4, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(0, 5, 10, bound=(0,30))),
                    MamdaniRule({'service': GaussianFuzzySet(5, 3, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(10, 15, 20, bound=(0,30))),
                    MamdaniRule({'service': SShapedFuzzySet(5, 10, bound=(0,10)),
                                 'food': LinearSFuzzySet(6, 9, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                    ]),
                    {
                        'service': 3,
                        'food': 8
                    },
                    'tip'),
                
                (MamdaniKnowledgeBase([
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
                    MamdaniRule({'food': LinearSFuzzySet(6, 9, bound=(0,10))}, 
                                 'tip',
                                 TriangularFuzzySet(20, 25, 30, bound=(0,30)))
                    ]),
                    {
                        'service': 3,
                        'food': 8
                    },
                    'tip-mamdani-corretto'),
            ])
    def test_evaluation(self,
                        kb: MamdaniKnowledgeBase,
                        params: dict[str, np.number],
                        file_name: str):
        output_fuzzy_sets = kb.evaluate(params)
        fuzzy_output = kb.aggregate_fuzzy_sets(output_fuzzy_sets)
        
        for k, v in fuzzy_output.items():
            sel = output_fuzzy_sets[k]
            inp = {str(i): sel[i].memberships_function for i in range(len(sel))}
            generate_plot(v.memberships_function, [], self.__PATH, file_name, additional_call=inp, start=v.bound[0], end=v.bound[1])
        

class TestTSKKnowledgeBase:
    @pytest.mark.parametrize(
            "rules_list,exception_expected", 
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
                TypeError),
                ('a',
                TypeError),
                ([
                    'a'
                ],
                TypeError),
                ([
                    TSKRule({'service': GaussianFuzzySet(5, 3, bound=(0,10)),
                             'food': GaussianFuzzySet(2, 3, bound=(0,10))},
                            [0.5, 1, 2],
                            'tip')
                ],
                None),
                ([],
                ValueError)
            ])
    def test_creation(self,
                      rules_list: list[FuzzyRule],
                      exception_expected: Exception):

        if exception_expected == None:
            with not_raises() as e_info:
                lfs = TSKKnowledgeBase(rules=rules_list)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = TSKKnowledgeBase(rules=rules_list)
    
    @pytest.mark.parametrize(
            "kb,params", 
            [
                (TSKKnowledgeBase([
                    TSKRule({'service': GaussianFuzzySet(5, 3, bound=(0,10)),
                             'food': GaussianFuzzySet(2, 3, bound=(0,10))},
                            [0.5, 1, 2],
                            'tip')
                    ]),
                    {
                        'service': 3,
                        'food': 2
                    }),
            ])
    def test_inference(self,
                      kb: MamdaniKnowledgeBase,
                      params: dict[str, np.number]):
        print(kb.infer(params))