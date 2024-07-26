from __future__ import annotations
from enum import Enum
from typing import Callable
import numpy as np

from softpy.fuzzy.operations import ContinuousFuzzyCombination, DiscreteFuzzyCombination, maximum, minimum
from .fuzzyset import FuzzySet, DiscreteFuzzySet, ContinuousFuzzySet
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
import scipy as sp



'''

class FuzzyControlSystem:
    An implementation of a FuzzyControlSystem: it can be either a Mamdani control system or a Sugeno control system, by specifying the appropriate type of
    rule. It is defined by a given set of rules (of the appropriate type) as well as (in the case of a Mamdani control system) by a tconorm operator.
    def __init__(self, 
                 inputs: dict[str, list[dict[str, FuzzySet]]],  # name input and fuzzy partition
                 outputs: dict[str, list[dict[str, FuzzySet]]],  # name output and fuzzy partition
                 rules: list[FuzzyRule], 
                 aggregation_type: AggregationType = AggregationType.FATI,
                 tconorm_aggregation: Callable = maximum, 
                 defuzzification_function: Callable = center_of_gravity):

        if not isinstance(input, dict):
            raise TypeError("input should be a dictionary")
        
        for k, v in input.items():
            if not isinstance(k, str):
                raise TypeError("k should be the name of a variable input")

        if not isinstance(aggregation_type, AggregationType):
            raise TypeError("aggregation_type should be a AggregationType")

        if not isinstance(rules, list):
            raise TypeError("rules should be a list")

        for r in rules:
            if not isinstance(r, FuzzyRule):
                raise TypeError("All rules should be a FuzzyRule")
            
        if not isinstance(tconorm_aggregation, Callable):
            raise TypeError("tconorm_aggregation should be a callable")
        
        if not isinstance(defuzzification_function, Callable):
            raise TypeError("defuzzification_function should be a callable")
        
        self.__rules: list[FuzzyRule] = rules
        self.__aggregation_type: AggregationType = aggregation_type
        self.__tconorm_aggregation: Callable  = tconorm_aggregation
        self.__defuzzification_function: Callable  = defuzzification_function

    def evaluate(self, 
                 input: dict[str, np.number]) -> dict[str, np.number]:
        It applies the control system at the given control input by applying all the rules and then returning an output signal.
        After applying all the rules in the control system rule base, it applies a different evaluation method based on whether
        it is a Mamdani or a Sugeno control system.
        results = {}
        outputs = {}
        for r in self.__rules:
            if r.name_conseguence in results.keys():
                results[r.name_conseguence].append(r.evaluate({key: input[key] 
                                                               for key in input.keys() & r.get_input_name()}))
            else:
                results[r.name_conseguence] = [r.evaluate({key: input[key] 
                                                           for key in input.keys() & r.get_input_name()})]

        if self.__aggregation_type == AggregationType.FATI:
            for name, ris_fuzzy_rules in results:
                aggregation = None
                if len(ris_fuzzy_rules) > 1:
                    curr = ris_fuzzy_rules[-1]
                    for i in range(1, len(ris_fuzzy_rules)):
                        curr =  self.__tconorm_aggregation(ris_fuzzy_rules[-1 - i], curr)
                    aggregation = curr
                else:
                    aggregation = ris_fuzzy_rules[0]
                outputs[name] = self.__defuzzification_function(aggregation)
        if self.__aggregation_type == AggregationType.FITA:
            for name, ris_fuzzy_rules in results:
                aggregation = None
                if len(ris_fuzzy_rules) > 1:
                    for f in ris_fuzzy_rules:
                        f = self.__defuzzification_function(f)
                    
                    curr = ris_fuzzy_rules[-1]
                    for i in range(1, len(ris_fuzzy_rules)):
                        curr =  self.__tconorm_aggregation(ris_fuzzy_rules[-1 - i], curr)
                    aggregation = curr
                else:
                    ris_fuzzy_rules[0] = self.__defuzzification_function(ris_fuzzy_rules[0])
                    aggregation = ris_fuzzy_rules[0]

                outputs[name] = aggregation

        return outputs
    '''
