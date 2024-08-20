from enum import Enum
from typing import Callable
import numpy as np

from softpy.fuzzy.fuzzy_rule import ClassifierRule, DNFRule, MamdaniRule, SingletonRule, TSKRule
from softpy.fuzzy.operations import maximum, minimum
from softpy.fuzzy.fuzzy_operation import ContinuousFuzzyCombination, DiscreteFuzzyCombination
from .fuzzyset import FuzzySet, DiscreteFuzzySet, ContinuousFuzzySet
from abc import ABC, abstractmethod
import scipy as sp

def center_of_gravity(fuzzy_set: ContinuousFuzzySet | DiscreteFuzzySet):
    defuzz = None
    partial_num = 0
    partial_den = 0

    if isinstance(fuzzy_set, ContinuousFuzzySet):
        partial_num = sp.integrate.quad(lambda u: fuzzy_set(u)*u,
                                fuzzy_set.bound[0],
                                fuzzy_set.bound[1], 
                                full_output=1)[0]
        partial_den = sp.integrate.quad(lambda u: fuzzy_set(u),
                                  fuzzy_set.bound[0],
                                  fuzzy_set.bound[1], 
                                  full_output=1)[0] + 0.001
    elif isinstance(fuzzy_set, DiscreteFuzzySet):
        for item in fuzzy_set.items:
            partial_num = partial_num + item * fuzzy_set(item)
            partial_den = partial_den + fuzzy_set(item)
    else:
        raise TypeError('fuzzy_set should be ContinuousFuzzyCombination or DiscreteFuzzyCombination')
    return partial_num / partial_den

class AggregationType(Enum):
    FITA = 1
    FATI = 2

class KnowledgeBaseABC(ABC):
    @abstractmethod
    def infer(self, params: dict[str, np.number]):
        pass

class MamdaniKnowledgeBase(KnowledgeBaseABC):
    def __init__(self, 
                 rules: list[MamdaniRule | DNFRule],    
                 aggregation_type: AggregationType = AggregationType.FATI,
                 tconorm_aggregation: Callable = maximum, 
                 defuzzification_function: Callable = center_of_gravity):
        if not isinstance(rules, list):
            raise TypeError("rules should be a list")
        
        self.__rules: dict[str, list[MamdaniRule | DNFRule]] = {}
        
        for r in rules:
            if not isinstance(r, MamdaniRule) and not isinstance(r, DNFRule):
                raise TypeError("All rules should be MamdaniRule or DNFRule")
            if r.name_conseguence in self.__rules.keys():
                self.__rules[r.name_conseguence].append(r)
            else:
                self.__rules[r.name_conseguence] = [r]

        if not isinstance(aggregation_type, AggregationType):
            raise TypeError("aggregation_type should be a AggregationType")
        
        if not isinstance(tconorm_aggregation, Callable):
            raise TypeError("self.__tconorm_aggregation should be a Callable")
        
        if not isinstance(defuzzification_function, Callable):
            raise TypeError("defuzzification_function should be a Callable")
        
        self.__aggregation_type: AggregationType = aggregation_type
        self.__tconorm_aggregation: Callable = tconorm_aggregation
        self.__defuzzification_function: Callable = defuzzification_function

    def infer(self, params: dict[str, np.number]) -> dict[str, np.number]:
        output_fuzzy_sets = self.evaluate(params)
        results = {}

        if self.__aggregation_type == AggregationType.FATI:
            fuzzy_output = self.aggregate_fuzzy_sets(output_fuzzy_sets)
            
            for name_output, output_fuzzy_set in fuzzy_output.items():
                results[name_output] = self.__defuzzification_function(output_fuzzy_set)
        else:
            weights = self.compute_rule_weights(params)
            defuzzification_output = {}
            sum_weights = {}
            for name_output, list_output_fuzzy_set in output_fuzzy_sets.items():
                sum_weights[name_output] = 0

                for i in range(len(list_output_fuzzy_set)):
                    if name_output in defuzzification_output:
                        defuzzification_output[name_output].append(self.__defuzzification_function(list_output_fuzzy_set[i]))
                    else:
                        defuzzification_output[name_output] = [self.__defuzzification_function(list_output_fuzzy_set[i])]
                    
                    sum_weights[name_output] = sum_weights[name_output] + weights[name_output][i]

            for name_output, list_output_deffuzzification in defuzzification_output.items():
                results[name_output]  = np.dot(weights[name_output], list_output_deffuzzification) / sum_weights[name_output]
        return results

    def evaluate(self, params: dict[str, np.number]) -> dict[str, list[FuzzySet]]:
        output = {}
        
        for name_output, lists_rules in self.__rules.items():
            for r in lists_rules:

                input_dict = {}
                for key in r.get_input_name():
                    if key in params.keys():
                        input_dict[key] = params[key]

                evaluation = r.evaluate(input_dict)

                if name_output in output:
                    output[name_output].append(evaluation)
                else:
                    output[name_output] = [evaluation]

        return output
    
    def aggregate_fuzzy_sets(self, params: dict[str, list[FuzzySet]]) -> dict[str, ContinuousFuzzyCombination | DiscreteFuzzyCombination]:
        aggregated_output = {}
        
        for name_output, fuzzy_sets in params.items():
            if isinstance(fuzzy_sets[0], ContinuousFuzzySet):
                aggregated_output[name_output] = ContinuousFuzzyCombination(fuzzy_sets, self.__tconorm_aggregation)
            else:
                aggregated_output[name_output] = DiscreteFuzzyCombination(fuzzy_sets, self.__tconorm_aggregation)
        
        return aggregated_output
    
    def compute_rule_weights(self, params: dict[str, list[FuzzySet]]) -> dict[str, list[np.number]]:
        evaluations = {}
        for name_output, lists_rules in self.__rules.items():
            for r in lists_rules:
                input_dict = {}
                for key in r.get_input_name():
                    if key in params.keys():
                        input_dict[key] = params[key]

                if name_output in evaluations.keys():
                    evaluations[name_output].append(r.evaluate_premises(input_dict))
                else:
                    evaluations[name_output] = [r.evaluate_premises(input_dict)]

        return evaluations
    

class TSKKnowledgeBase(KnowledgeBaseABC):
    def __init__(self,
                 rules: list[TSKRule]):
        
        if not isinstance(rules, list):
            raise TypeError("rules should be a list")
        
        if len(rules) == 0:
            raise ValueError("rules should be a non empty list")
        
        self.__rules: dict[str, list[TSKRule]] = {}
        
        for r in rules:
            if not isinstance(r, TSKRule):
                raise TypeError("All rules should be TSKRule")
            if r.name_conseguence in self.__rules.keys():
                self.__rules[r.name_conseguence].append(r)
            else:
                self.__rules[r.name_conseguence] = [r]
        
    def infer(self, params: dict[str, np.number]):
        results = self.__evaluate(params)
        
        return results

    def __evaluate(self, params: dict[str, np.number]) -> dict[str, np.number]:
        
        output = {}
        
        for name_output, lists_rules in self.__rules.items():
            
            partial_sum_numerator = 0 
            partial_sum_denumerator = 0 
            
            for r in lists_rules:

                input_dict = {}
                for key in r.get_input_name():
                    if key in params.keys():
                        input_dict[key] = params[key]

                output_rule, weight = r.evaluate(input_dict)

                partial_sum_denumerator = partial_sum_denumerator + weight
                partial_sum_numerator = partial_sum_numerator + weight * output_rule

            output[name_output] = partial_sum_numerator / partial_sum_denumerator

        return output

class SingletonKnowledgeBase(KnowledgeBaseABC):
    def __init__(self,
                 rules: list[SingletonRule]):
        
        if not isinstance(rules, list):
            raise TypeError("rules should be a list")
        
        if len(rules) == 0:
            raise ValueError("rules should be a non empty list")
        
        self.__rules: dict[str, list[SingletonRule]] = {}
        
        for r in rules:
            if not isinstance(r, SingletonRule):
                raise TypeError("All rules should be SingletonRule")
            if r.name_conseguence in self.__rules.keys():
                self.__rules[r.name_conseguence].append(r)
            else:
                self.__rules[r.name_conseguence] = [r]

                
    def infer(self, params: dict[str, np.number]):
        results = self.__evaluate(params)
        
        return results

    def __evaluate(self, params: dict[str, np.number]) -> dict[str, np.number]:
        
        output = {}
        
        for name_output, lists_rules in self.__rules.items():
            
            partial_sum_numerator = 0 
            partial_sum_denumerator = 0 
            
            for r in lists_rules:

                input_dict = {}
                for key in r.get_input_name():
                    if key in params.keys():
                        input_dict[key] = params[key]

                output_rule, weight = r.evaluate(input_dict)

                partial_sum_denumerator = partial_sum_denumerator + weight
                partial_sum_numerator = partial_sum_numerator + weight * output_rule

            output[name_output] = partial_sum_numerator / partial_sum_denumerator

        return output
    
class ClassfierKnowledgeBase(KnowledgeBaseABC):
    def __init__(self,
                 rules: list[ClassifierRule]):
        
        if not isinstance(rules, list):
            raise TypeError("rules should be a list")
        
        if len(rules) == 0:
            raise ValueError("rules should be a non empty list")
        
        self.__rules: dict[str, list[ClassifierRule]] = {}
        
        for r in rules:
            if not isinstance(r, ClassifierRule):
                raise TypeError("All rules should be SingletonRule")
            if r.name_conseguence in self.__rules.keys():
                self.__rules[r.name_conseguence].append(r)
            else:
                self.__rules[r.name_conseguence] = [r]

                
    def infer(self, params: dict[str, np.number]):
        results = self.__evaluate(params)
        
        return results

    def __evaluate(self, params: dict[str, np.number]) -> dict[str, np.number]:
        
        output = {}
        
        for name_output, lists_rules in self.__rules.items():

            max_weight = 0
            target = ''
            
            for r in lists_rules:

                input_dict = {}
                for key in r.get_input_name():
                    if key in params.keys():
                        input_dict[key] = params[key]

                output_rule, weight = r.evaluate(input_dict)

                if weight > max_weight:
                    target = output_rule

            output[name_output] = target
            
        return output
    
