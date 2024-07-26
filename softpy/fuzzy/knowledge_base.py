from enum import Enum
from typing import Callable
import numpy as np

from softpy.fuzzy.operations import ContinuousFuzzyCombination, DiscreteFuzzyCombination, maximum, minimum
from .fuzzyset import FuzzySet, DiscreteFuzzySet, ContinuousFuzzySet
from abc import ABC, abstractmethod
import scipy as sp

class TypeRule(Enum):
    Continuous = 0
    Discrete = 1

class FuzzyRule(ABC):
    '''
    An abstract class to represent a fuzzy rule.
    '''
 
    @abstractmethod
    def evaluate(self, params : dict) -> FuzzySet:
        pass


class MamdaniRule(FuzzyRule):
    '''
    An implementation of a fuzzy rule for a Mamdani-type control system.
    
    Rule is in the following form:
    T(F1, F2, ..., FN)
    '''

    def __init__(self, 
                 premises: dict[str, FuzzySet], 
                 name_conseguence: str,
                 conseguence: FuzzySet,
                 tnorm_operation: Callable[[FuzzySet, FuzzySet], 
                                           ContinuousFuzzyCombination | DiscreteFuzzyCombination] = minimum):
        
        if not isinstance(premises, dict):
            raise TypeError("premise should be a dict")

        if len(premises.keys()) == 0:
            raise ValueError("premises should have at least of almost 1 element")       
        
        for k, v in premises.items():
            if not isinstance(k, str) and k != '':
                raise TypeError("All keys should be a not empty string") 

        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a not empty string")
        
        if name_conseguence == '':
            raise ValueError("name_conseguence should be a not empty string")
        
        if not isinstance(conseguence, FuzzySet):
            raise TypeError("v should be a FuzzySet")

        if not isinstance(tnorm_operation, Callable):
            raise TypeError("tnorm_operation should be a tnorm fuzzy operation")  
        
        self.__tnorm_operation: Callable = tnorm_operation
        self.__premises: dict[str, FuzzySet] = premises
        self.__name_conseguence: str = name_conseguence
        self.__conseguence: FuzzySet = conseguence

        if isinstance(list(premises.values())[0], ContinuousFuzzySet):
            self.__rule: ContinuousFuzzyCombination = ContinuousFuzzyCombination(list(premises.values()), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Continuous
        else:
            self.__rule: DiscreteFuzzyCombination = DiscreteFuzzyCombination(list(premises.values()), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Discrete

    @property
    def rule(self) -> DiscreteFuzzyCombination | ContinuousFuzzyCombination:
        return self.__rule
    
    @property
    def premises(self) -> dict[str, FuzzySet]:
        return self.__premises
    
    @property
    def name_conseguence(self) -> str:
        return self.__name_conseguence
    
    @property
    def conseguence(self) -> FuzzySet:
        return self.__conseguence
    
    def get_input_name(self) -> list[str]:
        return self.__premises.keys()

    def evaluate(self, params: dict[str, np.number]) -> FuzzySet:
        '''
        It evaluates the MamdaniRule given a list of elements, ones per premise.
        '''
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")

        if set.intersection(set(params.keys()), 
                            set(self.get_input_name())) != set(self.get_input_name()):
            raise TypeError("params should have the same input of premises")
        
        combination_premises = self.__rule(list(params.values()))

        if self.__type_rule == TypeRule.Continuous:
            return ContinuousFuzzySet(lambda x: self.__tnorm_operation(combination_premises, 
                                                                       self.__conseguence(x)),
                                      self.__conseguence.bound)
        if self.__type_rule == TypeRule.Continuous:
            return DiscreteFuzzySet(self.__conseguence.items,
                                    [self.__tnorm_operation(combination_premises, m) for m in self.__conseguence.memberships],
                                    self.__conseguence.dynamic)

class DNFRule(FuzzyRule):
    '''
    An implementation of a DNF fuzzy rule.
    
    Rule is in the following form:
    T(TC(F1, F2, ..., FN), TC(F'1, F'2, ..., F'M), ...)
    '''
    def __init__(self, 
                 premises: list[dict[str, FuzzySet]], 
                 name_conseguence: str,
                 conseguence: FuzzySet,
                 tnorm_operation: Callable[[FuzzySet, FuzzySet], 
                                           ContinuousFuzzyCombination | DiscreteFuzzyCombination] = minimum,
                 tconorm_operation: Callable[[FuzzySet, FuzzySet], 
                                             ContinuousFuzzyCombination | DiscreteFuzzyCombination] = maximum):
        
        if not isinstance(premises, list):
            raise TypeError("premise should be a list")
        
        if len(premises) == 0:
            raise ValueError("premise should be a not empty list")
        
        if not isinstance(tnorm_operation, Callable):
            raise TypeError("tnorm_operation should be a tnorm fuzzy operation")
          
        if not isinstance(tconorm_operation, Callable):
            raise TypeError("tconorm_operation should be a tnorm fuzzy operation")  
        
        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a string")
        
        if name_conseguence == '':
            raise ValueError("name_conseguence should be a non empty string")  

        if not isinstance(conseguence, FuzzySet):
            raise TypeError("conseguence should be a FuzzySet")
        
        self.__rule: ContinuousFuzzyCombination | DiscreteFuzzyCombination = None
        self.__tnorm_operation: Callable = tnorm_operation
        self.__tconorm_operation: Callable = tconorm_operation
        self.__premises: list[dict[str, FuzzySet]] = premises
        self.__name_conseguence = name_conseguence
        self.__conseguence = conseguence
        self.__or_clausule_premises: list[FuzzySet] = []

        if isinstance(list(self.__premises[0].values())[0], ContinuousFuzzySet):
            self.__type_rule: TypeRule = TypeRule.Continuous
            for d in self.__premises:
                self.__or_clausule_premises.append(ContinuousFuzzyCombination(list(d.values()), 
                                                                              self.__tconorm_operation))
        else:
            self.__type_rule: TypeRule = TypeRule.Discrete
            for d in self.__premises:
                self.__or_clausule_premises.append(DiscreteFuzzyCombination(list(d.values()),
                                                                            self.__tconorm_operation))
        if self.__type_rule == TypeRule.Continuous:
            self.__rule = ContinuousFuzzyCombination(self.__or_clausule_premises, 
                                                     self.__tnorm_operation)
        else:
            self.__rule = DiscreteFuzzyCombination(self.__or_clausule_premises, 
                                                   self.__tnorm_operation)

        
    
    @property
    def rule(self) -> DiscreteFuzzyCombination | ContinuousFuzzyCombination:
        return self.__rule
    
    @property
    def or_clausule_premises(self) -> list[MamdaniRule]:
        return self.__or_clausule_premises

    @property
    def name_conseguence(self) -> str:
        return self.__name_conseguence
    
    @property
    def conseguence(self) -> FuzzySet:
        return self.__conseguence

    def get_input_name(self) -> list[str]:
        names: list[str] = []
        for d in self.__premises:
            names.extend(list(d.keys()))
        return names
    
    def evaluate(self, params: dict[np.number]) -> np.number:
        '''
        It evaluates the MamdaniRule given a list of elements, ones per premise.
        '''
        if not isinstance(params, dict):
            raise TypeError("params should be a list of list")
        
        if set.intersection(set(params.keys()), 
                            set(self.get_input_name())) != set(self.get_input_name()):
            raise ValueError("params should have the same input")

        input_params = []
        for d in self.__premises:
            input_params.append([params[key] 
                                 for key in d.keys()])


        combination_premises = self.__rule(input_params)

        if self.__type_rule == TypeRule.Continuous:
            return ContinuousFuzzySet(lambda x: self.__tnorm_operation(combination_premises, 
                                                                       self.__conseguence(x)),
                                      self.__conseguence.bound)
        if self.__type_rule == TypeRule.Continuous:
            return DiscreteFuzzySet(self.__conseguence.items,
                                    [self.__tnorm_operation(combination_premises, m) for m in self.__conseguence.memberships],
                                    self.__conseguence.dynamic)


class TSKRule(FuzzyRule):
    '''
    An implementation of a Takagi Sugeno Kang fuzzy rule.
    
    Rule is in the following form:
    dot((W1, W2, ..., WN), (F1, F2, ..., FN)) + W0
    '''
    def __init__(self, 
                 premises : dict[str, FuzzySet], 
                 weights: list[np.number],
                 name_conseguence: str,
                 conseguence: FuzzySet):
        
        if not isinstance(premises, dict):
            raise TypeError("premises should be a dict")
        
        if len(premises.keys()) <= 1:
            raise ValueError("premises should have at least of almost 2 elements")       
        
        for k, v in premises.items():
            if not isinstance(k, str) and k != '':
                raise TypeError("All keys should be a not empty string") 
            if not isinstance(v, FuzzySet):
                raise TypeError('All values should be a FuzzySet')  
            
        if not isinstance(weights, list):
            raise TypeError("weights should be a list")
        
        if len(weights) != len(premises) + 1:
            raise ValueError("premises and weights should have the same length")
        
        sum = 0
        for w in weights:
            if not np.issubdtype(type(w), np.number):
                raise TypeError("All weigths should be a number") 
            if w < 0 or w > 1:
                raise ValueError("All weigths should be between 0 and 1") 
            sum = sum + w
        if sum != 1:
            raise ValueError("Sum of weigths should be 1") 

        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a string")
        
        if name_conseguence == '':
            raise ValueError("name_conseguence should be a non empty string")  
        
        if not isinstance(conseguence, FuzzySet):
            raise ValueError("conseguence should be a FuzzySet")  
        
        self.__premises = premises
        self.__weights = weights
        self.__name_conseguence = name_conseguence
        self.__conseguence = conseguence
    
    @property
    def premises(self) -> dict[str, FuzzySet]:
        return self.__premises.keys()
    
    @property
    def name_conseguence(self) -> str:
        return self.__name_conseguence
    
    @property
    def conseguence(self) -> FuzzySet:
        return self.__conseguence
    
    def get_input_name(self) -> list[str]:
        names: list[str] = []
        for d in self.__premises:
            names.extend(d.keys())
        return names
    
    def memberships_function(self, x: np.number) -> np.number:
        memb = [f(x) for f in self.__premises.values()]
        return np.dot(self.__weights[1:], memb) + self.__weights[0]

    def evaluate(self, params: dict[str, np.number]):
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        
        if set.intersection(set(params.keys()), 
                            set(self.get_input_name())) != set(self.get_input_name()):
            raise ValueError("params should have the same keys of premises")
    
        for k in params:
            if not np.issubdtype(type(k), np.number):
                raise ValueError("every value should be a number")
        
        memb = [f(input) for f, input in zip(self.__premises.values(), params.values())]
        return self.__rule(memb)

def center_of_gravity(fuzzy_set: ContinuousFuzzySet | DiscreteFuzzySet):
    defuzz = None
    if isinstance(fuzzy_set, ContinuousFuzzySet):
        
        partial_num = 0
        partial_den = 0
        defuzz = None
        
        num = sp.integrate.quad(lambda u: fuzzy_set(u)*u,
                                fuzzy_set.bound[0],
                                fuzzy_set.bound[1], 
                                full_output=1)[0]
        denum = sp.integrate.quad(lambda u: fuzzy_set(u),
                                  fuzzy_set.bound[0],
                                  fuzzy_set.bound[1], 
                                  full_output=1)[0]
        if denum == 0:
            defuzz = 0
        else:
            defuzz = num / denum
            
    elif isinstance(fuzzy_set, DiscreteFuzzySet):
        partial_num = 0
        partial_den = 0
        for item in fuzzy_set.items:
            partial_num = partial_num + item * fuzzy_set(item)
            partial_den = partial_den + fuzzy_set(item)
        defuzz = partial_num / partial_den
    else:
        raise TypeError('fuzzy_set should be ContinuousFuzzyCombination or DiscreteFuzzyCombination')
    
    return defuzz

class AggregationType(Enum):
    FITA = 1
    FATI = 2

class KnowledgeBase:
    def __init__(self, 
                 rules: list[FuzzyRule],    
                 aggregation_type: AggregationType = AggregationType.FATI,
                 tconorm_aggregation: Callable = maximum, 
                 defuzzification_function: Callable = center_of_gravity):
        if not isinstance(rules, list):
            raise TypeError("rules should be a list")
        
        self.__rules: dict[str, list[FuzzyRule]] = {}
        
        for r in rules:
            if not isinstance(r, FuzzyRule):
                raise TypeError("All rules should be FuzzyRule")
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
        output_fuzzy_sets = self.__evaluate(params)
        results = {}

        if self.__aggregation_type == AggregationType.FATI:
            fuzzy_output = self.__aggregate_fuzzy_sets(output_fuzzy_sets)
            
            for name_output, output_fuzzy_set in fuzzy_output.items():
                results[name_output] = self.__defuzzification_function(output_fuzzy_set)
        else:
            for name_output, list_output_fuzzy_set in output_fuzzy_sets.items():
                for f in list_output_fuzzy_set:
                    if name_output in results.keys():
                        results[name_output].append(self.__defuzzification_function(f))
                    else:
                        results[name_output] = [self.__defuzzification_function(f)]
            results = self.__aggregate_crispy_values(results)
        return results

    def __evaluate(self, params: dict[str, np.number]) -> dict[str, list[FuzzySet]]:
        output = {}
        
        for name_output, lists_rules in self.__rules.items():
            for r in lists_rules:
                evaluation = r.evaluate({key: params[key] for key in r.get_input_name() and params.keys()})
                if name_output in output:
                    output[name_output].append(evaluation)
                else:
                    output[name_output] = [evaluation]

        return output

    def __aggregate_fuzzy_sets(self, params: dict[str, list[FuzzySet]]) -> dict[str, ContinuousFuzzyCombination | DiscreteFuzzyCombination]:
        aggregated_output = {}
        
        for name_output, fuzzy_sets in params.items():
            if isinstance(fuzzy_sets[0], ContinuousFuzzySet):
                aggregated_output[name_output] = ContinuousFuzzyCombination(fuzzy_sets, self.__tconorm_aggregation)
            else:
                aggregated_output[name_output] = DiscreteFuzzyCombination(fuzzy_sets, self.__tconorm_aggregation)
        
        return aggregated_output
    
    def __aggregate_crispy_values(self, params: dict[str, list[np.number]]) -> dict[str, np.number]:
        aggregated_output = {}
        
        for name_output, values in params.items():
            for v in values:
                if name_output in aggregated_output.keys():
                    aggregated_output[name_output] = self.__tconorm_aggregation(aggregated_output[name_output], v)
                else:
                    aggregated_output[name_output] = v
        
        return aggregated_output