from __future__ import annotations
from typing import Callable
import numpy as np

from softpy.fuzzy.operations import ContinuousFuzzyCombination, DiscreteFuzzyCombination, maximum, minimum
from .fuzzyset import FuzzySet, DiscreteFuzzySet, ContinuousFuzzySet
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
import scipy as sp

def get_combination_of_fuzzyset_list(list: list[FuzzySet], 
                                     op: Callable[[FuzzySet, FuzzySet], 
                                                  ContinuousFuzzyCombination | 
                                                  DiscreteFuzzyCombination]) -> ContinuousFuzzyCombination | DiscreteFuzzyCombination:
        if len(list) == 1:
            return list[0]
        
        curr = list[-1]
        for i in range(1, len(list)):
            curr = op(list[- 1 - i], curr)
        return curr

class FuzzyRule(ABC):
    '''
    An abstract class to represent a fuzzy rule.
    '''
    @abstractmethod
    def evaluate(self, params : dict):
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
                 tnorm_operation: Callable[[FuzzySet, FuzzySet], 
                                           ContinuousFuzzyCombination | DiscreteFuzzyCombination] = minimum):
        
        if not isinstance(premises, dict):
            raise TypeError("premise should be a dict")

        if len(premises.keys()) <= 1:
            raise ValueError("premises should have at least of almost 2 elements")       
        
        for k, v in premises.items():
            if not isinstance(k, str) and k != '':
                raise TypeError("All keys should be a not empty string") 
            if not isinstance(v, FuzzySet):
                raise TypeError('All values should be a FuzzySet')  

        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a not empty string")
        
        if name_conseguence == '':
            raise ValueError("name_conseguence should be a not empty string")

        if not isinstance(tnorm_operation, Callable):
            raise TypeError("tnorm_operation should be a tnorm fuzzy operation")  

        self.__rule: ContinuousFuzzyCombination | DiscreteFuzzyCombination = None
        self.__tnorm_operation: Callable = tnorm_operation
        self.__premises: dict[str, FuzzySet] = premises
        self.__rule = get_combination_of_fuzzyset_list(list(self.__premises.values()), 
                                                       self.__tnorm_operation)

    @property
    def rule(self) -> DiscreteFuzzyCombination | ContinuousFuzzyCombination:
        return self.__rule
    
    @property
    def premises(self) -> dict[str, FuzzySet]:
        return self.__premises
    
    def get_imput_name(self) -> list[str]:
        return self.__premises.keys()

    def evaluate(self, params: dict[str, np.number]) -> np.number:
        '''
        It evaluates the MamdaniRule given a list of elements, ones per premise.
        '''
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        
        if params.keys() != self.__premises.keys():
            raise TypeError("params should have the same input of premises")
        
        return self.__rule(params.values())

class DNFRule(FuzzyRule):
    '''
    An implementation of a DNF fuzzy rule.
    
    Rule is in the following form:
    T(TC(F1, F2, ..., FN), TC(F'1, F'2, ..., F'M), ...)
    '''
    def __init__(self, 
                 premises: list[dict[str, FuzzySet]], 
                 name_conseguence: str,
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

        self.__rule: ContinuousFuzzyCombination | DiscreteFuzzyCombination = None
        self.__tnorm_operation: Callable = tnorm_operation
        self.__tconorm_operation: Callable = tconorm_operation
        self.__premises: list[dict[str, FuzzySet]] = premises
        self.__or_clausule_premises: list[MamdaniRule] = []
        self.__name_conseguence = name_conseguence
        for d in self.__premises:
            self.__or_clausule_premises.append(MamdaniRule(d, 'cons', self.__tconorm_operation))
        

        cong: list[FuzzySet] = [c.rule for c in self.__or_clausule_premises] 
        self.__rule = get_combination_of_fuzzyset_list(cong, self.__tnorm_operation)
    
    @property
    def rule(self) -> DiscreteFuzzyCombination | ContinuousFuzzyCombination:
        return self.__rule
    
    @property
    def or_clausule_premises(self) -> list[MamdaniRule]:
        return self.__or_clausule_premises

    def evaluate(self, params: list[dict[np.number]]) -> np.number:
        '''
        It evaluates the MamdaniRule given a list of elements, ones per premise.
        '''
        if not isinstance(params, list):
            raise TypeError("params should be a list of list")
        
        if len(params) != len(self.__premises):
            raise ValueError("params should have the same length of premises")

        input = []
        for d1, d2 in zip(params, self.__premises):
            if not isinstance(d1, dict):
                raise TypeError("c should be a list of number")
        
            if len(d1) != len(d2):
                raise ValueError("params should have the same length of premises")
            
            if d1.keys() != d2.keys():
                raise ValueError("params should have the same input of premises")
            
            input.append(d1.values())

        return self.__rule(input)
    

class TSKRule(FuzzyRule):
    '''
    An implementation of a Takagi Sugeno Kang fuzzy rule.
    
    Rule is in the following form:
    dot((W1, W2, ..., WN), (F1, F2, ..., FN)) + W0
    '''
    def __init__(self, 
                 premises : dict[str, FuzzySet], 
                 weights: list[np.number],
                 name_conseguence: str):
        
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
        
        self.__premises = premises
        self.__weights = weights
        self.__name_conseguence = name_conseguence
    
    @property
    def premises(self) -> dict[str, FuzzySet]:
        return self.__premises
    
    def memberships_function(self, x: np.number) -> np.number:
        memb = [f(x) for f in self.__premises.values()]
        return np.dot(self.__weights[1:], memb) + self.__weights[0]

    def evaluate(self, params: dict[str, np.number]):
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        
        if params.keys() != self.__premises.keys():
            raise ValueError("params should have the same keys of premises")
    
        for k in params:
            if not np.issubdtype(type(k), np.number):
                raise ValueError("every value should be a number")
        
        memb = [f(input) for f, input in zip(self.__premises.values(), params.values())]
        return self.__rule(memb)

def center_of_gravity()

class FuzzyControlSystem:
    '''
    An implementation of a FuzzyControlSystem: it can be either a Mamdani control system or a Sugeno control system, by specifying the appropriate type of
    rule. It is defined by a given set of rules (of the appropriate type) as well as (in the case of a Mamdani control system) by a tconorm operator.
    '''
    def __init__(self, rules: list[FuzzyRule], tconorm = None, type=MamdaniRule):

        if type not in [MamdaniRule, SugenoRule]:
            raise TypeError("Rule type should be either MamdaniRule or SugenoRule")
        
        for r in rules:
            if not isinstance(r, type):
                raise TypeError("All rules should be of type %s" % type)
            
        self.rules = rules
        self.type = type
        self.tconorm = tconorm


    def evaluate_mamdani(self, results: dict, sets : dict):
        '''
        Applies the control for a Mamdani control system at the given input control, given the specified control results and the corresponding fuzzy sets.
        Specifically, it first combines the different controls for the same output using the specified tconorm operator, then applies defuzzification to
        reduce to a single control output.
        '''
        output = {}
        for n in results.keys():
            curr = results[n][0]
            for _, r in enumerate(results[n], 1):
                curr = self.tconorm(curr, r)
            
            max = -np.infty
            if isinstance(sets[n], DiscreteFuzzySet):
                for v in sets[n].items:
                    tmp = curr(v)
                    if tmp > max:
                        max = tmp
            elif isinstance(sets[n], ContinuousFuzzySet):
                max = sp.integrate.quad(lambda u: curr(u)*u, sets[n].min, sets[n].max)[0]
                max /= sp.integrate.quad(lambda u: curr(u), sets[n].min, sets[n].max)[0] + 0.001
            
            output[n] = max

        return output
    
    def evaluate_sugeno(self, results: dict):
        '''
        Applies the control for a Sugeno control system at the given input control, given the specified control results and fuzzy set memberships.
        Specifically, it combines the different controls for a given output by performing weighted average.
        '''
        output = {}
        for n in results.keys():
            num = 0
            denom = 0
            for i, _ in enumerate(results[n]):
                num += results[n][i][0]*results[n][i][1]
                denom += results[n][i][0]
            
            output[n] = num/denom

        return output


    def evaluate(self, params : dict, u = None) -> dict:
        '''
        It applies the control system at the given control input by applying all the rules and then returning an output signal.
        After applying all the rules in the control system rule base, it applies a different evaluation method based on whether
        it is a Mamdani or a Sugeno control system.
        '''
        results = {}
        sets = {}
        for r in self.rules:
            if r.consequent_name in results.keys():
                results[r.consequent_name].append(r.evaluate(params))
            else:
                results[r.consequent_name] = [r.evaluate(params)]
                sets[r.consequent_name] = r.consequent

        output = None
        if self.type == MamdaniRule:
            output = self.evaluate_mamdani(results, sets)
        elif self.type == SugenoRule:
            output = self.evaluate_sugeno(results)
        
        return output

    
            

