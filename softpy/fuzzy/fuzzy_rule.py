from enum import Enum
from typing import Callable
import numpy as np

from softpy.fuzzy.fuzzy_operation import ContinuousFuzzyCombination, DiscreteFuzzyCombination
from softpy.fuzzy.operations import  maximum, minimum
from .fuzzyset import FuzzySet, DiscreteFuzzySet, ContinuousFuzzySet, SingletonFuzzySet
from abc import ABC, abstractmethod

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
                 premises: dict[str, list[FuzzySet]], 
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
            if not isinstance(v, list):
                raise TypeError("All values should be a list") 
            for f in v:
                if not isinstance(f, FuzzySet):
                    raise TypeError("All list values should be a list of fuzzy set") 
                
        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a not empty string")
        
        if name_conseguence == '':
            raise ValueError("name_conseguence should be a not empty string")
        
        if not isinstance(conseguence, FuzzySet):
            raise TypeError("v should be a FuzzySet")

        if not isinstance(tnorm_operation, Callable):
            raise TypeError("tnorm_operation should be a tnorm fuzzy operation")  
        
        self.__tnorm_operation: Callable = tnorm_operation
        self.__premises: dict[str, list[FuzzySet]] = premises
        self.__name_conseguence: str = name_conseguence
        self.__conseguence: FuzzySet = conseguence

        if isinstance(self.get_premises_fuzzy_set()[0], ContinuousFuzzySet):
            self.__rule: ContinuousFuzzyCombination = ContinuousFuzzyCombination(self.get_premises_fuzzy_set(), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Continuous
        else:
            self.__rule: DiscreteFuzzyCombination = DiscreteFuzzyCombination(self.get_premises_fuzzy_set(), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Discrete

    @property
    def rule(self) -> DiscreteFuzzyCombination | ContinuousFuzzyCombination:
        return self.__rule
    
    @property
    def premises(self) -> dict[str, list[FuzzySet]]:
        return self.__premises
    
    @property
    def name_conseguence(self) -> str:
        return self.__name_conseguence
    
    @property
    def conseguence(self) -> FuzzySet:
        return self.__conseguence
    
    def get_input_name(self):
        return self.__premises.keys()

    def get_premises_fuzzy_set(self) -> list[FuzzySet]:
        all_fuzzy_set = []
        for l in self.__premises.values():
            all_fuzzy_set.extend(l)
        return all_fuzzy_set
    
    def evaluate(self, params: dict[str, np.number]) -> FuzzySet:
        '''
        It evaluates the MamdaniRule given a list of elements, ones per premise.
        '''
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")

        if list(params.keys()) != list(self.get_input_name()):
            raise TypeError("params should have the same input of premises")
        
        params_list = []
        for name, l_f in self.__premises.items():
            params_list.extend([params[name]] * len(l_f))

        combination_premises = self.__rule(params_list)

        if self.__type_rule == TypeRule.Continuous:
            return ContinuousFuzzySet(lambda x: self.__tnorm_operation(combination_premises, 
                                                                       self.__conseguence(x)),
                                      self.__conseguence.bound)
        if self.__type_rule == TypeRule.Continuous:
            return DiscreteFuzzySet(self.__conseguence.items,
                                    [self.__tnorm_operation(combination_premises, m) for m in self.__conseguence.memberships],
                                    self.__conseguence.dynamic)

    def evaluate_premises(self, params: dict[str, np.number]) -> np.number:
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")

        if list(params.keys()) != list(self.get_input_name()):
            raise TypeError("params should have the same input of premises")
        
        params_list = []
        for name, l_f in self.__premises.items():
            params_list.extend([params[name]] * len(l_f))

        ris = 1
        for f, param in zip(self.get_premises_fuzzy_set(), params_list):
            ris = self.__tnorm_operation(f(param), ris)

        return ris        
    
class TSKRule(FuzzyRule):
    '''
    An implementation of a Takagi Sugeno Kang fuzzy rule.
    
    Rule is in the following form:
    dot((W1, W2, ..., WN), (F1, F2, ..., FN)) + W0
    '''
    def __init__(self, 
                 premises : dict[str, list[FuzzySet]], 
                 weights: list[np.number],
                 name_conseguence: str,
                 tnorm_operation: Callable = minimum):
        
        if not isinstance(tnorm_operation, Callable):
            raise ValueError("tnorm_operation should be a callable")
        
        if not isinstance(premises, dict):
            raise TypeError("premises should be a dict")
        
        if len(premises.keys()) <= 0:
            raise ValueError("premises should have at least of almost 1 elements")       
        
        for k, v in premises.items():
            if not isinstance(k, str) and k != '':
                raise TypeError("All keys should be a not empty string") 
            if not isinstance(v, list):
                raise TypeError('All values should be a list')  
            if len(v) == 0:
                raise ValueError('All values should be a non empty list')  
            for f in v:
                if not isinstance(f, FuzzySet):
                    raise TypeError('All values should be a list of fuzzy sets')  

            
        
        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a string")

        if name_conseguence == '':
            raise ValueError("name_conseguence should be a non empty string")
        
        self.__premises = premises
        self.__fuzzy_composition = ContinuousFuzzyCombination(self.get_premises_fuzzy_set(), 
                                                              minimum)

        if not isinstance(weights, list):
            raise TypeError("weights should be a list")
        
        for w in weights:
            if not np.issubdtype(type(w), np.number):
                raise TypeError("All weigths should be a number") 
            
        if len(weights) != len(self.get_premises_fuzzy_set()) + 1:
            raise ValueError("premises and weights should have the same length")
        
        self.__name_conseguence = name_conseguence
        self.__weights = weights
    
    @property
    def premises(self) -> dict[str, FuzzySet]:
        return self.__premises
    
    @property
    def name_conseguence(self) -> str:
        return self.__name_conseguence
    
    def get_input_name(self) -> list[str]:
        names: list[str] = list(self.__premises.keys())
        return names

    def get_premises_fuzzy_set(self) -> list[FuzzySet]:
        all_fuzzy_set = []
        for l in self.__premises.values():
            all_fuzzy_set.extend(l)
        return all_fuzzy_set
    
    def evaluate(self, params: dict[str, np.number]) -> tuple[np.number, np.number]:
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        
        if list(params.keys()) != self.get_input_name():
            raise ValueError("params should have the same keys of premises")
    
        for k in params.values():
            if not np.issubdtype(type(k), np.number):
                raise ValueError("every value should be a number")
        
        params_list = []
        for name, l_f in self.__premises.items():
            params_list.extend([params[name]] * len(l_f))

        output_rule = np.dot(self.__weights[1:], params_list) + self.__weights[0]
        weight = self.__fuzzy_composition(params_list)

        return output_rule, weight
    
class SingletonRule(FuzzyRule):
    def __init__(self, 
                 premises : dict[str, list[FuzzySet]], 
                 name_conseguence: str,
                 conseguence: SingletonFuzzySet,
                 tnorm_operation: Callable = minimum):
        
        if not isinstance(tnorm_operation, Callable):
            raise ValueError("tnorm_operation should be a callable")
        
        if not isinstance(premises, dict):
            raise TypeError("premises should be a dict")
        
        if len(premises.keys()) <= 0:
            raise ValueError("premises should have at least of almost 2 elements")       
        
        for k, v in premises.items():
            if not isinstance(k, str) and k != '':
                raise TypeError("All keys should be a not empty string") 
            if not isinstance(v, list):
                raise TypeError('All values should be a list')  
            if len(v) == 0:
                raise ValueError('All values should be a non empty list')  
            for f in v:
                if not isinstance(f, FuzzySet):
                    raise TypeError('All values should be a list of fuzzy sets')
                
        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a not empty string")
        
        if name_conseguence == '':
            raise ValueError("name_conseguence should be a not empty string")
        
        if not isinstance(conseguence, FuzzySet):
            raise TypeError("v should be a FuzzySet")

        if not isinstance(tnorm_operation, Callable):
            raise TypeError("tnorm_operation should be a tnorm fuzzy operation")  
        
        self.__premises: dict[str, list[FuzzySet]] = premises
        self.__name_conseguence: str = name_conseguence
        self.__conseguence: SingletonFuzzySet = conseguence
        self.__tnorm_operation: Callable = tnorm_operation

        
        if isinstance(self.get_premises_fuzzy_set()[0], ContinuousFuzzySet):
            self.__rule: ContinuousFuzzyCombination = ContinuousFuzzyCombination(self.get_premises_fuzzy_set(), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Continuous
        else:
            self.__rule: DiscreteFuzzyCombination = DiscreteFuzzyCombination(self.get_premises_fuzzy_set(), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Discrete
        
    @property
    def premises(self) -> dict[str, FuzzySet]:
        return self.__premises
    
    @property
    def name_conseguence(self) -> str:
        return self.__name_conseguence
    
    def get_input_name(self) -> list[str]:
        names: list[str] = list(self.__premises.keys())
        return names

    def get_premises_fuzzy_set(self) -> list[FuzzySet]:
        all_fuzzy_set = []
        for l in self.__premises.values():
            all_fuzzy_set.extend(l)
        return all_fuzzy_set
    
    def evaluate(self, params: dict[str, np.number]) -> tuple[np.number, np.number]:
        '''
        It evaluates the MamdaniRule given a list of elements, ones per premise.
        '''

        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        
        if list(params.keys()) != self.get_input_name():
            raise ValueError("params should have the same keys of premises")
    
        for k in params.values():
            if not np.issubdtype(type(k), np.number):
                raise ValueError("every value should be a number")
        
        params_list = []
        for name, l_f in self.__premises.items():
            params_list.extend([params[name]] * len(l_f))

        output_rule = self.__conseguence.items[0]
        weight = self.__rule(params_list)

        return output_rule, weight
    
class ClassifierRule(FuzzyRule):
    def __init__(self, 
                 premises : dict[str, list[FuzzySet]], 
                 name_conseguence: str,
                 conseguence: np.number,
                 tnorm_operation: Callable = minimum):
        if not isinstance(tnorm_operation, Callable):
            raise ValueError("tnorm_operation should be a callable")
        
        if not isinstance(premises, dict):
            raise TypeError("premises should be a dict")
        
        if len(premises.keys()) <= 0:
            raise ValueError("premises should have at least of almost 1 element")       
        
        for k, v in premises.items():
            if not isinstance(k, str) and k != '':
                raise TypeError("All keys should be a not empty string") 
            if not isinstance(v, list):
                raise TypeError('All values should be a list')  
            if len(v) == 0:
                raise ValueError('All values should be a non empty list')  
            for f in v:
                if not isinstance(f, FuzzySet):
                    raise TypeError('All values should be a list of fuzzy sets')
                
        if not isinstance(name_conseguence, str):
            raise TypeError("name_conseguence should be a not empty string")
        
        if name_conseguence == '':
            raise ValueError("name_conseguence should be a not empty string")
        
        if not np.issubdtype(type(conseguence), np.number):
            raise TypeError("v should be a class")

        if not isinstance(tnorm_operation, Callable):
            raise TypeError("tnorm_operation should be a tnorm fuzzy operation")  
        
        self.__premises: dict[str, list[FuzzySet]] = premises
        self.__name_conseguence: str = name_conseguence
        self.__conseguence: np.number = conseguence
        self.__tnorm_operation: Callable = tnorm_operation

        
        if isinstance(self.get_premises_fuzzy_set()[0], ContinuousFuzzySet):
            self.__rule: ContinuousFuzzyCombination = ContinuousFuzzyCombination(self.get_premises_fuzzy_set(), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Continuous
        else:
            self.__rule: DiscreteFuzzyCombination = DiscreteFuzzyCombination(self.get_premises_fuzzy_set(), self.__tnorm_operation) 
            self.__type_rule: TypeRule = TypeRule.Discrete

    @property
    def premises(self) -> dict[str, FuzzySet]:
        return self.__premises
    
    @property
    def name_conseguence(self) -> str:
        return self.__name_conseguence
    
    def get_input_name(self) -> list[str]:
        names: list[str] = list(self.__premises.keys())
        return names

    def get_premises_fuzzy_set(self) -> list[FuzzySet]:
        all_fuzzy_set = []
        for l in self.__premises.values():
            all_fuzzy_set.extend(l)
        return all_fuzzy_set
    

    def evaluate(self, params: dict[str, np.number]) -> tuple[str, np.number]:
        '''
        It evaluates the MamdaniRule given a list of elements, ones per premise.
        '''

        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        
        if list(params.keys()) != self.get_input_name():
            raise ValueError("params should have the same keys of premises")
    
        for k in params.values():
            if not np.issubdtype(type(k), np.number):
                raise ValueError("every value should be a number")
        
        params_list = []
        for name, l_f in self.__premises.items():
            params_list.extend([params[name]] * len(l_f))

        output_rule = self.__conseguence
        weight = self.__rule(params_list)

        return output_rule, weight