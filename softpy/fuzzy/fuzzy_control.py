from functools import partial
import math
from typing import Callable
import numpy as np

from abc import ABC, abstractmethod

from softpy.fuzzy.knowledge_base import KnowledgeBaseABC

def linear_scaling(x, gamma = 1, v = 0): 
    return gamma * x + v

def non_linear_scaling(x, alpha = 2): 
    sign = partial(math.copysign, 1)
    return sign(x) * abs(x) ** alpha

class FuzzyControlSystemABC(ABC):
    @abstractmethod
    def evaluate(self, params: dict[str, np.number]):
        pass

class ControlSystem(FuzzyControlSystemABC):

    def __init__(self, 
                 kb: KnowledgeBaseABC,
                 input_scaling_func: Callable = linear_scaling,
                 output_scaling_func: Callable = linear_scaling) -> None:

        if not isinstance(input_scaling_func, Callable):
            raise TypeError('input_scaling_func should be a callable')
        
        if not isinstance(output_scaling_func, Callable):
            raise TypeError('output_scaling_func should be a callable')
        
        if not isinstance(kb, KnowledgeBaseABC):
            raise TypeError('kb should be a subclass of KnowledgeBase')

        self.__input_scaling_func = input_scaling_func
        self.__output_scaling_func = output_scaling_func
        self.__kb = kb

    def evaluate(self, params: dict[str, np.number]):
        for k in params.keys():
            params[k] = self.__input_scaling_func(params[k])

        output = self.__kb.infer(params)

        for k in output.keys():
            output[k] = self.__output_scaling_func(output[k])
        
        return output