from functools import partial
import math
from typing import Callable
import numpy as np

from abc import ABC, abstractmethod

from .knowledge_base import KnowledgeBaseABC

def linear_scaling(x, gamma = 1, v = 0): 
    return gamma * x + v

def non_linear_scaling(x, alpha = 2): 
    sign = partial(math.copysign, 1)
    return sign(x) * abs(x) ** alpha

class ControlSystemABC(ABC):
    '''
    Abstract class for a generic control system
    '''
    @abstractmethod
    def evaluate(self, params: dict[str, np.number]):
        pass

class FuzzyControlSystem(ControlSystemABC):
    '''
    Implements a fuzzy control system. The logic of the fuzzy control system itself is implemented in kb parameter. It provides the
    possibility to apply scaling (as well as arbitrary transformations) to both the input and output variables of the control system.

    Parameters
    ----------
    :param kb: a knowledge base that provides the logic for the fuzzy control system
    :type kb: KnowledgeBaseABC

    :param input_scaling_func: a Callable object to scale or transform the input variables to the system
    :type input_scaling_func: Callable, default=linear_scaling

    :param output_scaling_func: a Callable object to scale or transform the output controls of the system
    :type output_scaling_func: Callable, default=linear_scaling
    '''
    def __init__(self, 
                 kb: KnowledgeBaseABC,
                 input_scaling_func: Callable = linear_scaling,
                 output_scaling_func: Callable = linear_scaling):

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
        '''
        Computes the control to be applied based on the given input parameter values.
        '''
        for k in params.keys():
            params[k] = self.__input_scaling_func(params[k])

        output = self.__kb.infer(params)

        for k in output.keys():
            output[k] = self.__output_scaling_func(output[k])
        
        return output