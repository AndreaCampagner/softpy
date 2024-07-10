import numpy as np

from softpy.fuzzy.fuzzy_set_wrapper.fuzzy_set import FuzzySet

class LambdaFuzzySet(FuzzySet):
    '''Abstract Class for a fuzzy set defined by an explicitly specified membership function'''
    def __init__(self, func):
        self.__func = func

    def __call__(self, arg) -> np.number:
        return self.__func(arg)

    def __getitem__(self, alpha):
        pass

    @property
    def func(self): 
        '''property get of attribute func'''
        return self.__func

    @func.setter
    def func(self, func):
        self.__func = func

    def fuzziness(self):
        pass

    def hartley(self):
        pass
