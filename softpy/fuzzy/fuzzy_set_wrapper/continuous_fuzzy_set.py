from typing import Final
import numpy as np
import scipy as sp

from softpy.fuzzy.fuzzy_set_wrapper.fuzzy_set import FuzzySet

class ContinuousFuzzySet(FuzzySet):
    '''
    Abstract class for a continuous fuzzy set
    Note: each (bounded) continuous fuzzy set has a minimum and maximum elements 
    s.t. their membership degrees are > 0
    Epsilon is used to define an approximation degree for various operations that 
    cannot be implemented exactly 
    '''

    min : np.number
    max : np.number
    epsilon: Final[np.number] = 1e-3
    f: np.number = -1
    __h: np.number = -1

    
    def __getitem__(self, alpha: np.number) -> np.ndarray | tuple:
        '''
        In general, it is not possible to take alpha cuts of continuous fuzzy sets 
        analytically: we need to search

        explicitly for all values whose membership degree is >= alpha. Since it is 
        impossible to look all real numbers,

        we do a grid search where the grid step-size is defined by epsilon
        '''

        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be in [0,1], is {alpha}")



        grid = np.linspace(self.min, self.max, int((self.max - self.min)/self.epsilon))
        vals = np.array([v if self(v) >= alpha else np.nan for v in grid])
        return vals

    def __memberships_function(self, x):
        if np.abs(x - 0) <= self.epsilon or np.abs(x - 1) <= self.epsilon:
            return 0
        return self(x) * np.log(1/self(x)) + (1 - self(x))*np.log2(1/(1-self(x)))

    def fuzziness(self) -> np.number:
        '''
        As in the case of alpha cuts, it is generally impossible to compute the 
        fuzziness of a continuous fuzzy set
        analytically: thus, we perform a numerical integration of the fuzziness 
        function between the minimum and maximum
        values of the fuzzy set (it internally uses the __call__ method: notice 
        that it is not implemented in ContinuousFuzzySet!)
        '''
        try:
            return self.f
        except AttributeError:
            self.f : np.number = sp.integrate.quad(self.__memberships_function, 
                                                   self.min, 
                                                   self.max, 
                                                   epsabs=self.epsilon)[0]
            return self.f
