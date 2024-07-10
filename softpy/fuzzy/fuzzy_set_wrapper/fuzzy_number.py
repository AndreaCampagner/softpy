import numpy as np
import scipy as sp

from softpy.fuzzy.fuzzy_set_wrapper.continuous_fuzzy_set import ContinuousFuzzySet

class FuzzyNumber(ContinuousFuzzySet):
    '''
    Abstract class for a fuzzy number (convex, closed fuzzy set over the real numbers)
    '''
    h : np.number

    def __entropy(self, x):
        return np.log2(self[x][1] - self[x][0])

    def entropy(self) -> np.number:
        '''
        In the case of fuzzy numbers it is easy to compute the hartley entropy, 
        because we know that each
        alpha cut is an interval. Thus, for each alpha, we simply compute the 
        (logarithm of the) length
        of the corresponding interval and then integrate over alpha
        '''
        try:
            return self.h
        except AttributeError:
            self.h : np.number = sp.integrate.quad(self.__entropy, 0, 1)[0]
            return self.h


class IntervalFuzzyNumber(FuzzyNumber):
    '''
    Implements an interval fuzzy number (equivalently, an interval)
    '''
    def __init__(self, lower: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError(f"Lower should be float, is {type(lower)}")
        
        if not np.issubdtype(type(upper), np.number):
            raise TypeError(f"Upper should be float, is {type(upper)}")
        
        if lower > upper:
            raise ValueError("Lower should be smaller than Upper")

        self.lower = lower
        self.upper = upper
        self.min = lower
        self.max = upper

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError(f"Arg should be float, is {type(arg)}")

        if arg < self.lower or arg > self.upper:
            return 0.0
        return 1.0

    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be in [0,1], is {alpha}")

        return self.lower, self.upper

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalFuzzyNumber):
            return NotImplemented
        return self.lower == other.lower and self.upper == other.upper

    def fuzziness(self) -> np.number:
        return 0.0

    def hartley(self) -> np.number:
        return np.log2(self.upper - self.lower)



class RampFuzzyNumber(FuzzyNumber):
    '''
    Implements a ramp fuzzy number
    '''
    def __init__(self, lower: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError(f"Lower should be float, is {type(lower)}")

        if not np.issubdtype(type(upper), np.number):
            raise TypeError(f"Upper should be float, is {type(upper)}")

        self.lower = lower
        self.upper = upper
        self.min = np.min([lower,upper])
        self.max = np.max([lower,upper])

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError(f"Arg should be float, is {type(arg)}")

        if self.lower <= self.upper:
            if arg <= self.lower:
                return 0
            if arg >= self.upper:
                return 1
        else:
            if arg >= self.lower:
                return 0
            if arg <= self.upper:
                return 1

        return (arg - self.lower)/(self.upper - self.lower)


    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be in [0,1], is {alpha}")

        x = (self.upper - self.lower)*alpha + self.lower
        if self.upper > self.lower:
            return x, self.upper
        return self.upper, x

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RampFuzzyNumber):
            return NotImplemented
        return self.lower == other.lower and self.upper == other.upper
    
    def __memberships_function(self, x):
        partial = (x - self.lower)/(self.upper - self.lower)
        return partial*np.log(1/partial) + (1 - partial)*np.log2(1/(1-partial))
    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            self.f : np.number = sp.integrate.quad(self.__memberships_function, 
                                                   self.lower, 
                                                   self.upper)[0]
            return self.f




class TriangularFuzzyNumber(FuzzyNumber):
    '''
    Implements a triangular fuzzy number
    '''
    def __init__(self, lower: np.number, middle: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError(f"Lower should be floats, is {type(lower)}")

        if not np.issubdtype(type(middle), np.number):
            raise TypeError(f"Middle should be floats, is {type(middle)}")

        if not np.issubdtype(type(upper), np.number):
            raise TypeError(f"Higher should be floats, is {type(upper)}")

        if lower > middle or lower > upper:
            raise ValueError("Lower should be smaller than Middle and Upper")

        if middle > upper:
            raise ValueError("Middle should be smaller than Upper")

        self.lower = lower
        self.middle = middle
        self.upper = upper

        self.min = lower
        self.max = upper

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError(f"Arg should be float, is {type(arg)}")

        if arg < self.lower or arg > self.upper:
            return 0
        if self.lower <= arg <= self.middle:
            return (arg - self.lower)/(self.middle - self.lower)
        return (self.upper - arg)/(self.upper - self.middle)

    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be in [0,1], is {alpha}")

        low = alpha*(self.middle - self.lower) + self.lower
        upp = self.upper - alpha*(self.upper - self.middle)

        return low, upp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TriangularFuzzyNumber):
            return NotImplemented
        return (self.lower == other.lower and 
                self.middle == other.middle and 
                self.upper == other.upper)

    def __left_memberships_function(self, x):
        partial_comp = (x - self.lower)/(self.middle - self.lower)
        return (partial_comp*np.log(1/partial_comp) + 
                (1 - partial_comp)*np.log2(1/(1-partial_comp)))
    
    def __right_memberships_function(self, x):
        partial_comp = (self.upper - x)/(self.upper - self.middle)
        return (partial_comp*np.log(1/partial_comp) + 
                (1 - partial_comp)*np.log2(1/(1-partial_comp)))

    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            self.f : np.number = sp.integrate.quad(self.__left_memberships_function, 
                                                   self.lower, 
                                                   self.middle)[0]
            self.f += sp.integrate.quad(self.__right_memberships_function, 
                                        self.middle, 
                                        self.upper)[0]
            return self.f

class TrapezoidalFuzzyNumber(FuzzyNumber):
    '''
    Implements a trapezoidal fuzzy number 
    '''
    def __init__(self, lower: np.number, middle1: np.number, middle2: np.number, upper: np.number):
        if not np.issubdtype(type(lower), np.number):
            raise TypeError(f"Lower should be floats, is {type(lower)}")

        if not np.issubdtype(type(middle1), np.number):
            raise TypeError(f"Middle should be floats, is {type(middle1)}")

        if not np.issubdtype(type(middle2), np.number):
            raise TypeError(f"Middle should be floats, is {type(middle2)}")

        if not np.issubdtype(type(upper), np.number):
            raise TypeError(f"Higher should be floats, is {type(upper)}")

        if lower > middle1 or lower > middle2 or lower > upper:
            raise ValueError("Lower should be the smallest input")

        if middle1 > middle2 or middle1 > upper:
            raise ValueError("Middle1 should be smaller than Middle2 and Upper")

        if middle2 > upper:
            raise ValueError("Upper should be the largest input")

        self.lower = lower
        self.middle1 = middle1
        self.middle2 = middle2
        self.upper = upper

        self.min = lower
        self.max = upper

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError(f"Arg should be float, is {type(arg)}")

        if arg < self.lower or arg > self.upper:
            return 0
        if self.lower <= arg  <= self.middle1:
            return (arg - self.lower)/(self.middle1 - self.lower)
        if self.middle1 <= arg <= self.middle2:
            return 1
        return (self.upper - arg)/(self.upper - self.middle2)

    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be in [0,1], is {alpha}")

        low = alpha*(self.middle1 - self.lower) + self.lower
        upp = alpha*(self.upper - self.middle2) + self.middle2

        return low, upp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrapezoidalFuzzyNumber):
            return NotImplemented
        return (self.lower == other.lower and 
                self.middle1 == other.middle1 and 
                self.middle2 == other.middle2 and 
                self.upper == other.upper)
    
    def __left_memberships_function(self, x):
        partial_comp = (x - self.lower)/(self.middle1 - self.lower)
        return (partial_comp*np.log(1/partial_comp) + 
                (1 - partial_comp)*np.log2(1/(1-partial_comp)))
    
    def __right_memberships_function(self, x):
        partial_comp = (self.upper - x)/(self.upper - self.middle2)
        return (partial_comp*np.log(1/partial_comp) + 
                (1 - partial_comp)*np.log2(1/(1-partial_comp)))
    
    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            self.f = sp.integrate.quad(self.__left_memberships_function, 
                                       self.lower, 
                                       self.middle1)[0]
            self.f += sp.integrate.quad(self.__right_memberships_function, 
                                        self.middle2, 
                                        self.upper)[0]
            return self.f




class GaussianFuzzyNumber(FuzzyNumber):
    '''
    Implements a Gaussian fuzzy number
    '''
    def __init__(self, mean: np.number, std: np.number):
        if not np.issubdtype(type(mean), np.number):
            raise TypeError(f"Mean should be float, is {type(mean)}")

        if not np.issubdtype(type(std), np.number):
            raise TypeError(f"Middle should be floats, is {type(std)}")

        if std < 0:
            raise ValueError("std should be positive")

        self.mean = mean
        self.std = std

        self.min = self.mean - np.sqrt(2*self.std**2 * np.log(1/0.001))
        self.max = self.mean + np.sqrt(2*self.std**2 * np.log(1/0.001))

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError(f"Arg should be float, is {type(arg)}")
        return np.exp(-(arg - self.mean)**2/(2*self.std**2))

    def __getitem__(self, alpha: np.number) -> tuple[np.number, np.number]:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be in [0,1], is {alpha}")

        low = self.mean - np.sqrt(2*self.std**2 * np.log(1/alpha))
        upp = self.mean + np.sqrt(2*self.std**2 * np.log(1/alpha))
        return low, upp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussianFuzzyNumber):
            return NotImplemented
        return self.mean == other.mean and self.std == other.std
    
    def fuzziness(self) -> np.number:
        return self.fuzziness()
