from __future__ import annotations
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable
import numpy as np
import scipy as sp
from . import memberships_function as mf


class FuzzySet(ABC):
    '''Abstract Class for a generic fuzzy set'''
    @abstractmethod
    def __call__(self, arg) -> np.number:
        '''
        Gets the membership degree of arg
        '''
        pass

    @abstractmethod
    def __getitem__(self, alpha: np.number):
        '''
        Gets an alpha cut
        '''
        pass

    @abstractmethod
    def fuzziness(self) -> np.number:
        '''
        Computes the fuzziness
        '''
        pass

    @abstractmethod
    def hartley(self) -> np.number:
        '''
        Computes the hartley entropy (non-specificity)
        '''
        pass

    @abstractmethod
    def memberships_function(self, x) -> np.number:
        '''
        Utility function for returning the membership values
        '''
        pass

class DiscreteFuzzySet(FuzzySet):
    '''
    Implements a discrete fuzzy set. Requires as input a sequence (list or array) of objects and a sequence (list or array) of membership degrees
    Attribute dynamic controls whether the support set is exhaustive or not (i.e. there exist objects not in items whose membership degree is 0)

    Parameters
    -----------
    :param items: sequence of objects
    :type items: list|np.ndarray

    :param memberships: sequence of membership degrees
    :type items: list|np.ndarray

    :param dynamic: If False, it assumes that the closed world assumption hold (items is an exhaustive enumeration of all relevant objects)
    :type dynamic: bool, default=True
    '''

    def __init__(self, items: list|np.ndarray, memberships: list|np.ndarray, dynamic: bool=True):
        if not isinstance(items, list) and not isinstance(items, np.ndarray):
            raise TypeError("items should be list or numpy.array")
        
        if not isinstance(memberships, list) and not isinstance(memberships, np.ndarray):
            raise TypeError("memberships should be list or numpy.array")
        
        if len(items) != len(memberships):
            raise ValueError("items and memberships should have the same length")

        if not isinstance(dynamic, bool):
            raise TypeError("dynamic should be bool")
        
        self.__items = np.array(items)
        self.__set_items = dict(zip(items, range(len(items))))

        for m in memberships:
            if not np.issubdtype(type(m), np.number):
                raise TypeError("Membership degrees should be floats in [0,1], is %s" % type(m))
            if  m < 0 or m > 1:
                raise ValueError("Membership degrees should be floats in [0,1], is %s" % m)
            
        self.__memberships = np.array(memberships)
        self.__dynamic = dynamic

    @property
    def memberships(self) -> np.ndarray:
        return self.__memberships
    
    @property
    def set_items(self) -> set:
        return self.__set_items
    
    @property
    def items(self) -> list:
        return self.__items
    
    @property
    def dynamic(self) -> bool:
        return self.__dynamic
    
    def memberships_function(self, x) -> np.number:
        if x not in self.__set_items.keys():
            if self.__dynamic:
                self.__set_items[x] = len(self.__items)
                self.__items = np.append(self.__items, x)
                self.__memberships = np.append(self.__memberships, 0.0)
            else:
                raise ValueError("%s not in the support of the fuzzy set" % x)
        return self.__memberships[self.__set_items[x]]
    
    def __call__(self, arg):
        '''
        Gets the membership degree of arg. Uses self.__set_items to enable quick look-up.
        Behavior changes according to value of dynamic
        '''
        return self.memberships_function(arg)
        
    def __getitem__(self, alpha: np.number) -> np.ndarray:
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        return self.__items[self.__memberships >= alpha]
    
    def __eq__(self, other: object) -> bool:
        '''
        Checks whether two DiscreteFuzzySet instances are equal
        '''
        if not isinstance(other, DiscreteFuzzySet):
            return NotImplemented
        
        for v in list(self.__set_items.keys()) + list(other.set_items.keys()):
            try:
                v1 = self(v)
            except ValueError:
                return False
            try:
                v2 = other(v)
            except ValueError:
                return False
            
            if v1 != v2:
                return False
        return  True
    
    def fuzziness(self) -> np.number:
        try:
            return self.f
        except AttributeError:
            pos = self.__memberships[self.__memberships > 0]
            pos = pos*np.log2(1/pos)
            non = self.__memberships[self.__memberships < 1]
            non = (1-non)*np.log2(1/(1-non))
            self.f : np.number = np.sum(pos) + np.sum(non)
            return self.f
        
    def hartley(self) -> np.number:
        try:
            return self.h
        except AttributeError:
            pos = self.__memberships[self.__memberships > 0]
            sort = np.append(np.sort(pos)[::-1], 0)
            coeffs = sort[:-1] - sort[1:]
            sizes = np.log2(np.array([len(self[i]) for i in sort[:-1]]))
            self.h : np.number = np.sum(coeffs*sizes)
            return self.h
    

class ContinuousFuzzySet(FuzzySet):
    '''
    Class for a generic unbounded continuous fuzzy set. If the value of memberships function 
    exceeds interval [0,1], every exceeded memberships will be truncated at 0 or 1.

    :param memberships_function: a callable object that returns for each real number its corresponding membership degree
    :type memberships_function: Callable[[np.number], np.number]

    :param bound: a tuple representing the domain of definition of memberships_function
    :type bound: tuple[np.number, np.number], default=(-np.inf, np.inf)

    :param epsilon: tolerance parameter for numerical routines, which are implemented as approximate search
    :type epsilon: np.number, default=1e-3
    '''

    def __init__(self, 
                 memberships_function: Callable[[np.number], np.number], 
                 bound: tuple[np.number, np.number] = (-np.inf, np.inf),
                 epsilon: np.number = 1e-3):
        
        self.__epsilon: np.number = 1e-3
        self.__memberships_function: Callable[[np.number], np.number]
        self.__bound: tuple[np.number, np.number]
        self._f: np.number = -1
        self._h: np.number = -1
        
        if not isinstance(memberships_function, Callable):
            raise TypeError("Memberships function should be a lambda that takes" + 
                            "as input a real number and returns a value in [0,1]")
        
        if not isinstance(bound, tuple):
            raise TypeError("bound should be a tuple")
        
        if bound is None:
            bound = (-np.inf, np.inf)
        
        if bound[0] > bound[1]:
            raise ValueError("bound[0] should be less equal than bound[1]")

        if not np.issubdtype(type(epsilon), np.number):
            raise TypeError("epsilon should be a number")

        if epsilon >= 1 or epsilon <= 0:
            raise ValueError("Epsilon should be small positive number, ex: 1e-3")

        self.__memberships_function = memberships_function
        self.__epsilon = epsilon
        self.__bound = bound

        #self.fuzziness()
        #self.hartley()

    
    def memberships_function(self, x) -> np.number:
        if not np.issubdtype(type(x), np.number):
            raise TypeError("x should be a number")
        
        member = self.__memberships_function(x)
        
        if self.bound[0] <= x <= self.bound[1]:
            if member <= 0:
                return 0
            elif member >= 1:
                return 1
            return member
        
        return 0

    @property
    def epsilon(self) -> np.number:
        return self.__epsilon
    
    @property
    def bound(self) -> tuple:
        return self.__bound

    def __call__(self, arg: np.number) -> np.number:
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be a number")

        return self.memberships_function(arg)

    def __getitem__(self, alpha) -> np.ndarray:
        '''
        Implements a numerical approximate search for an alpha cut. Since for continuous fuzzy set is in general
        impossible to compute the alpha cuts in closed form, the method simply checks values in a grid and returns
        the list of all elements having membership degree larger than alpha
        '''
        if self.bound is None:
            raise AttributeError("You should define the interval on which you want compute alpha cut")

        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        step = int((self.bound[1] - self.bound[0] + 1)/self.epsilon)

        x_values = np.linspace(self.bound[0], 
                               self.bound[1], 
                               step)
        
        discr_memb_func = np.array([self(x) for x in  x_values])

        alpha_cut = np.array([v if memb >= alpha else np.nan for v, memb in zip(x_values, discr_memb_func)])

        return alpha_cut

    def fuzziness(self) -> np.number:
        '''
        As in the case of alpha cuts, it is generally impossible to compute the fuzziness of a continuous fuzzy set
        analytically: thus, we perform a numerical integration of the fuzziness function between the minimum and maximum
        values of the fuzzy set (it internally uses the __call__ method: notice that it is not implemented in ContinuousFuzzySet!)
        '''
        if self._f == -1:
            self._f = sp.integrate.quad(lambda x: 1 - np.abs(2 * self(x) - 1), 
                                       self.bound[0], 
                                       self.bound[1], 
                                       epsabs=self.epsilon)[0]
        return self._f
        
    def hartley(self) -> np.number:
        '''
        As in the case of alpha cuts, it is generally impossible to compute the fuzziness of a continuous fuzzy set
        analytically: thus, we perform a numerical integration of the fuzziness function between the minimum and maximum
        values of the fuzzy set (it internally uses the __call__ method: notice that it is not implemented in ContinuousFuzzySet!)
        '''

        if self._h == -1:
            
            self._h =  np.log2(sp.integrate.quad(lambda x: self.memberships_function(x), 
                                         self.bound[0], 
                                         self.bound[1], 
                                         epsabs=self.epsilon))[0]
        
        return self._h

class SingletonFuzzySet(DiscreteFuzzySet):
    '''
    Implements a DiscreteFuzzySet with a single element
    '''
    def __init__(self, value, memb: np.number = 1) -> None:
        super().__init__([value], [memb])

    
class TriangularFuzzySet(ContinuousFuzzySet):
    '''
    Implements a triangular fuzzy set

    Parameters
    ----------
    :param left: the left tail of the triangular fuzzy set
    :param left: np.number

    :param spike: the middle point of the triangular fuzzy set
    :type spike: np.number

    :param right: the right tail of the triangular fuzzy set
    :param right: np.number
    '''
    def __init__(self, 
                 left: np.number,
                 spike: np.number,
                 right: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(left), np.number):
            raise TypeError("a should be a number")
        
        if not np.issubdtype(type(spike), np.number):
            raise TypeError("b should be a number")
        
        if not np.issubdtype(type(right), np.number):
            raise TypeError("c should be a number")

        if not (left <= spike <= right):
            raise ValueError("Parameters a, b and c should be a <= b <= c")

        if bound is None:
            bound=(left, right)

        super().__init__(partial(mf.triangular, a = left, b = spike, c = right), 
                         epsilon=epsilon, 
                         bound=bound)

        self.__left = left
        self.__spike = spike
        self.__right = right

class TrapezoidalFuzzySet(ContinuousFuzzySet):
    '''
    Implements a trapezoidal fuzzy set

    Parameters
    ----------
    :param left_lower: the left tail of the trapezoidal fuzzy set
    :type left_lower: np.number

    :param left_upper: the upper left point of the trapezoidal fuzzy set
    :type left_upper: np.number

    :param right_upper: the upper right point of the trapezoidal fuzzy set
    :type right_upper: np.number

    :param right_lower: the right tail of the trapezoidal fuzzy set
    :type right_lower: np.number
    '''
    def __init__(self, 
                 left_lower: np.number,
                 left_upper: np.number,
                 right_upper: np.number,
                 right_lower: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(left_lower), np.number):
            raise TypeError("left_lower should be a number")
        
        if not np.issubdtype(type(left_upper), np.number):
            raise TypeError("left_upper should be a number")
        
        if not np.issubdtype(type(right_upper), np.number):
            raise TypeError("right_upper should be a number")
        
        if not np.issubdtype(type(right_lower), np.number):
            raise TypeError("right_lower should be a number")
        
        if not (left_lower <= left_upper <= right_upper <= right_lower):
            raise ValueError("Parameters left_lower, left_upper, right_upper and right_lower should be left_lower" +
                             "<= left_upper <= right_upper <= right_lower")

        if bound is None:
            bound=(left_lower, right_lower)
            
        super().__init__(partial(mf.trapezoidal, 
                                 a = left_lower, 
                                 b = left_upper, 
                                 c = right_upper, 
                                 d = right_lower), epsilon=epsilon, bound=bound)

        self.__left_lower = left_lower
        self.__left_upper = left_upper
        self.__right_upper = right_upper
        self.__right_lower = right_lower
        
class LinearZFuzzySet(ContinuousFuzzySet):
    '''
    Implements a linear z-shaped (descending ramp) fuzzy set

    Parameters
    ----------
    :param left_upper: the upper left tail of the triangular fuzzy set
    :param left_upper: np.number

    :param right_lower: the lower right tail of the triangular fuzzy set
    :param right_lower: np.number
    '''    
    def __init__(self, 
                 left_upper: np.number,
                 right_lower: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(left_upper), np.number):
            raise TypeError("left_upper should be a number")
        
        if not np.issubdtype(type(right_lower), np.number):
            raise TypeError("right_lower should be a number")
        
        if not (left_upper <= right_lower):
            raise ValueError("Parameters left_upper and right_lower should be left_upper <= right_lower")

        if bound is None:
            bound=(-np.inf, right_lower)

        super().__init__(partial(mf.linear_z_shaped, 
                                 a = left_upper, 
                                 b = right_lower), epsilon=epsilon, bound=bound)

        self.__left_upper = left_upper
        self.__right_lower = right_lower

class LinearSFuzzySet(ContinuousFuzzySet):

    '''
    Implements a linear s-shaped (ascending ramp) fuzzy set

    Parameters
    ----------
    :param left_lower: the lower left tail of the triangular fuzzy set
    :param left_lower: np.number

    :param right_upper: the upper right tail of the triangular fuzzy set
    :param right_upper: np.number
    '''    
    def __init__(self, 
                 left_lower: np.number,
                 right_upper: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(left_lower), np.number):
            raise TypeError("left_lower should be a number")
        
        if not np.issubdtype(type(right_upper), np.number):
            raise TypeError("right_upper should be a number")
        
        if not (left_lower <= right_upper):
            raise ValueError("Parameters left_lower and right_upper should be left_lower <= right_upper")

        if bound is None:
            bound= (left_lower, np.inf)

        super().__init__(partial(mf.linear_s_shaped, 
                                 a = left_lower, 
                                 b = right_upper), epsilon=epsilon, bound=bound)

        self.__left_lower = left_lower
        self.__right_upper = right_upper

class GaussianFuzzySet(ContinuousFuzzySet):
  
    '''
    Implements a gaussian fuzzy set

    Parameters
    ----------
    :param mean: the middle point of the gaussian fuzzy set
    :param mean: np.number

    :param std: the standard deviation of the gaussian fuzzy set
    :param std: np.number
    '''    
    def __init__(self, 
                 mean: np.number,
                 std: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(mean), np.number):
            raise TypeError("mean should be a number")
        
        if not np.issubdtype(type(std), np.number):
            raise TypeError("std should be a number")
        
        if std <= 0:
            raise ValueError("std should greater than 0")
        
        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.gaussian, mean=mean, std=std), epsilon=epsilon, bound=bound)

        self.__mean = mean
        self.__std = std

class Gaussian2FuzzySet(ContinuousFuzzySet):
    '''
    Implements a double gaussian fuzzy set

    Parameters
    ----------
    :param mean1: the middle point of the first gaussian fuzzy set
    :param mean1: np.number

    :param std1: the standard deviation of the first gaussian fuzzy set
    :param std1: np.number

    :param mean2: the middle point of the second gaussian fuzzy set
    :param mean2: np.number

    :param std2: the standard deviation of the second gaussian fuzzy set
    :param std2: np.number
    '''        
    def __init__(self, 
                 mean1: np.number, 
                 std1: np.number,
                 mean2: np.number, 
                 std2: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(mean1), np.number):
            raise TypeError("mean1 should be a number")
        
        if not np.issubdtype(type(std1), np.number):
            raise TypeError("std1 should be a number")
        
        if std1 <= 0:
            raise ValueError("std1 should greater than 0")
        
        if not np.issubdtype(type(mean2), np.number):
            raise TypeError("mean2 should be a number")
        
        if not np.issubdtype(type(std2), np.number):
            raise TypeError("std2 should be a number")
        
        if std2 <= 0:
            raise ValueError("std2 should greater than 0")

        if mean1 > mean2:
            raise ValueError("mean1 should be less equal than mean2")
        
        if bound is None:
            bound=(-np.inf, np.inf)
            
        super().__init__(partial(mf.gaussian2, 
                                 mean1=mean1, 
                                 std1=std1, 
                                 mean2=mean2, 
                                 std2=std2), epsilon=epsilon, bound=bound)

        self.__mean1 = mean1
        self.__std1 = std1
        self.__mean2 = mean2
        self.__std2 = std2

class GBellFuzzySet(ContinuousFuzzySet):
    '''
    Implements a generalized bell-shaped fuzzy set

    Parameters
    ----------
    :param width: controls the width of the generalized bell fuzzy set (set = 1 for a Gaussian fuzzy set)
    :param width: np.number

    :param slope: the exponent of the generalized bell fuzzy set (set = 1 for a Gaussian Fuzzy set)
    :param slope: np.number

    :param center: the middle point of the generalized bell fuzzy set
    :param center: np.number
    '''  
    def __init__(self, 
                 width: np.number, 
                 slope: np.number,
                 center: np.number, 
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(width), np.number):
            raise TypeError("width should be a number")
        
        if width <= 0:
            raise ValueError("width should be a positive value")
        
        if not np.issubdtype(type(slope), np.number):
            raise TypeError("slope should be a number")
        
        if slope < 0:
            raise ValueError("slope should be a positive value")
        
        if not np.issubdtype(type(center), np.number):
            raise TypeError("center should be a number")
        
        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.gbell, 
                                 a = width, 
                                 b = slope, 
                                 c = center), epsilon=epsilon, bound=bound)

        self.__width = width
        self.__slope = slope
        self.__center = center

class SigmoidalFuzzySet(ContinuousFuzzySet):
    '''
    Implements a sigmoid fuzzy set

    Parameters
    ----------
    :param width: controls the width of the sigmoid fuzzy set
    :param width: np.number

    :param center: the middle point of the sigmoid fuzzy set
    :param center: np.number
    '''  
    def __init__(self, 
                 width: np.number, 
                 center: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(width), np.number):
            raise TypeError("width should be a number")
        
        if width == 0:
            raise ValueError("width should be a positive value or a negative")
        
        if not np.issubdtype(type(center), np.number):
            raise TypeError("center should be a number")
        
        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.sigmoidal, 
                                 a = width, 
                                 c = center), epsilon=epsilon, bound=bound)

        self.__width = width
        self.__center = center

class DiffSigmoidalFuzzySet(ContinuousFuzzySet):
    '''
    Implements a fuzzy set defined by the difference of two sigmoids

    Parameters
    ----------
    :param width1: controls the width of the first sigmoid fuzzy set
    :param width1: np.number

    :param center1: the middle point of the first sigmoid fuzzy set
    :param center1: np.number

    :param width2: controls the width of the second sigmoid fuzzy set
    :param width2: np.number

    :param center2: the middle point of the second sigmoid fuzzy set
    :param center2: np.number
    '''  
    def __init__(self, 
                 width1: np.number, 
                 center1: np.number,
                 width2: np.number, 
                 center2: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(width1), np.number):
            raise TypeError("width1 should be a number")
        
        if width1 == 0:
            raise ValueError("width1 should be a positive value or a negative")
        
        if not np.issubdtype(type(center1), np.number):
            raise TypeError("center1 should be a number")
        
        if not np.issubdtype(type(width2), np.number):
            raise TypeError("width2 should be a number")
        
        if width2 == 0:
            raise ValueError("width2 should be a positive value or a negative")
        
        if not np.issubdtype(type(center2), np.number):
            raise TypeError("center2 should be a number")
        
        if center1 >= center2:
            raise ValueError("center1 should be less equal than center2")
        
        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.difference_sigmoidal, 
                                 a1 = width1, 
                                 c1 = center1, 
                                 a2 = width2, 
                                 c2 = center2), epsilon=epsilon, bound=bound)

        self.__width1 = width1
        self.__center1 = center1
        self.__width2 = width2
        self.__center2 = center2

        
class ProdSigmoidalFuzzySet(ContinuousFuzzySet):
    '''
    Implements a fuzzy set defined by the  product of two sigmoids

    Parameters
    ----------
    :param width1: controls the width of the first sigmoid fuzzy set
    :param width1: np.number

    :param center1: the middle point of the first sigmoid fuzzy set
    :param center1: np.number

    :param width2: controls the width of the second sigmoid fuzzy set
    :param width2: np.number

    :param center2: the middle point of the second sigmoid fuzzy set
    :param center2: np.number
    '''  
    def __init__(self, 
                 width1: np.number, 
                 center1: np.number,
                 width2: np.number, 
                 center2: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(width1), np.number):
            raise TypeError("width1 should be a number")
        
        if width1 == 0:
            raise ValueError("width1 should be a positive value or a negative")
        
        if not np.issubdtype(type(center1), np.number):
            raise TypeError("center1 should be a number")
        
        if not np.issubdtype(type(width2), np.number):
            raise TypeError("width2 should be a number")
        
        if width2 == 0:
            raise ValueError("width2 should be a positive value or a negative")
        
        if not np.issubdtype(type(center2), np.number):
            raise TypeError("center2 should be a number")
        
        if center1 >= center2:
            raise ValueError("center1 should be less equal than center2")
        
        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.product_sigmoidal, 
                                 a1 = width1, 
                                 c1 = center1, 
                                 a2 = width2, 
                                 c2 = center2), epsilon=epsilon, bound=bound)

        self.__width1 = width1
        self.__center1 = center1
        self.__width2 = width2
        self.__center2 = center2

class ZShapedFuzzySet(ContinuousFuzzySet):
    '''
    Implements a z-shaped fuzzy set (smoothed linear z)

    Parameters
    ----------
    :param left_upper: the largest value for which the membership degree is 1
    :param left_upper: np.number

    :param right_lower: the smallest value for which the membership degree is 0
    :param right_lower: np.number
    '''  
    def __init__(self, 
                 left_upper: np.number, 
                 right_lower: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(left_upper), np.number):
            raise TypeError("left_upper should be a number")
        
        if not np.issubdtype(type(right_lower), np.number):
            raise TypeError("right_lower should be a number")
        
        if left_upper > right_lower:
            raise ValueError("left_upper should be less equal than right_lower ")
                
        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.z_shaped, 
                                 a = left_upper, 
                                 b = right_lower), epsilon=epsilon, bound=bound)

        self.__left_upper = left_upper
        self.__right_lower = right_lower

class SShapedFuzzySet(ContinuousFuzzySet):
    '''
    Implements a s-shaped fuzzy set (smoothed linear s)

    Parameters
    ----------
    :param left_lower: the largest value for which the membership degree is 0
    :param left_lower: np.number

    :param right_upper: the smallest value for which the membership degree is 1
    :param right_upper: np.number
    '''  
    def __init__(self, 
                 left_lower: np.number, 
                 right_upper: np.number,
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(left_lower), np.number):
            raise TypeError("left_lower should be a number")
        
        if not np.issubdtype(type(right_upper), np.number):
            raise TypeError("right_upper should be a number")
        
        if left_lower > right_upper:
            raise ValueError("left_upper should be less equal than right_lower ")

        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.s_shaped, 
                                 a = left_lower, 
                                 b = right_upper), epsilon=epsilon, bound=bound)

        self.__left_lower = left_lower
        self.__right_upper = right_upper

class PiShapedFuzzySet(ContinuousFuzzySet):
    '''
    Implements a pi-shaped fuzzy set (smoothed trapezoidal)

    Parameters
    ----------
    :param left_lower: the largest value for which the membership degree is 0
    :param left_upper: np.number

    :param left_upper: the smallest value for which the membership degree is 1
    :param left_upper: np.number

    :param right_upper: the largest value for which the membership degree is 1
    :param right_upper: np.number

    :param right_lower: the smallest value for which the membership degree is 0
    :param right_lower: np.number
    '''  
    def __init__(self, 
                 left_lower: np.number, 
                 left_upper: np.number,
                 right_upper: np.number,
                 right_lower: np.number, 
                 epsilon: np.number = 0.001,
                 bound: tuple[np.number, np.number] = None) -> None:
        
        if not np.issubdtype(type(left_lower), np.number):
            raise TypeError("left_lower should be a number")
        
        if not np.issubdtype(type(left_upper), np.number):
            raise TypeError("left_upper should be a number")
        
        if not np.issubdtype(type(right_upper), np.number):
            raise TypeError("right_upper should be a number")
        
        if not np.issubdtype(type(right_lower), np.number):
            raise TypeError("right_lower should be a number")
        
        if not (left_lower <= left_upper < right_upper <= right_lower):
            raise ValueError("Parameters should be left_lower <= left_upper <= right_upper <= right_lower ")
                
        if bound is None:
            bound=(-np.inf, np.inf)

        super().__init__(partial(mf.pi_shaped, 
                                 a = left_lower, 
                                 b = left_upper,
                                 c = right_upper, 
                                 d = right_lower), epsilon=epsilon, bound=bound)

        self.__left_lower = left_lower
        self.__left_upper = left_upper
        self.__right_upper = right_upper
        self.__right_lower = right_lower

