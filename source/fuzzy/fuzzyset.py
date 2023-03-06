from abc import ABC, abstractmethod
import numpy as np

class FuzzySet(ABC):
    '''Abstract Class for a generic fuzzy set'''
    @abstractmethod
    def __call__(self, arg):
        pass

    @abstractmethod
    def __getitem__(self, alpha):
        pass


class DiscreteFuzzySet(FuzzySet):
    '''
    Implements a discrete fuzzy set
    
    '''
    def __init__(self, items, memberships):
        if type(items) != list and type(items) != np.array:
            raise TypeError("Support set should be list or numpy.array")
        
        if type(memberships) != list and type(memberships) != np.array:
            raise TypeError("Memberships should be list or numpy.array")
        
        self.items = np.array(items)
        self.set = dict(zip(items, range(len(items))))

        for m in memberships:
            if type(m) != float:
                raise TypeError("Membership degrees should be floats in [0,1], is %s" % type(m))
            if  m < 0 or m > 1:
                raise ValueError("Membership degrees should be floats in [0,1], is %s" % m)
            
        self.memberships = np.array(memberships)

    def __call__(self, arg):
        if arg not in self.set.keys():
            raise ValueError("%s not in the support of the fuzzy set" % arg)
        else:
            return self.memberships[self.set[arg]]
        
    def __getitem__(self, alpha):
        if type(alpha) != float:
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        return self.items[self.memberships >= alpha]
    
    def __eq__(self, other):
        return (self.items == other.items).all() and (self.memberships == other.memberships).all()
    

class ContinuousFuzzySet(FuzzySet):
    '''Abstract class for a continuous fuzzy set'''
    pass

class FuzzyNumber(ContinuousFuzzySet):
    '''Abstract class for a fuzzy number (convex, closed fuzzy set over the real numbers)'''
    pass

class IntervalFuzzyNumber(FuzzyNumber):
    '''
    Implements an interval fuzzy number (equivalently, an interval)
    '''
    def __init__(self, lower, upper):
        if type(lower) != float or type(upper) != float:
            raise TypeError("Lower and Higher should be floats")
        
        if lower > upper:
            raise ValueError("Lower should be smaller than Upper")
        
        self.lower = lower
        self.upper = upper

    def __call__(self, arg):
        if type(arg) != float:
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        if arg < self.lower or arg > self.upper:
            return 0
        else:
            return 1
        
    def __getitem__(self, alpha):
        if type(alpha) != float:
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        return self.lower, self.upper
    
    def __eq__(self, other):
        return self.lower == other.lower and self.upper == other.upper
    



class TriangularFuzzyNumber(FuzzyNumber):
    '''
    Implements a triangular fuzzy number
    '''
    def __init__(self, lower, middle, upper):
        if type(lower) != float or type(middle) != float or type(upper) != float:
            raise TypeError("Lower, Middle and Higher should be floats")
        
        if lower > middle or lower > upper:
            raise ValueError("Lower should be smaller than Middle and Upper")
        
        if middle > upper:
            raise ValueError("Middle should be smaller than Upper")
        
        self.lower = lower
        self.middle = middle
        self.upper = upper

    def __call__(self, arg):
        if type(arg) != float:
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        if arg < self.lower or arg > self.upper:
            return 0
        elif arg >= self.lower and arg <= self.middle:
            return (arg - self.lower)/(self.middle - self.lower)
        elif arg >= self.middle and arg <= self.upper:
            return (arg - self.middle)/(self.upper - self.middle)
        
    def __getitem__(self, alpha):
        if type(alpha) != float:
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        low = alpha*(self.middle - self.lower) + self.lower
        upp = alpha*(self.upper - self.middle) + self.middle
        
        return low, upp
    
    def __eq__(self, other):
        return self.lower == other.lower and self.middle == other.middle and self.upper == other.upper
