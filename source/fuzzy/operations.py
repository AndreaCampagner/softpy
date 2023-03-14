from __future__ import annotations
import numpy as np
from fuzzy.fuzzyset import FuzzySet,  DiscreteFuzzySet, ContinuousFuzzySet
from collections.abc import Sequence

class FuzzyCombination(FuzzySet):
    def __init__(self, left: FuzzySet, right: FuzzySet, op=None):
        if not isinstance(left, FuzzySet) or not isinstance(right, FuzzySet):
            raise TypeError("All arguments should be fuzzy sets")
            
        self.left = left
        self.right = right
        self.op = op

    def __call__(self, arg):
        if isinstance(arg, tuple) or isinstance(arg, np.ndarray) or isinstance(arg, list):
            return self.op(self.left(arg[0]), self.right(arg[1:]))
        else:
            return self.op(self.left(arg), self.right(arg))
        
    def __getitem__(self, alpha):
        pass

    def fuzziness(self):
        pass

    def hartley(self):
        pass


class ContinuousFuzzyOWA(ContinuousFuzzySet):
    '''
    Support class to implement the OWA operation between ContinuousFuzzySet instances.
    Notice: the result of OWA on ContinuousFuzzySet instances is a ContinuousFuzzySet
    '''

    def __init__(self, fuzzysets, weights):
        '''
        The constructor searches the given fuzzysets in order to determine the minimum and maximum of the result of OWA
        '''
        if not isinstance(fuzzysets, list) and not isinstance(fuzzysets, np.ndarray) and not isinstance(fuzzysets, tuple):
            raise TypeError("arr should be a sequence")
        
        if not isinstance(weights, list) and not isinstance(weights, np.ndarray) and not isinstance(weights, tuple):
            raise TypeError("w should be a sequence")
        
        if len(fuzzysets) != len(weights):
            raise ValueError("arr and w should have the same length")
        
        self.min = np.infty
        self.max = -np.infty
        
        for i in range(len(fuzzysets)):
            if not np.issubdtype(type(weights[i]), np.number) or weights[i] < 0 or weights[i] > 1:
                raise TypeError("w should be a sequence of floats in [0,1]")
            if not isinstance(fuzzysets[i], ContinuousFuzzySet):
                raise TypeError("Arguments should be all continuous fuzzy sets")
            if fuzzysets[i].min < self.min:
                self.min = fuzzysets[i].min
            if fuzzysets[i].max > self.max:
                self.max = fuzzysets[i].max
            
        self.fuzzysets = np.array(fuzzysets)
        self.weights = np.array(weights)

    def __call__(self, arg):
        '''
        Simply computes the membership degree of arg by evaluating the OWA with the given weights
        '''
        if isinstance(arg, tuple):
            ms = np.sort([self.fuzzysets[i](arg[i]) for i in range(len(self.fuzzysets))])
            return np.sum(ms*self.weights)  
        else:
            ms = np.sort([self.fuzzysets[i](arg) for i in range(len(self.fuzzysets))])
            return np.sum(ms*self.weights)
        
    def hartley(self):
        pass


class DiscreteFuzzyOWA(DiscreteFuzzySet):
    '''
    Support class to implement the OWA operation between DiscreteFuzzySet instances.
    The OWA operator is actually implemented in the constructor, that build a new DiscreteFuzzySet
    by computing the appropriate values of the membership degrees. All other methods directly rely on
    the base implementation of DiscreteFuzzySet
    Notice: the result of OWA on DiscreteFuzzySet instances is a DiscreteFuzzySet
    '''

    def __init__(self, fuzzysets, weights, dynamic=True):
        if not isinstance(fuzzysets, list) and not isinstance(fuzzysets, np.ndarray) and not isinstance(fuzzysets, tuple):
            raise TypeError("fuzzysets should be a sequence")
        
        if not isinstance(weights, list) and not isinstance(weights, np.ndarray) and not isinstance(weights, tuple):
            raise TypeError("weights should be a sequence")
        
        if len(fuzzysets) != len(weights):
            raise ValueError("fuzzysets and weights should have the same length")
        
        for i in range(len(fuzzysets)):
            if not np.issubdtype(type(weights[i]), np.number) or weights[i] < 0 or weights[i] > 1:
                raise TypeError("w should be a sequence of floats in [0,1]")
            if not isinstance(fuzzysets[i], DiscreteFuzzySet):
                raise TypeError("Arguments should be all discrete fuzzy sets")
            
        if type(dynamic) != bool:
            raise TypeError("dynamic should be bool")
        
        self.dynamic = dynamic
            
        self.fuzzysets = np.array(fuzzysets)
        self.weights = np.array(weights)

        self.items = set()
        for f in self.fuzzysets:
            self.items = self.items.union(set(f.items))
        self.items = np.array(list(self.items))
        self.set = dict(zip(list(self.items), range(len(self.items))))
            
        self.memberships = np.zeros(self.items.shape)

        take = lambda arr, i: arr[i] if 0 <= i < len(arr) else 0
        for e in self.set.keys():
            ms = np.sort([take(f.memberships, f.set.get(e, -1)) for f in self.fuzzysets])
            self.memberships[self.set[e]] = np.sum(ms*self.weights)


class ContinuousFuzzyCombination(ContinuousFuzzySet, FuzzyCombination):
    '''
    Implements a binary operator on ContinuousFuzzySet instances
    '''
    def __init__(self, left: ContinuousFuzzySet, right: ContinuousFuzzySet, op=None):
        if not isinstance(left, ContinuousFuzzySet) or not isinstance(right, ContinuousFuzzySet):
            raise TypeError("All arguments should be continuous fuzzy sets")
            
        self.left = left
        self.right = right
        self.op = op
        self.min = np.min([self.left.min, self.right.min])
        self.max = np.max([self.left.max, self.right.max])

    def __call__(self, arg):
        if (isinstance(arg, tuple) or isinstance(arg, np.ndarray) or isinstance(arg, list)):
            if len(arg) > 2:
                return self.op(self.left(arg[0]), self.right(arg[1:]))
            else:
                return self.op(self.left(arg[0]), self.right(arg[1]))
        else:
            return self.op(self.left(arg), self.right(arg))

    def hartley(self):
        pass

class DiscreteFuzzyCombination(DiscreteFuzzySet, FuzzyCombination):
    '''
    Implements a binary operator on DiscreteFuzzySet instances
    '''
    def __init__(self, left: DiscreteFuzzySet, right: DiscreteFuzzySet, op=None, dynamic=True):
        if not isinstance(left, DiscreteFuzzySet) or not isinstance(right, DiscreteFuzzySet):
            raise TypeError("All arguments should be discrete fuzzy sets")
        
        if type(dynamic) != bool:
            raise TypeError("dynamic should be bool")
        
        self.dynamic = dynamic
            
        self.left = left
        self.right = right
        self.op = op
        self.min = None
        self.max = None

        self.items = np.array(list(set(list(left.items) + list(right.items))))
        self.set = dict(zip(list(self.items), range(len(self.items))))
            
        self.memberships = np.zeros(self.items.shape)

        take = lambda arr, i: arr[i] if 0 <= i < len(arr) else 0
        for e in self.set.keys():
            self.memberships[self.set[e]] = self.op(take(left.memberships, self.left.set.get(e, -1)),
                                                    take(right.memberships, self.right.set.get(e, -1)))
            
    def __getitem__(self, alpha):
        return DiscreteFuzzySet.__getitem__(self, alpha)
    
    def __call__(self, arg):
        return DiscreteFuzzySet.__call__(self, arg)
    
    def fuzziness(self):
        return DiscreteFuzzySet.fuzziness(self)
    
    def hartley(self):
        return DiscreteFuzzySet.hartley(self)
    

class ContinuousFuzzyNegation(ContinuousFuzzySet):
    '''
    Implements a unary operator on ContinuousFuzzySet instances
    '''
    def __init__(self, fuzzy: ContinuousFuzzySet, op=None):
        if not isinstance(fuzzy, ContinuousFuzzySet):
            raise TypeError("Argument should be continuous fuzzy set")
            
        self.fuzzy = fuzzy
        self.op = op
        self.min = fuzzy.min
        self.max = fuzzy.max

    def __call__(self, arg):
        return self.op(self.f(arg))

    def fuzziness(self):
        return self.fuzzy.fuzziness()

    def hartley(self):
        pass
        


def negation(a: FuzzySet):
    if isinstance(a, DiscreteFuzzySet):
        return DiscreteFuzzySet(a.items, [1 - m for m in a.memberships])
    elif isinstance(a, ContinuousFuzzySet):
        return ContinuousFuzzyNegation(a, op=lambda x: 1 -x)
    else:
        raise TypeError("a should be either a discrete or continuous fuzzy set")

def minimum(a: FuzzySet, b: FuzzySet):
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=np.minimum)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=np.minimum)
    else:
        return FuzzyCombination(a,b, op=np.minimum)

def maximum(a: FuzzySet, b: FuzzySet):
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=np.maximum)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=np.maximum)
    else:
        return FuzzyCombination(a,b, op=maximum)
    
def product(a: FuzzySet, b: FuzzySet):
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=np.multiply)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=np.multiply)
    else:
        return FuzzyCombination(a,b, op=np.multiply)
    
def probsum(a: FuzzySet, b: FuzzySet):
    op = lambda x, y: 1 - (1-x)*(1-y)
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        return FuzzyCombination(a,b, op=op)
    
def lukasiewicz(a: FuzzySet, b: FuzzySet):
    op = lambda x, y: np.max([0, x + y - 1])
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        return FuzzyCombination(a,b, op=op)
    
def boundedsum(a: FuzzySet, b: FuzzySet):
    op = lambda x, y: np.min([x + y, 1])
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        return FuzzyCombination(a,b, op=op)
    
def drasticproduct(a: FuzzySet, b: FuzzySet):
    op = lambda x, y: 1 if (x == 1 or y == 1) else 0
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        return FuzzyCombination(a,b, op=op)
    
def drasticsum(a: FuzzySet, b: FuzzySet):
    op = lambda x, y: x if y == 0 else y if x == 0 else 1
    if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        return FuzzyCombination(a,b, op=op)

    
def weighted_average(arr: Sequence[FuzzySet], w : Sequence[np.number]):
    '''
    This implementation heavily relies on the fact that the weighted average is an associative operator. Indeed,
    w_1*v_1 + w_2*v_2 + ... + w_n*v_n = w_1*v1 + 1*(w_2*v2 + 1*(... + w_n*v_n))
    Could be implemented in a simpler way by simply allowing FuzzyCombination to take an array (rather than a pair) of
    argument
    '''

    if not isinstance(arr, list) and not isinstance(arr, np.ndarray) and not isinstance(arr, tuple):
        raise TypeError("arr should be a sequence")
    
    if not isinstance(w, list) and not isinstance(w, np.ndarray) and not isinstance(w, tuple):
        raise TypeError("w should be a sequence")
    
    if len(arr) != len(w):
        raise ValueError("arr and w should have the same length")
    
    t = DiscreteFuzzySet if isinstance(arr[0], DiscreteFuzzySet) else ContinuousFuzzySet if isinstance(arr[0], ContinuousFuzzySet) else FuzzySet
    
    for i in range(len(arr)):
        if not np.issubdtype(type(w[i]), np.number) or w[i] < 0 or w[i] > 1:
            raise TypeError("w should be a sequence of floats in [0,1]")
        
        if not isinstance(arr[i], t):
            t = FuzzySet

    w = np.array(w)
    w /= np.sum(w)
    arr = np.array(arr)

    last = len(arr) - 1
    curr = arr[-1]
    weight = w[-1]
    for i in range(1, len(arr)):
        op = lambda x, y: w[last - i]*x + weight*y
        if t == DiscreteFuzzySet:
            curr = DiscreteFuzzyCombination(arr[last - i], curr, op=op)
        elif t == ContinuousFuzzySet:
            curr = ContinuousFuzzyCombination(arr[last - i], curr, op=op)
        else:
            curr = FuzzyCombination(arr[last - i], curr, op=op)
        weight = 1
    return curr


def owa(arr, w):
    t = DiscreteFuzzySet if isinstance(arr[0], DiscreteFuzzySet) else ContinuousFuzzySet if isinstance(arr[0], ContinuousFuzzySet) else FuzzySet

    w = np.array(w)
    w /= np.sum(w)
    arr = np.array(arr)

    
    curr = DiscreteFuzzyOWA(arr, w) if t==DiscreteFuzzySet else ContinuousFuzzyOWA(arr, w)
    return curr




 