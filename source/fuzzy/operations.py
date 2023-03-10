from __future__ import annotations
import numpy as np
from fuzzyset import FuzzySet,  DiscreteFuzzySet, ContinuousFuzzySet
from typing import Type

def negation(a: Type[FuzzySet]):
    if issubclass(type(a), DiscreteFuzzySet):
        return DiscreteFuzzySet(a.items, [1 - m for m in a.memberships])
    else:
        return ContinuousFuzzyNegation(a, op=lambda x: 1 -x)

def minimum(a: Type[FuzzySet], b: Type[FuzzySet]):
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=np.minimum)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=np.minimum)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")

def maximum(a: Type[FuzzySet], b: Type[FuzzySet]):
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=np.maximum)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=np.maximum)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")
    
def product(a: Type[FuzzySet], b: Type[FuzzySet]):
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=np.multiply)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=np.multiply)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")
    
def probsum(a: Type[FuzzySet], b: Type[FuzzySet]):
    op = lambda x, y: 1 - (1-x)*(1-y)
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")
    
def lukasiewicz(a: Type[FuzzySet], b: Type[FuzzySet]):
    op = lambda x, y: np.max[0, x + y - 1]
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")
    
def boundedsum(a: Type[FuzzySet], b: Type[FuzzySet]):
    op = lambda x, y: np.min[x + y, 1]
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")
    
def drasticproduct(a: Type[FuzzySet], b: Type[FuzzySet]):
    op = lambda x, y: 1 if (x == 1 or y == 1) else 0
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")
    
def drasticsum(a: Type[FuzzySet], b: Type[FuzzySet]):
    op = lambda x, y: x if y == 0 else y if x == 0 else 1
    if issubclass(type(a), DiscreteFuzzySet) and issubclass(type(b), DiscreteFuzzySet):
        return DiscreteFuzzyCombination(a, b, op=op)
    elif issubclass(type(a), ContinuousFuzzySet) and issubclass(type(b), ContinuousFuzzySet):
        return ContinuousFuzzyCombination(a,b, op=op)
    else:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")

    
def weighted_average(arr, w):
    if not issubclass(type(arr), list) and not issubclass(type(arr), np.ndarray) and not issubclass(type(arr), tuple):
        raise TypeError("arr should be a sequence")
    
    if not issubclass(type(w), list) and not issubclass(type(w), np.ndarray) and not issubclass(type(w), tuple):
        raise TypeError("w should be a sequence")
    
    if len(arr) != len(w):
        raise ValueError("arr and w should have the same length")
    
    t = DiscreteFuzzySet if issubclass(type(arr[0]), DiscreteFuzzySet) else ContinuousFuzzySet if issubclass(type(arr[0]), ContinuousFuzzySet) else FuzzySet
    if t == FuzzySet:
        raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")
    
    for i in range(len(arr)):
        if not np.issubdtype(type(w[i]), np.number) or w[i] < 0 or w[i] > 1:
            raise TypeError("w should be a sequence of floats in [0,1]")
        
        if not issubclass(type(arr[i]), t):
            raise TypeError("Arguments should be fuzzy sets that are either all discrete or all continuous")

    w = np.array(w)
    w /= np.sum(w)
    arr = np.array(arr)

    last = len(arr) - 1
    curr = arr[-1]
    weight = w[-1]
    for i in range(1, len(arr)):
        op = lambda x, y: w[last - i]*x + weight*y
        curr = DiscreteFuzzyCombination(arr[last - i], curr, op=op) if t==DiscreteFuzzySet else ContinuousFuzzyCombination(arr[last - i], curr, op=op)
        weight = 1
    return curr


def owa(arr, w):
    t = DiscreteFuzzySet if issubclass(type(arr[0]), DiscreteFuzzySet) else ContinuousFuzzySet if issubclass(type(arr[0]), ContinuousFuzzySet) else FuzzySet

    w = np.array(w)
    w /= np.sum(w)
    arr = np.array(arr)

    
    curr = DiscreteFuzzyOWA(arr, w) if t==DiscreteFuzzySet else ContinuousFuzzyOWA(arr, w)
    return curr


class ContinuousFuzzyOWA(ContinuousFuzzySet):
    def __init__(self, fuzzysets, weights):
        if not issubclass(type(fuzzysets), list) and not issubclass(type(fuzzysets), np.ndarray) and not issubclass(type(fuzzysets), tuple):
            raise TypeError("arr should be a sequence")
        
        if not issubclass(type(weights), list) and not issubclass(type(weights), np.ndarray) and not issubclass(type(weights), tuple):
            raise TypeError("w should be a sequence")
        
        if len(fuzzysets) != len(weights):
            raise ValueError("arr and w should have the same length")
        
        self.min = np.infty
        self.max = -np.infty
        
        for i in range(len(fuzzysets)):
            if not np.issubdtype(type(weights[i]), np.number) or weights[i] < 0 or weights[i] > 1:
                raise TypeError("w should be a sequence of floats in [0,1]")
            if not issubclass(type(fuzzysets[i]), ContinuousFuzzySet):
                raise TypeError("Arguments should be all continuous fuzzy sets")
            if fuzzysets[i].min < self.min:
                self.min = fuzzysets[i].min
            if fuzzysets[i].max > self.max:
                self.max = fuzzysets[i].max
            
        self.fuzzysets = np.array(fuzzysets)
        self.weights = np.array(weights)

    def __call__(self, arg):
        if isinstance(arg, tuple):
            ms = np.sort([self.fuzzysets[i](arg[i]) for i in range(len(self.fuzzysets))])
            return np.sum(ms*self.weights)  
        else:
            ms = np.sort([self.fuzzysets[i](arg) for i in range(len(self.fuzzysets))])
            return np.sum(ms*self.weights)
        
    def hartley(self):
        pass


class DiscreteFuzzyOWA(DiscreteFuzzySet):
    def __init__(self, fuzzysets, weights, dynamic=True):
        if not issubclass(type(fuzzysets), list) and not issubclass(type(fuzzysets), np.ndarray) and not issubclass(type(fuzzysets), tuple):
            raise TypeError("fuzzysets should be a sequence")
        
        if not issubclass(type(weights), list) and not issubclass(type(weights), np.ndarray) and not issubclass(type(weights), tuple):
            raise TypeError("weights should be a sequence")
        
        if len(fuzzysets) != len(weights):
            raise ValueError("fuzzysets and weights should have the same length")
        
        for i in range(len(fuzzysets)):
            if not np.issubdtype(type(weights[i]), np.number) or weights[i] < 0 or weights[i] > 1:
                raise TypeError("w should be a sequence of floats in [0,1]")
            if not issubclass(type(fuzzysets[i]), DiscreteFuzzySet):
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


class ContinuousFuzzyCombination(ContinuousFuzzySet):
    def __init__(self, left: Type[ContinuousFuzzySet], right: Type[ContinuousFuzzySet], op=None):
        if not issubclass(type(left), ContinuousFuzzySet) or not issubclass(type(right), ContinuousFuzzySet):
            raise TypeError("All arguments should be continuous fuzzy sets")
            
        self.left = left
        self.right = right
        self.op = op
        self.min = np.min([self.left.min, self.right.min])
        self.max = np.max([self.left.max, self.right.max])

    def __call__(self, arg):
        if isinstance(arg, tuple):
            return self.op(self.left(arg[0]), self.right(arg[1:]))
        else:
            return self.op(self.left(arg), self.right(arg))

    def hartley(self):
        pass

class DiscreteFuzzyCombination(DiscreteFuzzySet):
    def __init__(self, left: Type[DiscreteFuzzySet], right: Type[DiscreteFuzzySet], op=None, dynamic=True):
        if not issubclass(type(left), DiscreteFuzzySet) or not issubclass(type(right), DiscreteFuzzySet):
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
    def __init__(self, f: Type[ContinuousFuzzySet], op=None):
        if not issubclass(type(f), ContinuousFuzzySet):
            raise TypeError("Argument should be continuous fuzzy set")
            
        self.f = f
        self.op = op
        self.min = f.min
        self.max = f.max

    def __call__(self, arg):
        return self.op(self.f(arg))

    def fuzziness(self):
        return self.f.fuzziness()

    def hartley(self):
        pass

 