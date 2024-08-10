from __future__ import annotations
from collections.abc import Sequence
from typing import Callable
import numpy as np
from softpy.fuzzy.fuzzyset import DiscreteFuzzySet, ContinuousFuzzySet, FuzzySet
from softpy.fuzzy.operations import negation

class ContinuousFuzzyOWA(ContinuousFuzzySet):
    '''
    Support class to implement the OWA operation between ContinuousFuzzySet instances.
    Notice: the result of OWA on ContinuousFuzzySet instances is a ContinuousFuzzySet
    '''

    def __init__(self, 
                 fuzzysets: list[ContinuousFuzzySet] | np.ndarray | tuple[np.number], 
                 weights: list[np.number] | np.ndarray | tuple[np.number]):
        '''
        The constructor searches the given fuzzysets in order to determine the minimum and maximum of the result of OWA
        '''
        if (not isinstance(fuzzysets, list) and 
            not isinstance(fuzzysets, np.ndarray) and 
            not isinstance(fuzzysets, tuple)):
            raise TypeError("arr should be a sequence")
        
        if (not isinstance(weights, list) and 
            not isinstance(weights, np.ndarray) and 
            not isinstance(weights, tuple)):
            raise TypeError("w should be a sequence")
        
        if len(fuzzysets) != len(weights) or len(fuzzysets) == 0:
            raise ValueError("arr and w should have the same length and greater than 0")
        
        if sum(weights) != 1:
            raise ValueError("sum of weights should be 1")

        self.__bound = [np.inf, -np.inf]
        
        for i in range(len(fuzzysets)):
            if not np.issubdtype(type(weights[i]), np.number) or weights[i] < 0 or weights[i] > 1:
                raise TypeError("w should be a sequence of floats in [0,1]")
            if not isinstance(fuzzysets[i], ContinuousFuzzySet):
                raise TypeError("Arguments should be all continuous fuzzy sets")
            if fuzzysets[i].bound[0] < self.__bound[0]:
                self.__bound[0] = fuzzysets[i].bound[0]
            if fuzzysets[i].bound[1] > self.__bound[1]:
                self.__bound[1] = fuzzysets[i].bound[1]
            
        self.__fuzzysets = np.array(fuzzysets)
        self.__weights = np.array(weights)

    def __call__(self, arg: np.number | list[np.number] | tuple[np.number] | np.ndarray):
        '''
        Computes the membership degree of arg by evaluating the OWA with the given weights
        '''
        if (not isinstance(arg, list) and 
            not isinstance(arg, tuple) and 
            not isinstance(arg, np.ndarray) and 
            not np.issubdtype(type(arg), np.number)):
            raise TypeError("arg should be a list | tuple | ndarray | number")
        
        if ((isinstance(arg, list) or 
             isinstance(arg, tuple) or 
             isinstance(arg, np.ndarray)) and len(arg) != len(self.__fuzzysets)):
            raise ValueError("arg should have equal length with fuzzysets")

        if np.issubdtype(type(arg), np.number):
            ms = np.sort([self.__fuzzysets[i](arg) for i in range(len(self.__fuzzysets))])
            return np.sum(ms*self.__weights)

        ms = np.sort([self.__fuzzysets[i](arg[i]) for i in range(len(self.__fuzzysets))])
        return np.sum(ms*self.__weights)  


class DiscreteFuzzyOWA(DiscreteFuzzySet):
    '''
    Support class to implement the OWA operation between DiscreteFuzzySet instances.
    The OWA operator is actually implemented in the constructor, that builds a new DiscreteFuzzySet
    by computing the appropriate values of the membership degrees. All other methods directly rely on
    the base implementation of DiscreteFuzzySet.
    Notice: the result of OWA on DiscreteFuzzySet instances is a DiscreteFuzzySet
    '''

    def __init__(self, 
                 fuzzysets: list[DiscreteFuzzySet] | np.ndarray | tuple[np.number], 
                 weights: list[np.number] | np.ndarray | tuple[np.number], 
                 dynamic: bool = True):
        if (not isinstance(fuzzysets, list) and 
            not isinstance(fuzzysets, np.ndarray) and 
            not isinstance(fuzzysets, tuple)):
            raise TypeError("fuzzysets should be a sequence")
        
        if (not isinstance(weights, list) and 
            not isinstance(weights, np.ndarray) and 
            not isinstance(weights, tuple)):
            raise TypeError("weights should be a sequence")
        
        if len(fuzzysets) != len(weights) or len(fuzzysets) == 0:
            raise ValueError("fuzzysets and weights should have the same length greater than 0")
        
        if sum(weights) != 1:
            raise ValueError("sum of weights should be 1")
        
        for i in range(len(fuzzysets)):
            if not np.issubdtype(type(weights[i]), np.number) or weights[i] < 0 or weights[i] > 1:
                raise TypeError("w should be a sequence of floats in [0,1]")
            if not isinstance(fuzzysets[i], DiscreteFuzzySet):
                raise TypeError("Arguments should be all discrete fuzzy sets")
            
        self.__fuzzysets: list[DiscreteFuzzySet] = np.array(fuzzysets)
        self.__weights: list[np.number] = np.array(weights)

        items = set()
        for f  in self.__fuzzysets:
            items = set.union(items, f.items)
        
        items = np.array(list(items))
        set_items = dict(zip(list(items), range(len(items))))
            
        memberships = np.zeros(len(items))

        take = lambda arr, i: arr[i] if 0 <= i < len(arr) else 0
        for e in set_items.keys():
            ms = np.sort([take(f.memberships, f.set_items.get(e, -1)) for f in self.__fuzzysets])
            memberships[set_items[e]] = np.sum(ms*self.__weights)
        
        super().__init__(items, memberships, dynamic)


class ContinuousFuzzyCombination(ContinuousFuzzySet):
    '''
    Implements a binary operator on ContinuousFuzzySet instances
    '''
    def __init__(self, 
                 fuzzy_sets: list[ContinuousFuzzySet], 
                 op: Callable):
        if not isinstance(fuzzy_sets, list):
            raise TypeError("fuzzy_sets should be a list")   
    
        if len(fuzzy_sets) == 0:
            raise TypeError("fuzzy_sets list should be not empty")
        
        if not isinstance(op, Callable):
            raise TypeError("op should be a callable")
        
        for f in fuzzy_sets:
            if not isinstance(f, ContinuousFuzzySet):
                raise TypeError("all fuzzy_sets should be ContinuousFuzzySet")    
            
        bound = list(fuzzy_sets[0].bound)
        for i in range(len(fuzzy_sets)):
            if bound[0] > fuzzy_sets[i].bound[0]:
                bound[0] = fuzzy_sets[i].bound[0]
            if bound[1] < fuzzy_sets[i].bound[1]:
                bound[1] = fuzzy_sets[i].bound[1]

        self.__fuzzy_sets: list[ContinuousFuzzySet] = fuzzy_sets
        self.__op: Callable = op


        super().__init__(self.__memberships_function, tuple(bound))

    def __memberships_function(self, x: list[np.number] | np.number):
        ris = 0
        if isinstance(x, list):
            ris = self.__fuzzy_sets[0](x[0]) if self.bound[0] <= x[0] <= self.bound[1] else 0
        else:
            ris = self.__fuzzy_sets[0](x) if self.bound[0] <= x <= self.bound[1] else 0

        for i in range(1, len(self.__fuzzy_sets)):
            if isinstance(x, list):
                memb_value = self.__fuzzy_sets[i](x[i]) if self.bound[0] <= x[i] <= self.bound[1] else 0
            else:
                memb_value = self.__fuzzy_sets[i](x) if self.bound[0] <= x <= self.bound[1] else 0
            ris = self.__op(ris, memb_value)
        return ris
    
    def __call__(self, arg: np.number | list):
        if not isinstance(arg, list) and not np.issubdtype(type(arg), np.number):
            raise TypeError("arg should be a list or a number")   

        return self.__memberships_function(arg)
        

class DiscreteFuzzyCombination(DiscreteFuzzySet):
    '''
    Implements a binary operator on DiscreteFuzzySet instances
    '''
    def __init__(self, 
                 fuzzy_sets: list[DiscreteFuzzySet], 
                 op: Callable,
                 dynamic: bool = True):
        if not isinstance(fuzzy_sets, list):
            raise TypeError("fuzzy_sets should be a list")   
    
        if len(fuzzy_sets) == 0:
            raise TypeError("fuzzy_sets list should be not empty")
        
        if not isinstance(op, Callable):
            raise TypeError("op should be a callable")
        
        items = []
        for f in fuzzy_sets:
            if not isinstance(f, DiscreteFuzzySet):
                raise TypeError("all fuzzy_sets should be DiscreteFuzzySet")
            items = items + list(f.items)
            
        items = np.array(list(set(items)))
        set_items = dict(zip(list(items), range(len(items))))
        memberships = np.zeros(len(items))

        take = lambda arr, i: arr[i] if 0 <= i < len(arr) else 0
        for e in set_items.keys():
            memberships[set_items[e]] = take(fuzzy_sets[0].memberships, fuzzy_sets[0].set_items.get(e, -1))
            for i in range(1, len(fuzzy_sets)):
                memberships[set_items[e]] = op(memberships[set_items[e]],
                                               take(fuzzy_sets[i].memberships, 
                                                    fuzzy_sets[i].set_items.get(e, -1)))
            
        super().__init__(items, memberships, dynamic)

class ContinuousFuzzyNegation(ContinuousFuzzySet):
    '''
    Implements a unary operator on ContinuousFuzzySet instances
    '''
    def __init__(self, fuzzy: ContinuousFuzzySet):
        if not isinstance(fuzzy, ContinuousFuzzySet):
            raise TypeError("Argument should be continuous fuzzy set")
            
        self.__fuzzy = fuzzy
        self.__op = negation
        super().__init__(lambda x: self.__op(self.__fuzzy(x)), 
                         (-np.inf, np.inf))

class DiscreteFuzzyNegation(DiscreteFuzzySet):
    '''
    Implements a unary operator on ContinuousFuzzySet instances
    '''
    def __init__(self, fuzzy: DiscreteFuzzySet):
        if not isinstance(fuzzy, DiscreteFuzzySet):
            raise TypeError("Argument should be discrete fuzzy set")
            
        self.__fuzzy = fuzzy
        self.__op = negation
        super().__init__(fuzzy.items,
                         [self.__op(m) for m in fuzzy.memberships], 
                         fuzzy.dynamic)
        

def weighted_average(arr: Sequence[FuzzySet], w : Sequence[np.number]):
    '''
    This implementation heavily relies on the fact that the weighted average is an associative operator. Indeed,
    w_1*v_1 + w_2*v_2 + ... + w_n*v_n = w_1*v1 + 1*(w_2*v2 + 1*(... + w_n*v_n))
    Could be implemented in a simpler way by simply allowing FuzzyCombination to take an array (rather than a pair) of
    arguments
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
        weight = 1
    return curr


def owa(arr, w):
    t = DiscreteFuzzySet if isinstance(arr[0], DiscreteFuzzySet) else ContinuousFuzzySet if isinstance(arr[0], ContinuousFuzzySet) else FuzzySet

    w = np.array(w)
    w /= np.sum(w)
    arr = np.array(arr)

    
    curr = DiscreteFuzzyOWA(arr, w) if t==DiscreteFuzzySet else ContinuousFuzzyOWA(arr, w)
    return curr




 