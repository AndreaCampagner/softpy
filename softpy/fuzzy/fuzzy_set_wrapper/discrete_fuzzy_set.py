from typing import List
import numpy as np

from softpy.fuzzy.fuzzy_set_wrapper.fuzzy_set import FuzzySet



class DiscreteFuzzySet(FuzzySet):
    '''
    Implements a discrete fuzzy set
    
    '''


    def __init__(self, 
                 items: list | np.ndarray,
                 memberships: list | np.ndarray,
                 dynamic: bool = True):
        '''
        Requires as input a sequence (list or array) or objects and a sequence 
        (list or array) of membership degrees

        Attribute dynamic controls whether the support set is exhaustive or not 
        (i.e. there exist objects not in items whose membership degree is 0)

        Internally the constructor uses a dictionary (self.set) to enable fast 
        look-up of membership degrees
        '''
        
        self.__items: np.ndarray = np.ndarray()
        self.__memberships: np.ndarray = np.ndarray()
        self.__set: dict = dict()
        self.__dynamic: bool = False
        self.__f: np.number = -1
        self.__h: np.number = -1

        if not isinstance(items, list) and not isinstance(items, np.ndarray):
            raise TypeError("items should be list or numpy.array")

        if not isinstance(memberships, list) and not isinstance(memberships, np.ndarray):
            raise TypeError("memberships should be list or numpy.array")

        if not isinstance(dynamic, bool):
            raise TypeError("dynamic should be bool")

        self.__items = np.array(items)
        self.__set = dict(zip(items, range(len(items))))

        for m in memberships:
            if not np.issubdtype(type(m), np.number):
                raise f"Membership degrees should be floats in [0,1], is {type(m)}" 
            if  m < 0 or m > 1:
                raise ValueError(f"Membership degrees should be floats in [0,1], is {m}")

        self.__memberships = np.array(memberships)
        self.__dynamic = dynamic

    def __call__(self, arg):
        '''
        Gets the membership degree of arg. Uses self.set to enable quick look-up.
        Behavior changes according to value of dynamic
        '''
        if arg not in self.__set.keys():
            if self.__dynamic:
                self.__set[arg] = len(self.__items)
                self.__items = np.append(self.__items, arg)
                self.__memberships = np.append(self.__memberships, 0.0)
            else:
                raise ValueError(f"{arg} not in the support of the fuzzy set")

        return self.__memberships[self.__set[arg]]

    def __getitem__(self, alpha: np.number) -> np.ndarray:
        '''
        Gets an alpha cut as [] semantic
        '''
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError(f"Alpha should be a float in [0,1], is {type(alpha)}")

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha should be in [0,1], is {alpha}")

        return self.__items[self.__memberships >= alpha]

    def __eq__(self, other: object) -> bool:
        '''
        Checks whether two DiscreteFuzzySet instances are equal
        '''
        if not isinstance(other, DiscreteFuzzySet):
            return NotImplemented

        for v in list(self.__set.keys()) + list(other.set.keys()):
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
        if self.__f == -1:
            pos = self.__memberships[self.__memberships > 0]
            pos = pos*np.log2(1/pos)
            non = self.__memberships[self.__memberships < 1]
            non = (1-non)*np.log2(1/(1-non))
            self.__f = np.sum(pos) + np.sum(non)
        return self.__f

    def __hartley_entropy(self) -> np.number:
        '''
        Computes the hartley entropy (non-specificity)
        '''
        pos = self.__memberships[self.__memberships > 0]
        sort = np.append(np.sort(pos)[::-1], 0)
        coeffs = sort[:-1] - sort[1:]
        sizes = np.log2(np.array([len(self[i]) for i in sort[:-1]]))
        H = np.sum(coeffs*sizes)
        return H
    
    def entropy(self) -> np.number:
        if self.__h == -1:
            self.__h = self.__hartley_entropy()
        return self.__h