import numpy as np
from typing import Callable 

class DiscreteMembershipFunction:
    def __init__(self,  
                 items: list[object] | np.ndarray,
                 memberships: list[np.number] | np.ndarray,
                 dynamic: bool = True) -> None:
        
        self.__items: np.ndarray = np.ndarray(shape=(0,))
        self.__memberships: np.ndarray = np.ndarray(shape=(0,))
        self.__dynamic: bool = False
        self.__set: dict = dict()

        if not isinstance(items, list) and not isinstance(items, np.ndarray):
            raise TypeError("items should be list or numpy.array")

        if not isinstance(memberships, list) and not isinstance(memberships, np.ndarray):
            raise TypeError("memberships should be list or numpy.array")

        if not isinstance(dynamic, bool):
            raise TypeError("dynamic should be bool")
        
        if len(self.__items) != len(self.__memberships):
            raise ValueError(f"Membership degrees and items have different dimension, respectevly: {len(self.__items)} and {len(self.__memberships)}")
        
        for m in memberships:
            if not np.issubdtype(type(m), np.number):
                raise f"Membership degrees should be floats in [0,1], is {type(m)}" 
            if  m < 0 or m > 1:
                raise ValueError(f"Membership degrees should be floats in [0,1], is {m}")
        
        self.__items = np.array(items)
        self.__set = dict(zip(items, range(len(items))))
        self.__memberships = np.array(memberships)
        self.__dynamic = dynamic

    def __call__(self, arg):
        '''
        Gets the membership degree of arg using () sintax. Uses self.set to enable quick look-up.
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
        Gets an alpha cut as [] sintax
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
        if not isinstance(other, DiscreteMembershipFunction):
            return False
        
        return  (self.__dynamic == other.__dynamic and
                 self.__memberships == other.__memberships and
                 self.__items == other.__items and 
                 self.__set == other.__set)
    
class ContinueMembershipFunction:
    def __init__(self,  
                 memberships_func: Callable[[np.number], np.number],
                 bound: tuple[np.number, np.number],
                 epsilon: np.number = 1e-3) -> None:
        
        self.__memberships_func: Callable[[np.number], 
                                          np.number] = (lambda x: 1 if x == 0 else 0) 
        self.__epsilon: np.number = 0
        self.__bound: tuple[np.number, np.number] = (0, 0) 

        if not isinstance(memberships_func, Callable[[np.number], np.number]):
            raise TypeError("memberships_func should be Callable[[np.number], np.number]")
        
        if not isinstance(bound, tuple[np.number, np.number]) and bound[0] > bound[1]:
            raise TypeError(f"bound should be a tuple[np.number, np.number] expressing bounds where support set is defined (outer included)")
        
        if not isinstance(epsilon, np.number):
            raise TypeError("epsilon should be a small float value, ex: 1e-3")
        
        discr_memb_func = np.linspace(bound[0], 
                                      bound[1] + 1, 
                                      int((bound[1] - bound[0] + 1)/epsilon))
        
        discr_memb_func = np.array([0 if 0 <= memberships_func(x) <= 1 else 1 
                                    for x in  discr_memb_func])

        if sum(discr_memb_func) != 0:
            raise ValueError(f"Membership degrees should be floats in [0,1]") 
        
        self.__memberships_func = memberships_func
        self.__epsilon = epsilon
        self.__bound = bound

    def __call__(self, arg):
        '''
        Gets the membership degree of arg using () sintax. Uses self.set to enable quick look-up.
        Behavior changes according to value of dynamic
        '''
        if not (self.__bound[0] <= arg <= self.__bound[1]):
            return 0
        
        return self.__memberships_func(arg)
    
    def __get_alpha_cut_interval(self, alpha: np.number) -> np.ndarray:
        alpha_cut_interval: list[np.number] = []
        step = int((self.__bound[1] - self.__bound[0] + 1)/self.__epsilon)

        x_values = np.linspace(self.__bound[0], 
                               self.__bound[1] + 1, 
                               step)
        
        discr_memb_func = np.array([self.__memberships_func(x) for x in  x_values])
        
        if len(x_values) == 1:
            if discr_memb_func[0] >= alpha:
                return np.ndarray([x_values[0]])
            return np.ndarray([])

        tmp_bound: tuple[np.number, np.number] = (x_values[0], x_values[0])
        for i in range(1, len(x_values)):
            if discr_memb_func[i] >= alpha:
                if x_values[i] == x_values[i-1] + step and discr_memb_func[i-1] >= alpha:
                    tmp_bound[1] = i
                else:
                    alpha_cut_interval.append(tmp_bound)
                    tmp_bound[0] = i

        return np.ndarray(alpha_cut_interval)
    
    def __getitem__(self, alpha: np.number) -> np.ndarray:
        '''
        Gets an alpha cut as [] sintax
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
        if not isinstance(other, ContinueMembershipFunction):
            return False
        
        return  (self.__epsilon == other.__epsilon and
                 self.__memberships_func == other.__memberships_func and
                 self.__bound == other.__bound)