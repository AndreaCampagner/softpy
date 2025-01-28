from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Type
from .candidate import Candidate
from .utils import Comparable

class MetaHeuristicsAlgorithm(ABC):
    '''
    An abstract class for implementing meta-heuristic optimization algorithms.
    All concrete sub-classes must provide access to a best and fitness_best objects
    '''
    best: Candidate
    fitness_best: Comparable

    @abstractmethod
    def __init__(self, candidate_type:Type[Candidate], fitness_func:Callable):
        pass
    
    @abstractmethod
    def fit(self, n_iters=10, keep_history=False):
        '''
        Implements the optimization routine
        '''
        pass

class RandomSearch(MetaHeuristicsAlgorithm):
    '''
    An implementation of single-state, as well as multi-state, random search.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.

    Parameters
    ----------
    :param candidate_type: the class specifying the representation of candidate solutions
    :type candidate_type: Type[Candidate]

    :param fitness_func: the fitness function
    :type fitness_func: Callable

    :param pop_size: the size of the population
    :type pop_size: int, default=1
    '''
    def __init__(self, candidate_type: Type[Candidate], fitness_func: Callable, pop_size: int = 1, **kwargs):
        if pop_size < 1:
            raise ValueError("pop_size must be an integer greater than 0, was %d" % pop_size)
        
        if not isinstance(fitness_func,Callable):
            raise ValueError("fitness_func must be of type Callable, was %s" % type(fitness_func))
        
        if not issubclass(candidate_type, Candidate):
            raise ValueError("candidate_type must be a subclass of Candidate")
        
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.kwargs = kwargs
        self.is_fitted = False


    def fit(self, n_iters:int=10, keep_history:bool=False):
        if n_iters < 1:
            raise ValueError("n_iters must be larger than 0, was %d" % n_iters)
        self.is_fitted = False
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters, dtype=self.candidate_type)
            self.fitness_h = [np.NINF]*n_iters


        for it in range(n_iters):
            population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
            fitness = np.vectorize(self.fitness_func)(population)
            v = np.max(fitness)
            self.best = population[np.argmax(fitness)] if v > self.fitness_best else self.best
            self.fitness_best = v if v > self.fitness_best else self.fitness_best

            if keep_history:
                self.best_h[it] = self.best
                self.fitness_h[it] = self.fitness_best

        self.is_fitted = True
        return self.best
    
class HillClimbing(MetaHeuristicsAlgorithm):
    '''
    An implementation of single-state hill climbing.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.

    Parameters
    ----------
    :param candidate_type: the class specifying the representation of candidate solutions
    :type candidate_type: Type[Candidate]

    :param fitness_func: the fitness function
    :type fitness_func: Callable

    :param test_size: the number of test candidates to evaluate at each iteration
    :type test_size: int, default=1
    '''
    def __init__(self, candidate_type: Type[Candidate], fitness_func: Callable, test_size: int = 1, **kwargs):
        if test_size < 1:
            raise ValueError("test_size must be an integer greater than 0, was %d" % test_size)
        
        if not isinstance(fitness_func,Callable):
            raise ValueError("fitness_func must be of type Callable, was %s" % type(fitness_func))
        
        if not issubclass(candidate_type, Candidate):
            raise ValueError("candidate_type must be a subclass of Candidate")
        
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.test_size = test_size
        self.kwargs = kwargs
        self.is_fitted = False


    def fit(self, n_iters:int=10, keep_history:bool=False):
        if n_iters < 1:
            raise ValueError("n_iters must be larger than 0, was %d" % n_iters)
        
        self.is_fitted = False
        self.best = self.candidate_type.generate(**self.kwargs)
        self.fitness_best = self.fitness_func(self.best)
        self.current = self.best

        if keep_history:
            self.best_h = np.empty(n_iters, dtype=self.candidate_type)
            self.best_h[0] = self.best
            self.fitness_h = [np.NINF]*n_iters
            self.fitness_h[0] = self.fitness_best

        for it in range(n_iters-1):
            population = [self.best.mutate() for t in range(self.test_size)]
            fitness = np.vectorize(self.fitness_func)(population)
            v = np.max(fitness)
            self.current = population[np.argmax(fitness)]
            self.best = self.current if v > self.fitness_best else self.best
            self.fitness_best = v if v > self.fitness_best else self.fitness_best

            if keep_history:
                self.best_h[it] = self.best
                self.fitness_h[it] = self.fitness_best

        self.is_fitted = True
        return self.best
    