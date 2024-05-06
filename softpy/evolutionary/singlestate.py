from abc import ABC, abstractmethod
import numpy as np

class MetaHeuristicsAlgorithm(ABC):
    '''
    An abstract class for implementing meta-heuristic optimization algorithms
    '''
    @abstractmethod
    def fit(self, n_iters=10, keep_history=False):
        pass

class RandomSearch(MetaHeuristicsAlgorithm):
    '''
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.
    '''
    def __init__(self, candidate_type, fitness_func, pop_size: int = 1, **kwargs):
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.kwargs = kwargs

    def fit(self, n_iters=10, keep_history=False):
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters)
            self.fitness_h[:] = np.NINF


        for it in range(n_iters):
            population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
            fitness = np.vectorize(self.fitness_func)(population)
            v = np.max(fitness)
            self.best = population[np.argmax(fitness)] if v > self.fitness_best else self.best
            self.fitness_best = v if v > self.fitness_best else self.fitness_best

            if keep_history:
                self.best_h[it] = self.best
                self.fitness_h[it] = self.fitness_best

        return self.best
    
class HillClimbing(MetaHeuristicsAlgorithm):
    '''
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.
    '''
    def __init__(self, candidate_type, fitness_func, test_size: int = 1, **kwargs):
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.test_size = test_size
        self.kwargs = kwargs

    def fit(self, n_iters=10, keep_history=False):
        self.best = self.candidate_type.generate(**self.kwargs)
        self.fitness_best = self.fitness_func(self.best)
        self.current = self.best

        if keep_history:
            self.best_h = np.empty(n_iters, dtype=self.candidate_type)
            self.best_h[0] = self.best
            self.fitness_h = np.zeros(n_iters)
            self.fitness_h[:] = np.NINF
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

        return self.best
    