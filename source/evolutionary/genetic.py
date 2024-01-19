from __future__ import annotations
import numpy as np
from evolutionary.singlestate import MetaHeuristicsAlgorithm

class GeneticAlgorithm(MetaHeuristicsAlgorithm):

    def __init__(self, pop_size: int, candidate_type, selection_func, fitness_func, elitism=False, n_elite=1, **kwargs):
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.elitism = elitism
        self.n_elite = n_elite

    def fit(self, n_iters=10, keep_history=False, show_iters=False):
        if self.elitism and (self.pop_size + self.n_elite)%2 != 0:
            self.pop_size += 1

        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = np.zeros(self.population.shape)
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = np.NINF

        for it in range(n_iters+1):
            if show_iters:
                print(it)
            self.fitness = np.vectorize(self.fitness_func)(self.population)
            v = np.max(self.fitness)
            self.best = self.population[np.argmax(self.fitness)] if v > self.fitness_best else self.best
            self.fitness_best = v if v > self.fitness_best else self.fitness_best

            if keep_history:
                self.best_h[it] = self.best
                self.fitness_h[it] = self.fitness_best


            q = np.empty(self.pop_size, dtype=self.candidate_type)
            if self.elitism:
                q[:self.n_elite] = self.population[np.argsort(self.fitness)[::-1][:self.n_elite]]
            
            i = self.n_elite if self.elitism else 0
            sub = self.n_elite if self.elitism else 0
            current_step = None

            idx = np.random.permutation(range(len(self.fitness)))
            self.population = self.population[idx]
            self.fitness = self.fitness[idx]
            for s in range(int((self.pop_size - sub)/2)):
                px1, current_step = self.selection_func(self.fitness, current_step=current_step)
                p1 = self.population[px1]
                px2, current_step = self.selection_func(self.fitness, current_step=current_step)
                p2 = self.population[px2]
                q[i] = p1.recombine(p2).mutate()
                q[i+1] = p2.recombine(p1).mutate()
                i += 2
            self.population = q

        return self.best
    

class SteadyStateGeneticAlgorithm(MetaHeuristicsAlgorithm):
    def __init__(self, pop_size: int, candidate_type, selection_func, fitness_func, **kwargs):
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs

    def fit(self, n_iters=10, keep_history=False):
        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = np.zeros(self.population.shape)
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = np.NINF

        self.fitness = np.vectorize(self.fitness_func)(self.population)
        v = np.max(self.fitness)
        self.best = self.population[np.argmax(self.fitness)] if v > self.fitness_best else self.best
        self.fitness_best = v if v > self.fitness_best else self.fitness_best

        if keep_history:
            self.best_h[0] = self.best
            self.fitness_h[0] = self.fitness_best

        for it in range(1, n_iters+1):
            p1, current_step = self.population[self.selection_func(self.fitness, current_step=current_step)]
            p2, current_step = self.population[self.selection_func(self.fitness, current_step=current_step)]
            c1 = p1.recombine(p2).mutate()
            c2 = p2.recombine(p1).mutate()

            f1 = self.fitness_func(c1)
            f2 = self.fitness_func(c2)
            if f1 > self.fitness_best:
                self.best = c1
                self.fitness_best = f1
            if f2 > self.fitness_best:
                self.best = c2
                self.fitness_best = f2

            if keep_history:
                self.best_h[it] = self.best
                self.fitness_h[it] = self.fitness_best 

            die = np.random.choice(range(self.fitness.shape[0]), 2, replace=False)
            self.population[die[0]] = c1
            self.population[die[1]] = c2

        return self.best
    



