from __future__ import annotations
import numpy as np

class GeneticAlgorithm():

    def __init__(self, pop_size: int, candidate_type, selection_func, fitness_func, elitism=False, n_elite=1, **kwargs):
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.elitism = elitism
        self.n_elite = n_elite

    def fit(self, n_iters=10, keep_history=False):
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
            for s in range(int((self.pop_size - sub)/2)):
                p1 = self.population[self.selection_func(self.fitness)]
                p2 = self.population[self.selection_func(self.fitness)]
                q[i] = p1.recombine(p2).mutate()
                q[i+1] = p2.recombine(p1).mutate()
                i += 2
            self.population = q

        return self.best




