from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats


class Candidate(ABC):
    @abstractmethod
    def mutate(self) -> Candidate:
        pass

    @abstractmethod
    def recombine(self, c: Candidate) -> Candidate:
        pass


class BitVectorCandidate(Candidate):

    def __init__(self, size: int, candidate: np.ndarray, p: np.number = 0.5, uniform: bool = False):
        self.size = size
        self.p = p
        self.candidate = candidate
        self.uniform = uniform

    def generate(size: int, p: np.number = 0.5, uniform: bool = False) -> BitVectorCandidate:
        candidate = np.random.choice(a=[True,False], size=size, replace=True)
        return BitVectorCandidate(size, candidate, p, uniform)

    def mutate(self) -> BitVectorCandidate:
        candidate = np.array([v if np.random.rand() >= self.p else bool(1-v) for v in self.candidate])
        return BitVectorCandidate(self.size, candidate, self.p, self.uniform)
    
    def recombine(self, c: BitVectorCandidate) -> BitVectorCandidate:
        if self.uniform:
            return self.recombine_uniform(c)
        else:
            return self.recombine_single_point(c)

    def recombine_single_point(self, c: BitVectorCandidate) -> BitVectorCandidate:
        idx = np.random.choice(range(self.size))
        candidate = np.empty(np.max([self.size, c.size]), dtype=bool)
        candidate[:idx] = self.candidate[:idx]
        candidate[idx:] = c.candidate[idx:]
        return BitVectorCandidate(self.candidate.shape[0], candidate, self.p, self.uniform)
    
    def recombine_uniform(self, c: BitVectorCandidate) -> BitVectorCandidate:
        candidate = np.empty(np.min([self.size, c.size]), dtype=bool)
        for i in range(len(candidate)):
            if np.random.rand() < 0.5:
                candidate[i] = self.candidate[i]
            else:
                candidate[i] = c.candidate[i]



class FloatVectorCandidate(Candidate):
    def __init__(self, size: int, candidate: np.ndarray, distribution, lower=0, upper=1):
        self.size = size
        self.candidate = candidate
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def generate(size: int, distribution, lower=0, upper=1) -> FloatVectorCandidate:
        candidate = stats.uniform.rvs(size=size, loc=lower, scale=upper)
        return FloatVectorCandidate(size, candidate, distribution, lower, upper)

    def mutate(self) -> FloatVectorCandidate:
        candidate = self.candidate + self.distribution.rvs(size=self.size)
        for i in range(self. size):
            while (candidate[i] > self.upper) or (candidate[i] < self. lower):
                candidate = self.candidate + self.distribution.rvs(size=1)
        return FloatVectorCandidate(self.size, candidate, self.distribution, self.lower, self.upper)

    def recombine(self, c: FloatVectorCandidate) -> FloatVectorCandidate:
        alpha = np.random.rand()
        candidate = alpha*self.candidate + (1-alpha)*c.candidate
        return FloatVectorCandidate(self.candidate.shape[0], candidate, self.distribution, self.lower, self.upper)


class PathCandidate(Candidate):
    def __init__(self, path: list, graph: np.ndarray, max_length: int = None, mutable_length: bool = False):
        self.path = path
        self.graph = graph
        self.max_length = max_length
        self.mutable_length = mutable_length

    def generate(graph: np.ndarray, max_length: int = None):
        num_vs = graph.shape[0]

        if max_length is None:
            max_length = num_vs
        elif max_length < 0:
            max_length = 1
            p = np.random.rand()
            while np.random.rand() < p:
                max_length += 1

        path = []
        node = np.random.choice(range(num_vs))
        while len((graph[node,:] > 0).nonzero()[0]) == 0:
            node = np.random.choice(range(num_vs))
        path.append(node)

        while len(path) < max_length:
            node = np.random.choice( (graph[path[-1],:] > 0).nonzero()[0] )
            if len((graph[node,:] > 0).nonzero()[0]) > 0 or (len(path)+1) == max_length:
                path.append(node)

        return PathCandidate(path, graph, max_length)
    
    def mutate(self) -> PathCandidate:
        max_length = self.max_length
        break_point = np.random.choice(self.path)
        path = []

        i=0
        while i < len(self.path) and self.path[i] != break_point:
            path.append(self.path[i])
            i+=1
        path.append(break_point)
        i+=1

        if self.mutable_length:
            max_length = np.random.choice([len(path), max_length])
            p = np.random.rand()
            while np.random.rand() < p:
                max_length += 1
        
        while len(path) < max_length:
            node = np.random.choice( (self.graph[path[-1],:] > 0).nonzero()[0] )
            if len((self.graph[node,:] > 0).nonzero()[0]) > 0 or (len(path)+1) == max_length:
                path.append(node)

        return PathCandidate(path, self.graph, max_length)
    
    def recombine(self, c: PathCandidate) -> PathCandidate:
        
        break_point = np.random.choice(self.path)
        new_path = []

        i = 0
        while i < len(self.path) and self.path[i] != break_point:
            new_path.append(self.path[i])
            i+=1
        new_path.append(break_point)

        i = 0
        while i < len(c.path) and c.path[i] != break_point:
            i+=1
        i+=1
        while i < len(c.path) and len(new_path) < self.max_length:
            new_path.append(c.path[i])
            i+=1

        if len((self.graph[new_path[-1],:] > 0).nonzero()[0]) == 0:
            new_path.remove(new_path[-1])
        while len(new_path) < self.max_length:
            node = np.random.choice( (self.graph[new_path[-1],:] > 0).nonzero()[0] )
            if len((self.graph[node,:] > 0).nonzero()[0]) > 0 or (len(new_path)+1) == self.max_length:
                new_path.append(node)
        
        return PathCandidate(new_path, self.graph, self.max_length)
