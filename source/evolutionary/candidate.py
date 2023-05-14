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

    def __init__(self, size: int, candidate: np.ndarray, p: np.number):
        self.size = size
        self.p = p
        self.candidate = candidate

    def generate(size: int, p: np.number) -> BitVectorCandidate:
        candidate = np.random.choice(a=[True,False], size=size, replace=True)
        return BitVectorCandidate(size, candidate, p)

    def mutate(self) -> BitVectorCandidate:
        candidate = np.array([v if np.random.rand() >= self.p else bool(1-v) for v in self.candidate])
        return BitVectorCandidate(self.size, candidate, self.p)

    def recombine(self, c: BitVectorCandidate) -> BitVectorCandidate:
        idx = np.random.choice(range(self.size))
        candidate = np.empty(np.max([self.size, c.size]), dtype=bool)
        candidate[:idx] = self.candidate[:idx]
        candidate[idx:] = c.candidate[idx:]
        return BitVectorCandidate(self.candidate.shape[0], candidate, self.p)


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