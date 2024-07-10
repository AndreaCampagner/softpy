"""
Definition of abstract class for a generic fuzzy set.
"""

from abc import ABC, abstractmethod
import numpy as np


class FuzzySet(ABC):
    '''Abstract Class for a generic fuzzy set'''

    @abstractmethod
    def __call__(self, arg) -> np.number:
        pass

    @abstractmethod
    def __getitem__(self, alpha):
        pass

    @abstractmethod
    def fuzziness(self) -> np.number:
        '''Abstract Method to compute fuzzyness of a fuzzy set'''

    @abstractmethod
    def entropy(self) -> np.number:
        '''Abstract Method to compute hartley entropy of a fuzzy set'''
