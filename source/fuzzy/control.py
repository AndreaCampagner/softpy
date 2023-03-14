import numpy as np
from fuzzy.fuzzyset import FuzzySet, LambdaFuzzySet
from fuzzy.operations import FuzzyCombination
from functools import partial
from typing import Type


class MamdaniRule:

    def __init__(self, premise : dict[str, FuzzySet], consequent_name: str, consequent: FuzzySet,
                 tnorm):
        
        if not isinstance(premise, dict):
            raise TypeError("premise should be a dictionary")
        
        if not isinstance(consequent_name, str):
            raise TypeError("consequent_name should be a string")

        for k in premise:
            if not isinstance(premise[k], FuzzySet):
                raise TypeError("All premises should be FuzzySet") 

        if not isinstance(consequent, FuzzySet):
            raise TypeError("consequent should be a FuzzySet")  

        
        self.premise = premise
        self.consequent = consequent
        self.consequent_name = consequent_name
        self.tnorm = tnorm

        self.names = np.array(list(self.premise.keys()))
        curr = self.premise[self.names[-1]]
        self.antecedent = None
        for k in self.names[::-1][1:]:
            self.antecedent = self.tnorm(self.premise[k], curr)
            curr = self.antecedent
        self.rule = self.tnorm(self.antecedent, self.consequent)      
        

    def evaluate(self, params : dict, u=None): 
        if not isinstance(params, dict):
            raise TypeError("params should be a dict")
        
        temp_vals = [None] *self.names.shape[0]
        
        for k in params.keys():
            idx = np.where(self.names == k)[0][0]
            print(idx)
            temp_vals[idx] = params[k]

        evals = self.antecedent(temp_vals)
        print(self.rule.op)
        return LambdaFuzzySet(lambda u : partial(self.rule.op, evals)(self.consequent(u)))
