from __future__ import annotations
import numpy as np
from fuzzy.fuzzyset import FuzzySet, LambdaFuzzySet, DiscreteFuzzySet, ContinuousFuzzySet
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
import scipy as sp



class FuzzyRule(ABC):
    '''
    An abstract class to represent a fuzzy rule.
    '''
    @abstractmethod
    def evaluate(self, params : dict):
        pass


class MamdaniRule(FuzzyRule):
    '''
    An implementation of a fuzzy rule for a Mamdani-type control system.
    The premise of the rule is a dictionary of FuzzySet instances, interpreted as a conjuction obtained using the specified tnorm operation.
    The tnorm itself is a function that constructs a FuzzyCombination object.
    The consequent is a FuzzySet instance.
    The contructor builds the rule as a new FuzzySet (specifically, a FuzzyCombination) by iteratively combining the FuzzySets in the premise
    and then with the consequent.
    '''
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
        if len(self.names) > 1:
            for k in self.names[::-1][1:]:
                self.antecedent = self.tnorm(self.premise[k], curr)
                curr = self.antecedent
        else:
            self.antecedent = curr
        self.rule = self.tnorm(self.antecedent, self.consequent)      
        

    def evaluate(self, params : dict):
        '''
        It evaluates the MamdaniRule at a certain set of values, by computing the corresponding membership degrees: in particular, the evaluate
        method accepts a specification of values for the FuzzySets in the premise and return a LambdaFuzzySet that evaluates the rule at any given
        value of the consequent.
        '''
        if not isinstance(params, dict):
            raise TypeError("params should be a dict or number")
        
        evals = None
        
        if len(self.names) > 1:
            temp_vals = [None] *self.names.shape[0]
            
            for k in params.keys():
                idx = np.where(self.names == k)[0]
                temp_vals[idx] = params[k]

            evals = self.antecedent(temp_vals)
        else:
            for k in params.keys():
                if k in self.names:
                    evals = self.antecedent(params[k])
        return LambdaFuzzySet(lambda u : partial(self.rule.op, evals)(self.consequent(u)))
    

class SugenoRule(FuzzyRule):
    '''
    An implementation of a fuzzy rule for a Mamdani-type control system.
    The premise of the rule is a dictionary of FuzzySet instances, interpreted as a conjuction obtained using the specified tnorm operation.
    The tnorm itself is a function that constructs a FuzzyCombination object.
    The consequent is a function (should be linear in the premise components).
    The contructor builds the rule as a new FuzzySet (specifically, a FuzzyCombination) by iteratively combining the FuzzySets in the premise.
    '''
    def __init__(self, premise : dict[str, FuzzySet], consequent_name: str, consequent,
                 tnorm):
        
        if not isinstance(premise, dict):
            raise TypeError("premise should be a dictionary")
        
        if not isinstance(consequent_name, str):
            raise TypeError("consequent_name should be a string")

        for k in premise:
            if not isinstance(premise[k], FuzzySet):
                raise TypeError("All premises should be FuzzySet") 

        
        self.premise = premise
        self.consequent = consequent
        self.consequent_name = consequent_name
        self.tnorm = tnorm

        self.names = np.array(list(self.premise.keys()))
        curr = self.premise[self.names[-1]]
        self.antecedent = None
        if len(self.names) > 1:
            for k in self.names[::-1][1:]:
                self.antecedent = self.tnorm(self.premise[k], curr)
                curr = self.antecedent
        else:
            self.antecedent = curr     
        

    def evaluate(self, params : dict):
        '''
        It evaluates the SugenoRule at a certain set of values, by computing the value: in particular, the evaluate
        method accepts a specification of values for the FuzzySets in the premise and returns the evaluation of the consequent function
        at these values.
        '''
        if not isinstance(params, dict):
            raise TypeError("params should be a dict or number")
        
        evals = None
        
        if len(self.names) > 1:
            temp_vals = [None] *self.names.shape[0]
            
            for k in params.keys():
                idx = np.where(self.names == k)[0]
                temp_vals[idx] = params[k]

            evals = self.antecedent(temp_vals)
        else:
            for k in params.keys():
                if k in self.names:
                    evals = self.antecedent(params[k])
        return evals, self.consequent(params)
    

class FuzzyControlSystem:
    '''
    An implementation of a FuzzyControlSystem: it can be either a Mamdani control system or a Sugeno control system, by specifying the appropriate type of
    rule. It is defined by a given set of rules (of the appropriate type) as well as (in the case of a Mamdani control system) by a tconorm operator.
    '''
    def __init__(self, rules: Sequence[FuzzyRule], tconorm = None, type=MamdaniRule):

        if type not in [MamdaniRule, SugenoRule]:
            raise TypeError("Rule type should be either MamdaniRule or SugenoRule")
        
        for r in rules:
            if not isinstance(r, type):
                raise TypeError("All rules should be of type %s" % type)
            
        self.rules = rules
        self.type = type
        self.tconorm = tconorm


    def evaluate_mamdani(self, results: dict, sets : dict):
        '''
        Applies the control for a Mamdani control system at the given input control, given the specified control results and the corresponding fuzzy sets.
        Specifically, it first combines the different controls for the same output using the specified tconorm operator, then applies defuzzification to
        reduce to a single control output.
        '''
        output = {}
        for n in results.keys():
            curr = results[n][0]
            for _, r in enumerate(results[n], 1):
                curr = self.tconorm(curr, r)
            
            max = -np.infty
            if isinstance(sets[n], DiscreteFuzzySet):
                for v in sets[n].items:
                    tmp = curr(v)
                    if tmp > max:
                        max = tmp
            elif isinstance(sets[n], ContinuousFuzzySet):
                max = sp.integrate.quad(lambda u: curr(u)*u, sets[n].min, sets[n].max)[0]
                max /= sp.integrate.quad(lambda u: curr(u), sets[n].min, sets[n].max)[0] + 0.001
            
            output[n] = max

        return output
    
    def evaluate_sugeno(self, results: dict):
        '''
        Applies the control for a Sugeno control system at the given input control, given the specified control results and fuzzy set memberships.
        Specifically, it combines the different controls for a given output by performing weighted average.
        '''
        output = {}
        for n in results.keys():
            num = 0
            denom = 0
            for i, _ in enumerate(results[n]):
                num += results[n][i][0]*results[n][i][1]
                denom += results[n][i][0]
            
            output[n] = num/denom

        return output


    def evaluate(self, params : dict, u = None) -> dict:
        '''
        It applies the control system at the given control input by applying all the rules and then returning an output signal.
        After applying all the rules in the control system rule base, it applies a different evaluation method based on whether
        it is a Mamdani or a Sugeno control system.
        '''
        results = {}
        sets = {}
        for r in self.rules:
            if r.consequent_name in results.keys():
                results[r.consequent_name].append(r.evaluate(params))
            else:
                results[r.consequent_name] = [r.evaluate(params)]
                sets[r.consequent_name] = r.consequent

        output = None
        if self.type == MamdaniRule:
            output = self.evaluate_mamdani(results, sets)
        elif self.type == SugenoRule:
            output = self.evaluate_sugeno(results)
        
        return output

    
            

