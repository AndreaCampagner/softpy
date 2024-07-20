import sys
from typing import Callable

import numpy as np
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.fuzzyset import ContinuousFuzzySet, DiffSigmoidalFuzzySet, GBellFuzzySet, Gaussian2FuzzySet, GaussianFuzzySet, LinearSFuzzySet, LinearZFuzzySet, PiShapedFuzzySet, ProdSigmoidalFuzzySet, SShapedFuzzySet, SigmoidalFuzzySet, TrapezoidalFuzzySet, TriangularFuzzySet, ZShapedFuzzySet
from softpy.fuzzy.memberships_function import gaussian
from tests.fuzzy_set.configuration import not_raises 

class TestContinuousFuzzySet:
    @pytest.mark.parametrize(
            "memberships,epsilon,bound,exception_expected", 
            [
                (lambda x: 1 if x == 0 else 0, 1e-3, (0, 0), None), #check membership function
                (1, 1e-3, (0, 0), TypeError),
                (lambda x: 2 if x == 0 else 0, 1e-3, (0, 0), None), 
                (lambda x: 1 if x == 0 else -1, 1e-3, (0, 0), None),
                (gaussian, 1e-3,  (0, 0), None),
                (gaussian, (0, 1),  (0, 0), TypeError), #check epsilon
                (gaussian, 2,  (0, 0), ValueError),
                (gaussian, -1e-3,  (0, 0), ValueError),
                (gaussian, 1e-3, 2, TypeError) #check bound
            ])
    def test_creation(self,
                      memberships: Callable, 
                      epsilon: np.number, 
                      bound: tuple, 
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = ContinuousFuzzySet(memberships, bound, epsilon)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = ContinuousFuzzySet(memberships, bound, epsilon)

    @pytest.mark.parametrize(
            "fuzzy_set,alpha_cut,exception_expected", 
            [
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.1, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.2, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.3, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.4, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.5, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.6, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.7, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.7, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.8, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.9, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 0.1, None),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 2, ValueError),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 'a', TypeError),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), 2.2, ValueError),
                (ContinuousFuzzySet(gaussian, bound = (0, 1), epsilon = 1e-3), -1.1, ValueError),
            ])
    def test_alpha_cut(self, 
                       fuzzy_set: ContinuousFuzzySet, 
                       alpha_cut: np.number,
                       exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                step = int((fuzzy_set.bound[1] - fuzzy_set.bound[0] + 1)/fuzzy_set.epsilon)

                x_values = np.linspace(fuzzy_set.bound[0], 
                                       fuzzy_set.bound[1], 
                                       step)
                
                discr_memb_func = np.array([fuzzy_set.memberships_function(x) for x in  x_values])
                alpha_cut_set = np.array([v if v >= alpha_cut else np.nan for v in discr_memb_func])
                assert np.array_equal(alpha_cut_set, fuzzy_set[alpha_cut], equal_nan=True)
        else:
            with pytest.raises(exception_expected) as e_info:
                alpha_cut_set = fuzzy_set[alpha_cut]


    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (ContinuousFuzzySet(gaussian), 'a', TypeError),
                (ContinuousFuzzySet(gaussian), 3, None),
                (ContinuousFuzzySet(gaussian), 0, None),
                (ContinuousFuzzySet(gaussian), 1, None),
                (ContinuousFuzzySet(gaussian), 1.2, None),
                (ContinuousFuzzySet(gaussian), 1.5, None),
                (ContinuousFuzzySet(gaussian), 1.7, None),
                (ContinuousFuzzySet(gaussian, bound = (1, 2)), 2, None),
                (ContinuousFuzzySet(lambda x: 2 if x == 2 else 0, (1, 2), 1e-3), 2, None),
                (ContinuousFuzzySet(lambda x: 1 if x == 2 else -1, (1, 2), 1e-3), 2, None),
            ]
            )
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: tuple,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert (fuzzy_set(arg) - (fuzzy_set.memberships_function)(arg)) <= fuzzy_set.epsilon
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestTriangularFuzzySet:
    @pytest.mark.parametrize(
            "left,spike,right,exception_expected", 
            [
                (0, 1, 2, None),
                (0, 1, 2, None),
                (0, 1, 2, None),
                (2, 1, 0, ValueError),
                ('a', 1, 2, TypeError),
                (0, 'a', 2, TypeError),
                (0, 1, 'a', TypeError),
            ])
    def test_creation(self,
                      left: np.number, 
                      spike: np.number, 
                      right: np.number, 
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = TriangularFuzzySet(left, spike, right)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = TriangularFuzzySet(left, spike, right)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (TriangularFuzzySet(0, 1, 2), 0.5, None),
                (TriangularFuzzySet(0, 1, 2), 0, None),
                (TriangularFuzzySet(0, 1, 2), 2, None),
                (TriangularFuzzySet(0, 1, 2), -1, None),
                (TriangularFuzzySet(0, 1, 2), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestTrapezoidalFuzzySet:
    @pytest.mark.parametrize(
            "left_lower,left_upper,right_upper,right_lower,exception_expected", 
            [
                (0, 1, 2, 3, None),
                ('a', 1, 2, 3, TypeError),
                (0, 'a', 2, 3, TypeError),
                (0, 1, 'a', 3, TypeError),
                (0, 1, 2, 'a', TypeError),
                (3, 2, 1, 0, ValueError),
            ])
    def test_creation(self,
                      left_lower: np.number,
                      left_upper: np.number,
                      right_upper: np.number,
                      right_lower: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = TrapezoidalFuzzySet(left_lower, 
                                          left_upper, 
                                          right_upper, 
                                          right_lower)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = TrapezoidalFuzzySet(left_lower, 
                                          left_upper, 
                                          right_upper, 
                                          right_lower)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (TrapezoidalFuzzySet(0, 1, 2, 3), 0, None),
                (TrapezoidalFuzzySet(0, 1, 2, 3), 1, None),
                (TrapezoidalFuzzySet(0, 1, 2, 3), 2, None),
                (TrapezoidalFuzzySet(0, 1, 2, 3), 3, None),
                (TrapezoidalFuzzySet(0, 1, 2, 3), -1, None),
                (TrapezoidalFuzzySet(0, 1, 2, 3), 4, None),
                (TrapezoidalFuzzySet(0, 1, 2, 3), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestLinearZFuzzySet:
    @pytest.mark.parametrize(
            "left_upper,right_lower,exception_expected", 
            [
                (0, 1, None),
                ('a', 1, TypeError),
                (0, 'a', TypeError),
                (1, 0, ValueError),
            ])
    def test_creation(self,
                      left_upper: np.number,
                      right_lower: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = LinearZFuzzySet(left_upper, 
                                      right_lower)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = LinearZFuzzySet(left_upper, 
                                      right_lower)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (LinearZFuzzySet(0, 1), 0, None),
                (LinearZFuzzySet(0, 1), 1, None),
                (LinearZFuzzySet(0, 1), 2, None),
                (LinearZFuzzySet(0, 1), 3, None),
                (LinearZFuzzySet(0, 1), -1, None),
                (LinearZFuzzySet(0, 1), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestLinearSFuzzySet:
    @pytest.mark.parametrize(
            "left_lower,right_upper,epsilon,exception_expected", 
            [
                (0, 1, 1e-3, None),
                ('a', 1, 1e-3, TypeError),
                (0, 'a', 1e-3, TypeError),
                (1, 0, 1e-3, ValueError),
            ])
    def test_creation(self,
                      left_lower: np.number,
                      right_upper: np.number,
                      epsilon: np.number, 
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = LinearSFuzzySet(left_lower, 
                                      right_upper, 
                                      epsilon)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = LinearSFuzzySet(left_lower, 
                                      right_upper, 
                                      epsilon)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (LinearSFuzzySet(0, 1), 0, None),
                (LinearSFuzzySet(0, 1), 1, None),
                (LinearSFuzzySet(0, 1), 2, None),
                (LinearSFuzzySet(0, 1), 3, None),
                (LinearSFuzzySet(0, 1), -1, None),
                (LinearSFuzzySet(0, 1), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestGaussianFuzzySet:
    @pytest.mark.parametrize(
            "mean,std,exception_expected", 
            [
                (0, 1, None),
                (0, -1, ValueError),
                (0, 0, ValueError),
                (2, 1, None),
                ('a', 1, TypeError),
                (2, 'a', TypeError),
            ])
    def test_creation(self,
                      mean: np.number,
                      std: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = GaussianFuzzySet(mean,
                                       std)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = GaussianFuzzySet(mean,
                                       std)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (GaussianFuzzySet(0, 1), 0, None),
                (GaussianFuzzySet(0, 1), 1, None),
                (GaussianFuzzySet(0, 1), 2, None),
                (GaussianFuzzySet(0, 1), 3, None),
                (GaussianFuzzySet(0, 1), -1, None),
                (GaussianFuzzySet(0, 1), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestGaussian2FuzzySet:
    @pytest.mark.parametrize(
            "mean1,std1,mean2,std2,exception_expected", 
            [
                (0, 1, 1, 1, None),
                (0, -1, 1, 1, ValueError),
                (0, 1, 1, -1, ValueError),
                (0, 1, -1, 1, ValueError),
                ('a', 1, 1, 1, TypeError),
                (0, 'a', 1, 1, TypeError),
                (0, 1, 'a', 1, TypeError),
                (0, 1, 1, 'a', TypeError),
            ])
    def test_creation(self,
                      mean1: np.number,
                      std1: np.number,
                      mean2: np.number,
                      std2: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = Gaussian2FuzzySet(mean1,
                                        std1,
                                        mean2,
                                        std2)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = Gaussian2FuzzySet(mean1,
                                        std1,
                                        mean2,
                                        std2)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (Gaussian2FuzzySet(0, 1, 1, 1), 0, None),
                (Gaussian2FuzzySet(0, 1, 1, 1), 1, None),
                (Gaussian2FuzzySet(0, 1, 1, 1), 2, None),
                (Gaussian2FuzzySet(0, 1, 1, 1), -1, None),
                (Gaussian2FuzzySet(0, 1, 1, 1), 0.5, None),
                (Gaussian2FuzzySet(0, 1, 1, 1), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestGBellFuzzySet:
    @pytest.mark.parametrize(
            "width,slope,center,exception_expected", 
            [
                (1, 1, 1, None),
                (1, 1, -1, None),
                (0, 0, 1, ValueError),
                (-1, 0, 1, ValueError),
                (1, -1, 1, ValueError),
                ('a', 1, 1, TypeError),
                (1, 'a', 1, TypeError),
                (1, 1, 'a', TypeError),
            ])
    def test_creation(self,
                      width: np.number,
                      slope: np.number,
                      center: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = GBellFuzzySet(width,
                                    slope,
                                    center)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = GBellFuzzySet(width,
                                    slope,
                                    center)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (GBellFuzzySet(1, 1, 1), 0, None),
                (GBellFuzzySet(1, 1, 1), 1, None),
                (GBellFuzzySet(1, 1, 1), 2, None),
                (GBellFuzzySet(1, 1, 1), -1, None),
                (GBellFuzzySet(1, 1, 1), 0.5, None),
                (GBellFuzzySet(1, 1, 1), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestSigmoidalFuzzySet:
    @pytest.mark.parametrize(
            "width,center,exception_expected", 
            [
                (1, 1, None),
                (1, -1, None),
                (1, 1, None),
                (-1, 1, None),
                (0, 1, ValueError),
                ('a', 1, TypeError),
            ])
    def test_creation(self,
                      width: np.number,
                      center: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = SigmoidalFuzzySet(width,
                                        center)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = SigmoidalFuzzySet(width,
                                        center)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (SigmoidalFuzzySet(1, 0), 0, None),
                (SigmoidalFuzzySet(1, 0), 1, None),
                (SigmoidalFuzzySet(1, 0), 2, None),
                (SigmoidalFuzzySet(1, 0), -1, None),
                (SigmoidalFuzzySet(1, 0), 0.5, None),
                (SigmoidalFuzzySet(1, 0), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestDiffSigmoidalFuzzySet:
    @pytest.mark.parametrize(
            "width1,center1,width2,center2,exception_expected", 
            [
                (1, 1, 1, 2, None),
                (1, -1, 1, 2, None),
                (-1, 1, 1, 2, None),
                (0, 1, 1, 2, ValueError),
                (1, 1, 0, 2, ValueError),
                (1, 2, 1, 1, ValueError),
                ('a', 1, 1, 2, TypeError),
                (1, 'a', 1, 2, TypeError),
                (1, 1, 'a', 2, TypeError),
                (1, 1, 1, 'a', TypeError),
            ])
    def test_creation(self,
                      width1: np.number,
                      center1: np.number,
                      width2: np.number,
                      center2: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = DiffSigmoidalFuzzySet(width1,
                                            center1,
                                            width2,
                                            center2)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = DiffSigmoidalFuzzySet(width1,
                                            center1,
                                            width2,
                                            center2)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (DiffSigmoidalFuzzySet(1, 1, 1, 2), 0, None),
                (DiffSigmoidalFuzzySet(1, 1, 1, 2), 1, None),
                (DiffSigmoidalFuzzySet(1, 1, 1, 2), 2, None),
                (DiffSigmoidalFuzzySet(1, 1, 1, 2), -1, None),
                (DiffSigmoidalFuzzySet(1, 1, 1, 2), 0.5, None),
                (DiffSigmoidalFuzzySet(1, 1, 1, 2), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestProdSigmoidalFuzzySet:
    @pytest.mark.parametrize(
            "width1,center1,width2,center2,exception_expected", 
            [
                (1, 1, 1, 2, None),
                (1, -1, 1, 2, None),
                (-1, 1, 1, 2, None),
                (1, 1, -1, 2, None),
                (0, 1, 1, 2, ValueError),
                (1, 1, 0, 2, ValueError),
                (1, 2, 0, 1, ValueError),
                ('a', 1, 1, 2, TypeError),
                (1, 'a', 1, 2, TypeError),
                (1, 1, 'a', 2, TypeError),
                (1, 1, 1, 'a', TypeError),
            ])
    def test_creation(self,
                      width1: np.number,
                      center1: np.number,
                      width2: np.number,
                      center2: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = ProdSigmoidalFuzzySet(width1,
                                            center1,
                                            width2,
                                            center2)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = ProdSigmoidalFuzzySet(width1,
                                            center1,
                                            width2,
                                            center2)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (ProdSigmoidalFuzzySet(1, 1, 1, 2), 0, None),
                (ProdSigmoidalFuzzySet(1, 1, 1, 2), 1, None),
                (ProdSigmoidalFuzzySet(1, 1, 1, 2), 2, None),
                (ProdSigmoidalFuzzySet(1, 1, 1, 2), -1, None),
                (ProdSigmoidalFuzzySet(1, 1, 1, 2), 0.5, None),
                (ProdSigmoidalFuzzySet(1, 1, 1, 2), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestZShapedFuzzySet:
    @pytest.mark.parametrize(
            "left_upper,right_lower,exception_expected", 
            [
                (1, 2, None),
                (1, 1, None),
                (1, 0, ValueError),
                ('a', 1, TypeError),
                (1, 'a', TypeError),
            ])
    def test_creation(self,
                      left_upper: np.number, 
                      right_lower: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = ZShapedFuzzySet(left_upper,
                                      right_lower)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = ZShapedFuzzySet(left_upper,
                                      right_lower)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (ZShapedFuzzySet(0, 1), 0, None),
                (ZShapedFuzzySet(0, 1), 1, None),
                (ZShapedFuzzySet(0, 1), 2, None),
                (ZShapedFuzzySet(0, 1), -1, None),
                (ZShapedFuzzySet(0, 1), 0.5, None),
                (ZShapedFuzzySet(0, 1), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestSShapedFuzzySet:
    @pytest.mark.parametrize(
            "left_upper,right_lower,exception_expected", 
            [
                (1, 2, None),
                (1, 1, None),
                (1, 0, ValueError),
                ('a', 1, TypeError),
                (1, 'a', TypeError),
            ])
    def test_creation(self,
                      left_upper: np.number, 
                      right_lower: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = SShapedFuzzySet(left_upper,
                                      right_lower)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = SShapedFuzzySet(left_upper,
                                      right_lower)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (SShapedFuzzySet(0, 1), 0, None),
                (SShapedFuzzySet(0, 1), 1, None),
                (SShapedFuzzySet(0, 1), 2, None),
                (SShapedFuzzySet(0, 1), -1, None),
                (SShapedFuzzySet(0, 1), 0.5, None),
                (SShapedFuzzySet(0, 1), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)

class TestPiShapedFuzzySet:
    @pytest.mark.parametrize(
            "left_lower,left_upper,right_upper,right_lower,exception_expected", 
            [
                (1, 2, 3, 4, None),
                (1, 1, 3, 4, None),
                (1, 2, 4, 4, None),
                (1, 2, 2, 4, ValueError),
                ('a', 2, 2, 4, TypeError),
                (1, 'a', 2, 4, TypeError),
                (1, 2, 'a', 4, TypeError),
                (1, 2, 3, 'a', TypeError),
            ])
    def test_creation(self,
                      left_lower: np.number, 
                      left_upper: np.number, 
                      right_upper: np.number,
                      right_lower: np.number,
                      exception_expected: Exception):
        
        if exception_expected == None:
            with not_raises() as e_info:
                lfs = PiShapedFuzzySet(left_lower,
                                       left_upper,
                                       right_upper,
                                       right_lower)
        else:
            with pytest.raises(exception_expected) as e_info:
                lfs = PiShapedFuzzySet(left_lower,
                                       left_upper,
                                       right_upper,
                                       right_lower)
    
    @pytest.mark.parametrize(
            "fuzzy_set,arg,exception_expected",
            [
                (PiShapedFuzzySet(1, 2, 3, 4), 0, None),
                (PiShapedFuzzySet(1, 2, 3, 4), 1, None),
                (PiShapedFuzzySet(1, 2, 3, 4), 2, None),
                (PiShapedFuzzySet(1, 2, 3, 4), -1, None),
                (PiShapedFuzzySet(1, 2, 3, 4), 0.5, None),
                (PiShapedFuzzySet(1, 2, 3, 4), 'a', TypeError),
            ])
    def test_memberships(self, 
                         fuzzy_set: ContinuousFuzzySet, 
                         arg: np.number,
                         exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                assert fuzzy_set(arg) == fuzzy_set.memberships_function(arg)
        else:
            with pytest.raises(exception_expected) as e_info:
                memberships = fuzzy_set(arg)