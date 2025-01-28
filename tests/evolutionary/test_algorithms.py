import sys

import numpy as np
import scipy.stats as stats
import pytest
from functools import partial

sys.path.append(__file__ + "/../..")

from softpy.evolutionary.candidate import Candidate,BitVectorCandidate
from softpy.evolutionary.singlestate import HillClimbing, RandomSearch
from softpy.evolutionary.genetic import GeneticAlgorithm, SteadyStateGeneticAlgorithm
from softpy.evolutionary.selection import tournament_selection

from tests.fuzzy_set.configuration import not_raises 

class TestRandomSearch:
     
    @pytest.mark.parametrize(
        "candidate_type,fitness_func,pop_size,exception_expected",
        [
            (Candidate,lambda x: 1, 1, None),
            (int,lambda x: 1, 1, ValueError),
            (Candidate,"a", 1, ValueError),
            (Candidate,lambda x: 1, 0, ValueError),
            (Candidate,lambda x: 1, "a", TypeError),
        ]
    )
    def test_creation(self, candidate_type,fitness_func,pop_size,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                RandomSearch(candidate_type,fitness_func,pop_size)
        else:
            with pytest.raises(exception_expected) as e_info:
                RandomSearch(candidate_type,fitness_func,pop_size)

    @pytest.mark.parametrize(
        "exception_expected",
        [
            (AttributeError),
            (None)
        ]
    )
    def test_access(self,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = RandomSearch(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=1)
                r.fit()
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = RandomSearch(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=1)
                r.best

    @pytest.mark.parametrize(
        "n_iters,exception_expected",
        [
            (0,ValueError),
            (1,None)
        ]
    )
    def test_iters(self,n_iters,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = RandomSearch(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=1)
                r.fit(n_iters)
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = RandomSearch(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=1)
                r.fit(n_iters)
                r.best



class TestHillClimbing:
     
    @pytest.mark.parametrize(
        "candidate_type,fitness_func,test_size,exception_expected",
        [
            (Candidate,lambda x: 1, 1, None),
            (int,lambda x: 1, 1, ValueError),
            (Candidate,"a", 1, ValueError),
            (Candidate,lambda x: 1, 0, ValueError),
            (Candidate,lambda x: 1, "a", TypeError),
        ]
    )
    def test_creation(self, candidate_type,fitness_func,test_size,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                HillClimbing(candidate_type,fitness_func,test_size)
        else:
            with pytest.raises(exception_expected) as e_info:
                HillClimbing(candidate_type,fitness_func,test_size)

    @pytest.mark.parametrize(
        "exception_expected",
        [
            (AttributeError),
            (None)
        ]
    )
    def test_access(self,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = HillClimbing(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, test_size=1)
                r.fit()
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = HillClimbing(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, test_size=1)
                r.best

    @pytest.mark.parametrize(
        "n_iters,exception_expected",
        [
            (0,ValueError),
            (1,None)
        ]
    )
    def test_iters(self,n_iters,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = HillClimbing(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, test_size=1)
                r.fit(n_iters)
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = HillClimbing(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, test_size=1)
                r.fit(n_iters)
                r.best



class TestGeneticAlgorithm:
     
    @pytest.mark.parametrize(
        "candidate_type,fitness_func,pop_size,selection_func,elitism,n_elite,exception_expected",
        [
            (Candidate,lambda x: 1, 4, tournament_selection,False,0, None),
            (int,lambda x: 1, 4, tournament_selection,False,0, ValueError),
            (Candidate,"a", 4, tournament_selection,False,0, ValueError),
            (Candidate,lambda x: 1, 0, tournament_selection,False,0, ValueError),
            (Candidate,lambda x: 1, "a", tournament_selection,False,0, TypeError),
            (Candidate,lambda x: 1, 4, "a",False,0, ValueError),
            (Candidate,lambda x: 1, 4, tournament_selection,True,0, ValueError),
            (Candidate,lambda x: 1, 4, tournament_selection,True,1, None),
        ]
    )
    def test_creation(self,candidate_type,fitness_func,pop_size,selection_func,elitism,n_elite,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                GeneticAlgorithm(candidate_type,fitness_func,pop_size,selection_func,elitism,n_elite)
        else:
            with pytest.raises(exception_expected) as e_info:
                GeneticAlgorithm(candidate_type,fitness_func,pop_size,selection_func,elitism,n_elite)

    @pytest.mark.parametrize(
        "exception_expected",
        [
            (AttributeError),
            (None)
        ]
    )
    def test_access(self,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = GeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection,elitism=False)
                r.fit()
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = GeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection,elitism=False)
                r.best

    @pytest.mark.parametrize(
        "n_iters,exception_expected",
        [
            (0,ValueError),
            (1,None)
        ]
    )
    def test_iters(self,n_iters,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = GeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection,elitism=False)
                r.fit(n_iters)
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = GeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection,elitism=False)
                r.fit(n_iters)
                r.best

    @pytest.mark.parametrize(
        "backend,n_jobs,exception_expected",
        [
            ("loky",1,None),
            ("loky",2,None),
            ("loky",0,ValueError),
            ("error",1,ValueError)
        ]
    )
    def test_parallelism(self,backend,n_jobs,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = GeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection,elitism=False)
                r.fit(backend=backend, n_jobs=n_jobs)
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = GeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection,elitism=False)
                r.fit(backend=backend,n_jobs=n_jobs)
                r.best




class TestSteadyStateGeneticAlgorithm:
     
    @pytest.mark.parametrize(
        "candidate_type,fitness_func,pop_size,selection_func,exception_expected",
        [
            (Candidate,lambda x: 1, 4, tournament_selection, None),
            (int,lambda x: 1, 4, tournament_selection,ValueError),
            (Candidate,"a", 4, tournament_selection, ValueError),
            (Candidate,lambda x: 1, 0, tournament_selection, ValueError),
            (Candidate,lambda x: 1, "a", tournament_selection, TypeError),
            (Candidate,lambda x: 1, 4, "a", ValueError),
            (Candidate,lambda x: 1, 4, tournament_selection, None),
        ]
    )
    def test_creation(self,candidate_type,fitness_func,pop_size,selection_func,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                SteadyStateGeneticAlgorithm(candidate_type,fitness_func,pop_size,selection_func)
        else:
            with pytest.raises(exception_expected) as e_info:
                SteadyStateGeneticAlgorithm(candidate_type,fitness_func,pop_size,selection_func)

    @pytest.mark.parametrize(
        "exception_expected",
        [
            (AttributeError),
            (None)
        ]
    )
    def test_access(self,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = SteadyStateGeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection)
                r.fit()
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = SteadyStateGeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection)
                r.best

    @pytest.mark.parametrize(
        "n_iters,exception_expected",
        [
            (0,ValueError),
            (1,None)
        ]
    )
    def test_iters(self,n_iters,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                r = SteadyStateGeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection)
                r.fit(n_iters)
                r.best
        else:
            with pytest.raises(exception_expected) as e_info:
                r = SteadyStateGeneticAlgorithm(candidate_type=BitVectorCandidate,fitness_func=lambda x: 1, pop_size=3,selection_func=tournament_selection)
                r.fit(n_iters)
                r.best
            