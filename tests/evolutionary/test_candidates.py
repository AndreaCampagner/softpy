import sys

import numpy as np
import scipy.stats as stats
import pytest
from functools import partial

sys.path.append(__file__ + "/../..")

from softpy.evolutionary.candidate import BitVectorCandidate, FloatVectorCandidate, PathCandidate, TreeCandidate, DictionaryCandidate
from tests.fuzzy_set.configuration import not_raises 

class TestBitVectorCandidate:

    @pytest.mark.parametrize(
        "size,p,uniform,exception_expected",
        [
            (1,0.5,False,None),
            (1,0.5,True,None),
            (-1,0.5,False,ValueError),
            (1,2,False,ValueError),
            (1,"error",False,TypeError),
            (1,0.5,"error",ValueError)
        ]
    )
    def test_creation(self,
                      size, p, uniform,
                      exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                np.random.seed(0)
                v = np.random.choice(a=[True,False], size=size, replace=True)
                np.random.seed(0)
                c = BitVectorCandidate.generate(size,p,uniform)
                assert c.candidate == v
        else:
            with pytest.raises(exception_expected) as e_info:
                c = BitVectorCandidate.generate(size,p,uniform)

    @pytest.mark.parametrize(
        "uniform,exception_expected",
        [
            (False,None)        ]
    )
    def test_recombine(self, uniform, exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                if uniform == True:
                    np.random.seed(42)
                    idx = np.random.randint(0,4)
                    np.random.seed(42)  
                    c1 = BitVectorCandidate.generate(4,0.2,uniform)
                    c2 = BitVectorCandidate.generate(4,0.2,uniform)
                    np.random.seed(42)
                    c = c1.recombine(c2)
                    print(c.candidate)
                    print(c1.candidate)
                    print(c2.candidate)
                    assert (c.candidate[:idx] == c1.candidate[:idx]).all() and (c.candidate[idx:] == c2.candidate[idx:]).all()
                else:
                    pass
        else:
            with pytest.raises(exception_expected) as e_info:
                pass

    @pytest.mark.parametrize(
        "exception_expected",
        [
            (None)
        ]
    )
    def test_mutate(self, exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                np.random.seed(42)
                vs = np.random.choice(a=[True,False], size=4, replace=True)
                np.random.seed(42)
                vs = np.array([v if np.random.rand() >= 0.2 else bool(1-v) for v in vs])
                np.random.seed(42)
                c = BitVectorCandidate.generate(4,0.2,False)
                np.random.seed(42)
                c = c.mutate()
                print(c.candidate)
                print(vs)
                assert (c.candidate == vs).all() 
        else:
            with pytest.raises(exception_expected) as e_info:
                pass



class TestFloatVectorCandidate:

    class DFDistrib:
        def __init__(self, distrib, df):
            self.distrib = distrib
            self.df = df

        def rvs(self, size):
            return self.distrib.rvs(df=self.df, size=size)

    @pytest.mark.parametrize(
        "size,distribution,lower,upper,intermediate,exception_expected",
        [
            (10,stats.uniform,0,1,False,None),
            (10,stats.norm,-1,1,True,None),
            (-1,stats.uniform,0,1,False,ValueError),
            (10,lambda x: x,0,1,False,ValueError)
        ]
    )
    def test_creation(self,
                      size, distribution, lower, upper, intermediate,
                      exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                c = FloatVectorCandidate.generate(size,distribution,lower,upper,intermediate)
        else:
            with pytest.raises(exception_expected) as e_info:
                print(e_info)
                c = FloatVectorCandidate.generate(size,distribution,lower,upper,intermediate)

    @pytest.mark.parametrize(
        "intermediate,exception_expected",
        [
            (False,None),
            (True,None)
        ]
    )
    def test_recombine(self, intermediate, exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                c1 = FloatVectorCandidate.generate(10,stats.uniform,0,1,intermediate)
                c2 = FloatVectorCandidate.generate(10,stats.uniform,0,1,intermediate)
                c = c1.recombine(c2)
                for i in range(10):
                    assert np.min([c1.candidate[i],c2.candidate[i]]) <= c.candidate[i]
                    assert c.candidate[i] <= np.max([c1.candidate[i],c2.candidate[i]])
        else:
            with pytest.raises(exception_expected) as e_info:
                pass

    @pytest.mark.parametrize(
        "size,distribution,lower,upper,intermediate,exception_expected",
        [
            (10,stats.uniform,0,1,False,None),
            (10,stats.norm,-1,1,False,None),
            (10,DFDistrib(stats.chi2, df=3),0,10,False,None),
            (10,DFDistrib(stats.t, df=3),-2,2,False,None)
        ]
    )
    def test_mutate(self, size,distribution,lower,upper,intermediate, exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                c = FloatVectorCandidate.generate(size,distribution,lower,upper,intermediate)
                for i in range(10):
                    assert lower <= c.candidate[i]
                    assert c.candidate[i] <= upper
        else:
            with pytest.raises(exception_expected) as e_info:
                pass

 #   graph: np.ndarray = np.empty(1), max_length: int | None = None, stop_early_prob:np.number=0.0, mutable_length:bool=False,
 #                mutate:Callable=None, recombine:Callable=None


class TestPathCandidate:

    graph = np.array([[0,1,0,0,1,0],
                      [1,0,1,0,1,0],
                      [0,1,0,1,0,0],
                      [0,0,1,0,1,1],
                      [1,1,0,1,0,0],
                      [0,0,0,1,0,0]])

    @pytest.mark.parametrize(
        "graph,max_length,stop_early_prob,mutable_length,exception_expected",
        [
            (graph,10,0.1,False,None),
            (graph,10,0.1,True,None),
            (graph,-1,0.1,False,None),
            (graph,0,0.1,False,ValueError)
        ]
    )
    def test_creation(self,graph,max_length,stop_early_prob,mutable_length,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c = PathCandidate.generate(graph,max_length,stop_early_prob,mutable_length)
                assert len(c.path) <= c.max_length
                i = 0
                while i < len(c.path)-1:
                    assert graph[c.path[i],c.path[i+1]] > 0
                    i+=1
        else:
            with pytest.raises(exception_expected) as e_info:
                c = PathCandidate.generate(graph,max_length,stop_early_prob,mutable_length)

    @pytest.mark.parametrize(
        "graph,max_length,stop_early_prob,mutable_length,exception_expected",
        [
            (graph,10,0.1,False,None),
            (graph,10,0.1,True,None),
            (graph,-1,0.1,False,None),
        ]
    )
    def test_mutate(self,graph,max_length,stop_early_prob,mutable_length,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c = PathCandidate.generate(graph,max_length,stop_early_prob,mutable_length)
                c = c.mutate()
                assert len(c.path) <= c.max_length
                i = 0
                while i < len(c.path)-1:
                    assert graph[c.path[i],c.path[i+1]] > 0
                    i+=1
        else:
            with pytest.raises(exception_expected) as e_info:
                c = PathCandidate.generate(graph,max_length,stop_early_prob,mutable_length)

    @pytest.mark.parametrize(
        "graph,max_length,stop_early_prob,mutable_length,exception_expected",
        [
            (graph,10,0.1,False,None),
            (graph,10,0.1,True,None),
            (graph,-1,0.1,False,None),
        ]
    )
    def test_recombine(self,graph,max_length,stop_early_prob,mutable_length,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c1 = PathCandidate.generate(graph,max_length,stop_early_prob,mutable_length)
                c2 = PathCandidate.generate(graph,max_length,stop_early_prob,mutable_length)
                c = c1.recombine(c2)
                assert len(c.path) <= c.max_length
                i = 0
                while i < len(c.path)-1:
                    assert graph[c.path[i],c.path[i+1]] > 0
                    i+=1
        else:
            with pytest.raises(exception_expected) as e_info:
                c = PathCandidate.generate(graph,max_length,stop_early_prob,mutable_length)



class TestTreeCandidate:

    funcs = ["+","*","/","-"]
    arities = [2, 2, 2, 1]

    def constant_generator(variables=["x"], low=-1.5, upp=1.5):
        r = np.random.rand()
        if r < 0.7:
            return np.random.choice(variables)
        else:
            return np.random.uniform(low, upp)

    @pytest.mark.parametrize(
        "function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob,exception_expected",
        [
            (funcs,arities,3,5,constant_generator,0.1,0.1,None),
            (funcs,arities,-1,5,constant_generator,0.1,0.1,ValueError),
            (funcs,arities,"a",5,constant_generator,0.1,0.1,TypeError),
            (funcs,arities,3,-1,constant_generator,0.1,0.1,ValueError),
            (funcs,arities,3,"a",constant_generator,0.1,0.1,TypeError),
            (funcs,arities,3,5,"prova",0.1,0.1,TypeError),
            (funcs,arities,3,5,constant_generator,"a",0.1,Exception),
            (funcs,arities,3,5,constant_generator,0.1,"a",Exception),
            (np.array(funcs)[:-1],arities,3,-1,constant_generator,0.1,0.1,ValueError),
        ]
    )
    def test_creation(self, function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                print(e_info)
                c = TreeCandidate.generate(function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob)
                nodes = [c.root]

                while len(nodes) != 0:
                    node: TreeCandidate.NodeCandidate = nodes.pop()
                    depth = node.depth
                    assert depth <= max_depth
                    num = 0 if node.children is None else len(node.children)
                    num = 0 if node.children is None else len(node.children)
                    for i in range(num):
                        nodes.append(node.children[i])
                    
        else:
            with pytest.raises(exception_expected) as e_info:
                print(e_info)
                c = TreeCandidate.generate(function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob)


    @pytest.mark.parametrize(
        "function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob,exception_expected",
        [
            (funcs,arities,3,5,constant_generator,0.1,0.1,None),
        ]
    )
    def test_mutate(self, function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c1 = TreeCandidate.generate(function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob)
                c = c1.mutate()
                nodes = [c.root]

                while len(nodes) != 0:
                    node: TreeCandidate.NodeCandidate = nodes.pop()
                    depth = node.depth
                    assert depth <= max_depth
                    num = 0 if node.children is None else len(node.children)
                    num = 0 if node.children is None else len(node.children)
                    for i in range(num):
                        nodes.append(node.children[i])
                    
        else:
            with pytest.raises(exception_expected) as e_info:
                print(e_info)
                c = TreeCandidate.generate(function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob)  

    @pytest.mark.parametrize(
        "function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob,exception_expected",
        [
            (funcs,arities,3,5,constant_generator,0.1,0.1,None),
        ]
    )
    def test_recombine(self, function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c1 = TreeCandidate.generate(function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob)
                c2 = TreeCandidate.generate(function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob)
                c = c1.recombine(c2)
                nodes = [c.root]

                while len(nodes) != 0:
                    node: TreeCandidate.NodeCandidate = nodes.pop()
                    depth = node.depth
                    assert depth <= max_depth
                    num = 0 if node.children is None else len(node.children)
                    num = 0 if node.children is None else len(node.children)
                    for i in range(num):
                        nodes.append(node.children[i])
                    
        else:
            with pytest.raises(exception_expected) as e_info:
                print(e_info)
                c = TreeCandidate.generate(function_set,arities,max_depth,max_absolute_depth,constant_generator,stop_early_prob,mutate_prob)  


class TestDictionaryCandidate:
    names = ["a","b","c","d"]
    gens = [range(1,5), stats.uniform.rvs, ["a","b","c"], partial(stats.t.rvs, 2)]
    discrete = [True, False, True, False]
    gens_wrong1 = [range(1,5), stats.uniform.rvs, ["a","b","c"], ["a","b","c"]]
    gens_wrong2 = [range(1,5), stats.uniform.rvs, stats.uniform.rvs, partial(stats.t.rvs, 2)]

    update_distrib = [
        lambda x: int(np.max([x+np.random.choice([-1,0,1]),1])) if np.random.rand() < 0.3 else x,
        lambda x: np.max([1.05,x + stats.norm(loc=0, scale=1).rvs()]),
        lambda x: np.random.choice(["a","b","c"]) if np.random.rand() < 0.3 else x,
        lambda x: x
    ]

    update_distrib_wrong = [
        lambda x: int(np.max([x+np.random.choice([-1,0,1]),1])) if np.random.rand() < 0.3 else x,
        lambda x: np.max([1.05,x + stats.norm(loc=0, scale=1).rvs()]),
        lambda x: np.random.choice(["a","b","c"]) if np.random.rand() < 0.3 else x,
        "a"
    ]

    @pytest.mark.parametrize(
        "names,gens,discrete,update_distrib,exception_expected",
        [
            (names,gens,discrete,update_distrib,None),
            (np.array(names)[:1],gens,discrete,update_distrib,ValueError),
            (names,gens,list(np.array(discrete)[:-1]) + ["True"],update_distrib,ValueError),
            (names,gens_wrong1,discrete,update_distrib,TypeError),
            (names,gens_wrong2,discrete,update_distrib,ValueError),
        ]
    )
    def test_creation(self,names,gens,discrete,update_distrib,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c = DictionaryCandidate.generate(names,gens,discrete,update_distrib)
        else:
            with pytest.raises(exception_expected) as e_info:
                c = DictionaryCandidate.generate(names,gens,discrete,update_distrib)

    @pytest.mark.parametrize(
        "names,gens,discrete,update_distrib,exception_expected",
        [
            (names,gens,discrete,update_distrib,None),
            (names,gens,discrete,update_distrib_wrong,TypeError),
        ]
    )
    def test_mutate(self,names,gens,discrete,update_distrib,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c = DictionaryCandidate.generate(names,gens,discrete,update_distrib)
                c.mutate()
        else:
            with pytest.raises(exception_expected) as e_info:
                c = DictionaryCandidate.generate(names,gens,discrete,update_distrib)
                c.mutate()

    @pytest.mark.parametrize(
        "names,gens,discrete,update_distrib,exception_expected",
        [
            (names,gens,discrete,update_distrib,None)
        ]
    )
    def test_recombine(self,names,gens,discrete,update_distrib,exception_expected):
        if exception_expected == None:
            with not_raises() as e_info:
                c1 = DictionaryCandidate.generate(names,gens,discrete,update_distrib)
                c2 = DictionaryCandidate.generate(names,gens,discrete,update_distrib)
                c1.recombine(c2)
        else:
            with pytest.raises(exception_expected) as e_info:
                c1 = DictionaryCandidate.generate(names,gens,discrete,update_distrib)
                c2 = DictionaryCandidate.generate(names,gens,discrete,update_distrib)
                c1.recombine(c2)