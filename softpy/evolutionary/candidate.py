from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import scipy.stats as stats
from copy import deepcopy


class Candidate(ABC):
    '''
    Abstract class for representing individuals/candidate solutions.
    Concrete sub-classes should be implemented as a factory, and instances should be initialized using the generate method.
    '''
    @abstractmethod
    def generate(mutate:Callable|None=None, recombine:Callable|None=None) -> Candidate:
        '''
        Factory method to generate an instance object.
        '''
        pass

    @abstractmethod
    def mutate(self) -> Candidate:
        '''
        Default implementation of the mutation function.
        '''
        pass

    @abstractmethod
    def recombine(self, c: Candidate) -> Candidate:
        '''
        Default implementation of the crossover function.
        '''
        pass


class BitVectorCandidate(Candidate):
    '''
    Implementation of candidate solutions represented in terms of binary vectors.
    The class is implemented as a factory, and instances should be initialized using the generate method.
    
    Parameters
    ----------
    :param size: the number of components in the candidate solution
    :type size: int

    :param candidate: the representation of the candidate solution
    :type candidate: np.ndarray

    :param p: the mutation probability
    :type p: np.number, default=0.5

    :param uniform: specifies whether to adopt uniform (if True) or single-point (if False) crossover
    :type uniform: bool, default=False

    :param mutate: optional user-defined function for mutation
    :type mutate: Callable|None, default=None

    :param recombine: optional user-defined function for crossover
    :type recombine: Callable|None, default=None
    '''
    def __init__(self, size: int, candidate: np.ndarray, p: np.number = 0.5, uniform: bool = False, mutate: Callable|None=None, recombine: Callable|None=None):
        self.size = size
        self.p = p
        self.candidate = candidate
        self.uniform = uniform
        if mutate is not None:
            self.mutate = mutate
        if recombine is not None:
            self.recombine = recombine

    def generate(size: int = 1, p: np.number = 0.5, uniform: bool = False, mutate:Callable|None=None, recombine:Callable|None=None) -> BitVectorCandidate:
        if type(uniform) != bool:
            raise ValueError("uniform must be of type bool, was %s" % type(uniform))
        if p < 0 or p > 1:
            raise ValueError("p must be a number between 0 and 1, was %s of type %s" % (p, type(p)))
        candidate = np.random.choice(a=[True,False], size=size, replace=True)
        return BitVectorCandidate(size, candidate, p, uniform, mutate, recombine)

    def mutate(self) -> BitVectorCandidate:
        candidate = np.array([v if np.random.rand() >= self.p else bool(1-v) for v in self.candidate])
        return BitVectorCandidate(self.size, candidate, self.p, self.uniform,mutate=self.mutate, recombine=self.recombine)
    
    def recombine(self, c: BitVectorCandidate) -> BitVectorCandidate:
        if self.uniform:
            return self.recombine_uniform(c)
        else:
            return self.recombine_single_point(c)

    def recombine_single_point(self, c: BitVectorCandidate) -> BitVectorCandidate:
        idx = np.random.randint(0,self.size)
        candidate = np.empty(np.max([self.size, c.size]), dtype=bool)
        candidate[:idx] = self.candidate[:idx]
        candidate[idx:] = c.candidate[idx:]
        return BitVectorCandidate(self.candidate.shape[0], candidate, self.p, self.uniform, mutate=self.mutate, recombine=self.recombine)
    
    def recombine_uniform(self, c: BitVectorCandidate) -> BitVectorCandidate:
        candidate = np.empty(np.min([self.size, c.size]), dtype=bool)
        for i in range(len(candidate)):
            if np.random.rand() < 0.5:
                candidate[i] = self.candidate[i]
            else:
                candidate[i] = c.candidate[i]
        return BitVectorCandidate(self.candidate.shape[0], candidate, self.p, self.uniform, mutate=self.mutate, recombine=self.recombine)




class FloatVectorCandidate(Candidate):
    '''
    Implementation of candidate solutions represented in terms of real-valued vectors.
    The distribution parameter should be compatible with the scipy random interface, in particular sampling should be implemented by calling the rvs method.
    The class is implemented as a factory, and instances should be initialized using the generate method.
    
    Parameters
    ----------
    :param size: the number of components in the candidate solution
    :type size: int

    :param candidate: the representation of the candidate solution
    :type candidate: np.ndarray

    :param distribution: the probability distribution used for mutation.
    :type p: object (must implement a rvs method)

    :param lower: lower bound of the interval of definition for generation and mutation
    :type lower: np.number, default=0

    :param upper: upper bound of the interval of definition for generation and mutation
    :type upper: np.number, default=1

    :param intermediate: specifies whether to use intermediate (if True) or line (if False) recombination for crossover
    :type intermediate: bool, default=False

    :param mutate: optional user-defined function for mutation
    :type mutate: Callable|None, default=None

    :param recombine: optional user-defined function for crossover
    :type recombine: Callable|None, default=None
    '''
    def __init__(self, size: int, candidate: np.ndarray, distribution, lower:np.number=0, upper:np.number=1,
                intermediate:bool=False, mutate:Callable|None=None, recombine:Callable|None=None):
        self.size = size
        self.candidate = candidate
        self.distribution = distribution
        self.lower = lower
        self.upper = upper
        self.intermediate = intermediate
        if mutate is not None:
            self.mutate = mutate
        if recombine is not None:
            self.recombine = recombine

    def generate(size: int = 1, distribution=stats.uniform, lower:np.number=0, upper:np.number=1, intermediate:bool=False,
                mutate:Callable|None=None, recombine:Callable|None=None) -> FloatVectorCandidate:
        try:
            _ = distribution.rvs(size=size)
        except:
            raise ValueError("distribution should have rvs method")
        candidate = stats.uniform.rvs(size=size, loc=lower, scale=upper)
        return FloatVectorCandidate(size, candidate, distribution, lower, upper, intermediate, mutate, recombine)

    def mutate(self) -> FloatVectorCandidate:
        candidate = self.candidate + self.distribution.rvs(size=self.size)
        for i in range(self.size):
            if type(self.lower) in [list, np.array]:
                while (candidate[i] > self.upper[i]) or (candidate[i] < self. lower[i]):
                    candidate = self.candidate + self.distribution.rvs(size=1)
            else:
                while (candidate[i] > self.upper) or (candidate[i] < self. lower):
                    candidate = self.candidate + self.distribution.rvs(size=1)
        return FloatVectorCandidate(self.size, candidate, self.distribution, self.lower, self.upper, mutate=self.mutate, recombine=self.recombine)

    def recombine(self, c: FloatVectorCandidate) -> FloatVectorCandidate:
        alpha = np.random.rand()
        if self.intermediate:
            alpha = np.random.rand(self.size)
        candidate = alpha*self.candidate + (1-alpha)*c.candidate
        return FloatVectorCandidate(self.candidate.shape[0], candidate, self.distribution, self.lower, self.upper,
                                    mutate=self.mutate, recombine=self.recombine)


class PathCandidate(Candidate):
    '''
    Implementation of candidate solutions represented in terms of paths on a graph. 
    It can also be used to implement algorithms based on a list representation, by using the graph parameter as specification of the grammar.
    The class is implemented as a factory, and instances should be initialized using the generate method.
    
    Parameters
    ----------
    :param path: the representation of the candidate solution
    :type path: list

    :param graph: provides a specification of the possible structure of the candidates. By default is assumed to be a graph specifying the connections between vertices
    :type p: np.ndarray

    :param max_length: maximum length of a candidate. None or negative values denote no maximum length.
    :type max_length: int|None, default=None

    :param mutable_length: specifies whether the max_length of candidates can be altered during mutation or crossover
    :type mutable_length: bool, default=False

    :param stop_early_prob: controls the probability to stop growing a candidate solution smaller than max_length
    :type stop_early_prob: np.number (must be between 0 and 1), default=0

    :param mutate: optional user-defined function for mutation
    :type mutate: Callable|None, default=None

    :param recombine: optional user-defined function for crossover
    :type recombine: Callable|None, default=None
    '''
    def __init__(self, path: list, graph: np.ndarray, max_length: int|None = None, mutable_length: bool = False,
                stop_early_prob:np.number=0.0, mutate:Callable|None=None, recombine:Callable|None=None):
        self.path = path
        self.graph = graph
        self.max_length = max_length
        self.mutable_length = mutable_length
        self.stop_early_prob=stop_early_prob
        if mutate is not None:
            self.mutate = mutate
        if recombine is not None:
            self.recombine = recombine

    def generate(graph: np.ndarray = np.empty(1), max_length: int | None = None, stop_early_prob:np.number=0.0, mutable_length:bool=False,
                 mutate:Callable|None=None, recombine:Callable|None=None):
        num_vs = graph.shape[0]

        if max_length == 0:
            raise ValueError("max_length should be a positive integer, None or a negative integer, was 0")

        if max_length is None or max_length < 0:
            max_length = 1
            p = np.random.rand()
            while np.random.rand() < p:
                max_length += 1

        stop_early_prob = np.clip(stop_early_prob, 0, 1)

        path = []
        node = np.random.choice(range(num_vs))
        while len((graph[node,:] > 0).nonzero()[0]) == 0:
            node = np.random.choice(range(num_vs))
        path.append(node)

        while len(path) < max_length and np.random.rand() > stop_early_prob:
            node = np.random.choice( (graph[path[-1],:] > 0).nonzero()[0] )
            if len((graph[node,:] > 0).nonzero()[0]) > 0 or (len(path)+1) == max_length:
                path.append(node)

        return PathCandidate(path=path, graph=graph, max_length=max_length, mutable_length=mutable_length,
                            stop_early_prob=stop_early_prob, mutate=mutate, recombine=recombine)
    
    def mutate(self) -> PathCandidate:
        if self.mutable_length:
            max_length = self.max_length
            p = np.random.rand()
            while np.random.rand() < p and max_length > 1:
                max_length += np.random.choice([-1,0,1])

            self.max_length = max_length
        break_point = np.random.choice(self.path)
        path = []

        i=0
        while i < len(self.path) and self.path[i] != break_point and i < self.max_length-1:
            path.append(self.path[i])
            i+=1
        path.append(break_point)
        i+=1

        
        
        while len(path) < self.max_length and np.random.rand() > self.stop_early_prob:
            node = np.random.choice( (self.graph[path[-1],:] > 0).nonzero()[0] )
            if len((self.graph[node,:] > 0).nonzero()[0]) > 0 or (len(path)+1) == self.max_length:
                path.append(node)

        return PathCandidate(path, self.graph, self.max_length, mutable_length=self.mutable_length, stop_early_prob=self.stop_early_prob,
                             mutate=self.mutate, recombine=self.recombine)
    
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
        while len(new_path) < self.max_length and np.random.rand() > self.stop_early_prob:
            node = np.random.choice( (self.graph[new_path[-1],:] > 0).nonzero()[0] )
            if len((self.graph[node,:] > 0).nonzero()[0]) > 0 or (len(new_path)+1) == self.max_length:
                new_path.append(node)
        
        return PathCandidate(new_path, self.graph, self.max_length, mutable_length=self.mutable_length, stop_early_prob=self.stop_early_prob,
                             mutate=self.mutate, recombine=self.recombine)
    

class TreeCandidate(Candidate):
    '''
    Implementation of candidate solutions represented in terms of trees. 
    It can be used to implement (tree-based) genetic programming solutions.
    The class is implemented as a factory, and instances should be initialized using the generate method.
    
    Parameters
    ----------
    :param function_set: a list of objects or names to be used for internal nodes
    :type function_set: list

    :param arities: the arities of the elements in function_set
    :type arities: list[int]|np.ndarray

    :param max_absolute_depth: the maximum allowed depth of candidates
    :type max_absolute_depth: int

    :param constant_generator: a Callable object that specifies how to generate leaf nodes
    :type constant_generator: Callable

    :param stop_early_prob: controls the probability to stop growing a candidate solution smaller than max_length
    :type stop_early_prob: np.number (must be between 0 and 1), default=0

    :param mutate_prob: specifies the mutation probability. The probability is applied independently for each sub-tree
    :type mutate_prob: np.number

    :param mutate: optional user-defined function for mutation
    :type mutate: Callable|None, default=None

    :param recombine: optional user-defined function for crossover
    :type recombine: Callable|None, default=None
    '''
    class NodeCandidate:
        '''
        Implements a node in a tree. Used internally by TreeCandidate
        '''
        def __init__(self, function, children, depth = 1, parent=None, child_id=None):
            self.function = function
            self.children = children
            self.depth = depth
            self.parent = parent
            self.child_id = child_id

    def __init__(self, root: TreeCandidate.NodeCandidate, function_set: list, arities: list[int]|np.ndarray, max_absolute_depth: int,
                constant_generator: Callable, stop_early_prob: np.number, mutate_prob: np.number, mutate:Callable|None=None, recombine:Callable|None=None):
        self.root = root
        self.function_set = function_set
        self.arities = arities
        self.max_absolute_depth = max_absolute_depth
        self.constant_generator = constant_generator
        self.stop_early_prob = stop_early_prob
        self.mutate_prob = mutate_prob
        if mutate is not None:
            self.mutate = mutate
        if recombine is not None:
            self.recombine = recombine

    def generate(function_set: list = [], arities: list[int]|np.ndarray = [], max_depth:int = 1, max_absolute_depth:int = 1,
                constant_generator:Callable = None, stop_early_prob:np.number = 0.0, mutate_prob:np.number=0.2,
                mutate:Callable|None=None, recombine:Callable|None=None) -> TreeCandidate:
        depth = 1
        parent = None

        nodes = []

        stop_early_prob = np.clip(stop_early_prob, 0, 1)
        mutate_prob = np.clip(mutate_prob, 0, 1)

        if len(function_set) != len(arities):
            raise ValueError("function_set and arities must have same length")
        
        if max_depth <= 0:
            raise ValueError("max_depth must be greater an integer than 0, was %s" % max_depth)
        
        if max_absolute_depth <= 0:
            raise ValueError("max_absolute_depth must be greater than 0, was %s" % max_absolute_depth)
        

        r = np.random.rand()
        if (depth >= max_depth)  or (r < stop_early_prob):
            val = constant_generator()
            nodes.append(TreeCandidate.NodeCandidate(val, None, depth, parent, None))
        else:
            idx = np.random.choice(range(len(function_set)))
            fun = function_set[idx]
            ari = arities[idx]
            nodes.append(TreeCandidate.NodeCandidate(fun, np.empty(ari, dtype=TreeCandidate.NodeCandidate), depth, parent, None))
            

        root = nodes[0]

        while len(nodes) != 0:
            node: TreeCandidate.NodeCandidate = nodes.pop()
            num = 0 if node.children is None else len(node.children)
            depth = node.depth
            for i in range(num):
                r = np.random.rand()
                if (depth + 1 == max_depth) or (r < stop_early_prob):
                    val = constant_generator()
                    child = TreeCandidate.NodeCandidate(val, None, depth+1, node, i)
                elif depth + 1 < max_depth:
                    idx = np.random.choice(range(len(function_set)))
                    fun = function_set[idx]
                    ari = arities[idx]
                    child = TreeCandidate.NodeCandidate(fun, np.empty(ari, dtype=TreeCandidate.NodeCandidate), depth+1, node, i)
                    nodes.append(child)

                node.children[i] = child

        return TreeCandidate(root, function_set, arities, max_absolute_depth, constant_generator, stop_early_prob, mutate_prob, mutate, recombine)


    def mutate(self) -> TreeCandidate:
        new_tree = deepcopy(self)
        nodes = [new_tree.root]
        while len(nodes) != 0:
            node: TreeCandidate.NodeCandidate = nodes.pop()
            r = np.random.rand()
            if node.depth == new_tree.max_absolute_depth or r < new_tree.stop_early_prob:
                node.function = new_tree.constant_generator()
                node.children = None
            else:
                r = np.random.rand()
                if r < new_tree.mutate_prob:
                    idx = np.random.choice(range(len(new_tree.function_set)))
                    node.function = new_tree.function_set[idx]
                    ari = new_tree.arities[idx]
                    node.children = np.empty(ari, dtype=TreeCandidate.NodeCandidate)
                    curr_nodes = [node]

                    while len(curr_nodes) != 0:
                        curr_node: TreeCandidate.NodeCandidate = curr_nodes.pop()
                        num = 0 if curr_node.children is None else len(curr_node.children)
                        curr_depth = curr_node.depth
                        for i in range(num):
                            r = np.random.rand()
                            if (curr_depth + 1 == new_tree.max_absolute_depth) or (r < new_tree.stop_early_prob):
                                val = new_tree.constant_generator()
                                child = TreeCandidate.NodeCandidate(val, None, curr_depth+1, curr_node, i)
                            elif curr_depth + 1 < new_tree.max_absolute_depth:
                                idx = np.random.choice(range(len(new_tree.function_set)))
                                fun = new_tree.function_set[idx]
                                ari = new_tree.arities[idx]
                                child = TreeCandidate.NodeCandidate(fun, np.empty(ari, dtype=TreeCandidate.NodeCandidate), curr_depth+1, curr_node, i)
                                curr_nodes.append(child)

                            curr_node.children[i] = child
                else:
                    num = 0 if node.children is None else len(node.children)
                    for i in range(num):
                        nodes.append(node.children[i])
        return new_tree


    def recombine(self, c: TreeCandidate) -> TreeCandidate:
        rec_tree = deepcopy(self)
        nodes_self = []
        queue_self = [rec_tree.root]

        while len(queue_self) != 0:
            node: TreeCandidate.NodeCandidate = queue_self.pop()
            nodes_self.append(node)
            num = 0 if node.children is None else len(node.children)
            for i in range(num):
                queue_self.append(node.children[i])

        old: TreeCandidate.NodeCandidate = np.random.choice(nodes_self)

        nodes_c = []
        queue_c = [c.root]

        while len(queue_c) != 0:
            node: TreeCandidate.NodeCandidate = queue_c.pop()
            nodes_c.append(node)
            num = 0 if node.children is None else len(node.children)
            for i in range(num):
                queue_c.append(node.children[i])

        new: TreeCandidate.NodeCandidate = deepcopy(np.random.choice(nodes_c))


        parent_old : TreeCandidate.NodeCandidate = old.parent
        idx = old.child_id
        depth = old.depth

        if parent_old is not None:
            parent_old.children[idx] = new
        else:
            rec_tree.root = new

        new.parent = parent_old
        new.child_id = idx
        new.depth = depth
        nodes = [new]
        while len(nodes) != 0:
            node: TreeCandidate.NodeCandidate = nodes.pop()
            num = 0 if node.children is None else len(node.children)
            depth = node.depth
            for i in range(num):
                node.children[i].depth = depth + 1
                nodes.append(node.children[i])
        
        return rec_tree


class DictionaryCandidate(Candidate):
    '''
    Generic implementation of candidate solutions that can be represented in terms of dictionaries.
    The class is implemented as a factory, and instances should be initialized using the generate method.
    
    Parameters
    ----------
    :param names: The possible keys for dictionary elements
    :type names: list

    :param gens: Specifies, for each possible key, the set of admissible values as well as the strategy to generate them for initialization. If a discrete parameter, should be a list, otherwise a Callable object.
    :type gens: list

    :param values: the dictionary representing the candidate solution
    :type values: dict

    :param discrete: for each possible key, specifies if its admissible value set is discrete or continuous
    :type discrete: list[bool]

    :param update_distrib: a list of Callable objects, specifying, for each possible key, how to update their corresponding value in mutation.
    :type update_distrib: list[Callable]

    :param mutate: optional user-defined function for mutation
    :type mutate: Callable|None, default=None

    :param recombine: optional user-defined function for crossover
    :type recombine: Callable|None, default=None
    '''
    def __init__(self, names: list, gens: list, values: dict, discrete: list[bool], update_distrib: list[Callable],
                mutate:Callable|None=None, recombine:Callable|None=None):
        self.names = names
        self.gens = gens
        self.values = values
        self.discrete = discrete
        self.update_distrib = update_distrib
        if mutate is not None:
            self.mutate = mutate
        if recombine is not None:
            self.recombine = recombine

    def generate(names: list = [], gens: list[bool] = [], discrete: list = [], update_distrib: list[Callable] = [],
                mutate:Callable|None=None, recombine:Callable|None=None) -> DictionaryCandidate:
        if len(names) != len(gens) or len(names) != len(discrete) or len(names) != len(update_distrib):
            raise ValueError("names, gens, discrete and update_distrib must be of the same length, were %d, %d, %d, %d" % (len(names),len(gens),len(discrete),len(update_distrib)))
        
        values = {}
        for i, name in enumerate(names):
            if discrete[i]:
                values[name] = np.random.choice(gens[i])
            else:
                values[name] = gens[i]()
        return DictionaryCandidate(names, gens, values, discrete, update_distrib, mutate, recombine)
    
    def mutate(self) -> DictionaryCandidate:
        values = {}
        for i, name in enumerate(self.names):
                values[name] = self.update_distrib[i](self.values[name])
        return DictionaryCandidate(self.names, self.gens, values, self.discrete, self.update_distrib, mutate=self.mutate, recombine=self.recombine)
    
    def recombine(self, c: DictionaryCandidate) -> DictionaryCandidate:
        values = {}
        for i, name in enumerate(self.names):
            if self.discrete[i]:
                if np.random.rand() < 0.5:
                    values[name] = self.values[name]
                else:
                    values[name] = c.values[name]
            else:
                alpha = np.random.rand()
                values[name] = alpha*self.values[name] + (1-alpha)*c.values[name]

        return DictionaryCandidate(self.names, self.gens, values, self.discrete, self.update_distrib, mutate=self.mutate, recombine=self.recombine)