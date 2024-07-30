from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from copy import deepcopy


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

    def mutate(self) -> 'BitVectorCandidate':
        #invertito il segno per la probabilita`
        #utilizzato lo XOR per invertire il bit in caso di mutazione
        mutation_arr = np.random.rand(self.size) < self.p
        mutated_candidate = np.logical_xor(self.candidate, mutation_arr)
        return BitVectorCandidate(self.size, mutated_candidate, self.p, self.uniform)
    
    def recombine(self, c: BitVectorCandidate) -> BitVectorCandidate:
        if self.uniform:
            return self.recombine_uniform(c)
        else:
            return self.recombine_single_point(c)

    def recombine_single_point(self, c: BitVectorCandidate) -> BitVectorCandidate:
        idx = np.random.randint(self.size)
        candidate = np.empty(self.size, dtype=bool)
        candidate[:idx] = self.candidate[:idx]
        candidate[idx:] = c.candidate[idx:]
        return BitVectorCandidate(self.size, candidate, self.p, self.uniform)      
    
    def recombine_uniform(self, c: BitVectorCandidate) -> BitVectorCandidate:
        mask = np.random.rand(self.size) < 0.5
        new_candidate = np.where(mask, self.candidate, c.candidate)
        return BitVectorCandidate(self.size, new_candidate, self.p, self.uniform)



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
    

class TreeCandidate(Candidate):

    class NodeCandidate:
        def __init__(self, function, children, depth = 1, parent=None, child_id=None):
            self.function = function
            self.children = children
            self.depth = depth
            self.parent = parent
            self.child_id = child_id

    def __init__(self, root: TreeCandidate.NodeCandidate, function_set, arities, max_absolute_depth, constant_generator, stop_early_prob, mutate_prob):
        self.root = root
        self.function_set = function_set
        self.arities = arities
        self.max_absolute_depth = max_absolute_depth
        self.constant_generator = constant_generator
        self.stop_early_prob = stop_early_prob
        self.mutate_prob = mutate_prob

    def generate(function_set, arities, max_depth, max_absolute_depth, constant_generator, stop_early_prob = 0.0, mutate_prob=0.2) -> TreeCandidate:
        depth = 1
        parent = None

        nodes = []

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

        return TreeCandidate(root, function_set, arities, max_absolute_depth, constant_generator, stop_early_prob, mutate_prob)


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
    def __init__(self, names, gens, values, discrete, update_distrib):
        self.names = names
        self.gens = gens
        self.values = values
        self.discrete = discrete
        self.update_distrib = update_distrib

    def generate(names, gens, discrete, update_distrib) -> DictionaryCandidate:
        values = {}
        for i, name in enumerate(names):
            if discrete[i]:
                values[name] = np.random.choice(gens[i])
            else:
                values[name] = gens[i]()
        return DictionaryCandidate(names, gens, values, discrete, update_distrib)
    
    def mutate(self) -> DictionaryCandidate:
        values = {}
        for i, name in enumerate(self.names):
                values[name] = self.update_distrib[i](self.values[name])
        return DictionaryCandidate(self.names, self.gens, values, self.discrete, self.update_distrib)
    
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

        return DictionaryCandidate(self.names, self.gens, values, self.discrete, self.update_distrib)
