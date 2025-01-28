from __future__ import annotations
from typing import Callable, Type, Union
import numpy as np
from .singlestate import MetaHeuristicsAlgorithm
from .candidate import Candidate
from .utils import Comparable



from joblib import Parallel, delayed
from joblib.parallel import ParallelBackendBase
from multiprocessing import shared_memory
import dill

def divide_number(num: int, parts: int):
    '''
    The divide_number function takes two input arguments, num and parts, and divides the number num into a specified number of equal parts given by parts. 
    The function ensures that the division is as equal as possible, with any remainder distributed among the first few parts.
    '''
    part_size = num // parts

    remainder = num % parts

    parts_list = [part_size] * parts

    for i in range(int(remainder)):
        parts_list[i] += 1

    return parts_list


def calculate_generation(iterations: int, sel: Callable, shared_pop, shared_fit, sizep: int, sizef: int):
    '''
    The calculate_generation function performs a generation of genetic algorithm operations, taking six parameters: iterations, sel, shared_pop, shared_fit, sizep, and sizef. 
    Initially, it loads the population and fitness data from shared memory. Then, it iterates through a specified number of iterations, each time selecting individuals using the sel function, recombining them, and applying
    mutation to create new individuals. After completing the iterations, the function closes and unlinks the shared memory to ensure proper cleanup. 
    Finally, it returns a list containing the new population
    '''

    #unpacking and loading population in shared memory      
    existing_shm_pop = shared_memory.SharedMemory(name=shared_pop)
    buffer_pop = np.ndarray((sizep,1), dtype=np.uint8, buffer=existing_shm_pop.buf)
    population = dill.loads(buffer_pop.tobytes())

    #unpacking and loading fitness in shared memory
    existing_shm_fit = shared_memory.SharedMemory(name=shared_fit)
    buffer_fit = np.ndarray((sizef,1), dtype=np.uint8, buffer=existing_shm_fit.buf)
    fitness = dill.loads(buffer_fit.tobytes())

    #calculating part of generation
    lst = np.empty(int(iterations)*2, dtype=type(population[0]))
    #list = []
    current_step=None
    for j in range(int(iterations)):  
        px1, current_step = sel(fitness, current_step=current_step)
        p1 = population[px1]
        px2, current_step = sel(fitness, current_step=current_step)
        p2 = population[px2]
        lst[2*j] = p1.recombine(p2).mutate()
        lst[2*j+1] = p2.recombine(p1).mutate()    

    #closing and unlinking shared mem
    existing_shm_pop.close()
    existing_shm_fit.close()
    #existing_shm_pop.unlink()
    #existing_shm_fit.unlink()

    return list(lst)



class GeneticAlgorithm(MetaHeuristicsAlgorithm):
    '''
    A generic implementation of an evolutionary algorithm. It supports different individual candidates' representation formats and can thus be used to
    implement the traditional genetic algorithm as well as variants such as genetic programming. It requires specification of the selection and fitness
    function, of the population size as well as whether the algorithm should apply elitism (and with how many elite individuals) or not.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    The implementation provides support for multi-core execution through joblib: this is specified through the n_jobs and backend parameters.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.

    Parameters
    ----------
    :param candidate_type: the class specifying the representation of candidate solutions
    :type candidate_type: Type[Candidate]

    :param fitness_func: the fitness function
    :type fitness_func: Callable

    :param pop_size: the size of the population
    :type pop_size: int, default=1

    :param selection_func: the function used to perform selection
    :type selection_func: Callable

    :param elitism: specifies whether to use elitism
    :type elitism: bool, default=False

    :param n_elite: the number of elite individuals to keep in the population. Only used if elitism=True
    :type n_elite: int, default=1
    '''
    def __init__(self, candidate_type: Type[Candidate], fitness_func: Callable, pop_size: int, selection_func: Callable,
                elitism: bool = False, n_elite: int = 1, **kwargs):
        if pop_size < 1:
            raise ValueError("pop_size must be an int greater than 0, was %d" % pop_size)
        
        if elitism and n_elite < 1:
            raise ValueError("n_elite must be an int greater than 0 when elite=True, was %d" % n_elite)
        
        if not isinstance(fitness_func,Callable):
            raise ValueError("fitness_func must be Callable, was %s" % type(fitness_func))
        
        if not isinstance(selection_func,Callable):
            raise ValueError("selection_func must be Callable, was %s" % type(selection_func))
        
        if not issubclass(candidate_type, Candidate):
            raise ValueError("candidate_type must be a subclass of Candidate")
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.elitism = elitism
        self.n_elite = n_elite
        self.is_fitted = False

    
    def fit(self, n_iters: int = 10, keep_history: bool = False, show_iters: bool = False, n_jobs: int = 1, backend: Union[ParallelBackendBase,str,None] = "loky"):
        '''
        Applies the genetic algorithm for a given number of iterations. Notice that the implemented recombination is non-standard as it is called two
        times rather than only once. The algorithm allows for global state tracking in the selection function (as in stochastic universal selection) by
        using an explictly defined state tracking variable (current_step). The individual candidates are randomly permuted at each iteration to avoid
        ordering bias. The population is entirely replaced at each iteration (unless elitism is used).
        '''
        self.is_fitted = False
        if n_iters < 1:
            raise ValueError("n_iters must be larger than 0, was %d" % n_iters)
        
        if backend != None and n_jobs < 1:
            raise ValueError("n_jobs must be an int larger than 0 when backend != None, was %d" % n_jobs)


        if not self.elitism:
            self.n_elite = 0
        if (self.elitism and (self.pop_size + self.n_elite)%2 != 0) or (not self.elitism and self.pop_size % 2 != 0):
            self.pop_size += 1

        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = [None] * self.population.shape[0]
        self.best = None
        self.fitness_best = -np.inf

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = -np.inf

        elab_tot = int(self.pop_size-self.n_elite)/2
        elab_thread = divide_number(elab_tot, n_jobs)
        
        
        shm_pop = None
        shm_fit = None

        with Parallel(n_jobs = n_jobs, backend=backend) as parallel:
            for it in range(n_iters+1):
                if show_iters:
                    print(it)

                idx = np.random.permutation(range(len(self.fitness)))
                self.population = self.population[idx]
                self.fitness = np.vectorize(self.fitness_func)(self.population) 
                v = np.max(self.fitness)
                self.best = self.population[np.argmax(self.fitness)] if v > self.fitness_best else self.best
                self.fitness_best = v if v > self.fitness_best else self.fitness_best

                if keep_history:
                    self.best_h[it] = self.best
                    self.fitness_h[it] = self.fitness_best

                q = np.empty(self.pop_size, dtype=self.candidate_type)
                if self.elitism:
                    q[:self.n_elite] = self.population[np.argsort(self.fitness)[::-1][:self.n_elite]]

                sub = self.n_elite if self.elitism else 0

                #Creating shared memory for population
                serialized_pop = dill.dumps(self.population)
                shm_pop = shared_memory.SharedMemory(create = True, size = len(serialized_pop))
                buffer_pop = np.ndarray((len(serialized_pop),), dtype=np.uint8, buffer=shm_pop.buf)
                buffer_pop[:] = np.frombuffer(serialized_pop, dtype=np.uint8)

                #Creating shared memory for fitness
                serialized_fit = dill.dumps(self.fitness)
                shm_fit = shared_memory.SharedMemory(create = True, size = len(serialized_fit))
                buffer_fit = np.ndarray((len(serialized_fit),), dtype=np.uint8, buffer=shm_fit.buf)
                buffer_fit[:] = np.frombuffer(serialized_fit, dtype=np.uint8)

                #Using 'n_jobs' cores to calculate sub-generation.  
                results = parallel(delayed(calculate_generation)(elab_thread[k], self.selection_func, shm_pop.name, shm_fit.name, len(serialized_pop), len(serialized_fit)) for k in range (int(n_jobs)))

                q[sub:] = np.concatenate(results)

                self.population = q

                shm_pop.unlink()
                shm_fit.unlink()
                shm_pop.close()
                shm_fit.close()
                

                serialized_pop = None
                buffer_pop = None

        self.is_fitted = True
        return self.best
    

class SteadyStateGeneticAlgorithm(MetaHeuristicsAlgorithm):
    '''
    A generic implementation of a steady-state evolutionary algorithm. It supports different individual candidates' representation formats and can thus be used to
    implement the traditional genetic algorithm as well as variants such as genetic programming. It requires specification of the selection and fitness
    function, as well as of the population size.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.

    Parameters
    ----------
    :param candidate_type: the class specifying the representation of candidate solutions
    :type candidate_type: Type[Candidate]

    :param fitness_func: the fitness function
    :type fitness_func: Callable

    :param pop_size: the size of the population
    :type pop_size: int, default=1

    :param selection_func: the function used to perform selection
    :type selection_func: Callable
    '''
    def __init__(self, candidate_type: Type[Candidate], fitness_func: Callable, pop_size: int,  selection_func: Callable, **kwargs):
        if pop_size < 1:
            raise ValueError("pop_size must be an int greater than 0, was %d" % pop_size)
        
        if not isinstance(fitness_func,Callable):
            raise ValueError("fitness_func must be Callable, was %s" % type(fitness_func))
        
        if not isinstance(selection_func,Callable):
            raise ValueError("selection_func must be Callable, was %s" % type(selection_func))
        
        if not issubclass(candidate_type, Candidate):
            raise ValueError("candidate_type must be a subclass of Candidate")
        
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.is_fitted = False


    def fit(self, n_iters: int = 10, keep_history: bool =False):
        '''
        Applies the genetic algorithm for a given number of iterations. Notice that the implemented recombination is non-standard as it is called two
        times rather than only once. The algorithm allows for global state tracking in the selection function (as in stochastic universal selection) by
        using an explictly defined state tracking variable (current_step). At each iteration, two individual candidate at random are selected for being replaced by
        the generated children individual candidates.
        '''
        self.is_fitted = False
        if n_iters < 1:
            raise ValueError("n_iters must be larger than 0, was %d" % n_iters)
        
        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = [None] * self.population.shape[0]
        self.best = None
        self.fitness_best = -np.inf

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = -np.inf

        self.fitness = np.vectorize(self.fitness_func)(self.population)
        v = np.max(self.fitness)
        self.best = self.population[np.argmax(self.fitness)] if v > self.fitness_best else self.best
        self.fitness_best = v if v > self.fitness_best else self.fitness_best

        if keep_history:
            self.best_h[0] = self.best
            self.fitness_h[0] = self.fitness_best

        for it in range(1, n_iters+1):
            idx = np.random.permutation(range(len(self.fitness)))
            self.population = self.population[idx]
            self.fitness = self.fitness[idx]
            current_step=None
            px1, current_step = self.selection_func(self.fitness, current_step=current_step)
            p1 = self.population[px1]
            px2, current_step = self.selection_func(self.fitness, current_step=current_step)
            p2 = self.population[px2]
            c1 = p1.recombine(p2).mutate()
            c2 = p2.recombine(p1).mutate()

            f1 = self.fitness_func(c1)
            f2 = self.fitness_func(c2)
            if f1 > self.fitness_best:
                self.best = c1
                self.fitness_best = f1
            if f2 > self.fitness_best:
                self.best = c2
                self.fitness_best = f2

            if keep_history:
                self.best_h[it] = self.best
                self.fitness_h[it] = self.fitness_best 

            die = np.random.choice(range(self.fitness.shape[0]), 2, replace=False)
            self.population[die[0]] = c1
            self.population[die[1]] = c2

        self.is_fitted = True
        return self.best
    



