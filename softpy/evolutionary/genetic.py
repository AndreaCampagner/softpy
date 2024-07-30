from __future__ import annotations
import numpy as np
from softpy.evolutionary.singlestate import MetaHeuristicsAlgorithm
from joblib import Parallel, delayed
from multiprocessing import shared_memory
import pickle


def calculate_generation(iterations, sel, shared_pop, shared_fit, sizep, sizef):
    '''
    The calculate_generation function performs a generation of genetic algorithm operations, taking six parameters: iterations, sel, shared_pop, shared_fit, sizep, and sizef. 
    Initially, it loads the population and fitness data from shared memory. Then, it iterates through a specified number of iterations, each time selecting individuals using the sel function, recombining them, and applying
    mutation to create new individuals. After completing the iterations, the function closes and unlinks the shared memory to ensure proper cleanup. 
    Finally, it returns a list containing the new population
    '''

    #unpacking and loading population in shared memory      
    existing_shm_pop = shared_memory.SharedMemory(name=shared_pop)
    buffer_pop = np.ndarray((sizep,1), dtype=np.uint8, buffer=existing_shm_pop.buf)
    population = pickle.loads(buffer_pop.tobytes())

    #unpacking and loading fitness in shared memory
    existing_shm_fit = shared_memory.SharedMemory(name=shared_fit)
    buffer_fit = np.ndarray((sizef,1), dtype=np.uint8, buffer=existing_shm_fit.buf)
    fitness = pickle.loads(buffer_fit.tobytes())
        
    #calculating part of generation
    list = []
    for j in range (int(iterations)):  
        px1, current_step = sel(fitness)
        p1 = population[px1]
        px2, current_step = sel(fitness)
        p2 = population[px2]
        list.append(p1.recombine(p2).mutate())
        list.append(p2.recombine(p1).mutate())    
    
    #closing and unlinking shared mem
    existing_shm_pop.close()
    existing_shm_fit.close()
    existing_shm_pop.unlink()
    existing_shm_fit.unlink()
    
    population = None
    buffer_pop = None
    
    return list



class GeneticAlgorithm(MetaHeuristicsAlgorithm):
    '''
    A generic implementation of an evolutionary algorithm. It supports different individual candidates' representation formats and can thus be used to
    implement the traditional genetic algorithm as well as variants such as genetic programming. It requires specification of the selection and fitness
    function, of the population size as well as whether the algorithm should apply elitism (and with how many elite individuals) or not.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.
    '''
    def __init__(self, pop_size: int, candidate_type, selection_func, fitness_func, elitism=False, n_elite=1, **kwargs):
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.elitism = elitism
        self.n_elite = n_elite

    
    def fit(self, n_iters=10, keep_history=False, show_iters=False, n_cores=1): 
        if self.elitism and (self.pop_size + self.n_elite)%2 != 0:
            self.pop_size += 1

        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = np.zeros(self.population.shape)
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = np.NINF

        elab_tot = int(self.pop_size-self.n_elite)/2
        elab_thread = self.divide_number(elab_tot, n_cores)
        
        with Parallel(n_jobs = n_cores) as parallel:
            for it in range(n_iters+1):
                if show_iters:
                    print(it)
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
                
                i = self.n_elite if self.elitism else 0
                sub = self.n_elite if self.elitism else 0
                current_step = None

                idx = np.random.permutation(range(len(self.fitness)))
                self.population = self.population[idx]
                self.fitness = self.fitness[idx]

                #Creating shared memory for population
                serialized_pop = pickle.dumps(self.population)
                shm_pop = shared_memory.SharedMemory(create = True, size = len(serialized_pop))
                buffer_pop = np.ndarray((len(serialized_pop),), dtype=np.uint8, buffer=shm_pop.buf)
                buffer_pop[:] = np.frombuffer(serialized_pop, dtype=np.uint8)
                
                #Creating shared memory for fitness
                serialized_fit = pickle.dumps(self.fitness)
                shm_fit = shared_memory.SharedMemory(create = True, size = len(serialized_fit))
                buffer_fit = np.ndarray((len(serialized_fit),), dtype=np.uint8, buffer=shm_fit.buf)
                buffer_fit[:] = np.frombuffer(serialized_fit, dtype=np.uint8)

                #Using 'n_cores' cores to calculate sub-generation.  
                results = parallel(delayed(calculate_generation)(elab_thread[k], self.selection_func, shm_pop.name, shm_fit.name, len(serialized_pop), len(serialized_fit)) for k in range (int(n_cores)))

                q[sub:] = np.concatenate(results)
                
                self.population = q
                
                shm_pop.close()
                shm_fit.close()
                shm_pop.unlink()
                shm_fit.unlink()

                serialized_pop = None
                buffer_pop = None          
                

        return self.best
    


    def divide_number(self, num, parts):
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



    

class SteadyStateGeneticAlgorithm(MetaHeuristicsAlgorithm):
    '''
    A generic implementation of a steady-state evolutionary algorithm. It supports different individual candidates' representation formats and can thus be used to
    implement the traditional genetic algorithm as well as variants such as genetic programming. It requires specification of the selection and fitness
    function, as well as of the population size.
    The **kwargs argument is used to provide additional arguments for individual candidates' initialization.
    Note that the algorithm is designed to solve a maximization problem: if a minimization problem is to be solved instead, this must be taken care of in the
    fitness function.
    '''
    def __init__(self, pop_size: int, candidate_type, selection_func, fitness_func, **kwargs):
        self.pop_size = pop_size
        self.selection_func = selection_func
        self.candidate_type = candidate_type
        self.fitness_func = fitness_func
        self.kwargs = kwargs

    def fit(self, n_iters=10, keep_history=False):
        '''
        Applies the genetic algorithm for a given number of iterations. Notice that the implemented recombination is non-standard as it is called two
        times rather than only once. The algorithm allows for global state tracking in the selection function (as in stochastic universal selection) by
        using an explictly defined state tracking variable (current_step). At each iteration, two individual candidate at random are selected for being replaced by
        the generated children individual candidates.
        '''
        self.population = np.array([self.candidate_type.generate(**self.kwargs) for i in range(self.pop_size)])
        self.fitness = np.zeros(self.population.shape)
        self.best = None
        self.fitness_best = np.NINF

        if keep_history:
            self.best_h = np.empty(n_iters+1, dtype=self.candidate_type)
            self.fitness_h = np.zeros(n_iters+1)
            self.fitness_h[:] = np.NINF

        self.fitness = np.vectorize(self.fitness_func)(self.population)
        v = np.max(self.fitness)
        self.best = self.population[np.argmax(self.fitness)] if v > self.fitness_best else self.best
        self.fitness_best = v if v > self.fitness_best else self.fitness_best

        if keep_history:
            self.best_h[0] = self.best
            self.fitness_h[0] = self.fitness_best

        for it in range(1, n_iters+1):
            p1, current_step = self.population[self.selection_func(self.fitness, current_step=current_step)]
            p2, current_step = self.population[self.selection_func(self.fitness, current_step=current_step)]
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

        return self.best
    

