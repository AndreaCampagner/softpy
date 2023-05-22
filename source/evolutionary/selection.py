import numpy as np

def tournament_selection(fitness: np.ndarray, tournament_size: int, current_step=None):
    idx = np.random.choice(range(fitness.shape[0]), tournament_size, replace=True)
    vals = fitness[idx]
    best = idx[np.argmax(vals)]
    return best, None

def fitness_prop_selection(fitness: np.ndarray, current_step=None):
    probs = fitness/np.sum(fitness)
    return np.random.choice(range(fitness.shape[0]), probs=probs), None

def stochastic_universal_selection(fitness: np.ndarray, current_step=None):
    s = np.sum(fitness)
    step_size = s/fitness.shape[0]
    probs = fitness/s

    pos = current_step + step_size 
    if pos >= s:
        pos - s

    if current_step is None:
        pos = np.random.rand()*step_size

    acc = current_step if current_step < pos else 0
    i = 0
    while acc + fitness[i] < pos:
        acc += fitness[i]
        i+=1
    return i, pos


