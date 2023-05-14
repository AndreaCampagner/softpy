import numpy as np

def tournament_selection(fitness: np.ndarray, tournament_size: int):
    idx = np.random.choice(range(fitness.shape[0]), tournament_size, replace=True)
    vals = fitness[idx]
    best = idx[np.argmax(vals)]
    return best

def fitness_prop_selection(fitness: np.ndarray):
    probs = fitness/np.sum(fitness)
    return np.random.choice(range(fitness.shape[0]), probs=probs)
