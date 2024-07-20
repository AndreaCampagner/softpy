from typing import Callable
import matplotlib.pyplot as plt
from contextlib import contextmanager

import numpy as np

@contextmanager
def not_raises():
    try:
        yield
        
    except Exception as err:
        raise AssertionError(
            # "Did raise exception {0} when it should not!".format(
                
            # )
            repr(err)
        )

def generate_plot(callable: Callable, params: list, path: str, name: str):

    x = np.linspace(0, 10, int(21 / 1e-2))
    y = [callable(x_i, *params) for x_i in x]        

    plt.plot(x, y)
    
    plt.xlabel('Data')
    plt.ylabel('Memberships values')
    plt.title(name)
    plt.grid()
    plt.savefig(path + name + '.png')    
    plt.close()    