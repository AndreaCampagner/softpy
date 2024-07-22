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

def generate_plot(callable: Callable, 
                  params: list, 
                  path: str,
                  name: str, 
                  additional_call: dict[str, Callable] = None):

    print(additional_call)
    x = np.linspace(0, 10, int(21 / 1e-2))
    y = [callable(x_i, *params) for x_i in x]
    
    add_y = []
    
    if additional_call != None:
        for _, call in additional_call.items():
            add_y.append([call(x_i) for x_i in x])
        for y_i, name_f in zip(add_y, additional_call.keys()):
            plt.plot(x, y_i, label=name_f)

    plt.plot(x, y, label='ris')

    plt.xlabel('Data')
    plt.ylabel('Memberships values')
    plt.title(name)
    plt.legend()
    plt.ylim(-0.1, 1.5)
    plt.grid()
    plt.savefig(path + name + '.png')
    plt.close()    