import matplotlib.pyplot as plt
import sys

import numpy as np
import pytest
from typing import Callable

sys.path.append(__file__ + "/..")

import softpy.fuzzy.memberships_function as mf



class TestMembershipFunction:
    __PATH: str = "./plots_memberships/"

    def __generate_plot(self, callable: Callable, params: list, name: str):

        x = np.linspace(0, 10, int(21 / 1e-2))
        y = [callable(x_i, *params) for x_i in x]        

        plt.plot(x, y)
        
        plt.xlabel('Data')
        plt.ylabel('Memberships values')
        plt.title(name)
        plt.grid()
        plt.savefig(self.__PATH + name + '.png')    
        plt.close()    


    @pytest.mark.parametrize(
            "name,memberships,args", 
            [
                ("triangular", mf.triangular, [3, 6, 8]),
                ("trapezoidal", mf.trapezoidal, [1, 5, 7, 8]),
                ("linear_z_shaped", mf.linear_z_shaped, [4,6]),
                ("linear_z_shaped_equal", mf.linear_z_shaped, [4,4]),
                ("linear_s_shaped", mf.linear_s_shaped, [4,6]),
                ("linear_s_shaped_equal", mf.linear_s_shaped, [4,4]),
                ("gaussian", mf.gaussian, [5, 2]),
                ("gaussian2", mf.gaussian2, [4, 2, 8, 1]),
                ("gbell", mf.gbell, [2, 4, 6]),
                ("sigmoidal", mf.sigmoidal, [2, 4]),
                ("difference_sigmoidal", mf.difference_sigmoidal, [5, 2, 5, 7]),
                ("product_sigmoidal", mf.product_sigmoidal, [2, 3, -5, 8]),
                ("z_shaped", mf.z_shaped, [3, 7]),
                ("s_shaped", mf.s_shaped, [1, 8]),
                ("pi_shaped", mf.pi_shaped, [1, 4, 5, 10]),
            ])
    def test_membership_function(self, name: str, memberships: Callable, args: list):
        self.__generate_plot(memberships, args, name)
        assert True