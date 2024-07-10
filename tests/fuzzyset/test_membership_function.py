import pytest
import sys, os
import importlib as ipl
from softpy.fuzzy.fuzzy_set_wrapper.membership_function import DiscreteMembershipFunction


class TestDescreteMembershipFunction:
    def test_creation(self):
        #DMF = ipl.import_module("../../softpy/fuzzy/fuzzyset/membership_function")
        #DMF = ipl.import_module("softpy.fuzzy.fuzzyset.membership_function")

        items = ['a', 'b', 'c']
        memberships = [0.5, 0.7, 1]
        dmf: DiscreteMembershipFunction = DiscreteMembershipFunction(items, memberships)

        assert (dmf.__dynamic == True and 
                dmf.__items == items and 
                dmf.__memberships == memberships and 
                dmf.__set == dict(zip(items, range(len(items)))))

