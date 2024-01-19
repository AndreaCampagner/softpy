import numpy as np
from .fuzzyset import DiscreteFuzzySet

# def operation_1(A :DiscreteFuzzySet, B: DiscreteFuzzySet):
#     result = []
#     for a, b in zip(A, B):
#         if a == 0:
#             result.append(0)
#         elif b == 0:
#             result.append(0)
#         else:
#             result.append(1)
#     return result

# def operation_2(A: ShadowedSet, B: ShadowedSet):
#     result = []
#     for a, b in zip(A, B):
#         if a == 0:
#             result.append(1)
#         else:
#             result.append(b)
#     return result

# def operation_3(A, B):
#     result = []
#     for a, b in zip(A, B):
#         if a == 0:
#             result.append(1)
#         else:
#             result.append(a)
#     return result


# def minimum(a: FuzzySet, b: FuzzySet):
#     if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
#         return DiscreteFuzzyCombination(a, b, op=np.minimum)
#     elif isinstance(a, ContinuousFuzzySet) and isinstance(b, ContinuousFuzzySet):
#         return ContinuousFuzzyCombination(a,b, op=np.minimum)
#     else:
#         return FuzzyCombination(a,b, op=np.minimum)
    
# def negation(a: DiscreteFuzzySet):
#     return DiscreteFuzzySet(a.items, [1 - m for m in a.memberships])


class ShadowedSetCombination(DiscreteFuzzySet):
    def __init__(self, left, right, op=None):
        if not isinstance(left, DiscreteFuzzySet) or not isinstance(right, DiscreteFuzzySet):
            raise TypeError("Both arguments should be shadowed (discrete fuzzy) sets")
            
        if op is None:
            raise ValueError("An operation function must be provided")
            
        if not callable(op):
            raise TypeError("The provided operation must be callable")
            
        if len(left.items) != len(right.items):
            raise ValueError("The shadowed (discrete fuzzy) sets must have the same number of items")
        
        self.left = left
        self.right = right
        self.op = op

        items = left.items  # assuming left and right have the same items
        memberships = [self.op(left(v), right(v)) for v in items]

        super().__init__(items, memberships)


# def negation(a):
#     return np.array(a.items, 1 - a.memberships)


def negation(a):
    if isinstance(a, DiscreteFuzzySet):
        return DiscreteFuzzySet(a.items, [1 - m for m in a.memberships])
    else:
        raise TypeError("a should be a shadowed (discrete fuzzy) set")


# def minimum(a, b):
#     min_memberships = np.minimum(a.memberships, b.memberships)
#     return np.array(a.items, min_memberships)


def minimum(a, b):
    # if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
    #     min_memberships = np.minimum(a.memberships, b.memberships)
    #     return DiscreteFuzzySet(a.items, min_memberships)
    # else:
        return ShadowedSetCombination(a,b, op=np.minimum)


# def maximum(a, b):
#     max_memberships = np.maximum(a.memberships, b.memberships)
#     return np.array(a.items, max_memberships)


def maximum(a, b):
    # if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
    #     max_memberships = np.maximum(a.memberships, b.memberships)
    #     return DiscreteFuzzySet(a.items, max_memberships)
    # else:
        return ShadowedSetCombination(a,b, op=np.maximum)


def product(a, b):
    # if isinstance(a, DiscreteFuzzySet) and isinstance(b, DiscreteFuzzySet):
    #     product_memberships = np.multiply(a.memberships, b.memberships)
    #     return DiscreteFuzzySet(a.items, product_memberships)
    # else:
       return ShadowedSetCombination(a,b, op=np.multiply)

def probsum(a, b):
    op = lambda x, y: 1 - (1-x)*(1-y)
    return ShadowedSetCombination(a, b, op=op)


def lukasiewicz(a, b):
    op = lambda x, y: np.max([0, x + y - 1])
    return ShadowedSetCombination(a, b, op=op)


def boundedsum(a, b):
    op = lambda x, y: np.min([x + y, 1])
    return ShadowedSetCombination(a, b, op=op)


def drasticproduct(a, b):
    op = lambda x, y: 1 if (x == 1 or y == 1) else 0
    return ShadowedSetCombination(a, b, op=op)


def drasticsum(a, b):
    op = lambda x, y: x if y == 0 else y if x == 0 else 1
    return ShadowedSetCombination(a, b, op=op)