import numpy as np
from .fuzzyset import DiscreteFuzzySet


def pedrycz_optimization(fuzzy_set, alpha_range, beta_range):
    min_difference = float('inf')
    best_alpha = None

    for alpha in alpha_range:
        for beta in beta_range:
            r = np.sum([mu for mu in fuzzy_set.memberships if mu < alpha])
            e = np.sum([(1 - mu) for mu in fuzzy_set.memberships if mu > beta])
            b = np.sum([1 for mu in fuzzy_set.memberships if alpha <= mu <= beta])

            # r = np.sum([mu for mu in fuzzy_set.memberships if fuzzy_set(mu) < alpha])
            # e = np.sum([(1 - mu) for mu in fuzzy_set.memberships if fuzzy_set(mu) > beta])
            # b = np.sum([1 for mu in fuzzy_set.memberships if alpha <= fuzzy_set(mu) <= beta])
            
            difference = abs(r + e - b)
            
            if difference < min_difference:
                min_difference = difference
                best_alpha = alpha
    
    return best_alpha


# def entropy_optimization(fuzzy_set, alpha):
#     phi = lambda x: -x * np.log(x) - (1 - x) * np.log(1 - x)
#     r = np.sum([phi(membership) for membership in fuzzy_set.memberships if membership < alpha])
#     e = np.sum([phi(membership) for membership in fuzzy_set.memberships if membership >= alpha])
    
#     return abs(r - e)


def phi(x):
    if x == 0 or x == 1:
        return 0  # Logarithm of 0 or 1 is 0
    else:
        return -x * np.log(x) - (1 - x) * np.log(1 - x)


def entropy_optimization(fuzzy_set, alpha_range):
    min_difference = float('inf')
    best_alpha = None
    # phi = lambda x: -x * np.log(x) - (1 - x) * np.log(1 - x)

    for alpha in alpha_range:
        phi_alpha = np.sum([phi(membership) for membership in fuzzy_set.memberships if membership < alpha])
        phi_beta = np.sum([phi(membership) for membership in fuzzy_set.memberships if membership >= alpha])
        # phi_alpha = np.sum([-mu * np.log(mu) - (1 - mu) * np.log(1 - mu) for mu in fuzzy_set.memberships if mu < alpha])
        # phi_beta = np.sum([mu * np.log(mu) + (1 - mu) * np.log(1 - mu) for mu in fuzzy_set.memberships if mu >= alpha])
        
        difference = abs(phi_alpha - phi_beta)
        
        if difference < min_difference:
            min_difference = difference
            best_alpha = alpha
    
    return best_alpha


def error_optimization(fuzzy_set, alpha_range):
    min_error = float('inf')
    best_alpha = None

    for alpha in alpha_range:
        # reduction_error = np.sum([mu for mu in fuzzy_set.memberships if mu < alpha])
        # fixing_error = np.sum([abs(mu - 0.5) for mu in fuzzy_set.memberships if alpha <= mu <= beta])
        # elevation_error = np.sum([1 - mu for mu in fuzzy_set.memberships if mu > beta])

        # red_error = np.sum(fuzzy_set.memberships[fuzzy_set.items <= alpha])
        # elv_error = np.sum(1 - fuzzy_set.memberships[fuzzy_set.items > alpha])
        # shd_error = np.sum(np.abs(0.5 - fuzzy_set.memberships[fuzzy_set.items > alpha]))

        # red_error = np.sum([mu for mu in fuzzy_set.memberships if mu < alpha])
        # elv_error = np.sum([1 - mu for mu in fuzzy_set.memberships if mu >= alpha])
        # shd_error = np.sum([abs(0.5 - mu) for mu in fuzzy_set.memberships if mu >= alpha])
        
        red_error = np.sum(fuzzy_set.memberships[fuzzy_set.memberships <= alpha])
        elv_error = np.sum(1 - fuzzy_set.memberships[fuzzy_set.memberships > alpha])
        shd_error = np.sum(np.abs(0.5 - fuzzy_set.memberships[fuzzy_set.memberships > alpha]))

        # total_error = reduction_error + fixing_error + elevation_error
        total_error = red_error + elv_error + shd_error
        
        if total_error < min_error:
            min_error = total_error
            best_alpha = alpha
    
    return best_alpha


# def alternative_uncertainty_balance(alpha):
#     ur = np.sum([phi(membership) for membership in fuzzy_set.memberships if membership < alpha])
#     ui = len(fuzzy_set.items) - np.sum([phi(membership) for membership in fuzzy_set.memberships if alpha <= membership <= beta])
    
#     return abs(ur - ui)


def alt_uncertainty_optimization(fuzzy_set, alpha_range):
    min_difference = float('inf')
    best_alpha = None

    for alpha in alpha_range:
        # elv_phi = fuzzy_set.memberships[fuzzy_set.items > alpha]
        # red_phi = fuzzy_set.memberships[fuzzy_set.items <= alpha]
        # shd_phi = 0.5 - fuzzy_set.memberships[(fuzzy_set.items > alpha) & (fuzzy_set.items <= 0.5)]
        # shd_cardinality = len(fuzzy_set.items[(fuzzy_set.items > alpha) & (fuzzy_set.items <= 0.5)])

        # elv_phi = np.array(fuzzy_set)[np.array(fuzzy_set) > alpha]
        # red_phi = np.array(fuzzy_set)[np.array(fuzzy_set) <= alpha]
        # shd_phi = 0.5 - np.array(fuzzy_set)[(np.array(fuzzy_set) > alpha) & (np.array(fuzzy_set) <= 0.5)]
        # shd_cardinality = len(np.array(fuzzy_set)[(np.array(fuzzy_set) > alpha) & (np.array(fuzzy_set) <= 0.5)])

        elv_phi = fuzzy_set.memberships[fuzzy_set.memberships > alpha]
        red_phi = fuzzy_set.memberships[fuzzy_set.memberships <= alpha]
        shd_phi = 0.5 - fuzzy_set.memberships[(fuzzy_set.memberships > alpha) & (fuzzy_set.memberships <= 0.5)]
        shd_cardinality = len(fuzzy_set.memberships[(fuzzy_set.memberships > alpha) & (fuzzy_set.memberships <= 0.5)])

        result = np.abs(np.sum(elv_phi) + np.sum(red_phi) + np.sum(shd_phi) - shd_cardinality)

        if result < min_difference:
            min_difference = result
            best_alpha = alpha
        
    return best_alpha


def calculate_uncertainty_alpha(fuzzy_set, alpha_range, method, beta_range = np.linspace(0, 1, num=100)):
    if method == 'pedrycz_opt':
        return pedrycz_optimization(fuzzy_set, alpha_range, beta_range)
    elif method == 'entropy_opt':
        return entropy_optimization(fuzzy_set, alpha_range)
    elif method == 'error_opt':
        return error_optimization(fuzzy_set, alpha_range)
    elif method == 'alt_uncertainty_opt':
        return alt_uncertainty_optimization(fuzzy_set, alpha_range)


def calculate_gradualness(item_membership):
    return np.abs(1 - item_membership)


def calculate_sharpness(fuzzy_set):
    sharpness = sum(abs(membership - 0.5) for membership in fuzzy_set.memberships)
    return sharpness


def calculate_gradualness_balance(fuzzy_set, alpha):
    Elv = []
    Red = []
    Shd = []

    for i, element in enumerate(fuzzy_set.items):
        membership = fuzzy_set.memberships[i]

        if membership >= 1 - alpha:
            Elv.append(element)
        elif membership <= alpha:
            Red.append(element)
        else:
            Shd.append(element)

    # Use np.where to find the indices of elements in Shd, Red, and Elv
    Shd_indices = np.where(np.isin(fuzzy_set.items, Shd))
    Red_indices = np.where(np.isin(fuzzy_set.items, Red))
    Elv_indices = np.where(np.isin(fuzzy_set.items, Elv))

    # Use the indices to calculate a, b, c, and d
    a = np.sum(fuzzy_set.memberships[Red_indices])
    b = np.sum(fuzzy_set.memberships[Shd_indices] - 0.5)
    c = np.sum(1 - fuzzy_set.memberships[Elv_indices])
    d = np.sum(0.5 - fuzzy_set.memberships[Shd_indices])

    return np.abs(a + b - c - d)


def calculate_sharpness_balance(fuzzy_set, alpha):
    Elv = []
    Red = []
    Shd = []

    for i, element in enumerate(fuzzy_set.items):
        membership = fuzzy_set.memberships[i]

        if membership >= 1 - alpha:
            Elv.append(element)
        elif membership <= alpha:
            Red.append(element)
        else:
            Shd.append(element)

    # Use np.where to find the indices of elements in Shd, Red, and Elv
    Shd_indices = np.where(np.isin(fuzzy_set.items, Shd))
    Red_indices = np.where(np.isin(fuzzy_set.items, Red))
    Elv_indices = np.where(np.isin(fuzzy_set.items, Elv))

    g = np.sum(np.abs(fuzzy_set.memberships[Shd_indices] - 0.5))
    h = np.sum(0.5 - np.abs(fuzzy_set.memberships[Red_indices]))
    f = np.sum(0.5 - np.abs(fuzzy_set.memberships[Elv_indices]))

    return abs(g - h - f)


def calculate_classification_score(fuzzy_set, alpha):
    Elv = []
    Red = []
    Shd = []

    for i, element in enumerate(fuzzy_set.items):
        membership = fuzzy_set.memberships[i]

        if membership >= 1 - alpha:
            Elv.append(element)
        elif membership <= alpha:
            Red.append(element)
        else:
            Shd.append(element)

    # Use np.where to find the indices of elements in Shd, Red, and Elv
    Shd_indices = np.where(np.isin(fuzzy_set.items, Shd))
    Red_indices = np.where(np.isin(fuzzy_set.items, Red))
    Elv_indices = np.where(np.isin(fuzzy_set.items, Elv))

    # Use the indices to calculate the score
    score = np.sum(fuzzy_set.memberships[Elv_indices])
    score += np.sum(fuzzy_set.memberships[Red_indices])
    score -= np.sum(fuzzy_set.memberships[Shd_indices])

    return score


def shadowed_set_induction(fuzzy_set, alpha_range, method):
    if method == 'sharpness_balance':
        min_difference = np.inf
        best_alpha = None

        for alpha in alpha_range:
            difference = calculate_sharpness_balance(fuzzy_set, alpha)

            if difference < min_difference:
                min_difference = difference
                best_alpha = alpha

        shadowed_set = DiscreteFuzzySet(fuzzy_set.items, fuzzy_set.memberships.copy())
        for i, element in enumerate(shadowed_set.items):
            membership = shadowed_set.memberships[i]

            if membership >= 1 - best_alpha:
                shadowed_set.memberships[i] = 1
            elif membership <= best_alpha:
                shadowed_set.memberships[i] = 0.5
            else:
                shadowed_set.memberships[i] = 0

        return shadowed_set, best_alpha
    
    elif method == 'tradeoff':
        max_score = float('-inf')
        best_alpha = None

        for alpha in alpha_range:
            score = calculate_classification_score(fuzzy_set, alpha)

            if score > max_score:
                max_score = score
                best_alpha = alpha

        shadowed_set = DiscreteFuzzySet(fuzzy_set.items, fuzzy_set.memberships.copy())
        for i, element in enumerate(shadowed_set.items):
            membership = shadowed_set.memberships[i]

            if membership >= 1 - best_alpha:
                shadowed_set.memberships[i] = 1
            elif membership <= best_alpha:
                shadowed_set.memberships[i] = 0
            else:
                shadowed_set.memberships[i] = 0.5

        return shadowed_set, best_alpha
    
    elif method == 'gradualness_balance':
        min_diff = float('inf')
        best_alpha = None

        for alpha in alpha_range:
            diff = calculate_gradualness_balance(fuzzy_set, alpha)

            if diff < min_diff:
                min_diff = diff
                best_alpha = alpha

        shadowed_set = DiscreteFuzzySet(fuzzy_set.items, fuzzy_set.memberships.copy())
        for i, element in enumerate(shadowed_set.items):
            membership = shadowed_set.memberships[i]

            if membership <= best_alpha:
                shadowed_set.memberships[i] = 0
            elif 0.5 <= fuzzy_set.memberships[i] <= 1 - best_alpha:
                shadowed_set.memberships[i] = 0.5
            else:
                shadowed_set.memberships[i] = 1

        return shadowed_set, best_alpha

    else:
        raise ValueError('Invalid method specified')


# ///////////////////////////////// Sharpness balance START //////////////////////////////////////

# import numpy as np
# import matplotlib.pyplot as plt

# class FuzzySet:
#     def __init__(self, elements, membership):
#         self.elements = elements
#         self.membership = membership

# def calculate_sharpness(fuzzy_set):
#     sharpness = sum(abs(membership - 0.5) for membership in fuzzy_set.membership)
#     return sharpness

# def calculate_sharpness_balance(fuzzy_set, alpha):
#     Elv = []
#     Red = []
#     Shd = []

#     for i, element in enumerate(fuzzy_set.elements):
#         membership = fuzzy_set.membership[i]

#         if membership >= 1 - alpha:
#             Elv.append(element)
#         elif membership <= alpha:
#             Red.append(element)
#         else:
#             Shd.append(element)

#     g = sum(abs(fuzzy_set.membership[fuzzy_set.elements.index(element)] - 0.5) for element in Shd)
#     h = sum(0.5 - abs(fuzzy_set.membership[fuzzy_set.elements.index(element)]) for element in Red)
#     f = sum(0.5 - abs(fuzzy_set.membership[fuzzy_set.elements.index(element)]) for element in Elv)

#     return abs(g - h - f)

# def shadowed_set_induction_sharpness_balance(fuzzy_set, alpha_range):
#     min_difference = np.inf
#     best_alpha = None

#     for alpha in alpha_range:
#         difference = calculate_sharpness_balance(fuzzy_set, alpha)

#         if difference < min_difference:
#             min_difference = difference
#             best_alpha = alpha

#     shadowed_set = FuzzySet(fuzzy_set.elements, fuzzy_set.membership.copy())
#     for i, element in enumerate(shadowed_set.elements):
#         membership = shadowed_set.membership[i]

#         if membership >= 1 - best_alpha:
#             shadowed_set.membership[i] = 1
#         elif membership <= best_alpha:
#             shadowed_set.membership[i] = 0.5

#     return shadowed_set, best_alpha

# ////////////////////////////////////////////////////////////////////////////////////////////


# ///////////////////////////////// Gradualness balance START //////////////////////////////////////

# import numpy as np
# import matplotlib.pyplot as plt

# class FuzzySet:
#     def __init__(self, elements, membership):
#         self.elements = elements
#         self.membership = membership

# def calculate_gradualness(element_membership):
#     return np.abs(1 - element_membership)

# def calculate_gradualness_balance(fuzzy_set, alpha):
#     Elv = []
#     Red = []
#     Shd = []

#     for i, element in enumerate(fuzzy_set.elements):
#         membership = fuzzy_set.membership[i]

#         if membership >= 1 - alpha:
#             Elv.append(element)
#         elif membership <= alpha:
#             Red.append(element)
#         else:
#             Shd.append(element)

#     a = sum(fuzzy_set.membership[fuzzy_set.elements.index(element)] for element in Red)
#     b = sum((fuzzy_set.membership[fuzzy_set.elements.index(element)] - 0.5) for element in Shd)
#     c = sum(1 - fuzzy_set.membership[fuzzy_set.elements.index(element)] for element in Elv)
#     d = sum((0.5 - fuzzy_set.membership[fuzzy_set.elements.index(element)]) for element in Shd)

#     return abs(a + b - c - d)

# def shadowed_set_induction_grad_balance(fuzzy_set, alpha_range):
#     min_diff = float('inf')
#     best_alpha = None

#     for alpha in alpha_range:
#         diff = calculate_gradualness_balance(fuzzy_set, alpha)

#         if diff < min_diff:
#             min_diff = diff
#             best_alpha = alpha

#     shadowed_set = FuzzySet(fuzzy_set.elements, fuzzy_set.membership.copy())
#     for i, element in enumerate(shadowed_set.elements):
#         membership = shadowed_set.membership[i]

#         if membership <= best_alpha:
#             shadowed_set.membership[i] = 0
#         elif 0.5 <= fuzzy_set.membership[i] <= 1 - best_alpha:
#             shadowed_set.membership[i] = 0.5

#     return shadowed_set, best_alpha

# ////////////////////////////////////////////////////////////////////////////////////////////

# ///////////////////////////////// Tradeoff START //////////////////////////////////////

# import numpy as np
# import matplotlib.pyplot as plt

# class FuzzySet:
#     def __init__(self, elements, membership):
#         self.elements = elements
#         self.membership = membership

# def calculate_classification_score(fuzzy_set, alpha):
#     Elv = []
#     Red = []
#     Shd = []

#     for i, element in enumerate(fuzzy_set.elements):
#         membership = fuzzy_set.membership[i]

#         if membership >= 1 - alpha:
#             Elv.append(element)
#         elif membership <= alpha:
#             Red.append(element)
#         else:
#             Shd.append(element)

#     score = sum(fuzzy_set.membership[fuzzy_set.elements.index(element)] for element in Elv)
#     score += sum(fuzzy_set.membership[fuzzy_set.elements.index(element)] for element in Red)
#     score -= sum(fuzzy_set.membership[fuzzy_set.elements.index(element)] for element in Shd)

#     return score

# def shadowed_set_induction_tradeoff(fuzzy_set, alpha_range):
#     max_score = float('-inf')
#     best_alpha = None

#     for alpha in alpha_range:
#         score = calculate_classification_score(fuzzy_set, alpha)

#         if score > max_score:
#             max_score = score
#             best_alpha = alpha

#     shadowed_set = FuzzySet(fuzzy_set.elements, fuzzy_set.membership.copy())
#     for i, element in enumerate(shadowed_set.elements):
#         membership = shadowed_set.membership[i]

#         if membership >= 1 - best_alpha:
#             shadowed_set.membership[i] = 1
#         elif membership <= best_alpha:
#             shadowed_set.membership[i] = 0

#     return shadowed_set, best_alpha

# ////////////////////////////////////////////////////////////////////////////////////////////


# ///////////////////////////////// Sharpness balance EXTRA //////////////////////////////////////

# import numpy as np

# class FuzzySet:
#     def __init__(self, values):
#         self.values = values

#     def __repr__(self):
#         return f"Values: {self.values}"


# def calculate_sharpness_balance(fuzzy_set):
#     num_elements = len(fuzzy_set.values)
#     sharpness_balance = 0.0

#     for i in range(num_elements):
#         sharpness_balance += abs(fuzzy_set.values[i] - 0.5)

#     sharpness_balance /= num_elements

#     return sharpness_balance


# def induce_shadowed_set_by_sharpness_balance(fuzzy_set, threshold):
#     shadow = np.zeros(len(fuzzy_set.values))

#     for i in range(len(fuzzy_set.values)):

#         if fuzzy_set.values[i] < threshold:
#             shadow[i] = fuzzy_set.values[i]
#         else:
#             shadow[i] = 1 - fuzzy_set.values[i]

#     return FuzzySet(shadow)


# # Example usage
# fuzzy_values = [0.2, 0.7, 0.4, 0.6, 0.3]
# fuzzy_set = FuzzySet(fuzzy_values)

# sharpness_balance = calculate_sharpness_balance(fuzzy_set)
# threshold = sharpness_balance

# shadowed_set = induce_shadowed_set_by_sharpness_balance(fuzzy_set, threshold)

# print("Original Fuzzy Set:")
# print(fuzzy_set)

# print("Induced Shadowed Set:")
# print(shadowed_set)

# ////////////////////////////////////////////////////////////////////////////////////////////