import numpy as np

def triangular(x: np.number,
               a: np.number,
               b: np.number,
               c: np.number) -> np.number:
    return max(min((x - a)/(b - a),
                   (c - x)/(c - b)), 0)

def trapezoidal(x: np.number,
                a: np.number, 
                b: np.number,
                c: np.number,
                d: np.number) -> np.number:
    return max(min((x - a)/(b - a),
                   1,
                   (d - x)/(d - c)), 0)

def linear_z_shaped(x: np.number,
                    a: np.number, 
                    b: np.number) -> np.number:
    if a == b:
        return 1 if x < a else 0
    
    if x <= a:
        return 1
    if x >= b:
        return 0
    return (b - x) / (b - a)

def linear_s_shaped(x: np.number,
                    a: np.number, 
                    b: np.number) -> np.number:
    if a == b:
        return 0 if x < a else 1

    if x < a:
        return 0
    if x > b:
        return 1
    return (x - a) / (b - a)

def gaussian(x: np.number,
             mean: np.number = 0, 
             std: np.number = 1) -> np.number:
    return np.exp(-1/2*(np.abs((x-mean)/std)**2))

def gaussian2(x: np.number,
              mean1: np.number, 
              std1: np.number,
              mean2: np.number, 
              std2: np.number) -> np.number:
    if mean1 <= x <= mean2:
        return 1
    if x < mean1 <= mean2:
        return gaussian(x, mean1, std1)
    return gaussian(x, mean2, std2)

def gbell(x: np.number,
          a: np.number, 
          b: np.number,
          c: np.number) -> np.number:
    return 1/(1+(np.abs((x - c) / a)**(2*b)))

def sigmoidal(x: np.number, 
              a: np.number, 
              c: np.number) -> np.number:
    return 1/(1 + np.exp((-a * (x - c))))

def difference_sigmoidal(x: np.number, 
                         a1: np.number, 
                         c1: np.number, 
                         a2: np.number, 
                         c2: np.number) -> np.number:
    return sigmoidal(x, a1, c1) - sigmoidal(x, a2, c2)

def product_sigmoidal(x: np.number, 
                      a1: np.number, 
                      c1: np.number, 
                      a2: np.number, 
                      c2: np.number) -> np.number:
    return sigmoidal(x, a1, c1) * sigmoidal(x, a2, c2)


def z_shaped(x: np.number, 
             a: np.number, 
             b: np.number) -> np.number:
    
    if x <= a:
        return 1

    if x > b:
        return 0

    if a < x <= (a+b)/2:
        return 1 - 2 * (((x - a)/(b-a)) ** 2)
    
    return 2 * (((x - b)/(b-a)) ** 2)


def s_shaped(x: np.number, 
             a: np.number, 
             b: np.number) -> np.number:
    
    if x <= a:
        return 0

    if x > b:
        return 1

    if a < x <= (a+b)/2:
        return 2 * (((x - a)/(b-a)) ** 2)
    
    return 1 - 2 * (((x - b)/(b-a)) ** 2)


def pi_shaped(x: np.number, 
              a: np.number,
              b: np.number,
              c: np.number,
              d: np.number) -> np.number:
    
    if x <= a:
        return 0

    if x > d:
        return 0
    
    if b <= x <= c:
        return 1
    
    if a <= x <= (a+b)/2:
        return 2 * (((x - a)/(b-a)) ** 2)
    
    if (a+b)/2 <= x <= b:
        return 1 - 2 * (((x - b)/(b-a)) ** 2)
    
    if c <= x <= (c + d)/2:
        return 1 - 2 * (((x - c)/(d-c)) ** 2)
    
    return 2 * (((x - d)/(d-c)) ** 2)

