from __future__ import annotations
import numpy as np

def negation(membership: np.number):
    return 1 - membership

def minimum(m_left: np.number, m_right: np.number):
    return np.minimum(m_left, m_right)

def maximum(m_left: np.number, m_right: np.number):
    return np.maximum(m_left, m_right)
    
def product(m_left: np.number, m_right: np.number):
    return np.product(m_left, m_right)
    
def probsum(m_left: np.number, m_right: np.number):
    return 1 - (1-m_left)*(1-m_right)
    
def lukasiewicz(m_left: np.number, m_right: np.number):
    return np.max([0, m_left + m_right - 1])
    
def boundedsum(m_left: np.number, m_right: np.number):
    return np.min([m_left + m_right, 1])
    
def drasticproduct(m_left: np.number, m_right: np.number):
    return 1 if (m_left == 1 or m_right == 1) else 0
    
def drasticsum(m_left: np.number, m_right: np.number):
    return  m_left if m_right == 0 else m_right if m_left == 0 else 1
    