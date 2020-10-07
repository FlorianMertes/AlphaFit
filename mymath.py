import numba as nb
from math import *
import numpy as np

lim = np.finfo(np.float64).epsneg

@nb.jit('float64(float64)',nopython=True,nogil=True,fastmath=True)
def erfc(x):
    val = erfc(x)
    if isnan(val):
        return 0.
    else:
        return val

@nb.jit('float64(float64)',nopython=True,nogil=True,fastmath=True)
def exp(x):
    return exp(x)
