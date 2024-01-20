#!/usr/bin/env python3
import numpy as np
from scipy import optimize as opt

def get_error(x):
    return np.abs(x - 1)

# this function must return scalar value
def get_y(x):
    return sum(get_error(x))


x_init = [0.5, -0.4, 0.5, 0.8]
result = opt.fmin(get_y, x_init)

# todo: use this reprj error for minimisation

print(result)


