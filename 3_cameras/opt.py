#!/usr/bin/env python3
import numpy as np
from scipy import optimize as opt
from scipy.spatial.transform import Rotation as Rot

# todo: implement these

def sampson_err_f():
    pass


def optimimze_init_Rt():
    pass


def reprojection_err_f():
    pass


def optimize_new_Rt():
    pass


def get_error(x):
    return np.abs(x - 1)


# this function must return scalar value
def get_y(x):
    return sum(get_error(x))


x_init = [0.5, -0.4, 0.5, 0.8]
result = opt.fmin(get_y, x_init)
print(get_y(result))
print(result)


R = np.eye(3, 3)
t = np.zeros((3, 1))
rvec = Rot.from_matrix(R).as_rotvec()  # 3 parametry do optimalizace

# rvec = [2*np.pi, 0, 0]
R = Rot.from_rotvec(rvec).as_matrix()  # zpět na rotační matici po optimalizaci
print(R)

