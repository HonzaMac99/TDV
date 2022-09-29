import numpy as np
import math


# euclidean to projective:
# transforms an array of 2D vectors to 3D homogenous coords
def e2p(u_e):
    u_p = np.vstack([u_e, np.ones([1, u_e.shape[1]])])
    return u_p


# projective to euclidean:
# transforms an array of 3D homogenous vectors to 2D euclidean coords
def p2e(u_p):
    u_e = np.zeros([2, u_p.shape[1]])
    for i in range(u_p.shape[1]):
        u_e[:, i] = u_p[:2, i]/u_p[2, i]
    return u_e


# counts vectors length
def vlen(x):
    l = np.zeros([1, x.shape[1]])
    for i in range(x.shape[1]):  # columns
        vec_len = 0
        for j in range(x.shape[0]):  # rows
            vec_len += pow(x[j, i], 2)
        l[i] = math.sqrt(vec_len)
    return l


# def sqc(vector_in):
# def sqc(vector_in):
# def sqc(vector_in):
# def sqc(vector_in):
