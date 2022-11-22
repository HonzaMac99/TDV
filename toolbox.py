import numpy as np
import math


class RotMatrix:
    def __init__(self, theta, dim=3):
        self.theta = theta
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        if dim == 3:
            self.Rx = np.array([[1,  0, 0], [0, c, -s], [ 0, s, c]])
            self.Ry = np.array([[c,  0, s], [0, 1,  0], [-s, 0, c]])
            self.Rz = np.array([[c, -s, 0], [s, c,  0], [ 0, 0, 1]])
        elif dim == 4:
            self.Rx = np.array([[1,  0, 0, 0], [0, c, -s, 0], [ 0, s, c, 0], [0, 0, 0, 1]])
            self.Ry = np.array([[c,  0, s, 0], [0, 1,  0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
            self.Rz = np.array([[c, -s, 0, 0], [s, c,  0, 0], [ 0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            print("Wrong RotMatrix dimension! Allowed dim is 3 or 4.")


def TMatrix(t_vec):
    T = np.eye(4)
    T[:3, 3] = t_vec.reshape(3,)
    return T

# ------- predefined toolbox functions -------------------------------------

# euclidean to projective:
# transforms an array of 2D vectors to 3D homogenous coords
def e2p(u_e):
    u_p = np.vstack([u_e, np.ones([1, u_e.shape[1]])])
    return u_p


# projective to euclidean:
# transforms an array of 3D homogenous vectors to 2D euclidean coords
def p2e(u_p):
    [rows, cols] = u_p.shape
    u_e = np.zeros([rows-1, cols])
    for i in range(cols):
        u_e[:, i] = u_p[:-1, i]/u_p[-1, i]
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


# produces a 3x3 skew-symmetric matrix from 3x1 vector
def sqc(x):
    S = np.array([[    0, -x[2],  x[1]],
                  [ x[2],     0, -x[0]],
                  [-x[1],  x[0],     0]])
    return S


# essential matrix decomposition with cheirality
def eutoRt(E, u1, u2):

    return [R, t]


# binocular reconstruction by DLT triangulation
# def pu2X(P1, P2, u1, u2):
    # return X


# sampson error on epipolar geometry
def err_F_sampson(F, u1, u2):
    e = (u2.T*F*u1)**2/(F@u1[0]**2 + F@u1[1]**2 + F.T@u2*2 + F.T@u2*2)
    e = math.sqrt(e)
    return e


# sampson correction of correspondences
# def u_correct_sampson(F, u1, u2):
    # return nu1, nu2
