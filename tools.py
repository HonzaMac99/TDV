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
    vec_len = np.zeros(x.shape[1])
    for i in range(x.shape[1]):  # columns
        vec_len[i] = math.sqrt(sum(x[:, i]**2))
    return vec_len


# produces a 3x3 skew-symmetric matrix from 3x1 vector
def sqc(x):
    S = np.array([[    0, -x[2],  x[1]],
                  [ x[2],     0, -x[0]],
                  [-x[1],  x[0],     0]])
    return S


# essential matrix decomposition with cheirality
def EutoRt(E, u1, u2):
    params = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    n_points = u1.shape[1]
    [U, D, V] = np.linalg.svd(E)

    for i in range(4):
        alpha, beta = params[i]
        W = np.array([[0, alpha, 0],
                      [-alpha, 0, 0],
                      [0, 0, 1]])
        R = U@W@V.T
        t = -beta*U[:, 2]
        t = np.vstack(t)

        P1 = np.eye(3, 4)
        P2 = np.hstack((R, t))

        correct = True
        for j in range(n_points):
            A = np.array([[P1[0] - P1[2]*u1[0, j]],
                          [P1[1] - P1[2]*u1[1, j]],
                          [P2[0] - P2[2]*u2[0, j]],
                          [P2[1] - P2[2]*u2[1, j]]])
            [Ua, Da, Va] = np.linalg.svd(A)
            X = Va[:, 3]
            X = X/X[3]  # convert to homogenous 4x1

            if ((P1@X)[2] < 0).any() or ((P2@X)[2].any() < 0).any():
                correct = False
                break

        if correct == True:
           return R, t

    # return this if there is no valid E decomposition
    R = np.array([])
    t = np.vstack([1, 0, 1])  # just random t
    return R, t


# binocular reconstruction by DLT triangulation
def Pu2X(P1, P2, u1, u2):
    #return X
    return []


# sampson error on epipolar geometry
def err_F_sampson(F, u1, u2):
    e = (u2.T*F*u1)**2/(F@u1[0]**2 + F@u1[1]**2 + F.T@u2*2 + F.T@u2*2)
    e = math.sqrt(e)
    return e


# sampson correction of correspondences
def u_correct_sampson(F, u1, u2):
    #return nu1, nu2
    return []
