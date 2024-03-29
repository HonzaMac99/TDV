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


def get_line_boundaries(plot_line, img):
    plot_line = plot_line.reshape(1, 3)
    min_x = 1
    min_y = 1
    max_x = img.shape[1]-1
    max_y = img.shape[0]-1

    boundaries = np.array([[min_x, min_x, max_x, max_x],
                           [min_y, max_y, max_y, min_y]])
    boundaries_hom = e2p(boundaries)

    # get line vectors of the boundaries
    a_line = np.cross(boundaries_hom[:, 0], boundaries_hom[:, 1])
    b_line = np.cross(boundaries_hom[:, 1], boundaries_hom[:, 2])
    c_line = np.cross(boundaries_hom[:, 2], boundaries_hom[:, 3])
    d_line = np.cross(boundaries_hom[:, 3], boundaries_hom[:, 0])
    bnd_lines = [a_line, b_line, c_line, d_line]

    line_boundaries = np.zeros([2, 2])
    count = 0
    for bnd_line in bnd_lines:
        line_end = p2e((np.cross(plot_line, bnd_line).reshape(3, 1)))
        x = line_end[0]
        y = line_end[1]
        # plt.plot(x, y, "oy")
        if 1 <= int(x) <= max_x and 1 <= int(y) <= max_y:
            line_end = np.reshape(line_end, (1, 2))
            line_boundaries[:, count] = line_end
            count += 1
    return line_boundaries


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

def norm(u_p):
    return e2p(p2e(u_p))

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
                  [-x[1],  x[0],     0]], dtype='float64')
    return S


# essential matrix decomposition with cheirality
def EutoRt(E, u1, u2):
    n_points = u1.shape[1]
    u1 = np.vstack((p2e(u1), np.ones((1, n_points))))
    u2 = np.vstack((p2e(u2), np.ones((1, n_points))))

    params = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    [U, D, V_t] = np.linalg.svd(E)
    U = np.linalg.det(U) * U
    V_t = np.linalg.det(V_t) * V_t

    for i in range(4):
        alpha, beta = params[i]
        W = np.array([[0, alpha, 0],
                      [-alpha, 0, 0],
                      [0, 0, 1]])
        R = U@W@V_t
        t = -beta*U[:, 2]
        t = np.vstack(t)

        P1 = np.eye(3, 4)
        P2 = np.hstack((R, t))

        correct = True
        for j in range(n_points):
            A = np.array([P1[2]*u1[0, j] - P1[0],
                          P1[2]*u1[1, j] - P1[1],
                          P2[2]*u2[0, j] - P2[0],
                          P2[2]*u2[1, j] - P2[1]])

            [Ua, Da, Va_t] = np.linalg.svd(A)
            X = Va_t[3, :]
            X = X/X[3]  # convert to homogenous 4x1

            if (P1@X)[2] < 0 or (P2@X)[2] < 0:
                correct = False
                break

        if correct == True:
           return R, t

    # return this if there is no valid E decomposition
    R = np.eye(3)
    t = np.zeros((3, 1))  # just random t
    return R, t


# binocular reconstruction by DLT triangulation
def Pu2X(P1, P2, u1, u2):
    n_points = u1.shape[1]
    # u1 = np.vstack((p2e(u1), np.ones((1, n_points))))
    # u2 = np.vstack((p2e(u2), np.ones((1, n_points))))
    assert u1[2].max() <= 1, "u1 not normalized!"
    assert u2[2].max() <= 1, "u2 not normalized!"

    X = np.zeros((4, n_points))
    for i in range(n_points):
        A = np.array([P1[2]*u1[0, i] - P1[0],
                      P1[2]*u1[1, i] - P1[1],
                      P2[2]*u2[0, i] - P2[0],
                      P2[2]*u2[1, i] - P2[1]])
        [Ua, Da, Vat] = np.linalg.svd(A)
        x = Vat[3, :]
        x = x/x[3]  # convert to homogenous 4x1
        X[:, i] = x

    return X


# sampson error on epipolar geometry
def err_F_sampson(F, u1, u2):
    n_points = u1.shape[1]

    u1 = np.vstack((p2e(u1), np.ones((1, n_points))))
    u2 = np.vstack((p2e(u2), np.ones((1, n_points))))

    e = np.zeros(n_points)
    for i in range(n_points):
        u1_i = np.vstack(u1[:, i])
        u2_i = np.vstack(u2[:, i])
        e[i] = float(u2_i.T@F@u1_i)**2/float((F@u1_i)[0]**2 + (F@u1_i)[1]**2 + (F.T@u2_i)[0]**2 + (F.T@u2_i)[1]**2)

    return e


# sampson correction of correspondences
def u_correct_sampson(F, u1, u2):
    n_points = u1.shape[1]

    u1 = np.vstack((p2e(u1), np.ones((1, n_points))))
    u2 = np.vstack((p2e(u2), np.ones((1, n_points))))

    # init the corrected img corresp
    nu1 = u1
    nu2 = u2

    for i in range(n_points):
        u1_i = np.vstack(u1[:, i])
        u2_i = np.vstack(u2[:, i])
        e = float(u2_i.T@F@u1_i)**2/float((F@u1_i)[0]**2 + (F@u1_i)[1]**2 + (F.T@u2_i)[0]**2 + (F.T@u2_i)[1]**2)
        u12_vec = np.vstack((u1_i[0],
                             u1_i[1],
                             u2_i[0],
                             u2_i[1]))

        J_t = np.vstack((F[:, 0]@u2_i,
                         F[:, 1]@u2_i,
                         F[0, :]@u1_i,
                         F[1, :]@u1_i))

        new_u12_vec = u12_vec - e * J_t
        nu1[0:2, i] = new_u12_vec[0:2].reshape(1, 2)
        nu2[0:2, i] = new_u12_vec[2:4].reshape(1, 2)
    # print(e)

    return nu1, nu2


def u_correct_sampson_2(F, u1, u2):
    n_points = u1.shape[1]

    u1 = np.vstack((p2e(u1), np.ones((1, n_points))))
    u2 = np.vstack((p2e(u2), np.ones((1, n_points))))

    S = [[1, 0, 0], [0, 1, 0]]
    SF_t = S @ F.T
    SF = S @ F

    # init the corrected img corresp
    nu1 = u1
    nu2 = u2

    for i in range(n_points):
        u1_i = np.vstack(u1[:, i])
        u2_i = np.vstack(u2[:, i])
        e = u2_i.T @ F @ u1_i

        u12_vec = np.vstack((u1_i[0],
                             u1_i[1],
                             u2_i[0],
                             u2_i[1]))

        J_t = np.concatenate((SF_t @ u2_i, SF @ u1_i))

        new_u12_vec = u12_vec - (e * J_t) / np.sum(J_t ** 2)
        nu1[0:2, i] = new_u12_vec[0:2].reshape(1, 2)
        nu2[0:2, i] = new_u12_vec[2:4].reshape(1, 2)
    # print(e / np.sum(J_t ** 2))

    return nu1, nu2


# compute F from P1, P2 - slides 108
def get_F_from_P(P1, P2):
    Q1, q1 = P1[:, :3], P1[:, 3].reshape(3, 1)
    Q2, q2 = P2[:, :3], P2[:, 3].reshape(3, 1)

    Q12 = Q1 @ np.linalg.inv(Q2)

    F = Q12.T @ sqc(q1 - Q12 @ q2)

    return F


# compute F from P1, P2 using K, Rs and ts - slides 79
# note: sqc(-t) @ R doesn't give us the precise E!!
def get_F_from_P_2(P1, P2):
    K = np.loadtxt('scene_1/K.txt', dtype='float')
    K_inv = np.linalg.inv(K)
    R1 = (K_inv @ P1)[:, :3]
    R2 = (K_inv @ P2)[:, :3]
    R21 = R2 @ R1.T
    t21 = P2[:, 3] - R21 @ P1[:, 3]
    E = sqc(-t21) @ R21

    F = K_inv.T @ E @ K_inv
    return F


def correct_crp_sampson(P1, P2, u1, u2):
    F = get_F_from_P(P1, P2)

    # nu1, nu2 = tb.u_correct_sampson(F, u1, u2)
    nu1, nu2 = u_correct_sampson_2(F, u1, u2)

    return nu1, nu2
