import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb
import math


def get_tf_points(X, P):
    X_tf = P @ tb.e2p(X)
    X_tf_img = tb.p2e(X_tf)
    return X_tf_img


X1 = np.array([[-0.5,  0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2,   0,  0.5],
               [-0.5, -0.5, 0.5,  0.5, -0.5, -0.7, -0.9, -0.9, -0.8,  -1, -0.5],
               [   4,    4,   4,    4,    4,    4,    4,    4,    4,   4,    4]])

X2 = np.array([[-0.5,  0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2,  0,    0.5],
               [-0.5, -0.5, 0.5,  0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1,   -0.5],
               [ 4.5,  4.5, 4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5]])

K = np.array([[1000,    0, 500],
              [   0, 1000, 500],
              [   0,    0,   1]])

t2 = np.array([   [0],  [-1],   [0]])
t3 = np.array([   [0], [0.5],   [0]])
t4 = np.array([   [0],  [-3], [0.5]])
t5 = np.array([   [0],  [-5], [4.2]])
t6 = np.array([[-1.5],  [-3], [1.5]])

E3 = np.eye(3)

R0 = tb.RotMatrix(math.pi/2).Ry @ tb.RotMatrix(math.pi/2).Rx
R4 = tb.RotMatrix(0.5).Rx
R5 = tb.RotMatrix(math.pi/2).Rx
R6 = tb.RotMatrix(0.8).Rx @ tb.RotMatrix(-0.5).Ry

P1 = K @ np.eye(3, 4)
P2 = K @ np.hstack([E3, -t2])
P3 = K @ np.hstack([E3, -t3])
P4 = K @ R4 @ np.hstack([E3, -t4])
P5 = K @ R5 @ np.hstack([E3, -t5])
P6 = K @ R6 @ np.hstack([E3, -t6])

P = [P1, P2, P3, P4, P5, P6]


for i in range(len(P)):
    X1_tf = get_tf_points(X1, P[i])
    [u1, v1] = X1_tf
    X2_tf = get_tf_points(X2, P[i])
    [u2, v2] = X2_tf

    plt.plot(u2, v2, 'b-', linewidth=2)
    plt.plot([u1, u2], [v1, v2], 'k-', linewidth=2)
    plt.plot(u1, v1, 'r-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.title("Camera at position P" + str(i+1))
    plt.axis('equal') # this kind of plots should be isotropic

    plt.show()
