#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tools as tb
import math
import sys

sys.path.append('python')
import p5


img1 = mpimg.imread('cam1.jpg')
img2 = mpimg.imread('cam2.jpg')
features1 = np.genfromtxt('features_01.txt', dtype='float')
features2 = np.genfromtxt('features_02.txt', dtype='float')
corresp = np.genfromtxt('corresp_01_02.txt', dtype='int')


# -------------------------------------- estimate the Esential matrix -----------------------------------------
print("Estimating E")

best_E = np.zeros((3, 3))
best_points1 = np.zeros((2, 4))
best_points2 = np.zeros((2, 4))
best_support = 0
best_inlier_idxs = []
E = np.zeros((3, 3))
R = best_R = np.eye(3)
t = best_t = np.zeros((3, 1))

K = np.array([[2080,    0, 1421],
              [   0, 2080,  957],
              [   0,    0,    1]])
K_inv = np.linalg.inv(K)

rng = np.random.default_rng()
n_crp = corresp.shape[0]
k = 0
k_max = 100
theta = 3  # pixels
probability = 0.95
while k <= k_max:
    random_corresp = rng.choice(corresp, 5, replace=False)
    points1 = np.zeros((2, 5))
    points2 = np.zeros((2, 5))
    for i in range(5):
        points1[:, i] = features1[random_corresp[i, 0]]
        points2[:, i] = features2[random_corresp[i, 1]]
    # the points should be rectified first!
    points1_hom = K_inv @ tb.e2p(points1)
    points2_hom = K_inv @ tb.e2p(points2)

    Es = p5.p5gb(points1_hom, points2_hom)

    # check every E solution from p5
    for E in Es:
        decomp = tb.EutoRt(E, points1_hom, points2_hom)
        if not decomp:
            continue
        [R, t] = decomp
        F = K_inv.T@E@K_inv

        # use P the K because the points are unrectified!
        P1 = K @ np.eye(3, 4)
        P2 = K @ np.hstack((R, t))

        support = 0
        inlier_idxs = []
        for i in range(n_crp):
            u1 = tb.e2p(features1[corresp[i, 0]].reshape((2, 1)))
            u2 = tb.e2p(features2[corresp[i, 1]].reshape((2, 1)))
            X = tb.Pu2X(P1, P2, u1, u2)

            # compute the sampson error only for points in front of the camera
            if (P1 @ X)[2] >= 0 and (P2 @ X)[2] >= 0:
                e_sampson = tb.err_F_sampson(F, u1, u2)
                if e_sampson < theta:
                    support += float(1 - e_sampson**2/theta**2)
                    inlier_idxs.append(i)

        if support > best_support:
            best_support = support
            best_E = E
            best_R = R
            best_t = t
            best_inlier_idxs = inlier_idxs
            best_points1 = points1
            best_points2 = points2
            print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

        k += 1
        w = (support + 1) / n_crp
        k_max = math.log(1 - probability) / math.log(1 - w ** 2)

print("[ k:", k-1, "/", k_max, "] [ support:", best_support, "/", n_crp, "]")
E = best_E
R = best_R
t = best_t
F = K_inv.T @ E @ K_inv
inlier_idxs = best_inlier_idxs

print()
# print(inlier_idxs)
print("Result E\n", E)
print("Result R\n", R)
print("Result t\n", t)

