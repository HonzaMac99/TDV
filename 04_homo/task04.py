#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cbook
import toolbox as tb
import math
import sys

book1 = mpimg.imread('book1.png')
book2 = mpimg.imread('book2.png')
features1 = np.genfromtxt('books_u1.txt', dtype='float')
features2 = np.genfromtxt('books_u2.txt', dtype='float')
corresp = np.genfromtxt('books_m12.txt', dtype='int')


# # show the images and their feature points
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# ax1.imshow(book2)
# ax1.scatter(features2[:, 0], features2[:, 1], s=0.5, c='black')
# ax1.set_title("book2.png")
#
# ax2.imshow(book1)
# ax2.scatter(features1[:, 0], features1[:, 1], s=0.5, c='black')
# ax2.set_title("book1.png")
#
# plt.show()
#
#
# # show the differences between both feature groups on the images
# fig2, (ax1, ax2) = plt.subplots(1, 2)
#
# ax1.imshow(book2)
# for i in range(corresp.shape[0]):
#     idx1 = corresp[i, 0]
#     idx2 = corresp[i, 1]
#     ax1.plot([features2[idx2, 0], features1[idx1, 0]],
#              [features2[idx2, 1], features1[idx1, 1]], color='green')
# ax1.scatter(features2[:, 0], features2[:, 1], marker='o', s=0.8, c='black')
# ax1.set_title("book2.png")
#
# ax2.imshow(book1)
# for i in range(corresp.shape[0]):
#     idx1 = corresp[i, 0]
#     idx2 = corresp[i, 1]
#     ax2.plot([features1[idx1, 0], features2[idx2, 0]],
#              [features1[idx1, 1], features2[idx2, 1]], color='red')
# ax2.scatter(features1[:, 0], features1[:, 1], marker='o', s=0.8, c='black')
# ax2.set_title("book1.png")
#
# plt.show()


# find the first homography Ha
best_H = np.zeros((3, 3))
best_points1 = np.zeros((2, 4))
best_points2 = np.zeros((2, 4))

rng = np.random.default_rng()
n_crp = corresp.shape[0]
k = 0
support = 0
k_max = 100
theta = 3  # 3 pixels
probability = 0.99
best_support = 0
while k <= k_max:
    random_corresp = rng.choice(corresp, 4, replace=False)
    points1 = np.zeros((2, 4))
    points2 = np.zeros((2, 4))
    for i in range(4):
        points1[:, i] = features1[random_corresp[i, 0]]
        points2[:, i] = features2[random_corresp[i, 1]]

    A = np.zeros((2 * 4, 9))
    for i in range(4):
        [u1, v1] = points1[:, i]
        [u2, v2] = points2[:, i]
        A[2*i]   = [ u1,  v1, 1.0, 0.0, 0.0, 0.0, -u2*u1, -u2*v1, -u2]
        A[2*i+1] = [0.0, 0.0, 0.0,  u1,  v1, 1.0, -v2*u1, -v2*v1, -v2]

    [U, S, V] = np.linalg.svd(A)
    h = V[-1]
    H = h.reshape((3, 3))
    H_inv = np.linalg.inv(H)

    support = 0
    for i in range(n_crp):
        p1 = features1[corresp[i, 0]].reshape((2, 1))
        p2 = features2[corresp[i, 1]].reshape((2, 1))
        p1_proj = tb.p2e(H@tb.e2p(p1))      # img1 --> img2
        p2_proj = tb.p2e(H_inv@tb.e2p(p2))  # img2 --> img1
        d1 = math.sqrt((p2[0] - p1_proj[0])**2 + (p2[1] - p1_proj[1])**2)
        d2 = math.sqrt((p1[0] - p2_proj[0])**2 + (p1[1] - p2_proj[1])**2)
        if (d1 + d2)/2 < theta:
            support += 1

    if support > best_support:
        best_support = support
        best_H = H
        best_points1 = points1
        best_points2 = points2
        print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

    k += 1
    # w = (support + 1) / n_crp
    # k_max = math.log(1 - probability) / math.log(1 - w ** 2)

    # print("k:", k)
    # print("k_max:", k_max)
    # print("support:", support)
    # print("-------------")
    # print("")

print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")


print("Best homography H:")
print(best_H)
best_H_inv = np.linalg.inv(best_H)

fig, (ax1, ax2) = plt.subplots(1, 2)

############################## first image ##############################
# ax1.imshow(book1)
ax1.set_title("Projected to img 1")

ax1.scatter(features1[:, 0], features1[:, 1], s=0.5, c='black')
ax1.scatter(features2[:, 0], features2[:, 1], s=0.2, c='gray')
f2_proj = tb.p2e(best_H_inv@tb.e2p(features2.T)).T
ax1.scatter(f2_proj[:, 0], f2_proj[:, 1], s=0.2, c='red')

# plot the 4 points used for H estimation
ax1.scatter(best_points1[0, :], best_points1[1, :], s=30.0, c='black', label="Orig. features")
ax1.scatter(best_points2[0, :], best_points2[1, :], s=12.0, c='gray', label="Corresp. features")
bp2_proj = tb.p2e(best_H_inv@tb.e2p(best_points2)).T
ax1.scatter(bp2_proj[:, 0], bp2_proj[:, 1], s=12.0, c='red', label="Projected corresp. features")

ax1.legend()

############################## second image ##############################
# ax2.imshow(book2)
ax2.set_title("Projected to img 2")
ax2.scatter(features2[:, 0], features2[:, 1], s=0.5, c='black')
ax2.scatter(features1[:, 0], features1[:, 1], s=0.2, c='gray')
f1_proj = tb.p2e(best_H@tb.e2p(features1.T)).T
ax2.scatter(f1_proj[:, 0], f1_proj[:, 1], s=0.2, c='green')

# plot the 4 points used for H estimation
ax2.scatter(best_points2[0, :], best_points2[1, :], s=30.0, c='black', label="Orig. features")
ax2.scatter(best_points1[0, :], best_points1[1, :], s=12.0, c='gray', label="Corresp. features")
bp1_proj = tb.p2e(best_H@tb.e2p(best_points1)).T
ax2.scatter(bp1_proj[:, 0], bp1_proj[:, 1], s=12.0, c='green', label="Projected corresp features")

ax2.legend()
plt.show()








