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

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(book2)
ax1.scatter(features2[:, 0], features2[:, 1], s=0.5, c='black')
ax1.set_title("book2.png")

ax2.imshow(book1)
ax2.scatter(features1[:, 0], features1[:, 1], s=0.5, c='black')
ax2.set_title("book1.png")

plt.show()

fig2, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(book2)
for i in range(corresp.shape[0]):
    idx1 = corresp[i, 0]
    idx2 = corresp[i, 1]
    ax1.plot([features2[idx2, 0], features1[idx1, 0]],
             [features2[idx2, 1], features1[idx1, 1]], color='green')
ax1.scatter(features2[:, 0], features2[:, 1], marker='o', s=0.8, c='black')
ax1.set_title("book2.png")

ax2.imshow(book1)
for i in range(corresp.shape[0]):
    idx1 = corresp[i, 0]
    idx2 = corresp[i, 1]
    ax2.plot([features1[idx1, 0], features2[idx2, 0]],
             [features1[idx1, 1], features2[idx2, 1]], color='red')
ax2.scatter(features1[:, 0], features1[:, 1], marker='o', s=0.8, c='black')
ax2.set_title("book1.png")

plt.show()


# find the first homography Ha
H = np.zeros((3, 3))
rng = np.random.default_rng()
n_points = 4
k = 0
k_max = 1000
best_support = 0
while k < k_max:
    random_corresp = rng.choice(corresp, n_points, replace=False)
    points1 = np.zeros((2, n_points))
    points2 = np.zeros((2, n_points))
    for i in range(n_points):
        points1[:, i] = features1[corresp[i, 0]]
        points2[:, i] = features2[corresp[i, 1]]

    A = np.zeros((2*n_points, 9))
    for i in range(4):
        [u1, v1] = points2[:, i]
        [u2, v2] = points1[:, i]
        A[i]   = [ u1,  v1, 1.0, 0.0, 0.0, 0.0, -u2*u1, -u2*v1, -u2]
        A[i+1] = [0.0, 0.0, 0.0,  u1,  v1, 1.0, -v2*u1, -v2*v1, -v2]

    [U, S, V] = np.linalg.svd(A)
    h = V[-1]
    H = h.reshape((3, 3))
    # TODO: calculate the repr. error, pick the best H

    k += 1
    # TODO: correctly update the k_max
    k_max = k_max

print(H)







