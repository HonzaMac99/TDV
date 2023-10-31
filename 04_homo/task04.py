#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cbook
import toolbox as tb
import math
import copy
import sys

def get_line_boundaries(plot_line, img):
    min_x = 1
    min_y = 1
    max_x = img.shape[1]
    max_y = img.shape[0]

    boundaries = np.array([[min_x, min_x, max_x, max_x],
                           [min_y, max_y, max_y, min_y]])
    boundaries_hom = tb.e2p(boundaries)

    # get line vectors of the boundaries
    a_line = np.cross(boundaries_hom[:, 0], boundaries_hom[:, 1])
    b_line = np.cross(boundaries_hom[:, 1], boundaries_hom[:, 2])
    c_line = np.cross(boundaries_hom[:, 2], boundaries_hom[:, 3])
    d_line = np.cross(boundaries_hom[:, 3], boundaries_hom[:, 0])
    bnd_lines = [a_line, b_line, c_line, d_line]

    line_boundaries = np.zeros([2, 2])
    count = 0
    for bnd_line in bnd_lines:
        line_end = tb.p2e((np.cross(plot_line, bnd_line).reshape(3, 1)))
        x = line_end[0]
        y = line_end[1]
        # plt.plot(x, y, "oy")
        if 1 <= int(x) <= max_x and 1 <= int(y) <= max_y:
            line_end = np.reshape(line_end, (1, 2))
            line_boundaries[:, count] = line_end
            count += 1
    return line_boundaries

# show the images and their feature points
def plot_features(img1, img2, features1, features2):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img1)
    ax1.scatter(features1[:, 0], features1[:, 1], s=0.5, c='black')
    ax1.set_title("book1.png")

    ax2.imshow(img2)
    ax2.scatter(features2[:, 0], features2[:, 1], s=0.5, c='black')
    ax2.set_title("book2.png")

    plt.show()


# show the differences between both feature groups on the images
def plot_corresp(img1, img2, features1, features2, corresp):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img1)
    for i in range(corresp.shape[0]):
        idx1 = corresp[i, 0]
        idx2 = corresp[i, 1]
        ax1.plot([features1[idx1, 0], features2[idx2, 0]],
                 [features1[idx1, 1], features2[idx2, 1]], color='green')
    ax1.scatter(features1[:, 0], features1[:, 1], marker='o', s=0.8, c='black')
    ax1.set_title("book2.png")

    ax2.imshow(img2)
    for i in range(corresp.shape[0]):
        idx1 = corresp[i, 0]
        idx2 = corresp[i, 1]
        ax2.plot([features2[idx2, 0], features1[idx1, 0]],
                 [features2[idx2, 1], features1[idx1, 1]], color='red')
    ax2.scatter(features2[:, 0], features2[:, 1], marker='o', s=0.8, c='black')
    ax2.set_title("book1.png")

    plt.show()


# show the results of the homography estimation
def plot_ha(img1, img2, features1, features2, best_points1, best_points2, best_H):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    best_H_inv = np.linalg.inv(best_H)

    # ax1.imshow(img1)
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

    # ax2.imshow(img2)
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



book1 = mpimg.imread('book1.png')
book2 = mpimg.imread('book2.png')
features1 = np.genfromtxt('books_u1.txt', dtype='float')
features2 = np.genfromtxt('books_u2.txt', dtype='float')
corresp = np.genfromtxt('books_m12.txt', dtype='int')


# show the images and their feature points
# plot_features(book2, book1, features2, features1)

# show the differences between both feature groups on the images
# plot_corresp(book2, book1, features2, features1, corresp)


# -------------------------------------- estimate the first homography Ha -----------------------------------------
print("Estimating Ha")
best_Ha = np.zeros((3, 3))
best_points1 = np.zeros((2, 4))
best_points2 = np.zeros((2, 4))
inlier_idxs = []
best_inlier_idxs = []

rng = np.random.default_rng()
n_crp = corresp.shape[0]
k = 0
k_max = 100
support = 0
best_support = 0
theta = 3  # pixels
probability = 0.99
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
    Ha = h.reshape((3, 3))
    Ha_inv = np.linalg.inv(Ha)

    support = 0
    inlier_idxs = []
    for i in range(n_crp):
        p1 = features1[corresp[i, 0]].reshape((2, 1))
        p2 = features2[corresp[i, 1]].reshape((2, 1))
        p1_proj = tb.p2e(Ha@tb.e2p(p1))      # img1 --> img2
        p2_proj = tb.p2e(Ha_inv@tb.e2p(p2))  # img2 --> img1
        d1 = math.sqrt((p2[0] - p1_proj[0])**2 + (p2[1] - p1_proj[1])**2)
        d2 = math.sqrt((p1[0] - p2_proj[0])**2 + (p1[1] - p2_proj[1])**2)
        if (d1 + d2)/2 < theta:
            support += 1
            inlier_idxs.append(i)

    if support > best_support:
        best_support = support
        best_inlier_idxs = inlier_idxs
        best_Ha = Ha
        best_points1 = points1
        best_points2 = points2
        print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

    k += 1
    w = (support + 1) / n_crp
    k_max = math.log(1 - probability) / math.log(1 - w ** 2)

print("[ k:", k-1, "/", k_max, "] [ support:", best_support, "/", n_crp, "]")
Ha = best_Ha
inlier_idxs = best_inlier_idxs
Ha_inv = np.linalg.inv(Ha)

# plot the results of the first (dominant) homography estimation
# plot_ha(book1, book2, features1, features2, best_points1, best_points2, best_Ha)

# create corresp array without current inliers
new_corresp = []
ha_inliers = []
for i in range(n_crp):
    if i in inlier_idxs:
        ha_inliers.append(corresp[i])
    else:
        new_corresp.append(corresp[i])
corresp = np.array(new_corresp)


# -------------------------------------- estimate the homology and second homography Hb -----------------------------------------
print("Estimating Hb")
best_Hb = np.zeros((3, 3))
best_points1 = np.zeros((2, 3))
best_points2 = np.zeros((2, 3))
inlier_idxs = []
a = np.zeros((1, 3))

rng = np.random.default_rng()
n_crp = corresp.shape[0]
k = 0
k_max = 1000
support = 0
best_support = 0
theta = 3  # pixels
probability = 0.99
while k <= k_max:
    random_corresp = rng.choice(corresp, 3, replace=False)
    points1 = np.zeros((2, 3))
    points2 = np.zeros((2, 3))
    for i in range(3):
        points1[:, i] = features1[random_corresp[i, 0]]
        points2[:, i] = features2[random_corresp[i, 1]]
    # project points img2 --> img1, then make them 2D and homogenous
    points2_proj = tb.e2p(tb.p2e(Ha_inv@tb.e2p(points2)))
    [u1, u2, u3] = tb.e2p(points1).T
    [u_1, u_2, u_3] = points2_proj.T  # "_" represents " ' "

    v = np.cross(np.cross(u1, u_1), np.cross(u2, u_2))

    A = np.vstack(((u_1[0]*v[2] - u_1[2]*v[0])*u1,
                   (u_2[0]*v[2] - u_2[2]*v[0])*u2,
                   (u_3[0]*v[2] - u_3[2]*v[0])*u3))

    b = np.array([u1[0]*u_1[2] - u1[2]*u_1[0],
                  u2[0]*u_2[2] - u2[2]*u_2[0],
                  u3[0]*u_3[2] - u3[2]*u_3[0]])

    # # solve Aa = b
    # a = np.linalg.solve(A, b)
    A_inv = np.linalg.inv(A)
    a = A_inv@b

    H = np.eye(3) + v.reshape(3, 1)@a.reshape((1, 3))

    Hb = Ha@H
    Hb_inv = np.linalg.inv(Hb)

    # fig, ax = plt.subplots()
    # ax.imshow(book1)
    # a1 = tb.p2e(a.reshape(3, 1))
    # a2 = tb.p2e(a.reshape(3, 1))
    # ax.plot([a1[0], a2[0]], [a1[1], a2[1]], "v--")
    # plt.show()

    support = 0
    inlier_idxs = []
    for i in range(n_crp):
        p1 = features1[corresp[i, 0]].reshape((2, 1))
        p2 = features2[corresp[i, 1]].reshape((2, 1))
        p1_proj = tb.p2e(Hb@tb.e2p(p1))      # img1 --> img2
        p2_proj = tb.p2e(Hb_inv@tb.e2p(p2))  # img2 --> img1
        d1 = math.sqrt((p2[0] - p1_proj[0])**2 + (p2[1] - p1_proj[1])**2)
        d2 = math.sqrt((p1[0] - p2_proj[0])**2 + (p1[1] - p2_proj[1])**2)
        if (d1 + d2)/2 < theta:
            support += 1
            inlier_idxs.append(i)

    if support > best_support:
        best_support = support
        best_inlier_idxs = inlier_idxs
        best_Hb = Hb
        best_points1 = points1
        best_points2 = points2
        print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

    k += 1
    w = (support + 1) / n_crp
    k_max = math.log(1 - probability) / math.log(1 - w ** 2)

print("[ k:", k-1, "/", k_max, "] [ support:", best_support, "/", n_crp, "]")
Hb = best_Hb
inlier_idxs = best_inlier_idxs
best_H_inv = np.linalg.inv(Hb)

# create corresp array without current inliers
outliers = []
hb_inliers = []
for i in range(n_crp):
    if i in inlier_idxs:
        hb_inliers.append(corresp[i])
    else:
        outliers.append(corresp[i])
outliers = np.array(outliers)

# plot the results
fig, ax = plt.subplots()
ax.set_title("results")
ax.imshow(book1)

# plot the outliers first
for i in range(len(outliers)):
    idx1 = outliers[i][0]
    idx2 = outliers[i][1]
    ax.scatter(features1[idx1, 0], features1[idx1, 1], marker='o', s=0.8, c='black')
    ax.plot([features1[idx1, 0], features2[idx2, 0]],
            [features1[idx1, 1], features2[idx2, 1]], color='black')

# plot inliers of the first homography
for i in range(len(ha_inliers)):
    idx1 = ha_inliers[i][0]
    idx2 = ha_inliers[i][1]
    ax.scatter(features1[idx1, 0], features1[idx1, 1], marker='o', s=0.8, c='red')
    ax.plot([features1[idx1, 0], features2[idx2, 0]],
            [features1[idx1, 1], features2[idx2, 1]], color='red')

# plot the inliers of the second homography
for i in range(len(hb_inliers)):
    idx1 = hb_inliers[i][0]
    idx2 = hb_inliers[i][1]
    ax.scatter(features1[idx1, 0], features1[idx1, 1], marker='o', s=0.8, c='green')
    ax.plot([features1[idx1, 0], features2[idx2, 0]],
            [features1[idx1, 1], features2[idx2, 1]], color='green')

# draw the division line between the homographies
a_boundaries = get_line_boundaries(a, book1)
plt.plot(a_boundaries[0], a_boundaries[1], "m-")

plt.show()










