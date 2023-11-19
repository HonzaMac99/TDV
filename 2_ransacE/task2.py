#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tools as tb
import math
import p5


# show the images and their feature points
def plot_features(img1, img2, fs1, fs2):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img2)
    ax1.scatter(fs2[:, 0], fs2[:, 1], s=0.5, c='black')
    ax1.set_title("book2.png")

    ax2.imshow(img1)
    ax2.scatter(fs1[:, 0], fs1[:, 1], s=0.5, c='black')
    ax2.set_title("book1.png")

    plt.show()


# show the differences between both feature groups on the images
def plot_corresp(img1, img2, fs1, fs2, corresp):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img2)
    for i in range(corresp.shape[0]):
        idx1 = corresp[i, 0]
        idx2 = corresp[i, 1]
        ax1.plot([fs2[idx2, 0], fs1[idx1, 0]],
                 [fs2[idx2, 1], fs1[idx1, 1]], color='green')
    ax1.scatter(fs2[:, 0], fs2[:, 1], marker='o', s=0.8, c='black')
    ax1.set_title("image 2")

    ax2.imshow(img1)
    for i in range(corresp.shape[0]):
        idx1 = corresp[i, 0]
        idx2 = corresp[i, 1]
        ax2.plot([fs1[idx1, 0], fs2[idx2, 0]],
                 [fs1[idx1, 1], fs2[idx2, 1]], color='red')
    ax2.scatter(fs1[:, 0], fs1[:, 1], marker='o', s=0.8, c='black')
    ax2.set_title("image 1")

    plt.show()


img1 = mpimg.imread('cam1.jpg')
img2 = mpimg.imread('cam2.jpg')
features1 = np.genfromtxt('features_01.txt', dtype='float')
features2 = np.genfromtxt('features_02.txt', dtype='float')
corresp = np.genfromtxt('corresp_01_02.txt', dtype='int')

# show the images and their feature points
plot_features(img1, img2, features1, features2)

# show the differences between both feature groups on the images
plot_corresp(img1, img2, features1, features2, corresp)


# -------------------------------------- estimate the Esential matrix -----------------------------------------
print("Estimating E")

best_E = np.zeros((3, 3))
best_points1 = np.zeros((2, 4))
best_points2 = np.zeros((2, 4))
best_support = 0
best_inlier_idxs = []

rng = np.random.default_rng()
n_crp = corresp.shape[0]
k = 0
k_max = 100
theta = 3  # pixels
probability = 0.99
while k <= k_max:
    random_corresp = rng.choice(corresp, 5, replace=False)
    points1 = np.zeros((2, 4))
    points2 = np.zeros((2, 4))
    for i in range(4):
        points1[:, i] = features1[random_corresp[i, 0]]
        points2[:, i] = features2[random_corresp[i, 1]]

    E = np.zeros((3, 3))
    E =

    support = 0
    inlier_idxs = []
    for i in range(n_crp):
        p1 = features1[corresp[i, 0]].reshape((2, 1))
        p2 = features2[corresp[i, 1]].reshape((2, 1))

        # TODO: check if the points are before camera
        # TODO: use the samson error

        error = 4
        if error < theta:
            support += 1
            inlier_idxs.append(i)

    if support > best_support:
        best_support = support
        best_E = E
        best_inlier_idxs = inlier_idxs
        best_points1 = points1
        best_points2 = points2
        print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

    k += 1
    w = (support + 1) / n_crp
    k_max = math.log(1 - probability) / math.log(1 - w ** 2)

print("[ k:", k-1, "/", k_max, "] [ support:", best_support, "/", n_crp, "]")
E = best_E
inlier_idxs = best_inlier_idxs

# TODO: plot the inliers of the E
# plot the results of the first (dominant) homography estimation
# plot_ha(book1, book2, features1, features2, best_points1, best_points2, best_Ha)
