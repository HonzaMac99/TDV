#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import sys
sys.path.append('..')
sys.path.append('p5/python')
sys.path.append('corresp/python')
sys.path.append('geom_export/python')
import tools as tb
import p5
import corresp
import p3p
import ge


def get_inliers_outliers(corresp, inlier_idxs):
    e_inliers = []
    outliers = []
    n_crp = corresp.shape[0]

    for i in range(n_crp):
        if i in inlier_idxs:
            e_inliers.append(corresp[i])
        else:
            outliers.append(corresp[i])
    return e_inliers, outliers


def plot_inliers(img1, features1, features2, corresp, inlier_idxs):

    e_inliers, outliers = get_inliers_outliers(corresp, inlier_idxs)

    # plot inliers and outliers
    fig, ax = plt.subplots()
    ax.set_title("results")
    ax.imshow(img1)

    # plot the outliers first
    for i in range(len(outliers)):
        idx1 = outliers[i][0]
        idx2 = outliers[i][1]
        ax.scatter(features1[idx1, 0], features1[idx1, 1], marker='o', s=0.8, c='black')
        ax.plot([features1[idx1, 0], features2[idx2, 0]],
                [features1[idx1, 1], features2[idx2, 1]], color='black')

    # plot inliers of the essential matrix E
    for i in range(len(e_inliers)):
        idx1 = e_inliers[i][0]
        idx2 = e_inliers[i][1]
        ax.scatter(features1[idx1, 0], features1[idx1, 1], marker='o', s=0.8, c='red')
        ax.plot([features1[idx1, 0], features2[idx2, 0]],
                [features1[idx1, 1], features2[idx2, 1]], color='red')

    plt.show()


def plot_e_lines(img1, img2, features1, features2, corresp, inlier_idxs, F):

    e_inliers, outliers = get_inliers_outliers(corresp, inlier_idxs)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)

    colors = ["darkred", "chocolate", "darkorange", "gold", "lime", "steelblue", "navy", "indigo", "orchid", "crimson"]
    for i in range(10):
        idx = (i * 10) % len(e_inliers)  # pick every tenth point
        idx1 = e_inliers[idx][0]
        idx2 = e_inliers[idx][1]
        m1 = features1[idx1].reshape((2, 1))
        m2 = features2[idx2].reshape((2, 1))
        l1 = F.T @ tb.e2p(m2)
        l2 = F @ tb.e2p(m1)
        l1_b = tb.get_line_boundaries(l1, img1)
        l2_b = tb.get_line_boundaries(l2, img2)

        ax1.scatter(m1[0], m1[1], marker='o', s=4.0, c=colors[i])
        ax1.plot(l1_b[0], l1_b[1], color=colors[i])

        ax2.scatter(m2[0], m2[1], marker='o', s=4.0, c=colors[i])
        ax2.plot(l2_b[0], l2_b[1], color=colors[i])

    plt.show()


def get_3d_points(features1, features2, corresp, inlier_idx, img, P1, P2):
    sparsity = 1
    Xs = np.zeros((4, 0))
    colors = []
    for i in range(0, len(inlier_idx), sparsity):
        idx = inlier_idx[i]
        u1 = tb.e2p(features1[corresp[idx, 0]].reshape((2, 1)))
        u2 = tb.e2p(features2[corresp[idx, 1]].reshape((2, 1)))
        X = tb.Pu2X(P1, P2, u1, u2)
        Xs = np.hstack((Xs, X))

        # for getting color from the img into the plot:
        img_point = features1[corresp[idx, 0]].reshape(2,)
        [x, y] = np.round(img_point).astype(int)

        [r, g, b] = img[y, x]
        color = (r/255.0, g/255.0, b/255.0)
        colors.append(color)

    return Xs, colors


def init_corresp(n_cameras):
    c = corresp.Corresp(n_cameras)
    c.verbose = 2
    for i in range(1, n_cameras):
        for j in range(i+1, n_cameras):
            f_name =
            corresps = np.genfromtxt('scene_1/corresp/m_01_02.txt', dtype='int')



# -------------------------------------- Esential matrix estimation -----------------------------------------
def ransac_E(features1, features2, corresp, K):
    print("Estimating E")

    best_E = np.zeros((3, 3))
    best_R = np.zeros((3, 3))
    best_t = np.zeros((3, 1))
    K_inv = np.linalg.inv(K)

    best_support = 0
    best_inlier_idxs = []
    rng = np.random.default_rng()
    n_crp = corresp.shape[0]
    k = 0
    k_max = 100
    theta = 3  # pixels
    probability = 0.95  # 0.40
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
                print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

            k += 1
            w = (support + 1) / n_crp
            k_max = math.log(1 - probability) / math.log(1 - w ** 2)

    print("[ k:", k-1, "/", k_max, "] [ support:", best_support, "/", n_crp, "]\n")

    # print(inlier_idxs)
    print("Result E\n", best_E)
    print("Result R\n", best_R)
    print("Result t\n", best_t)

    return best_E, best_R, best_t, best_inlier_idxs
# -----------------------------------------------------------------------------------------------------------


img1 = mpimg.imread('scene_1/images/01.jpg')
img2 = mpimg.imread('scene_1/images/02.jpg')
features1 = np.genfromtxt('scene_1/corresp/u_01.txt', dtype='float')
features2 = np.genfromtxt('scene_1/corresp/u_02.txt', dtype='float')
corresps = np.genfromtxt('scene_1/corresp/m_01_02.txt', dtype='int')
K = np.loadtxt('scene_1/K.txt', dtype='float')
print(K)
# K = np.array([[2080,    0, 1421],
#               [   0, 2080,  957],
#               [   0,    0,    1]])

# perform the actual E estimation
E, R, t, inls = ransac_E(features1, features2, corresps, K)

K_inv = np.linalg.inv(K)
F = K_inv.T @ E @ K_inv

# plot the inliers and outliers
plot_inliers(img1, features1, features2, corresps, inls)

# plot the epipolar lines
# plot_e_lines(img1, img2, features1, features2, corresps, inls, F)

I = np.eye(3, 3)
P1 = K @ np.eye(3, 4)
P2 = K @ np.hstack((R, t))

C1 = np.zeros((3, 1))
C2 = -R.T @ t
Cs = np.hstack((C1, C2))
# C2 = K_inv @ P2 @ np.vstack((C1, 1))

z1 = np.array([0, 0, 1]).reshape(3, 1)
z2 = C2 + R[2, :].reshape(3, 1)
z = np.hstack((z1, z2))
# z2 = K_inv @ P2 @ np.vstack((z1, 1))


# plot the centers of cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-2, 2)
# ax.axis('off')

# plot the camera centers and the baseline
ax.plot(Cs[0, :], Cs[1, :], Cs[2, :], marker='o', c='red')
for i in range(Cs.shape[1]):
    ax.scatter(Cs[0, i], Cs[1, i], Cs[2, i], marker='o', s=20, c='black')
    ax.text(Cs[0, i], Cs[1, i], Cs[2, i], str(i+1), fontsize=10, c='black')

    # plot the camera z axis
    ax.plot([Cs[0, i], z[0, i]],
            [Cs[1, i], z[1, i]],
            [Cs[2, i], z[2, i]], c='black')

# get the 3d point with colors
Xs, colors = get_3d_points(features1, features2, corresps, inls, img1, P1, P2)

print(Xs.shape[1])
for i in range(Xs.shape[1]):
    ax.scatter(Xs[0, i], Xs[1, i], Xs[2, i], marker='o', color=colors[i])

plt.show()

c = init_corresp()

g = ge.GePly('out.ply')
colors = np.array(colors).T
# g.points(Xs, colors) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
g.points(Xs) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
g.close()





