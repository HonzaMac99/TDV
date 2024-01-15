import numpy as np
import matplotlib.pyplot as plt
import math
import sys

sys.path.append('..')
sys.path.append('p5/python')
import tools as tb
import p5


# Esential matrix estimation
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
    probability = 0.9995
    k_max_reached = False
    while not k_max_reached:
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

            # use P with the K because the points are unrectified!
            P1 = K @ np.eye(3, 4)
            P2 = K @ np.hstack((R, t))

            support = 0
            inlier_idxs = []
            for i in range(n_crp):
                u1 = tb.e2p(features1[corresp[i, 0]].reshape((2, 1)))
                u2 = tb.e2p(features2[corresp[i, 1]].reshape((2, 1)))
                X = tb.Pu2X(P1, P2, u1, u2)

                # compute the sampson error only for points in front of the camera
                if (P1 @ X)[2] > 0 and (P2 @ X)[2] > 0:
                    e_sampson = tb.err_F_sampson(F, u1, u2)
                    if e_sampson < theta:
                        support += float(1 - e_sampson**2/theta**2)
                        inlier_idxs.append(i)

            k += 1
            w = (support + 1) / (n_crp + 1)
            k_max = math.log(1 - probability) / math.log(1 - w ** 2)

            if support > best_support:
                best_support = support
                best_E = E
                best_R = R
                best_t = t
                best_inlier_idxs = inlier_idxs
                print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

            if k >= k_max:
                k_max_reached = True
                break

    print("Result E\n", best_E)
    print("Result R\n", best_R)
    print("Result t\n", best_t)

    return best_E, best_R, best_t, best_inlier_idxs


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


if __name__ == '__main__':
    print("This is a file with function to estimate E, get inlier corresp, and 3D scene points")
