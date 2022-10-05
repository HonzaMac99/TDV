import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb
import math


# rng = np.random.default_rng()
# i = rng.choice(n, 2, replace=False)

def plot_results(points, sample, inliers, theta, best_line, best_support):
    inliers2plot = points[:, inliers[0]].reshape(2, 1)
    for i in range(len(inliers)-1):
        inliers2plot = np.hstack([inliers2plot, points[:, inliers[i]].reshape(2, 1)])

    line_x = [-best_line[2]/best_line[0], -(best_line[2] + 300*best_line[1])/best_line[0]]
    line_y = [0, 300]
    line_x = np.array(line_x)
    line_y = np.array(line_y)

    d = best_line[:2]/(math.sqrt(best_line[0]**2 + best_line[1]**2))
    boundary1_x = line_x + theta*d[0]
    boundary1_y = line_y + theta*d[1]
    boundary2_x = line_x - theta*d[0]
    boundary2_y = line_y - theta*d[1]

    plt.plot(line_x, line_y, "r-")
    plt.plot(boundary1_x, boundary1_y, "b--")
    plt.plot(boundary2_x, boundary2_y, "b--")
    plt.plot(inliers2plot[0], inliers2plot[1], "mx")
    plt.plot(sample[0], sample[1], 'go')
    plt.plot(points[0], points[1], 'ko', markersize=2)
    plt.title("Model with support:" + str(best_support))
    plt.show()


def ransac(points, theta, p, lsq=0):
    best_line = np.zeros(3)
    rng = np.random.default_rng()
    total_points = points.shape[1]
    k = 0
    k_max = 1000
    best_support = 0
    while k <= k_max:
        support = 0
        m = rng.choice(points.T, 2, replace=False).T
        m_hom = tb.e2p(np.array(m))
        n = np.cross(m_hom[:, 0], m_hom[:, 1])
        inliers2plot = []
        for i in range(total_points):
            point = tb.e2p(points[:, i].reshape(2, 1))
            error = abs(n@point/(math.sqrt(n[0]**2 + n[1]**2) * point[2]))
            if error < theta:
                inliers2plot.append(i)
                support += 1

        if support > best_support:
            best_support = support
            best_line = n

            # plot it
            plot_results(points, m, inliers2plot, theta, best_line, best_support)

        k += 1
        w = support/total_points
        k_max = math.log(1-p)/math.log(1-w**2)
        print("k:", k)
        print("k_max:", k_max)
        print("support:", support)
        print("-------------")
        print("")

        if k > k_max:
            plot_results(points, m, inliers2plot, theta, best_line, best_support)

    if lsq:
        print("TODO: do lsq over the inliers")

    return best_line


x = np.loadtxt("linefit_1.txt").T
x = np.array(x)

plt.plot(x[0], x[1], 'ko', markersize=2)
plt.show()

probability = 0.99
theta = 10
ransac(x, theta, probability)


