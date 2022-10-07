import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb
import math
import sys


# rng = np.random.default_rng()
# i = rng.choice(n, 2, replace=False)


def find_angle_dist(line):
    [a, b, c] = line
    angle = math.acos(b / math.sqrt(a**2 + b**2))
    angle = angle*180/math.pi  # from radians to deg
    if angle > 90:
        angle = 180 - angle
    elif angle < -90:
        angle = -180 - angle
    dist = abs(c / math.sqrt(a**2 + b**2))

    return angle, dist


def print_status(k, k_max, support):
    print("k:", k)
    print("k_max:", k_max)
    print("support:", support)
    print("-------------")
    print("")


def plot_results(points, sample, inliers, theta, best_line, best_support):
    inliers_plot = points[:, inliers[0]].reshape(2, 1)
    for i in range(len(inliers)-1):
        inliers_plot = np.hstack([inliers_plot, points[:, inliers[i]].reshape(2, 1)])

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
    plt.plot(inliers_plot[0], inliers_plot[1], "mx")
    plt.plot(sample[0], sample[1], 'go')
    plt.plot(points[0], points[1], 'ko', markersize=2)
    plt.title("Model with support:" + str(best_support))
    plt.show()


def do_lsq(inliers):
    n = inliers.shape[1]
    centroid = np.zeros([2, 1])
    for i in range(n):
        centroid += inliers[:, i].reshape([2, 1])
    centroid /= n

    A = np.zeros([2, n])
    for i in range(n):
        A[:, i] = inliers[:, i] - centroid.reshape([2, ])

    [U, S, V] = np.linalg.svd(A)

    [a, b] = U[:, 1]
    normal = np.array([a, b])
    c = -normal@centroid.reshape([2,])
    best_line = [a, b, c]
    return best_line



def find_model(points, theta, probability, mode=0, lsq=0):
    """
    find_model:
    finds the best fitting model of a line on a set of points with noise
    in:  points      ... set of points for the model
         theta       ... max distance of the points from the line
         probability ... max prob. that the model is optimal
         mode        ... 0=ransac, 1=mlesac
         lsq         ... 0=no lsq, 1=do a lsq regression on the resulting model and inliers
    out: angle       ... angle of the line
         dist        ... distance from origin
    """
    best_line = np.zeros(3)
    rng = np.random.default_rng()
    total_points = points.shape[1]
    k = 0
    k_max = 1000
    best_support = 0
    inliers_idx = []
    while k <= k_max:
        support = 0
        m = rng.choice(points.T, 2, replace=False).T
        m_hom = tb.e2p(np.array(m))
        n = np.cross(m_hom[:, 0], m_hom[:, 1])
        inliers_idx = []
        for i in range(total_points):
            point = tb.e2p(points[:, i].reshape(2, 1))
            error = abs(n@point/(math.sqrt(n[0]**2 + n[1]**2) * point[2]))
            if error < theta:
                inliers_idx.append(i)
                if mode == 0:  # ransac
                    support += 1
                elif mode == 1:  # mlesac
                    support += 1 - error**2/theta**2
                else:
                    print("Invalid mode! (only 0 or 1 allowed)")
                    return
        if support > best_support:
            best_support = support
            best_line = n
            # plot_results(points, m, inliers_idx, theta, best_line, best_support)
        k += 1
        w = support / total_points
        k_max = math.log(1 - probability) / math.log(1 - w ** 2)
        # print_status(k, k_max, support)

    num_inliers = len(inliers_idx)
    inliers = np.zeros([2, num_inliers])
    for i in range(num_inliers):
        inliers[:, i] = points[:, inliers_idx[i]]

    if lsq:
        best_line = do_lsq(inliers)

    return find_angle_dist(best_line)

probability = 0.99
theta = 10
input_points = np.array(np.loadtxt("linefit_1.txt").T)
# input_points = np.array(np.loadtxt("linefit_2.txt").T)
# input_points = np.array(np.loadtxt("linefit_3.txt").T)

# RANSAC 100 times
for i in range(1, 101):
    sys.stdout.flush()
    sys.stdout.write("\rRansac: %i %%" % i)

    line_angle, line_dist = find_model(input_points, theta, probability, 0, 0)
    plt.xlabel("angle [deg]")
    plt.ylabel("dist to origin")
    plt.plot(line_angle, line_dist, "mo", markersize=2)
print("")

# MLESAC 100 times
for i in range(1, 101):
    sys.stdout.flush()
    sys.stdout.write("\rMlesac: %i %%" % i)

    line_angle, line_dist = find_model(input_points, theta, probability, 1, 0)
    plt.xlabel("angle [deg]")
    plt.ylabel("dist to origin")
    plt.plot(line_angle, line_dist, "bo", markersize=2)
print("")

# RANSAC with after-MLE 5 times
for i in range(0, 101, 20):

    sys.stdout.flush()
    sys.stdout.write("\rRansac with MLE: %i %%" % i)

    line_angle, line_dist = find_model(input_points, theta, probability, 1, 1)
    plt.xlabel("angle [deg]")
    plt.ylabel("dist to origin")
    plt.plot(line_angle, line_dist, "ro", markersize=2)
print("")

# MLESAC with after-MLE 5 times
for i in range(0, 101, 20):

    sys.stdout.flush()
    sys.stdout.write("\rMlesac with MLE: %i %%" % i)

    line_angle, line_dist = find_model(input_points, theta, probability, 1, 1)
    plt.xlabel("angle [deg]")
    plt.ylabel("dist to origin")
    plt.plot(line_angle, line_dist, "co", markersize=2)

# ground truth
orig_line = np.array([-10, 3, 1200])
ol_angle, ol_dist = find_angle_dist(orig_line)
plt.plot(ol_angle, ol_dist, "go")

plt.show()
