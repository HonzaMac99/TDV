#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb

# plt.ginput ... for entering the points
# np.cross ... cross product

# homography
H = np.array([[1,     0.1,   0],
              [0.1,   1,     0],
              [0.004, 0.002, 1]])


# plot the image boundary
def plot_boundaries(boundaries):
    plt.plot(np.hstack([boundaries[0], boundaries[0, 0]]),
             np.hstack([boundaries[1], boundaries[1, 0]]), "k")


def get_points(count):
    input_pts = plt.ginput(count)
    output_pts = np.zeros([2, count])
    for i in range(count):
        input_point = list(input_pts[i])
        output_pts[:, i] = input_point
    return output_pts


def plot_points(points):
    plt.plot(points[0, :2], points[1, :2], "or")
    plt.plot(points[0, 2:4], points[1, 2:4], "ob")


# finds the parameters of the line defined by two points m1 and m2
def get_line(m1, m2):
    n = np.cross(m1, m2)
    return n


# finds the cross point of two lines n1 and n2
def get_cross(n1, n2):
    m = np.cross(n1, n2)
    return np.array([m]).T


def inside_img(coord_x, coord_y):
    return 1 <= int(coord_x) <= 800 and 1 <= int(coord_y) <= 600


def get_line_boundaries(line, boundaries):
    line_boundaries = np.zeros([2, 2])
    count = 0
    for bnd_line in boundaries:
        line_end = tb.p2e(get_cross(line, bnd_line))
        x = line_end[0]
        y = line_end[1]
        # plt.plot(x, y, "oy")
        if inside_img(x, y):
            line_end = np.reshape(line_end, (1, 2))
            line_boundaries[:, count] = line_end
            count += 1
    return line_boundaries


# boundary of the image
orig_boundaries = np.array([[1, 1, 800, 800], [1, 600, 600, 1]])
plot_boundaries(orig_boundaries)
orig_boundaries_hom = tb.e2p(orig_boundaries)

# get 4 points from the input
plt.gca().invert_yaxis()
orig_points = get_points(4)
plot_points(orig_points)
orig_points_hom = tb.e2p(orig_points)

# get line vectors of the boundaries
a_line = get_line(orig_boundaries_hom[:, 0], orig_boundaries_hom[:, 1])
b_line = get_line(orig_boundaries_hom[:, 1], orig_boundaries_hom[:, 2])
c_line = get_line(orig_boundaries_hom[:, 2], orig_boundaries_hom[:, 3])
d_line = get_line(orig_boundaries_hom[:, 3], orig_boundaries_hom[:, 0])
lines = [a_line, b_line, c_line, d_line]

# get parameters of the line defined by the first two points
m1_line = get_line(orig_points_hom[:, 0], orig_points_hom[:, 1])
m1_boundaries = get_line_boundaries(m1_line, lines)
plt.plot(m1_boundaries[0], m1_boundaries[1], "r--")

# get parameters of the line defined by the last two points
m2_line = get_line(orig_points_hom[:, 2], orig_points_hom[:, 3])
m2_boundaries = get_line_boundaries(m2_line, lines)
plt.plot(m2_boundaries[0], m2_boundaries[1], "b--")

# get the cross of the two lines
n1_cross = tb.p2e(get_cross(m1_line, m2_line))
if inside_img(n1_cross[0], n1_cross[1]):
    plt.plot(n1_cross[0], n1_cross[1], "og")

plt.show()

# get the transformed key points
tf_points = tb.p2e(H@(tb.e2p(orig_points)))
tf_boundaries = tb.p2e(H@(tb.e2p(orig_boundaries)))
tf_cross = tb.p2e(H@(tb.e2p(n1_cross)))
tf_m1_boundaries = tb.p2e(H@(tb.e2p(m1_boundaries)))
tf_m2_boundaries = tb.p2e(H@(tb.e2p(m2_boundaries)))

# plot the transformed image
plot_boundaries(tf_boundaries)
plot_points(tf_points)
if inside_img(n1_cross[0], n1_cross[1]):
    plt.plot(tf_cross[0], tf_cross[1], "og")
plt.plot(tf_m1_boundaries[0], tf_m1_boundaries[1], "r--")
plt.plot(tf_m2_boundaries[0], tf_m2_boundaries[1], "b--")

plt.gca().invert_yaxis()
plt.show()


