import numpy as np
import matplotlib.pyplot as plt

# plt.ginput ... for entering the points
# np.cross ... cross product

# homography
H = np.array([[1,     0.1,   0],
              [0.1,   1,     0],
              [0.004, 0.002, 1]])


# plot the image boundary
def plot_boundaries(boundaries):
    plt.plot(np.hstack([boundaries[:, 0], boundaries[0, 0]]),
             np.hstack([boundaries[:, 1], boundaries[0, 1]]), "k")


def plot_points(points):
    if len(points) != 4:
        print("Wrong number of points:", len(points))
        return
    for i in range(4):
        if i < 2:
            plt.plot(points[i][0], points[i][1], "or")
        else:
            plt.plot(points[i][0], points[i][1], "ob")


def find_line_params(point1, point2):
    a = np.array([[np.hstack([point1, 1])],
                  [np.hstack([point2, 1])]])
    print(a)
    b = np.array([0, 0])
    return np.linalg.solve(a, b)

# def find_cross(params1, params2):


bnds = np.array([[1, 1], [1, 800], [800, 800], [800, 1]])
plot_boundaries(bnds)

pts = plt.ginput(4)
plot_points(pts)
my_line = find_line_params(pts[0], pts[1])
my_line2 = find_line_params(pts[2], pts[3])
print(my_line)
print(my_line2)

# plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], "r--")
# plt.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], "b--")

plt.show()

ones_matrix = np.ones([4, 1])
points_hom = np.hstack([pts, ones_matrix])
bnds_hom = np.hstack([bnds, ones_matrix])

points_tf = np.zeros([4, 2])
bnds_tf = np.zeros([4, 2])

for i in range(4):
    point_tf = H@points_hom[i].T
    points_tf[i] = point_tf[:2]/point_tf[2]
    bnd_tf = H@bnds_hom[i].T
    bnds_tf[i] = bnd_tf[:2]/bnd_tf[2]

print(bnds_tf)

plot_boundaries(bnds_tf)
plot_points(points_tf)


plt.gca().invert_yaxis()
plt.show()


