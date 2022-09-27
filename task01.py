import numpy as np
import matplotlib.pyplot as plt

# plt.ginput ... for entering the points
# np.cross ... cross product

# homography
H = np.array([[1,     0.1,   0],
              [0.1,   1,     0],
              [0.004, 0.002, 1]])

# plot the image boundary
bnds = np.array([[1, 1], [1, 800], [800, 800], [800, 1]])
plt.plot(bnds[:, 0], bnds[:, 1], "k")
plt.plot([bnds[-1, 0], bnds[0, 0]], [bnds[-1, 1], bnds[0, 1]], "k")

points = plt.ginput(4)

for i in range(4):
    if i < 2:
        plt.plot(points[i][0], points[i][1], "or")
    else:
        plt.plot(points[i][0], points[i][1], "ob")
    print(points[i])

# plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], "r--")
# plt.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], "b--")

plt.show()

ones_matrix = np.ones([4, 1])
points_hom = np.hstack([points, ones_matrix])
bnds_hom = np.hstack([bnds, ones_matrix])

points_tf = np.zeros([4, 2])
bnds_tf = np.zeros([4, 2])

for i in range(4):
    point_tf = H@points_hom[i].T
    points_tf[i] = point_tf[:2]/point_tf[2]
    bnd_tf = H@bnds_hom[i].T
    bnds_tf[i] = bnd_tf[:2]/bnd_tf[2]

print(bnds_tf)

plt.plot(bnds_tf[:, 0], bnds_tf[:, 1], "k")
plt.plot([bnds_tf[-1, 0], bnds_tf[0, 0]], [bnds_tf[-1, 1], bnds_tf[0, 1]], "k")

plt.gca().invert_yaxis()
plt.show()


