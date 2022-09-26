import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

# plt.ginput ... for entering the points
# np.cross ... cross product

# homography
H = np.array([[1,     0.1,   0],
              [0.1,   1,     0],
              [0.004, 0.002, 1]])

# plot the image boundary
plt.plot([1, 800, 800, 1, 1], [1, 1, 600, 600, 1], "k")

points = plt.ginput(4)

for i in range(4):
    if i < 2:
        plt.plot(points[i][0], points[i][1], "or")
    else:
        plt.plot(points[i][0], points[i][1], "ob")
    print(points[i])

plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], "r--")
plt.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], "b--")

plt.show()

