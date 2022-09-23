import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

# plt.ginput ... for entering the points
# np.cross ... cross product

# test for changes

H = np.array([[1,     0.1,   0],
              [0.1,   1,     0],
              [0.004, 0.002, 1]])

plt.plot([1, 800, 800, 1, 1], [1, 1, 600, 600, 1], "k")
# plt.axis([0, 900, 0, 700])

points = plt.ginput(5)

for i in range(4):
    plt.plot(points[i][0], points[i][1], "ob")
    print(points[i])

plt.plot(200, 200, "r")

plt.show()
