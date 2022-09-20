import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

# plt.ginput ... for entering the points
# np.cross ... cross product

H = np.array([[1,     0.1,   0],
              [0.1,   1,     0],
              [0.004, 0.002, 1]])

plt.plot([1, 800, 800, 1, 1], [1, 1, 600, 600, 1], "k")
#plt.axis([0, 900, 0, 700])
plt.show()
