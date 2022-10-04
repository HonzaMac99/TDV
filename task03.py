import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb


# rng = np.random.default_rng()
# i = rng.choice(n, 2, replace=False)


x = np.loadtxt("linefit_1.txt").T
x = np.array(x)

plt.plot(x[0], x[1], 'ko', markersize=2)
plt.show()


