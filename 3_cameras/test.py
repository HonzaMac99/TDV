#!/usr/bin/env python3
import numpy as np
# import math
import matplotlib.pyplot as plt

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680801)

# N_points = 100000
# n_bins = 20

N_points = 20
n_bins = 20

# arr = np.array([1, 2, 3, 4, 4, 4, 4, 5, 6, 2, 4, 1])
arr = rng.standard_normal(N_points)

# Generate two normal distributions
dist1 = rng.standard_normal(N_points)

fig, ax = plt.subplots(1, 1, tight_layout=True)
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
ax.hist(arr, bins=n_bins)
# axs[1].hist(dist2, bins=n_bins)

plt.show()