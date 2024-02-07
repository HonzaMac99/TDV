import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
matrix = np.random.rand(100, 100)
matrix[45:55, 45:55] = np.nan
nan_mask = np.isnan(matrix)
matrix[nan_mask] = 0

disparities = matrix[matrix.nonzero()].flatten()

u1 = np.vstack(matrix.nonzero())
u2 = u1 + np.vstack((disparities, np.zeros((1, u1.shape[1]))))

plt.imshow(matrix != 0, cmap='jet')
plt.colorbar()
plt.show()

time.sleep(1)
end = time.time()
print("Elapsed time:", end-start, "s")
