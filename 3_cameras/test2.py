import numpy as np

P = [None for i in range(10)]
P[1] = np.ones((3, 4))

X = np.array([1, 2, 3, 4])
X = X.reshape(4, 1)
print(X)

X2 = P[1] @ X
print(X2)

a = np.arange(10) + 10
print(a)