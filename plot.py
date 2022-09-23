import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(1, 101, 1)

f1 = np.zeros(100)
for i in range(100):
    f1[i] = math.log(x[i], 10)
plt.plot(x, f1, "k--")

f2 = np.zeros(100)
for i in range(100):
    f2[i] = math.log(math.factorial(x[i]), 10)
plt.plot(x, f2, "b")

f3 = np.zeros(100)
for i in range(100):
    f3[i] = math.pow(i, 2)
plt.plot(x, f3, "r")

plt.show()
