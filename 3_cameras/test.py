#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tools as tb

E = np.array([[-0.00412352, 0.25999315,-0.01447761],
              [-0.43448699,-0.01803185, 0.55733086],
              [ 0.03831558,-0.65649552, 0.00085193]])

R = np.array([[ 0.95885891, 0.03092146, 0.28219398],
              [-0.03395483, 0.99940616, 0.00586403],
              [-0.28184508,-0.01520463, 0.95933944]])

t = np.array([[-0.92970636],
              [-0.02358887],
              [-0.36754545]])

K = np.array([[2080,    0, 1421],
              [   0, 2080,  957],
              [   0,    0,    1]])

P1 = np.eye(3, 4)
P2 = np.hstack((R, t))
C1 = np.zeros((3, 1))
C2 = P2 @ np.vstack((C1, 1))
Cs = np.hstack((C1, C2))

z1 = np.array([0, 0, 1]).reshape(3, 1)
z2 = P2 @ np.vstack((z1, 1))
z = np.hstack((z1, z2))

img1 = mpimg.imread('cam1.jpg')
img2 = mpimg.imread('cam2.jpg')

# plot the centers of cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.gca().set_aspect('equal', adjustable='box')
ax.axis('off')

ax.plot(Cs[0, :], Cs[1, :], Cs[2, :], marker='o', c='red')
for i in range(Cs.shape[1]):
    ax.scatter(Cs[0, i], Cs[1, i], Cs[2, i], marker='o', s=20, c='black')

    # Adding point numbers
    ax.text(Cs[0, i], Cs[1, i], Cs[2, i], str(i+1), fontsize=10, c='black')

    ax.plot([Cs[0, i], z[0, i]],
            [Cs[1, i], z[1, i]],
            [Cs[2, i], z[2, i]], c='black')

plt.show()



# # for getting color from the img into the plot:
# x = np.random.randint(1, img1.shape[1])
# y = np.random.randint(1, img1.shape[0])
# [r, g, b] = img1[y, x]
# c_arr = (r/255.0, g/255.0, b/255.0)
# ax.scatter(Cs[0, i], Cs[1, i], Cs[2, i], marker='o', s=20, c=c_arr)  # c='black'
