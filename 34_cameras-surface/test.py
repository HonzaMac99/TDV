#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import tools as tb
from task3_cameras import *

# testing the settings of the plot of the final pointcloud and cameras

Xs = tb.p2e(np.load('params/Xs.npy'))
colors = np.load('params/colors.npy').T
Cs = np.load('params/Cs.npy')
zs = np.load('params/zs.npy')
cam1_id, cam2_id = np.load('params/init_cams.npy')

# plot the centers of cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-125, azim=100, roll=180)
# ax.view_init(-75, -90)
ax.set_xlim([-1.5, 2.5])
ax.set_ylim([-3, 1])
ax.set_zlim([-2, 2])
# ax.axis('off')

# plot the camera centers and their axes
for i in range(Cs.shape[1]):
    idx = i
    if i == cam1_id or i == cam2_id:
        cam_color = 'red'
    else:
        cam_color = 'blue'
    ax.scatter(Cs[0, i], Cs[1, i], Cs[2, i], marker='o', s=20, c=cam_color)
    ax.text(Cs[0, i], Cs[1, i], Cs[2, i], str(idx+1), fontsize=10, c=cam_color)
    ax.plot([Cs[0, i], zs[0, i]],  # plot the camera z axis
            [Cs[1, i], zs[1, i]],
            [Cs[2, i], zs[2, i]], c=cam_color)

# plot the 3D points with colors
sparsity = 1  # 50
for i in range(Xs.shape[1]):
    if i % sparsity == 0:
        ax.scatter(Xs[0, i], Xs[1, i], Xs[2, i], marker='o', s=3, color=colors[:, i])

# plt.title(str(Xs.shape[1]) + " points")
plt.title("Sparse pointcloud from 12 cameras")
print(Xs.shape[1], "points")
plt.show()

