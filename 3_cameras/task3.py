#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import sys
sys.path.append('..')
sys.path.append('p5/python')
sys.path.append('corresp/python')
sys.path.append('geom_export/python')
import epipolar as ep
import tools as tb
import p5
import corresp
import p3p
import ge


# get the color from the images of the camera pair into the plot:
def get_3d_colors(f1, f2, crp, inliers_idx, img1, img2, sparsity=1):
    colors = []
    for i in range(0, len(inliers_idx), sparsity):
        idx = inliers_idx[i]
        img1_point = f1[crp[idx, 0]].reshape(2,)
        [x, y] = np.round(img1_point).astype(int)
        [r, g, b] = img1[y, x]
        color1 = (r/255.0, g/255.0, b/255.0)

        img2_point = f2[crp[idx, 1]].reshape(2,)
        [x, y] = np.round(img2_point).astype(int)
        [r, g, b] = img2[y, x]
        color2 = (r/255.0, g/255.0, b/255.0)

        color = ((color1[0] + color2[0])/2,
                 (color1[1] + color2[1])/2,
                 (color1[2] + color2[2])/2)
        colors.append(color)
    return colors


def get_3d_points(features1, features2, corresp, inliers_idx, P1, P2, sparsity=1):
    Xs = np.zeros((4, 0))
    for i in range(0, len(inliers_idx), sparsity):
        idx = inliers_idx[i]
        u1 = tb.e2p(features1[corresp[idx, 0]].reshape((2, 1)))
        u2 = tb.e2p(features2[corresp[idx, 1]].reshape((2, 1)))
        X = tb.Pu2X(P1, P2, u1, u2)
        Xs = np.hstack((Xs, X))

    return Xs


def get_fname(i, j):
    f_name = ''
    if i < 10 and j < 10:
        f_name = 'm_0{}_0{}.txt'.format(i, j)
    elif i >= 10 and j < 10:
        f_name = 'm_{}_0{}.txt'.format(i, j)
    elif i < 10 and j >= 10:
        f_name = 'm_0{}_{}.txt'.format(i, j)
    elif i >= 10 and j >= 10:
        f_name = 'm_{}_{}.txt'.format(i, j)
    return f_name


def init_corresp(n_cameras):
    c = corresp.Corresp(n_cameras)
    c.verbose = 2
    for i in range(n_cameras):
        for j in range(i+1, n_cameras):
            f_name = get_fname(i+1, j+1)
            path = 'scene_1/corresp/{}'.format(f_name)
            corresps = np.genfromtxt(path, dtype='int')
            c.add_pair(i, j, corresps)
    return c


if __name__ == '__main__':
    img1 = mpimg.imread('scene_1/images/01.jpg')
    img2 = mpimg.imread('scene_1/images/02.jpg')
    features1 = np.genfromtxt('scene_1/corresp/u_01.txt', dtype='float')
    features2 = np.genfromtxt('scene_1/corresp/u_02.txt', dtype='float')
    # corresps = np.genfromtxt('scene_1/corresp/m_01_02.txt', dtype='int')
    K = np.loadtxt('scene_1/K.txt', dtype='float')

    n_cams = 12
    c = init_corresp(n_cams)
    corresps = np.array(c.get_m(0, 1)).T

    # perform the actual E estimation
    E, R, t, inls = ep.ransac_E(features1, features2, corresps, K)

    # K_inv = np.linalg.inv(K)
    # F = K_inv.T @ E @ K_inv
    # ep.plot_inliers(img1, features1, features2, corresps, inls)
    # ep.plot_e_lines(img1, img2, features1, features2, corresps, inls, F)

    I = np.eye(3, 3)
    P1 = K @ np.eye(3, 4)
    P2 = K @ np.hstack((R, t))

    # camera centers
    C1 = np.zeros((3, 1))
    C2 = -R.T @ t
    Cs = np.hstack((C1, C2))
    # C2 = K_inv @ P2 @ np.vstack((C1, 1))

    # unit points on z axis
    z1 = np.array([0, 0, 1]).reshape(3, 1)
    z2 = C2 + R[2, :].reshape(3, 1)
    z = np.hstack((z1, z2))
    # z2 = K_inv @ P2 @ np.vstack((z1, 1))

    # get the 3d point and their corresponding colors
    Xs = get_3d_points(features1, features2, corresps, inls, P1, P2)
    colors = get_3d_colors(features1, features2, corresps, inls, img1, img2)

    X_ids = np.array([i for i in range(Xs.shape[1])])  # IDs of the reconstructed scene points
    c.start(0, 1, inls, X_ids)

    tent_cams = c.get_green_cameras()
    max_tent_crp = 0
    best_cam = None
    for cam in tent_cams:
        tent_crp = c.get_Xucount(cam)
        if tent_crp > max_tent_crp:
            max_tent_crp = tent_crp
            best_cam = cam

    best_cam = 3
    [X, u] = c.get_Xu(best_cam)

    # todo: implement p3p with ransac
    new_inls = []

    c.join_camera(best_cam, new_inls)

    nb_cam_list = c.get_cneighbours(best_cam)
    for nb_cam in nb_cam_list:
        cam_crp = c.get_m(best_cam, nb_cam)

        # todo: reconstruct 3d points
        Xs = get_3d_points(features1, features2, corresps, inls, P1, P2)
        colors = get_3d_colors(features1, features2, corresps, inls, img1, img2)

        # todo: for inliers check only in front camera position?
        inls = np.array([i for i in range(Xs.shape[1])])
        X_ids = np.array([i + len(X_ids) for i in range(Xs.shape[1])])  # IDs of the reconstructed scene points

        c.new_x(best_cam, nb_cam, inls, X_ids)

        # todo: verify the tentative corresp





    # ==============================================================================

    # plot the centers of cameras
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-125, azim=100, roll=180)
    ax.set_xlim([-1.5, 2.5])
    ax.set_ylim([-3, 1])
    ax.set_zlim([-2, 2])
    # ax.axis('off')

    # plot the camera centers and the baseline
    ax.plot(Cs[0, :], Cs[1, :], Cs[2, :], marker='o', c='red')
    for i in range(Cs.shape[1]):
        ax.scatter(Cs[0, i], Cs[1, i], Cs[2, i], marker='o', s=20, c='black')
        ax.text(Cs[0, i], Cs[1, i], Cs[2, i], str(i+1), fontsize=10, c='black')
        ax.plot([Cs[0, i], z[0, i]],  # plot the camera z axis
                [Cs[1, i], z[1, i]],
                [Cs[2, i], z[2, i]], c='black')

    # plot the 3D points with colors
    for i in range(Xs.shape[1]):
        ax.scatter(Xs[0, i], Xs[1, i], Xs[2, i], marker='o', s=5, color=colors[i])

    plt.show()











    # g = ge.GePly('out.ply')
    # colors = np.array(colors).T
    # # g.points(Xs, colors) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.points(Xs) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.close()





