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


class Camera:
    def __init__(self, cam_id):
        self.id = cam_id


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

def get_geometry(R, t):
    P1 = K @ np.eye(3, 4)
    P2 = K @ np.hstack((R, t))
    Ps = np.hstack((P1, P2))

    # camera centers
    C1 = np.zeros((3, 1))
    C2 = -R.T @ t
    Cs = np.hstack((C1, C2))

    # unit points on z axis
    z1 = np.array([0, 0, 1]).reshape(3, 1)
    z2 = C2 + R[2, :].reshape(3, 1)
    zs = np.hstack((z1, z2))

    return Ps, Cs, zs


def get_corresps(i, j):
    i += 1
    j += 1
    if i < 10 and j < 10:
        f_name = 'm_0{}_0{}.txt'.format(i, j)
    elif i >= 10 and j < 10:
        f_name = 'm_{}_0{}.txt'.format(i, j)
    elif i < 10 and j >= 10:
        f_name = 'm_0{}_{}.txt'.format(i, j)
    else:
        f_name = 'm_{}_{}.txt'.format(i, j)

    path = 'scene_1/corresp/{}'.format(f_name)
    corresps = np.genfromtxt(path, dtype='int')
    return corresps

def get_feats(i):
    i += 1
    if i < 10:
        f_name = 'u_0{}.txt'.format(i)
    else:
        f_name = 'u_{}.txt'.format(i)

    path = 'scene_1/corresp/{}'.format(f_name)
    feats = np.genfromtxt(path, dtype='int')
    return feats

def get_img(i):
    i += 1
    if i < 10:
        f_name = '0{}.jpg'.format(i)
    else:
        f_name = '{}.jpg'.format(i)

    img = mpimg.imread('scene_1/images/{}'.format(f_name))
    return img


def init_c(n_cameras):
    c = corresp.Corresp(n_cameras)
    c.verbose = 2
    for i in range(n_cameras):
        for j in range(i+1, n_cameras):
            corresps = get_corresps(i, j)
            c.add_pair(i, j, corresps)
    return c


def get_new_cam(cam_id, Xs, X_corresp, u_corresp, K):
    ''' get new camera parameters such as R, t and inlier idxs
        in: Xs ... [4xn] homogenous 3D points
        in: X_corresp ... array with idxs for Xs
        in: u_corresp ... array with idxs for features
    '''
    print("Estimating new cam R, t")

    best_support = 0
    best_R = np.zeros((3, 3))
    best_t = np.zeros((3, 1))
    best_inlier_idxs = []

    K_inv = np.linalg.inv(K)

    rng = np.random.default_rng()
    n_crp = corresp.shape[0]
    k = 0
    k_max = 100
    theta = 3  # pixels
    probability = 0.40  # 0.95
    while k <= k_max:
        n = len(X_corresp)
        rng_idx = rng.choice(np.arange(n), size=3, replace=False)

        Xw = np.zeros((4, 3))
        U = np.zeros((3, 3))
        features = get_feats(cam_id)
        for i in range(3):
            Xw[:, i] = Xs[rng_idx[i]]
            U[:, i] = features[u_corresp[rng_idx[i]]]
        # the points should be rectified first!
        U_hom = K_inv @ tb.e2p(U)

        Xc = p3p.p3p_grunert(Xw, U_hom)
        R, t = p3p.XX2Rt_simple(Xs, Xc)

        # use P with K because the points are unrectified!
        P1 = K @ np.eye(3, 4)
        P2 = K @ np.hstack((R, t))

        support = 0
        inlier_idxs = []
        for i in range(n_crp):
            u1 = tb.e2p(features1[corresp[i, 0]].reshape((2, 1)))
            u2 = tb.e2p(features2[corresp[i, 1]].reshape((2, 1)))
            X = tb.Pu2X(P1, P2, u1, u2)

            # compute the sampson error only for points in front of the camera
            if (P1 @ X)[2] >= 0 and (P2 @ X)[2] >= 0:
                e_sampson = tb.err_F_sampson(F, u1, u2)
                if e_sampson < theta:
                    support += float(1 - e_sampson**2/theta**2)
                    inlier_idxs.append(i)

        if support > best_support:
            best_support = support
            best_R = R
            best_t = t
            best_inlier_idxs = inlier_idxs
            print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

        k += 1
        w = (support + 1) / n_crp
        k_max = math.log(1 - probability) / math.log(1 - w ** 2)

    print("[ k:", k-1, "/", k_max, "] [ support:", best_support, "/", n_crp, "]\n")

    # print(inlier_idxs)
    print("Result R\n", best_R)
    print("Result t\n", best_t)

    return best_R, best_t, best_inlier_idxs


if __name__ == '__main__':
    img1 = get_img(0)
    img2 = get_img(1)
    features1 = get_feats(0)
    features2 = get_feats(1)
    # corresps = get_corresps(0, 1)
    K = np.loadtxt('scene_1/K.txt', dtype='float')

    n_cams = 12
    c = init_c(n_cams)
    corresps = np.array(c.get_m(0, 1)).T

    # perform the actual E estimation
    E, R, t, inls = ep.ransac_E(features1, features2, corresps, K)

    # K_inv = np.linalg.inv(K)
    # F = K_inv.T @ E @ K_inv
    # ep.plot_inliers(img1, features1, features2, corresps, inls)
    # ep.plot_e_lines(img1, img2, features1, features2, corresps, inls, F)

    P, C, z = get_geometry(R, t)

    # get initial 3d points and their corresponding colors
    Xs = get_3d_points(features1, features2, corresps, inls, P[0], P[1])
    X_ids = np.array([i for i in range(Xs.shape[1])])  # IDs of the reconstructed scene points
    colors = get_3d_colors(features1, features2, corresps, inls, img1, img2)

    c.start(0, 1, inls, X_ids)

    n_cluster_cams = 2
    while n_cluster_cams < n_cams:

        tent_cams = c.get_green_cameras()
        if not tent_cams:
            print("no more tentative cams")
            break

        # get tent cam with the most corresp
        max_tent_crp = 0
        new_cam = 0
        for cam in tent_cams:
            n_tent_crp = c.get_Xucount(cam)
            if n_tent_crp > max_tent_crp:
                max_tent_crp = n_tent_crp
                new_cam = cam

        # best_cam = 3
        print("best_cam is: ", new_cam)
        [X, u] = c.get_Xu(new_cam)


        # todo: implement p3p with ransac
        R, t, new_inls = get_new_cam()

        c.join_camera(new_cam, new_inls)
        nb_cam_list = c.get_cneighbours(new_cam)

        for nb_cam in nb_cam_list:
            cam_crp = c.get_m(new_cam, nb_cam)

            # todo: reconstruct 3d points
            Xs = get_3d_points(features1, features2, corresps, inls, P1, P2)
            colors = get_3d_colors(features1, features2, corresps, inls, img1, img2)

            # todo: for inliers check only in front camera position?
            inls = np.array([i for i in range(Xs.shape[1])])
            X_ids = np.array([i + len(X_ids) for i in range(Xs.shape[1])])  # IDs of the reconstructed scene points

            c.new_x(best_cam, nb_cam, inls, X_ids)

            # todo: verify the tentative corresp


        n_cluster_cams += 1




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





