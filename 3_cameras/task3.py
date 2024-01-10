#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import sys
sys.path.append('..')
sys.path.append('p5/python')
sys.path.append('corresp/python')
sys.path.append('p3p/python')
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


# get colors from the images of the camera pair into the 3D plot:
def get_3d_colors(cam1, cam2, inls, sparsity=1):
    f1 = get_feats(cam1)            # cam1 features
    f2 = get_feats(cam2)            # cam2 features
    crp = get_corresps(cam1, cam2)  # cams corresp. features indexes
    img1 = get_img(cam1)            # img from cam1
    img2 = get_img(cam2)            # img from cam2
    colors = []
    for i in range(0, len(inls), sparsity):
        idx = inls[i]
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


def get_3d_points(cam1, cam2, inls, P1, P2, sparsity=1):
    f1 = get_feats(cam1)            # cam1 features
    f2 = get_feats(cam2)            # cam2 features
    crp = get_corresps(cam1, cam2)  # cams corresp. f. indexes
    Xs = np.zeros((4, 0))
    for i in range(0, len(inls), sparsity):
        idx = inls[i]
        u1 = tb.e2p(f1[crp[idx, 0]].reshape((2, 1)))
        u2 = tb.e2p(f2[crp[idx, 1]].reshape((2, 1)))
        X = tb.Pu2X(P1, P2, u1, u2)
        Xs = np.hstack((Xs, X))

    return Xs


# this function gets the ids of the remaining corresps in original corresps and returns it as inlier index array
def crp2inls(cam1, cam2, crp_remaining):
    crp_orig = get_corresps(cam1, cam2)  # cams corresp. f. indexes
    inls = []
    for i, crp in enumerate(crp_orig):
        # Todo: check if crp_remaining is ordered and then improve performance
        if crp in crp_remaining:
            inls.append(i)
    return inls


def get_new_3d_colors(cam1, cam2, crp, sparsity=1):
    f1 = get_feats(cam1)            # cam1 features
    f2 = get_feats(cam2)            # cam2 features
    img1 = get_img(cam1)            # img from cam1
    img2 = get_img(cam2)            # img from cam2
    colors = []
    for i in range(0, len(crp), sparsity):
        img1_point = f1[crp[i, 0]].reshape(2,)
        [x, y] = np.round(img1_point).astype(int)
        [r, g, b] = img1[y, x]
        color1 = (r/255.0, g/255.0, b/255.0)

        img2_point = f2[crp[i, 1]].reshape(2,)
        [x, y] = np.round(img2_point).astype(int)
        [r, g, b] = img2[y, x]
        color2 = (r/255.0, g/255.0, b/255.0)

        color = ((color1[0] + color2[0])/2,
                 (color1[1] + color2[1])/2,
                 (color1[2] + color2[2])/2)
        colors.append(color)
    return colors


def get_new_3d_points(cam1, cam2, crp, P1, P2):
    f1 = get_feats(cam1)            # cam1 features
    f2 = get_feats(cam2)            # cam2 features
    Xs = np.zeros((4, 0))
    for i in range(0, len(crp)):
        u1 = tb.e2p(f1[crp[i, 0]].reshape((2, 1)))
        u2 = tb.e2p(f2[crp[i, 1]].reshape((2, 1)))
        X = tb.Pu2X(P1, P2, u1, u2)
        Xs = np.hstack((Xs, X))

    return Xs


# get the projection matrices, camera centres and unit z axis point for a pair of cameras
def get_geometry(K, R, t):
    # camera projections working with uncalibrated points
    P1 = K @ np.eye(3, 4)
    P2 = K @ np.hstack((R, t))
    Ps = [P1, P2]  # do not use ndarray here

    # camera centers
    C1 = np.zeros((3, 1))
    C2 = -R.T @ t
    Cs = np.hstack((C1, C2))

    # unit points on z axis
    z1 = np.array([0, 0, 1]).reshape(3, 1)
    z2 = C2 + R[2, :].reshape(3, 1)
    zs = np.hstack((z1, z2))

    return Ps, Cs, zs


def get_new_geometry(K_inv, P):
    assert P is not None, "Got P that is None!"
    R = (K_inv @ P)[:, :3]
    t = (K_inv @ P)[:, 3]

    new_C = (-R.T @ t).reshape(3, 1)
    new_z = new_C + R[2, :].reshape(3, 1)
    return new_C, new_z


def get_corresps(i, j):
    i += 1
    j += 1
    if i > j:
        i, j = j, i
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
    feats = np.genfromtxt(path, dtype='float')
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


def get_new_cam(cam_id, Xs, Xs_crp, u_crp, K):
    ''' get new camera parameters such as R, t and inlier idxs
        in: cam_id  ... [int] id of the camera
        in: Xs      ... [4xn] homogenous array of all 3D points
        in: Xs_crp  ... [int array] of idxs for 3D points Xs, obtained from c.get_Xu(cam_id)
        in: u_crp   ... [int array] of idxs for features u, obtained from c.get_Xu(cam_id)
        in: K       ... [3x3 matrix] of calibration

        out: R      ... [3x3 matrix] of cam rotation to global Xs
        out: t      ... [3x1 vector] of cam translation
        out: inliers ... [int array] of indexes of Xs_crp --> inliers in Xs
    '''
    print("Estimating new cam R, t")

    feats1 = get_feats(0)
    feats2 = get_feats(cam_id)

    best_support = 0
    best_R = np.zeros((3, 3))
    best_t = np.zeros((3, 1))
    best_inlier_idxs = []

    K_inv = np.linalg.inv(K)

    rng = np.random.default_rng()
    n_crp = len(Xs_crp)
    assert n_crp >= 3, "Number of corresp is less than 3!"

    k = 0
    k_max = 100
    theta = 3  # pixels
    probability = 0.40  # 0.95
    while k <= k_max and k <= 100:
        rand_idx = rng.choice(np.arange(n_crp), size=3, replace=False)

        Xs_triple = np.zeros((4, 3))
        Us_triple = np.zeros((2, 3))
        for i, idx in enumerate(rand_idx):
            Xs_triple[:, i] = Xs[:, Xs_crp[idx]]
            Us_triple[:, i] = feats2[u_crp[idx]]
        Us_triple = K_inv @ tb.e2p(Us_triple)  # the points should be rectified first by K_inv!

        Xs_local_arr = p3p.p3p_grunert(Xs_triple, Us_triple)  # compute local coords of the 3D points
        if not Xs_local_arr:
            continue
        for Xs_local in Xs_local_arr:
            R, t = p3p.XX2Rt_simple(Xs_triple, Xs_local)  # get R, t from glob Xs to the local 3D points

            # use P with K because the points are unrectified!
            # P1 = K @ np.eye(3, 4)
            P2 = K @ np.hstack((R, t))

            support = 0
            inlier_idxs = []
            for i in range(n_crp):
                # we have corresponding features in cam2:
                # we want to project the corresponding global 3D points to compare them
                # compute the distance error from it

                Xi = Xs[:, Xs_crp[i]].reshape(4, 1)

                ui_orig = feats2[u_crp[i]].reshape(2, 1)
                ui_orig = tb.e2p(ui_orig)        # [u, v] --> [u, v, 1]
                ui_new = P2 @ Xi
                ui_new = tb.e2p(tb.p2e(ui_new))  # normalize to the homogenous

                # compute the sampson error only for points in front of the camera
                if (P2 @ Xi)[2] > 0:  # !!assignment: (K_inv @ P2 @ Xi)[2] > 0:
                    # note1: we check just P2, because the 3D points are already in front of the P1
                    # note2: we cannot obtain F at this place --> use reprj. error instead of sampson
                    e_reprj = math.sqrt(
                        (ui_new[0]/ui_new[2] - ui_orig[0]/ui_orig[2])**2 +
                        (ui_new[1]/ui_new[2] - ui_orig[1]/ui_orig[2])**2)

                    if e_reprj < theta:  # !!assinment: e_reprj**2 < theta**2
                        support += float(1 - e_reprj**2/theta**2)

                        # remember idxs (not elements) of Xs_crp set to be able
                        # to select and keep the inls in c.join_camera
                        inlier_idxs.append(i)

            if support > best_support:
                best_support = support
                best_R = R
                best_t = t
                best_inlier_idxs = inlier_idxs
                print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

            k += 1
            epsilon = 1 - (support+1)/(n_crp+1)
            k_max = math.log(1 - probability) / math.log(1 - (1 - epsilon)**3)
            k_max = min(k_max, 100)

    print("[ k:", k-1, "/", k_max, "] [ support:", best_support, "/", n_crp, "]\n")

    # print(inlier_idxs)
    print("Result R\n", best_R)
    print("Result t\n", best_t)

    return best_R, best_t, best_inlier_idxs


if __name__ == '__main__':
    cam1, cam2 = 0, 1
    img1, img2 = get_img(cam1), get_img(cam2)
    feats1, feats2 = get_feats(cam1), get_feats(cam2)
    K = np.loadtxt('scene_1/K.txt', dtype='float')
    K_inv = np.linalg.inv(K)
    # F = K_inv.T @ E @ K_inv

    n_cams = 12
    P_arr = [None] * n_cams
    c = init_c(n_cams)
    corresps = np.array(c.get_m(0, 1)).T

    # perform the actual E estimation
    E, R, t, inls = ep.ransac_E(feats1, feats2, corresps, K)

    # ep.plot_inliers(img1, feats1, feats2, corresps, inls)
    # ep.plot_e_lines(img1, img2, feats1, feats2, corresps, inls, F)

    Ps, Cs, zs = get_geometry(K, R, t)
    P_arr[cam1] = Ps[0]
    P_arr[cam2] = Ps[1]

    # get initial 3d points and their corresponding colors
    Xs = get_3d_points(cam1, cam2, inls, Ps[0], Ps[1])
    colors = get_3d_colors(cam1, cam2, inls)
    X_ids = np.arange(Xs.shape[1])  # IDs of the reconstructed scene points

    c.start(0, 1, inls, X_ids)
    n_cluster_cams = 2

    # main loop
    while n_cluster_cams < n_cams:
        # break

        # why always n_Xu_crp <= n_tent_crp??
        tent_cams, n_Xu_crp = c.get_green_cameras()
        n_tent_crp, n_verif_crp = c.get_Xucount(tent_cams)  # scene to image corresp counts

        if tent_cams.shape[0] == 0:
            print("no more tentative cams")
            break

        # get new cam with the most tentative corresp
        new_cam = tent_cams[np.argmax(n_tent_crp)]
        print("best_cam is (from 0): ", new_cam)

        # get the transformation of the new camera from the global frame (cam1) by the p3p algorithm
        X_crp, u_crp, _ = c.get_Xu(new_cam)
        R, t, new_inls = get_new_cam(new_cam, Xs, X_crp, u_crp, K)
        P_arr[new_cam] = K @ np.hstack((R, t))

        # add the new camera to the cluster, OK
        c.join_camera(new_cam, new_inls)

        # get ids of cameras that still have corresp m[][] to the new_cam
        nb_cam_list = c.get_cneighbours(new_cam)

        for nb_cam in nb_cam_list:
            # we must know the transformations
            if P_arr[nb_cam] is None:
                continue
            cam_crp = np.array(c.get_m(new_cam, nb_cam)).T

            P1 = P_arr[new_cam]
            P2 = P_arr[nb_cam]

            # triangulate 3D points from the known Ps of the camera pair
            # inls = crp2inls(new_cam, nb_cam, cam_crp)
            new_Xs = get_new_3d_points(new_cam, nb_cam, cam_crp, P1, P2)
            new_colors = get_new_3d_colors(new_cam, nb_cam, cam_crp)

            # get inliers with pozitive z
            inls = []  # must be indices to corresp between new_cam and nb_cam
            for i in range(new_Xs.shape[1]):   # or cam_crp.shape[0], n of corresps
                new_X = new_Xs[:, i].reshape(4, 1)
                if (P1 @ new_X)[2] >= 0 and (P2 @ new_X)[2] >= 0:
                    Xs = np.hstack((Xs, new_X))     # append new 3D point
                    colors.append(new_colors[i])    # append new color
                    inls.append(i)                  # indexes to the cam_crp, OK

            X_ids = np.arange(len(inls)) + len(X_ids)  # IDs of the reconstructed scene points

            # store the new X-u corresps as selected for both cameras, delete the remaining u-u corresps, OK
            c.new_x(new_cam, nb_cam, inls, X_ids)

        # verify the tentative corresp emerged in the cluster --> the cluster should contain only verified X-u
        cam_cluster_list = c.get_selected_cameras()  # list of all cameras in the cluster
        for cl_cam in cam_cluster_list:
            [X_crp, u_crp, Xu_verified] = c.get_Xu(cl_cam)
            Xu_tentative = np.where(~Xu_verified)[0]

            # verify by reprojection error tentative X-u corresps
            cl_feats = get_feats(cl_cam)
            cl_P = P_arr[cl_cam]

            # loop through indexes of remaining unverified X-u correspondences
            crp_ok = []
            theta = 3
            for idx in Xu_tentative:
                Xi = Xs[:, X_crp[idx]].reshape(4, 1)

                ui_orig = cl_feats[u_crp[idx]].reshape(2, 1)
                ui_orig = tb.e2p(ui_orig)        # [u, v] --> [u, v, 1]

                ui_new = cl_P @ Xi
                ui_new = tb.e2p(tb.p2e(ui_new))  # normalize to the homogenous

                if (K_inv @ cl_P @ Xi)[2] > 0:
                    e_reprj = math.sqrt(
                        (ui_new[0]/ui_new[2] - ui_orig[0]/ui_orig[2])**2 +
                        (ui_new[1]/ui_new[2] - ui_orig[1]/ui_orig[2])**2)

                    if e_reprj < theta:  # !!assinment: e_reprj**2 < theta**2
                        # save the indexes to X-u correspondences, OK
                        crp_ok.append(idx)

            # remove the outliers from remaining tentative X-u corresps, OK
            c.verify_x(cl_cam, crp_ok)

        c.finalize_camera()
        print("camera finalized")
        n_cluster_cams += 1
        # break

    # ==============================================================================

    # plot the centers of cameras
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=-125, azim=100, roll=180)
    ax.set_xlim([-1.5, 2.5])
    ax.set_ylim([-3, 1])
    ax.set_zlim([-2, 2])
    # ax.axis('off')

    # get the remaining camera geometries
    cam_cluster_list = c.get_selected_cameras()  # list of all cameras in the cluster
    for i in range(2, len(cam_cluster_list)):
    # for idx in range(2, n_cams):
        idx = cam_cluster_list[i]
        new_C, new_z = get_new_geometry(K_inv, P_arr[idx])
        Cs = np.hstack((Cs, new_C))
        zs = np.hstack((zs, new_z))

    # plot the camera centers and their axes
    for i in range(Cs.shape[1]):
        idx = cam_cluster_list[i]
        if i < 2:
            cam_color = 'red'
        else:
            cam_color = 'blue'
        ax.scatter(Cs[0, i], Cs[1, i], Cs[2, i], marker='o', s=20, c=cam_color)
        ax.text(Cs[0, i], Cs[1, i], Cs[2, i], str(idx+1), fontsize=10, c=cam_color)
        ax.plot([Cs[0, i], zs[0, i]],  # plot the camera z axis
                [Cs[1, i], zs[1, i]],
                [Cs[2, i], zs[2, i]], c=cam_color)

    # plot the 3D points with colors
    sparsity = 20
    for i in range(Xs.shape[1]):
        if i % sparsity == 0:
            ax.scatter(Xs[0, i], Xs[1, i], Xs[2, i], marker='o', s=5, color=colors[i])
            # ax.scatter(Xs[0, i], Xs[1, i], Xs[2, i], marker='o', s=3, color="black")

    plt.show()











    # g = ge.GePly('out.ply')
    # colors = np.array(colors).T
    # Xs = tb.e2p(Xs)
    # # g.points(Xs, colors) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.points(Xs) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.close()





