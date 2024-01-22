#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy.spatial.transform import Rotation as Rot

import os, sys
sys.path.append('..')
sys.path.append('corresp/python')
sys.path.append('p3p/python')
sys.path.append('geom_export/python')
import epipolar as ep
import tools as tb
import corresp
import p3p
import ge

# custom functions
from load import *
from plot import *
from geometry import *


class Camera:
    id = -1
    K = np.zeros((3, 3))
    P = np.zeros((3, 4))
    f = np.zeros((2, 1))
    img = None

    def __init__(self, cam_id):
        self.id = cam_id
        self.K = np.loadtxt('scene_1/K.txt', dtype='float')

    def is_valid(self):
        return self.id >= 0 and self.P.any() and self.f.any()


def init_cameras(n_cameras):
    cameras = []
    for i in range(n_cameras):
        new_cam = Camera(i)
        new_cam.f = get_feats(i)
        new_cam.img = get_img(i)
        cameras.append(new_cam)
    return np.array(cameras)


def init_c(n_cameras, verbose):
    c = corresp.Corresp(n_cameras)
    c.verbose = verbose
    for i in range(n_cameras):
        for j in range(i+1, n_cameras):
            corresps = get_corresps(i, j)
            c.add_pair(i, j, corresps)
    return c


# p3p ransac to get R, t
def get_new_cam(cam_id, Xs, Xs_crp, u_crp, K):
    ''' get new camera parameters such as R, t and inlier idxs of X-u corrspondences
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

    feats = get_feats(cam_id)

    best_support = 0
    best_R = np.zeros((3, 3))
    best_t = np.zeros((3, 1))
    best_inlier_idxs = []
    best_errors = np.zeros(len(Xs_crp))

    K_inv = np.linalg.inv(K)

    rng = np.random.default_rng()
    n_crp = len(Xs_crp)
    assert n_crp >= 3, "Number of corresp is less than 3!"

    k = 0
    k_max = np.inf
    theta = 3  # pixels
    probability = 0.9999
    k_max_reached = False
    while not k_max_reached:
        rand_idx = rng.choice(np.arange(n_crp), size=3, replace=False)

        assert len(Xs_crp) >= 3, "Not enough crp to estimate R, t!"

        Xs_triple = Xs[:, Xs_crp[rand_idx]]    # [4x3] - 3 random 3D scene points from Xs that are Xs_crp
        Us_triple = feats[u_crp[rand_idx]].T   # [2x3] - 3 corresponding img points from features (u_crp)

        Us_triple = K_inv @ tb.e2p(Us_triple)  # the points should be rectified first by K_inv for the p3p!

        if not Xs_triple[:3, :].any():
            print("The Xs triple is degenerate - points are zeros:")
            print(Xs_triple)
            continue

        Xs_local_arr = p3p.p3p_grunert(Xs_triple, Us_triple)  # compute local coords of the Xs_triple
        if not Xs_local_arr:
            continue
        for Xs_local in Xs_local_arr:
            R, t = p3p.XX2Rt_simple(Xs_triple, Xs_local)  # get R, t from glob Xs to the local 3D points

            # use P with K because the points are unrectified!
            # P1 = K @ np.eye(3, 4)
            P2 = K @ np.hstack((R, t))

            # We want to project 3D points to camera with P2 to compute the distance error between
            # projection and the original image correspondence to X
            X = Xs[:, Xs_crp]
            P2_X = P2 @ X  # precompute the projection

            u_orig = tb.e2p(feats[u_crp].T)
            u_new  = tb.e2p(tb.p2e(P2_X))


            support = 0
            inlier_idxs = []
            errors = np.zeros((n_crp))
            for i in range(n_crp):
                # Xi = X[:, i].reshape(4, 1)
                ui_orig = u_orig[:, i].reshape(3, 1)
                ui_new  = u_new[:, i].reshape(3, 1)

                # note: Xi is already in front of the P1
                # note: when support is < 3, one or more of the selected 3D points project behind the camera P2
                # note: checking "(K_inv @ P2 @ Xi)[2] > 0" or checking "(P2 @ Xi)[2] > 0" makes no difference

                # compute the sampson error only for points in front of the camera
                if P2_X[2, i] > 0:

                    # euclidean reprojection error
                    e_reprj = math.sqrt(
                        (ui_new[0]/ui_new[2] - ui_orig[0]/ui_orig[2])**2 +
                        (ui_new[1]/ui_new[2] - ui_orig[1]/ui_orig[2])**2)

                    errors[i] = e_reprj

                    if e_reprj**2 <= theta**2:
                        support += float(1 - e_reprj**2/theta**2)
                        inlier_idxs.append(i)
                        # note: c.join_camera works with indexes (i) of Xs_crp and not its elements to keep the inliers

            # store the best results and display stats
            if support > best_support:
                best_support = support
                best_R = R
                best_t = t
                best_inlier_idxs = inlier_idxs
                best_errors = errors

                # update the k_max
                w = (support + 1) / (n_crp + 1)
                k_max = math.log(1 - probability) / math.log(1 - w ** 3)
                print("[ k:", k+1, "/", k_max, "] [ support:", support, "/", n_crp, "]")

            k += 1
            if k >= k_max:
                k_max_reached = True
                break

        # assert k >= 1000, "TIMEOUT on " + str(k) + "-th iteration!! -------------------------"

        iter_treshold = 5000
        if iter_treshold - 50 < k < iter_treshold:
            print("Warning: too much iterations:", k)
        elif k >= iter_treshold:
            print("TIMEOUT on " + str(k) + "-th iteration!! -------------------------")
            break

    print("Result R\n", best_R)
    print("Result t\n", best_t)

    # plot the results of P transformation
    plot_transformation(best_R, best_t, best_inlier_idxs, Xs, Xs_crp, feats, u_crp)

    # display a histogram of the reprj. error distribution
    plot_error_hist(best_errors, theta, len(best_inlier_idxs))

    return best_R, best_t, best_inlier_idxs


if __name__ == '__main__':

    n_cams = 12
    cameras = init_cameras(n_cams)
    cam1, cam2 = cameras[[0, 1]]
    # cam1, cam2 = cameras[[7, 11]]
    K = np.loadtxt('scene_1/K.txt', dtype='float')
    K_inv = np.linalg.inv(K)

    c_verbose = 0  # 0 or 1
    c = init_c(n_cams, c_verbose)
    corresps = np.array(c.get_m(cam1.id, cam2.id)).T

    # perform the E estimation of initial camera pair (or load them from files)
    # do not stop between refresh() and save_E_params() calls!!
    if refresh(cam1.id, cam2.id):
        E, R, t, inls = ep.ransac_E(cam1.f, cam2.f, corresps, K)
        # todo: R, t = optimize_init_Rt(cam1, cam2, inls, R, t)
        save_E_params(E, R, t, inls)
    else:
        E, R, t, inls = load_E_params()
        print("E, R, t, inls loaded from files")

    F = K_inv.T @ E @ K_inv

    # ep.plot_inliers(cam1.img, cam1.f, cam2.f, corresps, inls)
    # ep.plot_e_lines(cam1.img, cam2.img, cam1.f, cam2.f, corresps, inls, F)

    Ps, Cs, zs = get_init_geometry(K, R, t)
    cam1.P, cam2.P = Ps

    # get initial 3d points and their corresponding colors
    Xs = get_init_3d_points(cam1, cam2, inls, K, True)
    colors = get_init_3d_colors(cam1, cam2, inls)
    X_ids = np.arange(Xs.shape[1])  # IDs of the reconstructed scene points

    c.start(cam1.id, cam2.id, inls, X_ids)
    n_cluster_cams = 2
    # n_cams = 7  # fewer cameras for testing

    # ==============================================================================
    # =                               Main loop                                    =
    # ==============================================================================

    while n_cluster_cams < n_cams:

        # why always n_Xu_crp <= n_tent_crp??
        tent_cams, n_Xu_crp = c.get_green_cameras()
        n_tent_crp, n_verif_crp = c.get_Xucount(tent_cams)  # scene to image corresp counts

        if tent_cams.shape[0] == 0:
            print("no more tentative cams")
            break

        # get new cam with the most tentative corresp
        new_cam_id = tent_cams[np.argmax(n_tent_crp)]
        new_cam = cameras[new_cam_id]
        print("best_cam is (from 0): ", new_cam_id)

        X_crp, u_crp, _ = c.get_Xu(new_cam.id)
        cam_crp_prev = np.array(c.get_m(new_cam.id, cam1.id)).T

        # get the transformation of the new camera from the global frame (cam1) by the p3p algorithm
        R, t, new_inls = get_new_cam(new_cam.id, Xs, X_crp, u_crp, K)
        # todo: R, t = optimize_new_Rt(cam1, cam2, new_inls, R, t)

        new_cam.P = K @ np.hstack((R, t))

        # add the new camera to the cluster, OK
        c.join_camera(new_cam.id, new_inls)

        # get the img-img crp and 3D-img crp after the call of camera join for the asserts
        X_crp_after, u_crp_after, _ = c.get_Xu(new_cam.id)
        cam_crp_after = np.array(c.get_m(new_cam.id, cam1.id)).T

        assert np.array_equal(u_crp_after, u_crp[new_inls]), "New image points do not match the inliers!"
        assert np.array_equal(X_crp_after, X_crp[new_inls]), "New 3D points do not match the inliers!"
        # check also the img-img correspondences?

        # get ids of cameras that still have img-img corresps to the new_cam
        nb_cam_list = c.get_cneighbours(new_cam.id)
        for nb_cam_id in nb_cam_list:
            nb_cam = cameras[nb_cam_id]
            assert nb_cam.is_valid(), "We should know the transformations of neighbors!"

            cam_crp = np.array(c.get_m(new_cam.id, nb_cam.id)).T
            print("Reconstructing from", cam_crp.shape[0], new_cam.id, "->", nb_cam.id, "img-img correspondences")

            P1 = new_cam.P
            P2 = nb_cam.P

            # triangulate 3D points from the known Ps of the camera pair
            new_Xs = get_new_3d_points(new_cam, nb_cam, cam_crp, K, True)
            new_colors = get_new_3d_colors(new_cam, nb_cam, cam_crp)

            # precompute the transformations here for faster checks
            P1_new_X = P1 @ new_Xs
            P2_new_X = P2 @ new_Xs

            u1_orig = tb.e2p(new_cam.f[cam_crp[:, 0]].T)
            u2_orig = tb.e2p(nb_cam.f[cam_crp[:, 1]].T)

            # get inliers with pozitive z
            inls = []  # must be indices to corresp between new_cam and nb_cam
            theta = 3
            for i in range(new_Xs.shape[1]):   # n of corresps, cam_crp.shape[0] could also be used
                # if P1_new_X[2, i] > 0 and P2_new_X[2, i] > 0:

                # euclidean reprojection error
                e_reprj_1 = math.sqrt(
                    (P1_new_X[0, i]/P1_new_X[2, i] - u1_orig[0, i]/u1_orig[2, i])**2 +
                    (P1_new_X[1, i]/P1_new_X[2, i] - u1_orig[1, i]/u1_orig[2, i])**2)
                e_reprj_2 = math.sqrt(
                    (P2_new_X[0, i]/P2_new_X[2, i] - u2_orig[0, i]/u2_orig[2, i])**2 +
                    (P2_new_X[1, i]/P2_new_X[2, i] - u2_orig[1, i]/u2_orig[2, i])**2)
                e_reprj = (e_reprj_1 + e_reprj_2) / 2

                if e_reprj**2 < theta**2:
                    new_X = new_Xs[:, i].reshape(4, 1)
                    Xs = np.hstack((Xs, new_X))     # append new 3D point
                    colors.append(new_colors[i])    # append new color
                    inls.append(i)                  # indexes to the cam_crp, OK

            Xs = np.hstack((Xs, new_Xs[:, inls]))

            X_ids = np.arange(len(inls)) + len(X_ids)  # IDs of the reconstructed scene points

            # store the new X-u corresps as selected for both cameras, delete the remaining u-u corresps, OK
            c.new_x(new_cam.id, nb_cam.id, inls, X_ids)

        # verify the tentative corresp emerged in the cluster --> the cluster should contain only verified X-u
        cam_cluster_list = c.get_selected_cameras()  # list of all cameras in the cluster
        for cl_cam_id in cam_cluster_list:
            cl_cam = cameras[cl_cam_id]
            [X_crp, u_crp, Xu_verified] = c.get_Xu(cl_cam_id)
            Xu_tentative = np.where(~Xu_verified)[0]
            print("Verifying", Xu_tentative.shape[0], new_cam.id, "-->", cl_cam.id, "tentative Xu correspondences")

            # precompute here for faster checks
            P_Xs = cl_cam.P @ Xs
            K_inv_P_Xs = K_inv @ P_Xs

            # verify by reprojection error tentative X-u corresps
            # loop through indexes of remaining unverified X-u correspondences
            crp_ok = []
            theta = 3
            for idx in Xu_tentative:
                # get the homogenous image point [u, v, 1]
                ui_orig = tb.e2p(cl_cam.f[u_crp[idx]].reshape(2, 1))

                # get the corresp. X projection and normalize it to be also homogenous
                P_Xi = P_Xs[:, X_crp[idx]].reshape(3, 1)
                ui_new = tb.e2p(tb.p2e(P_Xi))

                if K_inv_P_Xs[2, X_crp[idx]] > 0:
                    e_reprj = math.sqrt(
                        (ui_new[0]/ui_new[2] - ui_orig[0]/ui_orig[2])**2 +
                        (ui_new[1]/ui_new[2] - ui_orig[1]/ui_orig[2])**2)

                    if e_reprj**2 < theta**2:
                        crp_ok.append(idx) # save the indexes to X-u correspondences, OK

            # remove the outliers from remaining tentative X-u corresps, OK
            c.verify_x(cl_cam.id, crp_ok)

        c.finalize_camera()
        print("Camera", new_cam.id,"finalized")
        n_cluster_cams += 1

    # ==============================================================================
    # =                                Plotting                                    =
    # ==============================================================================

    # plot the centers of cameras
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=-125, azim=100, roll=180)
    ax.view_init(-75, -90)
    ax.set_xlim([-1.5, 2.5])
    ax.set_ylim([-3, 1])
    ax.set_zlim([-2, 2])
    # ax.axis('off')

    # get the remaining camera geometries
    cam_cluster_list = c.get_selected_cameras()  # list of all cameras in the cluster
    for i in range(2, len(cam_cluster_list)):
        idx = cam_cluster_list[i]
        new_C, new_z = get_new_geometry(K_inv, cameras[idx].P)
        Cs = np.hstack((Cs, new_C))
        zs = np.hstack((zs, new_z))

    # plot the camera centers and their axes
    for i in range(Cs.shape[1]):
        idx = cam_cluster_list[i]
        if idx == cam1.id or idx == cam2.id:
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
            ax.scatter(Xs[0, i], Xs[1, i], Xs[2, i], marker='o', s=3, color=colors[i])

    # plt.title(str(Xs.shape[1]) + " points")
    plt.title("Sparse pointcloud from 12 cameras")
    print(Xs.shape[1], "points")
    plt.show()

    # ==============================================================================
    # =                                 Saving                                     =
    # ==============================================================================

    Xs = tb.p2e(Xs)
    colors = np.array(colors).T

    # include the camera positions in the pointcloud
    Xs = np.hstack((Xs, Cs))

    red_c = np.array([1.0, 0, 0]).reshape(3, 1)
    blue_c = np.array([0, 0, 1.0]).reshape(3, 1)
    for i in range(Cs.shape[1]):
        idx = cam_cluster_list[i]
        if idx == cam1.id or idx == cam2.id:
            colors = np.hstack((colors, red_c))
        else:
            colors = np.hstack((colors, blue_c))

    g = ge.GePly('params/points.ply')
    g.points(Xs, colors) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.points(Xs) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    g.close()





