#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import optimize as opt

import sys
sys.path.append('..')
sys.path.append('corresp/python')
sys.path.append('p3p/python')
sys.path.append('geom_export/python')
import epipolar as ep
import tools as tb
import corresp
import p3p
import ge


class Camera:
    def __init__(self, cam_id):
        self.id = cam_id


def correct_crp_sampson(P1, P2, u1, u2):
    Q1 = P1[:, :3]
    Q2 = P2[:, :3]

    q1 = P1[:, 3].reshape(3, 1)
    q2 = P2[:, 3].reshape(3, 1)

    Q12 = Q1 @ np.linalg.inv(Q2)

    # compute F from P1, P2 - slides 108
    F = Q12.T @ tb.sqc(q1 - Q12 @ q2)
    return F
    # nu1, nu2 = tb.u_correct_sampson(F, u1, u2)

    # return nu1, nu2


def correct_crp_sampson_2(P1, P2, u1, u2):
    K = np.loadtxt('scene_1/K.txt', dtype='float')
    K_inv = np.linalg.inv(K)
    R1 = (K_inv @ P1)[:, :3]
    R2 = (K_inv @ P2)[:, :3]
    R21 = R2 @ R1.T
    t21 = P2[:, 3] - R21 @ P1[:, 3]

    # compute F from P1, P2 using K, Rs and ts - slides 79
    F = K_inv.T @ tb.sqc(-t21) @ R21 @ K_inv
    return F
    # nu1, nu2 = tb.u_correct_sampson(F, u1, u2)

    # return nu1, nu2


# get colors from the images of the camera pair into the 3D plot:
def get_3d_colors(cam1, cam2, inls):
    f1 = get_feats(cam1)            # cam1 features
    f2 = get_feats(cam2)            # cam2 features
    crp = get_corresps(cam1, cam2)  # cams corresp. features indexes
    img1 = get_img(cam1)            # img from cam1
    img2 = get_img(cam2)            # img from cam2

    img1_points = f1[crp[:, 0]].T
    img2_points = f2[crp[:, 1]].T

    colors = []
    for i in range(0, len(inls)):
        [x, y] = np.round(img1_points[:, i]).astype(int)
        [r, g, b] = img1[y, x]
        color1 = (r/255.0, g/255.0, b/255.0)

        [x, y] = np.round(img2_points[:, i]).astype(int)
        [r, g, b] = img2[y, x]
        color2 = (r/255.0, g/255.0, b/255.0)

        color = ((color1[0] + color2[0])/2,
                 (color1[1] + color2[1])/2,
                 (color1[2] + color2[2])/2)
        colors.append(color)
    return colors


def get_3d_points(cam1, cam2, inls, P1, P2):
    f1 = get_feats(cam1)            # cam1 features
    f2 = get_feats(cam2)            # cam2 features
    crp = get_corresps(cam1, cam2)  # cams corresp. f. indexes

    # pick features u1, u2 at indexes defined by inlier corresps
    u1 = tb.e2p(f1[crp[inls, 0]].T)
    u2 = tb.e2p(f2[crp[inls, 1]].T)

    # fig, ax = plt.subplots(1, 1)
    # ax.invert_yaxis()
    # ax.scatter(u1[0, :], u1[1, :], marker='o', color='blue', s=20)

    # correct the correspondences using the sampson error correction
    # nu1, nu2 = correct_crp_sampson(P1, P2, u1, u2)

    # ax.scatter(nu1[0, :], nu1[1, :], marker='x', color='orange', s=20)
    # plt.show()

    R = (K_inv @ P2)[:, :3]
    t = (K_inv @ P2)[:, 3]

    F1 = correct_crp_sampson(P1, P2, u1, u2)
    F2 = correct_crp_sampson_2(P1, P2, u1, u2)

    Xs = tb.Pu2X(P1, P2, u1, u2)

    return Xs


def get_new_3d_colors(cam1, cam2, crp):
    f1 = get_feats(cam1)            # cam1 features
    f2 = get_feats(cam2)            # cam2 features
    img1 = get_img(cam1)            # img from cam1
    img2 = get_img(cam2)            # img from cam2

    img1_points = f1[crp[:, 0]].T
    img2_points = f2[crp[:, 1]].T

    colors = []
    for i in range(0, len(crp)):
        [x, y] = np.round(img1_points[:, i]).astype(int)
        [r, g, b] = img1[y, x]
        color1 = (r/255.0, g/255.0, b/255.0)

        [x, y] = np.round(img2_points[:, i]).astype(int)
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

    u1 = tb.e2p(f1[crp[:, 0]].T)
    u2 = tb.e2p(f2[crp[:, 1]].T)

    R = P2[:, :3]
    t = P2[:, 3]

    # correct the correspondences using the sampson error correction
    # nu1, nu2 = correct_crp_sampson(P1, P2, u1, u2)
    # nu1, nu2 = correct_crp_sampson_2(K, R, t, u1, u2)

    K = np.loadtxt('scene_1/K.txt', dtype='float')
    R = (K_inv @ P2)[:, :3]
    t = (K_inv @ P2)[:, 3]

    F1 = correct_crp_sampson(P1, P2, u1, u2)
    F2 = correct_crp_sampson_2(K, R, t, u1, u2)

    Xs = tb.Pu2X(P1, P2, u1, u2)

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


def init_c(n_cameras, verbose):
    c = corresp.Corresp(n_cameras)
    c.verbose = verbose
    for i in range(n_cameras):
        for j in range(i+1, n_cameras):
            corresps = get_corresps(i, j)
            c.add_pair(i, j, corresps)
    return c


# plot the comparison between the original features and the
# projections of corresponding 3D points (Xs) by the best estimated R and t
def plot_transformation(best_R, best_t, best_inlier_idxs, Xs, Xs_crp, feats, u_crp):
    fig, ax = plt.subplots(1, 1)
    ax.invert_yaxis()

    best_P = K @ np.hstack((best_R, best_t))
    n_crp = len(Xs_crp)

    best_outlier_idxs = np.ones(n_crp)
    best_outlier_idxs[best_inlier_idxs] = False
    best_outlier_idxs = best_outlier_idxs.nonzero()

    crp_feats = feats[u_crp, :].T
    crp_feats_in = crp_feats[:, best_inlier_idxs]
    crp_feats_out = crp_feats[:, best_outlier_idxs]

    crp_proj = best_P @ Xs[:, Xs_crp]
    crp_proj = tb.e2p(tb.p2e(crp_proj))  # normalize to the homogenous
    crp_proj_in = crp_proj[:, best_inlier_idxs]
    crp_proj_out = crp_proj[:, best_outlier_idxs]

    ax.scatter(crp_feats_out[0, :], crp_feats_out[1, :], marker='o', color='gray', s=20)
    ax.scatter(crp_proj_out[0, :], crp_proj_out[1, :], marker='x', color='gray', s=20)

    ax.scatter(crp_feats_in[0, :], crp_feats_in[1, :], marker='o', color='red', s=20)
    ax.scatter(crp_proj_in[0, :], crp_proj_in[1, :], marker='x', color='blue', s=20)

    plt.show()


def plot_error_hist(err_array, treshold, n_inliers):
    max_val = 100  # 3000
    n_bins = 100
    hist_range = (0, max_val)
    err_array.clip(min=None, max=max_val)
    plt.hist(err_array, bins=n_bins, range=hist_range)
    plt.axvline(x=treshold, color='red', linestyle='--', label='treshold')

    plt.title('Repr. error histogram: ' + str(n_inliers) + ' inliers')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.show()
    

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
            u_orig = tb.e2p(feats[u_crp].T)
            u_new  = tb.e2p(tb.p2e(P2 @ X))

            # precompute the transformations here for faster checks
            P2_X = P2 @ X

            support = 0
            inlier_idxs = []
            errors = np.zeros((n_crp))
            for i in range(n_crp):
                # Xi = X[:, i].reshape(4, 1)
                ui_orig = u_orig[:, i].reshape(3, 1)
                ui_new  = u_new[:, i].reshape(3, 1)

                # note: Xi is already in front of the P1
                # assert (K_inv @ np.eye(3, 4) @ Xi)[2] > 0, "Point from the global pointcloud should not have z negative"

                # note: when support is < 3, one or more of the selected 3D points project behind the camera P2

                # Check if computing with K_inv makes a difference --> no difference experienced so far
                # assert ((K_inv @ P2 @ Xi)[2] >  0 and (P2 @ Xi)[2] >  0) or \
                #        ((K_inv @ P2 @ Xi)[2] <= 0 and (P2 @ Xi)[2] <= 0), "Check: the conditions give different results"

                # compute the sampson error only for points in front of the camera
                if P2_X[2, i] > 0:

                    e_reprj = math.sqrt(
                        (ui_new[0]/ui_new[2] - ui_orig[0]/ui_orig[2])**2 +
                        (ui_new[1]/ui_new[2] - ui_orig[1]/ui_orig[2])**2)

                    errors[i] = e_reprj

                    if e_reprj**2 <= theta**2:
                        support += float(1 - e_reprj**2/theta**2)

                        # note: c.join_camera works with indexes (i) of Xs_crp and not its elements to keep the inliers
                        inlier_idxs.append(i)

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

    cam1, cam2 = 0, 1
    # cam1, cam2 = 5, 6
    img1, img2 = get_img(cam1), get_img(cam2)
    feats1, feats2 = get_feats(cam1), get_feats(cam2)
    K = np.loadtxt('scene_1/K.txt', dtype='float')
    K_inv = np.linalg.inv(K)

    n_cams = 12
    P_arr = [None] * n_cams
    corr_verbose = 0  # 1
    c = init_c(n_cams, corr_verbose)
    corresps = np.array(c.get_m(cam1, cam2)).T

    # perform the E estimation of initial camera pair, OK
    E, R, t, inls = ep.ransac_E(feats1, feats2, corresps, K)
    F = K_inv.T @ E @ K_inv

    # ep.plot_inliers(img1, feats1, feats2, corresps, inls)
    # ep.plot_e_lines(img1, img2, feats1, feats2, corresps, inls, F)

    Ps, Cs, zs = get_geometry(K, R, t)
    P_arr[cam1] = Ps[0]
    P_arr[cam2] = Ps[1]

    # get initial 3d points and their corresponding colors
    Xs = get_3d_points(cam1, cam2, inls, Ps[0], Ps[1])
    colors = get_3d_colors(cam1, cam2, inls)
    X_ids = np.arange(Xs.shape[1])  # IDs of the reconstructed scene points

    c.start(cam1, cam2, inls, X_ids)
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
        new_cam = tent_cams[np.argmax(n_tent_crp)]
        print("best_cam is (from 0): ", new_cam)

        X_crp, u_crp, _ = c.get_Xu(new_cam)
        cam_crp_prev = np.array(c.get_m(new_cam, cam1)).T

        # get the transformation of the new camera from the global frame (cam1) by the p3p algorithm
        R, t, new_inls = get_new_cam(new_cam, Xs, X_crp, u_crp, K)
        # todo: refine the R, t by numeric minimisation of reprj. error

        P_arr[new_cam] = K @ np.hstack((R, t))

        # add the new camera to the cluster, OK
        c.join_camera(new_cam, new_inls)

        # get the img-img crp and 3D-img crp after the call of camera join for the asserts
        X_crp_after, u_crp_after, _ = c.get_Xu(new_cam)
        cam_crp_after = np.array(c.get_m(new_cam, cam1)).T

        assert np.array_equal(u_crp_after, u_crp[new_inls]), "New image points do not match the inliers!"
        assert np.array_equal(X_crp_after, X_crp[new_inls]), "New 3D points do not match the inliers!"
        # check also the img-img correspondences?

        # get ids of cameras that still have img-img corresps to the new_cam
        nb_cam_list = c.get_cneighbours(new_cam)
        for nb_cam in nb_cam_list:
            assert P_arr[nb_cam] is not None, "We should know the transformations of neighbors!"

            cam_crp = np.array(c.get_m(new_cam, nb_cam)).T
            print("Reconstructing from", cam_crp.shape[0], new_cam, "->", nb_cam, "img-img correspondences")

            P1 = P_arr[new_cam]
            P2 = P_arr[nb_cam]

            # triangulate 3D points from the known Ps of the camera pair
            new_Xs = get_new_3d_points(new_cam, nb_cam, cam_crp, P1, P2)
            new_colors = get_new_3d_colors(new_cam, nb_cam, cam_crp)

            # precompute the transformations here for faster checks
            P1_new_X = P1 @ new_Xs
            P2_new_X = P2 @ new_Xs

            # get inliers with pozitive z
            inls = []  # must be indices to corresp between new_cam and nb_cam
            for i in range(new_Xs.shape[1]):   # n of corresps, cam_crp.shape[0] could also be used
                if P1_new_X[2, i] > 0 and P2_new_X[2, i] > 0:
                    new_X = new_Xs[:, i].reshape(4, 1)
                    Xs = np.hstack((Xs, new_X))     # append new 3D point
                    colors.append(new_colors[i])    # append new color
                    inls.append(i)                  # indexes to the cam_crp, OK

            # Xs = np.hstack((Xs, new_Xs[:, inls]))

            X_ids = np.arange(len(inls)) + len(X_ids)  # IDs of the reconstructed scene points

            # store the new X-u corresps as selected for both cameras, delete the remaining u-u corresps, OK
            c.new_x(new_cam, nb_cam, inls, X_ids)

        # verify the tentative corresp emerged in the cluster --> the cluster should contain only verified X-u
        cam_cluster_list = c.get_selected_cameras()  # list of all cameras in the cluster
        for cl_cam in cam_cluster_list:
            [X_crp, u_crp, Xu_verified] = c.get_Xu(cl_cam)
            Xu_tentative = np.where(~Xu_verified)[0]
            print("Verifying", Xu_tentative.shape[0], new_cam, "--", cl_cam, "tentative Xu correspondences")

            # verify by reprojection error tentative X-u corresps
            feats = get_feats(cl_cam)
            P = P_arr[cl_cam]

            # precompute here for faster checks
            P_Xs = P @ Xs
            K_inv_P_Xs = K_inv @ P_Xs

            # loop through indexes of remaining unverified X-u correspondences
            crp_ok = []
            theta = 3
            for idx in Xu_tentative:
                # get the homogenous image point [u, v, 1]
                ui_orig = tb.e2p(feats[u_crp[idx]].reshape(2, 1))

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
            c.verify_x(cl_cam, crp_ok)

        c.finalize_camera()
        print("Camera", new_cam,"finalized")
        n_cluster_cams += 1

    # ==============================================================================
    # =                                Plotting                                    =
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
    sparsity = 50
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





