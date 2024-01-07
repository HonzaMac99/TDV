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


def crp2inls(cam1, cam2, crp_remaining):
    crp_orig = get_corresps(cam1, cam2)  # cams corresp. f. indexes
    inls = []
    for i, crp in enumerate(crp_orig):
        # Todo: check if crp_remaining is ordered and then improve performance
        if crp in crp_remaining:
            inls.append(i)
    return inls


def get_3d_colors_c(cam1, cam2, corresp):
    inls = crp2inls(cam1, cam2, corresp)
    return get_3d_colors(cam1, cam2, inls)


def get_3d_points_c(cam1, cam2, corresp, P1, P2):
    inls = crp2inls(cam1, cam2, corresp)
    return get_3d_points(cam1, cam2, inls, P1, P2)


# get the projection matrices, camera centres and unit z axis point for a pair of cameras
def get_geometry(K, R, t):
    # camera projections working with uncalibrated points
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


def get_new_cam(cam_id, Xs, Xs_crp, u_crp, K):
    ''' get new camera parameters such as R, t and inlier idxs
        in: Xs      ... [4xn] homogenous array of all 3D points
        in: Xs_crp  ... array with idxs for Xs, obtained from c.get_Xu(cam_id)
        in: u_crp   ... array with idxs for features, obtained from c.get_Xu(cam_id)
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
    while k <= k_max:
        rand_idx = rng.choice(np.arange(n_crp), size=3, replace=False)

        Xs_triple = np.zeros((4, 3))
        Us_triple = np.zeros((3, 3))
        for i, idx in enumerate(rand_idx):
            Xs_triple[:, i] = Xs[Xs_crp[idx]]
            Us_triple[:, i] = feats2[u_crp[idx]]
        Us_triple = K_inv @ tb.e2p(Us_triple)  # the points should be rectified first by K_inv!

        Xs_local_arr = p3p.p3p_grunert(Xs_triple, Us_triple)  # compute local coords of the 3D points
        if not Xs_local_arr:
            continue
        for Xs_local in Xs_local_arr:
            R, t = p3p.XX2Rt_simple(Xs, Xs_local)  # get R, t from glob Xs to the local 3D points

            # use P with K because the points are unrectified!
            P1 = K @ np.eye(3, 4)
            P2 = K @ np.hstack((R, t))

            support = 0
            inlier_idxs = []
            for i in range(n_crp):
                # we have corresponding features in cam2:
                # we want to project the corresponding global 3D points to compare them
                # compute the distance error from it

                Xi = Xs[:, Xs_crp[i]].reshape(4, 1)

                ui_orig = feats2[u_crp[i]].reashape(2, 1)
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
                        inlier_idxs.append(Xs_crp[i])

            if support > best_support:
                best_support = support
                best_R = R
                best_t = t
                best_inlier_idxs = inlier_idxs
                print("[ k:", k, "/", k_max, "] [ support:", support, "/", n_crp, "]")

            k += 1
            epsilon = 1 - (support/n_crp)
            k_max = math.log(1 - probability) / math.log(1 - (1 - epsilon)**3)

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
    # K_inv = np.linalg.inv(K)
    # F = K_inv.T @ E @ K_inv

    n_cams = 12
    P_arr = [None for i in range(n_cams)]
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
    X_ids = np.arrange(Xs.shape[1])  # IDs of the reconstructed scene points

    c.start(0, 1, inls, X_ids)
    n_cluster_cams = 2

    # main loop
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

        R, t, new_inls = get_new_cam(new_cam, Xs, X, u, K)

        P_arr[new_cam] = K @ np.hstack((R, t))

        c.join_camera(new_cam, new_inls)
        nb_cam_list = c.get_cneighbours(new_cam)

        for nb_cam in nb_cam_list:
            # we must know the transformations
            if P_arr[nb_cam] is None:
                continue
            cam_crp = c.get_m(new_cam, nb_cam)

            # todo: add camera classes?
            P1 = P_arr[new_cam]
            P2 = P_arr[nb_cam]

            # todo: reconstruct 3d points
            # we already know the camera pair transformations, so just triangulate the 3D pts
            new_Xs = get_3d_points_c(new_cam, nb_cam, cam_crp, P1, P2)
            new_colors = get_3d_colors_c(new_cam, nb_cam, cam_crp)
            Xs = np.hstack(Xs, new_Xs)    # append new 3D points
            colors = colors + new_colors  # append new color array

            # todo: Check pozitive z for inliers and more?
            inls = np.arange(new_Xs.shape[1])
            X_ids = np.arange(new_Xs.shape[1]) + len(X_ids)  # IDs of the reconstructed scene points

            c.new_x(new_cam, nb_cam, inls, X_ids)

            # Todo: verify the tentative corresp
            ilist = c.get_selected_cameras()  # list of all cameras in the cluster
            for ic in ilist:
                [X, u, Xu_verified] = c.get_Xu(ic)
                # Xu_tentative = find(Xu_verified)

                # Todo: Verify (by reprojection error) scene-to-image correspondences in Xu_tentative. /
                #       A subset of good points is obtained.

                corr_ok = []
                c.verify_x(ic, corr_ok)

            c.finalize_camera()

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

    # Todo: create global Ps, Cs and zs

    # plot the camera centers and the baseline
    ax.plot(Cs[0, :], Cs[1, :], Cs[2, :], marker='o', c='red')
    for i in range(Cs.shape[1]):
        ax.scatter(Cs[0, i], Cs[1, i], Cs[2, i], marker='o', s=20, c='black')
        ax.text(Cs[0, i], Cs[1, i], Cs[2, i], str(i+1), fontsize=10, c='black')
        ax.plot([Cs[0, i], zs[0, i]],  # plot the camera z axis
                [Cs[1, i], zs[1, i]],
                [Cs[2, i], zs[2, i]], c='black')

    # plot the 3D points with colors
    for i in range(Xs.shape[1]):
        ax.scatter(Xs[0, i], Xs[1, i], Xs[2, i], marker='o', s=5, color=colors[i])

    plt.show()











    # g = ge.GePly('out.ply')
    # colors = np.array(colors).T
    # # g.points(Xs, colors) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.points(Xs) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.close()





