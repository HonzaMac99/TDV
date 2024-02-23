#!/usr/bin/env python3
import numpy as np
from scipy import optimize as opt
from scipy.spatial.transform import Rotation as Rot
import tools as tb
from load import *


def sampson_err_f(Rt_params, u1, u2, R, t, K_inv):
    R_params, t_params = Rt_params[:3], Rt_params[3:]

    R_new = R @ Rot.from_rotvec(R_params).as_matrix()  # from 3-vec to matrix
    t_new = t + t_params.reshape(3, 1)

    F = K_inv.T @ tb.sqc(-t_new) @ R_new @ K_inv

    return np.sum(tb.err_F_sampson(F, u1, u2))


def optimize_init_Rt(cam1, cam2, E, R, t, inls, K):
    f1, f2 = cam1.f, cam2.f               # feature points
    crp = get_corresps(cam1.id, cam2.id)  # cams corresp. f. indexes

    u1 = tb.e2p(f1[crp[inls, 0]].T)       # homogenous inliers in img1
    u2 = tb.e2p(f2[crp[inls, 1]].T)       # homogenous inliers in img2

    K_inv = np.linalg.inv(K)

    # note: it seems that when we are reconstructing E, sometimes E = [-t]x @ R
    #       gives us the negative E!!, should we check it before computaton?
    # F = K_inv.T @ E @ K_inv
    # F_test = K_inv.T @ tb.sqc(-t) @ R @ K_inv  # <-- for comparison

    Rt_params = np.zeros(6)

    opt_Rt_params = opt.fmin(sampson_err_f, Rt_params, (u1, u2, R, t, K_inv))
    best_Rp, best_tp = opt_Rt_params[:3], opt_Rt_params[3:]

    R_new = R @ Rot.from_rotvec(best_Rp).as_matrix()  # from 3-vec to matrix
    t_new = t + best_tp.reshape(3, 1)

    return R_new, t_new


def reprojection_err_f(Rt_params, X, u_orig, R, t, K):
    R_params = Rt_params[:3]
    R_new = R @ Rot.from_rotvec(R_params).as_matrix()  # from 3-vec to matrix
    t_new = t + Rt_params[3:].reshape(3, 1)

    # use P with K because the points are unrectified!
    P2 = K @ np.hstack((R_new, t_new))
    P2_X = P2 @ X  # precompute the projection
    u_new  = tb.norm(P2_X)

    e_reprj = np.sqrt((u_new[0, :] - u_orig[0, :])**2 + (u_new[1, :] - u_orig[1, :])**2)
    return np.sum(e_reprj)


def optimize_new_Rt(new_cam, Xs, Xs_crp, u_crp, K, R, t, new_inls):
    feats = new_cam.f
    Xs_crp = Xs_crp[new_inls]
    u_crp = u_crp[new_inls]

    X = Xs[:, Xs_crp]
    u_orig = tb.e2p(feats[u_crp].T)

    Rt_params = np.zeros(6)

    opt_Rt_params = opt.fmin(reprojection_err_f, Rt_params, (X, u_orig, R, t, K))
    best_Rp, best_tp = opt_Rt_params[:3], opt_Rt_params[3:]

    R_new = R @ Rot.from_rotvec(best_Rp).as_matrix()  # from 3-vec to matrix
    t_new = t + best_tp.reshape(3, 1)

    return R_new, t_new


# some testing
if __name__ == "__main__":

    def get_error(x):
        return np.abs(x - 1)


    # this function must return scalar value
    def get_y(x):
        return sum(get_error(x))


    x_init = [0.5, -0.4, 0.5, 0.8]
    result = opt.fmin(get_y, x_init)
    print(get_y(result))
    print(result)


    R = np.eye(3, 3)
    t = np.zeros((3, 1))
    rvec = Rot.from_matrix(R).as_rotvec()  # 3 parametry do optimalizace

    # rvec = [2*np.pi, 0, 0]
    R = Rot.from_rotvec(rvec).as_matrix()  # zpět na rotační matici po optimalizaci
    print(R)

