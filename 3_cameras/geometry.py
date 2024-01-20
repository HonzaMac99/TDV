import numpy as np
import tools as tb
from load import *


# get colors from the images of the camera pair into the 3D plot:
def get_3d_colors(cam1, cam2, inls):
    f1, f2 = cam1.f, cam2.f               # feature points in images
    img1, img2 = cam1.img, cam2.img       # camera images
    crp = get_corresps(cam1.id, cam2.id)  # corresponding feature indexes

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


def get_3d_points(cam1, cam2, inls, K, correct=True):
    P1, P2 = cam1.P, cam2.P  # projection matrices
    f1, f2 = cam1.f, cam2.f  # features
    crp = get_corresps(cam1.id, cam2.id)  # cams corresp. f. indexes
    K_inv = np.linalg.inv(K)

    # pick features u1, u2 at indexes defined by inlier corresps and calibrate them
    # do not use K_inv on the points here!!
    u1 = tb.e2p(f1[crp[inls, 0]].T)
    u2 = tb.e2p(f2[crp[inls, 1]].T)

    # correct the correspondences using the sampson error correction
    if correct:
        u1, u2 = tb.correct_crp_sampson(P1, P2, u1, u2)

    Xs = tb.Pu2X(P1, P2, u1, u2)

    # check for points with negative z in global coords

    return Xs


def get_new_3d_colors(cam1, cam2, crp):
    f1, f2 = cam1.f, cam2.f           # feature points in images
    img1, img2 = cam1.img, cam2.img   # camera images

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


def get_new_3d_points(cam1, cam2, crp, K, correct=True):
    P1, P2 = cam1.P, cam2.P  # projection matrices
    f1, f2 = cam1.f, cam2.f  # features
    K_inv = np.linalg.inv(K)

    # pick features u1, u2 at indexes defined by inlier corresps and calibrate them
    u1 = tb.e2p(f1[crp[:, 0]].T)
    u2 = tb.e2p(f2[crp[:, 1]].T)

    # Important: points have to be K_inv first!!
    u1 = tb.norm(K_inv @ u1)
    u2 = tb.norm(K_inv @ u2)

    # correct the correspondences using the sampson error correction
    if correct:
        u1, u2 = tb.correct_crp_sampson(P1, P2, u1, u2)

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
