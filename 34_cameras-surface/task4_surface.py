import numpy as np
import scipy.io
import pickle
import time
import sys

sys.path.append("..")
sys.path.append("rectify/python")
sys.path.append("corresp/python")
sys.path.append("geom_export/python")
import rectify
import corresp
import ge

# custom functions
import tools as tb
from plot import *
from geometry import *
from load import *

PLOT = False

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


def show_imgs_f(img1, img2, f1, f2, u1=None, u2=None):
    # pick all features if indexes u1, u2 not specified
    if u1 is None or u2 is None:
        u1 = np.arange(f1.shape[0])
        u2 = np.arange(f2.shape[0])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1, cmap='gray')
    axs[1].imshow(img2, cmap='gray')
    for i in u1:
        x, y = f1[i]
        axs[0].scatter(x, y, marker='.')
    for i in u2:
        x, y = f2[i]
        axs[1].scatter(x, y, marker='.')
    plt.tight_layout()
    plt.show()


# all horizontal pairs
cam_idxs = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11)
]

cameras = np.load('params/cameras.npy', allow_pickle=True)
with open('params/c.pickle', 'rb') as f:
    c = pickle.load(f)

if not os.path.isfile('stereo_out.mat'):
    print("Preparing cameras for rectified matching")

    # task variable for all data
    task = []

    # For every selected image pair with indices i_a and i_b
    for idx1, idx2 in cam_idxs:
        print("Rectifying cameras", idx1, "and", idx2)
        cam1, cam2 = cameras[[idx1, idx2]]
        f1, f2, = cam1.f, cam2.f
        crp = get_corresps(idx1, idx2)

        # load the image im_a, im_b, compute fundamental matrix F_ab
        F = tb.get_F_from_P(cam1.P, cam2.P)

        # load corresponing points u_a, u_b (keep only inliers w.r.t. F)
        X1, u1, _ = c.get_Xu(cam1.id)
        X2, u2, _ = c.get_Xu(cam2.id)

        X_intersec, X1_idxs, X2_idxs = np.intersect1d(X1, X2, assume_unique=False, return_indices=True)
        u1_new = u1[X1_idxs]
        u2_new = u2[X2_idxs]

        if PLOT:
            show_imgs_f(cam1.img, cam2.img, f1, f2, u1_new, u2_new)

        img_a, img_b = cam1.img, cam2.img
        [H_a, H_b, img_a_r, img_b_r] = rectify.rectify( F, img_a, img_b )

        # modify corresponding points by H_a, H_b
        f1_rect = tb.p2e(H_a @ tb.e2p(f1[u1_new].T))
        f2_rect = tb.p2e(H_b @ tb.e2p(f2[u2_new].T))

        if PLOT:
            show_imgs_f(img_a_r, img_b_r, f1_rect.T, f2_rect.T)

        # seeds are rows of coordinates [ x_a, x_b, y ] --> crp points have the same y coordinate when rectified
        seeds = np.vstack((f1_rect[0],
                           f2_rect[0],
                           (f1_rect[1] + f2_rect[1]) / 2)).T

        task_i = np.array([img_a_r, img_b_r, seeds], dtype=object)
        task += [task_i]

    # now all stereo tasks are prepared, save to a matlab file
    task = np.vstack(task)
    scipy.io.savemat('stereo_in.mat', {'task': task})
    print("params saved in 'stereo_in.mat'")

    # here run the gcs stereo in matlab, and then load the results
    # --> run stereo_matching_gcs.m in matlab

else:
    d = scipy.io.loadmat( 'stereo_out.mat' )
    Ds = d['Ds']
    Xs = np.zeros((3, 0))
    colors = np.zeros((3, 0))

    # a disparity map for i-th pair is in D[i,0]
    for i in range(Ds.shape[1]):
        Di = Ds[0, i]
        cam1, cam2 = cameras[np.array(cam_idxs[i])]
        start = time.time()
        print("Processing camera pair:", cam_idxs[i])

        if PLOT:
            plot_disp_field(Di)

        nan_mask = np.isnan(Di)
        y_real, x_real = np.where(~nan_mask)

        disparities = Di[y_real, x_real].flatten().astype(int)  # Di[y, x]

        u1_rect = np.vstack((x_real, y_real))                   # pixels (x, y)
        u2_rect = np.vstack((x_real + disparities, y_real))     # pixels (x + Di[y, x], y)

        F = tb.get_F_from_P(cam1.P, cam2.P)
        [H_a, H_b, img_a_r, img_b_r] = rectify.rectify(F, cam1.img, cam2.img)

        # make it smaller for debugging
        # u1_rect = u1_rect[:, np.arange(0, u1_rect.shape[1], 1000)]
        # u2_rect = u2_rect[:, np.arange(0, u2_rect.shape[1], 1000)]
        # show_imgs_f(img_a_r, img_b_r, u1_rect.T, u2_rect.T)

        print("  transforming 2D features ... ", end="")
        u1 = tb.norm(np.linalg.inv(H_a) @ tb.e2p(u1_rect))
        u2 = tb.norm(np.linalg.inv(H_b) @ tb.e2p(u2_rect))


        print("rectifying 3D points ... ", end="")
        new_Xs = tb.p2e(tb.Pu2X(cam1.P, cam2.P, u1, u2))

        xyz_mask = (-10 <= new_Xs) & (new_Xs <= 10)
        mask = xyz_mask[0] & xyz_mask[1] & xyz_mask[2]

        new_Xs = new_Xs[:, mask]
        u1 = u1[:, mask]
        u2 = u2[:, mask]

        print("extracting colors ...")
        u1_x = np.clip(np.round(u1[1]).astype(int), 0, cam1.img.shape[0]-1)
        u1_y = np.clip(np.round(u1[0]).astype(int), 0, cam1.img.shape[1]-1)
        u2_x = np.clip(np.round(u2[1]).astype(int), 0, cam2.img.shape[0]-1)
        u2_y = np.clip(np.round(u2[0]).astype(int), 0, cam2.img.shape[1]-1)

        new_colors_a = cam1.img[u1_x, u1_y] / 255.0
        new_colors_b = cam2.img[u2_x, u2_y] / 255.0
        new_colors = (new_colors_a + new_colors_b) / 2
        # new_colors = cam1.img[u1_x, u1_y] / 255.0

        # add to results
        Xs = np.hstack((Xs, new_Xs))
        colors = np.hstack((colors, new_colors.T))

        end = time.time()
        print("  total time:", end-start, "s")

    print(Xs.shape[1], "points")
    g = ge.GePly('params/pointcloud_dense.ply')
    g.points(Xs, colors) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    # g.points(Xs) # Xs contains euclidean points (3xn matrix), l RGB colors (3xn or 3x1, optional)
    g.close()
