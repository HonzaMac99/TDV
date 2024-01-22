import numpy as np
import matplotlib.pyplot as plt
import tools as tb


# plot the comparison between the original feature points and the
# projections of corresponding 3D points (Xs) by the best estimated R and t
def plot_transformation(best_R, best_t, best_inlier_idxs, Xs, Xs_crp, feats, u_crp, temp=True):
    fig, ax = plt.subplots(1, 1)
    ax.invert_yaxis()
    K = np.loadtxt('scene_1/K.txt', dtype='float')

    best_P = K @ np.hstack((best_R, best_t))
    n_crp = len(Xs_crp)

    best_outlier_idxs = np.ones(n_crp)
    best_outlier_idxs[best_inlier_idxs] = False
    best_outlier_idxs = best_outlier_idxs.nonzero()

    crp_feats = feats[u_crp, :].T
    crp_feats_in = crp_feats[:, best_inlier_idxs]
    crp_feats_out = crp_feats[:, best_outlier_idxs]

    crp_proj = best_P @ Xs[:, Xs_crp]
    crp_proj = tb.norm(crp_proj)  # normalize to the homogenous
    crp_proj_in = crp_proj[:, best_inlier_idxs]
    crp_proj_out = crp_proj[:, best_outlier_idxs]

    ax.scatter(crp_feats_out[0, :], crp_feats_out[1, :], marker='o', color='gray', s=20)
    ax.scatter(crp_proj_out[0, :], crp_proj_out[1, :], marker='x', color='gray', s=20)

    ax.scatter(crp_feats_in[0, :], crp_feats_in[1, :], marker='o', color='red', s=20)
    ax.scatter(crp_proj_in[0, :], crp_proj_in[1, :], marker='x', color='blue', s=20)

    if not temp:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def plot_error_hist(err_array, treshold, n_inliers, temp=True):
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

    if not temp:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(1)
        plt.close()

