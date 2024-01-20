import numpy as np
import os
import matplotlib.image as mpimg


def get_corresps(i, j):
    i += 1
    j += 1
    swapp = i > j
    if swapp:
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
    if swapp:
        corresps = corresps[:, [1, 0]]
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


def refresh(cam1, cam2):
    refr = True
    new_cams = np.array([cam1, cam2])
    if os.path.isfile('params/init_cams.npy'):
        old_cams = np.load('params/init_cams.npy')
        if np.array_equal(old_cams, new_cams):
            refr = False
    if refr:
        np.save('params/init_cams.npy', new_cams)
    elif not (os.path.isfile('params/E.npy') and os.path.isfile('params/R.npy') and
              os.path.isfile('params/t.npy') and os.path.isfile("params/inls.npy")):
        refr = True
    return refr


def load_E_params():
    E = np.load('params/E.npy')
    R = np.load('params/R.npy')
    t = np.load('params/t.npy')
    inls = np.load('params/inls.npy')
    return E, R, t, inls

def save_E_params(E, R, t, inls):
    np.save('params/E.npy', E)
    np.save('params/R.npy', R)
    np.save('params/t.npy', t)
    np.save('params/inls.npy', inls)
