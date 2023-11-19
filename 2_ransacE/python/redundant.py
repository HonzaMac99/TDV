# These are concepts of functions but replaced by the toolbox


def check_e_points(R, t, pts1, pts2):
    is_valid = True
    P1 = np.eye(3, 4)
    P2 = np.hstack((R, t))

    # triangulate the 5 points from estimation
    for i in range(len(pts1)):
        p1 = pts1[i]
        p2 = pts2[i]
        Q = np.vstack(((p1[0]*P1[2, :] - P1[0, :]),
                       (p2[0]*P1[2, :] - P1[1, :]),
                       (p1[1]*P2[2, :] - P2[0, :]),
                       (p2[1]*P2[2, :] - P2[1, :])))
        U, S, Vt = np.linalg.svd(Q)
        X = U[:, 3]  # triangulated homogenous point
        X = X[:3] / X[3]  # get real 3D position
        if X[2] <= 0:
            return False
    return is_valid


def decompose_e(E, pts1, pts2):
    U, _, Vt = np.linalg.svd(E)

    # Determinant of U and V should be positive
    # to form proper rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Rotation matrices
    Rs = []
    for alpha in [1, -1]:
        W = np.array([[    0, -alpha, 0],
                      [alpha,      0, 0],
                      [    0,      0, 1]])
        R = np.dot(U, np.dot(W, Vt))
        Rs.append(R)

    # Translation vectors
    beta = 1
    t1 = beta*U[:, 2]
    t2 = -beta*U[:, 2]
    ts = [t1.reshape((3, 1)), t2.reshape((3, 1))]

    # get the correct decomposition
    confs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idxs in confs:
        R = Rs[idxs[0]]
        t = ts[idxs[1]]
        if check_e_points(R, t, pts1, pts2):
            return [R, t]

    return []