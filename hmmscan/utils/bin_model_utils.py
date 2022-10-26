import numpy as np


def get_stat_from_trans(p):
    # Get the eigenvectors and eigenvalues
    eig = np.linalg.eig(p.T)

    # Index of eigenvector corresponding to eigenvalue 1
    eig_vec_idx = np.where(np.isclose(eig[0], 1))[0][0]

    # Normalized eigenvector
    stat_dist = np.real(eig[1][:, eig_vec_idx]/sum(eig[1][:, eig_vec_idx]))

    return np.array(stat_dist)