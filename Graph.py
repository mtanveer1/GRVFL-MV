import numpy as np
from scipy.linalg import pinv


def matrices_LFDA(data, lbls):
    N = data.shape[1]
    Dmat = np.sum(data.T ** 2, axis=1)[:, np.newaxis] + np.sum(data.T ** 2, axis=1) - 2 * (data.T @ data)
    sigma2 = np.mean(np.mean(Dmat))
    Amat = np.exp(-Dmat / (2 * sigma2))

    Wlfda_b = np.zeros_like(Amat)
    Wlfda_w = np.zeros_like(Amat)

    for ii in range(len(lbls)):
        curr_ind = np.where(lbls == lbls[ii])[0]
        other_ind = np.where(lbls != lbls[ii])[0]
        Wlfda_w[ii, curr_ind] = Amat[ii, curr_ind] / len(curr_ind)
        Wlfda_b[ii, curr_ind] = Amat[ii, curr_ind] * (1 / N - 1 / len(curr_ind))
        Wlfda_b[ii, other_ind] = 1 / N

    Wlfda_w = (Wlfda_w + Wlfda_w.T) / 2
    Wlfda_b = (Wlfda_b + Wlfda_b.T) / 2

    Dlfda_w = np.sum(Wlfda_w, axis=1)
    Llfda_w = np.diag(Dlfda_w) - Wlfda_w

    Dlfda_b = np.sum(Wlfda_b, axis=1)
    Llfda_b = np.diag(Dlfda_b) - Wlfda_b

    Slfda_w = data @ Llfda_w @ data.T
    Slfda_b = data @ Llfda_b @ data.T

    return Slfda_w, Slfda_b

def calc_ge_data2(train_data, train_lbls):
    S_w, S_b = matrices_LFDA(train_data, train_lbls)
    

    # calculate modified samples
    Rval = 1e-3
    S = pinv(S_b + Rval * np.eye(S_b.shape[0])) @ S_w

    return S
