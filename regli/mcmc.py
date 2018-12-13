import numpy as np


def sample_covariance(flatchain):
    """ evaluate covariance matrix by definition for MCMC sample

    Parameters
    ----------
    flatchain: n_chain x n_dim
        sampler.flatchain

    Return
    ------
    cov_: n_dim x n_dim
        covariance matrix

    """

    # mean values
    m = np.mean(flatchain, axis=0)

    # ndim
    ndim = flatchain.shape[1]

    # initiate cov_
    cov_ = np.zeros((ndim, ndim), np.float)

    ind_tril = np.tril_indices(ndim)
    for irow, icol in zip(*ind_tril):
        # evaluate cov_ elements by definition
        cov_[irow, icol] = np.mean((flatchain[:, irow] - m[irow]) * (flatchain[:, icol] - m[icol]))
        cov_[icol, irow] = cov_[irow, icol]

    return cov_


def cov_to_std(cov_):
    """ evaluate standard deviation based on cov_ matrix
    """
    return np.sqrt(np.diag(cov_))


def cov_to_rho(cov_):
    """ evaluate pearson correlation coefficient
    """
    s = cov_to_std(cov_)
    return cov_ / (s.reshape(-1, 1) * s.reshape(1, -1))