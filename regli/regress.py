import numpy as np


def costfun(x, r, v_obs, v_err):
    return (r.interpn(x) - v_obs) / v_err


def default_lnpost(x, r, obs, obs_err):
    lnpost = - 0.5 * np.sum(((r(x) - obs) / (obs_err + r.me))**2.)
    if np.isfinite(lnpost):
        return lnpost
    else:
        return -np.inf

