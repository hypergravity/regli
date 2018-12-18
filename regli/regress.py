import numpy as np


def costfun(x, r, v_obs, v_err=1.):
    return (r.interpn(x) - v_obs) / v_err


def default_lnlike(x, r, obs, obs_err, obs_tag):
    lnpost = - 0.5 * np.sum(((r(x) - obs) / (obs_err + r.me))**2.*obs_tag)
    if np.isfinite(lnpost):
        return lnpost
    else:
        return -np.inf


def best_match(mod, mod_err, obs, obs_err):
    lnpost = - 0.5 * np.sum(((mod - obs) / (obs_err + mod_err)) ** 2., axis=1)
    return np.nanargmax(lnpost)
