import numpy as np


def costfun(x, r, v_obs, v_err=1.):
    return (r.interpn(x) - v_obs) / v_err


def default_lnlike(x, r, obs, obs_err, obs_tag=None, **kwargs):
    if obs_tag is not None:
        lnpost = - 0.5 * np.sum(((r(x) - obs) / (obs_err + r.me) * obs_tag)**2.)
    else:
        lnpost = - 0.5 * np.sum(((r(x) - obs) / (obs_err + r.me)) ** 2.)

    if np.isfinite(lnpost):
        return lnpost
    else:
        return -np.inf


def best_match(mod, mod_err, obs, obs_err, mask=None):
    """ search for the best match (min chi2) template in regli database

    Parameters
    ----------
    mod:
        templates
    mod_err:
        errors of templates
    obs:
        observations
    obs_err:
        errors of observations
    mask:
        1 for bad, 0 for good.

    Return
    ------
    index of the best match template
    """
    res2 = ((mod - obs) / (obs_err + mod_err)) ** 2.
    ind_invalid = np.all(np.logical_not(np.isfinite(res2)), axis=1)
    lnpost = - 0.5 * np.nansum(res2, axis=1)

    if mask is None:
        lnpost = np.ma.MaskedArray(lnpost, ind_invalid)
        return np.nanargmax(lnpost)
    else:
        lnpost = np.ma.MaskedArray(lnpost, mask | ind_invalid)
        return np.nanargmax(lnpost)
