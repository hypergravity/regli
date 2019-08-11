import numpy as np


def costfun(x, r, v_obs, v_err=1.):
    return (r.interpn(x) - v_obs) / v_err


def default_lnlike(x, r, obs, obs_err, obs_weight=None, **kwargs):
    res2 = ((r(x) - obs) / (obs_err + r.me)) ** 2.
    if obs_weight is not None:
        res2 *= obs_weight
    if np.all(np.logical_not(np.isfinite(res2))):
        # in case every element is nan
        return  -np.inf
    else:
        lnpost = - 0.5 * np.nansum(res2, axis=1)


def best_match(mod, mod_err, obs, obs_err, obs_weight=None, mask=None):
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
    obs_weight:
        weight of observations, np.nan for bad values
    mask:
        1 for bad, 0 for good.

    Return
    ------
    index of the best match template
    """
    res2 = ((mod - obs) / (obs_err + mod_err)) ** 2.
    if obs_weight is not None:
        res2 *= obs_weight
    ind_invalid = np.all(np.logical_not(np.isfinite(res2)), axis=1)
    lnpost = - 0.5 * np.nansum(res2, axis=1)

    if mask is None:
        lnpost = np.ma.MaskedArray(lnpost, ind_invalid)
        return np.nanargmax(lnpost)
    else:
        lnpost = np.ma.MaskedArray(lnpost, mask | ind_invalid)
        return np.nanargmax(lnpost)
