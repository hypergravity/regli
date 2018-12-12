

def costfun(x, r, v_obs, v_err):
    return (r.interpn(x) - v_obs) / v_err


def default_lnpost(x, r, obs, obs_err):
    return 0.5 * ((r(x) - obs) / (obs_err + r.me))**2.

