# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:24:15 2018

@author: cham
"""

import bisect
import collections
from itertools import product

import numpy as np
from emcee import EnsembleSampler
from laspec.qconv import conv_spec_Gaussian, conv_spec_Rotation
from laspec.normalization import normalize_spectrum_spline
from laspec.ccf import RVM
from scipy.optimize import least_squares

from .regress import costfun, default_lnlike, best_match

# speed of light
SOL = 299792.458


def rand_pos(p0, nwalkers, eps=.1):
    p0 = np.array(p0)
    # do a 0.1 mag rand
    return [p0 + (np.random.rand(*p0.shape)-0.5)*2*eps for i in range(nwalkers)]


def grid_to_meshflat(*grids):
    """ convert a grid to meshflat arrays """
    # get dimensions
    Ns = [len(_) for _ in grids]
    # make indices of grids
    grids_ind = [np.arange(_, dtype=int) for _ in Ns]
    # make mesh_grids
    meshs = np.meshgrid(*grids)
    # make mesh_ind
    meshs_ind = np.meshgrid(*grids_ind)
    # make flat_mesh_grids
    flats = np.array([_.flatten() for _ in meshs]).T
    # make flat_mesh_ind
    flats_ind = np.array([_.flatten() for _ in meshs_ind]).T

    # make a dict of flat_mesh_ind:i
    ind_dict = dict()
    for i in range(len(flats_ind)):
        ind_dict[tuple(flats_ind[i])] = i

    return grids, grids_ind, meshs, meshs_ind, flats, flats_ind, ind_dict


def bisect_interval(edges=[1, 2, 3], x=.1):
    """ return the nearest edges using bisect """
    if edges[0] < x <= edges[-1]:
        _ = bisect.bisect_left(edges, x)
        return _ - 1, _
    elif edges[0] == x:
        return 0, 1
    else:
        # null value, returned when x is not in bounds
        return -9, -9


class Regli:
    """ Regular Grid Linear Interpolator """
    wave = None

    def __init__(self, *grids):

        self.ndim = len(grids)
        _ = grid_to_meshflat(*grids)
        self.grids = _[0]
        self.grids_ind = _[1]
        self.meshs = _[2]
        self.meshs_ind = _[3]
        self.flats = _[4]
        self.flats_ind = _[5]
        self.ind_dict = _[6]

        self.grid_shape = tuple([len(g) for g in grids])

        # set values
        self.set_values(np.array([]), test=True)
        # self.value_shape = 1
        # self.set_values(values)

        # define a subset for best match
        self.best_match_mask = None

        # mean error
        self.me = 0.

        # pre-proc parameters
        self.set_pre_proc_ind()

    def __repr__(self):
        s = "\n".join([
            "==== regli.Regli instance ====",
            "".join(["grid shape:  ", self.grid_shape.__repr__()]),
            "".join(["value shape: ", self.values.shape.__repr__()]),
            "==============================",
        ])
        return s

    def set_pre_proc_ind(self, pre_proc_ind=None):
        if pre_proc_ind is None:
            self.pre_proc_ind = pre_proc_ind
        else:
            self.pre_proc_ind = list(pre_proc_ind)

    def pre_proc(self, pos):
        if self.pre_proc_ind is None:
            return np.array(pos)
        else:
            return np.array(pos)[list(self.pre_proc_ind)]

    @staticmethod
    def init_from_flats(input_flats, verbose=False):
        """ try to parse flats and return a Regli instance
        This is a slow process and takes 86 min to parse a 400,000 point grid.

        Parameters
        ----------
        input_flats:
            flats
        verbose:
            if True, print verbose info

        Return
        ------

        """

        # determine n_dim
        ndim = input_flats.shape[1]

        # determine grid
        grids = [np.unique(input_flats[:, i]) for i in range(ndim)]

        # initiat Regli
        r = Regli(*grids)

        # determine eps
        eps_auto = np.min([np.min(np.diff(grid)) for grid in grids]) * 0.3

        # determine n_flats [number of grid points]
        nrow = np.prod(r.grid_shape)
        # ind_values = np.zeros((nrow,), np.int)

        c_missing = 0
        for i in range(len(r.flats)):
            # for each flat
            flat_ = r.flats[i]
            flat_ind_ = r.flats_ind[i]

            # evaluate norm, see if data exists
            norm_ = np.linalg.norm(input_flats - flat_, np.inf, axis=1)
            if np.min(norm_) < eps_auto:
                # data exists
                r.ind_dict[tuple(flat_ind_)] = np.argmin(norm_)
            else:
                # data missing
                r.ind_dict[tuple(flat_ind_)] = None
                c_missing += 1
                if verbose:
                    print("@Regli: grid value missing --> ", flat_)

        print("@Regli: {} values missing!".format(c_missing))

        return r

    @property
    def rgi_shape(self):
        return (*self.grid_shape, self.value_shape)

    def set_values(self, values, test=False):
        values = np.array(values)
        assert values.ndim in (1, 2)

        if values.ndim == 2:
            self.values = np.array(values)
            self.value_shape = self.values.shape[1]
            self.set_wave(np.arange(self.values.shape[1]))  # only available when 2D
        elif values.ndim == 1:
            if not test:
                assert len(values) == len(self.flats)
            self.values = np.array(values.reshape(-1, 1))
            self.value_shape = 1
        else:
            raise ValueError("Values shape not correct!")

    def interpns(self, poss):
        return np.array([self.interpn(pos) for pos in poss])

    def interp2(self, pos, null_value=np.nan):

        e1 = bisect_interval(self.grids[0], pos[0])
        e2 = bisect_interval(self.grids[1], pos[1])

        if e1[0] < -1 or e2[0] < -1:
            # out of bounds
            return np.ones((self.values.shape[1],)) * null_value
        else:
            # calculate nodes
            p1_0 = self.grids[0][e1[0]]
            p1_1 = self.grids[0][e1[1]]
            p2_0 = self.grids[1][e2[0]]
            p2_1 = self.grids[1][e2[1]]

            v_tot = (p1_1 - p1_0) * (p2_1 - p2_0)
            v_00 = (p1_1 - pos[0]) * (p2_1 - pos[1])
            v_01 = (p1_1 - pos[0]) * (pos[1] - p2_0)
            v_10 = (pos[0] - p1_0) * (p2_1 - pos[1])
            v_11 = (pos[0] - p1_0) * (pos[1] - p2_0)
            # v_000+v_001+v_010+v_011+v_100+v_101+v_110+v_111

            i_00 = self.ind_dict[e1[0], e2[0]]
            i_01 = self.ind_dict[e1[0], e2[1]]
            i_10 = self.ind_dict[e1[1], e2[0]]
            i_11 = self.ind_dict[e1[1], e2[1]]

            if None in (i_00, i_01, i_10, i_11):
                return np.ones((self.values.shape[1],)) * null_value

            w = np.array([v_00, v_01, v_10, v_11]).reshape(-1, 1) / v_tot
            value_interp = np.sum(self.values[np.array([i_00, i_01, i_10, i_11])] * w, axis=0)

            return value_interp

    def interp3(self, pos, null_value=np.nan):

        e1 = bisect_interval(self.grids[0], pos[0])
        e2 = bisect_interval(self.grids[1], pos[1])
        e3 = bisect_interval(self.grids[2], pos[2])

        if e1[0] < -1 or e2[0] < -1 or e3[0] < -1:
            # out of bounds
            return np.ones((self.values.shape[1],)) * null_value
        else:
            # calculate nodes
            p1_0 = self.grids[0][e1[0]]
            p1_1 = self.grids[0][e1[1]]
            p2_0 = self.grids[1][e2[0]]
            p2_1 = self.grids[1][e2[1]]
            p3_0 = self.grids[2][e3[0]]
            p3_1 = self.grids[2][e3[1]]

            v_tot = (p1_1 - p1_0) * (p2_1 - p2_0) * (p3_1 - p3_0)
            v_000 = (p1_1 - pos[0]) * (p2_1 - pos[1]) * (p3_1 - pos[2])
            v_100 = (pos[0] - p1_0) * (p2_1 - pos[1]) * (p3_1 - pos[2])
            v_010 = (p1_1 - pos[0]) * (pos[1] - p2_0) * (p3_1 - pos[2])
            v_110 = (pos[0] - p1_0) * (pos[1] - p2_0) * (p3_1 - pos[2])
            v_001 = (p1_1 - pos[0]) * (p2_1 - pos[1]) * (pos[2] - p3_0)
            v_101 = (pos[0] - p1_0) * (p2_1 - pos[1]) * (pos[2] - p3_0)
            v_011 = (p1_1 - pos[0]) * (pos[1] - p2_0) * (pos[2] - p3_0)
            v_111 = (pos[0] - p1_0) * (pos[1] - p2_0) * (pos[2] - p3_0)
            # v_000+v_001+v_010+v_011+v_100+v_101+v_110+v_111

            i_000 = self.ind_dict[e1[0], e2[0], e3[0]]
            i_001 = self.ind_dict[e1[0], e2[0], e3[1]]
            i_010 = self.ind_dict[e1[0], e2[1], e3[0]]
            i_011 = self.ind_dict[e1[0], e2[1], e3[1]]
            i_100 = self.ind_dict[e1[1], e2[0], e3[0]]
            i_101 = self.ind_dict[e1[1], e2[0], e3[1]]
            i_110 = self.ind_dict[e1[1], e2[1], e3[0]]
            i_111 = self.ind_dict[e1[1], e2[1], e3[1]]

            if None in (i_000, i_001, i_010, i_011, i_100, i_101, i_110, i_111):
                return np.ones((self.values.shape[1],)) * null_value

            w = np.array([v_000, v_001, v_010, v_011, v_100, v_101, v_110, v_111]).reshape(-1, 1) / v_tot
            # specs = spec[np.array([i_000, i_001, i_010, i_011, i_100, i_101, i_110, i_111])]
            # figure();plot(spec_wm);plot(specs.T)
            value_interp = np.sum(self.values[np.array([i_000, i_001, i_010, i_011, i_100, i_101, i_110, i_111])] * w,
                                  axis=0)

            return value_interp

    def interpn(self, pos, null_value=np.nan):
        pos = np.array(pos).flatten()
        # ndim x 2 edge array
        edges_ind = np.array([bisect_interval(self.grids[_], pos[_]) for _ in range(self.ndim)])
        edges = np.array([(self.grids[i][edges_ind[i]]) for i in range(self.ndim)])

        if np.any(edges_ind[:, 0] < -1):
            # out of bounds
            return np.ones((self.values.shape[1],))*null_value

        # make codes
        codes = np.array([_ for _ in product((0, 1), repeat=self.ndim)])

        # weight in each dimension
        frac_dist = np.fliplr((edges - pos.reshape(-1, 1)) * np.array([-1, 1])) / np.diff(edges)

        n_neighb = codes.shape[0]
        n_dim = self.ndim
        n_pix = self.values.shape[1]
        v_neighb = np.zeros((n_neighb, n_pix), float)
        w_neighb = np.zeros((1, n_neighb), float)
        _ind_neighb = np.arange(n_dim)

        for i_neighb in range(n_neighb):
            ind_value = self.ind_dict[tuple(edges_ind[_ind_neighb, codes[i_neighb]])]
            if ind_value is None:
                # out of bounds
                return np.ones((self.values.shape[1],)) * null_value
            v_neighb[i_neighb] = self.values[ind_value]
            w_neighb[0, i_neighb] = np.prod(frac_dist[_ind_neighb, codes[i_neighb]])
            # verbose #
            # print("fracdist", frac_dist)
            # print("ind_neighb", _ind_neighb)
            # print("codes", codes[i_neighb])
            # print(frac_dist[_ind_neighb, codes[i_neighb]], np.prod(frac_dist[_ind_neighb, codes[i_neighb]]))
            # print("w_neighb:",w_neighb)
            # print(np.sum(w_neighb))
            # print("=====")

        v_interp = np.dot(w_neighb, v_neighb)[0]
        return v_interp

    def __call__(self, pos):
        # pre-proc of input pos
        pos = self.pre_proc(pos)

        # optimized for ndim = {2,3}
        if self.ndim == 3:
            return self.interp3(pos)
        elif self.ndim == 2:
            return self.interp2(pos)
        else:
            return self.interpn(pos)

    def regress(self, x0, obs, obs_err, *args, **kwargs):
        return least_squares(costfun, x0, self, obs, obs_err, *args, **kwargs)

    def run_mcmc(self, obs, obs_err=0, obs_weight=None, p0=None, n_burnin=(100, 100),
                 n_step=1000, lnlike=None, lnprior=None,
                 lnlike_kwargs={}, lnprior_kwargs={},
                 pos_eps=.1, full=True, shrink="max",
                 ndim=None, nwalkers=None):
        """ run MCMC for (obs, obs_err, obs_weight)

        Parameters
        ----------
        obs:
            observable
        obs_err:
            error of observation
        obs_weight:
            additional weight, array(size(obs), 1 for good, 0/np.nan for bad.
        p0:
            initial points
        n_burnin:
            int/sequence, do 1 or multiple times of burn in process
        n_step:
            running step
        lnlike:
            define your own log likelihood function:
            lnlike(x, r, obs, obs_err, obs_weight=None, **kwargs)
        lnprior:
            lnpost(x) is the log prior
        pos_eps:
            random magnitude for starting position
        full:
            if True, return sampler, pos, prob, state
            otherwise, return sampler
        shrink:
            default "max": shrink to the maximum likelihood position

        Return
        ------
        EnsembleSampler instance

        """
        if obs_weight is None:
            # default obs_weight
            obs_weight = np.isfinite(obs) * 1
        else:
            obs_weight = np.isfinite(obs) * obs_weight
        # cope with bad obs
        if np.nansum(obs_weight) == 0:
            return None

        if p0 is None:
            # do best match for starting position
            p0 = self.best_match(obs, obs_err)
            print("@Regli.best_match: ", p0)

        if lnlike is None:
            # default gaussian likelihood function
            lnlike = default_lnlike
            print("@Regli: using the default gaussian *lnlike* function...")

        if lnprior is None:
            # no prior is specified
            lnpost = lnlike
            print("@Regli: No prior is adopted ...")
        else:
            # use user-defined prior
            print("@Regli: using user-defined *lnprior* function...")

            def lnpost(*_args, **_kwargs):
                lnpost_value = float(lnprior(_args[0], **_kwargs["lnprior_kwargs"])) + \
                               lnlike(*_args, **_kwargs["lnlike_kwargs"])
                if np.isfinite(lnpost_value):
                    return lnpost_value
                else:
                    return -np.inf

        # set parameters for sampler
        if ndim is None:
            ndim = self.ndim
        if nwalkers is None:
            nwalkers = 2 * ndim

        # initiate sampler
        sampler = EnsembleSampler(nwalkers, ndim, log_prob_fn=lnpost,
                                  args=(self, obs, obs_err, obs_weight),
                                  kwargs=dict(lnprior_kwargs=lnprior_kwargs,
                                              lnlike_kwargs=lnlike_kwargs))

        # generate random starting positions
        pos0 = rand_pos(p0, nwalkers=nwalkers, eps=pos_eps)

        if isinstance(n_burnin, collections.Iterable):
            # [o] multiple burn-ins
            for n_burnin_ in n_burnin:
                # run mcmc
                print("@Regli: running burn-in [{}]...".format(n_burnin_))
                pos, prob, state = sampler.run_mcmc(pos0, n_burnin_)

                # shrink to a new position
                if shrink == "max":
                    p1 = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
                else:
                    p1 = np.median(pos, axis=0)
                pos_std = np.std(sampler.flatchain, axis=0)*0.5

                # reset sampler
                sampler.reset()

                # randomly generate new start position
                pos0 = rand_pos(p1, nwalkers=nwalkers, eps=pos_std)

        else:
            # [o] single burn-in
            # run mcmc
            print("@Regli: running burn-in [{}]...".format(n_burnin))
            pos, prob, state = sampler.run_mcmc(pos0, n_burnin)

            # shrink to a new position
            if shrink == "max":
                p1 = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
            else:
                p1 = np.median(pos, axis=0)
            pos_std = np.std(sampler.flatchain, axis=0) * 0.5

            # reset sampler
            sampler.reset()

            # randomly generate new start position
            pos0 = rand_pos(p1, nwalkers=nwalkers, eps=pos_std)

        # run mcmc
        print("@Regli: running chains [{}]...".format(n_step))
        pos, prob, state = sampler.run_mcmc(pos0, n_step)

        if full:
            # return full result
            return sampler, pos, prob, state
        else:
            # return sampler only
            return sampler

    def set_best_match_mask(self, mask):
        """make a subset for best match search

        mask:
            1 for bad, 0 for good
        """
        self.best_match_mask = mask

    def best_match(self, obs, obs_err, obs_weight=None):
        """ returns the best matched position
        """
        # constrained best match!!!
        ind_maxlnpost = best_match(self.values, self.me, obs, obs_err,
                                   obs_weight=obs_weight,
                                   mask=self.best_match_mask)
        return self.flats[ind_maxlnpost]

    # set wavelength
    def set_wave(self, wave):
        self.wave = wave

    def generate_rvm(self, para_trial, n_collect=1000):
        """
        Parameters
        ----------
        para_trial:
            trial list
        n_collect:
            collect n_collect spectra if n_collect > 0

        Return
        ------
        RVM instance
        """
        para_list = []
        flux_list = []

        n_trial = len(para_trial)
        n_success = 0
        for i_trial in range(n_trial):
            this_param = para_trial[i_trial]
            this_flux = self.interpn(this_param)
            if np.isfinite(this_flux[0]):
                flux_list.append(normalize_spectrum_spline(self.wave, this_flux, niter=2)[0])
                para_list.append(this_param)
                n_success += 1
                print("i_trial = {}, progress: [{}/{}]".format(i_trial, n_success, n_collect))
            if 0 < n_collect <= n_success:
                break
        return RVM(np.asarray(para_list), self.wave, np.asarray(flux_list))

    # predict spectrum
    def predict_spectrum(self, pstar, R_hi=None, R_lo=None, vsini=0, epsilon=0.6, rv=0, wave_new=None, snr=None):
        """
        Parameters
        ----------
        pstar:
            stellar parameters
        R_hi:
            original resolution
        R_lo:
            target resolution
        vsini:
            projected rotational velocity
        epsilon:
            limb-darkening coefficient
        rv:
            radial velocity
        wave_new:
            target wavelength grid
        snr:
            signal-to-noise ratio

        Return
        ------
        flux:
            spectral flux
        """
        # interpolate flux
        flux = self.interpn(pstar)
        # wavelength
        if wave_new is None:
            wave_new = self.wave

        # degrade resolution
        if R_hi is not None and R_lo is not None:
            flux = conv_spec_Gaussian(self.wave, flux, R_hi=R_hi, R_lo=R_lo, wave_new=self.wave)[1]
        # convolve vsini
        if vsini > 0:
            flux = conv_spec_Rotation(self.wave, flux, vsini=vsini, epsilon=epsilon, wave_new=self.wave)[1]
        # radial velocity
        flux = np.interp(wave_new, self.wave * (1 + rv / SOL), flux)
        # add Gaussian noise
        if snr is not None:
            flux *= np.random.normal(1, 1 / snr, flux.shape)
        return flux


def test():

    x1 = np.linspace(-1, 1, 30)
    x2 = np.linspace(-1, 1, 30)
    x3 = np.linspace(-1, 1, 30)
    regli = Regli(x1, x2, x3)
    f = lambda _x1, _x2, _x3: _x1 + _x2 + _x3

    flats = regli.flats
    values = np.array([f(*_) for _ in flats]).reshape(-1, 1)
    regli.set_values(values)

    from scipy.interpolate import RegularGridInterpolator
    rgi = RegularGridInterpolator(regli.grids, values.reshape(30, 30, 30, -1))

    from time import time
    test_pos = (0.1, -0.111, 0.3)

    t0 = time()
    for i in range(10000):
        regli.interp3(test_pos)
    print("regli.interp3 x 10000: {} sec".format(time() - t0))

    t0 = time()
    for i in range(10000):
        regli.interpn(test_pos)
    print("regli.interpn x 10000: {} sec".format(time() - t0))

    regli.interpn((0.1, -0.111, 0.3))

    t0 = time()
    for i in range(10000):
        rgi(test_pos)
    print("rgi x 10000: {} sec".format(time() - t0))


if __name__ == "__main__":
    test()