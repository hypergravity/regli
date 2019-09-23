# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:11:15 2019

@author: cham
"""

import numpy as np
from .regli import Regli
from slam.normalization import normalize_spectrum, normalize_spectra_block
from scipy.interpolate import interp1d
from astropy import constants as const


SOL = 299792.458  # km/s


class SpIn(object):

    def __init__(self, ):

        self.ndim = 0
        self.nband = 0
        self.wave = list()
        self.reglis = list()

    def add_band(self, wave, flux, params):

        # the new band should have the same dimension
        ndim = params.shape[1]
        if self.nband > 0:
            assert ndim == self.ndim
        else:
            self.ndim = ndim

        # append the new band
        self.wave.append(wave)
        r = Regli.init_from_flats(params)
        r.set_values(flux)
        self.reglis.append(r)
        self.nband += 1

    def interp(self, p, band="all"):
        p = np.array(p)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        # n_p = p.shape[0]

        wave = []
        flux_interp = []

        # bands
        if band == "all":
            band = [_ for _ in range(self.nband)]
        elif isinstance(band, int):
            band = [band]
        else:
            band = list(band)

        # interpolate each band
        for iband in band:
            flux_interp.append(np.array([self.reglis[iband](p_) for p_ in p]))
            wave.append(self.wave[iband])

        return wave, flux_interp

    @staticmethod
    def generate_mock_p_uniform(n=10, ranges=[(0, 1)], seed=0):
        ranges = np.array(ranges)
        ndim_ = len(ranges)
        pmock = np.random.rand(n, ndim_) * np.diff(ranges, axis=1).reshape(1, -1) \
            + ranges[:, 0].reshape(1, -1)
        return pmock

    def generate_mock_flux(self, pmock, mock_snr=50, rv=0, wave_interp=None,
                           fill_flux=-1, fill_ivar=0., norm=True,
                           **norm_kwargs):
        if wave_interp is None:
            wave_interp = self.wave

        nmock = pmock.shape[0]
        # generate mock stellar spectra
        print("@SpIn: interpolate spectra ...")
        wave, mock_flux = self.interp(pmock, band="all")

        mock_z = np.random.randn(nmock) * rv / (const.c.value/1e3)

        mock_flux_rvin = []
        mock_ivar_rvin = []
        print("@SpIn: add noise to spectra ...")
        for iband in range(self.nband):
            # for each band, generate spectra + RV
            this_mock_flux_rvi = np.array([
                interp1d(wave[iband] * (1 + mock_z[i]), mock_flux[iband][i],
                         kind="linear", bounds_error=False,
                         fill_value=np.nan)(wave_interp[iband])
                for i in range(nmock)])

            # check nan
            this_mock_flux_rvi = np.where(np.isfinite(this_mock_flux_rvi),
                                           this_mock_flux_rvi, fill_flux)
            # this_mock_flux_rvin = np.where(np.isfinite(this_mock_flux_rvin),
            #                                this_mock_flux_rvin, fill_flux)

            # simulate S/N
            if mock_snr > 0:
                # add noise
                this_mock_flux_rvin, this_mock_ivar_rvin = add_gaussian_noise_to_spec(
                    this_mock_flux_rvi, snr=mock_snr, fill_ivar=0.)
                mock_flux_rvin.append(this_mock_flux_rvin)
                mock_ivar_rvin.append(this_mock_ivar_rvin)
            else:
                # return original interpolated spec

                mock_flux_rvin.append(this_mock_flux_rvi)
                mock_ivar_rvin.append(this_mock_flux_rvi*0)

        if not norm:
            return mock_flux_rvin, mock_ivar_rvin
        else:
            comb_norm_kwargs = dict(dwave=20,
                                    p=(1e-06, 1e-07),
                                    q=0.6,
                                    eps=1e-10,
                                    rsv_frac=1.0,
                                    n_jobs=1,
                                    verbose=1)
            comb_norm_kwargs.update(norm_kwargs)

            print("@SpIn: normalize spectra ...")
            mock_flux_rvin_norm = []
            mock_ivar_rvin_norm = []
            for iband in range(self.nband):
                this_mock_flux_rvin_norm, this_mock_flux_rvin_cont = \
                    normalize_spectra_block(wave_interp[iband],
                                            mock_flux_rvin[iband],
                                            wave_interp[iband][[0, -1]],
                                            **comb_norm_kwargs)
                mock_flux_rvin_norm.append(this_mock_flux_rvin_norm)
                mock_ivar_rvin_norm.append(mock_ivar_rvin[iband]*this_mock_flux_rvin_cont**2)

            return mock_flux_rvin_norm, mock_ivar_rvin_norm

    def select_finite(self, mock_flux, mock_ivar):
        pass

    def normalize_spectra_block(self, ):
        pass


def add_gaussian_noise_to_spec(spec, snr=100, fill_ivar=0.):
    """ add gaussian noise to spectra """
    specn = (1+np.random.randn(*spec.shape)/snr)*spec
    errn = spec/snr
    ivarn = errn**-2

    indbad = (errn <= 0) | (specn <= 0)
    ivarn = np.where(indbad, fill_ivar, ivarn)
    specn = np.where(indbad, 0., specn)
    return specn, ivarn



