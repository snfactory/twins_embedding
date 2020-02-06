"""
Simplest spectral indicator measurements

This code is originally from Sam Dixon (https://github.com/sam-dixon)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as US

LIMITS = {
    "CaIIHK": [3504, 3687, 3830, 3990],
    "SiII4000": [3830, 3963, 4034, 4150],
    "SiII5972": [5550, 5681, 5850, 6015],
    "SiII6355": [5850, 6015, 6250, 6365],
}


def vel_space(lam, lam0):
    """
    Convert a maximum absorption wavelength to a velocity
    using the relativistic Doppler formula

    Args:
        lam: maximum absorption wavelength
        lam0: rest-frame absorption wavelength

    Returns:
        v: ejecta velocity
    """
    z = lam / lam0
    return 3e5 * (z ** 2 - 1) / (z ** 2 + 1)


def wave_space(vel, lam0):
    """
    Convert a velocity to a maximum absorption wavelength

    Args:
        vel: velocity
        lam0: rest-frame absorption wavelength

    Returns:
        lam: maximum absorption wavelength
    """
    beta = vel / 3e5
    lam = lam0 * np.sqrt((1 + beta) / (1 - beta))
    return lam


class Spectrum(object):
    def __init__(self, wave, flux, var, smooth_type="spl"):
        """
        Class for measuring the following spectral indicators as simply as
        possible:
        + vSiII6355
        + EWSiII4000
        + EWCaIIHK

        Args:
            wave: wavelength the spectrum is observed at
            flux: flux observed
            var: flux variance

        Attributes:
            lamSiII6355: maximum absorption wavelength of SiII6355
            EWSiII4000: equivalent width of SiII4000 absorption feature
            EWCaIIHK: equivalent width of CaIIH&K absorption feature
        """
        self.wave = wave
        self.flux = flux
        self.var = var
        if self.var is None:
            self.var = np.abs(self.flux / 1e3)
        self.smoothed_wave = np.arange(min(wave), max(wave), 0.1)
        if smooth_type == "spl":
            self.smoothed_flux = self.smooth()
        elif smooth_type == "gauss_filt":
            self.smoothed_flux = self.gauss_filt()
        self._lamSiII6355 = None
        self._lamCaIIHK = None
        self._EWSiII4000 = None
        self._EWSiII5972 = None
        self._EWSiII6355 = None
        self._EWCaIIHK = None

    def smooth(self):
        """
        Smooth the input spectrum using a spline weighted by the variance

        Args:
            None

        Returns:
            smoothed_flux: spline evaluated along smoothed_wave
        """
        w = 1.0 / np.sqrt(self.var)
        spl = US(self.wave, self.flux, w=w)
        smoothed_flux = spl(self.smoothed_wave)
        return smoothed_flux

    def gauss_filt(self, smooth_fac=0.02, n_l=15):
        """
        Smooth the input spectrum using a inverse-variance weighted
        Gaussian filter (like in Blondin 2007)

        Args:
            None

        Returns:
            smoothed_flux: spline evaluated along smoothed_wave
        """
        y_smooth = []
        x = self.wave
        y = self.flux
        v = self.var
        for i in range(n_l, len(x) - n_l):
            sig_i = x[i] * smooth_fac
            g_i = np.array(
                [
                    1 / np.sqrt(2 * np.pi) * np.exp((-1 / sig_i * (x[j] - x[i])) ** 2)
                    for j in range(i - n_l, i + n_l)
                ]
            )
            w_i = g_i / v[i - n_l : i + n_l]
            y_smooth.append(np.dot(w_i, y[i - n_l : i + n_l]) / np.sum(w_i))
        spl = US(x[n_l - 1 : -n_l - 1], np.array(y_smooth), s=0, k=4)
        return spl(self.smoothed_wave)

    def pseudo_continuum(self, l_max_ind, r_max_ind):
        """
        Calculate the pseudo continuum

        Args:
            l_max_ind: index of maximum flux value in left region
            r_max_ind: index of maximum flux value in right region

        Returns:
            pc_sub_flux: pseudo-continuum subtracted smoothed flux
        """
        pc_delta_flux = self.smoothed_flux[r_max_ind] - self.smoothed_flux[l_max_ind]
        pc_delta_wave = self.smoothed_wave[r_max_ind] - self.smoothed_wave[l_max_ind]
        pseudo_cont_slope = pc_delta_flux / pc_delta_wave
        pseudo_cont_int = (
            self.smoothed_flux[r_max_ind]
            - pseudo_cont_slope * self.smoothed_wave[r_max_ind]
        )
        pc_sub_flux = self.smoothed_flux / (
            pseudo_cont_slope * self.smoothed_wave + pseudo_cont_int
        )
        return pc_sub_flux

    def find_extrema(self, feature_name, return_pc=False):
        """
        Find the extrema in the region defined by the given feature

        Args:
            feature_name: 'SiII6355', 'EWCaIIHK', or 'EWSiII4000'; feature
            to find extrema for
            return_pc: Return the pseudo-continuum subtracted flux if true

        Returns:
            l_max_ind: index of maximum flux value in left region
            c_min_ind: index of minimum pseudo-continuum subtracted flux
            value in center region
            r_max_ind: index of maximum flux value in right region
        """
        limit = LIMITS[feature_name]
        l_region = self.smoothed_flux[
            (limit[0] <= self.smoothed_wave) & (self.smoothed_wave < limit[1])
        ]
        r_region = self.smoothed_flux[
            (limit[2] <= self.smoothed_wave) & (self.smoothed_wave <= limit[3])
        ]
        try:
            l_max_ind = np.where(self.smoothed_flux == max(l_region))[0][0]
            r_max_ind = np.where(self.smoothed_flux == max(r_region))[0][0]
        except IndexError:
            raise ValueError
            # if return_pc:
            #     return None, None, None, None
            # else:
            #     return None, None, None
        pc_sub_flux = self.pseudo_continuum(l_max_ind, r_max_ind)
        c_region = pc_sub_flux[
            (limit[1] <= self.smoothed_wave) & (self.smoothed_wave < limit[2])
        ]
        c_min_ind = np.where(pc_sub_flux == min(c_region))[0][0]
        if return_pc:
            return l_max_ind, c_min_ind, r_max_ind, pc_sub_flux
        return l_max_ind, c_min_ind, r_max_ind

    @property
    def lamSiII6355(self):
        """
        Maximum absorption wavelength of the SiII6355 feature
        """
        if self._lamSiII6355 is None:
            _, c_min_ind, _ = self.find_extrema("SiII6355")
            if c_min_ind is not None:
                self._lamSiII6355 = self.smoothed_wave[c_min_ind]
            else:
                self._lamSiII6355 = np.nan
        return self._lamSiII6355

    @property
    def lamCaIIHK(self):
        """
        Maximum absorption wavelength of the CaIIHK feature
        """
        if self._lamCaIIHK is None:
            _, c_min_ind, _ = self.find_extrema("CaIIHK")
            if c_min_ind is not None:
                self._lamCaIIHK = self.smoothed_wave[c_min_ind]
            else:
                self._lamCaIIHK = np.nan
        return self._lamCaIIHK

    @property
    def EWSiII5972(self):
        """
        Equivalent width of the SiII5972 feature
        """
        if self._EWSiII5972 is None:
            l_max_ind, c_min_ind, r_max_ind, pc_sub_flux = self.find_extrema(
                "SiII5972", return_pc=True
            )
            if pc_sub_flux is not None:
                c_region = pc_sub_flux[l_max_ind:r_max_ind]
                self._EWSiII5972 = np.sum(1.0 - c_region) * 0.1
            else:
                self._EWSiII5972 = np.nan
        return self._EWSiII5972

    @property
    def EWSiII6355(self):
        """
        Equivalent width of the SiII6355 feature
        """
        if self._EWSiII6355 is None:
            l_max_ind, c_min_ind, r_max_ind, pc_sub_flux = self.find_extrema(
                "SiII6355", return_pc=True
            )
            if pc_sub_flux is not None:
                c_region = pc_sub_flux[l_max_ind:r_max_ind]
                self._EWSiII6355 = np.sum(1.0 - c_region) * 0.1
            else:
                self._EWSiII6355 = np.nan
        return self._EWSiII6355

    @property
    def EWSiII4000(self):
        """
        Equivalent width of the SiII4000 feature
        """
        if self._EWSiII4000 is None:
            l_max_ind, c_min_ind, r_max_ind, pc_sub_flux = self.find_extrema(
                "SiII4000", return_pc=True
            )
            if pc_sub_flux is not None:
                c_region = pc_sub_flux[l_max_ind:r_max_ind]
                self._EWSiII4000 = np.sum(1.0 - c_region) * 0.1
            else:
                self._EWSiII4000 = np.nan
        return self._EWSiII4000

    @property
    def EWCaIIHK(self):
        """
        Equivalent width of the CaIIH&K feature
        """
        if self._EWCaIIHK is None:
            l_max_ind, c_min_ind, r_max_ind, pc_sub_flux = self.find_extrema(
                "CaIIHK", return_pc=True
            )
            if pc_sub_flux is not None:
                c_region = pc_sub_flux[l_max_ind:r_max_ind]
                self._EWCaIIHK = np.sum(1.0 - c_region) * 0.1
            else:
                self._EWCaIIHK = np.nan
        return self._EWCaIIHK

    def get_spin_dict(self):
        """
        Get a dictionary with the available spectral indicators
        """
        spin_dict = {
            "lamSiII6355": self.lamSiII6355,
            "lamCaIIHK": self.lamCaIIHK,
            "vSiII6355": vel_space(self.lamSiII6355, 6355.0),
            "vCaIIHK": vel_space(self.lamCaIIHK, 3934.0),
            "EWSiII4000": self.EWSiII4000,
            "EWSiII5972": self.EWSiII5972,
            "EWSiII6355": self.EWSiII6355,
            "EWCaIIHK": self.EWCaIIHK,
        }
        return spin_dict
