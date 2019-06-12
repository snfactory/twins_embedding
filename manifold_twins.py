from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from hashlib import md5
from idrtools import Dataset, math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.stats import sigmaclip
from sklearn.manifold import Isomap
import extinction
import numpy as np
import os
import sys
import pickle
import pystan
import tqdm

from specind import Spectrum


idr_directory = '/home/scpdata06/kboone/snfactory/data/snfactory/idr/'

# default_idr = 'MARBLE'
# default_idr = 'BERNICE'
default_idr = 'CASCAD'
# default_idr = 'ALLEGv2'
# default_idr = 'KYLEPSF'
# default_idr = 'HICKORY'

cut_supernovae = [
    # Bad host subtraction failure in CASCAD and similar productions.
    'PTF12ecm', 
]


class ManifoldTwinsException(Exception):
    pass


def print_verbose(message, verbosity, threshold):
    if verbosity >= threshold:
        print(message)


def load_stan_code(path, cache_dir='./stan_cache', verbose=True):
    """Load Stan code at a given path.

    The compiled model is saved so that it can be reused later without having
    to recompile every time.
    """
    with open(path) as stan_code_file:
        model_code = stan_code_file.read()

    code_hash = md5(model_code.encode('ascii')).hexdigest()

    cache_path = '%s/stan-%s.pkl' % (cache_dir, code_hash)

    os.makedirs(cache_dir, exist_ok=True)

    try:
        model = pickle.load(open(cache_path, 'rb'))
        if verbose:
            print("Loaded cached stan model")
    except FileNotFoundError:
        print("Compiling stan model")
        model = pystan.StanModel(model_code=model_code)
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(model, cache_file)
        print("Compilation successful")

    return model


def frac_to_mag(fractional_difference):
    """Convert a fractional difference to a difference in magnitude

    Because this transformation is asymmetric for larger fractional changes, we
    take the average of positive and negative differences
    """
    pos_mag = 2.5*np.log10(1 + fractional_difference)
    neg_mag = 2.5*np.log10(1 - fractional_difference)
    mag_diff = (pos_mag - neg_mag) / 2.

    return mag_diff


def fill_mask(array, mask, fill_value=np.nan):
    """Fill in an array with masked out entries.
    
    Parameters
    ==========
    array : numpy.array with shape (N, ...)
        Array of elements for the masked entries. The first dimension, N,
        should be equal to the number of unmasked entries.
    mask : numpy.array of length M.
        Mask that was applied to select the entries in array. The selected
        entries should be set to True in mask, and there should be a total of N
        True values in mask.
    fill_value : scalar
        The value to fill with. Default: np.nan

    Returns
    =======
    filled_array : numpy.array with shape (M, ...)
        An array with the entries of the input array for the entries in mask,
        and fill_value elsewhere. filled_array[mask] will recover the original
        array.
    """
    filled_shape = array.shape
    filled_shape = (len(mask),) + filled_shape[1:]

    filled_array = np.zeros(filled_shape, dtype=array.dtype)
    filled_array[mask] = array
    filled_array[~mask] = fill_value

    return filled_array


class ManifoldTwinsAnalysis():
    def __init__(self, idr=default_idr, center_phase=0., phase_width=5.,
                 bin_velocity=1000., verbosity=1,
                 cut_supernovae=cut_supernovae, max_count=None):
        """Load the dataset"""
        if verbosity >= 1:
            print("Loading dataset...")
            print("IDR:          %s" % idr)
            print("Phase range: [%.1f, %.1f]" % (center_phase - phase_width,
                                                 center_phase + phase_width))
            print("Center phase: %.1f" % center_phase)
            print("Bin velocity: %.1f" % bin_velocity)
            if cut_supernovae:
                print("Cutting SNe:  %s" % cut_supernovae)
            if max_count is not None:
                print("WARNING: Only keeping first %d spectra!" % max_count)

        self.center_phase = center_phase
        self.phase_width = phase_width
        self.idr = idr

        self.dataset = Dataset.from_idr(
            os.path.join(idr_directory, idr), load_both_headers=True
        )

        # Load information about the extinction solutions to be able to cut bad
        # ones.
        self.ext_tab = Table.read('./data/extinction_solution_quality.txt',
                                  format='ascii')

        # Load information about the MFRs.
        self.mfr_tab = Table.read('./data/mfr_quality.fits')

        # An MFR is bad if it is missing any of the 3 filters that are used.
        # The SALT2 residuals show biases and increased dispersions when that
        # happens.
        has_f2 = (self.mfr_tab['f2_kernel_nstars'] > 0).astype(int)
        has_f3 = (self.mfr_tab['f3_kernel_nstars'] > 0).astype(int)
        has_f4 = (self.mfr_tab['f4_kernel_nstars'] > 0).astype(int)
        mfr_count = has_f2 + has_f3 + has_f4
        self.mfr_tab['good'] = mfr_count >= 2

        all_raw_spec = []
        center_mask = []

        sys.stdout.flush()

        for supernova in tqdm.tqdm(self.dataset.targets):
            if supernova.name in cut_supernovae:
                print_verbose("Cutting %s, explicit cut!" % supernova,
                              verbosity, 2)
                continue

            daymax_err = supernova['salt2.DayMax.err']
            if daymax_err > 1.0:
                print_verbose("Cutting %s, day max err %.2f too high" %
                              (supernova, daymax_err), verbosity, 2)
                continue

            if len(supernova.spectra) < 5:
                print_verbose("Cutting %s, not enough spectra to guarantee a "
                              "good LC fit" % supernova, verbosity, 2)
                continue

            # Get every spectrum for every supernova in the range.
            used_phases = []
            for spectrum in supernova.get_spectra_in_range(
                    center_phase - phase_width, center_phase +
                    phase_width):
                if self._check_spectrum(spectrum, verbosity):
                    all_raw_spec.append(spectrum)
                    used_phases.append(spectrum.phase)
                else:
                    spectrum.usable = False

                if max_count is not None and len(all_raw_spec) > max_count:
                    break

            used_phases = np.array(used_phases)
            if len(used_phases) > 0:
                # Figure out which spectrum was closest to the center of the
                # bin.
                target_center_mask = np.zeros(len(used_phases), dtype=bool)
                target_center_mask[np.argmin(np.abs(used_phases -
                                                    center_phase))] = True
                center_mask.extend(target_center_mask)

            if max_count is not None and len(all_raw_spec) > max_count:
                break

        all_flux = []
        all_fluxerr = []
        all_spec = []

        for spectrum in all_raw_spec:
            bin_spec = spectrum.bin_by_velocity(bin_velocity)
            all_flux.append(bin_spec.flux)
            all_fluxerr.append(bin_spec.fluxerr)
            all_spec.append(bin_spec)

        # All binned spectra have the same wavelengths, so save the wavelengths
        # from an arbitrary one of them.
        self.wave = all_spec[0].wave

        # Save the rest of the info
        self.flux = np.array(all_flux)
        self.fluxerr = np.array(all_fluxerr)
        self.raw_spectra = np.array(all_raw_spec)
        self.spectra = np.array(all_spec)
        self.center_mask = np.array(center_mask)

        # Pull out variables that we use all the time.
        self.salt_x1 = self.read_meta('salt2.X1')
        self.salt_color = self.read_meta('salt2.Color')
        self.salt_phases = self.read_meta('salt2.phase', center_only=False)
        self.redshifts = self.read_meta('host.zcmb')
        self.redshift_errs = self.read_meta('host.zhelio.err')
        self.distance_moduli = cosmo.distmod(self.redshifts).value

        # Build a list of targets and a map from spectra to targets.
        self.targets = np.unique([i.target for i in self.spectra])
        self.target_map = np.array([self.targets.tolist().index(i.target) for i
                                    in self.spectra])

        # Record which targets should be in the validation set.
        self.train_mask = np.array([i['idr.subset'] != 'validation' for i in
                                    self.targets])

    def _check_spectrum(self, spectrum, verbosity=1):
        """Check if a spectrum is valid or not"""
        spectrum.do_lazyload()

        timeon = np.min([spectrum['fits.timeon'],
                         spectrum['fits.r.timeon']])

        s2n_start = spectrum.get_signal_to_noise(3300, 3800)
        s2n_b = spectrum.get_snf_signal_to_noise('b')
        s2n_end = spectrum.get_signal_to_noise(8100, 8600)

        ext_row = self.ext_tab[self.ext_tab['night'] ==
                               spectrum['obs.exp'][:6]]
        std_weight = ext_row['std_weight']

        # Check if the MFR is good.
        photometric = spectrum['obs.photo']
        try:
            mfr_row = self.mfr_tab[self.mfr_tab['exp'] ==
                                   spectrum['obs.exp']][0]
            mfr_good = mfr_row['good']
        except IndexError:
            mfr_good = False

        redshift = spectrum.target['host.zcmb']
        airmass = spectrum['fits.airmass']

        if s2n_start < 100:
            # Signal-to-noise cut. We find that a signal-to-noise of
            # < ~100 in the U-band leads to an added core dispersion of
            # >0.1 mag in the U-band. This is unacceptable for the
            # twins analysis that relies on getting the color right for
            # a single spectrum.
            print_verbose("Cutting %s, start signal-to-noise %.2f "
                          "too low." % (spectrum, s2n_start),
                          verbosity, 2)
            return False
        # elif airmass < 1.5:
            # Airmass cut. We find that airmasses above 1.5 lead to large
            # residuals in the i-band.
            # print_verbose("Cutting %s, airmass %.2f too high." %
                          # (spectrum, airmass), verbosity, 2)
        # elif timeon < 30000:
            # Time on cut. Weird things happen when the detector was
            # recently turned on, including red wings. The noise in low
            # timeon images is very high and systematic, so we throw
            # them out.
            # print_verbose("Cutting %s, timeon %ds too short." %
                          # (spectrum, timeon), verbosity, 2)
            # return False
        # elif std_weight < 0.2:
            # Some nights have poor standard star choices leading to
            # bad extinction solutions. The "standard star weight" is
            # defined as the sum of the observed standard star
            # airmasses minus the mean airmass. In the case of only 2
            # stars observed, this is the difference in airmass between
            # them. If this value is too low (chosen to be 0.2 from
            # SALT2 residual tests), then the extinction solution can't
            # be properly measured.
            # print_verbose("Cutting %s, poor standard star "
                          # "distribution w=%.2f for extinction solution"
                          # % (spectrum, std_weight), verbosity, 2)
            # return False
        # elif redshift > 0.1:
            # print_verbose("Cutting %s, redshift %.2f > 0.1" %
                          # (spectrum, redshift), verbosity, 2)
        # elif (not photometric) and (not mfr_good):
            # print_verbose("Cutting %s, missing MFR filters" % spectrum,
                          # verbosity, 2)
            # return False

        # We made it!
        return True

    def read_meta(self, key, center_only=True):
        """Read a key from the meta data of each spectrum/target

        This will first attempt to read the key in the spectrum object's meta
        data. If it isn't there, then it will try to read from the target
        instead.

        If center_only is True, then a single value is returned for each
        target, from the spectrum closest to the center of the range if
        applicable. Otherwise, the values will be returned for each spectrum in
        the sample.
        """
        if key in self.spectra[0].meta:
            read_spectrum = True
        elif key in self.spectra[0].target.meta:
            read_spectrum = False
        else:
            raise KeyError("Couldn't find key %s in metadata." % key)

        if center_only:
            use_spectra = self.spectra[self.center_mask]
        else:
            use_spectra = self.spectra

        res = []
        for spec in use_spectra:
            if read_spectrum:
                val = spec.meta[key]
            else:
                val = spec.target.meta[key]
            res.append(val)

        res = np.array(res)

        return res

    def model_maximum_spectra(self, use_cache=True,
                              use_cached_model_uncertainty=False):
        """Run the phase interpolation algorithm to model spectra near maximum
        light.

        This algorithm uses all targets with multiple spectra to model how Type
        Ia supernovae evolve near maximum light. This method does not rely on
        knowing the underlying model of Type Ia supernovae and only models the
        differences. The model is generated in magnitude space, so anything
        static in between us and the supernova, like dust, does not affect the
        model.

        The fit is performed using Stan. We only use Stan as a minimizer here,
        and we do some analytic calculations inside to make the computation
        more robust that are not applicable for a proper Bayesian analysis!

        If use_cache is True, then the fitted model will be retrieved from a
        cache if it exists. Make sure to run with use_cache=False if making
        modifications to the model!

        If use_cached_model_uncertainty is True, then only the model
        uncertainty is used from the cached model. The model uncertainty is
        fairly stable across runs, but take a long time to fit properly.
        Setting use_cached_model_uncertainty to True overrides the value of
        use_cache.
        """
        # Try to load a cached output of this function if we can.
        if use_cache or use_cached_model_uncertainty:
            cache_result = self.load_interpolation_result('analytic')
            if cache_result is None:
                # No cache available, skip.
                pass
            elif use_cached_model_uncertainty:
                # Keep the model uncertainty from the previous run, but
                # recalculate everything else.
                print("Using cached model uncertainty, but refitting rest of "
                      "model")
                phase_dispersion_coefficients = \
                    cache_result['phase_dispersion_coefficients']
            elif use_cache:
                self.interpolation_result = cache_result
                self.maximum_flux = cache_result['maximum_flux']
                self.maximum_fluxerr = cache_result['maximum_fluxerr']
                return

        num_targets = len(self.targets)
        num_spectra = len(self.flux)
        num_wave = len(self.wave)
        num_phase_coefficients = 4

        if self.center_phase != 0:
            raise Exception("ERROR: Phase interpolation not yet implemented "
                            "for non-zero center phases")

        if num_phase_coefficients % 2 != 0:
            raise Exception("ERROR: Must have an even number of phase "
                            "coefficients.")

        spectra_targets = [i.target for i in self.spectra]
        spectra_target_counts = np.array(
            [spectra_targets.count(i.target) for i in self.spectra]
        )

        phase_coefficients = np.zeros((num_spectra, num_phase_coefficients))

        for i, phase in enumerate(self.salt_phases):
            phase_scale = np.abs((num_phase_coefficients / 2)
                                 * (phase / self.phase_width))

            full_bins = int(np.floor(phase_scale))
            remainder = phase_scale - full_bins

            for j in range(full_bins + 1):
                if j == full_bins:
                    weight = remainder
                else:
                    weight = 1

                if phase > 0:
                    phase_bin = num_phase_coefficients // 2 + j
                else:
                    phase_bin = num_phase_coefficients // 2 - 1 - j

                phase_coefficients[i, phase_bin] = weight

        def stan_init():
            init_params = {
                'phase_slope': np.zeros(num_wave),
                'phase_quadratic': np.zeros(num_wave),

                'phase_slope_x1': np.zeros(num_wave),
                'phase_quadratic_x1': np.zeros(num_wave),

                'phase_dispersion_coefficients': (
                    0.01 * np.ones((num_phase_coefficients, num_wave))),

                'gray_offsets': np.zeros(num_spectra),
                'gray_dispersion_scale': 0.02,
            }

            if not use_cached_model_uncertainty:
                init_params['phase_dispersion_coefficients'] = \
                    0.01 * np.ones((num_phase_coefficients, num_wave))

            return init_params

        self.target_map = np.array([self.targets.tolist().index(i.target) for i
                                    in self.spectra])

        stan_data = {
            'num_targets': num_targets,
            'num_spectra': num_spectra,
            'num_wave': num_wave,
            'measured_flux': self.flux,
            'measured_fluxerr': self.fluxerr,
            'phases': [i.phase for i in self.spectra],
            'phase_coefficients': phase_coefficients,
            'num_phase_coefficients': num_phase_coefficients,

            'spectra_target_counts': spectra_target_counts,
            'target_map': self.target_map + 1,  # stan uses 1-based indexing

            'salt_x1': self.salt_x1,

            # 'log_wavelength': np.log(self.wave),
        }

        if use_cached_model_uncertainty:
            stan_data['phase_dispersion_coefficients'] = \
                phase_dispersion_coefficients
            model_code = ('./stan_models/phase_interpolation_analytic_fixed_'
                          'model_uncertainty.stan')
        else:
            model_code = './stan_models/phase_interpolation_analytic.stan'

        self.stan_init = stan_init
        self.stan_data = stan_data

        model = load_stan_code(model_code)
        sys.stdout.flush()
        res = model.optimizing(data=stan_data, init=stan_init, verbose=True,
                               iter=20000, history_size=100)

        # For the analytic model, sampling doesn't work because we are
        # analytically calculating the MLE mean spectrum for each target.
        # res = model.sampling(data=stan_data, init=stan_init, verbose=True)

        # self.stan_model = model
        self.interpolation_result = res
        self.maximum_flux = self.interpolation_result['maximum_flux']
        self.maximum_fluxerr = self.interpolation_result['maximum_fluxerr']

        if use_cached_model_uncertainty:
            self.interpolation_result['phase_dispersion_coefficients'] = \
                phase_dispersion_coefficients

        # Save the output to cache
        cache_result = self.save_interpolation_result('analytic')

    def _get_interpolation_path(self, method):
        """Path to use for the results of a stan interpolation fit"""
        return "interpolations/interpolation_%s_%s_%.2f.pkl" % (
            self.idr, method, self.phase_width,
        )

    def save_interpolation_result(self, method):
        """Save a previously run interpolation to a pickle file"""
        path = self._get_interpolation_path(method)

        with open(path, 'wb') as outfile:
            pickle.dump(self.interpolation_result, outfile)

    def load_interpolation_result(self, method):
        """Load a previously run interpolation"""
        path = self._get_interpolation_path(method)

        try:
            with open(path, 'rb') as infile:
                print("Using saved interpolation result")
                return pickle.load(infile)
        except IOError:
            pass

        # No save result
        return None

    def read_between_the_lines(self, mask=None, mask_power_fraction=0.1,
                               blinded=True):
        """Run the read between the lines algorithm.

        This algorithm estimates the brightnesses and colors of every spectrum
        and produces dereddened spectra.

        If blinded is True, then the brightnesses of any validation supernovae
        are thrown out.

        The fit is performed using Stan. We only use Stan as a minimizer here.
        """
        color_law = extinction.fitzpatrick99(self.wave, 1., 2.8)

        if mask is None:
            mask = np.ones(len(self.targets), dtype=bool)

        use_targets = self.targets[mask]
        use_maximum_flux = self.maximum_flux[mask]
        use_maximum_fluxerr = self.maximum_fluxerr[mask]

        num_targets = len(use_targets)
        num_wave = len(self.wave)

        def stan_init():
            # Use the spectrum closest to maximum as a first guess of the
            # target's spectrum.
            start_mean_flux = np.mean(use_maximum_flux, axis=0)
            start_fractional_dispersion = 0.1 * np.ones(num_wave)

            return {
                'mean_flux': start_mean_flux,
                'fractional_dispersion': start_fractional_dispersion,

                'colors_raw': np.zeros(num_targets - 1),
                'magnitudes_raw': np.zeros(num_targets - 1),
            }

        stan_data = {
            'num_targets': num_targets,
            'num_wave': num_wave,
            'maximum_flux': use_maximum_flux,
            'maximum_fluxerr': use_maximum_fluxerr,
            'color_law': color_law,
        }

        model = load_stan_code('./stan_models/rbtl.stan')
        sys.stdout.flush()
        res = model.optimizing(data=stan_data, init=stan_init, verbose=True,
                               iter=5000)

        self.rbtl_result = res

        self.colors = fill_mask(res['colors'], mask)

        self.model_flux = fill_mask(res['model_flux'], mask)
        self.model_fluxerr = fill_mask(res['model_fluxerr'], mask)

        self.mean_flux = res['mean_flux']

        self.mags = fill_mask(res['magnitudes'], mask)

        if blinded:
            # Immediately discard validation magnitudes so that we can't
            # accidentally look at them.
            self.mags[~self.train_mask] = np.nan

        self.model_scales = fill_mask(res['model_scales'], mask)

        # Deredden the real spectra and set them to the same scale as the mean
        # spectrum.
        self.scale_flux = fill_mask(
            use_maximum_flux / res['model_scales'], mask
        )
        self.scale_fluxerr = fill_mask(
            use_maximum_fluxerr / res['model_scales'], mask
        )

        # Mask to select targets that have a measured magnitude
        self.mag_mask = ~np.isnan(self.mags)

        # Generate a mask that indicates which targets can be used for further
        # analysis. We start with the input mask, and reject any additional
        # targets whose interpolation uncertainty is large compared to the
        # intrinsic supernova dispersion.
        intrinsic_dispersion = frac_to_mag(res['fractional_dispersion'])
        intrinsic_power = np.sum(intrinsic_dispersion**2)
        interpolation_uncertainty = frac_to_mag(self.maximum_fluxerr /
                                                self.maximum_flux)
        interp_power = np.sum(interpolation_uncertainty**2, axis=1)
        self.interp_mask = (
            mask
            & (interp_power < mask_power_fraction * intrinsic_power)
        )
        print("Masking %d/%d targets whose interpolation uncertainty power is "
              "more than %.3f of the intrinsic power." %
              (np.sum(~self.interp_mask), len(self.interp_mask),
               mask_power_fraction))

        # Mask to select targets that have a magnitude that is expected to have
        # a reasonable dispersion in brightness.
        with np.errstate(invalid='ignore'):
            self.redshift_color_mask = (
                (self.redshift_errs < 0.004)
                & (self.redshifts > 0.02)
                & (self.colors - np.nanmedian(self.colors) < 0.5)
            )

            self.good_mag_mask = (
                self.mag_mask
                & self.interp_mask
                & self.redshift_color_mask
            )

    def do_embedding(self, n_neighbors=10, n_components=3):
        self.iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        mask = self.interp_mask
        self.trans = fill_mask(
            self.iso.fit_transform(self.scale_flux[mask] / self.mean_flux),
            mask
        )

    def get_indicators(self):
        """Calculate spectral indicators for all of the features"""
        all_indicators = []

        for idx in range(len(self.scale_flux)):
            spec = Spectrum(self.wave, self.scale_flux[idx],
                            self.scale_fluxerr[idx]**2)
            indicators = spec.get_spin_dict()
            all_indicators.append(indicators)

        all_indicators = Table(all_indicators)

        return all_indicators

    def do_blondin_plot(self, axis_1=0, axis_2=1, only_first=False):
        indicators = self.get_indicators()

        s1 = indicators['EWSiII6355']
        s2 = indicators['EWSiII5972']

        plt.figure()

        cut = s2 > 30
        plt.scatter(self.trans[cut, axis_1], self.trans[cut, axis_2], s=60,
                    c='r', label='Cool (CL)')
        cut = (s2 < 30) & (s1 < 70)
        plt.scatter(self.trans[cut, axis_1], self.trans[cut, axis_2], s=60,
                    c='g', label='Shallow silicon (SS)')
        cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
        plt.scatter(self.trans[cut, axis_1], self.trans[cut, axis_2], s=60,
                    c='black', label='Core normal (CN)')
        cut = (s2 < 30) & (s1 > 100)
        plt.scatter(self.trans[cut, axis_1], self.trans[cut, axis_2], s=60,
                    c='b', label='Broad line (BL)')

        plt.xlabel('First Isomap component')
        plt.ylabel('Second Isomap component')

        plt.legend()

        plt.savefig('./branch_classification.eps')

        if only_first:
            return

        plt.figure()

        cut = s2 > 30
        plt.scatter(s1[cut], s2[cut], s=60, c='r', label='Cool (CL)')
        cut = (s2 < 30) & (s1 < 70)
        plt.scatter(s1[cut], s2[cut], s=60, c='g',
                    label='Shallow silicon (SS)')
        cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
        plt.scatter(s1[cut], s2[cut], s=60, c='black',
                    label='Core normal (CN)')
        cut = (s2 < 30) & (s1 > 100)
        plt.scatter(s1[cut], s2[cut], s=60, c='b', label='Broad line (BL)')

        plt.xlabel('EW SiII 6355')
        plt.ylabel('EW SiII 5972')

        plt.legend()

    def do_blondin_plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        indicators = self.get_indicators()

        s1 = indicators['EWSiII6355']
        s2 = indicators['EWSiII5972']

        trans = self.trans

        cut = s2 > 30
        ax.scatter(trans[cut, 0], trans[cut, 1], trans[cut, 2], s=60, c='r',
                   label='Cool (CL)')
        cut = (s2 < 30) & (s1 < 70)
        ax.scatter(trans[cut, 0], trans[cut, 1], trans[cut, 2], s=60, c='g',
                   label='Shallow silicon (SS)')
        cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
        ax.scatter(trans[cut, 0], trans[cut, 1], trans[cut, 2], s=60,
                   c='black', label='Core normal (CN)')
        cut = (s2 < 30) & (s1 > 100)
        ax.scatter(trans[cut, 0], trans[cut, 1], trans[cut, 2], s=60, c='b',
                   label='Broad line (BL)')

        ax.set_xlabel('Component 0')
        ax.set_ylabel('Component 1')
        ax.set_zlabel('Component 2')

        ax.legend()

    def scatter(self, variable, cut=None, weak_cut=None, label='',
                axis_1=0, axis_2=1, axis_3=None, **kwargs):
        """Make a scatter plot of some variable against the embedded
        coefficients

        variable is the values to use for the color axis of the plot.

        A boolean array can be specified for cut to specify which points to use
        in the plot. If cut is None, then the full variable list is used.

        The target variable can be passed with or without the cut already
        applied. This function will check and automatically apply it or ignore
        it so that the variable array has the same length as the coefficient
        arrays.

        Optionally, a weak cut can be performed where spectra not passing the
        cut are plotted as small points rather than being completely omitted.
        To do this, specify the "weak_cut" parameter with a boolean array that
        has the length of the the variable array after the base cut.

        Any kwargs are passed to plt.scatter directly.
        """
        # Apply cut
        use_trans = self.trans
        if cut is not None:
            use_trans = use_trans[cut]
            if len(use_trans) != len(variable):
                variable = variable[cut]
            if len(use_trans) != len(variable):
                raise ManifoldTwinsException(
                    'Coefficients and variable have different lengths!'
                )

            if weak_cut is not None:
                if len(use_trans) != len(weak_cut):
                    weak_cut = weak_cut[cut]
                if len(use_trans) != len(weak_cut):
                    raise ManifoldTwinsException(
                        'Coefficients and weak_cut have different lengths!'
                    )

        if weak_cut is None:
            # Constant marker size
            marker_size = 50
        else:
            # Variable marker size
            marker_size = 5 + 45 * weak_cut

        fig = plt.figure()

        if use_trans.shape[1] >= 3 and axis_3 is not None:
            ax = fig.add_subplot(111, projection='3d')
            plot = ax.scatter(use_trans[:, axis_1], use_trans[:, axis_2],
                              use_trans[:, axis_3], s=marker_size, c=variable,
                              **kwargs)
            ax.set_zlabel('Component %d' % axis_3)
        else:
            ax = fig.add_subplot(111)
            plot = ax.scatter(use_trans[:, axis_1], use_trans[:, axis_2],
                              s=marker_size, c=variable, **kwargs)

        ax.set_xlabel('Component %d' % axis_1)
        ax.set_ylabel('Component %d' % axis_2)

        if label is not None:
            fig.colorbar(plot, label=label)
        else:
            fig.colorbar(plot)

    def _evaluate_polynomial(self, coordinates, coefficients, degree):
        """Evaluate a polynomial
        
        Parameters
        ==========
        coordinates : numpy.array
            Coordinates to evaluate the polynomial at. This should have the
            shape (num_points, num_dimensions). Where num_points is the number
            of different sets of coordinates to evaluate the polynomial at, and
            num_dimensions is the number of dimensions for each point.
        coefficients : numpy.array
            A list of N coefficients for the polynomial. The length of this
            depends on the degree and dimension.
        degree : int
            The degree of the polynomial. Only degrees of 0, 1 or 2 are
            supported.
        """ 
        dimension = coordinates.shape[-1]

        remaining_coefficients = coefficients

        # Degree 0
        model = coefficients[0]
        remaining_coefficients = coefficients[1:]

        # Degree 1
        if degree >= 1:
            num_coef = dimension
            linear_coef = remaining_coefficients[:num_coef]
            remaining_coefficients = remaining_coefficients[num_coef:]

            for dim in range(dimension):
                model += linear_coef[dim] * coordinates[:, dim]

        # Degree 2
        if degree >= 2:
            num_coef = dimension * (dimension + 1) // 2
            quad_coef = remaining_coefficients[:num_coef]
            remaining_coefficients = remaining_coefficients[num_coef:]

            coef_idx = 0
            for dim1 in range(dimension):
                for dim2 in range(dim1, dimension):
                    model += (quad_coef[coef_idx] * coordinates[:, dim1] *
                              coordinates[:, dim2])
                    coef_idx += 1

        return model

    def apply_polynomial_standardization(self, degree=1, kind='twins'):
        """Apply polynomial standardization to the dataset.

        The degree can be up to 2.

        All of the isomap coordinates are used along with color.
        """
        if degree not in [0, 1, 2]:
            message = ("Degree %d not supported for polynomial "
                       "standardization!" % dimension)
            raise ManifoldTwinsException(message)

        good_trans, good_mags, good_colors, good_mask = self._get_gp_mags(kind)
        full_trans, full_mags, full_colors, full_mask = \
            self._get_gp_mags(kind, full=True)

        # Figure out how many parameters we need for the fit. Note that there
        # is an additional linear term for color.
        dimension = good_trans.shape[-1] + 1
        num_parameters = 1
        if degree >= 1:
            num_parameters += dimension
        if degree >= 2:
            num_parameters += (dimension**2 + dimension) // 2

        def apply_corr(coefficients, trans, mags, colors):
            stack_x = np.concatenate((colors[:, None], trans), axis=1)

            model = self._evaluate_polynomial(stack_x, coefficients, degree)

            diff = mags - model

            return diff

        def to_min(x):
            corr_residuals = apply_corr(x, good_trans, good_mags, good_colors)
            result = np.sum(corr_residuals**2)
            return result

        res = minimize(to_min, np.zeros(num_parameters))
        print("Fitted coefficients:")
        print(res.x)

        self.corr_mags = fill_mask(
            apply_corr(res.x, full_trans, full_mags, full_colors),
            full_mask
        )

        good_corr_mags = self.corr_mags[good_mask]

        print("Fit NMAD:       ", math.nmad(good_corr_mags))
        print("Fit std:        ", np.std(good_corr_mags))

    def _build_gp(self, x, hyperparameters=None, phase=False):
        """Build a george Gaussian Process object and kernels.
        """
        import george
        from george import kernels

        if hyperparameters is None:
            hyperparameters = self.gp_hyperparameters

        yerr = hyperparameters[0] * np.ones(len(x))

        ndim = x.shape[-1]
        if phase:
            use_dim = list(range(ndim - 1))
        else:
            use_dim = list(range(ndim))

        kernel = (
            hyperparameters[1]**2 *
            kernels.Matern32Kernel([hyperparameters[2]**2]*len(use_dim),
                                     ndim=ndim, axes=use_dim)
        )

        if phase:
            # Additional kernel in phase direction.
            kernel += (
                hyperparameters[3]**2 *
                kernels.Matern32Kernel([hyperparameters[4]**2],
                                         ndim=ndim, axes=ndim-1)
            )

        gp = george.GP(kernel)
        gp.compute(x, yerr)

        return gp

    def _predict_gp(self, pred_x, x, y, hyperparameters=None, return_cov=False,
                    phase=False, **kwargs):
        """Predict a Gaussian Process on the given data with a single shared
        length scale and assumed intrinsic dispersion
        """
        gp = self._build_gp(x, hyperparameters, phase=phase)

        pred = gp.predict(y, np.atleast_2d(pred_x), return_cov=return_cov,
                          **kwargs)

        return pred

    def _predict_gp_oos(self, x, y, hyperparameters=None, condition_mask=None,
                        return_var=False, phase=False, groups=None):
        """Do out-of-sample Gaussian Process predictions given hyperparameters

        A binary mask can be specified as condition_mask to specify a subset of
        the data to use for conditioning the GP. The predictions will be done
        on the full sample.
        """
        if condition_mask is None:
            cond_x = x
            cond_y = y
        else:
            cond_x = x[condition_mask]
            cond_y = y[condition_mask]

        # Do out-of-sample predictions for element in the condition sample.
        cond_preds = []
        cond_vars = []
        for idx in range(len(cond_x)):
            if groups is not None:
                match_idx = groups[idx] == groups
                del_x = cond_x[~match_idx]
                del_y = cond_y[~match_idx]
            else:
                del_x = np.delete(cond_x, idx, axis=0)
                del_y = np.delete(cond_y, idx, axis=0)
            pred = self._predict_gp(cond_x[idx], del_x, del_y, hyperparameters,
                                    return_var=return_var, phase=phase)
            if return_var:
                cond_preds.append(pred[0][0])
                cond_vars.append(pred[1][0])
            else:
                cond_preds.append(pred[0])

        # Do standard predictions for elements that we aren't conditioning on.
        if condition_mask is None:
            all_preds = np.array(cond_preds)
            if return_var:
                all_vars = np.array(cond_vars)
        else:
            other_pred = self._predict_gp(x[~condition_mask], cond_x, cond_y,
                                          hyperparameters,
                                          return_var=return_var, phase=phase)
            all_preds = np.zeros(len(x))
            all_vars = np.zeros(len(x))

            if return_var:
                other_pred, other_vars = other_pred
                all_vars[condition_mask] = cond_vars
                all_vars[~condition_mask] = other_vars

            all_preds[condition_mask] = cond_preds
            all_preds[~condition_mask] = other_pred

        if return_var:
            return all_preds, all_vars
        else:
            return all_preds

    def _get_gp_mags(self, kind='twins', full=False):
        if kind == 'twins':
            mags = self.mags
            colors = self.colors
            if full:
                mask = self.mag_mask
            else:
                mask = self.good_mag_mask
        elif kind == 'salt' or kind == 'salt_raw':
            if kind == 'salt':
                mags = self.salt_hr
            elif kind == 'salt_raw':
                mags = self.salt_hr_raw
            colors = self.salt_color
            if full:
                mask = self.salt_mask & self.interp_mask
            else:
                mask = self.good_salt_mask & self.interp_mask
        else:
            raise ManifoldTwinsException("Unknown kind %s!" % kind)

        use_mags = mags[mask]
        use_colors = colors[mask]
        use_trans = self.trans[mask]

        return use_trans, use_mags, use_colors, mask

    def _calculate_gp_residuals(self, hyperparameters=None, kind='twins',
                                **kwargs):
        """Calculate the GP prediction residuals for a set of
        hyperparameters
        """
        use_trans, use_mags, use_colors, use_mask = self._get_gp_mags(kind)

        preds = self._predict_gp_oos(use_trans, use_mags, hyperparameters,
                                     **kwargs)
        residuals = use_mags - preds
        return residuals

    def _calculate_gp_dispersion(self, hyperparameters=None, metric=np.std,
                                 **kwargs):
        """Calculate the GP dispersion for a set of hyperparameters"""
        return metric(self._calculate_gp_residuals(hyperparameters, **kwargs))

    def fit_gp(self, verbose=True, kind='twins'):
        """Fit a Gaussian Process to predict magnitudes for the data."""
        print("Fitting GP hyperparameters...")

        good_trans, good_mags, use_colors, good_mask = self._get_gp_mags(kind)

        def to_min(x):
            gp = self._build_gp(good_trans, x)
            return -gp.log_likelihood(good_mags)

        res = minimize(to_min, [0.1, 0.3, 5])
        print("Fit result:")
        print(res)
        self.gp_hyperparameters = res.x

        full_trans, full_mags, full_colors, full_mask = \
            self._get_gp_mags(kind, full=True)

        preds = self._predict_gp_oos(
            full_trans, full_mags,
            condition_mask=good_mask[full_mask]
        )

        self.corr_mags = fill_mask(full_mags - preds, full_mask)
        good_corr_mags = self.corr_mags[good_mask]

        print("Fit NMAD:       ", math.nmad(good_corr_mags))
        print("Fit std:        ", np.std(good_corr_mags))

    def apply_gp_standardization(self, verbose=True, hyperparameters=None,
                                 phase=False, kind='twins'):
        """Use a Gaussian Process to predict magnitudes for the data.

        If hyperparameters is specified, then the hyperparameters are used
        directly. Otherwise, the hyperparameters are fit to the data.
        """
        if hyperparameters is None:
            print("Fitting GP hyperparameters...")

            def to_min(x):
                return self._calculate_gp_dispersion(x, phase=phase, kind=kind)

            res = minimize(to_min, [0.1, 0.3, 5])
            print("Fit result:")
            print(res)
            self.gp_hyperparameters = res.x
        else:
            print("Using fixed GP hyperparameters...")
            self.gp_hyperparameters = hyperparameters

        good_trans, good_mags, good_colors, good_mask = self._get_gp_mags(kind)
        full_trans, full_mags, full_colors, full_mask = \
            self._get_gp_mags(kind, full=True)

        preds = self._predict_gp_oos(
            full_trans, full_mags,
            condition_mask=good_mask[full_mask], phase=phase
        )

        self.corr_mags = fill_mask(full_mags - preds, full_mask)
        good_corr_mags = self.corr_mags[good_mask]

        print("Fit NMAD:       ", math.nmad(good_corr_mags))
        print("Fit std:        ", np.std(good_corr_mags))

    def predict_gp(self, x, hyperparameters=None, **kwargs):
        """Do the GP prediction at specific points using the full GP
        conditioning.

        Note: this function uses all of the training data to make predictions.
        Use _predict_gp_oos or something similar to properly do out of sample
        predictions if you want to predict on the training data.
        """
        use_trans = self.trans[self.good_mag_mask]
        use_mags = self.mags[self.good_mag_mask]

        preds = self._predict_gp(x, use_trans, use_mags, hyperparameters,
                                 **kwargs)

        return preds

    def plot_gp(self, axis_1=0, axis_2=1, hyperparameters=None, num_points=50,
                border=0.5):
        """Plot the GP predictions with data overlayed."""
        x = self.trans[:, axis_1]
        y = self.trans[:, axis_2]

        min_x = np.nanmin(x) - border
        max_x = np.nanmax(x) + border
        min_y = np.nanmin(y) - border
        max_y = np.nanmax(y) + border

        plot_x, plot_y = np.meshgrid(np.linspace(min_x, max_x, num_points),
                                     np.linspace(min_y, max_y, num_points))

        flat_plot_x = plot_x.flatten()
        flat_plot_y = plot_y.flatten()

        plot_coords = np.zeros((len(flat_plot_x), self.trans.shape[1]))

        plot_coords[:, axis_1] = flat_plot_x
        plot_coords[:, axis_2] = flat_plot_y

        print(plot_coords.shape)

        pred = self.predict_gp(plot_coords, hyperparameters)
        pred = pred.reshape(plot_x.shape)

        plt.figure()
        plt.imshow(pred[::-1], extent=(min_x, max_x, min_y, max_y),
                   cmap=plt.cm.coolwarm, vmin=-0.2, vmax=0.2)
        plt.scatter(x, y, c=self.mags, edgecolors='k', vmin=-0.2, vmax=0.2,
                    s=10 + 50 * self.good_mag_mask, cmap=plt.cm.coolwarm)

    def load_host_data(self):
        """Load host data from Rigault et al. 2019"""
        host_data = Table.read(
            './data/host_properties_rigault_2019.txt',
            format='ascii'
        )
        all_host_idx = []
        host_mask = []
        for target in self.targets:
            name = target.name
            match = host_data['name'] == name

            # Check if found
            if not np.any(match):
                host_mask.append(False)
                continue

            # row = host_data[match][0]
            all_host_idx.append(np.where(match)[0][0])
            host_mask.append(True)

        # Save the loaded data
        self.host_mask = np.array(host_mask)
        fill_host_data = fill_mask(host_data[all_host_idx].as_array(),
                                   self.host_mask)
        self.host_data = Table(fill_host_data, names=host_data.columns)

    def plot_host_variable(self, variable, mask=None, mag_type='twins',
                           match_masks=False, threshold=None):
        """Plot diagnostics for some host variable.

        Valid variable names are the keys in the host_data Table that comes
        from load_host_data.

        mag_type selects which magnitudes to plot. The options are:
        - twins: use the manifold twins magnitudes (default)
        - salt: use the SALT2 corrected Hubble residuals

        If match_masks is True, then the masks required for both the twins
        manifold and SALT2 are applied (leaving a smaller dataset).
        """
        if mask is None:
            mask = np.ones(len(self.targets), dtype=bool)

        if match_masks:
            mag_mask = mask & self.good_salt_mask & self.good_mag_mask
        elif mag_type == 'twins':
            mag_mask = mask & self.good_mag_mask
        elif mag_type == 'salt':
            mag_mask = mask & self.good_salt_mask

        if mag_type == 'twins':
            host_corr_mags = self.corr_mags
        elif mag_type == 'salt':
            host_corr_mags = self.salt_hr

        host_mask = self.host_mask

        # Find a threshold for a split
        host_values = np.squeeze(self.host_data[variable])

        if threshold is None:
            threshold = np.nanmedian(host_values)

        with np.errstate(invalid='ignore'):
            host_cut_1 = host_mask & (host_values < threshold)
            host_cut_2 = host_mask & (host_values > threshold)

        host_mags_1 = host_corr_mags[host_cut_1 & mag_mask]
        host_mags_2 = host_corr_mags[host_cut_2 & mag_mask]

        mean_diff = np.nanmean(host_mags_1) - np.nanmean(host_mags_2)
        mean_diff_err = np.sqrt(
            np.var(host_mags_1) / len(host_mags_1) +
            np.var(host_mags_2) / len(host_mags_2)
        )

        print("Threshold:   %.3f" % threshold)
        print("Mean diff:   %.4f ± %.4f mag" % (mean_diff, mean_diff_err))
        print("Median diff: %.4f mag" % (np.nanmedian(host_mags_1) -
              np.nanmedian(host_mags_2)))

        plt.figure(figsize=(8, 8))
        plt.title(variable)
        plt.subplot(2, 2, 1)
        plt.scatter(
            self.trans[:, 0],
            self.trans[:, 1],
            c=host_values,
            cmap=plt.cm.coolwarm,
            vmin=threshold - 0.1,
            vmax=threshold + 0.1
        )
        plt.xlabel('Isomap parameter 0')
        plt.ylabel('Isomap parameter 1')

        plt.subplot(2, 2, 2)
        plt.scatter(
            host_values,
            host_corr_mags,
            s=2 + 30 * mag_mask,
            cmap=plt.cm.coolwarm
        )
        plt.xlabel(variable)
        plt.ylabel('Corrected mags')

        plt.subplot(2, 2, 3)
        plt.scatter(
            host_values,
            self.trans[:, 1],
            c=self.trans[:, 0]
        )
        plt.xlabel(variable)
        plt.ylabel('Isomap parameter 1')

        plt.subplot(2, 2, 4)
        plt.scatter(
            host_values,
            self.trans[:, 0],
            c=self.trans[:, 1]
        )
        plt.xlabel(variable)
        plt.ylabel('Isomap parameter 0')

        plt.tight_layout()

    def plot_host(self, threshold=None, mask=None):
        """Make an interactive plot of host properties"""
        from ipywidgets import interact, fixed

        interact(self.plot_host_variable, variable=self.host_data.keys()[1:],
                 mask=fixed(mask), threshold=fixed(threshold),
                 mag_type=['twins', 'salt'])

    def plot_distances(self):
        """Plot the reconstructed distances from the embedding against the true
        distances
        """
        from scipy.spatial.distance import pdist

        use_spec = self.scale_flux / self.mean_flux
        spec_dists = pdist(use_spec)
        trans_dists = pdist(self.trans)

        plt.figure()
        plt.scatter(spec_dists, trans_dists * np.median(spec_dists) /
                    np.median(trans_dists), s=1, alpha=0.1)
        plt.xlabel('Twins distance')
        plt.ylabel('Scaled transformed distance')

    def plot_twin_distances(self, twins_percentile=10):
        """Plot a histogram of where twins show up in the transformed
        embedding.
        """
        from IPython.display import display
        from scipy.spatial.distance import pdist
        from scipy.stats import percentileofscore
        import pandas as pd

        use_spec = self.scale_flux / self.mean_flux
        spec_dists = pdist(use_spec)
        trans_dists = pdist(self.trans)

        splits = {
            'Best 10% of twins': (0, 10),
            '10-20%': (10, 20),
            '20-50%': (20, 50),
            'Worst 50% of twins': (50, 100),
        }

        # Set weight so that the histogram is 1 if we have every element in
        # that bin.
        weight = 100 / len(trans_dists)

        all_percentiles = []
        all_weights = []

        all_spec_cuts = []
        all_trans_cuts = []

        for label, (min_percentile, max_percentile) in splits.items():
            spec_cut = (
                (spec_dists >= np.percentile(spec_dists, min_percentile)) &
                (spec_dists < np.percentile(spec_dists, max_percentile))
            )
            trans_cut = (
                (trans_dists >= np.percentile(trans_dists, min_percentile)) &
                (trans_dists < np.percentile(trans_dists, max_percentile))
            )
            percentiles = []
            for dist in trans_dists[spec_cut]:
                percentiles.append(percentileofscore(trans_dists, dist))
            percentiles = np.array(percentiles)
            weights = np.ones(len(percentiles)) * weight

            all_percentiles.append(percentiles)
            all_weights.append(weights)
            all_spec_cuts.append(spec_cut)
            all_trans_cuts.append(trans_cut)

        plt.figure()
        plt.hist(
            all_percentiles,
            100, (0, 100),
            weights=all_weights,
            histtype='barstacked',
            label=splits.keys(),
        )
        plt.xlabel('Recovered twins percentile from the embedded space')
        plt.ylabel('Fraction in bin')
        plt.legend()

        plt.xlim(0, 100)
        plt.ylim(0, 1)

        for label, (min_percentile, max_percentile) in splits.items():
            plt.axvline(max_percentile, c='k', lw=2, ls='--')

        # Build leakage matrix.
        leakage_matrix = np.zeros((len(splits), len(splits)))
        for idx_1, label_1 in enumerate(splits.keys()):
            for idx_2, label_2 in enumerate(splits.keys()):
                spec_cut = all_spec_cuts[idx_1]
                trans_cut = all_trans_cuts[idx_2]
                leakage = np.sum(trans_cut & spec_cut) / np.sum(spec_cut)
                leakage_matrix[idx_1, idx_2] = leakage

        # Print the leakage matrix using pandas
        df = pd.DataFrame(
            leakage_matrix,
            index=['From %s' % i for i in splits.keys()],
            columns=['To %s' % i for i in splits.keys()],
        )
        display(df)

        return leakage_matrix

        # Print stats
        # twins_leakage = np.sum((trans_dists > np.percentile(trans_dists, 10)) &
                               # twins_cut) / len(twins_cut)
        # print("Twins leakage:", twins_leakage)
        # worst_leakage = np.sum((trans_dists < np.percentile(trans_dists, 50)) &
                               # worst_cut) / len(worst_cut)
        # print("Worst leakage:", worst_leakage)

    def plot_twin_pairings(self):
        """Plot the twins delta M as a function of twinness ala Fakhouri"""
        from scipy.spatial.distance import pdist
        from scipy.stats import percentileofscore

        scale_spec = (self.scale_flux / self.mean_flux)
        use_spec = scale_spec[self.malphaag_cut & self.train_cut]

        use_mag = self.mags[self.mag_cut[self.train_cut]]

        spec_dists = pdist(use_spec)
        delta_mags = pdist(use_mag[:, None])

        percentile = np.array([percentileofscore(spec_dists, i) for i in
                               spec_dists])

        mags_20 = delta_mags[percentile < 20]
        print("RMS  20%:", math.rms(mags_20) / np.sqrt(2))
        print("NMAD 20%:", math.nmad(mags_20) / np.sqrt(2))

        plt.figure()
        math.plot_binned_rms(percentile, delta_mags / np.sqrt(2), bins=20,
                             label='RMS', equal_bin_counts=True)
        math.plot_binned_nmad(percentile, delta_mags / np.sqrt(2), bins=20,
                              label='NMAD', equal_bin_counts=True)

        plt.xlabel('Twinness percentile')
        plt.ylabel('Single supernova dispersion in brightness (mag)')

        plt.legend()

        return mags_20

    def plot_same_target_pairings(self):
        """Plot the pairings with spectra from the same target labeled"""
        # color_mapping = {i.target: 'C%d' % (j % 10) for j, i in
                         # enumerate(np.unique(self.spectra))}
        color_mapping = {i.target: np.random.rand(3) for j, i in
                         enumerate(np.unique(self.spectra))}
        colors = [color_mapping[i.target] for i in self.spectra]

        plt.figure()
        plt.scatter(self.trans[:, 0], self.trans[:, 1], c=colors)

        for i in range(len(self.spectra)):
            target_1 = self.spectra[i].target
            trans_1 = self.trans[i]
            for j in range(len(self.spectra)):
                target_2 = self.spectra[j].target
                trans_2 = self.trans[j]

                if i >= j:
                    continue

                if target_1 != target_2:
                    continue

                plt.plot([trans_1[0], trans_2[0]], [trans_1[1], trans_2[1]],
                         c=color_mapping[target_1])

    def _evaluate_salt_hubble_residuals(self, MB, alpha, beta,
                                        intrinsic_dispersion):
        """Evaluate SALT Hubble residuals for a given set of standardization
        parameters

        Parameters
        ==========
        MB : float
            The intrinsic B-band brightness of Type Ia supernovae
        alpha : float
            Standardization coefficient for the SALT2 x1 parameter
        beta : float
            Standardization coefficient for the SALT2 color parameter
        intrinsic_dispersion : float
            Assumed intrinsic dispersion of the sample.

        Returns
        =======
        residuals : numpy.array
            The Hubble residuals for every target in the dataset
        residual_uncertainties : numpy.array
            The uncertainties on the Hubble residuals for every target in the
            dataset.
        """
        mb = np.array([i['salt2.RestFrameMag_0_B'] for i in self.targets])
        mb_err = np.array([i['salt2.RestFrameMag_0_B.err'] for i in
                           self.targets])
        x1_err = np.array([i['salt2.X1.err'] for i in self.targets])
        color_err = np.array([i['salt2.Color.err'] for i in self.targets])

        residuals = (
            mb
            - MB
            - self.distance_moduli
            + alpha * self.salt_x1
            - beta * self.salt_color
        )

        residual_uncertainties = np.sqrt(
            intrinsic_dispersion**2
            + mb_err**2
            + alpha**2 * x1_err**2
            + beta**2 * color_err**2
        )

        return residuals, residual_uncertainties

    def calculate_salt_hubble_residuals(self):
        """Calculate SALT hubble residuals"""
        # For SALT, can only use SNe that are in the good sample
        self.salt_mask = np.array(
            [i['idr.subset'] in ['training', 'validation'] for i in
             self.targets]
        )

        # We also require reasonable redshifts and colors for the determination
        # of standardization parameters. The redshift_color_mask produced by
        # the read_between_the_lines algorithm does this.
        self.good_salt_mask = self.salt_mask & self.redshift_color_mask

        mask = self.good_salt_mask

        # Starting value for intrinsic dispersion. We will update this in each
        # round to set chi2 = 1
        intrinsic_dispersion = 0.1

        for i in range(5):
            def calc_dispersion(MB, alpha, beta, intrinsic_dispersion):
                residuals, residual_uncertainties = \
                    self._evaluate_salt_hubble_residuals(
                        MB, alpha, beta, intrinsic_dispersion
                    )

                mask_residuals = residuals[mask]
                mask_residual_uncertainties = residual_uncertainties[mask]

                weights = 1 / mask_residual_uncertainties**2

                dispersion = np.sqrt(
                    np.sum(weights * mask_residuals**2) / np.sum(weights)
                    / ((len(mask_residuals) - 1) / len(mask_residuals))
                )

                return dispersion
            
            def to_min(x):
                return calc_dispersion(*x, intrinsic_dispersion)

            res = minimize(to_min, [-19, 0.13, 3.])
            
            MB, alpha, beta = res.x
            
            print("Pass %d, MB=%.3f, alpha=%.3f, beta=%.3f" % (i, MB, alpha,
                                                               beta))
            
            # Reestimate intrinsic dispersion.
            def chisq(intrinsic_dispersion):
                residuals, residual_uncertainties = \
                    self._evaluate_salt_hubble_residuals(
                        MB, alpha, beta, intrinsic_dispersion
                    )

                mask_residuals = residuals[mask]
                mask_residual_uncertainties = residual_uncertainties[mask]

                return (
                    np.sum(mask_residuals**2 / mask_residual_uncertainties**2)
                    / len(mask_residuals)
                )

            def to_min(x):
                chi2 = chisq(x[0])
                return (chi2 - 1)**2

            res_int_disp = minimize(to_min, [intrinsic_dispersion])

            intrinsic_dispersion = res_int_disp.x[0]
            print("  -> new intrinsic_dispersion=%.3f" % intrinsic_dispersion)

        self.salt_MB = MB
        self.salt_alpha = alpha
        self.salt_beta = beta
        self.salt_intrinsic_dispersion = intrinsic_dispersion

        residuals, residual_uncertainties = \
            self._evaluate_salt_hubble_residuals(
                MB, alpha, beta, intrinsic_dispersion
            )

        # Set median to 0 to remove absolute flux zeropoint
        self.salt_hr = residuals
        self.salt_hr_uncertainties = residual_uncertainties

        # Save raw residuals without alpha and beta corrections applied.
        raw_residuals, raw_residual_uncertainties = \
            self._evaluate_salt_hubble_residuals(
                MB, 0, 0, intrinsic_dispersion
            )
        self.salt_hr_raw = raw_residuals - np.mean(raw_residuals)

        print("SALT2 Hubble fit: ")
        print("    MB:   ", self.salt_MB)
        print("    alpha:", self.salt_alpha)
        print("    beta: ", self.salt_beta)
        print("    σ_int:", self.salt_intrinsic_dispersion)
        print("    std:  ", np.std(self.salt_hr[self.good_salt_mask]))
        print("    NMAD: ", math.nmad(self.salt_hr[self.good_salt_mask]))

    def calculate_salt_hubble_residuals_old(self):
        """Calculate SALT hubble residuals"""
        from astropy.cosmology import Planck15
        # For SALT, can only use SNe that are in the good sample
        self.salt_mask = np.array(
            [i['idr.subset'] in ['training', 'validation'] for i in
             self.targets]
        )

        # We also require reasonable redshifts and colors for the determination
        # of standardization parameters. The redshift_color_mask produced by
        # the read_between_the_lines algorithm does this.
        self.good_salt_mask = self.salt_mask & self.redshift_color_mask

        # Determine standardization parameters
        fit_redshift = self.redshifts[self.good_salt_mask]
        fit_color = self.salt_color[self.good_salt_mask]
        fit_x1 = self.salt_x1[self.good_salt_mask]

        all_salt_mb = np.array(
            [i.meta['salt2.RestFrameMag_0_B'] for i in self.targets]
        )
        fit_salt_mb = all_salt_mb[self.good_salt_mask]

        def calc_salt_hr(x, redshift, salt_mb, x1, color):
            salt_hr = (
                salt_mb
                - Planck15.distmod(redshift).value
                + x[0]*x1
                - x[1]*color
            )
            return salt_hr

        def to_min(x):
            salt_hr = calc_salt_hr(x, fit_redshift, fit_salt_mb, fit_x1,
                                   fit_color)
            return np.std(salt_hr)

        res = minimize(to_min, [0.15, 3.1])

        self.salt_alpha = res.x[0]
        self.salt_beta = res.x[1]
        self.salt_hr = calc_salt_hr(res.x, self.redshifts, all_salt_mb,
                                    self.salt_x1, self.salt_color)

        # Set median to 0 to remove absolute flux zeropoint
        self.salt_hr -= np.median(self.salt_hr)

        print("SALT2 Hubble fit: ")
        print("    alpha:", self.salt_alpha)
        print("    beta: ", self.salt_beta)
        print("    std:  ", np.std(self.salt_hr[self.good_salt_mask]))
        print("    NMAD: ", math.nmad(self.salt_hr[self.good_salt_mask]))
