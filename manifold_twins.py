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


basedir = '/home/scpdata06/kboone/snfactory/data/snfactory/'
# default_idr = 'MARBLE'
# default_idr = 'BERNICE'
default_idr = 'CASCAD'
# default_idr = 'ALLEGv2'
# default_idr = 'KYLEPSF'
# default_idr = 'MARBLE'

cut_supernovae = [
    # 'LSQ12fhs',   # Iax
    # 'SN2011ay',   # Iax
    # 'SN2011de',
    # 'SNF20070714-007',
    # 'SNNGC6430',
    # 'PTF11kjn',
    # 'PTF12iiq',
    # 'SNF20070403-000',
    # 'SNNGC6956',
    # 'SNF20080723-012',
]


class ManifoldTwinsException(Exception):
    pass


def print_verbose(message, verbosity, threshold):
    if verbosity >= threshold:
        print(message)


def load_stan_code(path, cache_dir='./stan_cache'):
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
        print("Loaded cached stan model")
    except FileNotFoundError:
        print("Compiling stan model")
        model = pystan.StanModel(model_code=model_code)
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(model, cache_file)
        print("Compilation successful")

    return model


class ManifoldTwinsAnalysis():
    def __init__(self, idr=default_idr, center_phase=0., phase_width=2.5,
                 bin_velocity=2000., verbosity=1, cut_supernovae=[],
                 all_spectra=False):
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

        self.dataset = Dataset.from_idr(basedir + 'idr/' + idr,
                                        load_both_headers=True)

        # Load information about the extinction solutions to be able to cut bad
        # ones.
        self.ext_tab = Table.read('../ext_sol/ext_offsets.txt', format='ascii')

        all_raw_spec = []

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
                print_verbose("Cutting %s, not enough spectra for LC fit" %
                              supernova, verbosity, 2)
                continue

            if all_spectra:
                # Get every spectrum for every supernova in the range.
                for spectrum in supernova.get_spectra_in_range(
                        center_phase - phase_width, center_phase +
                        phase_width):
                    if self._check_spectrum(spectrum, verbosity):
                        all_raw_spec.append(spectrum)
                    else:
                        spectrum.usable = False
            else:
                # Get the single spectrum closest to maximum.
                while True:
                    spectrum = supernova.get_nearest_spectrum(
                        center_phase, phase_width
                    )

                    if spectrum is None:
                        break
                    elif self._check_spectrum(spectrum, verbosity):
                        # Found one! Keep it and go to the next target.
                        all_raw_spec.append(spectrum)
                        break
                    else:
                        spectrum.usable = False
                        continue

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

        # Pull out variables that we use all the time.
        self.salt_x1 = self.read_meta('salt2.X1')
        self.salt_color = self.read_meta('salt2.Color')
        self.salt_phases = self.read_meta('salt2.phase')
        self.redshifts = self.read_meta('host.zcmb')
        self.redshift_errs = self.read_meta('host.zhelio.err')

        # Record which supernovae should be in the validation set.
        self.train_cut = np.array([i.target['idr.subset'] != 'validation' for i
                                   in all_spec])

    def _check_spectrum(self, spectrum, verbosity=1):
        """Check if a spectrum is valid or not"""
        spectrum.do_lazyload()

        timeon = np.min([spectrum['fits.timeon'],
                         spectrum['fits.r.timeon']])

        s2n_start = spectrum.get_signal_to_noise(3300, 3800)
        s2n_end = spectrum.get_signal_to_noise(8100, 8600)

        ext_row = self.ext_tab[self.ext_tab['night'] ==
                               spectrum['obs.exp'][:6]]
        std_weight = ext_row['std_weight']

        redshift = spectrum.target['host.zcmb']

        if timeon < 30000:
            # Time on cut. Weird things happen when the detector was
            # recently turned on, including red wings. The noise in low
            # timeon images is very high and systematic, so we throw
            # them out.
            print_verbose("Cutting %s, timeon %ds too short." %
                          (spectrum, timeon), verbosity, 2)
            return False
        # elif s2n_u < -150:
        elif s2n_start < 100:
            # Signal-to-noise cut. We find that a signal-to-noise of
            # < ~100 in the U-band leads to an added core dispersion of
            # >0.1 mag in the U-band. This is unacceptable for the
            # twins analysis that relies on getting the color right for
            # a single spectrum.
            # print_verbose("Cutting %s, U-band signal-to-noise %.2f "
                          # "too low." % (spectrum, s2n_u), verbosity, 2)
            print_verbose("Cutting %s, start signal-to-noise %.2f "
                          "too low." % (spectrum, s2n_start),
                          verbosity, 2)
            return False
        # elif s2n_end < 50:
            # # Similarly for the red channel, low signal-to-noises at
            # # the edge of the spectra lead to increased dispersion and
            # # outliers for red bands.
            # print_verbose("Cutting %s, end signal-to-noise %.2f "
                          # "too low." % (spectrum, s2n_end),
                          # verbosity, 2)
        elif std_weight < 0.2:
            # Some nights have poor standard star choices leading to
            # bad extinction solutions. The "standard star weight" is
            # defined as the sum of the observed standard star
            # airmasses minus the mean airmass. In the case of only 2
            # stars observed, this is the difference in airmass between
            # them. If this value is too low (chosen to be 0.2 from
            # SALT2 residual tests), then the extinction solution can't
            # be properly measured.
            print_verbose("Cutting %s, poor standard star "
                          "distribution w=%.2f for extinction solution"
                          % (spectrum, std_weight), verbosity, 2)
            return False
        # elif redshift > 0.1:
            # print_verbose("Cutting %s, redshift %.2f > 0.1" %
                          # (spectrum, redshift), verbosity, 2)

        # We made it!
        return True

    def read_meta(self, key):
        """Read a key from the meta data of each spectrum/target

        This will first attempt to read the key in the spectrum object's meta
        data. If it isn't there, then it will try to read from the target
        instead.
        """
        if key in self.spectra[0].meta:
            read_spectrum = True
        elif key in self.spectra[0].target.meta:
            read_spectrum = False
        else:
            raise KeyError("Couldn't find key %s in metadata." % key)

        res = []
        for spec in self.spectra:
            if read_spectrum:
                val = spec.meta[key]
            else:
                val = spec.target.meta[key]
            res.append(val)

        res = np.array(res)

        return res

    def read_between_the_lines(self, blinded=True):
        """Run the read between the lines algorithm.

        This algorithm estimates the brightnesses and colors of every spectrum
        and produces dereddened spectra.

        If blinded is True, then the brightnesses of any validation supernovae
        are thrown out.

        The fit is performed using Stan. We only use Stan as a minimizer here,
        although this model can also be used to produce the full Bayesian
        posterior of the fits.
        """
        # color_law = extinction.fm07(self.wave, 1.)
        color_law = extinction.fitzpatrick99(self.wave, 1., 2.8)

        N = len(self.flux)
        W = len(self.wave)

        def stan_init():
            start_mean_spectrum = np.mean(self.flux, axis=0)
            start_mean_spectrum /= np.sum(start_mean_spectrum)

            start_scales = np.mean(self.flux / start_mean_spectrum, axis=1)
            start_mags = -2.5*np.log10(start_scales)

            start_dispersion = 0.1 * start_mean_spectrum

            start_colors = np.zeros(N)

            return {
                'mean_spectrum': start_mean_spectrum,
                'mags': start_mags,
                'dispersion': start_dispersion,
                'colors': start_colors,
                'phase_slope': np.zeros(W),
                'phase_square': np.zeros(W),
            }

        stan_data = {
            'N': N,
            'W': W,
            'f': self.flux,
            'color_law': color_law,
            'phases': [i.phase for i in self.spectra],
        }

        model = load_stan_code('./rbtl_phase.stan')
        res = model.optimizing(data=stan_data, init=stan_init)

        self.stan_result = res

        self.raw_colors = res['colors']
        self.model_spectra = res['f_max']
        self.dispersion = res['dispersion']

        # Scale the mean spectrum so that its flux values are O(10). This isn't
        # strictly necessary, but it makes the distances come out to O(1)
        # numbers.
        raw_mean_spec = res['mean_spectrum']
        self.mean_spectrum = 10 * raw_mean_spec / np.mean(raw_mean_spec)

        raw_mags = res['mags']
        if blinded:
            # Immediately discard validation magnitudes so that we can't
            # accidentally look at them.
            raw_mags = raw_mags[self.train_cut]
        self.raw_mags = raw_mags

        # Zeropoint mags and colors.
        self.mags = self.raw_mags - np.median(self.raw_mags)
        self.colors = self.raw_colors - np.median(self.raw_colors)

        # Deredden the real spectra and set them to the same scale as the mean
        # spectrum.
        self.applied_scale = self.mean_spectrum / res['f_scale']
        self.scale_flux = self.flux * self.applied_scale
        self.scale_fluxerr = self.fluxerr * self.applied_scale

        # Setup a cut to select targets that should have reasonable
        # dispersions.
        self.mag_cut = (
            (self.redshift_errs < 0.004) &
            (self.redshifts > 0.02) &
            (self.colors < 0.5)
        )

    def do_embedding(self, n_neighbors=4, n_components=2):
        self.iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        self.trans = self.iso.fit_transform(self.scale_flux /
                                            self.mean_spectrum)

    def get_indicators(self):
        """Calculate spectral indicators for all of the features"""
        all_indicators = []

        for idx in range(len(self.flux)):
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

    def apply_polynomial_standardization(self, degree=1):
        """Apply polynomial standardization to the dataset.

        Note that this standardization uses the target residuals directly, so
        it is somewhat biased.

        The transformation can only be up to dimension 3 for now because that
        is the maximum dimension supported by numpy polyval.
        """
        opt_min_func = np.std

        # Figure out how many parameters we need for the fit. Note that there
        # is an additional linear term for color.
        dimension = self.trans.shape[-1]
        num_parameters = 1 + (1+degree)**dimension
        if dimension not in [1, 2, 3]:
            message = ("Dimension %d not supported for polynomial "
                       "standardization!" % dimension)
            raise ManifoldTwinsException(message)

        def extract_coefficients(x):
            color_slope = x[0]
            poly_coefficients = np.reshape(x[1:], (degree+1,)*dimension)
            return color_slope, poly_coefficients

        def apply_corr(x, cut=None):
            if cut is not None:
                cut_mags = self.mags[cut[self.train_cut]]
                cut_trans = self.trans[cut & self.train_cut]
                cut_colors = self.colors[cut & self.train_cut]
            else:
                cut_mags = self.mags
                cut_trans = self.trans[self.train_cut]
                cut_colors = self.colors[self.train_cut]

            color_slope, poly_coefficients = extract_coefficients(x)

            # Linear color correction
            model = color_slope * cut_colors

            # Correction based on embedding
            t = cut_trans
            poly = np.polynomial.polynomial
            if dimension == 1:
                model += poly.polyval(t[:, 0], poly_coefficients)
            elif dimension == 2:
                model += poly.polyval2d(t[:, 0], t[:, 1], poly_coefficients)
            elif dimension == 3:
                model += poly.polyval3d(t[:, 0], t[:, 1], t[:, 2],
                                        poly_coefficients)

            diff = cut_mags - model

            return diff

        def to_min(x):
            corr_residuals = apply_corr(x, self.mag_cut)
            return (
                # Target function
                opt_min_func(corr_residuals) +

                # Lagrange multiplier to keep mean 0.
                np.mean(corr_residuals)**2
            )

        res = minimize(to_min, np.zeros(num_parameters))
        color_slope, poly_coefficients = extract_coefficients(res.x)
        print("Fitted color slope:", color_slope)
        print("Fitted coefficients:")
        print(poly_coefficients)

        self.corr_mags = apply_corr(res.x)
        cut_corr_mags = self.corr_mags[self.mag_cut[self.train_cut]]

        print("Fit NMAD:       ", math.nmad(cut_corr_mags))
        print("Fit std:        ", np.std(cut_corr_mags))

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
            kernels.ExpSquaredKernel([hyperparameters[2]**2]*len(use_dim),
                                     ndim=ndim, axes=use_dim)
        )

        if phase:
            # Additional kernel in phase direction.
            kernel += (
                hyperparameters[3]**2 *
                kernels.ExpSquaredKernel([hyperparameters[4]**2],
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
        if full:
            use_trans = self.trans[self.train_cut]
            if kind == 'twins':
                use_mags = self.mags
            elif kind == 'salt2':
                use_mags = self.salt_hr[self.train_cut]
        else:
            use_trans = self.trans[self.mag_cut & self.train_cut]
            if kind == 'twins':
                use_mags = self.mags[self.mag_cut[self.train_cut]]
            elif kind == 'salt2':
                use_mags = self.salt_hr[self.mag_cut & self.train_cut]

        return use_trans, use_mags

    def _calculate_gp_residuals(self, hyperparameters=None, kind='twins',
                                **kwargs):
        """Calculate the GP prediction residuals for a set of
        hyperparameters
        """
        use_trans, use_mags = self._get_gp_mags(kind)

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

        use_trans, use_mags = self._get_gp_mags(kind)

        def to_min(x):
            gp = self._build_gp(use_trans, x)
            return -gp.log_likelihood(use_mags)

        res = minimize(to_min, [0.1, 0.3, 5])
        print("Fit result:")
        print(res)
        self.gp_hyperparameters = res.x

        full_trans, full_mags = self._get_gp_mags(kind, full=True)

        preds = self._predict_gp_oos(
            full_trans, full_mags,
            condition_mask=self.mag_cut[self.train_cut]
        )

        self.corr_mags = full_mags - preds
        cut_corr_mags = self.corr_mags[self.mag_cut[self.train_cut]]

        print("Fit NMAD:       ", math.nmad(cut_corr_mags))
        print("Fit std:        ", np.std(cut_corr_mags))

    def fit_gp_phase(self, verbose=True, kind='twins'):
        """Fit a Gaussian Process to predict magnitudes for the data."""
        print("Fitting GP hyperparameters...")

        use_trans, use_mags = self._get_gp_mags(kind)

        use_phases = self.salt_phases[self.mag_cut & self.train_cut]

        full_trans = np.vstack([use_trans.T, use_phases]).T

        def to_min(x):
            gp = self._build_gp(full_trans, x, phase=True)
            return -gp.log_likelihood(use_mags)

        res = minimize(to_min, [0.1, 0.3, 5, 0.3, 2])
        print("Fit result:")
        print(res)
        self.gp_hyperparameters = res.x

        full_trans, full_mags = self._get_gp_mags(kind, full=True)

        preds = self._predict_gp_oos(
            full_trans, full_mags,
            condition_mask=self.mag_cut[self.train_cut]
        )

        self.corr_mags = full_mags - preds
        cut_corr_mags = self.corr_mags[self.mag_cut[self.train_cut]]

        print("Fit NMAD:       ", math.nmad(cut_corr_mags))
        print("Fit std:        ", np.std(cut_corr_mags))

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

        full_trans, full_mags = self._get_gp_mags(kind, full=True)

        preds = self._predict_gp_oos(
            full_trans, full_mags,
            condition_mask=self.mag_cut[self.train_cut], phase=phase
        )

        self.corr_mags = full_mags - preds
        cut_corr_mags = self.corr_mags[self.mag_cut[self.train_cut]]

        print("Fit NMAD:       ", math.nmad(cut_corr_mags))
        print("Fit std:        ", np.std(cut_corr_mags))

    def predict_gp(self, x, hyperparameters=None, **kwargs):
        """Do the GP prediction at specific points using the full GP
        conditioning.

        Note: this function uses all of the training data to make predictions.
        Use _predict_gp_oos or something similar to properly do out of sample
        predictions if you want to predict on the training data.
        """
        use_trans = self.trans[self.mag_cut & self.train_cut]
        use_mags = self.mags[self.mag_cut[self.train_cut]]

        preds = self._predict_gp(x, use_trans, use_mags, hyperparameters,
                                 **kwargs)

        return preds

    def plot_gp(self, axis_1=0, axis_2=1, hyperparameters=None, num_points=50,
                border=0.5):
        """Plot the GP predictions with data overlayed."""
        x = self.trans[:, axis_1]
        y = self.trans[:, axis_2]

        min_x = np.min(x) - border
        max_x = np.max(x) + border
        min_y = np.min(y) - border
        max_y = np.max(y) + border

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
        plt.scatter(x[self.train_cut], y[self.train_cut], c=self.mags,
                    edgecolors='k', vmin=-0.2, vmax=0.2, s=10 + 50 *
                    self.mag_cut[self.train_cut], cmap=plt.cm.coolwarm)

    def load_host_data(self):
        """Load host data from Rigault et al. 2019"""
        host_data = Table.read(
            '../data/snfactory/aux/host_properties_rigault_2019.txt',
            format='ascii'
        )
        all_host_idx = []
        host_mask = []
        for spectrum in self.spectra:
            name = spectrum.target.name
            match = host_data['name'] == name

            # Check if found
            if not np.any(match):
                host_mask.append(False)
                continue

            # row = host_data[match][0]
            all_host_idx.append(np.where(match)[0])
            host_mask.append(True)

        # Save the loaded data
        self.host_data = host_data[all_host_idx]
        self.host_mask = np.array(host_mask)

    def plot_host_variable(self, variable, cut=None, mag_type='twins',
                           match_cuts=False):
        """Plot diagnostics for some host variable.

        Valid variable names are the keys in the host_data Table that comes
        from load_host_data.

        mag_type selects which magnitudes to plot. The options are:
        - twins: use the manifold twins magnitudes (default)
        - salt: use the SALT2 corrected Hubble residuals

        If match_cuts is True, then the cuts required for both the twins
        manifold and SALT2 are applied (leaving a smaller dataset).
        """
        if cut is None:
            cut = np.ones(len(self.spectra), dtype=bool)

        if mag_type == 'twins':
            if match_cuts:
                mag_cut = (self.salt_cut & self.host_mask)[self.train_cut]
                valid_cut = self.train_cut & self.salt_cut
            else:
                mag_cut = self.host_mask[self.train_cut]
                valid_cut = self.train_cut
            host_corr_mags = self.corr_mags[mag_cut]
        elif mag_type == 'salt':
            if match_cuts:
                mag_cut = self.salt_cut & self.host_mask & self.train_cut
                valid_cut = self.salt_cut & self.train_cut
            else:
                mag_cut = self.salt_cut & self.host_mask
                valid_cut = self.salt_cut
            host_corr_mags = self.salt_hr[mag_cut]

        # Find a threshold for a split
        threshold = np.nanmedian(self.host_data[variable])

        host_values = np.squeeze(self.host_data[variable])
        host_cut_1 = host_values < threshold
        host_cut_2 = host_values > threshold
        host_mags_1 = host_corr_mags[host_cut_1[valid_cut[self.host_mask]]
                                     & cut[valid_cut & self.host_mask]]
        host_mags_2 = host_corr_mags[host_cut_2[valid_cut[self.host_mask]]
                                     & cut[valid_cut & self.host_mask]]

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
            self.trans[self.host_mask, 0],
            self.trans[self.host_mask, 1],
            c=host_values,
            cmap=plt.cm.coolwarm,
            vmin=threshold - 0.1,
            vmax=threshold + 0.1
        )
        plt.xlabel('Isomap parameter 0')
        plt.ylabel('Isomap parameter 1')

        plt.subplot(2, 2, 2)
        plt.scatter(
            host_values[valid_cut[self.host_mask]],
            host_corr_mags,
            s=2 + 30 * cut[valid_cut & self.host_mask],
            cmap=plt.cm.coolwarm
        )
        plt.xlabel(variable)
        plt.ylabel('Corrected mags')

        plt.subplot(2, 2, 3)
        plt.scatter(
            host_values,
            self.trans[self.host_mask, 1],
            c=self.trans[self.host_mask, 0]
        )
        plt.xlabel(variable)
        plt.ylabel('Isomap parameter 1')

        plt.subplot(2, 2, 4)
        plt.scatter(
            host_values,
            self.trans[self.host_mask, 0],
            c=self.trans[self.host_mask, 1]
        )
        plt.xlabel(variable)
        plt.ylabel('Isomap parameter 0')

        plt.tight_layout()

    def plot_host(self, cut=None):
        """Make an interactive plot of host properties"""
        from ipywidgets import interact, fixed

        if cut is None:
            cut = self.mag_cut

        interact(self.plot_host_variable, variable=self.host_data.keys()[1:],
                 cut=fixed(cut), mag_type=['twins', 'salt'])

    def plot_distances(self):
        """Plot the reconstructed distances from the embedding against the true
        distances
        """
        from scipy.spatial.distance import pdist

        use_spec = self.scale_flux / self.mean_spectrum
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

        use_spec = self.scale_flux / self.mean_spectrum
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

        scale_spec = (self.scale_flux / self.mean_spectrum)
        use_spec = scale_spec[self.mag_cut & self.train_cut]
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

    def calculate_salt_hubble_residuals(self):
        """Calculate SALT hubble residuals"""
        from astropy.cosmology import Planck15
        # For SALT, can only use SNe that are in the good sample
        self.salt_cut = np.array(
            [i.target['idr.subset'] in ['training', 'validation'] for i in
             self.spectra]
        )

        # We also require reasonable redshifts and colors for the determination
        # of standardization parameters. The mag_cut previously defined works
        # for this.
        fit_cut = self.salt_cut & self.mag_cut

        # Determine standardization parameters
        fit_redshift = self.redshifts[fit_cut]
        fit_color = self.salt_color[fit_cut]
        fit_x1 = self.salt_x1[fit_cut]

        all_salt_mb = np.array(
            [i.target.meta['salt2.RestFrameMag_0_B'] for i in self.spectra]
        )
        fit_salt_mb = all_salt_mb[fit_cut]

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
        print("    std:  ", np.std(self.salt_hr[fit_cut]))
        print("    NMAD: ", math.nmad(self.salt_hr[fit_cut]))
