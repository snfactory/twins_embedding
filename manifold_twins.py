from astropy.table import Table
from hashlib import md5
from idrtools import Dataset, math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn.manifold import Isomap
import extinction
import numpy as np
import os
import sys
import tqdm

from settings import default_settings
from manifold_gp import ManifoldGaussianProcess
import utils
import specind


class ManifoldTwinsException(Exception):
    pass


class ManifoldTwinsAnalysis:
    def __init__(self, **kwargs):
        """Load the dataset and setup the analysis"""

        # Update the default settings with any arguments that came in from kwargs.
        self.settings = dict(default_settings, **kwargs)

        # Set the default matplotlib figure size from the settings.
        import matplotlib as mpl
        for key, value in self.settings['matplotlib_settings'].items():
            mpl.rcParams[key] = value

    def run_analysis(self):
        """Run the full analysis"""
        self.load_dataset()

        self.print_verbose("Estimating the spectra at maximum light...")
        self.model_differential_evolution()

        self.print_verbose("Reading between the lines...")
        self.read_between_the_lines()

        self.print_verbose("Building masks...")
        self.build_masks()

        self.print_verbose("Generating the manifold learning embedding...")
        self.embedding = self.generate_embedding()

        self.print_verbose("Calculating spectral indicators...")
        self.calculate_spectral_indicators()

        self.print_verbose("Fitting RBTL GP to magnitude residuals...")
        self.residuals_rbtl_gp = self.fit_gp_magnitude_residuals()

        self.print_verbose("Calculating SALT2 magnitude residuals...")
        self.residuals_salt = self.fit_salt_magnitude_residuals()

        self.print_verbose("Loading host galaxy data...")
        self.load_host_data()

        self.print_verbose("Loading peculiar SNe Ia data...")
        self.load_peculiar_data()

        self.print_verbose("Done!")

    def load_dataset(self):
        """Load the dataset"""
        self.print_verbose("Loading dataset...")
        self.print_verbose("    IDR:          %s" % self.settings['idr'])
        self.print_verbose(
            "    Phase range: [%.1f, %.1f] days"
            % (-self.settings['phase_range'], self.settings['phase_range'])
        )
        self.print_verbose("    Bin velocity: %.1f" % self.settings['bin_velocity'])

        self.dataset = Dataset.from_idr(
            os.path.join(self.settings['idr_directory'], self.settings['idr']),
            load_both_headers=True
        )

        # Do/load all of the SALT2 fits for this dataset
        self.dataset.load_salt_fits()

        all_raw_spec = []
        center_mask = []

        self.attrition_enough_spectra = 0
        self.attrition_salt_daymax = 0
        self.attrition_range = 0
        self.attrition_usable = 0

        for supernova in tqdm.tqdm(self.dataset.targets):
            if len(supernova.spectra) < 5:
                self.print_verbose(
                    "Cutting %s, not enough spectra to guarantee a "
                    "good LC fit" % supernova,
                    minimum_verbosity=2,
                )
                continue
            self.attrition_enough_spectra += 1

            daymax_err = supernova.salt_fit['t0_err']
            if daymax_err > 1.0:
                self.print_verbose(
                    "Cutting %s, day max err %.2f too high" % (supernova, daymax_err),
                    minimum_verbosity=2,
                )
                continue
            self.attrition_salt_daymax += 1

            range_spectra = supernova.get_spectra_in_range(
                -self.settings['phase_range'], self.settings['phase_range']
            )
            if len(range_spectra) > 0:
                self.attrition_range += 1

            used_phases = []
            for spectrum in range_spectra:
                if self._check_spectrum(spectrum):
                    all_raw_spec.append(spectrum)
                    used_phases.append(spectrum.phase)
                else:
                    spectrum.usable = False

            used_phases = np.array(used_phases)
            if len(used_phases) > 0:
                # Figure out which spectrum was closest to the center of the
                # bin.
                self.attrition_usable += 1
                target_center_mask = np.zeros(len(used_phases), dtype=bool)
                target_center_mask[np.argmin(np.abs(used_phases))] = True
                center_mask.extend(target_center_mask)

        all_flux = []
        all_fluxerr = []
        all_spec = []

        for spectrum in all_raw_spec:
            bin_spec = spectrum.bin_by_velocity(
                self.settings['bin_velocity'],
                self.settings['bin_min_wavelength'],
                self.settings['bin_max_wavelength'],
            )
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
        self.helio_redshifts = self.read_meta("host.zhelio")
        self.redshifts = self.read_meta("host.zcmb")
        self.redshift_errs = self.read_meta("host.zhelio.err")

        # Build a list of targets and a map from spectra to targets.
        self.targets = np.unique([i.target for i in self.spectra])
        self.target_map = np.array(
            [self.targets.tolist().index(i.target) for i in self.spectra]
        )

        # Pull out SALT fit info
        self.salt_fits = Table([i.salt_fit for i in self.targets])
        self.salt_x1 = self.salt_fits['x1'].data
        self.salt_colors = self.salt_fits['c'].data
        self.salt_phases = np.array([i.phase for i in self.spectra])
        self.salt_mask = np.array([i.has_valid_salt_fit() for i in self.targets])

        # Record which targets should be in the validation set.
        self.train_mask = np.array(
            [i["idr.subset"] != "validation" for i in self.targets]
        )

        # Build a hash that is unique to the dataset that we are working on.
        hash_info = (
            self.settings['idr']
            + ';' + str(self.settings['phase_range'])
            + ';' + str(self.settings['bin_velocity'])
            + ';' + str(self.settings['bin_min_wavelength'])
            + ';' + str(self.settings['bin_max_wavelength'])
            + ';' + str(self.settings['s2n_cut_min_wavelength'])
            + ';' + str(self.settings['s2n_cut_max_wavelength'])
            + ';' + str(self.settings['s2n_cut_threshold'])
        )
        self.dataset_hash = md5(hash_info.encode("ascii")).hexdigest()

        # Load a dictionary that maps IDR names into IAU ones.
        iau_data = np.genfromtxt('./data/iau_name_map.txt', dtype=str)
        self.iau_name_map = {i:j for i, j in iau_data}

    def _check_spectrum(self, spectrum):
        """Check if a spectrum is valid or not"""
        spectrum.do_lazyload()

        s2n_start = spectrum.get_signal_to_noise(
            self.settings['s2n_cut_min_wavelength'],
            self.settings['s2n_cut_max_wavelength'],
        )

        if s2n_start < self.settings['s2n_cut_threshold']:
            # Signal-to-noise cut. We find that a signal-to-noise of < ~100 in the
            # U-band leads to an added core dispersion of >0.1 mag in the U-band which
            # is much higher than expected from statistics. This is unacceptable for the
            # twins analysis that relies on getting the color right for a single
            # spectrum.
            self.print_verbose(
                "Cutting %s, start signal-to-noise %.2f "
                "too low." % (spectrum, s2n_start),
                minimum_verbosity=2,
            )
            return False

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

    def print_verbose(self, *args, minimum_verbosity=1):
        if self.settings['verbosity'] >= minimum_verbosity:
            print(*args)

    def model_differential_evolution(self, use_cache=True):
        """Estimate the spectra for each of our SNe Ia at maximum light.

        This algorithm uses all targets with multiple spectra to model the differential
        evolution of Type Ia supernovae near maximum light. This method does not rely on
        knowing the underlying model of Type Ia supernovae and only models the
        differences. The model is generated in magnitude space, so anything static in
        between us and the supernova, like dust, does not affect the model.

        The fit is performed using Stan. We only use Stan as a minimizer here,
        and we do some analytic tricks inside to speed up the computation. Don't try to
        run this in sampling model, the analytic tricks will mess up the uncertainties
        of a Bayesian analysis!

        If use_cache is True, then the fitted model will be retrieved from a
        cache if it exists. Make sure to run with use_cache=False if making
        modifications to the model!

        If use_x1 is True, a SALT2 x1-dependent term will be included in the
        model.
        """
        # Load the stan model
        model_path = "./stan_models/phase_interpolation_analytic.stan"
        model_hash, model = utils.load_stan_model(
            model_path,
            verbosity=self.settings['verbosity']
        )

        # Build a hash that is unique to this dataset/analysis
        hash_info = (
            self.dataset_hash
            + ';' + model_hash
            + ';' + str(self.settings['differential_evolution_num_phase_coefficients'])
            + ';' + str(self.settings['differential_evolution_use_salt_x1'])
        )
        self.differential_evolution_hash = md5(hash_info.encode("ascii")).hexdigest()

        # If we ran this model before, read the cached result if we can.
        if use_cache:
            cache_result = utils.load_stan_result(self.differential_evolution_hash)
            if cache_result is not None:
                # Found the cached result. Load it and don't redo the fit.
                self.differential_evolution_result = cache_result
                self.maximum_flux = cache_result["maximum_flux"]
                self.maximum_fluxerr = cache_result["maximum_fluxerr"]
                return

        num_targets = len(self.targets)
        num_spectra = len(self.flux)
        num_wave = len(self.wave)
        num_phase_coefficients = self.settings[
            'differential_evolution_num_phase_coefficients'
        ]

        if num_phase_coefficients % 2 != 0:
            raise Exception("ERROR: Must have an even number of phase " "coefficients.")

        spectra_targets = [i.target for i in self.spectra]
        spectra_target_counts = np.array(
            [spectra_targets.count(i.target) for i in self.spectra]
        )

        phase_coefficients = np.zeros((num_spectra, num_phase_coefficients))

        for i, phase in enumerate(self.salt_phases):
            phase_scale = np.abs(
                (num_phase_coefficients / 2) * (phase / self.settings['phase_range'])
            )

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
                "phase_slope": np.zeros(num_wave),
                "phase_quadratic": np.zeros(num_wave),
                "phase_slope_x1": np.zeros(num_wave),
                "phase_quadratic_x1": np.zeros(num_wave),
                "phase_dispersion_coefficients": (
                    0.01 * np.ones((num_phase_coefficients, num_wave))
                ),
                "gray_offsets": np.zeros(num_spectra),
                "gray_dispersion_scale": 0.02,
            }

            return init_params

        if self.settings['differential_evolution_use_salt_x1']:
            x1 = self.salt_x1
        else:
            x1 = np.zeros(num_targets)

        stan_data = {
            "num_targets": num_targets,
            "num_spectra": num_spectra,
            "num_wave": num_wave,
            "measured_flux": self.flux,
            "measured_fluxerr": self.fluxerr,
            "phases": [i.phase for i in self.spectra],
            "phase_coefficients": phase_coefficients,
            "num_phase_coefficients": num_phase_coefficients,
            "spectra_target_counts": spectra_target_counts,
            "target_map": self.target_map + 1,  # stan uses 1-based indexing
            "maximum_map": np.where(self.center_mask)[0] + 1,
            "salt_x1": x1,
        }

        sys.stdout.flush()
        result = model.optimizing(
            data=stan_data, init=stan_init, verbose=True, iter=20000, history_size=100
        )

        self.differential_evolution_result = result
        self.maximum_flux = result["maximum_flux"]
        self.maximum_fluxerr = result["maximum_fluxerr"]

        # Save the output to cache it for future runs.
        utils.save_stan_result(self.differential_evolution_hash, result)

    def read_between_the_lines(self, use_cache=True):
        """Run the read between the lines algorithm.

        This algorithm estimates the brightnesses and colors of every spectrum
        and produces dereddened spectra.

        The fit is performed using Stan. We only use Stan as a minimizer here.
        """
        # Load the fiducial color law.
        self.rbtl_color_law = extinction.fitzpatrick99(
            self.wave, 1.0, self.settings['rbtl_fiducial_rv']
        )

        # Load the stan model
        model_path = "./stan_models/read_between_the_lines.stan"
        model_hash, model = utils.load_stan_model(
            model_path,
            verbosity=self.settings['verbosity']
        )

        # Build a hash that is unique to this dataset/analysis
        hash_info = (
            self.differential_evolution_hash
            + ';' + model_hash
            + ';' + str(self.settings['rbtl_fiducial_rv'])
        )
        self.rbtl_hash = md5(hash_info.encode("ascii")).hexdigest()

        # If we ran this model before, read the cached result if we can.
        if use_cache:
            cache_result = utils.load_stan_result(self.rbtl_hash)
            if cache_result is not None:
                # Found the cached result. Load it and don't redo the fit.
                self._parse_rbtl_result(cache_result)
                return

        use_targets = self.targets

        num_targets = len(use_targets)
        num_wave = len(self.wave)

        def stan_init():
            # Use the spectrum closest to maximum as a first guess of the
            # target's spectrum.
            start_mean_flux = np.mean(self.maximum_flux, axis=0)
            start_fractional_dispersion = 0.1 * np.ones(num_wave)

            return {
                "mean_flux": start_mean_flux,
                "fractional_dispersion": start_fractional_dispersion,
                "colors_raw": np.zeros(num_targets - 1),
                "magnitudes_raw": np.zeros(num_targets - 1),
            }

        stan_data = {
            "num_targets": num_targets,
            "num_wave": num_wave,
            "maximum_flux": self.maximum_flux,
            "maximum_fluxerr": self.maximum_fluxerr,
            "color_law": self.rbtl_color_law,
        }

        sys.stdout.flush()
        result = model.optimizing(data=stan_data, init=stan_init, verbose=True,
                                  iter=5000)

        # Save the output to cache it for future runs.
        utils.save_stan_result(self.rbtl_hash, result)

        # Parse the result
        self._parse_rbtl_result(result)

    def _parse_rbtl_result(self, result):
        """Parse and save the result of a run of the RBTL analysis"""
        self.rbtl_result = result

        self.rbtl_colors = result["colors"]
        self.rbtl_mags = result["magnitudes"]
        self.mean_flux = result["mean_flux"]

        if self.settings['blinded']:
            # Immediately discard validation magnitudes so that we can't
            # accidentally look at them.
            self.rbtl_mags[~self.train_mask] = np.nan

        # Deredden the real spectra and set them to the same scale as the mean
        # spectrum.
        self.scale_flux = self.maximum_flux / result['model_scales']
        self.scale_fluxerr = self.maximum_fluxerr / result['model_scales']

        # Calculate fractional differences from the mean spectrum.
        self.fractional_differences = self.scale_flux / self.mean_flux - 1

    def build_masks(self):
        """Build masks that are used in the various manifold learning and magnitude
        analyses
        """
        # For the manifold learning analysis, we need to make sure that the spectra at
        # maximum light have reasonable uncertainties on their spectra at maximum light.
        # We define "reasonable" by comparing the variance of each spectrum to the
        # size of the intrinsic supernova variation measured in the RBTL analysis.
        intrinsic_dispersion = utils.frac_to_mag(
            self.rbtl_result["fractional_dispersion"]
        )
        intrinsic_power = np.sum(intrinsic_dispersion**2)
        maximum_uncertainty = utils.frac_to_mag(
            self.maximum_fluxerr / self.maximum_flux
        )
        maximum_power = np.sum(maximum_uncertainty**2, axis=1)
        self.maximum_uncertainty_fraction = maximum_power / intrinsic_power
        self.uncertainty_mask = (
            self.maximum_uncertainty_fraction <
            self.settings['mask_uncertainty_fraction']
        )
        self.print_verbose(
            "    Masking %d/%d targets whose uncertainty power is \n"
            "    more than %.3f of the intrinsic power."
            % (np.sum(~self.uncertainty_mask), len(self.uncertainty_mask),
               self.settings['mask_uncertainty_fraction'])
        )

        # Mask to select targets that have a magnitude that is expected to have a large
        # dispersion in brightness.
        with np.errstate(invalid="ignore"):
            self.redshift_color_mask = (
                (self.redshift_errs < 0.004)
                & (self.helio_redshifts > 0.02)
                & (self.rbtl_colors - np.median(self.rbtl_colors) < 0.5)
            )

    def generate_embedding(self, num_neighbors=None, num_components=-1):
        """Generate a manifold learning embedding."""
        if num_neighbors is None:
            num_neighbors = self.settings['isomap_num_neighbors']

        if num_components == -1:
            num_components = self.settings['isomap_num_components']

        isomap = Isomap(n_neighbors=num_neighbors, n_components=num_components)

        good_mask = self.uncertainty_mask

        # Build the embedding using well-measured targets
        ref_embedding = isomap.fit_transform(self.fractional_differences[good_mask])

        # Evaluate the coordinates in the embedding for the remaining targets.
        other_embedding = isomap.transform(self.fractional_differences[~good_mask])

        # Combine everything into a single array.
        embedding = np.zeros((len(self.targets), ref_embedding.shape[1]))
        embedding[good_mask] = ref_embedding
        embedding[~good_mask] = other_embedding

        return embedding

    def calculate_spectral_indicators(self):
        """Calculate spectral indicators for all of the features"""
        all_indicators = []

        for idx in range(len(self.scale_flux)):
            spec = specind.Spectrum(
                self.wave, self.scale_flux[idx], self.scale_fluxerr[idx]**2
            )
            indicators = spec.get_spin_dict()
            all_indicators.append(indicators)

        all_indicators = Table(all_indicators)

        self.spectral_indicators = all_indicators

    def calculate_peculiar_velocity_uncertainties(self, redshifts):
        """Calculate dispersion added to the magnitude due to host galaxy
        peculiar velocity
        """
        pec_vel_dispersion = (5 / np.log(10)) * (
            self.settings['peculiar_velocity'] / 3e5 / redshifts
        )

        return pec_vel_dispersion

    def _get_gp_data(self, kind="rbtl"):
        """Return the data needed for GP fits along with the corresponding masks.

        Parameters
        ----------
        kind : {'rbtl', 'salt', 'salt_raw'}
            The kind of magnitude data to return. The options are:
            - rbtl: RBTL magnitudes and colors.
            - salt: Corrected SALT2 magnitudes and colors.
            - salt_raw: Uncorrected SALT2 magnitudes and colors.

        Returns
        -------
        coordinates : numpy.array
            The coordinates to evaluate the GP over.
        mags : numpy.array
            A list of magnitudes for each supernova in the sample.
        mag_errs : numpy.array
            The uncertainties on the magnitudes. This only includes measurement
            uncertainties, not model ones (since the GP will handle that). Since we are
            dealing with high signal-to-noise light curves/spectra, the color and
            magnitude measurement errors are very small and difficult to propagate so I
            ignore them. This therefore only includes contributions from peculiar
            velocity.
        colors : numpy.array
            A list of colors for each supernova in the sample.
        condition_mask : numpy.array
            The mask that should be used for conditioning the GP.
        """
        if kind == "rbtl":
            mags = self.rbtl_mags
            colors = self.rbtl_colors
            condition_mask = self.uncertainty_mask & self.redshift_color_mask

            # Assume that we can ignore measurement uncertainties for the magnitude errors,
            # so the only contribution is from peculiar velocities.
            mag_errs = self.calculate_peculiar_velocity_uncertainties(self.redshifts)
        elif kind == "salt_raw" or kind == "salt":
            if kind == "salt_raw":
                # Evaluate the residuals with all model terms set to zero.
                mags, mag_errs = self._evaluate_salt_magnitude_residuals(
                    [], 0., 0., 0., 0.
                )
            elif kind == "salt":
                # Use the standard SALT2 fit as a baseline.
                mags = self.residuals_salt['residuals']
                mag_errs = self.residuals_salt['raw_residual_uncertainties']

            colors = self.salt_colors
            condition_mask = (
                self.salt_mask
                & self.uncertainty_mask
                & self.redshift_color_mask
            )
        else:
            raise ManifoldTwinsException("Unknown kind %s!" % kind)

        # Use the Isomap embedding for the GP coordinates.
        coordinates = self.embedding

        # If the analysis is blinded, only use the training data for conditioning.
        if self.settings['blinded']:
            condition_mask &= self.train_mask

        return coordinates, mags, mag_errs, colors, condition_mask

    def fit_gp_magnitude_residuals(self, kind="rbtl", mask=None,
                                   additional_covariates=[], verbosity=None):
        """Calculate magnitude residuals using a GP over a given latent space."""
        if verbosity is None:
            verbosity = self.settings['verbosity']

        # Fit the hyperparameters on the full conditioning sample.
        coordinates, mags, mag_errs, colors, raw_mask = self._get_gp_data(kind)

        # Build a list of linear covariates to use in the model that includes the color
        # and any user-specified covariates.
        covariates = [
            colors,
        ]

        if additional_covariates:
            covariates.append(additional_covariates)

        covariates = np.vstack(covariates)

        # Apply the user-specified mask if one was given.
        if mask is None:
            mask = raw_mask
        else:
            mask = mask & raw_mask

        manifold_gp = ManifoldGaussianProcess(
            self.embedding,
            mags,
            mag_errs,
            covariates,
            mask,
        )

        manifold_gp.fit(verbosity=verbosity)

        return manifold_gp

    def load_host_data(self):
        """Load host data from Rigault et al. 2019"""
        host_data = Table.read(
            # "./data/host_properties_rigault_2019.txt", format="ascii"
            # "./data/host_properties_rigault_full.csv"
            "./data/host_properties_rigault_valid.csv"
        )
        all_host_idx = []
        host_mask = []
        for target in self.targets:
            name = target.name

            # Note: not applicable if we are using the full list from private
            # communication.
            # Rigault et al. 2019 uses IAU names, so convert our names if appropriate.
            # name = self.iau_name_map.get(name, name)

            match = host_data["name"] == name

            # Check if found
            if not np.any(match):
                host_mask.append(False)
                continue

            # row = host_data[match][0]
            all_host_idx.append(np.where(match)[0][0])
            host_mask.append(True)

        # Save the loaded data
        self.host_mask = np.array(host_mask)
        fill_host_data = utils.fill_mask(host_data[all_host_idx].as_array(),
                                         self.host_mask)
        self.host_data = Table(fill_host_data, names=host_data.columns)

    def load_peculiar_data(self):
        """Load peculiar SNe Ia information from Lin. et al 2020"""
        peculiar_data = Table.read("./data/peculiar_lin_2020.csv", format="ascii.csv")

        all_peculiar_idx = []
        peculiar_mask = []
        for target in self.targets:
            name = target.name

            match = peculiar_data["name"] == name

            # Check if found
            if not np.any(match):
                peculiar_mask.append(True)
                continue

            all_peculiar_idx.append(np.where(match)[0][0])
            peculiar_mask.append(False)

        # Save the loaded data
        self.peculiar_mask = np.array(peculiar_mask)
        fill_peculiar_data = utils.fill_mask(peculiar_data[all_peculiar_idx].as_array(),
                                             ~self.peculiar_mask)
        self.peculiar_data = Table(fill_peculiar_data, names=peculiar_data.columns)

    def plot_host_variable(self, variable, mask=None, mag_type="rbtl",
                           match_masks=False, threshold=None):
        """Plot diagnostics for some host variable.

        Valid variable names are the keys in the host_data Table that comes
        from load_host_data.

        mag_type selects which magnitudes to plot. The options are:
        - rbtl: use the RBTL magnitudes (default)
        - salt: use the SALT2 corrected magnitude residuals

        If match_masks is True, then the masks required for both the Isomap
        manifold and SALT2 are applied (leaving a smaller dataset).
        """
        if mask is None:
            mask = np.ones(len(self.targets), dtype=bool)

        if match_masks:
            mag_mask = mask & self.good_salt_mask & self.good_mag_mask
        elif mag_type == "rbtl":
            mag_mask = mask & self.good_mag_mask
        elif mag_type == "salt":
            mag_mask = mask & self.good_salt_mask

        if mag_type == "rbtl":
            host_corr_mags = self.corr_mags
        elif mag_type == "salt":
            host_corr_mags = self.salt_hr

        host_mask = self.host_mask

        # Find a threshold for a split
        host_values = np.squeeze(self.host_data[variable])

        if threshold is None:
            threshold = np.nanmedian(host_values)

        with np.errstate(invalid="ignore"):
            host_cut_1 = host_mask & (host_values < threshold)
            host_cut_2 = host_mask & (host_values > threshold)

        host_mags_1 = host_corr_mags[host_cut_1 & mag_mask]
        host_mags_2 = host_corr_mags[host_cut_2 & mag_mask]

        mean_diff = np.nanmean(host_mags_1) - np.nanmean(host_mags_2)
        mean_diff_err = np.sqrt(
            np.var(host_mags_1) / len(host_mags_1)
            + np.var(host_mags_2) / len(host_mags_2)
        )

        print("Threshold:   %.3f" % threshold)
        print("Mean diff:   %.4f ± %.4f mag" % (mean_diff, mean_diff_err))
        print(
            "Median diff: %.4f mag"
            % (np.nanmedian(host_mags_1) - np.nanmedian(host_mags_2))
        )

        plt.figure(figsize=(8, 8))
        plt.title(variable)
        plt.subplot(2, 2, 1)
        plt.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            c=host_values,
            cmap=plt.cm.coolwarm,
            vmin=threshold - 0.1,
            vmax=threshold + 0.1,
        )
        plt.xlabel("Isomap parameter 0")
        plt.ylabel("Isomap parameter 1")

        plt.subplot(2, 2, 2)
        plt.scatter(
            host_values, host_corr_mags, s=2 + 30 * mag_mask, cmap=plt.cm.coolwarm
        )
        plt.xlabel(variable)
        plt.ylabel("Corrected mags")

        plt.subplot(2, 2, 3)
        plt.scatter(host_values, self.embedding[:, 1], c=self.embedding[:, 0])
        plt.xlabel(variable)
        plt.ylabel("Isomap parameter 1")

        plt.subplot(2, 2, 4)
        plt.scatter(host_values, self.embedding[:, 0], c=self.embedding[:, 1])
        plt.xlabel(variable)
        plt.ylabel("Isomap parameter 0")

        plt.tight_layout()

    def plot_host(self, threshold=None, mask=None):
        """Make an interactive plot of host properties"""
        from ipywidgets import interact, fixed

        interact(
            self.plot_host_variable,
            variable=self.host_data.keys()[1:],
            mask=fixed(mask),
            threshold=fixed(threshold),
            mag_type=["rbtl", "salt"],
        )

    def plot_distances(self):
        """Plot the reconstructed distances from the embedding against the true
        distances
        """
        from scipy.spatial.distance import pdist

        mask = self.uncertainty_mask

        spec_dists = pdist(self.fractional_differences[mask])
        embedding_dists = pdist(self.embedding[mask])

        plt.figure()
        plt.scatter(
            spec_dists,
            embedding_dists * np.median(spec_dists) / np.median(embedding_dists),
            s=1,
            alpha=0.1,
        )
        plt.xlabel("Twins distance")
        plt.ylabel("Scaled transformed distance")

    def plot_twin_distances(self, twins_percentile=10, figsize=None):
        """Plot a histogram of where twins show up in the transformed
        embedding.
        """
        from IPython.display import display
        from scipy.spatial.distance import pdist
        from scipy.stats import percentileofscore
        import pandas as pd

        mask = self.uncertainty_mask

        spec_dists = pdist(self.fractional_differences[mask])
        embedding_dists = pdist(self.embedding[mask])

        splits = {
            "Best 10% of twinness": (0, 10),
            "10-20%": (10, 20),
            "20-50%": (20, 50),
            "Worst 50% of twinness": (50, 100),
        }

        # Set weight so that the histogram is 1 if we have every element in
        # that bin.
        weight = 100 / len(embedding_dists)

        all_percentiles = []
        all_weights = []

        all_spec_cuts = []
        all_embedding_cuts = []

        for label, (min_percentile, max_percentile) in splits.items():
            spec_cut = (spec_dists >= np.percentile(spec_dists, min_percentile)) & (
                spec_dists < np.percentile(spec_dists, max_percentile)
            )
            embedding_cut = (embedding_dists >= np.percentile(embedding_dists, min_percentile)) & (
                embedding_dists < np.percentile(embedding_dists, max_percentile)
            )
            percentiles = []
            for dist in embedding_dists[spec_cut]:
                percentiles.append(percentileofscore(embedding_dists, dist))
            percentiles = np.array(percentiles)
            weights = np.ones(len(percentiles)) * weight

            all_percentiles.append(percentiles)
            all_weights.append(weights)
            all_spec_cuts.append(spec_cut)
            all_embedding_cuts.append(embedding_cut)

        plt.figure(figsize=figsize)
        plt.hist(
            all_percentiles,
            100,
            (0, 100),
            weights=all_weights,
            histtype="barstacked",
            label=splits.keys(),
        )
        plt.xlabel("Recovered twinness percentile in the embedded space")
        plt.ylabel("Fraction in bin")
        plt.legend()

        plt.xlim(0, 100)
        plt.ylim(0, 1)

        for label, (min_percentile, max_percentile) in splits.items():
            plt.axvline(max_percentile, c="k", lw=2, ls="--")

        # Build leakage matrix.
        leakage_matrix = np.zeros((len(splits), len(splits)))
        for idx_1, label_1 in enumerate(splits.keys()):
            for idx_2, label_2 in enumerate(splits.keys()):
                spec_cut = all_spec_cuts[idx_1]
                embedding_cut = all_embedding_cuts[idx_2]
                leakage = np.sum(embedding_cut & spec_cut) / np.sum(spec_cut)
                leakage_matrix[idx_1, idx_2] = leakage

        # Print the leakage matrix using pandas
        df = pd.DataFrame(
            leakage_matrix,
            index=["From %s" % i for i in splits.keys()],
            columns=["To %s" % i for i in splits.keys()],
        )
        display(df)

        return leakage_matrix

    def plot_twin_pairings(self, mask=None, show_nmad=False):
        """Plot the twins delta M as a function of twinness ala Fakhouri"""
        from scipy.spatial.distance import pdist
        from scipy.stats import percentileofscore

        if mask is None:
            mask = self.good_mag_mask

        use_spec = self.fractional_differences[mask]
        use_mag = self.rbtl_mags[mask]

        use_mag -= np.mean(use_mag)

        spec_dists = pdist(use_spec)
        delta_mags = pdist(use_mag[:, None])

        percentile = np.array([percentileofscore(spec_dists, i) for i in spec_dists])

        mags_20 = delta_mags[percentile < 20]
        self.twins_rms = math.rms(mags_20) / np.sqrt(2)
        self.twins_nmad = math.nmad_centered(mags_20) / np.sqrt(2)

        print("RMS  20%:", self.twins_rms)
        print("NMAD 20%:", self.twins_nmad)

        plt.figure()
        math.plot_binned_rms(
            percentile,
            delta_mags / np.sqrt(2),
            bins=20,
            label="RMS",
            equal_bin_counts=True,
        )
        if show_nmad:
            math.plot_binned_nmad_centered(
                percentile,
                delta_mags / np.sqrt(2),
                bins=20,
                label="NMAD",
                equal_bin_counts=True,
            )

        plt.xlabel("Twinness percentile")
        plt.ylabel("Single supernova dispersion in brightness (mag)")

        plt.legend()

        return mags_20

    def _evaluate_salt_magnitude_residuals(self, additional_covariates,
                                           intrinsic_dispersion, ref_mag, alpha, beta,
                                           *covariate_slopes):
        """Evaluate SALT2 magnitude residuals for a given set of standardization
        parameters

        Parameters
        ----------
        additional_covariates : list of arrays
            Additional covariates to use in the fits (e.g. host properties). This should
            be a list of arrays, each of which has the same length as the number of
            SNe Ia in the dataset.
        intrinsic_dispersion : float
            Assumed intrinsic dispersion of the sample.
        ref_mag : float
            The intrinsic B-band brightness of Type Ia supernovae
        alpha : float
            Standardization coefficient for the SALT2 x1 parameter
        beta : float
            Standardization coefficient for the SALT2 color parameter
        covariate_slopes : list
            Slopes for each of the additional covariates.

        Returns
        -------
        residuals : numpy.array
            The SALT2 magnitude residuals for every target in the dataset
        residual_uncertainties : numpy.array
            The associated uncertainties on the SALT2 magnitude residuals.
        """
        salt_fits = self.salt_fits

        mb = -2.5*np.log10(salt_fits['x0'].data)
        x0_err = salt_fits['x0_err'].data
        mb_err = utils.frac_to_mag(x0_err / salt_fits['x0'].data)
        x1_err = salt_fits['x1_err'].data
        color_err = salt_fits['c_err'].data

        cov_mb_x1 = salt_fits['covariance'].data[:, 1, 2] * -mb_err / x0_err
        cov_color_mb = salt_fits['covariance'].data[:, 1, 3] * -mb_err / x0_err
        cov_color_x1 = salt_fits['covariance'].data[:, 2, 3]

        peculiar_velocity_uncertainties = \
            self.calculate_peculiar_velocity_uncertainties(self.redshifts)

        model = (
            ref_mag
            - alpha * salt_fits['x1'].data
            + beta * salt_fits['c'].data
        )

        for slope, covariate in zip(covariate_slopes, additional_covariates):
            model += slope * covariate

        residual_uncertainties = np.sqrt(
            intrinsic_dispersion**2
            + peculiar_velocity_uncertainties**2
            + mb_err**2
            + alpha**2 * x1_err**2
            + beta**2 * color_err**2
            + 2 * alpha * cov_mb_x1
            - 2 * beta * cov_color_mb
            - 2 * alpha * beta * cov_color_x1
        )

        residuals = mb - model

        return residuals, residual_uncertainties

    def fit_salt_magnitude_residuals(self, mask=None, additional_covariates=[],
                                     bootstrap=False, verbosity=None):
        """Calculate SALT2 magnitude residuals

        This follows the standard procedure of estimating the alpha and beta correction
        parameters using an assumed intrinsic dispersion, then solving for the intrinsic
        dispersion that sets the chi-square to 1. We repeat this procedure until the
        intrinsic dispersion converges.
        """
        if verbosity is None:
            verbosity = self.settings['verbosity']

        # Start with a complete mask if there wasn't a user specified one.
        if mask is None:
            mask = np.ones(len(self.salt_fits), dtype=bool)
        else:
            mask = mask.copy()

        # Reject bad SALT2 fits.
        mask &= self.salt_mask

        # Require reasonable redshifts and colors for the determination of
        # standardization parameters. The redshift_color_mask produced by the
        # read_between_the_lines algorithm does this.
        mask &= self.redshift_color_mask

        if bootstrap:
            # Do a bootstrap resampling of the dataset. We can use this to estimate
            # uncertainties on all of our parameters.
            mask = np.random.choice(np.where(mask)[0], np.sum(mask))

        # Starting value for intrinsic dispersion. We will update this in each
        # round to set chi2 = 1
        intrinsic_dispersion = 0.1

        for i in range(10):
            def calc_dispersion(*fit_parameters):
                residuals, residual_uncertainties = \
                    self._evaluate_salt_magnitude_residuals(additional_covariates,
                                                            *fit_parameters)

                mask_residuals = residuals[mask]
                mask_residual_uncertainties = residual_uncertainties[mask]

                weights = 1 / mask_residual_uncertainties**2

                dispersion = np.sqrt(
                    np.sum(weights * mask_residuals**2)
                    / np.sum(weights)
                )

                return dispersion

            def to_min_fit_parameters(x):
                return calc_dispersion(intrinsic_dispersion, *x)

            start_vals = [-10, 0.13, 3.0] + [0.] * len(additional_covariates)

            res = minimize(to_min_fit_parameters, start_vals)
            fit_parameters = res.x

            if verbosity >= 2:
                print(f"Pass {i}, ref_mag={fit_parameters[0]:.3f}, "
                      f"alpha={fit_parameters[1]:.3f}, "
                      f"beta={fit_parameters[2]:.3f}")

            # Reestimate intrinsic dispersion.
            def chisq(intrinsic_dispersion):
                residuals, residual_uncertainties = \
                    self._evaluate_salt_magnitude_residuals(
                        additional_covariates, intrinsic_dispersion, *res.x
                    )

                mask_residuals = residuals[mask]
                mask_residual_uncertainties = residual_uncertainties[mask]

                dof = 4 + len(additional_covariates)

                return np.sum(
                    mask_residuals**2 / mask_residual_uncertainties**2
                ) / (len(mask_residuals) - dof)

            def to_min_intrinsic_dispersion(x):
                chi2 = chisq(x[0])
                return (chi2 - 1)**2

            res_int_disp = minimize(
                to_min_intrinsic_dispersion,
                [intrinsic_dispersion],
                bounds=[(0, None)],
            )

            old_intrinsic_dispersion = intrinsic_dispersion
            intrinsic_dispersion = res_int_disp.x[0]

            if verbosity >= 2:
                print("  -> new intrinsic_dispersion=%.3f" % intrinsic_dispersion)

            if np.abs(intrinsic_dispersion - old_intrinsic_dispersion) < 1e-5:
                break
        else:
            raise Exception("Intrinsic dispersion didn't converge!")

        # Calculate the SALT2 magnitude residuals.
        residuals, residual_uncertainties = self._evaluate_salt_magnitude_residuals(
            additional_covariates, intrinsic_dispersion, *fit_parameters
        )

        # Calculate SALT2 uncertainties without the intrinsic dispersion component.
        raw_uncertainties = np.sqrt(
            residual_uncertainties**2 - intrinsic_dispersion**2
        )

        result = {
            'mask': mask,
            'ref_mag': res.x[0],
            'alpha': res.x[1],
            'beta': res.x[2],
            'intrinsic_dispersion': intrinsic_dispersion,
            'wrms': res.fun,
            'rms': np.std(residuals[mask]),
            'nmad': math.nmad(residuals[mask]),
            'residuals': residuals,
            'residual_uncertainties': residual_uncertainties,
            'raw_residual_uncertainties': raw_uncertainties,
        }

        if verbosity >= 1:
            print("SALT2 magnitude residuals fit: ")
            print(f"    ref_mag: {result['ref_mag']:.3f}")
            print(f"    alpha:   {result['alpha']:.3f}")
            print(f"    beta:    {result['beta']:.3f}")
            print(f"    σ_int:   {result['intrinsic_dispersion']:.3f}")
            print(f"    RMS:     {result['rms']:.3f}")
            print(f"    NMAD:    {result['nmad']:.3f}")
            print(f"    WRMS:    {result['wrms']:.3f}")

        for i in range(len(additional_covariates)):
            covariate_amplitude = res.x[3 + i]
            result[f'covariate_amplitude_{i}'] = covariate_amplitude
            if verbosity >= 1:
                print(f"    amp[{i}]:  {covariate_amplitude:.3f}")

        return result

    def bootstrap_salt_magnitude_residuals(self, num_samples=100, *args, **kwargs):
        """Bootstrap the SALT2 magnitude residuals fit to get parameter uncertainties.

        Parameters
        ----------
        num_samples : int
            The number of bootstrapping samples to do.
        *args, **kwargs
            Additional parameters passed to calculate_salt_magnitude_residuals.

        Returns
        -------
        reference : dict
            The reference values for the non-bootstrapped data.
        samples : `astropy.table.Table`
            A Table with all of the keys from calculate_salt_magnitude_residuals with
            one row per bootstrap.
        """
        # Calculate reference result
        reference = self.fit_salt_magnitude_residuals(*args, verbosity=0, **kwargs)

        # Do bootstrapping
        samples = []
        for i in tqdm.tqdm(range(num_samples)):
            samples.append(
                self.fit_salt_magnitude_residuals(*args, bootstrap=True, verbosity=0,
                                                  **kwargs)
            )

        samples = Table(samples)

        return reference, samples

    def scatter(self, variable, mask=None, weak_mask=None, label="", axis_1=0, axis_2=1,
                axis_3=None, marker_size=60, invert_colorbar=False, **kwargs):
        """Make a scatter plot of some variable against the Isomap coefficients

        variable is the values to use for the color axis of the plot.

        A boolean array can be specified for cut to specify which points to use in the
        plot. If cut is None, then the full variable list is used.

        The target variable can be passed with or without the cut already applied. This
        function will check and automatically apply it or ignore it so that the variable
        array has the same length as the coefficient arrays.

        Optionally, a weak cut can be performed where spectra not passing the cut are
        plotted as small points rather than being completely omitted. To do this,
        specify the "weak_cut" parameter with a boolean array that has the length of the
        the variable array after the base cut.

        Any kwargs are passed to plt.scatter directly.
        """
        use_embedding = self.embedding
        use_var = variable

        if mask is not None:
            use_embedding = use_embedding[mask]
            use_var = use_var[mask]

        cmap = self.settings['colormap']

        if invert_colorbar:
            cmap = cmap.reversed()

        if weak_mask is None:
            # Constant marker size
            marker_size = marker_size
        else:
            # Variable marker size
            marker_size = 10 + (marker_size - 10) * weak_mask[mask]

        fig = plt.figure()

        if use_embedding.shape[1] >= 3 and axis_3 is not None:
            ax = fig.add_subplot(111, projection="3d")
            plot = ax.scatter(
                use_embedding[:, axis_1],
                use_embedding[:, axis_2],
                use_embedding[:, axis_3],
                s=marker_size,
                c=use_var,
                cmap=cmap,
                edgecolors='k',
                linewidths=0.7,
                **kwargs
            )
            ax.set_zlabel("Component %d" % axis_3)
        else:
            ax = fig.add_subplot(111)
            plot = ax.scatter(
                use_embedding[:, axis_1],
                use_embedding[:, axis_2],
                s=marker_size,
                c=use_var,
                cmap=cmap,
                edgecolors='k',
                linewidths=0.7,
                **kwargs
            )

        ax.set_xlabel("Component %d" % (axis_1 + 1))
        ax.set_ylabel("Component %d" % (axis_2 + 1))

        if label is not None:
            cb = fig.colorbar(plot, label=label)
        else:
            cb = fig.colorbar(plot)

        if invert_colorbar:
            # workaround: in my version of matplotlib, the ticks disappear if
            # you invert the colorbar y-axis. Save the ticks, and put them back
            # to work around that bug.
            ticks = cb.get_ticks()
            cb.ax.invert_yaxis()
            cb.set_ticks(ticks)


    def do_component_blondin_plot(self, axis_1=0, axis_2=1, marker_size=40):
        indicators = self.spectral_indicators

        s1 = indicators["EWSiII6355"]
        s2 = indicators["EWSiII5972"]

        plt.figure()

        cut = s2 > 30
        plt.scatter(
            self.embedding[cut, axis_1],
            self.embedding[cut, axis_2],
            s=marker_size,
            c="r",
            label="Cool (CL)",
        )
        cut = (s2 < 30) & (s1 < 70)
        plt.scatter(
            self.embedding[cut, axis_1],
            self.embedding[cut, axis_2],
            s=marker_size,
            c="g",
            label="Shallow silicon (SS)",
        )
        cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
        plt.scatter(
            self.embedding[cut, axis_1],
            self.embedding[cut, axis_2],
            s=marker_size,
            c="black",
            label="Core normal (CN)",
        )
        cut = (s2 < 30) & (s1 > 100)
        plt.scatter(
            self.embedding[cut, axis_1],
            self.embedding[cut, axis_2],
            s=marker_size,
            c="b",
            label="Broad line (BL)",
        )

        plt.xlabel("Component %d" % (axis_1 + 1))
        plt.ylabel("Component %d" % (axis_2 + 1))

        plt.legend()

    def do_blondin_plot(self, marker_size=40):
        indicators = self.spectral_indicators

        s1 = indicators["EWSiII6355"]
        s2 = indicators["EWSiII5972"]

        plt.figure()

        cut = s2 > 30
        plt.scatter(s1[cut], s2[cut], s=marker_size, c="r", label="Cool (CL)")
        cut = (s2 < 30) & (s1 < 70)
        plt.scatter(
            s1[cut], s2[cut], s=marker_size, c="g", label="Shallow silicon (SS)"
        )
        cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
        plt.scatter(
            s1[cut], s2[cut], s=marker_size, c="black", label="Core normal (CN)"
        )
        cut = (s2 < 30) & (s1 > 100)
        plt.scatter(s1[cut], s2[cut], s=marker_size, c="b", label="Broad line (BL)")

        plt.xlabel("SiII 6355 Equivalent Width")
        plt.ylabel("SiII 5972 Equivalent Width")

        plt.legend()

    def do_blondin_plot_3d(self, marker_size=40):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        indicators = self.spectral_indicators

        s1 = indicators["EWSiII6355"]
        s2 = indicators["EWSiII5972"]

        embedding = self.embedding

        cut = s2 > 30
        ax.scatter(
            embedding[cut, 0],
            embedding[cut, 1],
            embedding[cut, 2],
            s=marker_size,
            c="r",
            label="Cool (CL)",
        )
        cut = (s2 < 30) & (s1 < 70)
        ax.scatter(
            embedding[cut, 0],
            embedding[cut, 1],
            embedding[cut, 2],
            s=marker_size,
            c="g",
            label="Shallow silicon (SS)",
        )
        cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
        ax.scatter(
            embedding[cut, 0],
            embedding[cut, 1],
            embedding[cut, 2],
            s=marker_size,
            c="black",
            label="Core normal (CN)",
        )
        cut = (s2 < 30) & (s1 > 100)
        ax.scatter(
            embedding[cut, 0],
            embedding[cut, 1],
            embedding[cut, 2],
            s=marker_size,
            c="b",
            label="Broad line (BL)",
        )

        ax.set_xlabel("Component 0")
        ax.set_ylabel("Component 1")
        ax.set_zlabel("Component 2")

        ax.legend()

    def plot_flux(self, ax, flux, fluxerr=None, *args, c=None, label=None,
                  uncertainty_label=None, **kwargs):
        """Plot a spectrum.

        See settings.py for details about the normalization and labeling of spectra.
        """
        wave = self.wave

        plot_format = self.settings['spectrum_plot_format']

        if plot_format == 'f_nu':
            plot_scale = wave**2 / 5000.**2
        elif plot_format == 'f_lambda':
            plot_scale = 1.
        else:
            raise ManifoldTwinsException(f"Invalid plot format {plot_format}")

        flux = np.atleast_2d(flux)
        if fluxerr is not None:
            fluxerr = np.atleast_2d(fluxerr)

        for idx in range(len(flux)):
            if label is None:
                use_label = None
            elif np.isscalar(label):
                if idx == 0:
                    use_label = label
                else:
                    use_label = None
            else:
                use_label = label[idx]

            ax.plot(wave, flux[idx] * plot_scale, *args, c=c, label=use_label, **kwargs)

            if fluxerr is not None:
                if uncertainty_label is None:
                    use_uncertainty_label = None
                elif np.isscalar(uncertainty_label):
                    if idx == 0:
                        use_uncertainty_label = uncertainty_label
                    else:
                        use_uncertainty_label = None
                else:
                    use_uncertainty_label = uncertainty_label[idx]

                ax.fill_between(
                    wave,
                    (flux[idx] - fluxerr[idx]) * plot_scale,
                    (flux[idx] + fluxerr[idx]) * plot_scale,
                    facecolor=c,
                    alpha=0.3,
                    label=use_uncertainty_label,
                )

        ax.set_xlabel(self.settings['spectrum_plot_xlabel'])
        ax.set_ylabel(self.settings['spectrum_plot_ylabel'])
        ax.autoscale()
        ax.set_ylim(0, None)

        if label is not None:
            ax.legend()

    def savefig(self, filename, figure=None, **kwargs):
        """Save a matplotlib figure

        Parameters
        ----------
        filename : str
            The output filename. This will be placed in the directory specified by
            self.settings['figure_directory']
        figure : `matplotlib.pyplot.figure` instance or None
            The matplotlib figure to save. If figure is None, then we get the current
            figure from matplotlib and save that.
        **kwargs
            By default, the settings for savefig are taken from
            self.settings["matplotlib_savefig_keywords"], but they can be overridden
            for individual figures with kwargs.
        """
        if figure is None:
            figure = plt.gcf()

        directory = self.settings['figure_directory']
        os.makedirs(directory, exist_ok=True)

        path = os.path.join(directory, filename)

        figure.savefig(
            path,
            **self.settings["matplotlib_savefig_keywords"],
            **kwargs
        )
