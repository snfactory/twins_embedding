from astropy import table
from hashlib import md5
from idrtools import Dataset, math
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn.manifold import Isomap
import extinction
import numpy as np
import os
import pickle
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

        self.print_verbose("Loading other indicators of diversity...")
        self.load_indicators()

        self.print_verbose("Fitting RBTL GP to magnitude residuals...")
        self.residuals_rbtl_gp = self.fit_gp_magnitude_residuals()

        self.print_verbose("Calculating SALT2 magnitude residuals...")
        self.residuals_salt = self.fit_salt_magnitude_residuals()

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
        self.salt_fits = table.Table([i.salt_fit for i in self.targets])
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

        # The signs of the embedding are arbitrary... flip the sign of some of them to
        # make them match up with well-known indicators in the literature.
        for component in self.settings['isomap_flip_components']:
            if num_components > component:
                embedding[:, component] *= -1

        return embedding

    def load_indicators(self):
        """Calculate/load a range of different indicators of intrinsic diversity"""
        all_indicators = []

        # Dummy table with the target name
        target_table = table.Table({'name': [i.name for i in self.targets]},
                                   masked=True)
        all_indicators.append(target_table)

        # Add in all of the different indicators that are available.
        all_indicators.append(self.load_isomap_indicators())
        all_indicators.append(self.load_salt_indicators())
        all_indicators.append(self.calculate_spectral_indicators())
        all_indicators.append(self.load_nordin_colors())
        all_indicators.append(self.load_sugar_components())
        all_indicators.append(self.load_snemo_components())
        all_indicators.append(self.load_host_data())
        all_indicators.append(self.load_peculiar_data())

        all_indicators = table.hstack(all_indicators)

        self.indicators = all_indicators

        # Extract masks that we will use extensively.
        self.peculiar_mask = (self.indicators['peculiar_type'] == 'Normal').filled()
        self.host_mask = ~self.indicators['host_lssfr'].mask

    def load_isomap_indicators(self):
        """Extract Isomap indicators"""
        columns = []
        for i in range(self.settings['isomap_num_components']):
            columns.append(table.MaskedColumn(
                self.embedding[:, i],
                name=f'isomap_c{i+1}',
                mask=~self.uncertainty_mask,
            ))

        return table.Table(columns)

    def load_salt_indicators(self):
        """Extract SALT2.4 indicators from the fits"""
        salt_indicators = table.Table(self.salt_fits[['c', 'x1']], masked=True)
        salt_indicators['c'].name = 'salt_c'
        salt_indicators['x1'].name = 'salt_x1'
        salt_indicators['salt_c'].mask = ~self.salt_mask
        salt_indicators['salt_x1'].mask = ~self.salt_mask

        return salt_indicators

    def calculate_spectral_indicators(self):
        """Calculate spectral indicators for all of the features"""
        spectral_indicators = []

        for idx in range(len(self.scale_flux)):
            spec = specind.Spectrum(
                self.wave, self.scale_flux[idx], self.scale_fluxerr[idx]**2
            )
            indicators = spec.get_spin_dict()
            spectral_indicators.append(indicators)

        spectral_indicators = table.Table(spectral_indicators, masked=True)

        # Figure out Branch classifications
        all_si6355 = spectral_indicators["EWSiII6355"]
        all_si5972 = spectral_indicators["EWSiII5972"]

        branch_classifications = []
        branch_plot_colors = []

        for si6355, si5972 in zip(all_si6355, all_si5972):
            if si5972 >= 30:
                branch_classifications.append("Cool")
            elif (si5972 < 30) & (si6355 < 70):
                branch_classifications.append("Shallow Silicon")
            elif (si5972 < 30) & (si6355 >= 70) & (si6355 < 100):
                branch_classifications.append("Core Normal")
            elif (si5972 < 30) & (si6355 >= 100):
                branch_classifications.append("Broad Line")

        spectral_indicators['branch_classification'] = branch_classifications

        for colname in spectral_indicators.colnames:
            # Mask out indicators that we shouldn't be using.
            spectral_indicators[colname].mask = ~self.uncertainty_mask

            if 'branch' not in colname:
                spectral_indicators.rename_column(colname, f'spectrum_{colname}')

        return spectral_indicators

    def _load_table(self, path, name_key):
        """Read a table from a given path and match it to our list of targets"""
        # Read the table
        data = table.Table.read(path)

        # Make a dummy table with the names of each of our SNe~Ia
        name_table = table.Table({name_key: [i.name for i in self.targets]})

        # Join the tables
        ordered_table = table.join(name_table, data, join_type='left')

        return ordered_table

    def load_sugar_components(self):
        """Load the SUGAR components from Leget et al. 2019"""
        pickle_data = open('./data/sugar_parameters.pkl').read() \
            .replace('\r\n', '\n').encode('latin1')
        sugar_data = pickle.loads(pickle_data, encoding='latin1')

        sugar_keys = ['q1', 'q2', 'q3', 'Av', 'grey']

        sugar_rows = []
        for target in self.targets:
            try:
                row = sugar_data[target.name.encode('latin1')]
                sugar_rows.append([row[i] for i in sugar_keys])
            except KeyError:
                sugar_rows.append(np.ma.masked_array([np.nan]*len(sugar_keys),
                                                     [1]*len(sugar_keys)))

        sugar_components = table.Table(
            rows=sugar_rows,
            names=[f'sugar_{i}' for i in sugar_keys],
        )

        return sugar_components

    def load_nordin_colors(self):
        """Load the U-band colors from Nordin et al. 2018"""
        nordin_table = self._load_table('./data/nordin_2018_colors.csv', 'name')

        for colname in nordin_table.colnames:
            nordin_table.rename_column(colname, f'nordin_{colname}')

        return nordin_table

    def load_snemo_components(self):
        """Load the SNEMO components from Saunders et al. 2018"""
        snemo_table = self._load_table('./data/snemo_salt_coefficients_snf.csv', 'SN')

        for colname in snemo_table.colnames:
            if 'snemo' not in colname:
                snemo_table.rename_column(colname, f'snemo_{colname}')
                continue

        return snemo_table

    def load_host_data(self):
        """Load host data from Rigault et al. 2019"""
        host_data = self._load_table('./data/host_properties_rigault_valid.csv', 'name')

        # Note: This list is from private communication and has the same names as our
        # dataset. The data table in Rigault et al. 2019 uses IAU names which can be
        # converted using self.iau_name_map.get(name, name)
        for original_colname in host_data.colnames:
            colname = original_colname

            if 'host' not in colname:
                colname = f'host_{colname}'

            colname = colname.replace('.', '_')

            host_data.rename_column(original_colname, colname)

        return host_data

    def load_peculiar_data(self):
        """Load peculiar SNe Ia information from Lin. et al 2020"""
        raw_peculiar_data = self._load_table('./data/peculiar_lin_2020.csv', 'name')

        peculiar_type = raw_peculiar_data['kind'].filled('Normal')
        peculiar_reference = raw_peculiar_data['reference'].filled('')

        peculiar_table = table.Table({
            'peculiar_type': peculiar_type,
            'peculiar_reference': peculiar_reference,
        }, masked=True)

        return peculiar_table

    def find_best_transformation(self, target_indicator,
                                 quadratic_reference_indicators=[],
                                 linear_reference_indicators=[], mask=True,
                                 shuffle=False):
        """Find the best transformation of a set of indicators to reproduce a different
        indicator.

        The indicators can be either keys corresponding to columns in the
        self.indicators table or arrays of values directly. Masks will automatically be
        extracted if the indicators are `MaskedColumn` or `numpy.ma.masked_array`
        instances. A mask can also be explicitly passed to this function which will be
        used in addition to any extracted masks.

        Parameters
        ----------
        target_indicator : str or array
            The indicator to attempt to reproduce.
        quadratic_reference_indicators : list of strs or arrays, optional
            Indicators to transform, with up to quadratic terms (including cross-terms)
            allowed in each of these indicators.
        linear_reference_indicators : list of strs or arrays, optional
            Indicators to transform, with only linear terms allowed in each of these
            indicators.
        mask : array of bools, optional
            A mask to apply (in addition to ones extracted from the indicators).
        shuffle : bool, optional
            If True, shuffle the reference indicators randomly before doing the
            transformation. This can be used to determine the significance of any
            relations. (default False)

        Returns
        -------
        explained_variance : float
            Fraction of variance that is explained by the tranformation.
        coefficients : array
            Coefficients of the transformation.
        best_transformation : array
            Transformation of the reference values that best matches the target values.
        mask : array
            Mask that was used for the transformation
        """
        def parse_column(col):
            if isinstance(col, str):
                return self.indicators[col]
            else:
                return col

        quad_ref_columns = [parse_column(i) for i in quadratic_reference_indicators]
        lin_ref_columns = [parse_column(i) for i in linear_reference_indicators]
        target_column = parse_column(target_indicator)

        # Build the mask taking masked columns into account if applicable.
        for column in quad_ref_columns + lin_ref_columns + [target_column]:
            try:
                mask = mask & ~column.mask
            except AttributeError:
                continue

        # Get basic numpy arrays for everything and apply the masks. This has a
        # surprisingly large effect on performance.
        quad_ref_values = [np.asarray(i)[mask] for i in quad_ref_columns]
        lin_ref_values = [np.asarray(i)[mask] for i in lin_ref_columns]
        target_values = np.asarray(target_column)[mask]

        if shuffle:
            # Reorder the references randomly
            order = np.random.permutation(np.arange(len(target_values)))
            quad_ref_values = [i[order] for i in quad_ref_values]
            lin_ref_values = [i[order] for i in lin_ref_values]

        num_linear_terms = len(quad_ref_values) + len(lin_ref_values)
        num_quadratic_terms = (len(quad_ref_values) + 1) * len(quad_ref_values) // 2

        num_terms = 1 + num_linear_terms + num_quadratic_terms

        def evaluate(x):
            zeropoint = x[0]
            linear_coeffs = x[1:num_linear_terms+1]
            quadratic_coeffs = x[num_linear_terms+1:]

            # Start the model with the zeropoint
            model = zeropoint

            # Linear terms. Note that the quadratic terms also have a linear one.
            lin_idx = 0
            for val in lin_ref_values:
                model += linear_coeffs[lin_idx] * val
                lin_idx += 1
            for val in quad_ref_values:
                model += linear_coeffs[lin_idx] * val
                lin_idx += 1

            # Quadratic terms.
            quad_idx = 0
            for i, val1 in enumerate(quad_ref_values):
                for j, val2 in enumerate(quad_ref_values):
                    if i > j:
                        continue
                    model += quadratic_coeffs[quad_idx] * val1 * val2
                    quad_idx += 1

            return model

        norm = 1. / len(target_values) / np.var(target_values)

        def calc_unexplained_variance(x):
            model = evaluate(x)
            diff = target_values - model
            return np.sum(diff**2) * norm

        res = minimize(calc_unexplained_variance, [0] * num_terms)
        best_guess = utils.fill_mask(evaluate(res.x), mask)
        explained_variance = 1 - calc_unexplained_variance(res.x)

        return explained_variance, res.x, best_guess, mask

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
            self,
            self.embedding,
            mags,
            mag_errs,
            covariates,
            mask,
        )

        manifold_gp.fit(verbosity=verbosity)

        return manifold_gp

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

        samples = table.Table(samples)

        return reference, samples

    def scatter(self, variable, mask=None, weak_mask=None, label=None, axis_1=0,
                axis_2=1, axis_3=None, invert_colorbar=False, **kwargs):
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

        marker_size = self.settings['scatter_plot_marker_size']

        if weak_mask is not None:
            # Variable marker size
            marker_size = 10 + (marker_size - 10) * weak_mask[mask]

        plot_kwargs = {
            's': marker_size,
            'edgecolors': 'gray',
            'cmap': cmap,
        }
        plot_kwargs.update(kwargs)

        fig = plt.figure()

        if use_embedding.shape[1] >= 3 and axis_3 is not None:
            ax = fig.add_subplot(111, projection="3d")
            plot = ax.scatter(
                use_embedding[:, axis_1],
                use_embedding[:, axis_2],
                use_embedding[:, axis_3],
                c=use_var,
                **plot_kwargs
            )
            ax.set_zlabel("Component %d" % axis_3)
        else:
            ax = fig.add_subplot(111)
            plot = ax.scatter(
                use_embedding[:, axis_1],
                use_embedding[:, axis_2],
                c=use_var,
                **plot_kwargs
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

    def scatter_combined(self, variable, mask=None, label=None, axis_1=0, axis_2=1,
                         axis_3=2, vmin=None, vmax=None, discrete_color_map=None,
                         invert_colorbar=False, **kwargs):
        """Scatter plot that shows three components simultaneously while preserving
        aspect ratios.

        The height of the figure will be adjusted automatically to produce the right
        aspect ratio.
        """
        use_embedding = self.embedding

        if np.ndim(variable) == 2:
            c12 = variable[0]
            c13 = variable[1]
            c32 = variable[2]
        else:
            c12 = c13 = c32 = variable

        if mask is not None:
            use_embedding = use_embedding[mask]
            c12 = c12[mask]
            c13 = c13[mask]
            c32 = c32[mask]

        if discrete_color_map is not None:
            cmap = ListedColormap(discrete_color_map.values())
            color_id_map = {j:i for i, j in enumerate(discrete_color_map)}
            c12 = [color_id_map[i] for i in c12]
            c13 = [color_id_map[i] for i in c13]
            c32 = [color_id_map[i] for i in c32]
        else:
            cmap = self.settings['colormap']

            if invert_colorbar:
                cmap = cmap.reversed()

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,
                                                                     vmax=vmax))
            sm._A = []

            c12 = sm.to_rgba(c12)
            c13 = sm.to_rgba(c13)
            c32 = sm.to_rgba(c32)

        min_1 = np.min(use_embedding[:, axis_1])
        max_1 = np.max(use_embedding[:, axis_1])
        min_2 = np.min(use_embedding[:, axis_2])
        max_2 = np.max(use_embedding[:, axis_2])
        min_3 = np.min(use_embedding[:, axis_3])
        max_3 = np.max(use_embedding[:, axis_3])

        range_1 = max_1 - min_1
        range_2 = max_2 - min_2
        range_3 = max_3 - min_3

        border = 0.1

        min_1 -= border * range_1
        max_1 += border * range_1
        min_2 -= border * range_2
        max_2 += border * range_2
        min_3 -= border * range_3
        max_3 += border * range_3

        range_1 *= (1 + 2. * border)
        range_2 *= (1 + 2. * border)
        range_3 *= (1 + 2. * border)

        if discrete_color_map:
            # Don't show a colorbar
            ncols = 2
            width_ratios = [range_1, range_3]
        else:
            # Add axes for a colorbar
            colorbar_frac = 0.025

            plot_width = 1 - colorbar_frac# - blank_frac
            width_1 = plot_width * range_1 / (range_1 + range_3)
            width_3 = plot_width * range_3 / (range_1 + range_3)

            ncols = 3
            width_ratios = [width_1, width_3, colorbar_frac]

        # Set the figure width. The height will be adjusted automatically to produce the
        # right aspect ratio.
        fig_width = self.settings['combined_scatter_plot_width']
        fig = plt.figure(figsize=(fig_width, fig_width))
        gs = GridSpec(
            2, ncols,
            figure=fig,
            height_ratios=[range_3, range_2],
            width_ratios=width_ratios,
        )

        ax12 = fig.add_subplot(gs[1, 0])
        ax13 = fig.add_subplot(gs[0, 0], sharex=ax12)
        ax32 = fig.add_subplot(gs[1, 1], sharey=ax12)

        if discrete_color_map:
            # Show the legend in the middle of the upper right open space.
            legend_ax = fig.add_subplot(gs[0, 1])
            legend_ax.axis('off')
        else:
            # Show the colorbar on the right side of everything.
            cax = fig.add_subplot(gs[:, 2])

        plot_kwargs = {
            's': self.settings['scatter_plot_marker_size'],
            'edgecolors': 'gray',
        }

        if discrete_color_map:
            plot_kwargs['cmap'] = cmap

        plot_kwargs.update(kwargs)

        scatter = ax12.scatter(
            use_embedding[:, axis_1],
            use_embedding[:, axis_2],
            c=c12,
            **plot_kwargs,
        )
        ax12.set_xlabel(f'Component {axis_1 + 1}')
        ax12.set_ylabel(f'Component {axis_2 + 1}')
        ax12.set_xlim(min_1, max_1)
        ax12.set_ylim(min_2, max_2)

        ax13.scatter(
            use_embedding[:, axis_1],
            use_embedding[:, axis_3],
            c=c13,
            **plot_kwargs
        )
        ax13.set_ylabel(f'Component {axis_3 + 1}')
        ax13.tick_params(labelbottom=False)
        ax13.set_ylim(min_3, max_3)

        ax32.scatter(
            use_embedding[:, axis_3],
            use_embedding[:, axis_2],
            c=c32,
            **plot_kwargs
        )
        ax32.set_xlabel(f'Component {axis_3 + 1}')
        ax32.tick_params(labelleft=False)
        ax32.set_xlim(min_3, max_3)

        if discrete_color_map:
            # Show a legend with the discrete colors
            legend_ax.legend(handles=scatter.legend_elements()[0],
                             labels=discrete_color_map.keys(),
                             loc='center')
        else:
            # Show a colorbar
            if label is not None:
                cb = fig.colorbar(sm, cax=cax, label=label)
            else:
                cb = fig.colorbar(sm, cax=cax)

            if invert_colorbar:
                # workaround: in my version of matplotlib, the ticks disappear if
                # you invert the colorbar y-axis. Save the ticks, and put them back
                # to work around that bug.
                ticks = cb.get_ticks()
                cb.ax.invert_yaxis()
                cb.set_ticks(ticks)

        # Calculate the aspect ratio, and regenerate the figure a few times until we get
        # it right.
        while True:
            fig.canvas.draw()

            coord = ax12.get_position() * fig.get_size_inches()
            plot_width = coord[1][0] - coord[0][0]
            plot_height = coord[1][1] - coord[0][1]
            plot_ratio = plot_height / plot_width

            aspect_ratio = plot_ratio / ax12.get_data_ratio()

            if np.abs(aspect_ratio - 1) < 0.001:
                # Good enough
                break

            fig.set_size_inches([fig_width, fig.get_size_inches()[1] / aspect_ratio])

        return ax12, ax13, ax32

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
            Additional kwargs to pass to savefig.
        """
        if figure is None:
            figure = plt.gcf()

        directory = self.settings['figure_directory']
        os.makedirs(directory, exist_ok=True)

        path = os.path.join(directory, filename)

        figure.savefig(
            path,
            **kwargs
        )
