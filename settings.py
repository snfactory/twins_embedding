"""Default settings to use for the analysis"""

from matplotlib import pyplot as plt

default_settings = {
    # Is the analysis being blinded?
    "blinded": True,

    # Location of the SNfactory dataset.
    "idr_directory": "/home/kyle/data/snfactory/idr/",

    # Data release to use.
    "idr": "BLACKSTON",

    # Range of phases to allow for spectra in the analysis in restframe days.
    "phase_range": 5.0,

    # Bin the spectrum with equally space bins in velocity before running the analysis.
    "bin_min_wavelength": 3300.,
    "bin_max_wavelength": 8600.,
    "bin_velocity": 1000.,

    # Verbosity.
    # 0 = suppress most output
    # 1 = normal output
    # 2 = debug
    "verbosity": 1,

    # Cut on signal-to-noise
    "s2n_cut_min_wavelength": 3300,
    "s2n_cut_max_wavelength": 3800,
    "s2n_cut_threshold": 100,

    # Parameters for the differential evolution model used to model spectra at maximum
    # light.
    "differential_evolution_num_phase_coefficients": 4,
    "differential_evolution_use_salt_x1": False,

    # Parameters for the read between the lines algorithm.
    "rbtl_fiducial_rv": 2.8,

    # For the manifold learning analysis, we reject spectra with too large of
    # uncertainties on the estimates of their spectra at maximum light. This sets the
    # threshold for "too large" as the ratio of total variance of the spectrum at
    # maximum light to the total intrinsic variance of Type Ia supernovae from the RBTL
    # analysis.
    "mask_uncertainty_fraction": 0.1,

    # Parameters for the Isomap algorithm
    "isomap_num_neighbors": 10,
    "isomap_num_components": 3,

    # Peculiar velocity (in km/s)
    "peculiar_velocity": 360,

    # Figure parameters

    # Matplotlib settings for all figures.
    "matplotlib_settings": {
        "figure.figsize": (5., 4.),
        "figure.autolayout": True,
        "figure.max_open_warning": 1000,
    },

    # Colormap to use
    "colormap": plt.cm.coolwarm,

    # Size of full-page spectra figures
    "spectrum_plot_figsize": (9., 4.),

    # Choose how to plot spectra. Options are "f_nu" or "f_lambda". In this analysis, we
    # do everything in F_lambda, but plots of SNe Ia look a lot better in F_nu because
    # the overall spectrum is flatter so we do that by default. Note that the overall
    # scale of our spectra is arbitrary.
    "spectrum_plot_format": "f_nu",

    # Default labels for spectrum plots
    "spectrum_plot_xlabel": "Wavelength ($\AA$)",
    "spectrum_plot_ylabel": "Normalized flux (erg/$cm^2$/s/Hz)"
}
