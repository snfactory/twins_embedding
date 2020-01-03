"""Default settings to use for the analysis"""

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
    "maximum_num_phase_coefficients": 4,

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
}
