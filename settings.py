"""Default settings to use for the analysis"""

default_settings = {
    # Location of the SNfactory dataset.
    "idr_directory": "/home/kyle/data/snfactory/idr/",

    # Data release to use.
    "idr": "BLACKSTON",

    # Range of phases to allow for spectra in the analysis in restframe days.
    "phase_range": 5.0,

    # Velocity in km/s to bin the input spectra to before processing.
    "bin_velocity": 1000.,

    # Verbosity.
    # 0 = suppress most output
    # 1 = normal output
    # 2 = debug
    "verbosity": 1,


}
