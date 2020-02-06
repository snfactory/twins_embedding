"""Utility functions for the Manifold learning analysis"""

from hashlib import md5
import os
import numpy as np
import pickle
import pystan

from idrtools import math


def compile_stan_model(model_code, verbosity=1, cache_dir='./stan_cache'):
    """Compile the given Stan code.

    The compiled model is saved so that it can be reused later without having
    to recompile every time.
    """
    code_hash = md5(model_code.encode("ascii")).hexdigest()

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = "%s/model_%s.pkl" % (cache_dir, code_hash)

    try:
        model = pickle.load(open(cache_path, "rb"))
        if verbosity >= 1:
            print("    Loaded cached stan model")
    except FileNotFoundError:
        if verbosity >= 1:
            print("    Compiling stan model...")
        model = pystan.StanModel(model_code=model_code)
        with open(cache_path, "wb") as cache_file:
            pickle.dump(model, cache_file)
        if verbosity >= 1:
            print("    Compilation successful")

    return code_hash, model


def load_stan_model(path, *args, **kwargs):
    """Load Stan code at a given path.

    The compiled model is saved so that it can be reused later without having
    to recompile every time.
    """
    with open(path) as stan_code_file:
        model_code = stan_code_file.read()

    return compile_stan_model(model_code, *args, **kwargs)


def save_stan_result(hash_str, result, cache_dir='./stan_cache'):
    """Save the result of a Stan model to a pickle file

    Parameters
    ----------
    hash_str : str
        A string that is unique to the stan model that was run.
    result
        The result of the Stan run
    cache_dir : str
        The directory to save the cached results to.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = "%s/result_%s.pkl" % (cache_dir, hash_str)

    with open(path, "wb") as outfile:
        pickle.dump(result, outfile)


def load_stan_result(hash_str, cache_dir='./stan_cache'):
    """Load the result of a previously run Stan model"""
    os.makedirs(cache_dir, exist_ok=True)
    path = "%s/result_%s.pkl" % (cache_dir, hash_str)

    try:
        with open(path, "rb") as infile:
            print("    Using saved stan result")
            return pickle.load(infile)
    except IOError:
        pass

    # No saved result
    return None


def frac_to_mag(fractional_difference):
    """Convert a fractional difference to a difference in magnitude

    Because this transformation is asymmetric for larger fractional changes, we
    take the average of positive and negative differences
    """
    pos_mag = 2.5 * np.log10(1 + fractional_difference)
    neg_mag = 2.5 * np.log10(1 - fractional_difference)
    mag_diff = (pos_mag - neg_mag) / 2.0

    return mag_diff


def latex_print(file, text):
    """Helper for writing out latex automatically.

    This just prints both to a file and to stdout so that we can see what we're doing
    """
    print(text)
    print(text, file=file)


def latex_command(file, name, formatstr, val):
    """Generate a latex command to define a variable."""
    latex_print(file, "\\newcommand{\\%s}{%s}" % (name, formatstr % val))


def latex_std(file, name, val):
    """Generate a latex command to capture the standard deviation of a parameter"""
    std, std_err = math.bootstrap_statistic(np.std, val, ddof=1)
    latex_command(file, name, '%.3f $\\pm$ %.3f', (std, std_err))


def latex_nmad(file, name, val):
    """Generate a latex command to capture the NMAD of a parameter"""
    nmad, nmad_err = math.bootstrap_statistic(math.nmad, val)
    latex_command(file, name, '%.3f $\\pm$ %.3f', (nmad, nmad_err))


def fill_mask(array, mask, fill_value=np.nan):
    """Fill in an array with masked out entries.

    Parameters
    ----------
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
    -------
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
