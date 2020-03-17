import numpy as np
from scipy.optimize import minimize
from idrtools import math
from matplotlib import pyplot as plt


def fit_step(side_probabilities, residuals, uncertainties, mask, verbosity=1,
             calculate_covariance=True):
    fit_side_probabilities = side_probabilities[mask]
    fit_residuals = residuals[mask]
    fit_uncertainties = uncertainties[mask]

    def calc_likelihood(x):
        offset_1, offset_2, dispersion_1, dispersion_2 = x

        var_1 = fit_uncertainties**2 + dispersion_1**2
        var_2 = fit_uncertainties**2 + dispersion_2**2

        likelihood = np.sum(-np.log(
            (1 - fit_side_probabilities) / np.sqrt(2 * np.pi * var_1)
                * np.exp(-(fit_residuals - offset_1)**2 / 2. / var_1)
            + fit_side_probabilities / np.sqrt(2 * np.pi * var_2)
                * np.exp(-(fit_residuals - offset_2)**2 / 2. / var_2)
        ))

        return likelihood

    bounds =  [(None, None), (None, None), (0., None), (0., None)]

    res = minimize(
        calc_likelihood,
        [0., 0., 0.1, 0.1],
        bounds=bounds
    )

    if not res['success']:
        raise Exception("Fit failed!")

    parameter_names = ['offset_1', 'offset_2', 'dispersion_1', 'dispersion_2']

    result = {
        'fit_result': res.message,
        'step_size': res.x[1] - res.x[0],
    }

    for parameter_name, value in zip(parameter_names, res.x):
        result[parameter_name] = value

    if calculate_covariance:
        cov = math.calculate_covariance_finite_difference(
            calc_likelihood,
            ["offset_1", "offset_2", "dispersion_1", "dispersion_2"],
            res.x,
            bounds,
            verbose=verbosity >= 3
        )
        parameter_uncertainties = np.sqrt(np.diag(cov))

        # Estimate the variances with the intrinsic components.
        total_uncertainties = np.sqrt(
            uncertainties**2
            + (1 - side_probabilities) * res.x[2]**2
            + side_probabilities * res.x[3]**2
        )
        result['total_uncertainties'] = total_uncertainties

        # Calculate the total uncertainty on the step size.
        step_size_uncertainty = np.sqrt(cov[0, 0] + cov[1, 1] + 2.*cov[0, 1])
        result['step_size_uncertainty'] = step_size_uncertainty

        for name, uncertainty in zip(parameter_names, parameter_uncertainties):
            result[name + '_uncertainty'] = uncertainty

        if verbosity >= 1:
            print(
                f"    Step: {result['step_size']:+.3f} ± "
                f"{result['step_size_uncertainty']:.3f} mag, "
                f"σ1: {result['dispersion_1']:.3f} ± "
                f"{result['dispersion_1_uncertainty']:.3f} mag, "
                f"σ2: {result['dispersion_2']:.3f} ± "
                f"{result['dispersion_2_uncertainty']:.3f} mag"
            )

    return result


def plot_mean(ax, min_x, max_x, mean, mean_uncertainty, color):
    ax.plot(
        [min_x, max_x],
        [mean] * 2,
        c='k',
        zorder=-1
    )
    ax.fill_between(
        [min_x, max_x],
        [mean - mean_uncertainty] * 2,
        [mean + mean_uncertainty] * 2,
        color=color,
        alpha=0.5,
        zorder=-3
    )


def plot_step(variable, residuals, residual_uncertainties, host_data, mask, title=None,
              **kwargs):
    if variable == 'host_lssfr':
        threshold = -10.8
        x_label = 'log(lsSFR)'
        z_label = '$P_{Young}$'
        probability_tag = 'host_p(prompt)'
    elif variable == 'host_gmass':
        threshold = 10.
        x_label = 'log($M_* / M_\odot$) (global)'
        z_label = '$P_{High mass}$'
        probability_tag = 'host_p(highgmass)'
    else:
        raise Exception(f"Unknown variable {variable}!")

    host_values = host_data[variable]
    host_values_down = host_data[variable + '_err_down']
    host_values_up = host_data[variable + '_err_up']
    host_probabilities = host_data[probability_tag]

    step_result = fit_step(host_probabilities, residuals, residual_uncertainties, mask,
                           **kwargs)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 4),
                                   gridspec_kw={'width_ratios': [5, 1]})

    if title is not None:
        fig.suptitle(title)

    ax1.errorbar(
        host_values[mask],
        residuals[mask],
        xerr=(host_values_down[mask], host_values_up[mask]),
        yerr=step_result['total_uncertainties'][mask],
        fmt='.',
        c='gray',
        alpha=0.5,
        zorder=-2
    )
    scatter = ax1.scatter(
        host_values[mask],
        residuals[mask],
        s=100,
        c=host_probabilities[mask],
        edgecolors='gray',
        cmap=plt.cm.viridis_r
    )

    # Threshold
    ax1.axvline(threshold, c='k', lw=2, ls='--')

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Magnitude residuals")

    ax1.set_ylim(-0.6, 0.6)

    weights_1 = host_probabilities[mask]# / np.sum(host_probabilities[mask])
    weights_2 = (1 - host_probabilities)[mask]# / np.sum((1 - host_probabilities)[mask])

    ax2.set_axis_off()
    ax2.hist(residuals[mask], 15, (-0.6, 0.6), orientation="horizontal", linewidth=1.,
             edgecolor='gray', facecolor=plt.cm.viridis(0), weights=weights_1,
             zorder=-3)
    ax2.hist(residuals[mask], 15, (-0.6, 0.6), orientation="horizontal", linewidth=1.,
             edgecolor='gray', facecolor=plt.cm.viridis(1000), weights=-weights_2,
             zorder=-3)

    # Show means of each side on each plot.
    for ax, middle in [(ax1, threshold), (ax2, 0)]:
        plot_min, plot_max = ax.get_xlim()
        plot_mean(
            ax,
            plot_min,
            middle,
            step_result['offset_1'],
            step_result['offset_1_uncertainty'],
            plt.cm.viridis(1000)
        )
        plot_mean(
            ax,
            middle,
            plot_max,
            step_result['offset_2'],
            step_result['offset_2_uncertainty'],
            plt.cm.viridis(0)
        )
        ax.set_xlim(plot_min, plot_max)

    plt.colorbar(scatter, label=z_label, aspect=30)

    return ax1, ax2
