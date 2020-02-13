import george
from george import kernels
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import numpy as np
from idrtools import math


def _build_george_gp(coordinates, target_value_uncertainties, parameters):
    """Build a george Gaussian Process object and kernels."""
    total_target_value_uncertainties = np.sqrt(
        target_value_uncertainties**2
        + parameters[0]**2 * np.ones(len(coordinates))
    )

    ndim = coordinates.shape[-1]
    use_dim = list(range(ndim))

    kernel = parameters[1]**2 * kernels.Matern32Kernel(
        [parameters[2]**2] * len(use_dim), ndim=ndim, axes=use_dim
    )

    gp = george.GP(kernel)
    gp.compute(coordinates, total_target_value_uncertainties)

    return gp


class ManifoldGaussianProcess():
    """Class to build and evaluate a Gaussian Process over a given manifold"""
    def __init__(self, coordinates, target_values, target_value_uncertainties,
                 covariates=None, condition_mask=None, parameters=None):
        self.coordinates = coordinates
        self.target_values = target_values
        self.target_value_uncertainties = target_value_uncertainties
        self.condition_mask = condition_mask

        # If we have additional covariates, parse them into a nice form.
        if covariates is None:
            self.covariates = covariates
        else:
            self.covariates = np.atleast_2d(covariates)

        # Estimate the fit parameters if none were given.
        if parameters is None:
            self.parameters = self._get_default_parameters()
        else:
            self.parameters = parameters

    def _get_default_parameters(self):
        """Return a default set of parameters to use for fits."""
        default_parameters = [
            0.1,        # intrinsic dispersion
            0.2,        # kernel amplitude
            3.,         # kernel length scale
            0.,         # zeropoint offset
        ]

        # Slopes for each covariate
        if self.covariates is not None:
            default_parameters += [0.] * self.covariates.shape[0]

        return default_parameters

    @property
    def parameter_names(self):
        parameter_names = [
            'intrinsic_dispersion',
            'gp_kernel_amplitude',
            'gp_length_scale',
            'offset',
        ]

        for i in range(self.covariates.shape[0]):
            parameter_names.append(f'covariate_slope_{i}')

        return parameter_names

    @property
    def parameter_bounds(self):
        parameter_bounds = [
            (0., None),
            (0., None),
            (0.1, None),
            (None, None),
        ]

        for i in range(self.covariates.shape[0]):
            parameter_bounds.append((None, None))

        return parameter_bounds

    @property
    def parameter_dict(self):
        return dict(zip(self.parameter_names, self.parameters))

    def _parse_parameters(self, parameters):
        """Parse the parameters list, and return the GP parameters, offset and
        covariate slopes separately
        """
        gp_parameters = parameters[:3]
        offset = parameters[3]
        covariate_slopes = parameters[4:]

        return gp_parameters, offset, covariate_slopes

    def negative_log_likelihood(self, parameters=None):
        """Calculate the negative log likelihood for this Gaussian Process"""
        if parameters is None:
            parameters = self.parameters

        gp_parameters, offset, covariate_slopes = self._parse_parameters(parameters)

        gp = _build_george_gp(
            self.coordinates[self.condition_mask],
            self.target_value_uncertainties[self.condition_mask],
            gp_parameters
        )

        # Calculate the covariate model for the conditioning dataset.
        model = offset
        if len(covariate_slopes) > 0:
            model = offset + self.covariates.T.dot(covariate_slopes)

        condition_residuals = (self.target_values - model)[self.condition_mask]

        result = -gp.log_likelihood(condition_residuals)

        return result

    def fit(self, verbosity=1):
        """Fit the parameters to the given dataset"""

        initial_parameters = self.parameters

        # For some reason BFGS has some convergence issues... use Nelder-Mead instead.
        # The bounds aren't crucial, they are just requiring that the standard
        # deviations are greater than zero. We only ever deal with variances anyway, so
        # there is simply a degeneracy in the likelihood with "negative standard
        # deviations" that can be flipped to be positive if they are encountered.

        result = minimize(
            self.negative_log_likelihood,
            initial_parameters,
            bounds=self.parameter_bounds,
        )

        self.fit_result = result
        self.parameters = result.x

        # Calculate the parameter covariance using a custom code that numerically
        # estimates the Hessian using a finite difference method with adaptive step
        # sizes.

        cov = math.calculate_covariance_finite_difference(
            self.negative_log_likelihood,
            self.parameter_names,
            self.parameters,
            self.parameter_bounds,
            verbose=verbosity >= 3,
        )
        self.parameter_covariance = cov
        self.parameter_uncertainties = np.sqrt(np.diag(cov))

        # Calculate the residuals for out-of-sample predictions.
        predictions, prediction_uncertainties = self.predict_out_of_sample()
        self.residuals = self.target_values - predictions
        self.raw_residual_uncertainties = np.sqrt(
            self.target_value_uncertainties**2
            + prediction_uncertainties**2
        )

        self.residual_uncertainties = np.sqrt(
            self.raw_residual_uncertainties**2
            + self.parameter_dict['intrinsic_dispersion']**2
        )

        if verbosity >= 1:
            print(f"GP magnitude residuals fit:")
            print(f"    Fit result:           {result['message']}")
            for parameter_name, value, uncertainty in \
                    zip(self.parameter_names, self.parameters,
                        self.parameter_uncertainties):
                print(f"    {parameter_name:25s} {value:.3f} ± {uncertainty:.3f}")

            # Calculate statistics
            good_residuals = self.residuals[self.condition_mask]
            nmad = math.nmad(good_residuals)
            std = np.std(good_residuals)

            print(f"    {'Fit NMAD':25s} {nmad:.3f} mag")
            print(f"    {'Fit std':25s} {std:.3f} mag")

    def predict(self, prediction_coordinates, prediction_covariates=None,
                 parameters=None, condition_mask=None, return_uncertainties=True):
        """Predict a Gaussian Process on the given data."""
        if parameters is None:
            parameters = self.parameters

        if condition_mask is None:
            condition_mask = np.ones(len(self.target_values), dtype=bool)

        if self.condition_mask is not None:
            condition_mask = condition_mask & self.condition_mask

        gp_parameters, offset, covariate_slopes = self._parse_parameters(parameters)

        gp = _build_george_gp(
            self.coordinates[condition_mask],
            self.target_value_uncertainties[condition_mask],
            gp_parameters
        )

        # Calculate the covariate model for the conditioning dataset.
        model = offset
        if len(covariate_slopes) > 0:
            model += self.covariates.T.dot(covariate_slopes)

        condition_residuals = (self.target_values - model)[condition_mask]

        predictions = gp.predict(
            condition_residuals,
            np.atleast_2d(prediction_coordinates),
            return_cov=False,
            return_var=return_uncertainties,
        )

        if return_uncertainties:
            predictions, prediction_variances = predictions
            prediction_uncertainties = np.sqrt(prediction_variances)

        # Add in zeropoint offset
        predictions += offset

        # Add in covariates if they were passed in.
        if prediction_covariates is not None:
            prediction_covariates = np.atleast_2d(prediction_covariates)
            predictions += prediction_covariates.T.dot(covariate_slopes)

        if return_uncertainties:
            return predictions, prediction_uncertainties
        else:
            return predictions

    def predict_out_of_sample(self):
        """Do out-of-sample Gaussian Process predictions.

        For data that was used to condition the GP, we evaluate the GP prediction at
        each location using all entries other than the specific data point at that
        location. Predictions for the rest of the sample are done using the full
        conditioning sample.
        """
        predictions = np.zeros(len(self.target_values))
        prediction_uncertainties = np.zeros(len(self.target_values))

        # Do out-of-sample predictions for data in the conditioning sample.
        locs = np.where(self.condition_mask)[0]
        for loc in locs:
            mask = np.zeros(len(self.target_values), dtype=bool)
            mask[loc] = True

            prediction, prediction_uncertainty = self.predict(
                self.coordinates[mask],
                self.covariates[:, mask],
                condition_mask = ~mask,
                return_uncertainties=True,
            )

            predictions[mask] = prediction
            prediction_uncertainties[mask] = prediction_uncertainty

        other_predictions, other_prediction_uncertainties = self.predict(
            self.coordinates[~self.condition_mask],
            self.covariates[:, ~self.condition_mask],
        )

        predictions[~self.condition_mask] = other_predictions
        prediction_uncertainties[~self.condition_mask] = other_prediction_uncertainties

        return predictions, prediction_uncertainties

    def plot(self, axis_1=0, axis_2=1, num_points=200, border=0.1, vmin=-0.2, vmax=0.2,
             **kwargs):
        # Only show the data that were used for conditioning the GP.
        condition_coordinates = self.coordinates[self.condition_mask]

        scatter_x = condition_coordinates[:, axis_1]
        scatter_y = condition_coordinates[:, axis_2]

        min_x = np.nanmin(scatter_x)
        max_x = np.nanmax(scatter_x)
        min_y = np.nanmin(scatter_y)
        max_y = np.nanmax(scatter_y)

        width_x = max_x - min_x
        width_y = max_y - min_y

        min_x -= border * width_x
        max_x += border * width_x
        min_y -= border * width_y
        max_y += border * width_y

        plot_x, plot_y = np.meshgrid(
            np.linspace(min_x, max_x, num_points), np.linspace(min_y, max_y, num_points)
        )

        flat_plot_x = plot_x.flatten()
        flat_plot_y = plot_y.flatten()

        plot_coords = np.zeros((len(flat_plot_x), self.coordinates.shape[1]))

        plot_coords[:, axis_1] = flat_plot_x
        plot_coords[:, axis_2] = flat_plot_y

        predictions = self.predict(plot_coords, return_uncertainties=False)
        predictions = predictions.reshape(plot_x.shape)

        # Subtract out the fitted offset.
        gp_parameters, offset, covariate_slopes = self._parse_parameters(
            self.parameters
        )
        predictions -= offset
        plot_target_values = self.target_values[self.condition_mask] - offset

        # Apply corrections for covariates.
        if len(covariate_slopes) > 0:
            covariates_model = self.covariates.T.dot(covariate_slopes)
            plot_target_values -= covariates_model[self.condition_mask]

        fig, ax = plt.subplots()

        plot = ax.scatter(
            scatter_x,
            scatter_y,
            s=60,
            c=plot_target_values,
            cmap=plt.cm.coolwarm.reversed(),
            vmin=vmin,
            vmax=vmax,
            edgecolors='k',
            linewidths=0.7,
            **kwargs,
        )

        ax.set_xlabel("Component %d" % (axis_1 + 1))
        ax.set_ylabel("Component %d" % (axis_2 + 1))

        cb = fig.colorbar(plot, label="Magnitude residuals")

        # workaround: in my version of matplotlib, the ticks disappear if you invert the
        # colorbar y-axis. Save the ticks, and put them back to work around that bug.
        ticks = cb.get_ticks()
        cb.ax.invert_yaxis()
        cb.set_ticks(ticks)

        ax.imshow(
            predictions[::-1],
            extent=(min_x, max_x, min_y, max_y),
            cmap=plt.cm.coolwarm.reversed(),
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
