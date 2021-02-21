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
    """Class to build and evaluate a Gaussian Process over a given manifold."""
    def __init__(self, analysis, coordinates, target_values, target_value_uncertainties,
                 covariates=None, mask=None, parameters=None):
        self.analysis = analysis
        self.coordinates = coordinates
        self.target_values = target_values
        self.target_value_uncertainties = target_value_uncertainties

        if mask is None:
            self.mask = np.ones(len(target_values), dtype=bool)
        else:
            self.mask = mask

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

        if self.covariates is not None:
            for i in range(self.covariates.shape[0]):
                parameter_names.append(f'covariate_slope_{i}')

        return parameter_names

    @property
    def parameter_bounds(self):
        parameter_bounds = [
            (1e-5, None),
            (1e-5, None),
            (0.1, None),
            (None, None),
        ]

        if self.covariates is not None:
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
            self.coordinates[self.mask],
            self.target_value_uncertainties[self.mask],
            gp_parameters
        )

        # Calculate the covariate model for the conditioning dataset.
        model = offset
        if len(covariate_slopes) > 0:
            model = offset + self.covariates.T.dot(covariate_slopes)

        condition_residuals = (self.target_values - model)[self.mask]

        result = -gp.log_likelihood(condition_residuals)

        return result

    def fit(self, cov=True, verbosity=1, options={}, **kwargs):
        """Fit the parameters to the given dataset"""

        initial_parameters = self.parameters

        if cov:
            # Need a very accurate minimum to be able to measure the covariance since we
            # are using finite differences.
            use_options = {
                'ftol': 1e-14
            }
        else:
            use_options = {}

        use_options.update(options)

        # For some reason BFGS has some convergence issues with default parameters.
        # Using a small value for ftol fixes this.
        result = minimize(
            self.negative_log_likelihood,
            initial_parameters,
            bounds=self.parameter_bounds,
            options=use_options,
            **kwargs,
        )

        self.fit_result = result
        self.parameters = result.x

        if cov:
            # Calculate the parameter covariance using a custom code that numerically
            # estimates the Hessian using a finite difference method with adaptive step
            # sizes.
            param_cov = math.calculate_covariance_finite_difference(
                self.negative_log_likelihood,
                self.parameter_names,
                self.parameters,
                self.parameter_bounds,
                verbose=verbosity >= 3,
                allow_no_effect=True,
            )

            self.parameter_covariance = param_cov
            self.parameter_uncertainties = np.sqrt(np.diag(param_cov))
        else:
            self.parameter_covariance = np.nan * np.ones((len(self.parameters),
                                                          len(self.parameters)))
            self.parameter_uncertainties = np.nan * np.ones(len(self.parameters))

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
            print("GP magnitude residuals fit:")
            print(f"    Fit result:           {result['message']}")
            for parameter_name, value, uncertainty in \
                    zip(self.parameter_names, self.parameters,
                        self.parameter_uncertainties):
                if cov:
                    print(f"    {parameter_name:25s} {value:.3f} ± {uncertainty:.3f}")
                else:
                    print(f"    {parameter_name:25s} {value:.3f}")

            # Calculate statistics
            good_residuals = self.residuals[self.mask]
            nmad = math.nmad(good_residuals)
            std = np.std(good_residuals, ddof=1)

            print(f"    {'Fit NMAD':25s} {nmad:.3f} mag")
            print(f"    {'Fit std':25s} {std:.3f} mag")

    def _calculate_covariate_model(self, prediction_covariates=None, parameters=None):
        """Calculate the model with a given set of covariates"""
        if parameters is None:
            parameters = self.parameters

        gp_parameters, offset, covariate_slopes = self._parse_parameters(parameters)

        # Add in zeropoint offset
        model = offset

        # Add in covariates.
        if prediction_covariates is not None:
            prediction_covariates = np.atleast_2d(prediction_covariates)
            model += prediction_covariates.T.dot(covariate_slopes)

        return model

    def predict(self, prediction_coordinates, prediction_covariates=None,
                parameters=None, mask=None, return_uncertainties=True):
        """Predict a Gaussian Process on the given data."""
        if parameters is None:
            parameters = self.parameters

        if mask is None:
            mask = np.ones(len(self.target_values), dtype=bool)

        if self.mask is not None:
            mask = mask & self.mask

        gp_parameters, offset, covariate_slopes = self._parse_parameters(parameters)

        gp = _build_george_gp(
            self.coordinates[mask],
            self.target_value_uncertainties[mask],
            gp_parameters
        )

        # Calculate the covariate model for the conditioning dataset, and subtract it
        # out to get the model residuals.
        model = self._calculate_covariate_model(self.covariates, parameters)
        condition_residuals = (self.target_values - model)[mask]

        predictions = gp.predict(
            condition_residuals,
            np.atleast_2d(prediction_coordinates),
            return_cov=False,
            return_var=return_uncertainties,
        )

        if return_uncertainties:
            predictions, prediction_variances = predictions
            prediction_uncertainties = np.sqrt(prediction_variances)

        # Add the covariate model back in to the predictions.
        prediction_model = self._calculate_covariate_model(prediction_covariates,
                                                           parameters)
        predictions += prediction_model

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
        locs = np.where(self.mask)[0]
        for loc in locs:
            oos_mask = np.zeros(len(self.target_values), dtype=bool)
            oos_mask[loc] = True

            if self.covariates is not None:
                use_covariates = self.covariates[:, oos_mask]
            else:
                use_covariates = None

            prediction, prediction_uncertainty = self.predict(
                self.coordinates[oos_mask],
                use_covariates,
                mask=~oos_mask,
                return_uncertainties=True,
            )

            predictions[oos_mask] = prediction
            prediction_uncertainties[oos_mask] = prediction_uncertainty

        if self.covariates is not None:
            use_covariates = self.covariates[:, ~self.mask]
        else:
            use_covariates = None

        other_predictions, other_prediction_uncertainties = self.predict(
            self.coordinates[~self.mask],
            use_covariates,
        )

        predictions[~self.mask] = other_predictions
        prediction_uncertainties[~self.mask] = other_prediction_uncertainties

        return predictions, prediction_uncertainties

    def plot(self, axis_1=0, axis_2=1, axis_3=2, vmin=-0.3, vmax=0.3, num_points=200,
             edgecolors='0.3', **kwargs):
        """Plot the GP predictions over the parameter space"""
        # We want to only show the residuals, so we need to take out the covariates and
        # zeropoint offset.
        covariate_model = self._calculate_covariate_model(self.covariates)
        residuals = self.target_values - covariate_model

        # Set the median value of the residuals to 0 for the plot.
        zeropoint = np.median(residuals)
        residuals -= zeropoint

        ax12, ax13, ax32 = self.analysis.scatter_combined(
            residuals,
            mask=self.mask,
            label='Magnitude residuals',
            axis_1=axis_1,
            axis_2=axis_2,
            axis_3=axis_3,
            vmin=vmin,
            vmax=vmax,
            invert_colorbar=True,
            edgecolors=edgecolors,
            **kwargs
        )

        subplot_datas = [
            (ax12, axis_1, axis_2),
            (ax13, axis_1, axis_3),
            (ax32, axis_3, axis_2),
        ]

        for ax, axis_x, axis_y in subplot_datas:
            min_x, max_x = ax.get_xlim()
            min_y, max_y = ax.get_ylim()

            plot_x, plot_y = np.meshgrid(
                np.linspace(min_x, max_x, num_points),
                np.linspace(min_y, max_y, num_points)
            )

            flat_plot_x = plot_x.flatten()
            flat_plot_y = plot_y.flatten()

            plot_coords = np.zeros((len(flat_plot_x), self.coordinates.shape[1]))
            plot_coords[:, axis_x] = flat_plot_x
            plot_coords[:, axis_y] = flat_plot_y

            # Predict the GP residuals over the manifold without covariates.
            predictions = self.predict(plot_coords, return_uncertainties=False)
            predictions -= self.parameter_dict['offset']
            predictions -= zeropoint
            predictions = predictions.reshape(plot_x.shape)

            ax.imshow(
                predictions[::-1],
                extent=(min_x, max_x, min_y, max_y),
                cmap=plt.cm.coolwarm.reversed(),
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
