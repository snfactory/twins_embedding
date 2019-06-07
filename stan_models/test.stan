data {
    int<lower=0> num_targets;
    int<lower=0> num_spectra;
    int<lower=0> num_wave;
    vector[num_wave] measured_flux[num_spectra];
    vector[num_wave] color_law;
    vector[num_spectra] phases;
    int<lower=0> target_map[num_spectra];
    real log_wavelength[num_wave];
}
parameters {
    simplex[num_wave] mean_spectrum;
    vector[num_wave] phase_slope;
    vector[num_wave] phase_quadratic;
    vector[num_targets] colors;
    vector[num_targets] magnitudes;
    vector<lower=0>[num_wave] fractional_dispersion;

    real<lower=0> length_scale;
    vector[num_wave] max_spectra_mag[num_targets];
}
transformed parameters {
    vector[num_wave] model_flux[num_spectra];
    vector[num_wave] model_fluxerr[num_spectra];

    matrix[num_wave, num_wave] gp_cov = (
        cov_exp_quad(log_wavelength, 1., length_scale) *
        (fractional_dispersion * to_row_vector(fractional_dispersion))
    );

    for (s in 1:num_spectra) {
        model_flux[s] = exp(-0.4 * log(10) * (
            max_spectra_mag[target_map[s]] +
            phase_slope * phases[s] +
            phase_quadratic * phases[s] * phases[s]
        ));

        model_fluxerr[s] = 0.02 * model_flux[s];
    }
}
model {
    for (t in 1:num_targets) {
        max_spectra_mag[t] ~ multi_normal(
            -2.5 * log(mean_spectrum) / log(10) +
            magnitudes[t] +
            color_law * colors[t],
            gp_cov
        );
    }

    sum(colors) ~ normal(0, 0.1);

    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
