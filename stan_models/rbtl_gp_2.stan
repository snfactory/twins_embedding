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
    vector[num_wave] max_spectra_diff[num_targets];
}
transformed parameters {
    vector[num_wave] max_scales[num_targets];
    vector[num_wave] full_scales[num_spectra];
    vector[num_wave] model_flux[num_spectra];
    vector[num_wave] model_fluxerr[num_spectra];

    matrix[num_wave, num_wave] gp_cov;


    gp_cov = (
        cov_exp_quad(log_wavelength, 0.1, 0.1) +
        diag_matrix(rep_vector(1e-10, num_wave))
    );

    for (t in 1:num_targets) {
        max_scales[t] =
            magnitudes[t] +
            color_law * colors[t];
    }

    max_scales[1] = max_scales[1] + max_spectra_diff[1];

    for (s in 1:num_spectra) {
        full_scales[s] =
            max_scales[target_map[s]] +
            phase_slope * phases[s] +
            phase_quadratic * phases[s] * phases[s];

        model_flux[s] = mean_spectrum .* exp(-0.4 * log(10) * full_scales[s]);
        model_fluxerr[s] = fractional_dispersion .* model_flux[s];
    }
}
model {
    for (t in 1:2) {
        max_spectra_diff[t] ~ multi_normal(rep_vector(0, num_wave), gp_cov);
    }
    sum(colors) ~ normal(0, 0.1);
    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
