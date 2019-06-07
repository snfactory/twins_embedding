data {
    int<lower=0> num_targets;
    int<lower=0> num_spectra;
    int<lower=0> num_wave;
    vector[num_wave] measured_flux[num_spectra];
    vector[num_wave] color_law;
    vector[num_spectra] phases;
    int<lower=0> target_map[num_spectra];
}
transformed data{
    // Sum-to-zero transformations
    matrix[num_targets, num_targets] sum_zero_mat =
        diag_matrix(rep_vector(1, num_targets));
    matrix[num_targets, num_targets-1] sum_zero_qr;
    for (i in 1:num_targets-1) sum_zero_mat[num_targets,i] = -1;
    sum_zero_mat[num_targets, num_targets] = 0;
    sum_zero_qr = qr_Q(sum_zero_mat)[ , 1:(num_targets-1)];
}
parameters {
    vector[num_wave] mean_spectrum;
    vector[num_wave] phase_slope;
    vector[num_wave] phase_quadratic;
    vector[num_targets-1] colors_raw;
    vector[num_targets-1] magnitudes_raw;
    vector[num_wave] target_diff[num_targets];
    vector<lower=0>[num_wave] fractional_dispersion;
    real<lower=0> phase_quadratic_dispersion;
}
transformed parameters {
    vector[num_wave] max_scales[num_targets];
    vector[num_wave] max_model_scales[num_targets];
    vector[num_wave] max_flux[num_targets];

    vector[num_wave] full_scales[num_spectra];
    vector[num_wave] model_flux[num_spectra];
    vector[num_wave] model_fluxerr[num_spectra];

    vector[num_targets] colors = sum_zero_qr * colors_raw;
    vector[num_targets] magnitudes = sum_zero_qr * magnitudes_raw;

    for (t in 1:num_targets) {
        max_scales[t] =
            magnitudes[t] +
            color_law * colors[t];

        max_model_scales[t] =
            max_scales[t] +
            target_diff[t];

        max_flux[t] = mean_spectrum .* exp(-0.4 * log(10) *
            max_model_scales[t]);
    }

    for (s in 1:num_spectra) {
        full_scales[s] =
            max_model_scales[target_map[s]] +
            phase_slope * phases[s] +
            phase_quadratic * phases[s] * phases[s];

        model_flux[s] = mean_spectrum .* exp(-0.4 * log(10) * full_scales[s]);
        // model_fluxerr[s] = fractional_dispersion .* model_flux[s];
        model_fluxerr[s] = 1e-5 + 0.05 * model_flux[s];
    }
}
model {
    for (t in 1:num_targets) {
        target_diff[t] ~ normal(rep_vector(0, num_wave), fractional_dispersion);
    }

    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
