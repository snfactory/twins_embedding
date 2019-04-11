data {
    int<lower=0> num_targets;
    int<lower=0> num_spectra;
    int<lower=0> num_wave;
    vector[num_wave] measured_flux[num_spectra];
    vector[num_wave] measured_fluxerr[num_spectra];
    vector[num_wave] color_law;
    vector[num_spectra] phases;
    int<lower=0> target_map[num_spectra];
    vector[num_targets] salt_x1;
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
    vector[num_wave] target_spectra[num_targets];

    vector[num_wave] phase_slope;
    vector[num_wave] phase_quadratic;
    vector[num_wave] phase_slope_x1;
    vector[num_wave] phase_quadratic_x1;

    vector<lower=0>[num_wave] target_dispersion;
    // real<lower=0> measurement_dispersion_floor;
    vector<lower=0>[num_wave] phase_quadratic_dispersion;
    vector[num_spectra] gray_offsets;
    real<lower=0> gray_dispersion;

    vector[num_targets-1] colors_raw;
    vector[num_targets-1] magnitudes_raw;
}
transformed parameters {
    vector[num_wave] mean_flux;

    vector[num_wave] target_ref[num_targets];
    vector[num_wave] target_diff[num_targets];

    vector[num_wave] model_spectra[num_spectra];
    vector[num_wave] model_flux[num_spectra];
    vector[num_wave] model_fluxerr[num_spectra];
    vector[num_wave] model_diff[num_spectra];

    vector[num_targets] colors = sum_zero_qr * colors_raw;
    vector[num_targets] magnitudes = sum_zero_qr * magnitudes_raw;

    mean_flux = exp(-0.4 * log(10) * mean_spectrum);

    for (t in 1:num_targets) {
        target_ref[t] =
            mean_spectrum + 
            magnitudes[t] +
            color_law * colors[t];
            
        target_diff[t] = target_spectra[t] - target_ref[t];
    }

    for (s in 1:num_spectra) {
        model_diff[s] = 
            gray_offsets[s]
            + phases[s] * (phase_slope
                + salt_x1[target_map[s]] * phase_slope_x1)
            + phases[s] * phases[s] * (phase_quadratic
                + salt_x1[target_map[s]] * phase_quadratic_x1);

        model_spectra[s] =
            target_spectra[target_map[s]]
            + model_diff[s];

        model_flux[s] = exp(-0.4 * log(10) * model_spectra[s]);
        // model_fluxerr[s] = fractional_dispersion .* model_flux[s];
        // model_fluxerr[s] = 1e-5 + 0.05 * model_flux[s];
        model_fluxerr[s] = sqrt(
            square(measured_fluxerr[s]) +
            // square(measurement_dispersion_floor * model_flux[s]) +
            square(phase_quadratic_dispersion * phases[s] * phases[s])
        );
    }
}
model {
    mean_spectrum ~ normal(0, 10);
    target_dispersion ~ uniform(0, 1);
    // measurement_dispersion_floor ~ uniform(0, 1);
    phase_quadratic_dispersion ~ uniform(0, 1);
    gray_dispersion ~ uniform(0, 1);

    gray_offsets ~ normal(0, gray_dispersion);

    for (t in 1:num_targets) {
        target_spectra[t] ~ normal(0, 10);
        target_diff[t] ~ normal(rep_vector(0, num_wave), target_dispersion);
    }

    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
generated quantities {
    vector[num_wave] target_flux[num_targets];
    vector[num_wave] scale_spectra[num_targets];
    vector[num_wave] scale_flux[num_targets];
    vector[num_wave] scale_fluxerr[num_targets];

    for (t in 1:num_targets) {
        target_flux[t] = exp(-0.4 * log(10) * target_spectra[t]);

        scale_spectra[t] =
            target_spectra[t]
            - magnitudes[t]
            - color_law * colors[t];

        scale_flux[t] = exp(-0.4 * log(10) * scale_spectra[t]);
        // TODO: Better error model here! Should I sample?
        scale_fluxerr[t] = sqrt(
            square(0.02 * scale_flux[t])
        );
    }
}
