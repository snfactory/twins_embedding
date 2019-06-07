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
    vector[num_wave] model_flux[num_spectra];
    vector[num_wave] model_fluxerr[num_spectra];

    matrix[num_wave, num_wave] gp_cov;
    matrix[num_wave, num_wave] gp_L_cov;

    vector[num_wave] zeros;

    zeros = rep_vector(0, num_wave);

    //gp_cov = (
        //cov_exp_quad(log_wavelength, 0.1, length_scale)
        //* (fractional_dispersion * to_row_vector(fractional_dispersion))
    //);

    //for (w in 1:num_wave) {
        //gp_cov[w, w] = gp_cov[w, w] + 1e-12;
    //}

    //gp_L_cov = cholesky_decompose(gp_cov);

    for (s in 1:num_spectra) {
        model_flux[s] = mean_spectrum .* exp(-0.4 * log(10) * (
            magnitudes[target_map[s]] +
            color_law * colors[target_map[s]] +
            // max_spectra_diff[target_map[s]] +
            phase_slope * phases[s] +
            phase_quadratic * phases[s] * phases[s]
        ));

        model_fluxerr[s] = 0.02 * model_flux[s];
    }
}
model {
    sum(colors) ~ normal(0, 0.1);

    # for (t in 1:num_targets) {
        # max_spectra_diff[t] ~ multi_normal_cholesky(zeros, gp_L_cov);
    # }

    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
