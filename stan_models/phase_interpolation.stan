data {
    int<lower=0> num_targets;
    int<lower=0> num_spectra;
    int<lower=0> num_wave;
    vector[num_wave] measured_flux[num_spectra];
    vector[num_wave] measured_fluxerr[num_spectra];
    vector[num_spectra] phases;
    int<lower=0> target_map[num_spectra];
    int<lower=0> spectra_target_counts[num_spectra];
    vector[num_targets] salt_x1;
}
parameters {
    vector[num_wave] maximum_spectra[num_targets];

    vector[num_wave] phase_slope;
    vector[num_wave] phase_quadratic;
    vector[num_wave] phase_slope_x1;
    vector[num_wave] phase_quadratic_x1;

    // real<lower=0> measurement_dispersion_floor;
    vector<lower=0>[num_wave] phase_quadratic_dispersion;
    vector[num_spectra] gray_offsets;
    real<lower=0> gray_dispersion_scale;
    // real<lower=1> gray_dispersion_df;
}
transformed parameters {
    vector[num_wave] model_spectra[num_spectra];
    vector[num_wave] model_flux[num_spectra];
    vector[num_wave] model_fluxerr[num_spectra];
    vector[num_wave] model_diff[num_spectra];

    for (s in 1:num_spectra) {
        model_diff[s] = (
            gray_offsets[s]
            + phases[s] * phase_slope
            + phases[s] * phases[s] * phase_quadratic
            // + phases[s] * salt_x1[target_map[s]] * phase_slope_x1
            // + phases[s] * phases[s] * salt_x1[target_map[s]]
                // * phase_quadratic_x1;
        );

        model_spectra[s] =
            maximum_spectra[target_map[s]]
            + model_diff[s];

        model_flux[s] = exp(-0.4 * log(10) * model_spectra[s]);

        model_fluxerr[s] = sqrt(
            square(measured_fluxerr[s]) +
            square(measured_flux[s] .* phase_quadratic_dispersion * phases[s] *
                phases[s])
        );
    }
}
model {
    // Note: the dispersion variables have lower bounds at 0, so are are using
    // weak half-normal priors.
    phase_quadratic_dispersion ~ normal(0, 1);
    gray_dispersion_scale ~ normal(0, 1);

    // gray_dispersion_df ~ gamma(2, 0.1);

    // for (s in 1:num_spectra) {
        // if (spectra_target_counts[s] >= 2) {
            // If there are multiple spectra, draw from the gray dispersion
            // model.
            // gray_offsets[s] ~ normal(0, gray_dispersion_scale);
        // } else {
            // If there is only a single spectrum, effectively fix the gray
            // dispersion to 0 because we can't do anything with it.
            // gray_offsets[s] ~ normal(0, 0.0001);
        // }
    // }

    gray_offsets ~ normal(0, gray_dispersion_scale);
    // gray_offsets ~ student_t(gray_dispersion_df, 0, gray_dispersion_scale);

    for (t in 1:num_targets) {
        maximum_spectra[t] ~ normal(0, 10);
    }

    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
generated quantities {
    vector[num_wave] maximum_flux[num_targets];

    for (t in 1:num_targets) {
        maximum_flux[t] = exp(-0.4 * log(10) * maximum_spectra[t]);
    }
}
