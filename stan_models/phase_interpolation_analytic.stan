data {
    int<lower=0> num_targets;
    int<lower=0> num_spectra;
    int<lower=0> num_wave;
    int<lower=0> num_phase_coefficients;
    vector[num_wave] measured_flux[num_spectra];
    vector[num_wave] measured_fluxerr[num_spectra];
    vector[num_spectra] phases;
    int<lower=0> target_map[num_spectra];
    int<lower=0> spectra_target_counts[num_spectra];
    real phase_coefficients[num_spectra, num_phase_coefficients];
    // int phase_indices[num_spectra];
    vector[num_targets] salt_x1;
}
parameters {
    vector[num_wave] phase_slope;
    vector[num_wave] phase_quadratic;
    vector[num_wave] phase_slope_x1;
    vector[num_wave] phase_quadratic_x1;

    // real<lower=0> measurement_dispersion_floor;
    // vector<lower=0>[num_wave] phase_quadratic_dispersion;
    // vector<lower=0>[num_wave] phase_quadratic_dispersion;
    // vector<lower=0>[num_wave] phase_dispersion_fall_slope;
    // vector<lower=0>[num_wave] phase_dispersion_fall_quadratic;
    // vector<lower=0>[num_wave] phase_dispersion_rise_slope;
    // vector<lower=0>[num_wave] phase_dispersion_rise_quadratic;
    vector<lower=0>[num_wave] phase_dispersion_coefficients[
        num_phase_coefficients];

    vector[num_spectra] gray_offsets;
    real<lower=0> gray_dispersion_scale;
    // real<lower=1> gray_dispersion_df;
}
transformed parameters {
    vector[num_wave] model_diffs[num_spectra];
    vector[num_wave] model_scales[num_spectra];
    vector[num_wave] model_flux[num_spectra];
    vector[num_wave] model_fluxerr[num_spectra];
    vector[num_wave] shift_flux[num_spectra];
    vector[num_wave] shift_fluxerr[num_spectra];

    vector[num_wave] phase_dispersions[num_spectra];

    // vector[num_wave] model_spectra[num_spectra];
    // vector[num_wave] model_flux[num_spectra];
    // vector[num_wave] model_fluxerr[num_spectra];

    vector[num_wave] weighted_mean_num[num_targets];
    vector[num_wave] weighted_mean_denom[num_targets];
    vector[num_wave] maximum_flux[num_targets];
    vector[num_wave] maximum_fluxerr[num_targets];

    for (t in 1:num_targets) {
        weighted_mean_num[t] = rep_vector(0., num_wave);
        weighted_mean_denom[t] = rep_vector(0., num_wave);
    }

    for (s in 1:num_spectra) {
        // if (phases[s] > 0) {
            // phase_dispersions[s] = (
                // phase_dispersion_fall_slope * abs(phases[s])
                // + phase_dispersion_fall_quadratic * phases[s] * phases[s]
            // );
        // } else {
            // phase_dispersions[s] = (
                // phase_dispersion_rise_slope * abs(phases[s])
                // + phase_dispersion_rise_quadratic * phases[s] * phases[s]
            // );
        // }
        phase_dispersions[s] = rep_vector(0., num_wave);
        for (c in 1:num_phase_coefficients) {
            phase_dispersions[s] += phase_dispersion_coefficients[c] *
                phase_coefficients[s, c];
        }
        // if (phase_indices[s] == 0) {
            // phase_dispersions[s] = rep_vector(0., num_wave);
        // } else {
            // phase_dispersions[s] = phase_dispersion_coefficients[phase_indices[s]];
        // }

        model_fluxerr[s] = sqrt(
            square(measured_fluxerr[s]) +
            // square(measured_flux[s] .* phase_quadratic_dispersion * phases[s] *
                // phases[s])
            square(measured_flux[s] .* phase_dispersions[s])
        );

        model_diffs[s] = (
            gray_offsets[s]
            + phases[s] * phase_slope
            + phases[s] * phases[s] * phase_quadratic
            + phases[s] * salt_x1[target_map[s]] * phase_slope_x1
            + phases[s] * phases[s] * salt_x1[target_map[s]]
                * phase_quadratic_x1
        );

        model_scales[s] = exp(-0.4 * log(10) * model_diffs[s]);

        shift_flux[s] = measured_flux[s] ./ model_scales[s];
        shift_fluxerr[s] = model_fluxerr[s] ./ model_scales[s];

        // Analytically evaluate the maximum spectrum from the weighted mean
        // of the estimates from each individual spectrum.
        weighted_mean_num[target_map[s]] += (
            shift_flux[s] ./ square(shift_fluxerr[s])
        );
        weighted_mean_denom[target_map[s]] += 1. ./ square(shift_fluxerr[s]);
    }

    for (t in 1:num_targets) {
        maximum_flux[t] = weighted_mean_num[t] ./ weighted_mean_denom[t];
        maximum_fluxerr[t] = sqrt(1. ./ weighted_mean_denom[t]);
    }

    for (s in 1:num_spectra) {
        model_flux[s] = maximum_flux[target_map[s]] .* model_scales[s];
    }
}
model {
    // Note: the dispersion variables have lower bounds at 0, so are are using
    // weak half-normal priors.
    // phase_quadratic_dispersion ~ normal(0, 1);
    // phase_dispersion_fall_slope ~ normal(0, 1);
    // phase_dispersion_fall_quadratic ~ normal(0, 1);
    // phase_dispersion_rise_slope ~ normal(0, 1);
    // phase_dispersion_rise_quadratic ~ normal(0, 1);
    gray_dispersion_scale ~ normal(0, 1);

    gray_offsets ~ normal(0, gray_dispersion_scale);

    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
