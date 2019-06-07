data {
    int<lower=0> num_targets;
    int<lower=0> num_spectra;
    real measured_flux[num_spectra];
    real measured_fluxerr[num_spectra];
    real phases[num_spectra];
    int<lower=0> target_map[num_spectra];
    int<lower=0> spectra_target_counts[num_spectra];
    vector[num_targets] salt_x1;
}
parameters {
    real maximum_spectra[num_targets];
    // real gray_offsets[num_spectra];
    vector[num_spectra] gray_offsets;

    real phase_slope;
    real phase_quadratic;
    real phase_slope_x1;
    real phase_quadratic_x1;

    real<lower=0> phase_slope_dispersion;

    real<lower=0> gray_dispersion;

    // real<lower=0> dispersion_maximum;
    // real<lower=0> dispersion_slope_fall;
    // real<lower=0> dispersion_slope_rise;
}
transformed parameters {
    real model_spectra[num_spectra];
    real model_flux[num_spectra];
    real model_fluxerr[num_spectra];
    real model_diff[num_spectra];
    real model_dispersions[num_spectra];

    matrix[num_spectra, num_spectra] model_covariance;

    real cov;

    for (s in 1:num_spectra) {
        model_diff[s] = (
            phases[s] * phase_slope
            + phases[s] * phases[s] * phase_quadratic
            // + phases[s] * salt_x1[target_map[s]] * phase_slope_x1
            // + phases[s] * phases[s] * salt_x1[target_map[s]]
                // * phase_quadratic_x1;
            + gray_offsets[s]
        );

        // model_dispersions[s] = dispersion_maximum;
        // if (phases[s] > 0) {
            // model_dispersions[s] += dispersion_slope_fall * phases[s];
        // } else {
            // model_dispersions[s] += dispersion_slope_rise * -phases[s];
        // }

        model_spectra[s] =
            maximum_spectra[target_map[s]]
            + model_diff[s];

        model_flux[s] = exp(-0.4 * log(10) * model_spectra[s]);

        model_fluxerr[s] = sqrt(
            + square(measured_fluxerr[s])
            // + square(model_dispersions[s] * model_flux[s])
        );
    }

    for (i in 1:num_spectra) {
        for (j in i:num_spectra) {
            cov = 0;
            if (i == j) {
                cov += square(gray_dispersion);
            }
            if (target_map[i] == target_map[j]) {
                // Add GP kernel
                cov += square(phase_slope_dispersion) * phases[i] * phases[j];
            }
            model_covariance[i, j] = cov;
            model_covariance[j, i] = cov;
        }
    }
}
model {
    // Note: the dispersion variables have lower bounds at 0, so are are using
    // weak half-normal priors.
    // phase_quadratic_dispersion ~ normal(0, 1);
    // gray_dispersion_scale ~ normal(0, 1);
    // dispersion_maximum ~ normal(0, 1);
    // dispersion_slope_fall ~ normal(0, 1);
    // dispersion_slope_rise ~ normal(0, 1);
    gray_dispersion ~ normal(0, 1);
    phase_slope_dispersion ~ normal(0, 1);
    // gray_offsets ~ normal(0, gray_dispersion);

    // gray_offsets[1:5] ~ normal(0, 1);

    gray_offsets ~ multi_normal(rep_vector(0., num_spectra), model_covariance);

    /*
    {
        int cur_target = -1;
        int next_target = -1;
        int start = 1;
        int end = 1;
        while (start <= num_spectra) {
            while (end <= num_spectra) {
                next_target = target_map[end];
                if (next_target == cur_target || cur_target == -1) {
                    cur_target = next_target;
                    end += 1;
                } else {
                    break;
                }
            }
            print(cur_target, start, end);
            {
                int num_target_spectra = end - start + 1;
                vector target_model_offsets[num_target_spectra];
                

                // Build the covariance matrix for this target;
                // gray_offsets[start:end] ~ multi_normal(
                    // diag_matrix(square(rep_vector(gray_dispersion, end-start+1)))
                // );
            }
            start = end + 1;
            cur_target = -1;
        }
    }
    */

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

    // gray_offsets ~ normal(0, gray_dispersion_scale);
    // gray_offsets ~ student_t(gray_dispersion_df, 0, gray_dispersion_scale);

    for (t in 1:num_targets) {
        maximum_spectra[t] ~ normal(0, 10);
    }

    for (s in 1:num_spectra) {
        measured_flux[s] ~ normal(model_flux[s], model_fluxerr[s]);
    }
}
generated quantities {
    real maximum_flux[num_targets];

    for (t in 1:num_targets) {
        maximum_flux[t] = exp(-0.4 * log(10) * maximum_spectra[t]);
    }
}
