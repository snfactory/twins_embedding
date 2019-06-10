data {
    int<lower=0> num_targets;
    int<lower=0> num_wave;
    vector[num_wave] maximum_flux[num_targets];
    vector[num_wave] maximum_fluxerr[num_targets];
    vector[num_wave] color_law;
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
    vector[num_wave] mean_flux;
    vector<lower=0>[num_wave] fractional_dispersion;

    vector[num_targets-1] colors_raw;
    vector[num_targets-1] magnitudes_raw;
}
transformed parameters {
    vector[num_wave] model_diffs[num_targets];
    vector[num_wave] model_scales[num_targets];
    vector[num_wave] model_flux[num_targets];
    vector[num_wave] model_fluxerr[num_targets];

    vector[num_targets] colors = sum_zero_qr * colors_raw;
    vector[num_targets] magnitudes = sum_zero_qr * magnitudes_raw;

    for (t in 1:num_targets) {
        model_diffs[t] = magnitudes[t] + color_law * colors[t];
        model_scales[t] = exp(-0.4 * log(10) * model_diffs[t]);
        model_flux[t] = mean_flux .* model_scales[t];
        model_fluxerr[t] = sqrt(
            square(maximum_fluxerr[t])
            + square(fractional_dispersion .* model_flux[t])
        );
    }
}
model {
    colors ~ normal(0, 1.0);
    for (t in 1:num_targets) {
        maximum_flux[t] ~ normal(model_flux[t], model_fluxerr[t]);
    }
}
