data {
    int<lower=0> N; // number of data points
    int<lower=0> W; // number of wavelengths
    vector[W] f[N]; // measured flux
    vector[W] color_law; // color law
    vector[N] phases; // phases
}
parameters {
    simplex[W] mean_spectrum;
    vector[W] phase_slope;
    vector[W] phase_square;
    vector[N] colors;
    vector[N] mags;
    vector<lower=0>[W] dispersion;
}
transformed parameters {
    vector[W] max_scales[N]; // max scales
    vector[W] full_scales[N]; // full scales
    vector[W] f_max[N]; // flux at maximum
    vector[W] f_scale[N]; // scaled flux
    vector[W] ferr_scale[N]; // scaled error

    for (n in 1:N) {
        max_scales[n] = 
            mags[n] +
            color_law * colors[n];

        full_scales[n] = 
            max_scales[n] +
            phase_slope * phases[n] +
            phase_square * phases[n] * phases[n];

        f_max[n] = mean_spectrum .* exp(-0.4 * log(10) * max_scales[n]);

        f_scale[n] = mean_spectrum .* exp(-0.4 * log(10) * full_scales[n]);
        ferr_scale[n] = dispersion .* exp(-0.4 * log(10) * full_scales[n]);
    }
}
model {
    sum(colors) ~ normal(0, 0.1);
    for (n in 1:N) {
        f[n] ~ normal(f_scale[n], ferr_scale[n]);
    }
}
