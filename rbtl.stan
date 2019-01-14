data {
    int<lower=0> N; // number of data points
    int<lower=0> W; // number of wavelengths
    vector[W] f[N]; // measured flux
    vector[W] color_law; // color law
}
parameters {
    simplex[W] mean_spectrum;
    vector[N] colors;
    vector[N] mags;
    vector<lower=0>[W] dispersion;
}
transformed parameters {
    vector[W] full_scales[N]; // full scales
    vector[W] f_scale[N]; // scaled flux
    vector[W] ferr_scale[N]; // scaled error

    for (n in 1:N) {
        full_scales[n] = mags[n] + color_law * colors[n];
        f_scale[n] = mean_spectrum .* exp(-0.4 * log(10) * full_scales[n]);
        ferr_scale[n] = dispersion .* exp(-0.4 * log(10) * full_scales[n]);
    }
}
model {
    colors ~ normal(0, 1.0);
    for (n in 1:N) {
        f[n] ~ normal(f_scale[n], ferr_scale[n]);
    }
}
