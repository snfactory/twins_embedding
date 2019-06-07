real interpolate_linear(real target, real min_node, real max_node,
        real[] coefficients, int num_coefficients) {
    // Do linear interpolation with evenly spaced nodes. 
    int ref_index;
    real range_fraction;
    real residual_fraction;
    real left_val;
    real right_val;

    range_fraction = (target - min_node) / (max_node - min_node);

    ref_index = floor(num_coefficients * range_fraction);
    right_fraction = range_fraction - floor(range_fraction);

    left_val = coefficients[ref_index + 1];
    right_val = coefficients[ref_index + 2];

    return left_val * (1 - right_fraction) + right_val * right_fraction;
}
