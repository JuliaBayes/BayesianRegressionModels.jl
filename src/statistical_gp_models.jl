# One statistical source for the native catalogue and formula emission.
const _GP_COVARIANCE_MODELS = (
    gp_pair_locations = :(
gp_pair_locations(x) = begin
    indices = BayesianRegressionModels.StatisticalPreparation._gp_indices(x)
    locations = Float64.(x)
    left = locations
    right = reshape(locations, 1, :)
    row = indices
    col = reshape(indices, 1, :)
    return left, right, row, col
end
    ),
    gp_exp_quad_cov = :(
gp_exp_quad_cov_graph(x, sigma, rho, jitter) = begin
    left, right, row, col = gp_pair_locations(x)
    scale = BayesianRegressionModels.StatisticalPreparation._gp_positive(sigma, :sigma)
    width = BayesianRegressionModels.StatisticalPreparation._gp_positive(rho, :rho)
    jit = BayesianRegressionModels.StatisticalPreparation._gp_jitter(jitter)
    variance = scale^2
    denominator = 2 * width^2
    covariance = plate(left, right, row, col, Ref(variance), Ref(denominator), Ref(jit)) do xi, xj, i, j, s2, denom, eps
        distance = xi - xj
        squared_distance = distance * distance
        cell = s2 * exp(-squared_distance / denom)
        diagonal_jitter = ifelse(i == j, eps, 0.0)
        cell + diagonal_jitter
    end
    return covariance
end
    ),
    gp_periodic_cov = :(
gp_periodic_cov_graph(x, sigma, rho, period, jitter) = begin
    left, right, row, col = gp_pair_locations(x)
    scale = BayesianRegressionModels.StatisticalPreparation._gp_positive(sigma, :sigma)
    width = BayesianRegressionModels.StatisticalPreparation._gp_positive(rho, :rho)
    per = BayesianRegressionModels.StatisticalPreparation._gp_period(period)
    jit = BayesianRegressionModels.StatisticalPreparation._gp_jitter(jitter)
    variance = scale^2
    squared_width = width^2
    covariance = plate(left, right, row, col, Ref(variance), Ref(squared_width), Ref(per), Ref(jit)) do xi, xj, i, j, s2, r2, p, eps
        distance = abs(xi - xj)
        sine = sin(pi * distance / p)
        cell = s2 * exp(-2 * sine * sine / r2)
        diagonal_jitter = ifelse(i == j, eps, 0.0)
        cell + diagonal_jitter
    end
    return covariance
end
    ),
)
