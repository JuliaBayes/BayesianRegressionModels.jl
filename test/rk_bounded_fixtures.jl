# Independent public fixtures for declaration-only coordinate bounds.
using BayesianRegressionModels, Distributions

const bounded_data = (; y=[-0.7, 0.2, 0.6], lo=fill(-0.7, 3),
    hi=fill(0.6, 3), nu=6.0)
const bounded_normal_builder = @brm begin
    scale ~ Normal(0.0, 1.0; lower=0.0)
    y ~ Normal(0.0, scale)
end
const bounded_df_builder = @brm begin
    invdf ~ Exponential(0.125; lower=0.0, upper=0.5)
    nu_obs = [1/invdf, 1/invdf, 1/invdf]
    y ~ censored(LocationScale(0.0, 1.0, TDist(nu_obs)); lower=lo, upper=hi)
end
const bounded_scalar_array_builder = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    mu = [a, b, a+b]
    y ~ Normal(mu, 1.0)
end
const bounded_scalar_df_builder = @brm begin
    mu ~ Normal(0.0, 1.0)
    y ~ censored(LocationScale(mu, 1.0, TDist(nu)); lower=lo, upper=hi)
end

const bounded_data_builder = @brm begin
    scale ~ Normal(0.0, 1.0; lower=lower_bound, upper=upper_bound)
    y ~ Normal(0.0, scale)
end
const bounded_upper_builder = @brm begin
    location ~ Normal(0.0, 1.0; upper=0.5)
    y ~ Normal(location, 1.0)
end
const bounded_live_builder = @brm begin
    location ~ Normal(0.0, 1.0)
    bound = location + 2.0
    value ~ Normal(0.0, 1.0; lower=location, upper=bound)
    y ~ Normal(value, 1.0)
end
const bounded_student_prior_builder = @brm begin
    value ~ LocationScale(0.0, 1.0, TDist(6.0); lower=-0.5, upper=1.5)
    y ~ Normal(value, 1.0)
end
const bounded_beta_builder = @brm begin
    value ~ Beta(2.0, 3.0; lower=-1.0, upper=0.8)
    y ~ Normal(value, 1.0)
end
const bounded_normalized_builder = @brm begin
    scale ~ truncated(Normal(0.0, 1.0); lower=0.0)
    y ~ Normal(0.0, scale)
end

const bounded_shifted_builder = @brm begin
    value ~ Normal(0.4, 0.7; lower=0.2)
    y ~ Normal(value, 1.0)
end
const bounded_beta_interval_builder = @brm begin
    value ~ Beta(2.0, 3.0; lower=0.2, upper=0.8)
    y ~ Normal(value, 1.0)
end
