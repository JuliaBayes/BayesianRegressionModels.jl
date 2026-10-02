# Shared fixtures for RK parity and the independent BridgeStan spline anchors.
using BayesianRegressionModels
using Distributions
using Random: Xoshiro, randn

# BridgeStan full-posterior values (propto=false, Jacobian included) with
# canonical range-projection signs, StanBlocks eeee3bad, BridgeStan 2.9.0.
const SPLINE_S_SB_ANCHOR = -1141.2148096907295
const SPLINE_T2_SB_ANCHOR = -359.94318407167754
const SPLINE_ACCEL_SB_ANCHOR = -69.1125421088841
const SPLINE_ACCEL_SB_ZERO_ANCHOR = -46.468310103713605

function spline_s_parity_case()
    rng = Xoshiro(7207)
    x = 5 .* randn(rng, 80)
    y = sin.(x) .+ 0.3 .* randn(rng, 80)
    brmi = @brm (; x, y) begin
        mu ~ 1 + s(x)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    (; brmi, x, y)
end

function spline_t2_parity_case()
    rng = Xoshiro(7208)
    x = 4 .* randn(rng, 80)
    z = 3 .* randn(rng, 80) .+ 1.0
    y = 0.5 .* x .- 0.25 .* z .+ 0.3 .* randn(rng, 80)
    brmi = @brm (; x, z, y) begin
        mu ~ 1 + t2(x, z; k=(3, 3))
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ Normal(0, 5)
        sigma ~ Exponential(1)
    end
    (; brmi, x, z, y)
end

# Public RK test_smooth_sb.jl's accel_splines toy probe (12 observations),
# with BRM's constant null columns retained in both predictors.
function spline_accel_parity_case()
    x = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    x2 = copy(x)
    y = [-1.4, -1.2, -0.5, 0.3, 1.1, 0.8, -0.2, -0.9, -1.4, -0.7, 0.2, 0.9]
    brmi = @brm (; x, x2, y) begin
        mu ~ 1 + s(x)
        log(sigma) ~ 1 + s(x2)
        y ~ Normal(mu, sigma)
        effect(mu, Intercept) ~ LocationScale(-13, 36, TDist(3))
        effect(sigma, Intercept) ~ LocationScale(0, 10, TDist(3))
        sd(mu, s(x)) ~ LocationScale(0, 36, TDist(3))
        sd(sigma, s(x2)) ~ LocationScale(0, 10, TDist(3))
    end
    (; brmi, x, x2, y)
end
