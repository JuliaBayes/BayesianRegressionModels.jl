# Statistical model bodies adopted from RKPPL library.jl at
# e005f231c1feb93e4eb1c636f238ccb02fc1cb9e; input SHA256:
# fa87d6cb5ede8b258a413ed61dbb5afd298cf6bf51e105159353ff466c0c8f0c.
# PK-specific bodies belong to downstream RKPPLBench.
# The RK extension installs these compiler-independent syntax trees as
# ordinary @rkppl submodels. Shared preparation serves BRM's backends.
const _BRM_STATISTICAL_MODELS = (
    ordered_logistic = :(
ordered_logistic(eta) = begin
    cutpoints ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
    y .~ OrderedLogistic.(eta, Ref(cutpoints))
    return y
end
    ),
    penalized_smooth = :(
penalized_smooth(X, Z) = begin
    b[axes(X, 2)] .~ Flat.()
    sd ~ HalfNormal(1)
    z[axes(Z, 2)] .~ Normal.(0, 1)
    return X * b .+ Z * (sd .* z)
end
    ),
    t2_smooth = :(
t2_smooth(X, Zrr, Zrn, Znr) = begin
    b[axes(X, 2)] .~ Flat.()
    sd[1:3] .~ HalfNormal.(1)
    z_rr[axes(Zrr, 2)] .~ Normal.(0, 1)
    z_rn[axes(Zrn, 2)] .~ Normal.(0, 1)
    z_nr[axes(Znr, 2)] .~ Normal.(0, 1)
    return X * b .+ Zrr * (sd[1] .* z_rr) .+ Zrn * (sd[2] .* z_rn) .+
        Znr * (sd[3] .* z_nr)
end
    ),
    hsgp_effect = :(
hsgp_effect(PHI, lambda) = begin
    rho_floor = maximum(hsgp_rho_floors(lambda))
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ LogNormal(0, 1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_sqrt_spd(lambda, sigma, rho) .* z)
end
    ),
    hsgp_periodic_effect = :(
hsgp_periodic_effect(PHI, harmonics) = begin
    rho_floor = hsgp_periodic_rho_floor(harmonics)
    rho ~ truncated(LogNormal(0, 1), rho_floor, Inf)
    sigma ~ LogNormal(0, 1)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_periodic_sqrt_spd(harmonics, sigma, rho) .* z)
end
    ),
    hsgp_grouped_effect = :(
hsgp_grouped_effect(PHI, lambda, g) = begin
    rho_floor = maximum(hsgp_rho_floors(lambda))
    rho_mu ~ Normal(0, 1)
    rho_sd ~ HalfNormal(1)
    rho_z[levels(g)] .~ Normal.(0, 1)
    rho = max.(exp.(rho_mu .+ rho_sd .* rho_z), rho_floor)
    sigma_mu ~ Normal(0, 1)
    sigma_sd ~ HalfNormal(1)
    sigma_z[levels(g)] .~ Normal.(0, 1)
    sigma = exp.(sigma_mu .+ sigma_sd .* sigma_z)
    z[axes(PHI, 2)] .~ Normal.(0, 1)
    return PHI * (hsgp_grouped_sqrt_spd(lambda, sigma, rho) .* z)
end
    ),
    monotonic = :(
monotonic(c, zeta) = begin
    return cumsum(vcat(0.0, zeta))[c]
end
    ),
    differenced_ar1 = :(
differenced_ar1(beta, sigma) = begin
    @scan begin
        level[1] = 0.0
        increment[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            increment[t] = beta * increment[t - 1] + sigma * z
            level[t] = level[t - 1] + increment[t]
        end
    end
    return level
end
    ),
    r2d2_coefs = :(
r2d2_coefs(X, alpha) = begin
    R2 ~ Beta(1, 1)
    phi ~ Dirichlet(alpha)
    tau ~ HalfNormal(1)
    varx = var.(eachcol(X))
    b[axes(X, 2)] .~ Normal.(0, sqrt.(phi .* R2 .* tau^2 ./ varx))
    return b
end
    ),
    horseshoe_coefs = :(
horseshoe_coefs(X) = begin
    tau ~ HalfCauchy(1)
    lambda[axes(X, 2)] .~ HalfCauchy.(1)
    z[axes(X, 2)] .~ Normal.(0, 1)
    return z .* lambda .* tau
end
    ),
    varying_coefs = :(
varying_coefs(g) = begin
    sd ~ HalfNormal(1)
    z[levels(g)] .~ Normal.(0, 1)
    return sd .* z
end
    ),
    varying_coefs_correlated = :(
varying_coefs_correlated(g, K) = begin
    sd[1:K] .~ HalfNormal.(1)
    L ~ LKJCholesky(K, 1.0)
    z[levels(g), 1:K] .~ Normal.(0, 1)
    return z * (sd .* L)'
end
    ),
    varying_coefs_centered = :(
varying_coefs_centered(g) = begin
    sd ~ HalfNormal(1)
    c[levels(g)] .~ Normal.(0, sd)
    return c
end
    ),
    varying_coefs_centered_correlated = :(
varying_coefs_centered_correlated(g, K) = begin
    sd[1:K] .~ HalfNormal.(1)
    L ~ LKJCholesky(K, 1.0)
    F = sd .* L
    eachrow(c[levels(g), 1:K]) .~ MvNormalCholesky(zeros(K), F)
    return c
end
    ),
    varying_stratified = :(
varying_stratified(g, s) = begin
    sd[levels(s)] .~ HalfNormal.(1)
    z[levels(g)] .~ Normal.(0, 1)
    return sd[s] .* z[g]
end
    ),
    varying_stratified_correlated = :(
varying_stratified_correlated(g, s, K) = begin
    sd[levels(s), 1:K] .~ HalfNormal.(1)
    @plate for k in levels(s)
        L[k] ~ LKJCholesky(K, 1.0)
    end
    z[levels(g), 1:K] .~ Normal.(0, 1)
    @plate for i in eachindex(g)
        b[i, 1:K] = (sd[s[i], :] .* L[s[i]]) * z[g[i], :]
    end
    return b
end
    ),
)

"""Return a BRM-owned statistical RKPPL submodel after loading ReactiveKernelsPPL."""
function rkppl_model end

"""
    rk_model(name::Symbol)

Return a BRM-owned native statistical KernelSpec after loading ReactiveKernels
and ReactiveKernelsPPL. Available models: `:gp_exp_quad_cov`, `:gp_periodic_cov`
and `:dual_hsgp`. Covariance graphs compose into authored RK/RKPPL models;
`StatisticalPreparation` provides the matching covariance call wrappers.
The dual HSGP preserves its 44 packed coordinates, live partial centeredness,
LogNormal(0,4) hyperpriors, standard-normal weights and normalized likelihood.
"""
function rk_model end

# Shared statistical algebra over explicitly declared random coordinates.
# The emitter keeps prior statements at their current author paths, and uses
# these ordinary submodels to make repeated block composition readable.
const _BRM_STATISTICAL_VALUES = (
    brm_r2d2m2_scale = :(
brm_r2d2m2_scale(reference, phi, r2, share, variance) = begin
    allocated = phi[share] * r2
    residual = (1.0 - r2) * variance
    ratio = allocated / residual
    return reference * sqrt(ratio)
end
    ),
    brm_completed_covariate = :(
brm_completed_covariate(observed, missing, lookup, mask) = begin
    drawn = missing[lookup]
    completed = observed .+ mask .* drawn
    return completed
end
    ),
    # An ordinary whole-value contrast composes with whole-value smooths.
    # Reuse the statistical model's law rather than maintaining a second one.
    brm_monotonic_contrast = let definition = deepcopy(_BRM_STATISTICAL_MODELS.monotonic)
        first(definition.args).args[1] = :brm_monotonic_contrast
        Expr(:function, definition.args...)
    end,
    brm_scaled_random_coefficients = :(
brm_scaled_random_coefficients(sd, z) = begin
    z .* sd[1]
end
    ),
    brm_correlated_random_coefficients = :(
brm_correlated_random_coefficients(sd, L, z) = begin
    z * transpose(sd .* L)
end
    ),
    brm_hsgp_summand = :(
brm_hsgp_summand(PHI, omega2, sigma, rho, z) = begin
    PHI * (brm_hsgp_sqrt_spd(omega2, sigma, rho) .* z)
end
    ),
    brm_periodic_hsgp_summand = :(
brm_periodic_hsgp_summand(PHI, harmonics, sigma, rho, z) = begin
    PHI * (brm_hsgp_periodic_sqrt_spd(harmonics, sigma, rho) .* z)
end
    ),
    brm_grouped_hsgp_summand = :(
brm_grouped_hsgp_summand(PHI, omega2, z, group_index, rho, sigma, reader) = begin
    reader(PHI, omega2, z, group_index, rho, sigma)
end
    ),
)
