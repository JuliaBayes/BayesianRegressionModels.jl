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

# Shared numerical algebra over explicitly declared coordinates. These are
# ordinary functions or numerical graphs, not submodels: statistical blocks
# that allocate parameters are generated per block (see `_rk_ast_block_call!`).
const _BRM_STATISTICAL_VALUES = (
    brm_multinomial_scores = :(
brm_multinomial_scores(count_columns, trials, probabilities) = begin
    pointwise = ReactiveKernels.plate(eachindex(first(count_columns)),
            Ref(count_columns), Ref(trials), Ref(probabilities)) do row, columns, totals, probs
        total = totals isa Integer ? totals : totals[row]
        cells = sum(eachindex(columns)) do category
            count = columns[category][row]
            mass = count == 0 ? zero(probs[category]) : count * log(probs[category])
            mass - BayesianRegressionModels.loggamma(count + 1)
        end
        BayesianRegressionModels.loggamma(total + 1) + cells
    end
    return pointwise
end),

    brm_covariate_mean = :(
brm_covariate_mean(values) = begin
    mean = BayesianRegressionModels._brm_fit_mean_numeric(
        values, :predictor, :center, ArgumentError)
    return mean
end),

    brm_covariate_sd = :(
brm_covariate_sd(values) = begin
    fit = BayesianRegressionModels._brm_fit_zscale_numeric(
        values, :predictor, ArgumentError)
    return fit.scale
end),

    brm_flatten_response = :(
brm_flatten_response(cells) = begin
    values = reduce(vcat, cells; init=eltype(eltype(cells))[])
    return values
end),

    brm_gather_response = :(
brm_gather_response(raw, rows) = begin
    values = raw[rows]
    return values
end),

    brm_matrix_column = :(
brm_matrix_column(inputs, column) = begin
    matrix = only(inputs)
    values = Int.(matrix[:, column])
    return values
end
    ),
    brm_covariate_observed = :(
brm_covariate_observed(values) = begin
    observed = Float64.(collect(skipmissing(values)))
    return observed
end
    ),
    brm_covariate_observed_rows = :(
brm_covariate_observed_rows(values) = begin
    rows = findall(!ismissing, values)
    return rows
end
    ),
    brm_covariate_missing_rows = :(
brm_covariate_missing_rows(values) = begin
    rows = findall(ismissing, values)
    return rows
end
    ),
    brm_structured_inputs = :(
function brm_structured_inputs(prepared_inputs, indices)
    prepared = only(prepared_inputs)
    fields = map((field, index) -> merge(field, (; idx=index)), prepared.state.fields, indices)
    state = merge(prepared.state, (; fields))
    return [BayesianRegressionModels._BRMPreparedTerm(prepared.callable,
        prepared.source, state, prepared.dependencies)]
end
    ),
    brm_factor_dummy = :(
brm_factor_dummy(values, level) = begin
    dummy = ReactiveKernels.plate(values, Ref(level)) do value, selected
        1.0 * isequal(value, selected)
    end
    return dummy
end
    ),
    brm_prepared_indices = :(
brm_prepared_indices(values, level_values) = begin
    indices = ReactiveKernels.plate(values, Ref(level_values)) do value, declared
        Int(findfirst(isequal(value), declared))
    end
    return indices
end
    ),
    brm_covariate_geometry = :(
brm_covariate_geometry(observed, observed_rows, missing_rows) = begin
    rows = 1:(length(observed_rows) + length(missing_rows))
    observed_by_row = Dict(zip(observed_rows, observed))
    missing_by_row = Dict(zip(missing_rows, eachindex(missing_rows)))
    observed_component = ReactiveKernels.plate(rows, Ref(observed_by_row)) do row, values
        get(values, row, 0.0)
    end
    missing_lookup = ReactiveKernels.plate(rows, Ref(missing_by_row)) do row, indices
        get(indices, row, 1)
    end
    missing_mask = ReactiveKernels.plate(rows, Ref(missing_by_row)) do row, indices
        1.0 * haskey(indices, row)
    end
    return (observed_component, missing_lookup, missing_mask)
end
    ),
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
)
