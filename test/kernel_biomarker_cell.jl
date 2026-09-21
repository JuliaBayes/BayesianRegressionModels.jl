# test/kernel_biomarker_cell.jl — biomarker shape via general kernel primitives
# (todo 16q2mu4; evidence for 13u7ueb).
#
# The baked-in `biomarker_hierarchical_parametric` emitter (src/sbimpl.jl:12587+)
# is DEAD on the current tree — its documented formula does not transpile
# (dangling `sigma_rate`/`lloq`/`uloq`, dead kwargs, zero tests/examples/users;
# see the 16q2mu4 pivot comment). Before/after parity is therefore impossible;
# this file proves the AFTER side standalone: the biomarker math (6 correlated
# per-series params + time/dose response bells + TruncatedNormal obs) composed
# from general primitives only —
#   - six formula LPs sharing ONE `(1 | p | series)` bucket (the correlated
#     6-vec; replaces the legacy `_sb_term_group_block` + `ranef_correlated_draws`
#     floor the baked-in emitter used);
#   - one `kernel(...)` do-block cell per series calling the SAME StanBlocks
#     `biomarker_time_response` / `biomarker_dose_response` builtins;
#   - per-obs time/dose/obs/bounds via the `ragged` secondary axis;
#   - `truncated_normal` obs in-cell with the lifted-scalar spelling.
#
# Deliberate demo simplifications vs the documented math (structural
# correspondence, not draw parity): strictly-positive synthetic inputs so
# plain `log()` replaces the `broadcasted_max` clamp, and an all-affected
# panel (the `affectable` multiplier stays in the expression).
#
# Gate: `compiles()`/stanc + finite BridgeStan log-density/gradient (kernel
# primer rule — never settle for `transpiles()`).
# RUN: `julia --project=test test/kernel_biomarker_cell.jl`
# BRM_KERNEL_BIOMARKER_RUNTIME=0 skips the BridgeStan layer.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: transpiles, compiles

const BIOMARKER_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_BIOMARKER_RUNTIME", "1") != "0"
const BIOMARKER_CACHE = joinpath(tempdir(), "brm-kernel-biomarker")

function biomarker_kernel_df()
    persons = ["P1", "P2"]
    biomarkers = ["B1", "B2"]
    series = [p * b for p in persons for b in biomarkers]
    times = [1.0, 2.0, 3.0]
    flat_series, flat_time, flat_dose, flat_obs = String[], Float64[], Float64[], Float64[]
    for (si, _s) in enumerate(series), (ti, t) in enumerate(times)
        push!(flat_series, series[si])
        push!(flat_time, t)
        push!(flat_dose, 100.0)
        push!(flat_obs, 0.3 + 0.1 * ti + 0.05 * si)
    end
    n_obs = length(flat_obs)
    (;
        series,
        series_flat = flat_series,
        time_flat = flat_time,
        dose_flat = flat_dose,
        log_obs_flat = flat_obs,
        lloq_flat = fill(0.1, n_obs),
        uloq_flat = fill(5.0, n_obs),
        affectable_flat = fill(1.0, n_obs),
    )
end

biomarker_kernel_brm(df) = @brm df begin
    sigma ~ Exponential(1)
    baseline ~ 1 + (1 | p | series)
    time_loc ~ 1 + (1 | p | series)
    time_log_slope ~ 1 + (1 | p | series)
    time_mag ~ 1 + (1 | p | series)
    dose_loc ~ 1 + (1 | p | series)
    dose_log_slope ~ 1 + (1 | p | series)
    pred ~ kernel(
        ragged(time_flat, series_flat),
        ragged(dose_flat, series_flat),
        ragged(log_obs_flat, series_flat),
        ragged(lloq_flat, series_flat),
        ragged(uloq_flat, series_flat),
        ragged(affectable_flat, series_flat),
        baseline, time_loc, time_log_slope, time_mag, dose_loc, dose_log_slope,
    ) do ts, ds, yy, ll, uu, aff, b, tloc, tls, tmag, dloc, dls
        log_time = log(ts)
        log_dose = log(ds)
        time_response = biomarker_time_response(log_time, rep_vector(tloc, dims(ts)[1]), rep_vector(tls, dims(ts)[1]), rep_vector(tmag, dims(ts)[1]))
        dose_response = exp(biomarker_dose_response(log_dose, rep_vector(dloc, dims(ts)[1]), rep_vector(dls, dims(ts)[1])))
        mu = b + aff .* time_response .* dose_response
        yy ~ truncated_normal(mu, rep_vector(sigma, dims(mu)[1]), ll, uu)
        mu
    end
end

function biomarker_bridgestan_finite(model)
    isdir(BIOMARKER_CACHE) || mkpath(BIOMARKER_CACHE)
    code = StanBlocks.stan_code(model)
    path = joinpath(BIOMARKER_CACHE, string(hash(code)) * ".stan")
    prob = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(prob)
    q = [0.1 * ((i % 5) - 2) for i in 1:dim]
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, q)
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

biomarker_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

@testset "biomarker shape — kernel composition of the 6-param response model" begin
    sb = SBBRMI(biomarker_kernel_brm(biomarker_kernel_df()); mod = @__MODULE__)
    @test transpiles(sb.model)
    @test compiles(sb.model)
    @test biomarker_stanc_ok(sb.model)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("biomarker_time_response", code)
    @test occursin("biomarker_dose_response", code)
    @test occursin("lkj_corr_cholesky", code)
    BIOMARKER_RUN_BRIDGESTAN && @test biomarker_bridgestan_finite(sb.model)
end
