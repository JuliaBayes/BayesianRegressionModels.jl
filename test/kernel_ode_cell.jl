# test/kernel_ode_cell.jl — ODE-based kernel cell gate (todo quyu87).
#
# Question: does a `@deffun` kernel cell support Stan ODE integration through
# the `@brm kernel(...)` surface? Minimal 1-compartment elimination
# `dy/dt = -ke*y` solved with `ode_rk45` inside the cell, per-subject `ke`
# threaded as a grouped LP through the `(1 | p | subject)` bucket machinery.
# Calling convention: stanblocks-use §29 (`@deffun` RHS + `to_array_1d` /
# `[:, k]` / `to_vector` at the call site).
#
# Gate: `compiles()`/stanc + finite BridgeStan log-density/gradient (kernel
# primer rule — never settle for `transpiles()`).
# RUN: `julia --project=test test/kernel_ode_cell.jl`
# BRM_KERNEL_ODE_RUNTIME=0 skips the BridgeStan layer.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: transpiles, compiles

const ODE_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_ODE_RUNTIME", "1") != "0"
const ODE_CACHE = joinpath(tempdir(), "brm-kernel-ode")

# 1-compartment elimination RHS — the `(t, y, args...)` signature Stan
# requires. Symbolic `ny`: indexed literally, so the count lives in the
# body, the `y0`, and the extraction together (§29).
StanBlocks.@deffun begin
    one_cmt_rhs(t::real, y::vector[ny], ke::real)::vector[ny] = begin
        dy::vector[ny]
        dy[1] = -ke * y[1]
        dy
    end
end

function ode_kernel_df()
    subjects = ["S1", "S2", "S3", "S4"]
    ke_true = [0.25, 0.35, 0.2, 0.3]
    dose = 100.0
    nt = 8
    (;
        subject = subjects,
        t_grid = [collect(1.0:nt) for _ in subjects],
        dv = [[dose * exp(-ke * t) for t in 1:nt] for ke in ke_true],
    )
end

ode_kernel_brm(df) = @brm df begin
    sigma ~ Exponential(1)
    log_ke ~ 1 + (1 | p | subject)
    pred ~ kernel(t_grid, dv, log_ke) do ts, yy, lke
        ke = exp(lke)
        y0v = rep_vector(100.0, 1)
        c_pred = to_vector(ode_rk45(one_cmt_rhs, y0v, 0.0, to_array_1d(ts), ke)[:, 1])
        yy ~ normal(c_pred, sigma)
        c_pred
    end
end

function ode_bridgestan_finite(model)
    isdir(ODE_CACHE) || mkpath(ODE_CACHE)
    code = StanBlocks.stan_code(model)
    path = joinpath(ODE_CACHE, string(hash(code)) * ".stan")
    prob = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(prob)
    q = [0.1 * ((i % 5) - 2) for i in 1:dim]
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, q)
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

ode_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

@testset "ODE kernel cell — 1-cmt ode_rk45 through @brm kernel(...)" begin
    sb = SBBRMI(ode_kernel_brm(ode_kernel_df()); mod = @__MODULE__)
    @test transpiles(sb.model)
    @test compiles(sb.model)
    @test ode_stanc_ok(sb.model)
    @test occursin("ode_rk45", StanBlocks.stan_code(sb.model))
    ODE_RUN_BRIDGESTAN && @test ode_bridgestan_finite(sb.model)
end
