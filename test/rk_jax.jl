# test/rk_jax.jl — RKBRMI backends as Python/JAX modules,
# `jax_source(rk.model, rk_bound_plan(rk))` (brm-use
# references/reactivekernels.md, "A Python/JAX module"; snag
# `rkbrmi-to-rk-jax-a465a405`).
#
# Every run checks that representative backends, fits and replays, emit a
# module whose `prepare_data` holds BRM's preprocessing (level codes, fitted
# `zscale`/`center` anchors, cell flattens) as translated host code. With
# `RK_JAX_PYTHON` naming a Python that has `jax`, each module's `log_density`
# and `jax.grad` are also compared with the native density and its central
# differences, through the pinned RK checkout's own driver
# (`test/jax_python.jl`); without it those checks are skipped.
#
# Run: RK_JAX_PYTHON=<python with jax> julia --project=test test/rk_jax.jl
using Test
using BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL

include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

module JaxModels
using BayesianRegressionModels, Distributions
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) + center(w) + (1 | g)
    y ~ Normal(mu, sigma)
end
counts(d) = @brm d begin
    log(lambda) ~ 1 + x + (1 | g)
    c ~ Poisson(lambda)
end
# A per-subject `zscale` read in a `@plate`: its anchor is computed in the
# program, not bound, so it reaches the module's `prepare_data`.
pk(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dose[i] .* exp.(-exp(log_CL[i]) .* t[i])
        y[i] ~ normal(conc[i], sigma)
    end
end
end
const JM = JaxModels

const TRAIN = (; x=collect(range(0.3, 4.1; length=9)), w=cos.(1:9),
    g=["a", "b", "c"][mod1.(1:9, 3)], y=sin.(1:9), c=[0, 2, 1, 3, 0, 1, 4, 2, 1])
const PK = (; subject=["s1", "s2", "s3", "s4", "s5"],
    weight=[60.0, 72.0, 81.0, 55.0, 90.0], dose=[100.0, 120.0, 80.0, 100.0, 150.0],
    t=[[0.5, 1.0, 2.0], [0.25, 1.5], [0.5, 1.0, 2.0, 4.0], [1.0], [0.5, 2.0]],
    y=[[1.0, 0.8, 0.5], [1.2, 0.7], [0.9, 0.7, 0.4, 0.2], [0.6], [1.3, 0.5]])

point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
jax_function(src, name) = only(f for f in src.functions if f.name == name)

# The module's `prepare_data` takes the bound columns it lists, by name.
function raw_inputs(rk, src)
    bound = rk_bound_data(rk)
    Tuple(bound[Symbol(first(input))] for input in jax_function(src, "prepare_data").inputs)
end

function check_module(rk; anchors)
    src = jax_source(rk.model, rk_bound_plan(rk))
    @test src isa JAXSource
    @test issubset(("prepare_data", "log_density", "log_prior", "log_likelihood",
        "log_jacobian", "constrain"), [f.name for f in src.functions])
    # `prepare_data` takes `rk_bound_data(rk)`'s columns by the names it
    # lists. A data-free replay binds more (its omitted fit-role columns),
    # which `prepare_data` does not take.
    inputs = Set(Symbol(first(input)) for input in jax_function(src, "prepare_data").inputs)
    @test issubset(inputs, keys(rk_bound_data(rk)))
    # A fitted anchor computed in the program is translated host code, its
    # guards included.
    @test occursin("requires at least one training value", src.text) == anchors
    if rk_jax_available()
        u = point(rk)
        density = prepare_query(rk.model, rk_bound_plan(rk), :sampler)
        r = rk_jax_evaluate(src, "log_density", (u, raw_inputs(rk, src)...);
            grad=true, prepare=true)
        @test r.value ≈ Base.invokelatest(density, u) rtol=1e-12
        @test r.grad ≈ rk_fd_gradient(v -> Base.invokelatest(density, v), u) rtol=1e-6 atol=1e-6
    else
        @test_skip "RK_JAX_PYTHON names no Python with jax"
    end
    src
end

@testset "bound anchors and level codes: the fit and a replay" begin
    rk = RKBRMI(JM.grouped(TRAIN))
    check_module(rk; anchors=false)
    check_module(reprocess(rk, (; x=[1.0, 2.0, 3.0, 4.0], w=[0.1, -0.2, 0.3, 0.0],
        g=["a", "c", "c", "b"], y=[0.2, -0.1, 0.4, 0.0])); anchors=false)
    check_module(RKBRMI(JM.counts(TRAIN)); anchors=false)
end

const PK_DESIGN = (; subject=["s2", "s4"], weight=[72.0, 55.0], dose=[120.0, 100.0],
    t=[[0.25, 1.5, 3.0], [1.0, 2.0]], y=[[1.1, 0.6, 0.3], [0.5, 0.4]])

@testset "an anchor computed in the program is prepare_data host code" begin
    rk = RKBRMI(JM.pk(PK))
    src = check_module(rk; anchors=true)
    @test occursin("requires nonzero sample variance", src.text)
    check_module(reprocess(rk, PK_DESIGN); anchors=true)
end

# `RKBRMI(build(design), brm_fitted_state(rk))` binds the fitted values, not
# the training rows: they are `prepare_data` inputs, which `rk_bound_data`
# holds under the names `src.functions` lists.
@testset "a data-free replay passes its fitted values to prepare_data" begin
    rk = RKBRMI(JM.grouped(TRAIN))
    design = (; x=[1.0, 2.0, 3.0, 4.0], w=[0.1, -0.2, 0.3, 0.0],
        g=["a", "c", "c", "b"], y=[0.2, -0.1, 0.4, 0.0])
    check_module(RKBRMI(JM.grouped(design), brm_fitted_state(rk)); anchors=false)
    pk = RKBRMI(JM.pk(PK))
    check_module(RKBRMI(JM.pk(PK_DESIGN), brm_fitted_state(pk)); anchors=false)
end
