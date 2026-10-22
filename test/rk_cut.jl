# test/rk_cut.jl — cut posteriors on RK (todo `0fdr8mz`; shape `1bgtn3d`).
#
# A synthetic two-stage model: a per-subject elimination rate (stage 1, PK
# reads) feeds an Emax response (stage 2, PD reads), both computed in one
# `@plate` cell, the fused plan a joint fit uses. Oracles:
# - the coordinate cut's conditional is the joint density with the upstream
#   declarations pinned: equal gradients on the free coordinates, the density
#   off by a constant per draw;
# - an explicit stage-2 model reading the stage-1 values as data is the same
#   conditional (function and value-name interfaces);
# - joint draws take each draw's upstream coordinates from its upstream draw,
#   and the poor cut pairs every upstream draw with one downstream draw.
# Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/rk_cut.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, Statistics, WarmupHMC

module CutModels
using BayesianRegressionModels, Distributions
joint(d) = @brm d begin
    sigma_pk ~ Exponential(1)
    sigma_pd ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    effect(log_k, Intercept) ~ Normal(-1, 0.5)
    sd(:, p) ~ Exponential(1)
    E0 ~ Normal(0, 1)
    log_Emax ~ Normal(0, 1)
    log_EC50 ~ Normal(0, 1)
    @plate for i in eachindex(log_k)
        y_pk[i] ~ Normal(log(dose[i]) .- exp(log_k[i]) .* t_pk[i], sigma_pk)
        c_pd = dose[i] .* exp.(-exp(log_k[i]) .* t_pd[i])
        y_pd[i] ~ Normal(E0 .+ exp(log_Emax) .* c_pd ./ (exp(log_EC50) .+ c_pd), sigma_pd)
    end
end
# Stage 2 alone, on the concentrations at the PD reads (one row per read).
pd_reads(d) = @brm d begin
    sigma_pd ~ Exponential(1)
    E0 ~ Normal(0, 1)
    log_Emax ~ Normal(0, 1)
    log_EC50 ~ Normal(0, 1)
    mu = E0 + exp(log_Emax) * c / (exp(log_EC50) + c)
    y_pd ~ Normal(mu, sigma_pd)
end
# Stage 2 alone, on each subject's elimination rate, as data.
pd_rates(d) = @brm d begin
    sigma_pd ~ Exponential(1)
    E0 ~ Normal(0, 1)
    log_Emax ~ Normal(0, 1)
    log_EC50 ~ Normal(0, 1)
    @plate for i in eachindex(log_k)
        c_pd = dose[i] .* exp.(-exp(log_k[i]) .* t_pd[i])
        y_pd[i] ~ Normal(E0 .+ exp(log_Emax) .* c_pd ./ (exp(log_EC50) .+ c_pd), sigma_pd)
    end
end
# A model sharing no declaration with the joint one.
unrelated(d) = @brm d begin
    s_other ~ Exponential(1)
    z ~ Normal(0, s_other)
end
end

const AD = AutoEnzyme(; mode=Enzyme.Reverse)
const N = 6
const RNG = Xoshiro(20261022)
const T_PK = [sort(rand(RNG, 4)) .* 5 for _ in 1:N]
const T_PD = [sort(rand(RNG, 5)) .* 5 for _ in 1:N]
const DOSE = fill(10.0, N)
const K_TRUE = exp.(-1 .+ 0.3 .* randn(RNG, N))
const Y_PK = [log(DOSE[i]) .- K_TRUE[i] .* T_PK[i] .+ 0.1 .* randn(RNG, 4) for i in 1:N]
truth(i) = DOSE[i] .* exp.(-K_TRUE[i] .* T_PD[i])
const Y_PD = [0.5 .+ 2.0 .* truth(i) ./ (1.0 .+ truth(i)) .+ 0.2 .* randn(RNG, 5) for i in 1:N]
const DATA = (; subject=collect(1:N), dose=DOSE, t_pk=T_PK, t_pd=T_PD, y_pk=Y_PK, y_pd=Y_PD)
# Stage 1: the joint model without the downstream response.
const STAGE1 = Base.structdiff(DATA, NamedTuple{(:y_pd,)})

const JOINT = RKBRMI(CutModels.joint(DATA))
const UP = RKBRMI(CutModels.joint(STAGE1))
const JOINT_NAMES = BRM._rk_layout_coordinate_names(JOINT)
const UP_NAMES = BRM._rk_layout_coordinate_names(UP)
const JOINT_PROBLEM = rk_logdensity_problem(JOINT; ad_backend=AD)
# Upstream draws: points of the upstream model's coordinates.
const U = 0.3 .* randn(RNG, length(UP_NAMES), 3)

# Joint coordinates of a downstream point `x` given upstream draw `u`.
function joint_point(family, u, x)
    [from === :upstream ? u[at] : x[at] for (from, at) in family.joint_sources]
end

@stestset "the upstream model samples the upstream part" begin
    @test Set(UP_NAMES) == Set(setdiff(JOINT_NAMES, [:sigma_pd, :E0, :log_Emax, :log_EC50]))
end

const COORD = brm_cut_family(JOINT, UP, U; ad_backend=AD)

@stestset "the coordinate cut is the joint density with the upstream pinned" begin
    @test COORD.names == [:sigma_pd, :E0, :log_Emax, :log_EC50]
    @test COORD.joint_names == JOINT_NAMES
    @test length(COORD) == size(U, 2)
    for k in 1:length(COORD)
        offsets = map(1:2) do _
            x = 0.4 .* randn(RNG, length(COORD.names))
            v, g = LogDensityProblems.logdensity_and_gradient(COORD.targets[k], x)
            vj, gj = LogDensityProblems.logdensity_and_gradient(JOINT_PROBLEM,
                joint_point(COORD, U[:, k], x))
            free = [findfirst(==(n), JOINT_NAMES) for n in COORD.names]
            @test g ≈ gj[free] atol=1e-10 rtol=1e-10
            v - vj
        end
        @test offsets[1] ≈ offsets[2] atol=1e-10
    end
    @test COORD.features[2] == U[:, 2]
    @test all(iszero, COORD.starts[1])
end

# Stage-1 values at a draw: each subject's elimination rate, and the
# concentrations at the PD reads.
rates(u) = only(rk_draw_values(UP, reshape(u, :, 1), (:log_k,)).log_k)
reads(u) = (; c=reduce(vcat, [DOSE[i] .* exp.(-exp(rates(u)[i]) .* T_PD[i]) for i in 1:N]))

@stestset "an explicit stage-2 model is the same conditional" begin
    reads_data = (; y_pd=reduce(vcat, Y_PD))
    reads_down = RKBRMI(CutModels.pd_reads(merge(reads_data, reads(U[:, 1]))))
    by_function = brm_cut_family(reads, reads_down, reads_data, UP, U;
        ad_backend=AD, joint=JOINT)
    rates_data = (; dose=DOSE, t_pd=T_PD, y_pd=Y_PD)
    rates_down = RKBRMI(CutModels.pd_rates(merge(rates_data, (; log_k=rates(U[:, 1])))))
    by_name = brm_cut_family((:log_k,), rates_down, rates_data, UP, U;
        ad_backend=AD, joint=JOINT)
    for family in (by_function, by_name)
        @test Set(family.names) == Set(COORD.names)
        @test family.joint_names == JOINT_NAMES
        order = [findfirst(==(n), family.names) for n in COORD.names]
        for k in 1:length(COORD)
            offsets = map(1:2) do _
                x = 0.4 .* randn(RNG, length(COORD.names))
                v, g = LogDensityProblems.logdensity_and_gradient(COORD.targets[k], x)
                y = zeros(length(x)); y[order] = x
                ve, ge = LogDensityProblems.logdensity_and_gradient(family.targets[k], y)
                @test ge[order] ≈ g atol=1e-10 rtol=1e-10
                @test joint_point(family, U[:, k], y) ≈ joint_point(COORD, U[:, k], x)
                v - ve
            end
            @test offsets[1] ≈ offsets[2] atol=1e-10
        end
    end
end

@stestset "cut draws are placed in the joint coordinates" begin
    draws = [fill(Float64(k), length(COORD.names), k + 1) for k in 1:length(COORD)]
    fit = brm_cut_fit((targets, starts, features) -> draws, COORD)
    @test fit.downstream == draws
    @test fit.source == [1, 1, 2, 2, 2, 3, 3, 3, 3]
    @test fit.joint_names == JOINT_NAMES
    for (j, k) in enumerate(fit.source), (i, (from, at)) in enumerate(COORD.joint_sources)
        @test fit.joint[i, j] == (from === :upstream ? U[at, k] : Float64(k))
    end
    # refused: a result per conditional, each in the conditionals' coordinates.
    @test_throws "returned 2 results for 3 conditionals" brm_cut_fit((t, s, f) -> draws[1:2], COORD)
    @test_throws "is not a 4 × draws matrix" brm_cut_fit((t, s, f) -> [zeros(3, 2) for _ in t], COORD)
end

@stestset "the poor cut collapses the upstream draws to their median" begin
    point = brm_cut_point(U)
    @test point == reshape([median(U[i, :]) for i in axes(U, 1)], :, 1)
    poor = brm_cut_family(JOINT, UP, point; ad_backend=AD)
    @test length(poor) == 1
    fit = brm_cut_fit((t, s, f) -> [reshape(collect(1.0:20.0), 4, 5)], poor)
    out = brm_cut_propagate(fit, poor, U; rng=Xoshiro(1))
    @test size(out.joint) == (length(JOINT_NAMES), size(U, 2))
    @test allunique(out.pairing)
    for k in axes(U, 2), (i, (from, at)) in enumerate(poor.joint_sources)
        @test out.joint[i, k] == (from === :upstream ? U[at, k] : fit.downstream[1][at, out.pairing[k]])
    end
    # refused: the poor cut's family is one conditional.
    @test_throws "has one conditional" brm_cut_propagate(fit, COORD, U)
end

@stestset "cut draws post-process on one value kernel" begin
    values = rk_draw_values(UP, U, (:log_k,))
    @test length(values.log_k) == size(U, 2)
    @test values.log_k[2] == rates(U[:, 2])
end

@stestset "refusals name the fix" begin
    # refused: the draws are columns in the upstream model's coordinates.
    @test_throws "the model has $(length(UP_NAMES)) coordinates" brm_cut_family(
        JOINT, UP, U[1:end-1, :]; ad_backend=AD)
    # refused: an upstream declaration the joint model lacks cannot be pinned.
    other = RKBRMI(CutModels.unrelated((; z=[0.1, -0.2])))
    @test_throws "`s_other` is not a declaration of the joint model" brm_cut_family(
        JOINT, other, zeros(1, 1); ad_backend=AD)
end

@stestset "a cut fits with independent fits per conditional" begin
    function independent(targets, starts, features)
        results = adaptive_warmup_mcmc([Xoshiro(k) for k in eachindex(targets)],
            targets; n_draws=100)
        [r.posterior_position for r in results]
    end
    fit = brm_cut_fit(independent, COORD)
    @test length(fit.downstream) == length(COORD)
    @test all(D -> size(D, 1) == length(COORD.names) && all(isfinite, D), fit.downstream)
    @test size(fit.joint, 1) == length(JOINT_NAMES)
end
