# test/mvnormal_vector_parameters.jl — vector-valued parameters via `x ~ MvNormal(...)`.
#
# Run: julia --project=test test/mvnormal_vector_parameters.jl
# Set BRM_MVNORMAL_RUNTIME=0 to skip the BridgeStan density/gradient probes.
#
# Decision `187g4va` (2026-09-16): a non-data LHS with an `MvNormal` RHS declares a Stan
# `vector[n]` parameter with Distributions.jl's constructor semantics. Before this seam the
# statement parsed but died in sbimpl with "distribution `MvNormal` has no Stan translation",
# and the only way to get a vector latent (a random-walk's innovations, per-patch seeds) was a
# one-cell `kernel(...)` with a dummy grouping ranef (measured in research/epi_renewal/).

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using LinearAlgebra: Diagonal, I
using Distributions: MvNormal, Normal
import BayesianRegressionModels as BRM

const MVN_CACHE = joinpath(tempdir(), "brm-mvnormal-vector-parameters")
const MVN_RUNTIME = get(ENV, "BRM_MVNORMAL_RUNTIME", "1") != "0"

mvn_df() = (;
    time      = collect(1.0:8),
    y         = [0.4, 1.1, 0.9, 1.6, 1.2, 2.0, 1.8, 2.5],
    seed_mean = [log(50.0), log(0.05), log(0.05)],
)

code_of(sb) = StanBlocks.stan_code(sb.model)

function stanc_accepts(model)
    result = StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic=false)
    result.ok || @error "stanc rejected MvNormal vector-parameter model" output=result.output
    result.ok
end

function instantiate(model)
    isdir(MVN_CACHE) || mkpath(MVN_CACHE)
    code = StanBlocks.stan_code(model)
    StanBlocks.stan_instantiate(model; path=joinpath(MVN_CACHE, string(hash(code)) * ".stan"))
end

fixed_q(dim) = [0.1 * ((i % 5) - 2) for i in 1:dim]

# A top-level `@brm` assignment is Julia-evaluated, so Stan builtins reach it through a
# `@deffun` (a Julia binding + a Stan function), exactly as any other model function does.
StanBlocks.@deffun begin
    rw_path(init::real, sig::real, eps::vector[K])::vector[K + 1] =
        append_row(init, init + sig * cumulative_sum(eps))
end

# A parameter nothing reads is lowered to a generated quantity by StanBlocks' activity
# analysis (by design), so every case below CONSUMES its vector.
param_block(code) = code[findfirst("parameters {", code)[1]:findfirst("model {", code)[1]]

function probe(model, expected_dim)
    MVN_RUNTIME || return true
    problem = instantiate(model)
    dim = LogDensityProblems.dimension(problem)
    lp, g = LogDensityProblems.logdensity_and_gradient(problem, fixed_q(dim))
    dim == expected_dim && isfinite(lp) && all(isfinite, g)
end

# a random walk whose innovations are a top-level vector parameter: no kernel, no dummy ranef
rw_builder(df) = @brm df begin
    sig  ~ Normal(0.0, 0.05; lower=0.0)
    init ~ Normal(0.0, 1.0)
    eps  ~ MvNormal(zeros(length(time) - 1), 1.0)
    walk = rw_path(init, sig, eps)
    y ~ Normal(walk, 0.3)
end

@testset "MvNormal vector parameters" begin
    df = mvn_df()

    @testset "data-only mean expression + literal std, used by a top-level assignment" begin
        sb = SBBRMI(rw_builder(df); mod=@__MODULE__)
        code = code_of(sb)
        @test occursin(r"vector\[\w+\] eps;", param_block(code))
        @test occursin("eps ~ normal(eps_mu, 1.0)", code) || occursin("eps ~ normal(eps_mu, 1)", code)
        @test occursin("walk = rw_path(init, sig, eps)", code)
        @test sb.data[:eps_n] == 7
        @test sb.data[:eps_mu] == zeros(7)
        @test stanc_accepts(sb.model)
        @test probe(sb.model, 2 + 7)                # sig, init, eps[1:7]
    end

    @testset "data column mean + literal std (per-patch seeds)" begin
        sb = SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            lI0 ~ MvNormal(seed_mean, 0.5)
            y ~ Normal(sig + sum(lI0), 1.0)
        end); mod=@__MODULE__)
        code = code_of(sb)
        @test occursin(r"vector\[\w+\] lI0;", param_block(code))
        @test occursin("lI0 ~ normal(seed_mean, 0.5)", code)
        @test sb.data[:lI0_n] == 3
        @test !haskey(sb.data, :lI0_mu)              # the column is referenced by its own name
        @test stanc_accepts(sb.model)
        @test probe(sb.model, 1 + 3)
    end

    @testset "sampled scalar std (isotropic, parameter-bearing scale)" begin
        sb = SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            eps ~ MvNormal(zeros(4), sig)
            y ~ Normal(sig + sum(eps), 1.0)
        end); mod=@__MODULE__)
        @test occursin("eps ~ normal(eps_mu, sig)", code_of(sb))
        @test stanc_accepts(sb.model)
        @test probe(sb.model, 1 + 4)
    end

    @testset "integer dimension, zero mean" begin
        sb = SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            eps ~ MvNormal(5, 1.0)
            y ~ Normal(sig + sum(eps), 1.0)
        end); mod=@__MODULE__)
        code = code_of(sb)
        @test occursin(r"vector\[\w+\] eps;", param_block(code))
        @test occursin("rep_vector(0.0, eps_n)", code) || occursin("rep_vector(0, eps_n)", code)
        @test sb.data[:eps_n] == 5
        @test stanc_accepts(sb.model)
        @test probe(sb.model, 1 + 5)
    end

    @testset "full covariance -> multi_normal" begin
        sb = SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            v ~ MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0])
            y ~ Normal(sig + sum(v), 1.0)
        end); mod=@__MODULE__)
        code = code_of(sb)
        @test occursin("v ~ multi_normal(v_mu, v_scale)", code)
        @test sb.data[:v_scale] == [1.0 0.5; 0.5 1.0]
        @test stanc_accepts(sb.model)
        @test probe(sb.model, 1 + 2)
    end

    @testset "zero-mean covariance form MvNormal(Σ)" begin
        sb = SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            v ~ MvNormal([2.0 0.0; 0.0 3.0])
            y ~ Normal(sig + sum(v), 1.0)
        end); mod=@__MODULE__)
        code = code_of(sb)
        @test occursin("v ~ multi_normal(rep_vector(0.0, v_n), v_scale)", code) ||
              occursin("v ~ multi_normal(rep_vector(0, v_n), v_scale)", code)
        @test sb.data[:v_n] == 2
        @test stanc_accepts(sb.model)
    end

    @testset "Diagonal(variances) -> vectorised normal with standard deviations" begin
        sb = SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            v ~ MvNormal([0.0, 1.0], Diagonal([4.0, 9.0]))
            y ~ Normal(sig + sum(v), 1.0)
        end); mod=@__MODULE__)
        @test occursin("v ~ normal(v_mu, v_scale)", code_of(sb))
        @test sb.data[:v_scale] == [2.0, 3.0]
        @test stanc_accepts(sb.model)
    end

    @testset "vector of standard deviations" begin
        sb = SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            v ~ MvNormal([0.0, 1.0], [0.5, 2.0])
            y ~ Normal(sig + sum(v), 1.0)
        end); mod=@__MODULE__)
        @test occursin("v ~ normal(v_mu, v_scale)", code_of(sb))
        @test sb.data[:v_scale] == [0.5, 2.0]
        @test stanc_accepts(sb.model)
    end

    @testset "refusals are loud and specific" begin
        # parameter-bearing mean with a scalar scale: no data-determinable size
        @test_throws ErrorException SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            v ~ MvNormal(sig, 1.0)
            y ~ Normal(sig, 1.0)
        end); mod=@__MODULE__)
        err = try
            SBBRMI((@brm df begin
                sig ~ Normal(0.0, 1.0; lower=0.0)
                v ~ MvNormal(sig, 1.0)
                y ~ Normal(sig, 1.0)
            end); mod=@__MODULE__)
            nothing
        catch e
            sprint(showerror, e)
        end
        @test occursin("determinable from data", err)
        # mean / scale size mismatch
        @test_throws ErrorException SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            v ~ MvNormal(zeros(3), [1.0 0.0; 0.0 1.0])
            y ~ Normal(sig, 1.0)
        end); mod=@__MODULE__)
        # keywords (bounds) are not a thing on a multivariate normal parameter
        @test_throws ErrorException SBBRMI((@brm df begin
            sig ~ Normal(0.0, 1.0; lower=0.0)
            v ~ MvNormal(zeros(3), 1.0; lower=0.0)
            y ~ Normal(sig, 1.0)
        end); mod=@__MODULE__)
    end
end
