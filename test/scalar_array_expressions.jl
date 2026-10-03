using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using BridgeStan
include(joinpath(@__DIR__, "testset_filter.jl"))

const ARRAY_DATA = (; y=[0.1, -0.2], shift=0.4)
const ARRAY_SCALARS = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    v = [a, b]
    mu = sum(v)
    y ~ Normal(mu, 1.0)
end
const ARRAY_MIXED = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    v = [a, b + shift, 0.5]
    mu = v[1] + sum([b, 0.5]) + v[2]
    y ~ Normal(mu, 1.0)
end
const ARRAY_INLINE = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    y ~ Normal(sum([a, b]), 1.0)
end
const ARRAY_SINGLE = @brm begin
    a ~ Normal(0.0, 1.0)
    v = [a]
    y ~ Normal(v[1], 1.0)
end
const ARRAY_CONSTANT = @brm begin
    v = [0.25, 0.5]
    y ~ Normal(sum(v), 1.0)
end

@stestset "scalar arrays retain ordinary expression semantics" begin
    builders = (ARRAY_SCALARS, ARRAY_MIXED, ARRAY_INLINE, ARRAY_SINGLE, ARRAY_CONSTANT)
    for builder in builders
        brmi = builder(ARRAY_DATA)
        before = deepcopy(ARRAY_DATA)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        code = BayesianRegressionModels.stan_code(sb)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test isequal(ARRAY_DATA, before)
        @test BayesianRegressionModels.stan_code(
            SBBRMI(brmi; mod=@__MODULE__, total_groups=())) == code
    end
    sb = SBBRMI(ARRAY_MIXED(ARRAY_DATA); mod=@__MODULE__, total_groups=())
    @test sb.data[:shift] == ARRAY_DATA.shift
    replay = reprocess(sb, merge(ARRAY_DATA, (; shift=-0.3, y=[-0.1, 0.3])))
    # Existing replay semantics keep scalar data constants and rebind rows.
    @test replay.data[:shift] == sb.data[:shift]
    @test replay.data[:y] == [-0.1, 0.3]
    fresh = SBBRMI(ARRAY_MIXED(merge(ARRAY_DATA, (; shift=-0.3)));
        mod=@__MODULE__, total_groups=())
    @test fresh.data[:shift] == -0.3
    @test BayesianRegressionModels.stan_code(replay) == BayesianRegressionModels.stan_code(sb)
end

@stestset "scalar arrays match independent densities gradients and outputs" begin
    for (builder, weights, offset) in (
        (ARRAY_SCALARS, [1.0, 1.0], 0.0),
        (ARRAY_MIXED, [1.0, 2.0], ARRAY_DATA.shift + 0.5),
    )
        sb = SBBRMI(builder(ARRAY_DATA); mod=@__MODULE__, total_groups=())
        problem = StanBlocks.stan_instantiate(sb.model)
        names = BridgeStan.param_unc_names(problem.model)
        @test Set(names) == Set(["a", "b"])
        full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        descriptor = brm_descriptor(sb)
        for (a, b) in ((0.0, 0.0), (0.3, -0.7), (-1.1, 0.4))
            q = [name == "a" ? a : b for name in names]
            mu = weights[1] * a + weights[2] * b + offset
            expected_lp = logpdf(Normal(), a) + logpdf(Normal(), b) +
                sum(logpdf(Normal(mu, 1.0), y) for y in ARRAY_DATA.y)
            gradient = zeros(length(q))
            lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
                propto=false, jacobian=true)
            expected_gradient = [
                -(name == "a" ? a : b) + weights[name == "a" ? 1 : 2] *
                    sum(ARRAY_DATA.y .- mu) for name in names
            ]
            @test lp ≈ expected_lp atol=1e-12 rtol=1e-12
            @test gradient ≈ expected_gradient atol=1e-12 rtol=1e-12
            full = BridgeStan.param_constrain(problem.model, q;
                include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, 42))
            expected_v = builder === ARRAY_SCALARS ? [a, b] : [a, b + ARRAY_DATA.shift, 0.5]
            @test full[brm_output_coordinates(descriptor, :v, full_names)] ≈ expected_v
            @test only(full[brm_output_coordinates(descriptor, :mu, full_names)]) ≈ mu
        end
    end
end
