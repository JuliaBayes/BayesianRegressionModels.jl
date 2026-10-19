# Weighted observations whose response column is omitted, and StatsBase weight
# constructors inside `@plate for` cells.
#
#   julia --startup-file=no --project=test test/weighted_omitted_and_cell_weights.jl

using Test
using BayesianRegressionModels, StanBlocks, Distributions
using BridgeStan, LogDensityProblems
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels

omit(data, keys::Symbol...) = Base.structdiff(data, NamedTuple{keys})
stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
rk_problem(brmi) = rk_logdensity_problem(RKBRMI(brmi);
    ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

const flat = (; x=[1.0, 2.0, 0.5, 1.5], y=[0.1, 0.2, 0.3, 0.5],
    z=[0.0, 0.4, 0.2, 0.9], w=[1.0, 1.0, 0.0, 2.0], n=[1, 3, 2, 1],
    k=[2.0, 3.0, 1.0, 4.0])

weighted_builder(kind) =
    kind === :power ? (@brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ weighted(Normal(mu, sigma), weights(w))
    end) : kind === :frequency ? (@brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ weighted(Normal(mu, sigma), fweights(n))
    end) : (@brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ weighted(Normal(mu, sigma), aweights(k))
    end)

# The emitted call that draws `y`, wherever it sits in the program.
predictive_law(kind) = kind === :analytic ? "(sigma ./ sqrt(brm_weight_y))" :
    "brm_weight_y, mu, sigma)"

@testset "an omitted weighted response forward-simulates its fitted predictive law" begin
    for kind in (:power, :frequency, :analytic)
        builder = weighted_builder(kind)
        omitted = SBBRMI(builder(omit(flat, :y, :z)); mod=@__MODULE__)
        bound = SBBRMI(builder(omit(flat, :z)); mod=@__MODULE__)
        code = BRM.stan_code(omitted)
        @test stanc_ok(omitted)
        @test occursin(predictive_law(kind), code)
        # The fitted program draws its `y_gen` from that same law.
        @test occursin(predictive_law(kind), BRM.stan_code(bound))
        kind === :analytic || @test occursin("weighted_vector_normal_rng(", code)
        @test omitted.data[:brm_weight_y] == Float64.(getproperty(flat,
            kind === :power ? :w : kind === :frequency ? :n : :k))
        y = only(d for d in generative_plan(omitted).declarations if d.target === :y)
        @test y.role === :observation
        @test isnothing(y.data_source)
        descriptor = brm_descriptor(omitted)
        @test brm_output(descriptor, :y; role=:posterior_predictive).name === :y_gen
        @test [o.name for o in descriptor.outputs if o.logical === :y] == [:y_gen]
        @test RKBRMI(builder(omit(flat, :y, :z))) isa RKBRMI
    end
end

@testset "an omitted analytic draw uses the precision-scaled Normal" begin
    # Two rows share x, so mu cancels: Var(y1 - y2) = 0.5^2 (1/k1 + 1/k2).
    # The unweighted base law would give 0.5^2 * 2.
    twin = @brm begin
        mu ~ 1 + x
        y ~ weighted(Normal(mu, 0.5), aweights(k))
    end
    sb = SBBRMI(twin((; x=[0.3, 0.3], k=[1.0, 100.0])); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    first_row, second_row = findfirst(==("y_gen.1"), names), findfirst(==("y_gen.2"), names)
    rng = BridgeStan.StanRNG(problem.model, 7)
    u = zeros(LogDensityProblems.dimension(problem))
    differences = map(1:20_000) do _
        draw = BridgeStan.param_constrain(problem.model, u;
            include_tp=true, include_gq=true, rng)
        draw[first_row] - draw[second_row]
    end
    mean_difference = sum(differences) / length(differences)
    variance = sum(abs2, differences .- mean_difference) / (length(differences) - 1)
    @test variance ≈ 0.25 * (1 + 1 / 100) rtol=0.05
end

@testset "a fitted response beside an omitted weighted one" begin
    mixed(kind) = kind === :plain ? (@brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        nu ~ 1 + x
        z ~ Normal(nu, 1.0)
        y ~ Normal(mu, sigma)
    end) : kind === :power ? (@brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        nu ~ 1 + x
        z ~ Normal(nu, 1.0)
        y ~ weighted(Normal(mu, sigma), weights(w))
    end) : (@brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        nu ~ 1 + x
        z ~ Normal(nu, 1.0)
        y ~ weighted(Normal(mu, sigma), aweights(k))
    end)
    dimension(kind) = LogDensityProblems.dimension(
        StanBlocks.stan_instantiate(SBBRMI(mixed(kind)(omit(flat, :y));
            mod=@__MODULE__).model))
    for kind in (:power, :analytic)
        sb = SBBRMI(mixed(kind)(omit(flat, :y)); mod=@__MODULE__)
        @test stanc_ok(sb)
        @test occursin(predictive_law(kind), BRM.stan_code(sb))
        # Only `z` conditions the fit, exactly as with an unweighted `y`.
        @test dimension(kind) == dimension(:plain)
        @test brm_output(brm_descriptor(sb), :y; role=:posterior_predictive).name === :y_gen
        rk = RKBRMI(mixed(kind)(omit(flat, :y)))
        @test coordinate_names(rk.model.layout) ==
              coordinate_names(RKBRMI(mixed(:plain)(omit(flat, :y))).model.layout)
    end
end

@testset "reprocess rebinds the weights of an omitted response" begin
    sb = SBBRMI(weighted_builder(:analytic)(omit(flat, :y, :z)); mod=@__MODULE__)
    replay = reprocess(sb, (; x=[0.0, 1.0], k=[5.0, 6.0]))
    @test replay.data[:brm_weight_y] == [5.0, 6.0]
    @test_throws ErrorException reprocess(sb, (; x=[0.0, 1.0], k=[5.0, -1.0]))
end

const cell = (; x=[[1.0, 2.0], [0.5, 1.5, -0.4]], y=[[0.1, 0.2], [0.3, 0.5, -0.1]],
    w=[[1.0, 0.0], [1.0, 1.0, 0.5]], n=[[1.0, 3.0], [2.0, 1.0, 1.0]],
    k=[[2.0, 3.0], [1.0, 4.0, 2.0]])

# Each typed-weight cell next to the spelling it must equal.
cell_builder(kind) =
    kind === :analytic ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(Normal(m, sigma), aweights(k[i]))
        end
    end) : kind === :analytic_spelled ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ Normal(m, sigma ./ sqrt.(k[i]))
        end
    end) : kind === :power ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(Normal(m, sigma), weights(w[i]))
        end
    end) : kind === :power_bare ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(Normal(m, sigma), w[i])
        end
    end) : kind === :frequency ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(Normal(m, sigma), fweights(n[i]))
        end
    end) : kind === :frequency_bare ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(Normal(m, sigma), n[i])
        end
    end) : kind === :token_analytic ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(normal, aweights(k[i]), m, sigma)
        end
    end) : kind === :token_power ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(normal, weights(w[i]), m, sigma)
        end
    end) : kind === :token_power_bare ? (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(normal, w[i], m, sigma)
        end
    end) : (@brm begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = s .* x[i]
            y[i] ~ weighted(Normal(m, sigma), pweights(k[i]))
        end
    end)

@testset "typed weights inside a cell keep their formula-level meaning" begin
    for (typed, spelled) in ((:analytic, :analytic_spelled), (:power, :power_bare),
                             (:frequency, :frequency_bare),
                             (:token_analytic, :analytic_spelled),
                             (:token_power, :token_power_bare))
        @test BRM.stan_code(SBBRMI(cell_builder(typed)(cell); mod=@__MODULE__)) ==
              BRM.stan_code(SBBRMI(cell_builder(spelled)(cell); mod=@__MODULE__))
        typed_problem = rk_problem(cell_builder(typed)(cell))
        spelled_problem = rk_problem(cell_builder(spelled)(cell))
        for u in ([0.3, log(0.7)], [-0.2, 0.1])
            typed_value, typed_gradient =
                LogDensityProblems.logdensity_and_gradient(typed_problem, u)
            spelled_value, spelled_gradient =
                LogDensityProblems.logdensity_and_gradient(spelled_problem, u)
            @test typed_value ≈ spelled_value atol=1e-12
            @test typed_gradient ≈ spelled_gradient atol=1e-12
        end
    end
    # RK used to read `aweights(k[i])` as a power weight `k[i]`.
    problem = rk_problem(cell_builder(:analytic)(cell))
    u = [0.3, log(0.7)]
    s, sigma = u[1], exp(u[2])
    prior = logpdf(Normal(0, 1), s) + logpdf(Exponential(1), sigma) + u[2]
    analytic = sum(logpdf(Normal(s * x, sigma / sqrt(k)), y)
        for (xs, ys, ks) in zip(cell.x, cell.y, cell.k) for (x, y, k) in zip(xs, ys, ks))
    @test LogDensityProblems.logdensity(problem, u) ≈ prior + analytic atol=1e-12

    omitted = omit(cell, :y)
    sb = SBBRMI(cell_builder(:analytic)(omitted); mod=@__MODULE__)
    @test stanc_ok(sb)
    @test occursin("sqrt(", BRM.stan_code(sb))
    @test RKBRMI(cell_builder(:analytic)(omitted)) isa RKBRMI
end

@testset "sampling weights inside a cell" begin
    # refused: sampling weights are not likelihood weights, as at formula level
    # (`_brm_observation_weight_plan`; brm-use weights-and-families).
    @test_throws r"sampling-weight semantics" SBBRMI(cell_builder(:pweights)(cell);
        mod=@__MODULE__)
    @test_throws r"sampling-weight semantics" RKBRMI(cell_builder(:pweights)(cell))
end
