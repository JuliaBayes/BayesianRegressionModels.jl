using Test
using BayesianRegressionModels
using Distributions
using LogDensityProblems
using Statistics
using StanBlocks
using Turing

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const SB_TERM_SEMANTICS_CACHE = joinpath(tempdir(), "brm-sb-term-prior-semantics")

asymmetric_simplex_prior(a, b, c) = Dirichlet([a, b, c])
BRM.brm_distribution_type(::typeof(asymmetric_simplex_prior)) = Dirichlet
BRM._sb_stan_dist_name(::typeof(asymmetric_simplex_prior)) = :asymmetric_simplex

StanBlocks.@deffun begin
    @lpxf asymmetric_simplex_lpdf(
        value::simplex[n], a::real, b::real, c::real
    )::real = dirichlet_lpdf(value, [a, b, c])
    asymmetric_simplex_rng(simplex[n], a::real, b::real, c::real)::simplex[n] =
        dirichlet_rng([a, b, c])
end

term_semantics_df = (;
    y = [0.1, -0.2, 0.3, 0.4],
    c = [1, 2, 3, 4],
)

function instantiate_term_semantics(sb)
    code = BRM.stan_code(sb)
    mkpath(SB_TERM_SEMANTICS_CACHE)
    StanBlocks.stan_instantiate(sb.model;
        path=joinpath(SB_TERM_SEMANTICS_CACHE, string(hash(code)) * ".stan"))
end

@testset "StanBlocks monotonic simplex prior semantics" begin
    scalar = @brm term_semantics_df begin
        y ~ Normal(mu, 1)
        mu ~ 0 + mo1(c)
        simplex(mu, mo1(c)) ~ Dirichlet(2)
    end
    scalar_code = BRM.stan_code(SBBRMI(scalar; mod=@__MODULE__))
    @test occursin(
        "mo1_c_simplex_incr ~ dirichlet(rep_vector(2.0, 3));", scalar_code)

    variadic = @brm term_semantics_df begin
        y ~ Normal(mu, 1)
        mu ~ 0 + mo1(c)
        simplex(mu, mo1(c)) ~ Dirichlet(1, 2, 4)
    end
    variadic_code = BRM.stan_code(SBBRMI(variadic; mod=@__MODULE__))
    @test occursin(
        "mo1_c_simplex_incr ~ dirichlet([1.0, 2.0, 4.0]');", variadic_code)
    @test StanBlocks.stanc_check(variadic_code; warn_pedantic=false).ok

    vector_data = merge(term_semantics_df, (; alpha=[1.0, 2.0, 4.0]))
    vector_concentration = @brm vector_data begin
        y ~ Normal(mu, 1)
        mu ~ 1 + mo1(c)
        simplex(mu, mo1(c)) ~ Dirichlet(alpha)
    end
    vector_sb_code = BRM.stan_code(SBBRMI(vector_concentration; mod=@__MODULE__))
    @test occursin("vector[alpha_n] alpha;", vector_sb_code)
    @test occursin("simplex[alpha_n] mo1_c_simplex_incr;", vector_sb_code)
    @test occursin("mo1_c_simplex_incr ~ dirichlet(alpha);", vector_sb_code)
    @test !occursin("dirichlet(rep_vector(alpha", vector_sb_code)
    @test StanBlocks.stanc_check(vector_sb_code; warn_pedantic=false).ok

    vector_turing = TuringBRMI(vector_concentration)
    vector_term = only(only(vector_turing.plan.predictors).terms)
    @test vector_term.state.alpha == vector_data.alpha
    simplex_probe = [0.2, 0.3, 0.5]
    @test Turing.logjoint(
        BRM._brm_turing_term_model(vector_term, length(vector_data.y)),
        (; simplex_incr=simplex_probe),
    ) ≈ logpdf(Dirichlet(vector_data.alpha), simplex_probe)

    fitted = @brm term_semantics_df begin
        y ~ Normal(mu, 1)
        mu ~ 0 + mo1(c)
        simplex(mu, mo1(c)) ~ asymmetric_simplex_prior(1, 2, 4)
    end
    fitted_sb = SBBRMI(fitted; mod=@__MODULE__)
    fitted_code = BRM.stan_code(fitted_sb)
    @test occursin("simplex[3] mo1_c_simplex_incr;", fitted_code)
    @test occursin(
        "mo1_c_simplex_incr ~ asymmetric_simplex(1, 2, 4);", fitted_code)
    @test StanBlocks.stanc_check(fitted_code; warn_pedantic=false).ok

    fitted_problem = instantiate_term_semantics(fitted_sb)
    @test LogDensityProblems.dimension(fitted_problem) == 2
    raw = zeros(2)
    names = BS.param_names(fitted_problem.model; include_tp=true)
    values = BS.param_constrain(fitted_problem.model, raw; include_tp=true)
    constrained = Dict(zip(names, values))
    simplex_value = [constrained["mo1_c_simplex_incr.$i"] for i in 1:3]
    monotonic = [0.0; cumsum(simplex_value)]
    expected = logpdf(Dirichlet([1.0, 2.0, 4.0]), simplex_value) +
        sum(logpdf.(Normal.(monotonic[term_semantics_df.c], 1.0),
                    term_semantics_df.y))
    @test BS.log_density(fitted_problem.model, raw;
        propto=false, jacobian=false) ≈ expected atol=1e-10

    prior = @brm (; c=term_semantics_df.c) begin
        mu ~ 0 + mo1(c)
        simplex(mu, mo1(c)) ~ asymmetric_simplex_prior(1, 2, 4)
    end
    prior_sb = SBBRMI(prior; mod=@__MODULE__)
    prior_code = BRM.stan_code(prior_sb)
    @test occursin("asymmetric_simplex_simplex_rng(3, 1, 2, 4)", prior_code)
    @test occursin(r"parameters\s*\{\s*\}", prior_code)
    @test StanBlocks.stanc_check(prior_code; warn_pedantic=false).ok

    prior_problem = instantiate_term_semantics(prior_sb)
    @test LogDensityProblems.dimension(prior_problem) == 0
    prior_names = BS.param_names(
        prior_problem.model; include_tp=true, include_gq=true)
    simplex_indices = [findfirst(==("mo1_c_simplex_incr.$i"), prior_names)
                       for i in 1:3]
    @test all(!isnothing, simplex_indices)
    simplex_indices = Int[index for index in simplex_indices]
    draws = Matrix{Float64}(undef, 256, 3)
    for draw_index in axes(draws, 1)
        draw = BS.param_constrain(prior_problem.model, Float64[];
            include_tp=true, include_gq=true,
            rng=BS.StanRNG(prior_problem.model, 10_000 + draw_index))
        draws[draw_index, :] = draw[simplex_indices]
    end
    @test all(draws .>= 0)
    @test all(isapprox.(vec(sum(draws; dims=2)), 1.0; atol=1e-12))
    @test vec(mean(draws; dims=1)) ≈ [1, 2, 4] ./ 7 atol=0.04
end
