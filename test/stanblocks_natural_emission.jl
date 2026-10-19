# Focused regression for natural Stan emission from the StanBlocks backend.
#
# Run: julia --startup-file=no --project=test test/stanblocks_natural_emission.jl

using Test
using BayesianRegressionModels
using Distributions: Cauchy, Exponential, Normal, Uniform
using LogDensityProblems
using Statistics
using StanBlocks

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const NATURAL_EMISSION_CACHE = joinpath(tempdir(), "brm-natural-emission")

natural_df = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    g = [1, 1, 2, 2, 3, 3],
    y = [-1.4, -0.8, -0.1, 0.4, 1.0, 1.7],
)

@testset "native matrix conversion and homogeneous ranef SD prior" begin
    homogeneous = @brm natural_df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(homogeneous; mod=@__MODULE__)
    code = BRM.stan_code(sb)

    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test !occursin("matrix reshape(", code)
    @test !occursin(" = reshape(", code)
    @test occursin(
        "b_p_g_z = to_matrix(b_p_g_z_flat, n_terms_p_g, n_g);", code)
    @test occursin("vector<lower=0.0>[n_terms_p_g] b_p_g_tau;", code)
    @test occursin("b_p_g_tau ~ exponential((1.0 ./ 2));", code)
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", code)

    block = only(ranef_blocks(sb))
    @test (block.binding, block.n_terms, block.n_groups) == (:b_p_g, 2, 3)

    heterogeneous = @brm natural_df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        sd(mu, p, x) ~ Normal(0, 0.5)
        y ~ Normal(mu, sigma)
    end
    heterogeneous_code = BRM.stan_code(SBBRMI(heterogeneous; mod=@__MODULE__))
    @test StanBlocks.stanc_check(heterogeneous_code; warn_pedantic=false).ok
    # Differing builtin families are native per-family statements over one
    # `vector<lower=0.0>` declaration (StanBlocks `vcat`), not a generated family.
    @test occursin("vector<lower=0.0>[n_terms_p_g] b_p_g_tau;", heterogeneous_code)
    @test occursin("b_p_g_tau[1] ~ exponential(0.5);", heterogeneous_code)
    @test occursin("b_p_g_tau[2] ~ normal(0.0, 0.5);", heterogeneous_code)
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", heterogeneous_code)
    @test !occursin("negative_infinity()", heterogeneous_code)

    # One family with margin-specific hyperparameters stays one native
    # vectorised statement: differing hyperparameters become a vector literal.
    same_family = @brm natural_df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        sd(mu, p, x) ~ Exponential(0.5)
        y ~ Normal(mu, sigma)
    end
    same_family_code = BRM.stan_code(SBBRMI(same_family; mod=@__MODULE__))
    @test StanBlocks.stanc_check(same_family_code; warn_pedantic=false).ok
    @test occursin("vector<lower=0.0>[n_terms_p_g] b_p_g_tau;", same_family_code)
    @test occursin("b_p_g_tau ~ exponential([0.5, 2.0]');", same_family_code)
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", same_family_code)

    # A hyperparameter every margin shares stays scalar.
    shared_location = @brm natural_df begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Normal(0, 1)
        sd(mu, p, x) ~ Normal(0, 0.5)
        y ~ Normal(mu, sigma)
    end
    shared_location_code = BRM.stan_code(SBBRMI(shared_location; mod=@__MODULE__))
    @test StanBlocks.stanc_check(shared_location_code; warn_pedantic=false).ok
    @test occursin("b_p_g_tau ~ normal(0.0, [1.0, 0.5]');", shared_location_code)
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", shared_location_code)
end

# Reference family spelling the previous homogeneous lowering: one scalar
# Exponential kernel per vector coordinate. It is used only to compare the
# normalized density and unconstrained coordinate map against Stan's native
# vectorized Exponential statement.
function brm_test_iid_exponential end
function brm_test_iid_normal end
StanBlocks.@deffun begin
    @lhs @lpxf brm_test_iid_exponential_lpdf(
        x::vector[n], rate::real)::real = begin
        lp = 0.0
        for i in 1:n
            lp += exponential_lpdf(x[i], rate)::real
        end
        lp
    end
    @lhs @lpxf brm_test_iid_normal_lpdf(
        x::vector[n], location::real, scale::real)::real = begin
        lp = 0.0
        for i in 1:n
            lp += normal_lpdf(x[i], location, scale)::real
        end
        lp
    end
end

@testset "homogeneous vector density and prior RNG preserve semantics" begin
    direct = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ exponential(0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    coordinatewise = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ brm_test_iid_exponential(0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    direct_code = StanBlocks.stan_code(direct)
    coordinatewise_code = StanBlocks.stan_code(coordinatewise)
    @test StanBlocks.stanc_check(direct_code; warn_pedantic=false).ok
    @test StanBlocks.stanc_check(coordinatewise_code; warn_pedantic=false).ok

    mkpath(NATURAL_EMISSION_CACHE)
    direct_problem = StanBlocks.stan_instantiate(
        direct; path=joinpath(NATURAL_EMISSION_CACHE, "direct.stan"))
    coordinatewise_problem = StanBlocks.stan_instantiate(
        coordinatewise;
        path=joinpath(NATURAL_EMISSION_CACHE, "coordinatewise.stan"))
    @test LogDensityProblems.dimension(direct_problem) == 2
    @test LogDensityProblems.dimension(coordinatewise_problem) == 2
    @test BS.param_unc_names(direct_problem.model) ==
          BS.param_unc_names(coordinatewise_problem.model) ==
          ["theta.1", "theta.2"]
    for q in ([-0.7, 0.4], [0.0, 0.0], [0.8, -0.2])
        @test BS.log_density(direct_problem.model, q;
            propto=false, jacobian=true) ≈
              BS.log_density(coordinatewise_problem.model, q;
                  propto=false, jacobian=true) atol=1e-12
    end

    # A lower-constrained Normal sampling statement is the same unnormalised
    # positive-support kernel as the former coordinate-wise helper. In
    # particular, native emission must not turn it into a truncated density
    # with a parameter-dependent normalising constant.
    direct_normal = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ normal(0.1, 0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    coordinatewise_normal = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta ~ brm_test_iid_normal(0.1, 0.5; n=2, lower=0.0)
        y ~ normal(theta, 1.0)
    end
    direct_normal_problem = StanBlocks.stan_instantiate(
        direct_normal; path=joinpath(NATURAL_EMISSION_CACHE, "direct-normal.stan"))
    coordinatewise_normal_problem = StanBlocks.stan_instantiate(
        coordinatewise_normal;
        path=joinpath(NATURAL_EMISSION_CACHE, "coordinatewise-normal.stan"))
    for q in ([-0.7, 0.4], [0.0, 0.0], [0.8, -0.2])
        @test BS.log_density(direct_normal_problem.model, q;
            propto=false, jacobian=true) ≈
              BS.log_density(coordinatewise_normal_problem.model, q;
                  propto=false, jacobian=true) atol=1e-12
    end

    # Prior draws keep the observation statement and omit its response column.
    prior = @brm (; x=natural_df.x, g=natural_df.g) begin
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        y ~ Normal(mu, 1)
    end
    prior_sb = SBBRMI(prior; mod=@__MODULE__)
    prior_code = BRM.stan_code(prior_sb)
    @test StanBlocks.stanc_check(prior_code; warn_pedantic=false).ok
    @test occursin(
        "b_p_g_tau = exponential_vector_rng(n_terms_p_g, (1.0 ./ 2));",
        prior_code)
    @test occursin(r"parameters\s*\{\s*\}", prior_code)

    prior_problem = StanBlocks.stan_instantiate(
        prior_sb.model; path=joinpath(NATURAL_EMISSION_CACHE, "prior.stan"))
    @test LogDensityProblems.dimension(prior_problem) == 0
    names = BS.param_names(prior_problem.model; include_tp=true, include_gq=true)
    tau_indices = Int[findfirst(==("b_p_g_tau.$i"), names) for i in 1:2]
    draws = Matrix{Float64}(undef, 256, 2)
    for draw_index in axes(draws, 1)
        values = BS.param_constrain(
            prior_problem.model, Float64[];
            include_tp=true, include_gq=true,
            rng=BS.StanRNG(prior_problem.model, 20_000 + draw_index))
        draws[draw_index, :] = values[tau_indices]
    end
    @test all(draws .>= 0.0)
    @test vec(mean(draws; dims=1)) ≈ [2.0, 2.0] atol=0.3
end

# Draw `b_p_g_tau` from a response-free SBBRMI program's prior predictive.
function natural_prior_tau_draws(sb, label; n_draws=256)
    problem = StanBlocks.stan_instantiate(
        sb.model; path=joinpath(NATURAL_EMISSION_CACHE, "$label.stan"))
    @test LogDensityProblems.dimension(problem) == 0
    names = BS.param_names(problem.model; include_tp=true, include_gq=true)
    tau_indices = Int[findfirst(==("b_p_g_tau.$i"), names) for i in 1:2]
    draws = Matrix{Float64}(undef, n_draws, 2)
    for draw_index in axes(draws, 1)
        values = BS.param_constrain(
            problem.model, Float64[];
            include_tp=true, include_gq=true,
            rng=BS.StanRNG(problem.model, 30_000 + draw_index))
        draws[draw_index, :] = values[tau_indices]
    end
    draws
end

@testset "same-family margins keep the generated family's law" begin
    base = StanBlocks.@slic (; y=[0.2, -0.1]) begin
        theta::vector[2]
        y ~ normal(theta, 1.0)
    end
    # (margins, positive): positive scales and unrestricted coefficients.
    cases = (
        exponential=([BRM.ExprColumn(Exponential, 2.0), BRM.ExprColumn(Exponential, 0.5)], true),
        normal=([BRM.ExprColumn(Normal, 0.1, 1.0), BRM.ExprColumn(Normal, 0.1, 0.5)], true),
        cauchy=([BRM.ExprColumn(Cauchy, 0.0, 1.0), BRM.ExprColumn(Cauchy, 0.0, 2.0)], false),
        uniform=([BRM.ExprColumn(Uniform, 0.2, 0.8), BRM.ExprColumn(Uniform, 0.2, 0.8)], false),
    )
    mkpath(NATURAL_EMISSION_CACHE)
    for (label, (priors, positive)) in pairs(cases)
        native = BRM._sb_native_vector_prior(:theta, priors, 2; positive)
        @test !isnothing(native)
        family, args = BRM._sb_vector_prior_family(priors; positive, mod=@__MODULE__)
        generated = Expr(:call, :~, :theta,
            Expr(:call, family, Expr(:parameters, Expr(:kw, :n, 2)), args...))
        native_problem = StanBlocks.stan_instantiate(Base.merge(base, native.statement);
            path=joinpath(NATURAL_EMISSION_CACHE, "native-$label.stan"))
        generated_problem = StanBlocks.stan_instantiate(Base.merge(base, generated);
            path=joinpath(NATURAL_EMISSION_CACHE, "generated-$label.stan"))
        @test BS.param_unc_names(native_problem.model) ==
              BS.param_unc_names(generated_problem.model) ==
              ["theta.1", "theta.2"]
        for q in ([-0.7, 0.4], [0.0, 0.0], [0.8, -0.2])
            @test BS.log_density(native_problem.model, q;
                propto=false, jacobian=true) ≈
                  BS.log_density(generated_problem.model, q;
                      propto=false, jacobian=true) atol=1e-12
        end
    end

    exponential_prior = @brm (; x=natural_df.x, g=natural_df.g) begin
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        sd(mu, p, x) ~ Exponential(0.5)
        y ~ Normal(mu, 1)
    end
    exponential_sb = SBBRMI(exponential_prior; mod=@__MODULE__)
    exponential_code = BRM.stan_code(exponential_sb)
    @test StanBlocks.stanc_check(exponential_code; warn_pedantic=false).ok
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", exponential_code)
    exponential_draws = natural_prior_tau_draws(exponential_sb, "prior-exponential")
    @test all(exponential_draws .>= 0.0)
    @test vec(mean(exponential_draws; dims=1)) ≈ [2.0, 0.5] atol=0.3

    # Positive Normal margins redraw through the lower-conditioned RNG, now
    # with a vector scale.
    normal_prior = @brm (; x=natural_df.x, g=natural_df.g) begin
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Normal(0, 1)
        sd(mu, p, x) ~ Normal(0, 0.5)
        y ~ Normal(mu, 1)
    end
    normal_sb = SBBRMI(normal_prior; mod=@__MODULE__)
    normal_code = BRM.stan_code(normal_sb)
    @test StanBlocks.stanc_check(normal_code; warn_pedantic=false).ok
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", normal_code)
    normal_draws = natural_prior_tau_draws(normal_sb, "prior-normal")
    @test all(normal_draws .>= 0.0)
    @test vec(mean(normal_draws; dims=1)) ≈ sqrt(2 / pi) .* [1.0, 0.5] atol=0.15
end

@testset "mixed-family margins keep the generated family's law" begin
    base = StanBlocks.@slic (; y=[0.2, -0.1, 0.4]) begin
        theta::vector[3]
        y ~ normal(theta, 1.0)
    end
    # (margins, positive): consecutive margins of one family and support form
    # one `vcat` block; a family that recurs later starts a new block.
    cases = (
        scales=([BRM.ExprColumn(Exponential, 2.0), BRM.ExprColumn(Exponential, 0.5),
                 BRM.ExprColumn(Normal, 0.0, 0.5)], true),
        recurring=([BRM.ExprColumn(Normal, 0.0, 1.0), BRM.ExprColumn(Cauchy, 0.0, 2.0),
                    BRM.ExprColumn(Normal, 0.5, 1.0)], false),
        bounded=([BRM.ExprColumn(Normal, 0.0, 1.0), BRM.ExprColumn(Uniform, 0.2, 0.8),
                  BRM.ExprColumn(Uniform, 0.3, 0.9)], false),
    )
    mkpath(NATURAL_EMISSION_CACHE)
    for (label, (priors, positive)) in pairs(cases)
        native = BRM._sb_native_vector_prior(:theta, priors, 3; positive)
        @test !isnothing(native)
        @test native.statement.args[3].args[1] === :vcat
        family, args = BRM._sb_vector_prior_family(priors; positive, mod=@__MODULE__)
        generated = Expr(:call, :~, :theta,
            Expr(:call, family, Expr(:parameters, Expr(:kw, :n, 3)), args...))
        native_code = StanBlocks.stan_code(Base.merge(base, native.statement))
        @test StanBlocks.stanc_check(native_code; warn_pedantic=false).ok
        @test !occursin(r"brm_vector_prior_[0-9a-f]+", native_code)
        native_problem = StanBlocks.stan_instantiate(Base.merge(base, native.statement);
            path=joinpath(NATURAL_EMISSION_CACHE, "native-mixed-$label.stan"))
        generated_problem = StanBlocks.stan_instantiate(Base.merge(base, generated);
            path=joinpath(NATURAL_EMISSION_CACHE, "generated-mixed-$label.stan"))
        @test BS.param_unc_names(native_problem.model) ==
              BS.param_unc_names(generated_problem.model) ==
              ["theta.1", "theta.2", "theta.3"]
        for q in ([-0.7, 0.4, 0.1], [0.0, 0.0, 0.0], [0.8, -0.2, -1.3])
            @test BS.log_density(native_problem.model, q;
                propto=false, jacobian=true) ≈
                  BS.log_density(generated_problem.model, q;
                      propto=false, jacobian=true) atol=1e-12
        end
    end

    # Exponential intercept and half-Normal slope margins redraw through their
    # own families inside the shared positive support.
    prior = @brm (; x=natural_df.x, g=natural_df.g) begin
        mu ~ 1 + x + (1 + x | p | g)
        sd(:, p) ~ Exponential(2)
        sd(mu, p, x) ~ Normal(0, 0.5)
        y ~ Normal(mu, 1)
    end
    prior_sb = SBBRMI(prior; mod=@__MODULE__)
    prior_code = BRM.stan_code(prior_sb)
    @test StanBlocks.stanc_check(prior_code; warn_pedantic=false).ok
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", prior_code)
    draws = natural_prior_tau_draws(prior_sb, "prior-mixed")
    @test all(draws .>= 0.0)
    @test vec(mean(draws; dims=1)) ≈ [2.0, 0.5 * sqrt(2 / pi)] atol=0.3
end

@testset "same-family coefficient margins redraw coordinatewise" begin
    prior = @brm (; x=natural_df.x, z=reverse(natural_df.x)) begin
        mu ~ 1 + x + z
        effect(mu, :) ~ Cauchy(0, 1)
        effect(mu, x) ~ Cauchy(0, 2)
        y ~ Normal(mu, 1)
    end
    sb = SBBRMI(prior; mod=@__MODULE__, total_groups=())
    code = BRM.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test !occursin(r"brm_vector_prior_[0-9a-f]+", code)
    problem = StanBlocks.stan_instantiate(
        sb.model; path=joinpath(NATURAL_EMISSION_CACHE, "prior-cauchy.stan"))
    @test LogDensityProblems.dimension(problem) == 0
    names = BS.param_names(problem.model; include_tp=true, include_gq=true)
    beta = Int[findfirst(==("pop_mu_beta_pop.$i"), names) for i in 1:3]
    draws = Matrix{Float64}(undef, 512, 3)
    for draw_index in axes(draws, 1)
        values = BS.param_constrain(
            problem.model, Float64[];
            include_tp=true, include_gq=true,
            rng=BS.StanRNG(problem.model, 40_000 + draw_index))
        draws[draw_index, :] = values[beta]
    end
    # The median absolute Cauchy(0, s) draw is s: (Intercept, x, z).
    @test vec(median(abs.(draws); dims=1)) ≈ [1.0, 2.0, 1.0] rtol=0.25
end
