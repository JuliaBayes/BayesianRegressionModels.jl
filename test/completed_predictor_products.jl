# Run: julia --project=. test/completed_predictor_products.jl [testset substrings...]
using Test, Random, Statistics, LinearAlgebra
using BayesianRegressionModels, StanBlocks, Distributions, BridgeStan
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM_PRODUCT = BayesianRegressionModels

const PRODUCT_DATA = (;
    x=Union{Missing,Float64}[0.4, missing, -0.8, 1.3, missing],
    z=Union{Missing,Float64}[missing, 0.6, 1.4, missing, 0.9],
    y=[0.2, -0.3, 0.7, 0.1, -0.6],
)
const COMPLETED_PRODUCT = @brm begin
    mi(x) ~ Normal(0, 1)
    mi(z) ~ LogNormal(0, 0.5)
    squared = x * x
    mu ~ 1 + x * z + squared * z
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end
const COMPLETED_INTERACTION = @brm begin
    mi(x) ~ Normal(0, 1)
    mi(z) ~ LogNormal(0, 0.5)
    squared = x * x
    mu ~ 1 + x & z + squared & z
    effect(mu, x & z) ~ Normal(0, 1)
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end
const COMPLETED_ALIASES = @brm begin
    mi(x) ~ Normal(0, 1)
    mi(z) ~ LogNormal(0, 0.5)
    squared = x * x
    product = x * z
    squared_product = squared * z
    mu ~ 1 + product + squared_product
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end
const OBSERVED_PRODUCT = @brm begin
    mu ~ 1 + x * z
    y ~ Normal(mu, 1)
end
const OBSERVED_INTERACTION = @brm begin
    mu ~ 1 + x & z
    y ~ Normal(mu, 1)
end
const SINGLE_COMPLETED_PRODUCT = @brm begin
    mi(x) ~ Normal(0, 1)
    squared = x * x
    mu ~ 1 + x * z + squared * z
    y ~ Normal(mu, 1)
end
const SINGLE_COMPLETED_INTERACTION = @brm begin
    mi(x) ~ Normal(0, 1)
    squared = x * x
    mu ~ 1 + x & z + squared & z
    y ~ Normal(mu, 1)
end
const COMPLETED_CATEGORY = @brm begin
    mi(x) ~ Normal(0, 1)
    mu ~ 1 + x & group
    y ~ Normal(mu, 1)
end
const CATEGORY_COMPLETED = @brm begin
    mi(x) ~ Normal(0, 1)
    mu ~ 1 + group & x
    y ~ Normal(mu, 1)
end
const LOG_COMPLETED_CATEGORY = @brm begin
    mi(z) ~ LogNormal(0, 0.5)
    mu ~ 1 + log(z) & group
    y ~ Normal(mu, 1)
end
const STANDARDIZED_COMPLETED_CATEGORY = @brm begin
    mi(x) ~ Normal(0, 1)
    mu ~ 1 + standardize(x) & group
    y ~ Normal(mu, 1)
end

function product_reference(q, names, df; z_modelled=true)
    at = Dict(zip(names, q))
    function complete(raw, stem, positive)
        j = 0
        [if ismissing(value)
            j += 1
            u = at["$(stem)_y_mis.$j"]
            positive ? exp(u) : u
        else
            Float64(value)
        end for value in raw]
    end
    x, z = complete(df.x, "x", false), complete(df.z, "z", true)
    beta = [at["pop_mu_beta_pop.$i"] for i in 1:3]
    squared = x .* x
    mu = beta[1] .+ beta[2] .* x .* z .+ beta[3] .* squared .* z
    pointwise = logpdf.(Normal.(mu, 1), df.y)
    z_lp = z_modelled ? sum(logpdf.(LogNormal(0, 0.5), z)) +
        sum((at["z_y_mis.$j"] for j in 1:count(ismissing, df.z)); init=0.0) : 0.0
    lp = sum(logpdf.(Normal(), x)) + z_lp +
        sum(logpdf.(Normal(), beta)) + sum(pointwise)
    (; lp, x, z, squared, mu, pointwise)
end

function product_oracle(sb, df; z_modelled=true)
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_unc_names(problem.model)
    @test length(names) == 3 + count(ismissing, df.x) + count(ismissing, df.z)
    descriptor = brm_descriptor(sb)
    full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    for shift in (0.0, 0.13, -0.19)
        q = [shift + 0.11 * (i - 4) for i in eachindex(names)]
        expected = product_reference(q, names, df; z_modelled)
        lp, gradient = BridgeStan.log_density_gradient(problem.model, q;
            propto=false, jacobian=true)
        @test lp ≈ expected.lp atol=1e-9 rtol=0
        fd = BRM_PRODUCT._sb_central_diff(
            v -> product_reference(v, names, df; z_modelled).lp, q, 1e-6)
        @test gradient ≈ fd atol=3e-7 rtol=3e-7
        full = BridgeStan.param_constrain(problem.model, q;
            include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, 71))
        for field in (z_modelled ? (:x, :z, :squared, :mu) : (:x, :squared, :mu))
            coords = brm_output_coordinates(descriptor, field, full_names)
            @test full[collect(Int, coords)] ≈ getproperty(expected, field) atol=1e-12 rtol=1e-12
        end
    end
    problem
end

@stestset "completed numeric products and interactions have independent density and output oracles" begin
    original = deepcopy(PRODUCT_DATA)
    artifacts = map((COMPLETED_PRODUCT, COMPLETED_INTERACTION, COMPLETED_ALIASES)) do builder
        sb = SBBRMI(builder(PRODUCT_DATA); mod=@__MODULE__, total_groups=())
        code = BRM_PRODUCT.stan_code(sb)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test length(popcoefnames(sb.parent, :mu)) == 3
        @test all(label -> occursin(r"^[A-Za-z][A-Za-z0-9_]*$", String(label)),
            popcoefnames(sb.parent, :mu))
        product_oracle(sb, PRODUCT_DATA)
        sb
    end
    @test popcoefnames(artifacts[2].parent, :mu) ==
        [:Intercept, :int_x_x_z, :int_squared_x_z]
    @test isequal(PRODUCT_DATA, original)
end

@stestset "completed product replay keeps fitted names and refreshes observed operands" begin
    changed = merge(PRODUCT_DATA, (;
        x=Union{Missing,Float64}[0.9, missing, -0.6, 1.8, missing],
        z=Union{Missing,Float64}[missing, 0.8, 1.7, missing, 1.1]))
    for builder in (COMPLETED_PRODUCT, COMPLETED_INTERACTION)
        sb = SBBRMI(builder(PRODUCT_DATA); mod=@__MODULE__, total_groups=())
        replay = reprocess(sb, changed)
        @test popcoefnames(replay.parent, :mu) == popcoefnames(sb.parent, :mu)
        @test BRM_PRODUCT.stan_code(replay) == BRM_PRODUCT.stan_code(sb)
        @test replay.data[:x_obs] == [0.9, -0.6, 1.8]
        @test replay.data[:z_obs] == [0.8, 1.7, 1.1]
        product_oracle(replay, changed)
    end
end

@stestset "completed products retain gradients with an observed numeric operand" begin
    df = merge(PRODUCT_DATA, (; z=coalesce.(PRODUCT_DATA.z, 1.0)))
    for builder in (SINGLE_COMPLETED_PRODUCT, SINGLE_COMPLETED_INTERACTION)
        sb = SBBRMI(builder(df); mod=@__MODULE__, total_groups=())
        @test StanBlocks.stanc_check(BRM_PRODUCT.stan_code(sb); warn_pedantic=false).ok
        @test length(popcoefnames(sb.parent, :mu)) == 3
        product_oracle(sb, df; z_modelled=false)
    end
end

@stestset "observed numeric products compile and explicit coefficients retain assignment semantics" begin
    df = (; x=[0.4, 0.7, -0.8, 1.3, -0.2], z=[1.1, 0.6, 1.4, 0.5, 0.9], y=PRODUCT_DATA.y)
    for builder in (OBSERVED_PRODUCT, OBSERVED_INTERACTION)
        sb = SBBRMI(builder(df); mod=@__MODULE__, total_groups=())
        @test StanBlocks.stanc_check(BRM_PRODUCT.stan_code(sb); warn_pedantic=false).ok
        problem = StanBlocks.stan_instantiate(sb.model)
        names = BridgeStan.param_unc_names(problem.model)
        q = [0.17, -0.31]
        at = Dict(zip(names, q))
        mu = at["pop_mu_beta_pop.1"] .+ at["pop_mu_beta_pop.2"] .* df.x .* df.z
        expected = sum(logpdf.(Normal(), q)) + sum(logpdf.(Normal.(mu, 1), df.y))
        @test BridgeStan.log_density(problem.model, q; propto=false) ≈ expected atol=1e-9 rtol=0
    end
    # Explicit coefficients enter a predictor through `=`, never through a
    # formula summand allocating another coefficient (84434c7; user decision 1pg29y9).
    guard_data = merge(PRODUCT_DATA, (; z=coalesce.(PRODUCT_DATA.z, 1.0), group=[1,2,3,1,2]))
    for body in (
            "beta ~ Normal(0,1); mu ~ 0 + beta*z; y ~ Normal(mu,1)",
            "mi(x) ~ Normal(0,1); beta ~ Normal(0,1); mu ~ 0 + beta*x; y ~ Normal(mu,1)",
            "mi(x) ~ Normal(0,1); beta ~ Normal(0,1); scaled = beta*x; mu ~ 0 + scaled*z; y ~ Normal(mu,1)",
            "mi(x) ~ Normal(0,1); beta ~ Normal(0,1); scaled = beta*x; mu ~ 0 + scaled&group; y ~ Normal(mu,1)")
        raw = Core.eval(@__MODULE__, BRM_PRODUCT._brm(body; df=guard_data))
        @test_throws r"sampled coefficient" SBBRMI(raw; mod=@__MODULE__, total_groups=())
    end
    assignment = @brm begin
        mi(x) ~ Normal(0, 1)
        beta ~ Normal(0, 1)
        mu = beta * x
        y ~ Normal(mu, 1)
    end
    sb = SBBRMI(assignment(PRODUCT_DATA); mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM_PRODUCT.stan_code(sb); warn_pedantic=false).ok
end

@stestset "completed covariates use categorical contrasts and transformed interaction operands" begin
    df = merge(PRODUCT_DATA, (; group=[1,2,3,1,2]))
    builders = (COMPLETED_CATEGORY, CATEGORY_COMPLETED,
        LOG_COMPLETED_CATEGORY, STANDARDIZED_COMPLETED_CATEGORY)
    for (index, builder) in enumerate(builders)
        sb = SBBRMI(builder(df); mod=@__MODULE__, total_groups=())
        @test length(popcoefnames(sb.parent, :mu)) == 3
        @test StanBlocks.stanc_check(BRM_PRODUCT.stan_code(sb); warn_pedantic=false).ok
        problem = StanBlocks.stan_instantiate(sb.model)
        names = BridgeStan.param_unc_names(problem.model)
        q = [0.12 * (i-2) for i in eachindex(names)]
        at = Dict(zip(names, q))
        stem = index == 3 ? :z : :x
        j = 0
        completed = [if ismissing(v)
            j += 1
            u = at["$(stem)_y_mis.$j"]
            stem === :z ? exp(u) : u
        else
            Float64(v)
        end for v in getproperty(df, stem)]
        values = index == 3 ? log.(completed) : index == 4 ?
            (completed .- mean(skipmissing(df.x))) ./ std(collect(skipmissing(df.x))) : completed
        mu = at["pop_mu_beta_pop.1"] .+ values .* (
            at["pop_mu_beta_pop.2"] .* (df.group .== 2) .+
            at["pop_mu_beta_pop.3"] .* (df.group .== 3))
        full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        full = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(problem.model, 72))
        coordinates = collect(Int, brm_output_coordinates(brm_descriptor(sb), :mu, full_names))
        @test full[coordinates] ≈ mu atol=1e-12 rtol=1e-12
    end
end

const JOINT_COMPLETED_PRODUCTS = @brm begin
    L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
    mi([x, z]) ~ MvNormalCholesky([0.0, 0.0], L)
    mu ~ 1 + x * z + x & z
    y ~ Normal(mu, 1)
end
@stestset "correlated block completed products have a normalized density and gradient oracle" begin
    df = PRODUCT_DATA
    sb = SBBRMI(JOINT_COMPLETED_PRODUCTS(df); mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM_PRODUCT.stan_code(sb); warn_pedantic=false).ok
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_unc_names(problem.model)
    @test length(names) == 10
    function reference(q)
        at = Dict(zip(names, q))
        scales = exp.([at["L_scales.$i"] for i in 1:2])
        rho = tanh(at[only(filter(name -> startswith(name, "L_L_corr."), names))])
        covariance = Symmetric(Diagonal(scales) * [1.0 rho; rho 1.0] * Diagonal(scales))
        joint = MvNormal(zeros(2), covariance)
        # K=2, eta=2: density 3/4*(1-rho^2), plus tanh/positive-scale Jacobians.
        factor_prior = sum(logpdf.(Exponential(1), scales)) + sum(log.(scales)) +
            log(0.75) + 2 * log1p(-rho^2)
        cursor = 0
        completed = Matrix{Float64}(undef, length(df.y), 2)
        for row in eachindex(df.y), (column, raw) in enumerate((df.x, df.z))
            if ismissing(raw[row])
                cursor += 1
                completed[row, column] = at["brm_joint_x__z_y_mis.$cursor"]
            else
                completed[row, column] = raw[row]
            end
        end
        beta = [at["pop_mu_beta_pop.$i"] for i in 1:3]
        mu = beta[1] .+ (beta[2] + beta[3]) .* completed[:, 1] .* completed[:, 2]
        lp = factor_prior + sum(logpdf(joint, collect(row)) for row in eachrow(completed)) + sum(logpdf.(Normal(), beta)) +
            sum(logpdf.(Normal.(mu, 1), df.y))
        (; lp, mu)
    end
    for shift in (0.0, 0.17, -0.13)
        q = [shift + 0.07*(i-3) for i in eachindex(names)]
        lp, gradient = BridgeStan.log_density_gradient(problem.model, q; propto=false)
        @test lp ≈ reference(q).lp atol=1e-9 rtol=0
        fd = BRM_PRODUCT._sb_central_diff(v -> reference(v).lp, q, 1e-6)
        @test gradient ≈ fd atol=3e-7 rtol=3e-7
        full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        full = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(problem.model, 73))
        coordinates = collect(Int, brm_output_coordinates(brm_descriptor(sb), :mu, full_names))
        @test full[coordinates] ≈ reference(q).mu atol=1e-12 rtol=1e-12
    end
end

@stestset "grouped scalar composition accepts completed products and interactions" begin
    rng = Xoshiro(393)
    x = Union{Missing,Float64}[exp(0.4 * randn(rng)) for _ in 1:16]
    x[[4, 9, 15]] .= missing
    df = (; x, z=collect(range(-1.0,1.0;length=16)),
        subject=repeat(1:4;inner=4), outcome=0.1 .* randn(rng,16))
    for operator in ("*", "&")
        body = """
            mi(x) ~ LogNormal(0.0, 0.5)
            x_sq = x * x
            loc ~ 1 + log(x) + standardize(x) + x $operator z + x_sq + (1 | p | subject)
            effect(loc, :) ~ Normal(0.0, 1.0)
            sd(:, p) ~ Exponential(1.0)
            outcome ~ Normal(loc, 1.0)
        """
        raw = Core.eval(@__MODULE__, BRM_PRODUCT._brm(body; df=df))
        sb = SBBRMI(raw; mod=@__MODULE__, total_groups=())
        @test StanBlocks.stanc_check(BRM_PRODUCT.stan_code(sb); warn_pedantic=false).ok
    end
end
