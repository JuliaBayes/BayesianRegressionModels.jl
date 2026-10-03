# Public synthetic arithmetic fixtures; no application model or data.
using Test, BayesianRegressionModels, StanBlocks, BridgeStan, Distributions

const PRODUCT_DATA = (; y=[0.2, -0.1, 0.4], x=[0.7, -0.2, 1.3])
const PRODUCT_FLAT = @brm begin
    b ~ Normal(0, 1); r ~ Uniform(-1, 1)
    s ~ Exponential(1); u ~ Exponential(1); z ~ Normal(0, 1)
    mu = b + r * (s / u) * z
    y ~ Normal(mu, 1)
end
const PRODUCT_BINARY = @brm begin
    b ~ Normal(0, 1); r ~ Uniform(-1, 1)
    s ~ Exponential(1); u ~ Exponential(1); z ~ Normal(0, 1)
    ratio = s / u
    weight = r * ratio
    mu = b + weight * z
    y ~ Normal(mu, 1)
end
const PRODUCT_ARRAY = @brm begin
    b ~ Normal(0, 1); r ~ Uniform(-1, 1)
    s ~ Exponential(1); u ~ Exponential(1); z ~ Normal(0, 1)
    mu = b + r * (s / u) * z * x
    y ~ Normal(mu, 1)
end
const PRODUCT_ARRAY_BINARY = @brm begin
    b ~ Normal(0, 1); r ~ Uniform(-1, 1)
    s ~ Exponential(1); u ~ Exponential(1); z ~ Normal(0, 1)
    ratio = s / u
    weight = r * ratio
    amplitude = weight * z
    mu = b + amplitude * x
    y ~ Normal(mu, 1)
end
const PRODUCT_SLIC = StanBlocks.@slic PRODUCT_DATA begin
    b ~ normal(0, 1); r ~ uniform(-1, 1)
    s ~ exponential(1); u ~ exponential(1); z ~ normal(0, 1)
    mu = b + r * (s / u) * z
    y ~ normal(mu, 1)
end

@testset "flat scalar and array products preserve full model semantics" begin
    models = (
        (:flat, PRODUCT_FLAT, ones(3)),
        (:binary, PRODUCT_BINARY, ones(3)),
        (:array, PRODUCT_ARRAY, PRODUCT_DATA.x),
        (:array_binary, PRODUCT_ARRAY_BINARY, PRODUCT_DATA.x),
        (:direct_slic, nothing, ones(3)),
    )
    @testset "$label" for (label, builder, weights) in models
        model = isnothing(builder) ? PRODUCT_SLIC :
            SBBRMI(builder(PRODUCT_DATA); mod=@__MODULE__, total_groups=()).model
        check = StanBlocks.stanc_check(StanBlocks.stan_code(model))
        @test check.ok
        problem = StanBlocks.stan_instantiate(model)
        names = BridgeStan.param_unc_names(problem.model)
        @test Set(names) == Set(["b", "r", "s", "u", "z"])
        full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        for raw in ((b=0.2, r=-0.4, s=0.3, u=-0.5, z=-0.7),
                    (b=-0.4, r=0.5, s=-0.3, u=1.2, z=0.7),
                    (b=0.1, r=-1.4, s=-0.9, u=-1.7, z=-0.2))
            t = inv(1 + exp(-raw.r))
            r = -1 + 2t
            s, u = exp(raw.s), exp(raw.u)
            ratio = s / u
            mu = raw.b .+ (r * ratio * raw.z) .* weights
            residual = sum(PRODUCT_DATA.y .- mu)
            weighted_residual = sum((PRODUCT_DATA.y .- mu) .* weights)
            expected_lp = logpdf(Normal(), raw.b) + logpdf(Normal(), raw.z) +
                logpdf(Uniform(-1, 1), r) + logpdf(Exponential(), s) +
                logpdf(Exponential(), u) +
                sum(logpdf(Normal(m, 1), y) for (m, y) in zip(mu, PRODUCT_DATA.y)) +
                log(2) + log(t) + log1p(-t) + raw.s + raw.u
            expected_gradient = (
                b=-raw.b + residual,
                r=weighted_residual * 2t * (1 - t) * ratio * raw.z + 1 - 2t,
                s=weighted_residual * r * ratio * raw.z + 1 - s,
                u=-weighted_residual * r * ratio * raw.z + 1 - u,
                z=-raw.z + weighted_residual * r * ratio,
            )
            q = [getproperty(raw, Symbol(name)) for name in names]
            gradient = zeros(length(q))
            lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
                propto=false, jacobian=true)
            @test lp ≈ expected_lp atol=2e-12 rtol=2e-12
            @test gradient ≈ [getproperty(expected_gradient, Symbol(name)) for name in names] atol=2e-12 rtol=2e-12
            full = BridgeStan.param_constrain(problem.model, q;
                include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, 41))
            mu_names = label in (:array, :array_binary) ?
                ["mu.$i" for i in eachindex(mu)] : ["mu"]
            expected_mu = label in (:array, :array_binary) ? mu : [first(mu)]
            @test [full[only(findall(==(name), full_names))] for name in mu_names] ≈ expected_mu atol=1e-14
            for (i, y) in enumerate(PRODUCT_DATA.y)
                @test full[only(findall(==("y_likelihood.$i"), full_names))] ≈ logpdf(Normal(mu[i], 1), y) atol=2e-12
            end
        end
    end
end
