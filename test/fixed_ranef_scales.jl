# test/fixed_ranef_scales.jl — random-effect scales and correlations held at
# known values: `sd(<lp|:>, ID[, coefficient]) = value` and `cor(:, ID) = R`,
# nlmixr2's `fix()` on an omega (snag `no-documented-wa-41b05d57`, user
# decision `1kicac2`).
#
# Run: julia --project=test test/fixed_ranef_scales.jl
# Set BRM_FIXED_RANEF_RUNTIME=0 to skip the BridgeStan/RK density probes.
#
# The density checks compare each known-value model against the SAME model
# with the known quantities sampled, evaluated at the known values: the two
# log densities (no Jacobian) must differ by exactly the sampled model's prior
# density of those values, at every point. That pins "known = conditioned at
# the value" against the established sampled emission rather than against a
# hand-written Stan program.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra
using Distributions: Exponential, Normal, LKJCholesky, TDist, Dirac, logpdf

const FIXED_RUNTIME = get(ENV, "BRM_FIXED_RANEF_RUNTIME", "1") != "0"
const FIXED_CACHE = joinpath(tempdir(), "brm-fixed-ranef-scales")

Random.seed!(20261016)
const SUBJECTS = repeat(1:6, inner=4)
const DF = (; subject=SUBJECTS, x=round.(randn(24); digits=3),
            y=round.(randn(24); digits=3), omega_sd=0.4)
const R = [1 0.3; 0.3 1]

sampled_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
end

# One known SD next to a sampled one, with its own prior.
mixed_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(mu, p, Intercept) = sqrt(0.1)
    sd(mu, p, x) ~ Exponential(0.5)
    y ~ Normal(mu, sigma)
end
mixed_sampled_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(mu, p, x) ~ Exponential(0.5)
    y ~ Normal(mu, sigma)
end

# Every SD from a data field, and the correlation.
known_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) = omega_sd
    cor(:, p) = [1 0.3; 0.3 1]
    y ~ Normal(mu, sigma)
end

# Only the correlation.
cor_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    cor(:, p) = [1 0.3; 0.3 1]
    y ~ Normal(mu, sigma)
end

# nlmixr2's `eta.ka ~ fix(0.1)`: a one-margin ID'd block.
single_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | ka | subject)
    sd(:, ka) = sqrt(0.1)
    y ~ Normal(mu, sigma)
end
single_sampled_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | ka | subject)
    y ~ Normal(mu, sigma)
end

studentt_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(mu, p, Intercept) = 0.3
    ranef(:, p) ~ TDist(4)
    y ~ Normal(mu, sigma)
end

# A BRMixr-shaped per-subject `@plate` cell reading predictors of a block with
# one known SD (synthetic one-compartment-like decay).
const PK = (; subject=collect(1:5), time=[[0.5, 1.0, 2.0] for _ in 1:5],
            dv=[round.(abs.(randn(3)) .+ 0.5; digits=3) for _ in 1:5])
plate_builder = @brm begin
    sigma ~ Exponential(1)
    log_cl ~ 1 + (1 | pk | subject)
    log_v ~ 1 + (1 | pk | subject)
    sd(log_v, pk) = sqrt(0.1)
    @plate for i in eachindex(log_cl)
        cp[i] = exp(log_cl[i] - log_v[i]) .* exp.(-time[i])
        dv[i] ~ LogNormal(log.(cp[i]), sigma)
    end
end

build(builder; kwargs...) =
    SBBRMI(builder(DF); mod=@__MODULE__, total_groups=(), kwargs...)
stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok

function instantiate(sb)
    mkpath(FIXED_CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(FIXED_CACHE, string(hash(code)) * ".stan"))
end

# The Stan data dict a block's known values were bound to.
stan_data_of(sb) = BRM.stan_data(sb)

@stestset "parse, introspection and composition" begin
    brmi = mixed_builder(DF)
    specs = ranef_effect_priors(brmi)
    known = only(s for s in specs if s.family === BRM._brm_known)
    @test known.class === :sd && known.id === :p
    @test known.predictor === :mu && known.coefficient === :Intercept
    # A known value is a prior-address statement, not model structure.
    @test length(ranef_effect_priors(priors_of(brmi))) == 2
    @test isempty(ranef_effect_priors(structure_of(brmi)))
    shown = sprint(show, brmi)
    @test occursin(" = sqrt(0.1)", shown)
    @test !occursin("_brm_known", shown)

    # One address carries either a prior or a value; merge swaps them.
    structure = sampled_builder(DF)
    fixed = @brm DF begin
        sd(:, p) = omega_sd
        cor(:, p) = [1 0.3; 0.3 1]
    end
    @test BRM.stan_code(SBBRMI(Base.merge(structure, fixed); mod=@__MODULE__,
        total_groups=())) == BRM.stan_code(build(known_builder))
    estimated = @brm DF begin
        sd(:, p) ~ Exponential(1)
    end
    swapped = Base.merge(known_builder(DF), estimated)
    @test any(s -> s.class === :sd && s.family === Exponential, ranef_effect_priors(swapped))

    # refused: one address holds either a prior or a known value (`1kicac2`).
    @test_throws "duplicate `sd(:, p)` prior statement" @eval @brm $DF begin
        mu ~ 1 + x + (1 | p | subject)
        sd(:, p) ~ Exponential(1)
        sd(:, p) = 0.3
        y ~ Normal(mu, 1)
    end
    # refused: only random-effect `sd`/`cor` addresses take a known value
    # (`1kicac2`); a known population term is `offset(...)`.
    @test_throws "offset(...)" @eval @brm $DF begin
        mu ~ 1 + x
        effect(mu, x) = 0.5
        y ~ Normal(mu, 1)
    end
    @test_throws "random-effect blocks only" @eval @brm $DF begin
        mu ~ 1 + s(x)
        sd(mu, s(x)) = 0.5
        y ~ Normal(mu, 1)
    end
    @test_throws "only random-effect `sd(...)` and `cor(...)`" @eval @brm $DF begin
        mu ~ 1 + (1 | p | subject)
        ranef(:, p) = 0.5
        y ~ Normal(mu, 1)
    end
end

@stestset "SBBRMI emission" begin
    for builder in (mixed_builder, known_builder, cor_builder, single_builder,
                    studentt_builder)
        @test stanc_ok(build(builder))
    end
    @test stanc_ok(build(mixed_builder; cv_groups=[:subject]))
    plate = SBBRMI(plate_builder(PK); mod=@__MODULE__, total_groups=())
    @test stanc_ok(plate)
    @test only(ranef_blocks(plate)).family === :ranef_correlated_draws_known_scale

    families = Dict(b => only(ranef_blocks(build(b))).family for b in
        (mixed_builder, known_builder, cor_builder, single_builder, studentt_builder))
    @test families[mixed_builder] === :ranef_correlated_draws_known_scale
    @test families[known_builder] === :ranef_correlated_draws_known_cor
    @test families[cor_builder] === :ranef_correlated_draws_known_cor
    @test families[single_builder] === :ranef_correlated_draws_known_scale
    @test families[studentt_builder] === :ranef_correlated_draws_known_scale_student_t

    # Known values are data: every scale known binds `<bucket>_tau`; a mixed
    # block binds only its known part and samples `<bucket>_tau_free`.
    data = stan_data_of(build(known_builder))
    @test data[:b_p_subject_tau] == [0.4, 0.4]
    @test data[:b_p_subject_L_known] ≈ Matrix(cholesky(R).L)
    mixed = build(mixed_builder)
    data = stan_data_of(mixed)
    @test data[:b_p_subject_tau_known] == [sqrt(0.1)]
    @test data[:b_p_subject_tau_free_idx] == [2]
    code = BRM.stan_code(mixed)
    @test occursin("b_p_subject_tau_free", code)
    @test !occursin("b_p_subject_L_known", code)

    # refused: a known-value block has no centered emission yet (follow-up).
    @test_throws "centered emission of a known-value block" build(mixed_builder;
        centered_groups=[:subject])
    # refused: an R2D2 decomposition derives the block's scales, so it cannot
    # share the block with a known scale or correlation.
    @test_throws "r2d2" SBBRMI(eval(:(@brm $DF begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        sd(:, p) ~ r2d2(reference_scale=sigma)
        sd(mu, p, x) = 0.3
        y ~ Normal(mu, sigma)
    end)); mod=@__MODULE__)
    @test_throws "r2d2" SBBRMI(eval(:(@brm $DF begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        sd(:, p) ~ r2d2(reference_scale=sigma)
        cor(:, p) = [1 0.3; 0.3 1]
        y ~ Normal(mu, sigma)
    end)); mod=@__MODULE__)
    # refused: a known value is a number or a correlation matrix (`1kicac2`).
    bad(stmt) = eval(:(@brm $DF begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        $stmt
        y ~ Normal(mu, sigma)
    end))
    @test_throws "strictly positive" SBBRMI(bad(:(sd(:, p) = -0.1)); mod=@__MODULE__)
    @test_throws "needs a known value" SBBRMI(bad(:(sd(:, p) = sigma)); mod=@__MODULE__)
    @test_throws "unit diagonal" SBBRMI(bad(:(cor(:, p) = [2 0.1; 0.1 2])); mod=@__MODULE__)
    @test_throws "symmetric" SBBRMI(bad(:(cor(:, p) = [1 0.1; 0.2 1])); mod=@__MODULE__)
    @test_throws "positive definite" SBBRMI(bad(:(cor(:, p) = [1 1.0; 1.0 1])); mod=@__MODULE__)
    @test_throws "2×2 correlation" SBBRMI(bad(:(cor(:, p) = [1 0 0; 0 1 0; 0 0 1])); mod=@__MODULE__)
    @test_throws "no correlation to fix" SBBRMI(eval(:(@brm $DF begin
        mu ~ 1 + (1 | ka | subject)
        cor(:, ka) = [1;;]
        y ~ Normal(mu, 1)
    end)); mod=@__MODULE__)
end

# The log density (propto=false, jacobian=false) of a compiled model at a
# constrained point given by name.
function stan_lp(model, values)
    theta = [values[n] for n in BridgeStan.param_names(model)]
    BridgeStan.log_density(model, BridgeStan.param_unconstrain(model, theta);
        propto=false, jacobian=false)
end

function lkj_factor(prefix, r)
    L = Matrix(cholesky([1 r; r 1]).L)
    Dict("$prefix.1.1" => L[1, 1], "$prefix.2.1" => L[2, 1],
         "$prefix.1.2" => 0.0, "$prefix.2.2" => L[2, 2])
end

# Random values for the parameters every model below shares.
function shared_point(rng, block, K)
    values = Dict{String,Float64}("sigma" => 0.5 + rand(rng),
        "pop_mu_beta_pop.1" => randn(rng), "pop_mu_beta_pop.2" => randn(rng))
    for i in 1:(K * 6)
        values["$(block)_z_flat.$i"] = randn(rng)
    end
    values
end

FIXED_RUNTIME && @stestset "BridgeStan: known equals sampled at the known value" begin
    rng = MersenneTwister(7)
    # A one-margin block; the sampled twin draws `tau ~ half-normal(0, 1)`.
    known, twin = instantiate(build(single_builder)).model,
        instantiate(build(single_sampled_builder)).model
    tau = sqrt(0.1)
    for _ in 1:3
        # A one-margin block keeps its 1x1 correlation factor on both sides.
        base = merge(shared_point(rng, "b_ka_subject", 1),
            Dict("b_ka_subject_L.1.1" => 1.0))
        sampled = stan_lp(twin, merge(base, Dict("b_ka_subject_tau.1" => tau)))
        @test sampled - stan_lp(known, base) ≈ logpdf(Normal(), tau) atol=1e-10
    end
    # One known and one sampled margin: the free scale keeps its own prior.
    known, twin = instantiate(build(mixed_builder)).model,
        instantiate(build(mixed_sampled_builder)).model
    for _ in 1:3
        r, free = 0.6rand(rng) - 0.3, 0.2 + rand(rng)
        base = merge(shared_point(rng, "b_p_subject", 2), lkj_factor("b_p_subject_L", r))
        sampled = stan_lp(twin, merge(base, Dict("b_p_subject_tau.1" => sqrt(0.1),
            "b_p_subject_tau.2" => free)))
        fixed = stan_lp(known, merge(base, Dict("b_p_subject_tau_free.1" => free)))
        @test sampled - fixed ≈ logpdf(Normal(), sqrt(0.1)) atol=1e-10
    end
    # Known scales and correlation: the twin samples `tau` (half-normal) and `L`
    # (LKJ(1)); the difference is their prior density, constant in the rest.
    known, twin = instantiate(build(known_builder)).model,
        instantiate(build(sampled_builder)).model
    extra = merge(Dict("b_p_subject_tau.1" => 0.4, "b_p_subject_tau.2" => 0.4),
        lkj_factor("b_p_subject_L", 0.3))
    differences = map(1:3) do _
        base = shared_point(rng, "b_p_subject", 2)
        stan_lp(twin, merge(base, extra)) - stan_lp(known, base)
    end
    @test maximum(differences) - minimum(differences) <= 1e-10
    @test first(differences) ≈ 2logpdf(Normal(), 0.4) +
        logpdf(LKJCholesky(2, 1.0), Cholesky(LowerTriangular(Matrix(cholesky(R).L)))) atol=1e-8
end

FIXED_RUNTIME && @stestset "RKBRMI: the same law as SBBRMI" begin
    for (name, builder, data) in (("mixed", mixed_builder, DF), ("known", known_builder, DF),
            ("cor", cor_builder, DF), ("single", single_builder, DF),
            ("studentt", studentt_builder, DF), ("plate", plate_builder, PK))
        brmi = builder(data)
        rk, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        stan = instantiate(sb)
        stan_names = BridgeStan.param_unc_names(stan.model)
        transport = brm_coordinate_transport(rk, sb, stan_names)
        @test length(transport) == length(coordinate_names(rk.model.layout)) == length(stan_names)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))))
            stan_u = brm_rk_point_to_stan(transport, u)
            checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
            @test checked.max_error <= 1e-12
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

FIXED_RUNTIME && @stestset "compositions: prediction, replay and prior programs" begin
    sb = build(mixed_builder)
    unc = BridgeStan.param_unc_names(instantiate(sb).model)
    draws = reshape(collect(range(-0.4, 0.5; length=2length(unc))), 2, length(unc))
    pop = population_draws(sb, draws, unc; groups=:subject)
    z = [i for (i, n) in enumerate(unc) if startswith(n, "b_p_subject_z_flat")]
    @test length(z) == 12 && all(iszero, pop[:, z])
    keep = setdiff(eachindex(unc), z)
    @test pop[:, keep] == draws[:, keep]

    # New subjects re-draw in generated quantities under the known scale.
    new_subjects = (; subject=[7, 7, 8], x=[0.1, -0.2, 0.3], y=[0.0, 0.1, -0.1],
                    omega_sd=0.4)
    cv = reprocess(sb, new_subjects; resample_groups=[:subject])
    @test stanc_ok(cv)
    @test BRM.stan_data(cv)[:b_p_subject_tau_known] == [sqrt(0.1)]

    # A prior program (the response omitted) keeps the known block whole.
    prior = SBBRMI(known_builder((; subject=DF.subject, x=DF.x, omega_sd=0.4));
        mod=@__MODULE__, total_groups=())
    @test stanc_ok(prior)
    @test only(ranef_blocks(prior)).family === :ranef_correlated_draws_known_cor
end

FIXED_RUNTIME && @stestset "brm_start: known values are restated, sampled ones start" begin
    # sd = (sqrt(0.1), 0.5); the first matches the known SD of `mixed_builder`.
    Sigma = [0.1 0.02; 0.02 0.25]
    contradicting = [0.2 0.0; 0.0 0.25]
    sb = build(mixed_builder)
    model = instantiate(sb).model
    names = BridgeStan.param_names(model)
    u = brm_start(sb, model, (; p=Sigma, sigma=0.7))
    values = BridgeStan.param_constrain(model, u)
    @test values[only(findall(==("b_p_subject_tau_free.1"), names))] ≈ 0.5
    # refused: an initial value cannot contradict a known one (`1kicac2`).
    @test_throws "contradicts it" brm_start(sb, model, (; p=contradicting))
    rk = RKBRMI(mixed_builder(DF))
    @test length(brm_start(rk, (; p=Sigma, sigma=0.7))) ==
        length(coordinate_names(rk.model.layout))
    @test_throws "contradicts it" brm_start(rk, (; p=contradicting))
    # A known correlation: restating it starts, another one is refused.
    known = build(known_builder)
    known_model = instantiate(known).model
    @test brm_start(known, known_model, (; p=0.16 .* R)) isa Vector{Float64}
    @test_throws "contradicts it" brm_start(known, known_model,
        (; p=0.16 .* [1 0.5; 0.5 1]))
    @test_throws "contradicts it" brm_start(RKBRMI(known_builder(DF)),
        (; p=0.16 .* [1 0.5; 0.5 1]))
end

@stestset "descriptors, descriptions and the loud edges" begin
    sb = build(mixed_builder)
    d = brm_descriptor(sb; name=:fixed)
    # refused: a known SD is not a parameter, so it has no posterior coordinate.
    @test_throws "held at the known value" brm_ranef_sd_coordinates(d, :mu,
        String[]; id=:p, coefficient=:Intercept)
    # The sampled margin of the same block is its element of `<binding>_tau`.
    if FIXED_RUNTIME
        names = BridgeStan.param_names(instantiate(sb).model; include_tp=true,
            include_gq=true)
        resolved = brm_ranef_sd_coordinates(d, :mu, names; id=:p, coefficient=:x)
        @test names[only(resolved.coordinates)] == "b_p_subject_tau.2"
    end

    description = brm_description(d)
    known = [p for p in description.priors if get(p.source, :kind, nothing) === :known]
    @test length(known) == 1

    # refused: NLMEEstimation cannot hold a covariance family fixed yet
    # (NLMEEstimation snag `fixed-omega-fami-a1322422`).
    @test_throws "fixed-omega-fami-a1322422" brm_nlme_view(RKBRMI(mixed_builder(DF)))
    # refused: adaptive centering of a known-value block is a follow-up.
    @test_throws "known-value block" adaptive_centering_blocks(sb, String[])
end
