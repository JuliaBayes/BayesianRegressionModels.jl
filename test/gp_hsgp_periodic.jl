# test/gp_hsgp_periodic.jl — `cov=:periodic` for `hsgp` and `gp`.
#
# Run: julia --project=test test/gp_hsgp_periodic.jl
# Set BRM_GP_RUNTIME=0 to skip the BridgeStan gates.
#
# The periodic Hilbert-space basis (Riutort-Mayol et al. 2023, "periodic
# kernel") approximates Stan's `gp_periodic_cov`,
#
#     k(x, x') = sigma^2 exp(-2 sin^2(pi |x - x'| / period) / rho^2),
#
# with `k` harmonics of `2pi / period`: columns cos(j w0 x) and sin(j w0 x),
# each weighted by q_j = sigma sqrt(2 exp(-a) I_j(a)), a = 1 / rho^2. Every
# reference quantity below is rebuilt here from `SpecialFunctions.besselix`
# and the analytic kernel, never read back out of BRM, so the tests cannot
# agree with the implementation by construction.

using Test
using BayesianRegressionModels
using StanBlocks
using LinearAlgebra
using LogDensityProblems
using SpecialFunctions: besselix
using Distributions: LogNormal, Normal, Uniform

const PERIODIC_RUNTIME = get(ENV, "BRM_GP_RUNTIME", "1") != "0"
const PERIODIC_CACHE = joinpath(tempdir(), "brm-gp-periodic")
const PERIODIC_N = 16
const PERIOD = 24.0

periodic_df(; shift=0.0, x=nothing) = begin
    xs = isnothing(x) ? collect(range(0.0, 20.0; length=PERIODIC_N)) .+ shift : x
    (; x=xs, y=sin.(2pi .* xs ./ PERIOD), z=cos.(xs),
       g=repeat(["a", "b"], inner=PERIODIC_N ÷ 2))
end

sbb(brmi) = SBBRMI(brmi; mod=@__MODULE__)
code_of(sb) = StanBlocks.stan_code(sb.model)
transpiles_and_stanc(sb) =
    StanBlocks.stan.transpiles(sb.model) &&
        StanBlocks.stanc_check(code_of(sb); warn_pedantic=false).ok

# The DECLARATION line of the length-scale parameter (see gp_hsgp_priors.jl).
_rho_declaration(code::AbstractString) = begin
    hits = collect(m.match for m in eachmatch(r"^[^\n]*\b\w*_rho(?:_iso)?;[ \t]*$"m, code))
    length(hits) == 1 || error("expected exactly one `rho` declaration, got $(hits)")
    only(hits)
end

# Reference construction, independent of BRM's helpers.
ref_basis(x, k, period) = begin
    w0 = 2pi / period
    hcat([cos.(w0 * j .* x) for j in 1:k]..., [sin.(w0 * j .* x) for j in 1:k]...)
end
ref_weights(k, sigma, rho) = begin
    a = 1 / rho^2
    q = [sigma * sqrt(2 * besselix(j, a)) for j in 1:k]
    vcat(q, q)
end
ref_kernel(x, sigma, rho, period) =
    [sigma^2 * exp(-2 * sin(pi * abs(xi - xj) / period)^2 / rho^2)
     for xi in x, xj in x]

periodic_builder = @brm begin
    y ~ Normal(mu, 1.)
    mu ~ 1 + hsgp(x; k=5, cov=:periodic, period=24.0)
end

# The periodic weights call Stan's `log_modified_bessel_first_kind`, registered
# in StanBlocks from `bec23bc3c52303ebde60a026af48c435e4c81330` (snag
# `log-modified-bes-60f43dd8`, filed from this lane); the test pin carries it.


# ------------------------------------------------------------- mathematics

@testset "the cosine/sine basis with Bessel weights reproduces gp_periodic_cov" begin
    x = periodic_df().x
    sigma, rho = 0.7, 0.8
    k = 40
    PHI = ref_basis(x, k, PERIOD)
    q = ref_weights(k, sigma, rho)
    approx = PHI * Diagonal(q .^ 2) * PHI'
    # The dropped j = 0 harmonic is the kernel's constant term.
    exact = ref_kernel(x, sigma, rho, PERIOD) .- sigma^2 * besselix(0, 1 / rho^2)
    @test maximum(abs, approx - exact) < 1e-10
    # ...and BRM's Julia-side basis IS that reference basis.
    sb = sbb(periodic_builder(periodic_df()))
    @test sb.data[:PHI_hsgp_x] ≈ ref_basis(x, 5, PERIOD)
end

# ------------------------------------------------------------ validation

@testset "public contract and refusals" begin
    df = periodic_df()
    brmi = periodic_builder(df)
    @test popcoefnames(brmi, :mu) == [:Intercept]
    @test isempty(grouping_factors(brmi, :mu))

    cases = [
        ("requires a numeric `period=`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, cov=:periodic) end)),
        ("meaningful only with `cov=:periodic`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, period=24.0) end)),
        ("finite positive numeric", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, cov=:periodic, period=-24.0) end)),
        ("finite positive numeric", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, cov=:periodic, period=0) end)),
        ("supports `:exp_quad` and `:periodic`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, cov=:matern32) end)),
        ("supports exactly one axis", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x, z; k=5, cov=:periodic, period=24.0) end)),
        ("does not accept `c=`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, c=1.5, cov=:periodic, period=24.0) end)),
        ("does not accept `domain=`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, domain=(0.0, 24.0), cov=:periodic, period=24.0) end)),
        ("does not accept `orthogonal_to=`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + x + hsgp(x; k=5, orthogonal_to=:linear, cov=:periodic, period=24.0) end)),
        ("does not accept `by=`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, by=g, cov=:periodic, period=24.0) end)),
        ("`iso=false` has no meaning", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=5, iso=false, cov=:periodic, period=24.0) end)),
        ("requires a numeric `period=`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + gp(x; cov=:periodic) end)),
        ("supports exactly one axis", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + gp(x, z; cov=:periodic, period=24.0) end)),
        ("`iso=false` has no meaning", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + gp(x; cov=:periodic, period=24.0, iso=false) end)),
        ("meaningful only with `cov=:periodic`", () -> sbb(@brm df begin
            y ~ Normal(mu, 1.); mu ~ 1 + gp(x; period=24.0) end)),
    ]
    for (fragment, build) in cases
        err = try
            build()
            nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin(fragment, sprint(showerror, err))
    end

    # A model-derived axis has no Julia-time values to build the cosine/sine
    # columns from; it is refused by name rather than falling into the
    # exp-quad latent path.
    latent = @brm df begin
        log_x ~ 1 + z
        w = exp(log_x)
        mu ~ 1 + hsgp(w; k=5, cov=:periodic, period=24.0)
        y ~ Normal(mu, 1.)
    end
    @test_throws "requires a raw-data axis" sbb(latent)
end

# -------------------------------------------------------- data and floor

@testset "basis data, harmonics, and the periodic validity floor" begin
    df = periodic_df()
    sb = sbb(periodic_builder(df))
    @test size(sb.data[:PHI_hsgp_x]) == (PERIODIC_N, 10)
    @test sb.data[:harmonics_hsgp_x] == Float64[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    entry = sb.preproc[:PHI_hsgp_x]
    @test entry.kind === :hsgp
    @test entry.const_.cov === :periodic
    @test entry.const_.period == PERIOD
    @test entry.const_.K == 5
    @test entry.raw_ref == (:x,)
    @test only(i for i in brm_descriptor(sb).inputs
               if i.transform === :hsgp).name === :PHI_hsgp_x

    # The floor solves the exp-quad amplitude-ratio rule for the periodic
    # weights: I_k(a) / I_1(a) = 100^-2 at a = 1 / rho_lower^2.
    floor5 = sb.data[:rho_lower_hsgp_x]
    @test floor5 > 0
    a5 = 1 / floor5^2
    @test besselix(5, a5) / besselix(1, a5) ≈ 1e-4 rtol=1e-8

    floor10 = sbb(@brm df begin
        y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=10, cov=:periodic, period=24.0)
    end).data[:rho_lower_hsgp_x]
    a10 = 1 / floor10^2
    @test besselix(10, a10) / besselix(1, a10) ≈ 1e-4 rtol=1e-8
    @test floor10 < floor5                     # more harmonics reach smaller scales

    @test sbb(@brm df begin
        y ~ Normal(mu, 1.); mu ~ 1 + hsgp(x; k=1, cov=:periodic, period=24.0)
    end).data[:rho_lower_hsgp_x] == 0.0        # nothing truncated to bound
end

# ------------------------------------------------------- emitted program

@testset "emitted Stan, priors, and their addresses" begin
    df = periodic_df()
    sb = sbb(periodic_builder(df))
    code = code_of(sb)
    @test occursin("brm_hsgp_periodic_sqrt_spd", code)
    @test occursin("log_modified_bessel_first_kind", code)
    @test !occursin("brm_hsgp_sqrt_spd(", code)
    @test occursin("rho_lower_hsgp_x", _rho_declaration(code))
    @test occursin("hsgp_x_rho_iso ~ lognormal(0.0, 1.0);", code)
    @test transpiles_and_stanc(sb)

    exact = sbb(@brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + gp(x; cov=:periodic, period=24.0)
    end)
    exact_code = code_of(exact)
    @test occursin("brm_periodic_cov", exact_code)
    @test occursin("gp_periodic_cov", exact_code)
    @test !occursin("gp_exp_quad_cov", exact_code)
    @test occursin("real<lower=0.0> gp_x_rho;", exact_code)
    @test transpiles_and_stanc(exact)

    # The term-prior addresses are the same as for the exp-quad basis, and
    # the default density is REPLACED, floor included.
    configured = sbb(@brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5, cov=:periodic, period=24.0)
        length_scale(:, hsgp(x)) ~ Uniform(0.5, 3.0)
        sd(mu, hsgp(x)) ~ Normal(0, 0.5)
    end)
    ccode = code_of(configured)
    @test occursin("hsgp_x_rho_iso ~ uniform(0.5, 3.0);", ccode)
    @test occursin("hsgp_x_sigma ~ normal(0.0, 0.5);", ccode)
    @test !occursin("lognormal(", ccode)
    decl = _rho_declaration(ccode)
    @test occursin("lower=0.5", decl) && occursin("upper=3.0", decl)
    @test !occursin("rho_lower_hsgp_x", decl)
    @test transpiles_and_stanc(configured)
    specs = term_priors(periodic_builder(df))
    @test isempty(specs)
    @test length(term_priors(@brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5, cov=:periodic, period=24.0)
        length_scale(:, hsgp(x)) ~ Uniform(0.5, 3.0)
        sd(mu, hsgp(x)) ~ Normal(0, 0.5)
    end)) == 2

    exact_configured = sbb(@brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + gp(x; cov=:periodic, period=24.0)
        length_scale(:, gp(x)) ~ LogNormal(0.0, 0.5)
    end)
    @test occursin("gp_x_rho ~ lognormal(0.0, 0.5);", code_of(exact_configured))
    @test transpiles_and_stanc(exact_configured)
end

# --------------------------------------------------------------- replay

@testset "reprocess rebuilds the periodic basis on new rows with no fitted constant" begin
    df = periodic_df()
    sb = sbb(periodic_builder(df))
    new_df = periodic_df(; shift=3.25)
    replay = reprocess(sb, new_df)
    @test replay.data[:PHI_hsgp_x] ≈ ref_basis(new_df.x, 5, PERIOD)
    @test replay.data[:harmonics_hsgp_x] == sb.data[:harmonics_hsgp_x]
    @test replay.data[:rho_lower_hsgp_x] == sb.data[:rho_lower_hsgp_x]
    @test code_of(replay) == code_of(sb)
    @test replay.preproc[:PHI_hsgp_x].const_ == sb.preproc[:PHI_hsgp_x].const_
    # Nothing is fitted, so fresh-fit replay is byte-identical to frozen replay.
    refit = reprocess(sb, new_df; freeze_constants=false)
    @test refit.data[:PHI_hsgp_x] == replay.data[:PHI_hsgp_x]
    @test restan_data(sb, new_df)[:PHI_hsgp_x] == replay.data[:PHI_hsgp_x]
    # A constant prediction axis is fine: there is no domain to degenerate.
    constant = reprocess(sb, periodic_df(; x=fill(6.0, PERIODIC_N)))
    @test all(r -> r == constant.data[:PHI_hsgp_x][1, :],
              eachrow(constant.data[:PHI_hsgp_x]))
    # Extrapolation past the training range is the periodic basis's point.
    far = reprocess(sb, periodic_df(; shift=1000.0))
    @test far.data[:PHI_hsgp_x] ≈ ref_basis(periodic_df(; shift=1000.0).x, 5, PERIOD)

    plan = generative_plan(periodic_builder, df; mod=@__MODULE__)
    replayed_plan = reprocess(plan, new_df)
    @test replayed_plan.data[:PHI_hsgp_x] ≈ ref_basis(new_df.x, 5, PERIOD)
    @test BayesianRegressionModels.stan_code(replayed_plan) ==
          BayesianRegressionModels.stan_code(plan)
    @test :reprocess in Symbol[op.name for op in brm_descriptor(sb).operations]
end

# -------------------------------------------------------------- runtime

@testset "BridgeStan: descriptor coordinates, term_draws, and Stan-side weights" begin
    if PERIODIC_RUNTIME
        isdir(PERIODIC_CACHE) || mkpath(PERIODIC_CACHE)
        df = periodic_df()
        exact = sbb(@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x; cov=:periodic, period=24.0)
        end)
        exact_code = code_of(exact)
        exact_prob = StanBlocks.stan_instantiate(
            exact.model; path=joinpath(PERIODIC_CACHE, string(hash(exact_code)) * ".stan"))
        exact_dimension = LogDensityProblems.dimension(exact_prob)
        qe = [0.04 * ((i % 5) - 2) for i in 1:exact_dimension]
        lpe, ge = LogDensityProblems.logdensity_and_gradient(exact_prob, qe)
        @test isfinite(lpe)
        @test all(isfinite, ge)

        d = brm_descriptor(periodic_builder, df; mod=@__MODULE__)
        prob = brm_execute(d, :fit)
        dimension = LogDensityProblems.dimension(prob)
        q = [0.05 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(prob, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)

        names = StanBlocks.BridgeStan.param_names(
            prob.model; include_tp=true, include_gq=false)
        constrained = StanBlocks.BridgeStan.param_constrain(
            prob.model, q; include_tp=true, include_gq=false)
        rho = brm_term_coordinates(d, :mu, names; term=:hsgp_x, parameter=:length_scale)
        amplitude = brm_term_coordinates(d, :mu, names; term=:hsgp_x, parameter=:sd)
        weights = brm_term_coordinates(d, :mu, names; term=:hsgp_x, parameter=:basis_weights)
        @test length(rho.coordinates) == 1
        @test length(amplitude.coordinates) == 1
        @test length(weights.coordinates) == 10
        @test weights.coordinates == findall(
            name -> startswith(name, "hsgp_x_beta_raw."), names)

        # The Stan-side Bessel weights agree with SpecialFunctions: the emitted
        # term value equals the reference basis times the reference weights.
        term_coordinates = brm_output_coordinates(d, :hsgp_x, names)
        rho_value = constrained[only(rho.coordinates)]
        sigma_value = constrained[only(amplitude.coordinates)]
        beta = constrained[weights.coordinates]
        @test rho_value >= sbb(periodic_builder(df)).data[:rho_lower_hsgp_x]
        reference = ref_basis(df.x, 5, PERIOD) * (ref_weights(5, sigma_value, rho_value) .* beta)
        @test constrained[term_coordinates] ≈ reference atol=1e-9
        @test maximum(abs, reference) > 0

        # `term_draws` zeroes exactly the 2k basis weights in unconstrained space.
        unc = StanBlocks.BridgeStan.param_unc_names(prob.model)
        draws = reshape(copy(q), 1, :)
        zeroed = term_draws(d, draws, unc; predictor=:mu, term=:hsgp_x)
        weight_unc = brm_term_coordinates(
            d, :mu, unc; term=:hsgp_x, parameter=:basis_weights).coordinates
        @test length(weight_unc) == 10
        @test all(iszero, zeroed[1, weight_unc])
        others = setdiff(1:dimension, weight_unc)
        @test zeroed[1, others] == draws[1, others]
        removed = StanBlocks.BridgeStan.param_constrain(
            prob.model, vec(zeroed); include_tp=true, include_gq=false)
        @test all(iszero, removed[term_coordinates])
    else
        @info "Skipping BridgeStan periodic GP/HSGP runtime gate (BRM_GP_RUNTIME=0)"
        @test true
    end
end
