# test/gp_hsgp_priors.jl — priors on a gp/hsgp term's own scale parameters.
#
# Run: julia --project=. test/gp_hsgp_priors.jl
# Set BRM_GP_PRIOR_RUNTIME=0 to skip the BridgeStan density/gradient probe.
#
# `test/term_priors.jl` covers the term-internal address shape for `s`, `t2`,
# `mo` and `me`. A Gaussian process owns two more private parameters, and until
# this file they had no address at all:
#
#     length_scale(<lp|:>, gp(x...))    the GP length scale `rho`
#     length_scale(<lp|:>, hsgp(x...))  ditto, for the Hilbert-space basis
#     sd(<lp|:>, gp(x...))              the marginal amplitude `sigma`
#     sd(<lp|:>, hsgp(x...))            ditto
#
# Three invariants the tests below pin down:
#
#   * The base density is REPLACED, not doubled. Two priors on one parameter is
#     a valid Stan program that transpiles, passes stanc and gives a finite
#     density, so every override test asserts the default `lognormal` is GONE —
#     the presence assertion alone would not catch it.
#   * The typed declaration SURVIVES. Three of the six gp/hsgp submodels declare
#     `rho` per-axis (`vector[n_axes]`) and three declare it as a plain scalar;
#     `Base.merge` replaces a matching-named statement WHOLESALE, so an override
#     that does not reproduce the declaration silently drops the type and the
#     constraint (StanBlocks snag `merge-plain-over-f228c5b2`).
#   * Declaration bounds and density support AGREE. A `Uniform(a, b)` length
#     scale emits `<lower=a, upper=b>` on the parameter, so the sampler cannot
#     propose outside the support — the shape a hand-written Stan port uses to
#     hold an HSGP length scale above its own approximation-validity threshold.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Exponential, Gamma, InverseGamma, LogNormal, Normal, Uniform, Beta

const GP_PRIOR_CACHE = joinpath(tempdir(), "brm-gp-priors")
const GP_PRIOR_RUNTIME = get(ENV, "BRM_GP_PRIOR_RUNTIME", "1") != "0"

const GP_PRIOR_N = 24
gp_prior_df() = (;
    y  = [sin(3i) for i in 1:GP_PRIOR_N],
    x  = collect(range(-1.0, 1.0; length=GP_PRIOR_N)),
    x2 = collect(range(-1.0, 1.0; length=GP_PRIOR_N)) .^ 2,
    g  = repeat(["a", "b"], inner=GP_PRIOR_N ÷ 2),
)

code_of(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)
transpiles_and_stanc(brmi) = begin
    sb = SBBRMI(brmi; mod=@__MODULE__)
    StanBlocks.stan.transpiles(sb.model) &&
        StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
end

# The DECLARATION of the length-scale parameter itself -- the line whose
# declared identifier ends `_rho` or `_rho_iso`, which excludes both the
# `rho_lower_*` data and the `<name>_rho = rep_vector(...)` transformed value.
# Asserting against a bare `<lower=...>` substring of the whole program does
# not work: `sigma` carries one too, so such a test keeps passing after `rho`'s
# bound changes and reports on a parameter nobody asked about.
_rho_declaration(code::AbstractString, label=nothing) = begin
    hits = collect(m.match for m in eachmatch(r"^[^\n]*\b\w*_rho(?:_iso)?;[ \t]*$"m, code))
    length(hits) == 1 || error(
        "gp_hsgp_priors: expected exactly one `rho` declaration for $label, got " *
        "$(length(hits)): $(hits). Did the parameter get renamed?")
    only(hits)
end

# The paper's bound, recomputed here from the raw axis rather than read back
# out of BRM, so the test cannot agree with the implementation by construction.
hsgp_validity_bound(x, k, c) = begin
    mu = sum(x) / length(x)
    L = c * maximum(abs, x .- mu)
    (4 * L / pi) * sqrt(log(100.0) / (k^2 - 1))
end

rho_lower_data(brmi) = begin
    d = SBBRMI(brmi; mod=@__MODULE__).data
    keys_ = filter(k -> startswith(String(k), "rho_lower_"), collect(keys(d)))
    length(keys_) == 1 || error("gp_hsgp_priors: expected one rho_lower entry, got $keys_")
    d[only(keys_)]
end

# --------------------------------------------------------------------- default

@testset "unconfigured gp/hsgp keep their historical density" begin
    df = gp_prior_df()
    unconfigured = [
        :gp => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x)
        end),
        :gp_aniso => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x, x2; iso=false)
        end),
        :hsgp => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
        end),
        :hsgp_aniso => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false)
        end),
        :hsgp_by => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=4, by=g)
        end),
    ]
    for (label, m) in unconfigured
        code = code_of(m)
        @test occursin("lognormal(0.0, 1.0)", code)
        @test !occursin("uniform(", code)
        @test transpiles_and_stanc(m)
        @test isempty(term_priors(m))
        # Assert the bound on `rho` ITSELF, by name. A bare `<lower=0.0>`
        # substring test passes on `sigma` alone, so it stayed green after
        # `hsgp` gained its validity floor and silently stopped observing the
        # thing it was written to observe.
        rho_decl = _rho_declaration(code, label)
        if startswith(String(label), "hsgp")
            # An hsgp length scale is bounded below by the approximation's
            # validity threshold, supplied as data (decision 13keyez).
            @test occursin("<lower=rho_lower_", rho_decl)
        else
            @test occursin("<lower=0.0>", rho_decl)
        end
    end
end

# ------------------------------------------- the hsgp validity floor (13keyez)

# An HSGP whose length scale falls below `(4L/pi)*sqrt(log(100)/(k^2-1))` is no
# longer approximating the kernel it was asked for -- and it fails SILENTLY: it
# transpiles, samples and returns finite draws. `hsgp` therefore declares `rho`
# with that floor by DEFAULT. Exact `gp` has no basis truncation and is
# untouched.
@testset "an unconfigured hsgp is bounded by its approximation-validity floor" begin
    df = gp_prior_df()

    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ hsgp(x; k=5, c=1.5)
    end
    code = code_of(m)
    @test occursin("<lower=rho_lower_hsgp_x>", _rho_declaration(code, :hsgp))
    # Passed as DATA, not baked into the source: `L` comes from the covariate,
    # so a non-frozen `reprocess` has to be able to move it.
    @test occursin(r"^\s*real rho_lower_hsgp_x;"m, code)
    @test rho_lower_data(m) ≈ hsgp_validity_bound(df.x, 5, 1.5)
    @test transpiles_and_stanc(m)

    # The default is a DEFAULT: an explicit declaration replaces the whole
    # statement, bound included, and the floor does not apply.
    over = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ hsgp(x; k=5, c=1.5)
        length_scale(:, hsgp(x)) ~ LogNormal(0, 1)
    end
    over_code = code_of(over)
    @test occursin("<lower=0.0>", _rho_declaration(over_code, :hsgp_overridden))
    @test !occursin("rho_lower", over_code)   # not even as unused data
    @test transpiles_and_stanc(over)

    # One shared `rho` across several axes must satisfy the STRICTEST axis.
    iso2 = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ hsgp(x, x2; k=5, c=1.5)
    end
    @test rho_lower_data(iso2) ≈ max(hsgp_validity_bound(df.x, 5, 1.5),
                                     hsgp_validity_bound(df.x2, 5, 1.5))

    # The anisotropic spelling bounds each axis separately.
    aniso = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ hsgp(x, x2; k=5, c=1.5, iso=false)
    end
    @test rho_lower_data(aniso) ≈ [hsgp_validity_bound(df.x, 5, 1.5),
                                   hsgp_validity_bound(df.x2, 5, 1.5)]
    @test occursin("<lower=rho_lower_hsgp_x_x2>", _rho_declaration(code_of(aniso), :aniso))
    @test transpiles_and_stanc(aniso)

    # `k = 1` puts the formula at infinity, which is not a declarable bound;
    # that degenerate basis stays unbounded instead of emitting an
    # impossible-to-satisfy declaration.
    k1 = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ hsgp(x; k=1, c=1.5)
    end
    @test rho_lower_data(k1) == 0.0
    @test transpiles_and_stanc(k1)

    # Exact GP has no truncation and so no floor.
    exact = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ gp(x)
    end
    @test !occursin("rho_lower", code_of(exact))
end

# --------------------------------------------- length_scale, isotropic (scalar)

@testset "length_scale on an isotropic term keeps its scalar declaration" begin
    df = gp_prior_df()

    m_gp = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + gp(x)
        length_scale(:, gp(x)) ~ InverseGamma(5, 5)
    end
    code = code_of(m_gp)
    @test occursin("real<lower=0.0> gp_x_rho;", code)
    @test occursin("gp_x_rho ~ inv_gamma(5.0, 5.0);", code)
    # `sigma` was not addressed, so it keeps the default -- and the ONE
    # remaining `lognormal` in the program is its.
    @test occursin("gp_x_sigma ~ lognormal(0.0, 1.0);", code)
    @test count(l -> occursin("lognormal", l), split(code, '\n')) == 1
    @test transpiles_and_stanc(m_gp)

    # The isotropic HSGP samples ONE length scale named `rho_iso` and broadcasts
    # it with `rep_vector`; the override has to name that parameter, not `rho`.
    m_hsgp = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ InverseGamma(5, 5)
    end
    code = code_of(m_hsgp)
    @test occursin("real<lower=0.0> hsgp_x_rho_iso;", code)
    @test occursin("hsgp_x_rho_iso ~ inv_gamma(5.0, 5.0);", code)
    @test occursin("rep_vector(hsgp_x_rho_iso", code)
    @test !occursin("hsgp_x_rho_iso ~ lognormal", code)
    @test transpiles_and_stanc(m_hsgp)
end

# ------------------------------------------ length_scale, anisotropic (vector)

@testset "length_scale on an anisotropic term keeps its per-axis declaration" begin
    df = gp_prior_df()
    for (label, m, prefix) in (
        (:gp, (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x, x2; iso=false)
            length_scale(:, gp(x, x2)) ~ LogNormal(1.0, 0.3)
        end), "gp_x_x2"),
        (:hsgp, (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false)
            length_scale(:, hsgp(x, x2)) ~ LogNormal(1.0, 0.3)
        end), "hsgp_x_x2"),
    )
        code = code_of(m)
        # The declaration a plain-LHS override would silently drop: BOTH the
        # per-axis container and the positivity constraint have to survive.
        @test occursin("vector<lower=0.0>[$(prefix)_n_axes] $(prefix)_rho;", code)
        @test occursin("$(prefix)_rho ~ lognormal(1.0, 0.3);", code)
        @test !occursin("$(prefix)_rho ~ lognormal(0.0, 1.0);", code)
        @test transpiles_and_stanc(m)
    end
end

# ---------------------------------------------------- bounded uniform + amplitude

@testset "a Uniform length scale bounds the declaration to its support" begin
    df = gp_prior_df()
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ Uniform(0.8366, 2.0)
        sd(:, hsgp(x)) ~ Normal(0, 0.5)
    end
    code = code_of(m)
    @test occursin("real<lower=0.8366, upper=2.0> hsgp_x_rho_iso;", code)
    @test occursin("hsgp_x_rho_iso ~ uniform(0.8366, 2.0);", code)
    # Amplitude addressed through `sd`, so it is a half-normal, not lognormal.
    @test occursin("real<lower=0.0> hsgp_x_sigma;", code)
    @test occursin("hsgp_x_sigma ~ normal(0.0, 0.5);", code)
    # Both parameters were configured, so NO default survives anywhere.
    @test !occursin("lognormal", code)
    @test transpiles_and_stanc(m)
end

@testset "a per-group hsgp shares one configured length scale" begin
    df = gp_prior_df()
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=4, by=g)
        length_scale(:, hsgp(x)) ~ Uniform(0.5, 2.0)
        sd(:, hsgp(x)) ~ Exponential(2.0)
    end
    code = code_of(m)
    @test occursin("real<lower=0.5, upper=2.0> hsgp_x_by_g_rho_iso;", code)
    @test occursin("hsgp_x_by_g_rho_iso ~ uniform(0.5, 2.0);", code)
    # Distributions.Exponential takes a SCALE; Stan's exponential takes a rate.
    @test occursin("hsgp_x_by_g_sigma ~ exponential((1.0 ./ 2.0));", code)
    @test !occursin("lognormal", code)
    # Only the tensor-basis weights vary per group (decision `7p44fo`); the
    # configured hyperparameters stay shared.
    @test occursin("rep_vector(hsgp_x_by_g_rho_iso", code)
    @test transpiles_and_stanc(m)
end

# ----------------------------------------------------- parameterisation mapping

@testset "Julia constructors map onto Stan's parameterisation" begin
    df = gp_prior_df()
    cases = [
        (:(Gamma(2.0, 0.5)), "gamma(2.0, (1.0 ./ 0.5))"), # scale -> rate
        (:(Exponential(4.0)), "exponential((1.0 ./ 4.0))"), # scale -> rate
        (:(InverseGamma(5, 5)), "inv_gamma(5.0, 5.0)"),  # direct
        (:(LogNormal(0.0, 0.5)), "lognormal(0.0, 0.5)"), # direct
        (:(Normal(0, 2.0)), "normal(0.0, 2.0)"),         # half-normal at lower=0
    ]
    for (call, expected) in cases
        m = @eval @brm $df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x)
            length_scale(:, gp(x)) ~ $call
        end
        code = code_of(m)
        @test occursin("gp_x_rho ~ $expected;", code)
        @test occursin("real<lower=0.0> gp_x_rho;", code)
        @test transpiles_and_stanc(m)
    end
end

# ------------------------------------------------------------------ predictors

@testset "the predictor slot resolves like every other term address" begin
    df = gp_prior_df()
    # A named linear predictor beats the `:` default layer.
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ InverseGamma(5, 5)
        length_scale(mu, hsgp(x)) ~ Uniform(0.5, 2.0)
    end
    code = code_of(m)
    @test occursin("hsgp_x_rho_iso ~ uniform(0.5, 2.0);", code)
    @test !occursin("inv_gamma", code)
    @test transpiles_and_stanc(m)
end

# ------------------------------------------------------------------ inspection

@testset "term_priors reports the new class" begin
    df = gp_prior_df()
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ Uniform(0.8366, 2.0)
        sd(mu, hsgp(x)) ~ Normal(0, 0.5)
    end
    specs = term_priors(m)
    @test length(specs) == 2
    ls = only(p for p in specs if p.class === :term_length_scale)
    @test ls.term === Symbol("hsgp(x)")
    @test isnothing(ls.predictor)
    @test isnothing(ls.component)
    @test ls.family === Uniform
    @test ls.arguments == (0.8366, 2.0)
    amp = only(p for p in specs if p.class === :term_sd)
    @test amp.term === Symbol("hsgp(x)")
    @test amp.predictor === :mu
end

# -------------------------------------------------------------------- refusals

@testset "mis-addressed gp/hsgp scale priors are refused by name" begin
    df = gp_prior_df()
    # Parse-time refusals: the target slot must be a term, not a bare symbol.
    @test_throws "must be a term as the formula spells it" (@eval @brm $df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, g) ~ Uniform(0.5, 2.0)
    end)

    # Resolve-time refusals: the class reaches a term with no such parameter, or
    # explicit bounds are not usable for a positive scale.
    generic = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ Beta(2, 2)
    end
    generic_code = code_of(generic)
    @test occursin("real<lower=0.0, upper=1> hsgp_x_rho_iso;", generic_code)
    @test occursin("hsgp_x_rho_iso ~ beta(2.0, 2.0);", generic_code)
    @test transpiles_and_stanc(generic)

    cases = [
        ("has no length scale to configure", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + s(x)
            length_scale(:, s(x)) ~ Uniform(0.5, 2.0)
        end)),
        ("0 <= lower < upper", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
            length_scale(:, hsgp(x)) ~ Uniform(-1.0, 2.0)
        end)),
        ("matches no `hsgp(x)` term", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x)
            length_scale(:, hsgp(x)) ~ Uniform(0.5, 2.0)
        end)),
    ]
    for (fragment, m) in cases
        err = try
            code_of(m)
            nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin(fragment, sprint(showerror, err))
    end
end

# -------------------------------------------------------------------- runtime

@testset "BridgeStan finite density and gradient" begin
    if GP_PRIOR_RUNTIME
        df = gp_prior_df()
        m = @brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
            length_scale(:, hsgp(x)) ~ Uniform(0.8366, 2.0)
            sd(:, hsgp(x)) ~ Normal(0, 0.5)
        end
        sb = SBBRMI(m; mod=@__MODULE__)
        isdir(GP_PRIOR_CACHE) || mkpath(GP_PRIOR_CACHE)
        code = StanBlocks.stan_code(sb.model)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(GP_PRIOR_CACHE, string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping BridgeStan gp/hsgp prior runtime gate (BRM_GP_PRIOR_RUNTIME=0)"
        @test true
    end
end
