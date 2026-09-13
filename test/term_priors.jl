# test/term_priors.jl — priors on a formula TERM's own internal parameters.
#
# Run: julia --project=. test/term_priors.jl
# Set BRM_TERM_PRIOR_RUNTIME=0 to skip the BridgeStan density/gradient probe.
#
# The population surface addresses a COEFFICIENT (`effect(mu, x)`) and the
# random-effect surface a GROUPING FACTOR (`sd(:, subject)`). Neither can reach
# a parameter a term owns privately inside its submodel — the smoothing scale of
# `s(x)`, the Dirichlet increments of `mo(c)`, the latent covariate of `me(x)`.
# This file covers the third address shape, whose target slot is the term as the
# formula spells it:
#
#     sd(<lp|:>, s(x))                 smoothing scale of a thin-plate smooth
#     sd(<lp|:>, t2(x, z), <block>)    one of the three tensor penalties
#     simplex(<lp|:>, mo(c))           monotonic-effect Dirichlet concentration
#     latent(<lp|:>, me(x))            latent true covariate of a me() term
#
# Two invariants the tests below pin down:
#
#   * The submodels are PARAMETERIZED, not duplicated. A configured and an
#     unconfigured `s(x)` reach the same `_sb_s`; only the values Julia supplies
#     for `sd_family` / `sd_rate` differ. So the default is written down once,
#     in the Julia-side emitter, rather than as a second copy of the submodel.
#   * Only MODEL-SCALE quantities are exposed. `b_pen_raw` and friends stay iid
#     standard normal — scaling a standardized innovation would duplicate the
#     smoothing SD and change the advertised parameterization (decision
#     `145tp0o`).

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Cauchy, Dirichlet, Exponential, Normal

const TERM_PRIOR_CACHE = joinpath(tempdir(), "brm-term-priors")
const TERM_PRIOR_RUNTIME = get(ENV, "BRM_TERM_PRIOR_RUNTIME", "1") != "0"

const TERM_N = 40
term_df() = (;
    y  = [sin(i) for i in 1:TERM_N],
    x  = [cos(i) for i in 1:TERM_N],
    z  = [cos(2i) for i in 1:TERM_N],
    xo = [sin(3i) for i in 1:TERM_N],
    c  = [1 + (i % 4) for i in 1:TERM_N],
    subject = repeat(1:8; inner=5),
)

code_of(brmi) = BayesianRegressionModels.stan_code(SBBRMI(brmi; mod=@__MODULE__))
transpiles_and_stanc(brmi) = begin
    sb = SBBRMI(brmi; mod=@__MODULE__)
    StanBlocks.stan.transpiles(sb.model) &&
        StanBlocks.stanc_check(BayesianRegressionModels.stan_code(sb); warn_pedantic=false).ok
end

# --------------------------------------------------------------------- default

@testset "unconfigured terms keep their historical density" begin
    unconfigured = [
        :s => (@brm term_df() begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + s(x)
        end),
        :t2 => (@brm term_df() begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + t2(x, z)
        end),
        :mo1 => (@brm term_df() begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + mo1(c)
        end),
        :mo => (@brm term_df() begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + mo(c)
        end),
        :me => (@brm term_df() begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + me(xo, 0.3)
        end),
    ]
    expected = Dict(
        # family 0 is the half-standard-normal; the rate is then unused.
        :s   => "s_x_sd_pen ~ std_normal();",
        :t2  => "t2_mu_x_z_sd_pen ~ std_normal();",
        # `c` has four levels, so the increment simplex has three.
        :mo1 => "mo1_c_simplex_incr ~ dirichlet(rep_vector(1.0, 3));",
        :mo  => "mo_c_simplex_incr ~ dirichlet(rep_vector(1.0, 3));",
        :me  => "me_xo_x_true ~ normal(0.0, 1.0);",
    )
    for (label, m) in unconfigured
        @test isempty(term_priors(m))
        code = code_of(m)
        @test occursin(expected[label], code)
        @test transpiles_and_stanc(m)
    end
end

# ----------------------------------------------------------------- smooth `sd`

@testset "sd(<lp|:>, s(x)) sets the smoothing scale" begin
    m = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + s(x)
        sd(:, s(x)) ~ Exponential(2)
    end

    spec = only(term_priors(m))
    @test spec.class === :term_sd
    @test spec.term === Symbol("s(x)")
    @test spec.predictor === nothing        # `:` — the default layer
    @test spec.component === nothing
    @test spec.family <: Exponential

    code = code_of(m)
    # Distributions.Exponential is scale-parameterized, Stan's is rate: 1/2.
    @test occursin(r"s_x_sd_pen ~ brm_vector_prior_[0-9a-f]+", code)
    # The penalized coefficients stay standardized (decision `145tp0o`).
    @test occursin("s_x_b_pen_raw ~ std_normal();", code)
    @test occursin("vector<lower=0.0>[1] s_x_sd_pen;", code)
    @test transpiles_and_stanc(m)
end

@testset "smoothing scales accept an arbitrary translated scalar prior" begin
    m = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + s(x)
        sd(:, s(x)) ~ Cauchy(0, 2)
    end
    code = code_of(m)
    @test occursin("cauchy_lpdf(x[1] | arg_1, arg_2)", code)
    @test occursin("vector<lower=0.0>[1] s_x_sd_pen;", code)
    @test transpiles_and_stanc(m)
end

@testset "sd(<lp|:>, t2(x, z), <block>) sets one tensor penalty" begin
    m = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + t2(x, z)
        sd(mu, t2(x, z), rr) ~ Exponential(3)
        sd(:, t2(x, z), nr) ~ Exponential(0.5)
    end

    specs = term_priors(m)
    @test [s.component for s in specs] == [:rr, :nr]
    @test [s.predictor for s in specs] == [:mu, nothing]
    @test all(s -> s.term === Symbol("t2(x,z)"), specs)

    code = code_of(m)
    # (rr, rn, nr) is the sampled order; the unmentioned `rn` keeps family 0.
    @test occursin(r"t2_mu_x_z_sd_pen ~ brm_vector_prior_[0-9a-f]+", code)
    @test occursin("vector<lower=0.0>[3] t2_mu_x_z_sd_pen;", code)
    @test transpiles_and_stanc(m)
end

# ------------------------------------------------------------ monotonic effect

@testset "simplex(<lp|:>, mo(c)) sets the Dirichlet concentration" begin
    symmetric = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + mo1(c)
        simplex(:, mo1(c)) ~ Dirichlet(2)
    end
    @test only(term_priors(symmetric)).class === :term_simplex
    @test occursin("mo1_c_simplex_incr ~ dirichlet(rep_vector(2.0, 3));",
                   code_of(symmetric))
    @test transpiles_and_stanc(symmetric)

    # One concentration per increment, in increment order.
    elementwise = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + mo(c)
        simplex(mu, mo(c)) ~ Dirichlet(1, 2, 3)
    end
    @test occursin("mo_c_simplex_incr ~ dirichlet([1.0, 2.0, 3.0]');",
                   code_of(elementwise))
    @test transpiles_and_stanc(elementwise)

    sampled = @brm term_df() begin
        concentration ~ Exponential(1)
        y ~ Normal(mu, 1.)
        mu ~ 1 + mo(c)
        simplex(mu, mo(c)) ~ Dirichlet(concentration)
    end
    sampled_code = code_of(sampled)
    @test occursin("concentration ~ exponential", sampled_code)
    @test occursin("mo_c_simplex_incr ~ dirichlet(rep_vector(concentration, 3));",
                   sampled_code)
    @test transpiles_and_stanc(sampled)
end

# --------------------------------------------------------------- measurement error

@testset "latent(<lp|:>, me(x)) sets the latent covariate prior" begin
    m = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + me(xo, 0.3)
        latent(:, me(xo)) ~ Normal(1, 4)
    end
    spec = only(term_priors(m))
    @test spec.class === :term_latent
    @test spec.term === Symbol("me(xo)")    # the numeric `sd` arg is not part of the address

    code = code_of(m)
    @test occursin("me_xo_x_true ~ normal(1, 4);", code)
    # The observation likelihood is never configurable.
    @test occursin("xo ~ normal(me_xo_x_true, sd_xo);", code)
    @test transpiles_and_stanc(m)

    bounded = @brm term_df() begin
        center ~ Normal(0, 1)
        y ~ Normal(mu, 1.)
        mu ~ 1 + me(xo, 0.3)
        latent(mu, me(xo)) ~ Cauchy(center, 1; lower=-0.5, upper=0.5)
    end
    bounded_code = code_of(bounded)
    @test occursin("vector<lower=-0.5, upper=0.5>[num_elements(xo)] me_xo_x_true;",
                   bounded_code)
    @test occursin("me_xo_x_true ~ cauchy(center, 1);", bounded_code)
    @test transpiles_and_stanc(bounded)
end

@testset "one me latent column is reused by population and random slopes" begin
    m = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + me(xo, 0.3) + (1 + me(xo, 0.3) | p | subject)
        latent(mu, me(xo)) ~ Normal(0, 2)
    end

    @test popcoefnames(m, :mu) == [:Intercept, :me_xo]
    @test ranefcoefnames(m, :p) == [
        (predictor=:mu, coefficient=:Intercept),
        (predictor=:mu, coefficient=:me_xo),
    ]

    code = code_of(m)
    @test count(line -> occursin("me_xo_x_true ~ normal(0, 2);", line),
                eachline(IOBuffer(code))) == 1
    @test transpiles_and_stanc(m)
end

# ------------------------------------------------------------------ precedence

@testset "`:` is the default layer a named predictor overrides" begin
    # `t2` is the only one of these terms that namespaces its emitted column by
    # linear predictor, so it is the only one that can carry ONE term key in TWO
    # predictors at once -- which is what makes `:` distinguishable from a named
    # predictor. (`s`/`me`/`mo`/`mo1` name their column from the data column
    # alone, so the same term in two predictors collides in StanBlocks before
    # any prior is involved; that predates this surface.)
    m = @brm term_df() begin
        y ~ Normal(mu + nu, 1.)
        mu ~ 1 + t2(x, z)
        nu ~ 0 + t2(x, z)
        sd(:, t2(x, z), rr) ~ Exponential(2)
        sd(nu, t2(x, z), rr) ~ Exponential(8)
    end
    code = code_of(m)
    # mu takes the `:` default layer; nu's own statement is strictly more
    # specific and wins there.
    @test occursin(r"t2_mu_x_z_sd_pen ~ brm_vector_prior_[0-9a-f]+", code)
    @test occursin(r"t2_nu_x_z_sd_pen ~ brm_vector_prior_[0-9a-f]+", code)
    @test transpiles_and_stanc(m)
end

@testset "several terms in one predictor stay independent" begin
    m = @brm term_df() begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + s(x) + mo(c) + me(xo, 0.3)
        sd(:, s(x)) ~ Exponential(4)
        latent(mu, me(xo)) ~ Normal(0, 10)
    end
    code = code_of(m)
    @test occursin(r"s_x_sd_pen ~ brm_vector_prior_[0-9a-f]+", code)
    @test occursin("me_xo_x_true ~ normal(0, 10);", code)
    # Unaddressed, so untouched.
    @test occursin("mo_c_simplex_incr ~ dirichlet(rep_vector(1.0, 3));", code)
    @test transpiles_and_stanc(m)
end

# ---------------------------------------------------------------- refusals

@testset "a term address that cannot be honoured is refused by name" begin
    cases = [
        # (what the statement gets wrong, a distinctive fragment of the message)
        "sd(:, s(z)) ~ Exponential(2)"          => "matches no `s(z)` term in any linear predictor",
        "sd(mu, s(z)) ~ Exponential(2)"         => "matches no `s(z)` term in `mu`",
        "simplex(:, s(x)) ~ Dirichlet(2)"       => "`s` has no simplex to configure",
        "latent(:, s(x)) ~ Normal(0, 1)"        => "`s` has no latent covariate to configure",
    ]
    for (stmt, fragment) in cases
        expr = Meta.parse("""
            code_of(@brm term_df() begin
                y ~ Normal(mu, 1.)
                mu ~ 1 + s(x)
                $stmt
            end)""")
        err = try
            @eval($expr); nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin(fragment, sprint(showerror, err))
    end

    # A tensor smooth has three scales, so the component slot is mandatory —
    # and must name a real block.
    t2_cases = [
        "sd(:, t2(x, z)) ~ Exponential(2)"      => "a tensor smooth has three",
        "sd(:, t2(x, z), qq) ~ Exponential(2)"  => "names no penalty block",
    ]
    for (stmt, fragment) in t2_cases
        expr = Meta.parse("""
            code_of(@brm term_df() begin
                y ~ Normal(mu, 1.)
                mu ~ 1 + t2(x, z)
                $stmt
            end)""")
        err = try
            @eval($expr); nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin(fragment, sprint(showerror, err))
    end

    # `c` has four levels, so three increments — two concentrations name neither
    # one symmetric value nor one per increment.
    bad_arity = Meta.parse("""
        code_of(@brm term_df() begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + mo1(c)
            simplex(:, mo1(c)) ~ Dirichlet(1, 2)
        end)""")
    err = try
        @eval($bad_arity); nothing
    catch e
        e
    end
    @test !isnothing(err)
    @test occursin("either one concentration or 3 of them", sprint(showerror, err))

    # Two spellings of one key make the address ambiguous rather than picking
    # whichever copy the walker reached first.
    ambiguous = Meta.parse("""
        code_of(@brm term_df() begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + s(x) + s(x)
            sd(:, s(x)) ~ Exponential(2)
        end)""")
    err = try
        @eval($ambiguous); nothing
    catch e
        e
    end
    @test !isnothing(err)
    @test occursin("carries 2 terms spelled `s(x)`", sprint(showerror, err))
end

@testset "the heads do not blur into each other at parse time" begin
    # A term is a call, a grouping factor a bare symbol, so `sd` tells them
    # apart unambiguously -- but `effect`/`cor` address neither.
    for (stmt, fragment) in [
        "effect(:, s(x)) ~ Normal(0, 1)" =>
            "addresses a coefficient or a grouping factor, not a term's own parameters",
        "cor(:, s(x)) ~ Normal(0, 1)" =>
            "addresses a coefficient or a grouping factor, not a term's own parameters",
        # `simplex`/`latent` take exactly two slots; there is no component.
        "simplex(:, mo1(c), rr) ~ Dirichlet(2)" =>
            "takes exactly `simplex(<linear_predictor|:>, <term>)`",
    ]
        expr = Meta.parse("""
            @brm term_df() begin
                y ~ Normal(mu, 1.)
                mu ~ 1 + s(x) + mo1(c)
                $stmt
            end""")
        err = try
            @eval($expr); nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin(fragment, sprint(showerror, err))
    end
end

# ------------------------------------------------------------------- runtime

@testset "BridgeStan finite density and gradient" begin
    if TERM_PRIOR_RUNTIME
        m = @brm term_df() begin
            concentration ~ Exponential(1)
            center ~ Normal(0, 1)
            y ~ Normal(mu, 1.)
            mu ~ 1 + s(x) + t2(x, z) + mo(c) + me(xo, 0.3) +
                 (1 + me(xo, 0.3) | p | subject)
            sd(:, s(x)) ~ Exponential(2)
            sd(:, t2(x, z), rr) ~ Exponential(3)
            simplex(:, mo(c)) ~ Dirichlet(concentration)
            latent(:, me(xo)) ~ Cauchy(center, 1; lower=-5, upper=5)
        end
        sb = SBBRMI(m; mod=@__MODULE__)
        isdir(TERM_PRIOR_CACHE) || mkpath(TERM_PRIOR_CACHE)
        code = StanBlocks.stan_code(sb.model)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(TERM_PRIOR_CACHE, string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping BridgeStan term-prior runtime gate (BRM_TERM_PRIOR_RUNTIME=0)"
        @test true
    end
end
