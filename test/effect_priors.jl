# test/effect_priors.jl — named population-coefficient prior overrides.
#
# Run: julia --project=. test/effect_priors.jl
# Set BRM_EFFECT_RUNTIME=0 to skip the BridgeStan density/gradient probe.

# These fixtures inspect conventional emitted parameter names.
# Exact total-prior equivalence is covered in total_effects_integration.jl.
using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Cauchy, Dirichlet, Exponential, LKJCholesky, Normal, Uniform

const EFFECT_PRIOR_CACHE = joinpath(tempdir(), "brm-effect-priors")
const EFFECT_PRIOR_RUNTIME = get(ENV, "BRM_EFFECT_RUNTIME", "1") != "0"

df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        subject=[1, 1, 2, 2, 3, 3],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

builder = @brm begin
    log_ka ~ 1 + x + (1 | pk | subject)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, x) ~ Normal(0, 0.1)
    y ~ Normal(log_ka, 0.2)
end

@testset "effect prior capture and lowering" begin
    brmi = builder(df)
    @test popcoefnames(brmi, :log_ka) == [:Intercept, :x]
    @test length(linear_predictors(brmi)) == 1

    priors = effect_priors(brmi)
    @test length(priors) == 2
    @test [(p.predictor, p.coefficient) for p in priors] ==
          [(:log_ka, :Intercept), (:log_ka, :x)]
    @test all(p -> p.family <: Normal, priors)
    @test all(isempty(p.keywords) for p in priors)
    @test occursin("effect(log_ka, Intercept) ~ Normal", sprint(show, brmi))

    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("vector[pop_log_ka_n_covariates] pop_log_ka_beta_pop;", code)
    @test occursin("pop_log_ka_beta_pop ~ normal(", code)
    @test occursin(string(log(1 / 8)), code)
    @test occursin("[0.8, 0.1]'", code)

    plan = generative_plan(sb)
    pop_decl = only(d for d in plan.declarations if d.target === :pop_log_ka)
    @test pop_decl.family === :_popefs_normal
    @test pop_decl.role === :prior

    descriptor = brm_descriptor(sb)
    byname = Dict(o.name => o for o in descriptor.outputs)
    @test byname[:pop_log_ka].role === :population_effect
    @test byname[:pop_log_ka_beta_pop].role === :population_effect
    @test byname[:pop_log_ka_beta_pop].labels == [:Intercept, :x]

    if EFFECT_PRIOR_RUNTIME
        isdir(EFFECT_PRIOR_CACHE) || mkpath(EFFECT_PRIOR_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model;
            path=joinpath(EFFECT_PRIOR_CACHE, string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping effect-prior BridgeStan gate (BRM_EFFECT_RUNTIME=0)"
    end
end

@testset "bounded scalar priors retain their Stan declaration support" begin
    bounded = @brm df begin
        sigma ~ Normal(0, 2; lower=0.0)
        y ~ Normal(0, sigma)
    end
    bounded_sb = SBBRMI(bounded; mod=@__MODULE__, total_groups=())
    bounded_code = BayesianRegressionModels.stan_code(bounded_sb)
    @test occursin("real<lower=0.0> sigma;", bounded_code)
    @test occursin("sigma ~ normal(0, 2);", bounded_code)
    @test StanBlocks.stanc_check(bounded_code; warn_pedantic=false).ok

    @test_throws "requires `lower < upper`" SBBRMI((@brm df begin
        sigma ~ Normal(0, 2; lower=1.0, upper=0.0)
        y ~ Normal(0, sigma)
    end); mod=@__MODULE__, total_groups=())
    @test_throws "constructor keywords (:typo,)" SBBRMI((@brm df begin
        sigma ~ Normal(0, 2; typo=0.0)
        y ~ Normal(0, sigma)
    end); mod=@__MODULE__, total_groups=())
end

@testset "sampled scalar parameter in an effect-prior argument" begin
    sampled_scale = @brm df begin
        slope_prior_sd ~ Exponential(10.0)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0.0, slope_prior_sd)
        y ~ Normal(mu, 1.0)
    end

    sb = SBBRMI(sampled_scale; mod=@__MODULE__, total_groups=())
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("slope_prior_sd ~ exponential((1.0 ./ 10.0));", code)
    @test occursin(
        "pop_mu_beta_pop ~ normal([0.0, 0.0]', [1.0, slope_prior_sd]');",
        code)
    @test first(findfirst("slope_prior_sd ~ exponential", code)) <
          first(findfirst("pop_mu_beta_pop ~ normal", code))

    # One scalar coefficient cannot consume a whole observation vector as its SD.
    data_arg = @brm df begin
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0.0, y)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "model-level prior references `y`" SBBRMI(data_arg; mod=@__MODULE__, total_groups=())

    # A vector-valued prior declaration has the same sampling-expression
    # carrier, but its whole value is not a scalar Normal scale.
    vector_arg = @brm df begin
        simplex_scale ~ Dirichlet([1.0, 1.0])
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0.0, simplex_scale)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "needs scalar location and scale" SBBRMI(
        vector_arg; mod=@__MODULE__, total_groups=())

    if EFFECT_PRIOR_RUNTIME
        isdir(EFFECT_PRIOR_CACHE) || mkpath(EFFECT_PRIOR_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model;
            path=joinpath(EFFECT_PRIOR_CACHE, string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping sampled-effect-prior BridgeStan gate (BRM_EFFECT_RUNTIME=0)"
    end
end

@testset "defaults, `:` default layer, and validation" begin
    default = @brm df begin
        mu ~ 1 + x
        y ~ Normal(mu, 1)
    end
    @test occursin("pop_mu_beta_pop ~ std_normal();",
                   BayesianRegressionModels.stan_code(SBBRMI(default; mod=@__MODULE__, total_groups=())))

    # `:` in the predictor slot is the default layer: it reaches every
    # predictor owning that column, so a single-predictor model reads exactly
    # like the removed concise form did.
    colon_lp = @brm df begin
        mu ~ 1 + x
        effect(:, x) ~ Normal(0, 0.25)
        y ~ Normal(mu, 1)
    end
    @test occursin("[1.0, 0.25]'",
                   BayesianRegressionModels.stan_code(SBBRMI(colon_lp; mod=@__MODULE__, total_groups=())))

    # Reaching several predictors is the POINT of `:`, not an ambiguity: the old
    # concise form errored here, the default layer sets both.
    both = @brm df begin
        a ~ 1
        b ~ 1
        effect(:, Intercept) ~ Normal(0, 2)
        y ~ Normal(a + b, 1)
    end
    both_code = BayesianRegressionModels.stan_code(SBBRMI(both; mod=@__MODULE__, total_groups=()))
    @test occursin("pop_a_beta_pop ~ normal([0]', [2]');", both_code)
    @test occursin("pop_b_beta_pop ~ normal([0]', [2]');", both_code)

    excluded = @brm df begin
        mu ~ 1 + x + (1 | subject)
        effect(mu, subject) ~ Normal(0, 0.5)
        y ~ Normal(mu, 1)
    end
    @test_throws "not a population coefficient" SBBRMI(excluded; mod=@__MODULE__, total_groups=())

    generic = @brm df begin
        mu ~ 1 + x
        effect(mu, x) ~ Cauchy(0, 1)
        y ~ Normal(mu, 1)
    end
    generic_sb = SBBRMI(generic; mod=@__MODULE__, total_groups=())
    generic_code = BayesianRegressionModels.stan_code(generic_sb)
    @test occursin("cauchy_lpdf(x[2] | arg_3, arg_4)", generic_code)
    @test StanBlocks.stanc_check(generic_code; warn_pedantic=false).ok
    generic_output = only(o for o in brm_descriptor(generic_sb).outputs
                          if o.name === :pop_mu_beta_pop)
    @test generic_output.role === :population_effect
    @test generic_output.labels == [:Intercept, :x]

    bounded_vector = @brm df begin
        mu ~ 1 + x
        effect(mu, :) ~ Uniform(0.2, 0.8)
        y ~ Normal(mu, 1)
    end
    bounded_vector_code = BayesianRegressionModels.stan_code(
        SBBRMI(bounded_vector; mod=@__MODULE__, total_groups=()))
    @test occursin("vector<lower=0.2, upper=0.8>", bounded_vector_code)
    @test occursin("uniform_lpdf(x[1] | arg_1, arg_2)", bounded_vector_code)
    @test StanBlocks.stanc_check(bounded_vector_code; warn_pedantic=false).ok

    heterogeneous_bounds = @brm df begin
        mu ~ 1 + x
        effect(mu, Intercept) ~ Uniform(0.2, 0.8)
        effect(mu, x) ~ Uniform(0.3, 0.9)
        y ~ Normal(mu, 1)
    end
    heterogeneous_code = BayesianRegressionModels.stan_code(
        SBBRMI(heterogeneous_bounds; mod=@__MODULE__, total_groups=()))
    @test occursin("vector[pop_mu_n_covariates] pop_mu_beta_pop;", heterogeneous_code)
    @test occursin("if((x[1] < 0.2))", heterogeneous_code)
    @test occursin("if((x[2] > 0.9))", heterogeneous_code)
    @test StanBlocks.stanc_check(heterogeneous_code; warn_pedantic=false).ok

    prior_only = @brm df begin
        mu ~ 1 + x
        effect(mu, :) ~ Uniform(0.2, 0.8)
        y ~ Normal(0, 1)
    end
    prior_only_sb = SBBRMI(prior_only; mod=@__MODULE__, total_groups=())
    prior_only_code = BayesianRegressionModels.stan_code(prior_only_sb)
    @test occursin(r"pop_mu_beta_pop = brm_vector_prior_[0-9a-f]+_vector_rng",
                   prior_only_code)
    if EFFECT_PRIOR_RUNTIME
        problem = StanBlocks.stan_instantiate(
            prior_only_sb.model;
            path=joinpath(EFFECT_PRIOR_CACHE, string(hash(prior_only_code)) * ".stan"))
        @test LogDensityProblems.dimension(problem) == 0
        names = StanBlocks.BridgeStan.param_names(
            problem.model; include_tp=true, include_gq=true)
        draw = StanBlocks.BridgeStan.param_constrain(
            problem.model, Float64[]; include_tp=true, include_gq=true,
            rng=StanBlocks.BridgeStan.StanRNG(problem.model, 417))
        beta = [v for (nm, v) in zip(names, draw)
                if startswith(String(nm), "pop_mu_beta_pop.")]
        @test length(beta) == 2
        @test all(x -> 0.2 <= x <= 0.8, beta)
    end
end

# The head-position grammar's refusals all fire in the PARSER, before any
# backend runs, so a mis-shaped address can never reach lowering.
@testset "prior-address grammar refusals" begin
    # Both `effect` slots are mandatory; the concise predictor-inferring form
    # is gone, and its removal is what the diagnostic must say.
    @test_throws "concise predictor-inferring form was removed" eval(quote
        @brm begin
            mu ~ 1 + x
            effect(x) ~ Normal(0, 1)
            y ~ Normal(mu, 1)
        end
    end)

    # Correlation priors are block-wide by construction: one shared `|ID|`
    # covariance block spans every predictor that slices it, so there is no
    # per-predictor correlation to address.
    @test_throws "correlation priors are block-wide" eval(quote
        @brm begin
            eta ~ 1 + (1 + x | p | subject)
            cor(eta, p) ~ LKJCholesky(2, 2)
            y ~ Normal(eta, 1)
        end
    end)

    # Collections in an address slot are DEFERRED, not rejected on principle,
    # so they must fail by name rather than fall through to the generic
    # bare-symbol error -- adding them later stays purely additive.
    @test_throws "are not supported yet" eval(quote
        @brm begin
            mu ~ 1 + x + z
            effect(mu, (x, z)) ~ Normal(0, 1)
            y ~ Normal(mu, 1)
        end
    end)
    @test_throws "are not supported yet" eval(quote
        @brm begin
            mu ~ 1 + x + z
            effect(mu, [x, z]) ~ Normal(0, 1)
            y ~ Normal(mu, 1)
        end
    end)

    # The reserved prior-address heads are exactly `effect`, `sd`, `cor`,
    # `simplex`, `latent` and `length_scale`. Anything else is NOT claimed by
    # the prior grammar — it falls through to ordinary formula handling and
    # fails there, on its own terms, naming the unknown function rather than any
    # prior-address diagnostic. That is deliberate, and it is what made each
    # term-internal head purely additive as it arrived: `simplex` / `latent`
    # with `cf6ebfb`, `length_scale` with the gp/hsgp scale surface.
    not_a_prior_head = @brm begin
        mu ~ 1 + x
        nugget(mu, x) ~ InverseGamma(5, 5)
        y ~ Normal(mu, 1)
    end
    @test_throws "nugget" not_a_prior_head(df)

    # A claimed head, in contrast, is refused BY the grammar: `length_scale`
    # addresses a term's own parameter, so a bare coefficient in its target slot
    # fails at parse time with a diagnostic naming the shape it wanted.
    @test_throws "must be a term as the formula spells it" eval(quote
        @brm begin
            mu ~ 1 + x
            length_scale(mu, x) ~ InverseGamma(5, 5)
            y ~ Normal(mu, 1)
        end
    end)
end

# `linear_predictors` reports EVERY non-data `~` binding, not just population
# formulas: a scalar parameter prior (`log_F_bottle ~ Normal(0, 0.5)`) and a
# submodel binding (`pred ~ kernel(...) do … end`) are both in it, and neither
# has `beta_pop` columns for `popcoefnames` to name. Resolving labels for every
# entry the moment ANY effect prior existed therefore made one unrelated
# statement fatal for the whole model — `effect(...)` overrides and ordinary
# scalar priors could not coexist. Labels are now resolved lazily, per address.
@testset "unrelated statements do not block effect-prior resolution" begin
    explicit = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test [lp.name for lp in linear_predictors(explicit)] == [:log_F_bottle, :mu]
    @test popcoefnames(explicit, :mu) == [:Intercept, :x]
    # The scalar prior is exactly the entry `popcoefnames` cannot name.
    @test_throws "cannot resolve the `beta_pop` column(s)" popcoefnames(explicit, :log_F_bottle)

    sb = SBBRMI(explicit; mod=@__MODULE__, total_groups=())
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    # The scalar prior stays an ordinary parameter…
    @test occursin("real log_F_bottle;", code)
    @test occursin("log_F_bottle ~ normal(0, 0.5);", code)
    # …and the override still reaches only the addressed population column.
    @test occursin("pop_mu_beta_pop ~ normal([0.0, 0]', [1.0, 0.25]');", code)

    # `:` is the path that enumerates candidates, so it must skip the
    # unnameable entries rather than fail on them.
    colon_lp = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(:, x) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test occursin("pop_mu_beta_pop ~ normal([0.0, 0]', [1.0, 0.25]');",
                   BayesianRegressionModels.stan_code(SBBRMI(colon_lp; mod=@__MODULE__, total_groups=())))

    # Addressing the scalar prior itself is still an error, and names what IS
    # addressable instead of leaking `popcoefnames`' internal failure.
    at_prior = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(log_F_bottle, x) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test_throws "names no linear predictor with population coefficients" SBBRMI(
        at_prior; mod=@__MODULE__, total_groups=())

    # An unknown label is still rejected — tolerance must not mute this.
    unknown = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(:, nope) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test_throws "matches no population coefficient" SBBRMI(unknown; mod=@__MODULE__, total_groups=())
end

# The PMX shape the snag was reported against: a population PK model carries a
# scalar prior (`sigma ~ Exponential(1)`) AND a `kernel(...)` submodel binding
# whose columns `popcoefnames` cannot name — two distinct unnameable entries —
# alongside `effect(...)` overrides on the per-subject linear predictors.
@testset "effect priors coexist with a kernel(...) PMX model" begin
    n = 6
    pk_df = (; t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:n],
               dose    = fill(100.0, n),
               dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:n],
               subject = collect(1:n))

    pk = @brm pk_df begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + (1 | p | subject)
        log_V  ~ 1 + (1 | p | subject)
        log_Ka ~ 1 + (1 | p | subject)
        effect(log_CL, Intercept) ~ Normal(log(1 / 8), 0.8)
        pred ~ kernel(t, dose, dv, log_CL, log_V, log_Ka) do ts, d, yy, lCL, lV, lKa
            CL = exp(lCL); Vc = exp(lV); Ka = exp(lKa)
            ke = CL / Vc
            mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
            yy ~ normal(mu, sigma)
            mu
        end
    end
    @test_throws "cannot resolve the `beta_pop` column(s)" popcoefnames(pk, :sigma)
    @test_throws "cannot resolve the `beta_pop` column(s)" popcoefnames(pk, :pred)
    @test popcoefnames(pk, :log_CL) == [:Intercept]

    sb = SBBRMI(pk; mod=@__MODULE__, total_groups=())
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("pop_log_CL_beta_pop ~ normal([$(log(1 / 8))]', [0.8]');", code)
    # Unaddressed predictors keep the default, and the scalar prior is untouched.
    @test occursin("pop_log_V_beta_pop ~ std_normal();", code)
    @test occursin("sigma ~ exponential(", code)
end

# An LHS LINK TRANSFORMATION (`log(Vc) ~ 1 + x`, §2 of `brm-use`) and the
# `effect(...)` prior surface each had their own coverage, and NOTHING composed
# them — so the coefficient label for a linked declaration was undocumented and
# a consumer moving off the inert-name spelling had to guess between
# `effect(Vc, …)`, `effect(log(Vc), …)` and `effect(log_Vc, …)`
# (snag `how-do-you-attac-a30ba40b`, a downstream PKPD consumer). Two names are in play and
# they are NOT interchangeable:
#
#   PUBLIC  address  = the bare inner name `Vc` — what `linear_predictors`
#                      reports, and what `popcoefnames` / `effect(...)` /
#                      `ranefcoefnames` take.
#   EMITTED artefact = the LINKED spelling `log_Vc` — `pop_log_Vc_beta_pop`,
#                      `X_log_Vc`, `Z_log_Vc_p_subject`.
#
# The emitted half is what makes the switch posterior-preserving: it is
# byte-identical to the inert-name twin, so the same priors attach to the same
# coefficients and a cache keyed on emitted source keeps its identity.
@testset "LHS link transformation composes with `effect(...)`" begin
    link_df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                 subject=[1, 1, 2, 2, 3, 3],
                 y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

    linked = @brm link_df begin
        log(Vc) ~ 1 + x + (1 + x | p | subject)
        effect(Vc, Intercept) ~ Normal(log(10), 0.8)
        effect(Vc, x) ~ Normal(0, 0.1)
        sd(:, p) ~ Exponential(0.3)
        cor(:, p) ~ LKJCholesky(2, 2.0)
        y ~ Normal(log(Vc), 0.2)
    end

    # The public address is the BARE name; the link is reported beside it.
    @test [(l.name, l.link_lhs_fn) for l in linear_predictors(linked)] == [(:Vc, log)]
    @test popcoefnames(linked, :Vc) == [:Intercept, :x]
    # The linked spelling is NOT a public address — it names no linear predictor.
    @test popcoefnames(linked, :log_Vc) === nothing
    @test [(p.predictor, p.coefficient) for p in effect_priors(linked)] ==
          [(:Vc, :Intercept), (:Vc, :x)]

    linked_sb = @test_nowarn SBBRMI(linked; mod=@__MODULE__, total_groups=())
    linked_code = BayesianRegressionModels.stan_code(linked_sb)
    @test StanBlocks.stan.transpiles(linked_sb.model)
    @test StanBlocks.stanc_check(linked_code; warn_pedantic=false).ok
    # Emitted under the LINKED spelling, and the inverse link is bound for the
    # declared quantity — so `Vc` is usable downstream on its natural scale.
    @test occursin("pop_log_Vc_beta_pop ~ normal([$(log(10)), 0]', [0.8, 0.1]');",
                   linked_code)
    @test occursin("Vc = exp(log_Vc);", linked_code)

    # The descriptor's population labels survive the link. This is the whole
    # point of mounting a descriptor, and it silently regressed to `nothing`
    # because `pop_lp` derived the block name from the PUBLIC name.
    linked_byname = Dict(o.name => o for o in brm_descriptor(linked_sb).outputs)
    @test linked_byname[:pop_log_Vc_beta_pop].role === :population_effect
    @test linked_byname[:pop_log_Vc_beta_pop].labels == [:Intercept, :x]

    # Ranef addresses are keyed by the `|ID|` bucket, never by the LP name, so
    # `sd(:, p)` / `cor(:, p)` are untouched by the switch — but `ranefcoefnames`
    # margins are labelled with the PUBLIC predictor name.
    @test ranefcoefnames(linked, :p) ==
          [(predictor=:Vc, coefficient=:Intercept), (predictor=:Vc, coefficient=:x)]
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(2.0);", linked_code)
    @test occursin(r"b_p_subject_tau ~ brm_vector_prior_[0-9a-f]+\(", linked_code)

    # Same model with the quantity declared as an INERT name and the link undone
    # by hand: identical parameter block, so the posterior is the same fit.
    inert = @brm link_df begin
        log_Vc ~ 1 + x + (1 + x | p | subject)
        effect(log_Vc, Intercept) ~ Normal(log(10), 0.8)
        effect(log_Vc, x) ~ Normal(0, 0.1)
        sd(:, p) ~ Exponential(0.3)
        cor(:, p) ~ LKJCholesky(2, 2.0)
        y ~ Normal(log_Vc, 0.2)
    end
    inert_code = BayesianRegressionModels.stan_code(SBBRMI(inert; mod=@__MODULE__, total_groups=()))
    params(code) = begin
        i = findfirst("parameters {", code)
        code[first(i):findnext("}", code, last(i))[end]]
    end
    @test params(linked_code) == params(inert_code)
    @test popcoefnames(inert, :log_Vc) == popcoefnames(linked, :Vc)

    # The two wrong guesses fail LOUDLY, and the population one names the right
    # address in its message — neither can silently attach elsewhere.
    wrong_underscore = @brm link_df begin
        log(Vc) ~ 1 + x
        effect(log_Vc, Intercept) ~ Normal(log(10), 0.8)
        y ~ Normal(log(Vc), 0.2)
    end
    @test_throws "Available predictors: Vc" SBBRMI(wrong_underscore; mod=@__MODULE__, total_groups=())
    @test_throws "must be bare symbols" @eval @brm $link_df begin
        log(Vc) ~ 1 + x
        effect(log(Vc), Intercept) ~ Normal(log(10), 0.8)
        y ~ Normal(log(Vc), 0.2)
    end
end

# The consumer shape the snag was reported for (a downstream PKPD web app,
# `linear_pk_brm5`): the PK model declares its per-subject quantities with LHS
# links so the cell no longer spells `exp(...)` itself, and `kernel(...)` is fed
# the NATURAL-scale bindings. The population priors stay on the log scale and
# keep their bare-name addresses.
@testset "LHS link transformation feeds `kernel(...)` on its natural scale" begin
    n = 6
    pk_df = (; t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:n],
               dose    = fill(100.0, n),
               dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:n],
               weight  = collect(range(-1.0, 1.0; length=n)),
               subject = collect(1:n))

    pk = @brm pk_df begin
        sigma ~ Exponential(1)
        log(CL) ~ 1 + weight + (1 | p | subject)
        log(Vc) ~ 1 + (1 | p | subject)
        log(Ka) ~ 1 + (1 | p | subject)
        effect(CL, Intercept) ~ Normal(log(1 / 8), 0.8)
        effect(CL, weight) ~ Normal(0, 0.1)
        pred ~ kernel(t, dose, dv, CL, Vc, Ka) do ts, d, yy, CLi, Vci, Kai
            ke = CLi / Vci
            mu = d * Kai / (Vci * (Kai - ke)) * (exp(-ke * ts) - exp(-Kai * ts))
            yy ~ normal(mu, sigma)
            mu
        end
    end

    @test popcoefnames(pk, :CL) == [:Intercept, :weight]
    @test popcoefnames(pk, :log_CL) === nothing
    # One shared `|p|` block over three link-declared predictors: the margins are
    # labelled with the bare names, in formula order.
    @test ranefcoefnames(pk, :p) == [(predictor=:CL, coefficient=:Intercept),
                                     (predictor=:Vc, coefficient=:Intercept),
                                     (predictor=:Ka, coefficient=:Intercept)]

    sb = SBBRMI(pk; mod=@__MODULE__, total_groups=())
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("pop_log_CL_beta_pop ~ normal([$(log(1 / 8)), 0]', [0.8, 0.1]');",
                   code)
    @test occursin("pop_log_Vc_beta_pop ~ std_normal();", code)
    # Every declared quantity reaches the cell on its natural scale.
    for q in (:CL, :Vc, :Ka)
        @test occursin("$q = exp(log_$q);", code)
    end

    byname = Dict(o.name => o for o in brm_descriptor(sb).outputs)
    @test byname[:pop_log_CL_beta_pop].labels == [:Intercept, :weight]
    @test byname[:pop_log_Vc_beta_pop].labels == [:Intercept]
    @test byname[:pop_log_Ka_beta_pop].labels == [:Intercept]
end

# A categorical / `factor(...)` predictor does NOT live in `beta_pop`: it owns a
# `cat_<lp>_<c>_beta` contrast vector emitted by its own submodel, which is exactly
# why `popcoefnames` excludes it. Before this, that vector's `std_normal()` was
# hardcoded with no seam at all — the only way to change it was to hand-build
# K-1 dummy columns, which loses `factor()`'s frozen training levels (and so its
# unseen-level guard in `reprocess`), `ref=`, and the `:factor` preprocessing
# record. The address is the COLUMN name, never the emitted `cat_` name.
@testset "categorical contrast priors via `effect(lp, column)`" begin
    cat_df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                g=[1, 2, 3, 1, 2, 3],
                y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

    code_of(m) = BayesianRegressionModels.stan_code(SBBRMI(m; mod=@__MODULE__, total_groups=()))

    # Unconfigured: the hardcoded default, byte for byte as before.
    plain = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        y ~ Normal(mu, 1)
    end
    @test occursin("cat_mu_g_beta ~ std_normal();", code_of(plain))
    # The categorical is still deliberately absent from `beta_pop`'s labels.
    @test popcoefnames(plain, :mu) == [:Intercept, :x]

    configured = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        effect(mu, g) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1)
    end
    sb = SBBRMI(configured; mod=@__MODULE__, total_groups=())
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("cat_mu_g_beta ~ normal(0.0, 0.5);", code)
    # One shared `(location, scale)` over the K-1 contrasts; `beta_pop` untouched.
    @test occursin("pop_mu_beta_pop ~ std_normal();", code)
    @test popcoefnames(configured, :mu) == [:Intercept, :x]

    # A `:` predictor slot reaches the same block, and a bare integer column
    # (no `factor(...)` wrapper) is the same term downstream.
    concise = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        effect(:, g) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1)
    end
    bare = @brm cat_df begin
        mu ~ 1 + g + x
        effect(mu, g) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1)
    end
    @test code_of(concise) == code
    @test code_of(bare) == code

    # A non-default reference level renames the emitted block; the COLUMN name
    # still addresses it, and the renamed block is addressable too.
    reffed = @brm cat_df begin
        mu ~ 1 + factor(g; ref=3) + x
        effect(mu, g) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1)
    end
    @test occursin("cat_mu_g__ref_3_beta ~ normal(0.0, 0.5);", code_of(reffed))

    # An intercept-less formula has no `beta_pop` labels at all; the contrast
    # block must stay reachable anyway.
    no_intercept = @brm cat_df begin
        mu ~ 0 + factor(g)
        effect(mu, g) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1)
    end
    @test popcoefnames(no_intercept, :mu) == Symbol[]
    @test occursin("cat_mu_g_beta ~ normal(0.0, 0.5);", code_of(no_intercept))

    # Same column at two reference levels: each EXACT emitted name binds to its
    # own block, so the plain block never loses its name to the other's alias.
    both = @brm cat_df begin
        mu ~ 1 + factor(g) + factor(g; ref=3)
        effect(mu, g) ~ Normal(0.0, 0.5)
        effect(mu, g__ref_3) ~ Normal(0.0, 0.25)
        y ~ Normal(mu, 1)
    end
    both_code = code_of(both)
    @test occursin("cat_mu_g_beta ~ normal(0.0, 0.5);", both_code)
    @test occursin("cat_mu_g__ref_3_beta ~ normal(0.0, 0.25);", both_code)

    # But an alias wanted by two blocks binds to neither — refuse, never guess.
    ambiguous_ref = @brm cat_df begin
        mu ~ 1 + factor(g; ref=2) + factor(g; ref=3)
        effect(mu, g) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1)
    end
    @test_throws "g__ref_2" SBBRMI(ambiguous_ref; mod=@__MODULE__, total_groups=())

    # The EMITTED name is not the address — and the error says which name is.
    emitted_guess = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        effect(mu, cat_g) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1)
    end
    @test_throws "address one by that name" SBBRMI(emitted_guess; mod=@__MODULE__, total_groups=())

    # `:` and an explicit predictor reach the same block, but they are NOT a
    # duplicate: `:` is the default layer and the more specific address wins.
    layered = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        effect(:, g) ~ Normal(0.0, 0.5)
        effect(mu, g) ~ Normal(0.0, 0.25)
        y ~ Normal(mu, 1)
    end
    layered_code = BayesianRegressionModels.stan_code(SBBRMI(layered; mod=@__MODULE__, total_groups=()))
    @test occursin("cat_mu_g_beta ~ normal(0.0, 0.25);", layered_code)
    @test !occursin("cat_mu_g_beta ~ normal(0.0, 0.5);", layered_code)

    # Two addresses of EQUAL specificity reaching one block have no winner, and
    # that is an error rather than a silent last-writer-wins.
    tied = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        effect(:, g) ~ Normal(0.0, 0.5)
        effect(mu, :) ~ Normal(0.0, 0.25)
        y ~ Normal(mu, 1)
    end
    @test_throws "equally specific" SBBRMI(tied; mod=@__MODULE__, total_groups=())

    generic_family = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        effect(mu, g) ~ Cauchy(0, 1)
        y ~ Normal(mu, 1)
    end
    generic_family_sb = SBBRMI(generic_family; mod=@__MODULE__, total_groups=())
    generic_family_code = BayesianRegressionModels.stan_code(generic_family_sb)
    @test count("cauchy_lpdf(x[", generic_family_code) == 2
    @test occursin(r"cat_mu_g_beta ~ brm_vector_prior_[0-9a-f]+", generic_family_code)
    @test StanBlocks.stanc_check(generic_family_code; warn_pedantic=false).ok

    # A configured contrast block and a configured `beta_pop` coexist.
    mixed = @brm cat_df begin
        mu ~ 1 + factor(g) + x
        effect(mu, g) ~ Normal(0.0, 0.5)
        effect(mu, x) ~ Normal(0.0, 0.25)
        y ~ Normal(mu, 1)
    end
    mixed_code = code_of(mixed)
    @test occursin("cat_mu_g_beta ~ normal(0.0, 0.5);", mixed_code)
    @test occursin("[1.0, 0.25]'", mixed_code)

    # One categorical source may be reused across any number of response
    # formulas. Every formula owns a separate contrast parameter and its own
    # `effect(response, source)` prior, including linked LHS predictors whose
    # emitted namespace carries the link spelling.
    repeated_df = (;
        indication=[1, 2, 1, 2],
        y=zeros(4),
    )
    repeated = @brm repeated_df begin
        log(Vc) ~ 1 + indication
        log(k10) ~ 1 + indication
        log(k12) ~ 1 + indication
        log(k21) ~ 1 + indication
        log(ka) ~ 1 + indication
        qt_base ~ 1 + indication
        qt_slope ~ 1 + indication

        effect(Vc, indication) ~ Normal(0.0, 0.11)
        effect(k10, indication) ~ Normal(0.0, 0.12)
        effect(k12, indication) ~ Normal(0.0, 0.13)
        effect(k21, indication) ~ Normal(0.0, 0.14)
        effect(ka, indication) ~ Normal(0.0, 0.15)
        effect(qt_base, indication) ~ Normal(0.0, 0.16)
        effect(qt_slope, indication) ~ Normal(0.0, 0.17)

        y ~ Normal(Vc + k10 + k12 + k21 + ka + qt_base + qt_slope, 1.0)
    end
    repeated_sb = SBBRMI(repeated; mod=@__MODULE__, total_groups=())
    repeated_code = BayesianRegressionModels.stan_code(repeated_sb)
    @test StanBlocks.stan.transpiles(repeated_sb.model)
    @test StanBlocks.stanc_check(repeated_code; warn_pedantic=false).ok
    EFFECT_PRIOR_RUNTIME && @test StanBlocks.stan.compiles(repeated_sb.model)
    @test !isempty(brm_descriptor(repeated_sb).outputs)
    for (block, scale) in (
        (:cat_log_Vc_indication_beta, 0.11),
        (:cat_log_k10_indication_beta, 0.12),
        (:cat_log_k12_indication_beta, 0.13),
        (:cat_log_k21_indication_beta, 0.14),
        (:cat_log_ka_indication_beta, 0.15),
        (:cat_qt_base_indication_beta, 0.16),
        (:cat_qt_slope_indication_beta, 0.17),
    )
        @test occursin("$block ~ normal(0.0, $scale);", repeated_code)
    end

    # `factor()`'s frozen training levels — the guard the hand-built-dummy
    # workaround silently gives up — still fire on the configured model.
    @test_throws "not a training level" BayesianRegressionModels.reprocess(
        sb, (; x=[0.0, 1.0], g=[1, 9], y=[0.0, 0.0]))
end
