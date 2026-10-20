# test/output_predictors.jl — `BRMOutput.predictors` / `term`, and `component`
# for the parts of population, categorical and structured-term blocks: which
# linear predictor, which formula term and which part of it every output is
# (snag `descriptor-has-n-9c0e42d5`, design decision `0yh0jw3`).
#
# The declaration half builds every population-level term family and checks
# each output's address, found through `brm_outputs(d; predictor=, term=,
# component=)` and never through an emitted name. The runtime half compiles
# two models and checks that every address resolves back to its own output:
# a categorical block through `brm_population_effect_coordinates(...;
# coefficient=term)`, a term part through `brm_term_coordinates(...; term,
# parameter=component)`.
#
# Run: julia --project=test test/output_predictors.jl
# (set BRM_OUTPUT_PREDICTORS_RUNTIME=0 to run only the declaration half.)

using Test
using BayesianRegressionModels
using StanBlocks
using CategoricalArrays
using Random
using Distributions: Exponential, Beta

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const RUNTIME = get(ENV, "BRM_OUTPUT_PREDICTORS_RUNTIME", "1") != "0"

rng = MersenneTwister(5)
const N = 60
const DF = (; y = randn(rng, N), x = randn(rng, N), z = randn(rng, N),
    t = collect(range(0, 1; length=N)),
    treatment = categorical(rand(rng, ["a", "b", "c"], N)),
    site = categorical(rand(rng, ["s1", "s2"], N)),
    g = rand(rng, 1:3, N), dose = rand(rng, 1:4, N),
    subject = repeat(1:12, inner=5),
    arm = ifelse.(repeat(1:12, inner=5) .<= 6, "a", "b"),
    grp = categorical(repeat(["u", "v", "w"], N ÷ 3)),
    week = repeat(1:(N ÷ 2), inner=2), patch = repeat(["a", "b"], N ÷ 2),
    C = [1.0 0.5; 0.5 1.0])

one_output(d; kw...) = only(brm_outputs(d; kw...))
sampled(d) = [o for o in d.outputs if o.kind === :parameter]

const MIXED = @brm begin
    mu ~ 1 + x + treatment + mo(dose) + hsgp(t) + (1 | subject)
    log(scale) ~ 1 + treatment + site
    y ~ Normal(mu, scale)
end

@testset "categorical, mo, hsgp and population blocks name their predictor" begin
    d = brm_descriptor(MIXED, DF; mod=@__MODULE__)
    # The same column in two predictors: one block each, told apart by predictor.
    mu_cat = one_output(d; predictor=:mu, term=:treatment, component=:contrasts)
    scale_cat = one_output(d; predictor=:scale, term=:treatment, component=:contrasts)
    @test mu_cat.predictors == (:mu,) && scale_cat.predictors == (:scale,)
    @test mu_cat.name !== scale_cat.name
    @test one_output(d; predictor=:scale, term=:site, component=:contrasts).kind === :parameter
    @test one_output(d; predictor=:mu, term=:treatment, component=:contribution).kind ===
          :transformed_parameter
    # A linked predictor is addressed by its public name, never `log_scale`.
    @test isempty(brm_outputs(d; predictor=:log_scale))
    pop = one_output(d; predictor=:scale, role=:population_effect, component=:coefficients)
    @test pop.labels == [:Intercept] && isnothing(pop.term)
    @test one_output(d; predictor=:mu, role=:population_effect,
                     component=:coefficients).labels == [:Intercept, :x, :mo_dose]
    @test one_output(d; predictor=:mu, role=:population_effect,
                     component=:contribution).kind === :transformed_parameter

    # Term parts use the parameter names `brm_term_coordinates` takes.
    @test one_output(d; term=:mo_dose, component=:simplex).predictors == (:mu,)
    hsgp = brm_outputs(d; predictor=:mu, term=:hsgp_t)
    @test Set(o.component for o in hsgp if o.kind === :parameter) ==
          Set([:length_scale, :sd, :basis_weights])
    @test one_output(d; term=:hsgp_t, component=:value).kind === :transformed_parameter
    @test all(o -> o.component === :auxiliary,
              [o for o in hsgp if o.kind === :transformed_parameter && o.component !== :value])

    # Random-effect blocks keep their components and gain the predictor.
    @test Set(o.component for o in brm_outputs(d; role=:random_effect, predictor=:mu)) ==
          Set([:log_scale, :standardized, :contribution])
    # The predictor's own value; outputs no formula term owns carry ().
    @test one_output(d; role=:linear_predictor, predictor=:scale).logical === :scale
    @test all(o -> isempty(o.predictors) && isnothing(o.term),
              brm_outputs(d; role=(:posterior_predictive, :pointwise_loglik)))
    # Every sampled coordinate is grouped by a predictor.
    @test all(o -> !isempty(o.predictors), sampled(d))
    # Filters take collections like every other `brm_outputs` filter.
    @test length(brm_outputs(d; predictor=(:mu, :scale))) ==
          length(brm_outputs(d; predictor=:mu)) + length(brm_outputs(d; predictor=:scale))
    @test Set(o.term for o in brm_outputs(d; term=(:treatment, :site))) ==
          Set([:treatment, :site])
end

@testset "a shared |ID| block lists every predictor it feeds" begin
    d = brm_descriptor(@brm(begin
        mu ~ 1 + x + rw(t) + (1 | p | subject)
        log(scale) ~ 1 + (1 | p | subject)
        y ~ Normal(mu, scale)
    end), DF; mod=@__MODULE__)
    shared = brm_outputs(d; role=:random_effect)
    @test !isempty(shared) && all(o -> o.predictors == (:mu, :scale), shared)
    @test one_output(d; role=:random_effect, component=:scale) in brm_outputs(d; predictor=:scale)
    @test one_output(d; role=:random_effect, component=:scale) in brm_outputs(d; predictor=:mu)
    @test Set(o.component for o in brm_outputs(d; predictor=:mu, term=:rw_mu_t)) ==
          Set([:sd, :innovations, :value])
end

@testset "every structured term family" begin
    d = brm_descriptor(@brm(begin
        mu ~ 1 + factor(g; ref=3) + mo1(dose) + dar(t) + s(x)
        nu ~ 0 + treatment
        y ~ Normal(mu + nu, 1.0)
    end), DF; mod=@__MODULE__)
    # `factor(g; ref=3)` is addressed by its column, as `coefficient=:g` is.
    @test one_output(d; term=:g, component=:contrasts).predictors == (:mu,)
    # An intercept-free predictor's first factor is cell-mean coded.
    @test one_output(d; predictor=:nu, term=:treatment, component=:cell_means).kind === :parameter
    @test one_output(d; term=:mo1_dose, component=:simplex).predictors == (:mu,)
    @test one_output(d; term=:mo1_dose, component=:value).kind === :transformed_parameter
    @test Set(o.component for o in brm_outputs(d; term=:dar_mu_t, kind=:parameter)) ==
          Set([:ar, :sd, :innovations])
    # `s` has no public parameter roles: grouped and labelled, no parts.
    spline = brm_outputs(d; term=Symbol("s(x)"))
    @test !isempty(spline) && all(o -> o.predictors == (:mu,) && isnothing(o.component), spline)
    @test all(o -> !isempty(o.predictors), sampled(d))

    d = brm_descriptor(@brm(begin
        mu ~ 1 + gp(x) + t2(x, z) + ar(t) + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, 1.0)
    end), DF; mod=@__MODULE__)
    for term in (Symbol("gp(x)"), Symbol("t2(x,z)"), Symbol("ar(t)"))
        @test any(o -> o.kind === :parameter, brm_outputs(d; predictor=:mu, term))
    end
    @test Set(o.component for o in brm_outputs(d; term=:cdar_mu_week, kind=:parameter)) ==
          Set([:ar, :sd, :innovations])
    @test all(o -> o.predictors == (:mu,), sampled(d))

    # A grouped hsgp's basis weights live in a per-group block beside the term;
    # a term repeated in two predictors stays each predictor's own.
    d = brm_descriptor(@brm(begin
        mu ~ 1 + me(x, 0.5) + hsgp(t; by=grp) + s(z) + mo(dose)
        log(scale) ~ 1 + s(z) + mo(dose)
        y ~ Normal(mu, scale)
    end), DF; mod=@__MODULE__)
    @test one_output(d; term=:hsgp_t_by_grp, component=:basis_weights, kind=:parameter).predictors ==
          (:mu,)
    @test Set(o.component for o in brm_outputs(d; term=:hsgp_t_by_grp, kind=:parameter)) ==
          Set([:length_scale, :sd, :basis_weights])
    @test one_output(d; predictor=:mu, term=:mo_dose, component=:simplex).name !==
          one_output(d; predictor=:scale, term=:mo_dose, component=:simplex).name
    @test length(brm_outputs(d; term=Symbol("s(z)"), kind=:parameter, predictor=:scale)) ==
          length(brm_outputs(d; term=Symbol("s(z)"), kind=:parameter, predictor=:mu))
    @test any(o -> o.kind === :parameter, brm_outputs(d; predictor=:mu, term=Symbol("me(x)")))
    @test all(o -> !isempty(o.predictors), sampled(d))
end

@testset "prior structure, exact totals, S2Z and stratified blocks" begin
    flat = brm_descriptor(SBBRMI(@brm(DF, begin
        mu ~ 1 + x + z + (1 | g)
        effect(mu, :) ~ r2d2(R2=Beta(2, 3), tau_bsv=0.5)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__, total_groups=()))
    @test all(o -> o.predictors == (:mu,), sampled(flat))

    joint = brm_descriptor(SBBRMI(@brm(DF, begin
        sigma_y ~ Exponential(1)
        a ~ 1 + (1 | p | subject)
        b ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=4, concentration=0.7, reference_scale=sigma_y)
        y ~ Normal(a + b, sigma_y)
    end); mod=@__MODULE__, total_groups=()))
    # The budget spans both predictors; the author's reference scale is the author's.
    @test one_output(joint; logical=:sigma_y, kind=:parameter).predictors == ()
    @test all(o -> o.predictors == (:a, :b),
              [o for o in sampled(joint) if o.logical !== :sigma_y && o.role !== :population_effect])

    hs = brm_descriptor(SBBRMI(@brm(DF, begin
        mu ~ 1 + x
        effect(mu, x) ~ Horseshoe(local_scale=0.5, global_scale=0.2)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__, total_groups=()))
    @test all(o -> o.predictors == (:mu,), sampled(hs))
    # The horseshoe's local and global scales are internals of the population block.
    @test one_output(hs; role=:population_effect, component=:coefficients).predictors == (:mu,)
    @test any(o -> o.component === :auxiliary, sampled(hs))

    totals = brm_descriptor(@brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | p | g)
        y ~ Normal(mu, sigma)
    end), DF; mod=@__MODULE__)
    @test !isempty(total_effect_blocks(totals.plan))
    @test all(o -> o.predictors == (:mu,), [o for o in sampled(totals) if o.logical !== :sigma])
    @test one_output(totals; role=:random_effect, component=:effects).predictors == (:mu,)

    s2z = brm_descriptor(SBBRMI(@brm(DF, begin
        mu ~ 1 + (1 | g)
        effect(mu, Intercept) ~ Normal(0, 5)
        y ~ Normal(mu, 1)
    end); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.5, total_groups=()))
    @test all(o -> o.predictors == (:mu,), sampled(s2z))

    strat = brm_descriptor(SBBRMI(@brm(DF, begin
        mu ~ 1 + x + (1 + x | gr(subject, by=arm))
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__, total_groups=()))
    @test all(o -> o.predictors == (:mu,), brm_outputs(strat; role=:random_effect))
end

@testset "hand-built outputs keep compiling" begin
    o = BRM.BRMOutput(:x, :parameter, :real, (), (;), :posterior, nothing,
                      :parameter, nothing, nothing, nothing, nothing, nothing)
    @test o.predictors == () && isnothing(o.term)
end

if RUNTIME
    # Every address resolves back to its own output through the public queries.
    function round_trip(d, names)
        checked = 0
        for o in d.outputs
            isnothing(o.term) && continue
            lp = only(o.predictors)
            if o.component in (:contrasts, :cell_means)
                q = brm_population_effect_coordinates(d, lp, names; coefficient=o.term)
                @test q.output.name === o.name
                @test q.coordinates == brm_output_coordinates(o, names)
                checked += 1
            elseif !isnothing(o.component) && o.component !== :auxiliary &&
                   o.component !== :contribution
                q = brm_term_coordinates(d, lp, names; term=o.term, parameter=o.component)
                @test q.output.name === o.name
                checked += 1
            end
        end
        checked
    end
    for (label, builder) in (
            "mixed" => MIXED,
            "families" => @brm(begin
                mu ~ 1 + factor(g; ref=3) + mo1(dose) + dar(t) + treatment
                nu ~ 0 + site + rw(t)
                y ~ Normal(mu + nu, 1.0)
            end))
        @testset "round trip: $label" begin
            sb = SBBRMI(builder(DF); mod=@__MODULE__)
            d = brm_descriptor(sb)
            sm = BRM.stan_instantiate(sb).model
            names = BS.param_names(sm; include_tp=true, include_gq=true)
            @test round_trip(d, names) >= 5
            # Grouping by predictor reaches every sampled coordinate.
            grouped = Set(names[i] for o in sampled(d) if !isempty(o.predictors)
                          for i in brm_output_coordinates(o, names))
            @test Set(BS.param_names(sm)) ⊆ grouped
        end
    end
end
