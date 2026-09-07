# test/descriptor.jl — BRMDescriptor acceptance.
#
# Covers the five things the descriptor must be able to promise a consumer:
#   1. reflection            — inputs/outputs/operations derived from ONE declaration
#   2. stable identity       — same declaration, same id; different model, different id
#   3. schema propagation    — Stan inputs carry their dataframe column + transform
#   4. generative semantics  — BRM roles on the outputs, coefficient labels
#   5. execution             — BRM -> StanBlocks -> BridgeStan actually runs
# plus the fail-closed cases and the documented extension points.
#
# Run: julia --project=. test/descriptor.jl
# (BRM has no runnable test harness — see the primer; these are standalone.)

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, LogNormal, Normal

df = (; x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        g=[1, 1, 2, 2, 3, 3],
        y=[1.0, 1.5, 2.0, 2.4, 3.1, 3.4],
        z=[2.0, 2.5, 3.0, 3.4, 4.1, 4.4])

hier_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | g)
    y ~ Normal(mu, sigma)
    z ~ Normal(mu, 2 * sigma)
end

module DescriptorFragments
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

StanBlocks.@deffun descriptor_fragment_inner(x::vector[n])::vector[n] = x .* x
StanBlocks.@deffun descriptor_fragment_outer(x::vector[n])::vector[n] =
    descriptor_fragment_inner(x) .+ 1.0

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(descriptor_fragment_outer(mu), sigma)
end

df = (; x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0])
end

@testset "reflection — one declaration, derived surface" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)

    @test d isa BRMDescriptor
    @test d.name === :hier
    @test !isempty(d.id)

    # Operations are DERIVED, not listed. This model has parameters, a
    # likelihood, predictive draws and pointwise log-liks, and was built from a
    # reusable builder -> replay plus frozen same-group reprocess.
    ops = Symbol[op.name for op in d.operations]
    @test :transpile in ops
    @test :instantiate in ops
    @test :fit in ops
    @test :predict in ops
    @test :pointwise_loglik in ops
    @test :replay in ops
    @test :reprocess in ops
    @test isempty(d.unpredictable)  # both observations get a real *_gen

    # An operation's origin says which namespace its `inputs` live in.
    @test brm_operation(d, :fit).origin === :stan
    @test brm_operation(d, :replay).origin === :brm
    @test brm_operation(d, :replay).inputs == d.columns

    # :predict reports only draws the program actually emits.
    @test Set(brm_operation(d, :predict).outputs) == Set([:y_gen, :z_gen])

    # required inputs exclude the sizes StanBlocks derives from other inputs.
    req = required_brm_inputs(d)
    @test :y in req && :x in req && :g_idx in req
    @test :y_n ∉ req && :x_n ∉ req
end

@testset "stable identity" begin
    a = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)
    b = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:something_else)
    # Identity is content, not label.
    @test a.id == b.id
    @test a.name !== b.name
    # ...and it is the key the compiled artifact caches under.
    @test a.id == StanBlocks.stan_descriptor(a.plan.model).id

    flat_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test brm_descriptor(flat_builder, df; mod=@__MODULE__).id != a.id
end

@testset "configured hsgp term priors keep both descriptor entry points" begin
    hsgp_df = (;
        y=sin.(1:12),
        op_log_dose=collect(range(-1.0, 1.0; length=12)),
    )
    hsgp_builder = @brm begin
        y ~ Normal(mu, 1.0)
        mu ~ 1 + hsgp(op_log_dose; k=5)
        length_scale(:, hsgp(op_log_dose)) ~ LogNormal(0, 1)
        sd(:, hsgp(op_log_dose)) ~ LogNormal(0, 1)
    end
    sb = SBBRMI(hsgp_builder(hsgp_df); mod=@__MODULE__)

    from_sb = brm_descriptor(sb; name=:hsgp_term_priors, highlights=())
    from_builder = brm_descriptor(
        hsgp_builder, hsgp_df;
        mod=@__MODULE__, name=:hsgp_term_priors, highlights=())

    code = BayesianRegressionModels.stan_code(sb)
    @test BayesianRegressionModels.stan_code(from_sb.plan) == code
    @test BayesianRegressionModels.stan_code(from_builder.plan) == code
    @test from_sb.id == from_builder.id
    @test from_sb.plan.model.model !== sb.model.model
    @test Set(from_sb.columns) == Set((:y, :op_log_dose))
    @test brm_output(from_sb, :y; role=:posterior_predictive).logical === :y
    @test :fit in (op.name for op in from_builder.operations)
end

@testset "named Stan definition highlights" begin
    d = brm_descriptor(
        DescriptorFragments.builder,
        DescriptorFragments.df;
        mod=DescriptorFragments,
        highlights=(
            :descriptor_fragment_outer => "Displayed model calculation",
            "descriptor_fragment_inner",
        ),
    )

    @test [h.name for h in d.highlights] ==
          [:descriptor_fragment_outer, :descriptor_fragment_inner]
    @test [h.caption for h in d.highlights] ==
          ["Displayed model calculation", nothing]
    inventory = Dict(def.name => def for def in d.stan.definitions)
    @test all(h -> haskey(inventory, h.name), d.highlights)
    @test all(h -> h.definition === inventory[h.name], d.highlights)
    @test all(h -> StanBlocks.stan_definition(d.stan, h.name).name === h.name,
              d.highlights)
    @test all(h -> [def.signature for def in h.closure] ==
                         [def.signature for def in
                          StanBlocks.stan_definition_closure(d.stan, h.definition)],
              d.highlights)
    @test occursin("descriptor_fragment_inner",
                   inventory[:descriptor_fragment_outer].source)
    @test :descriptor_fragment_inner in
          inventory[:descriptor_fragment_outer].dependencies
    @test :descriptor_fragment_inner in
          (def.name for def in first(d.highlights).closure)

    # Presentation metadata does not alter the executable model identity.
    plain = brm_descriptor(DescriptorFragments.builder, DescriptorFragments.df;
                           mod=DescriptorFragments)
    @test d.id == plain.id

    # Replay/reprocessing resolves the same names against the NEW descriptor,
    # preserving order and captions rather than carrying stale definition
    # objects from the old traced model.
    d2 = brm_execute(d, :reprocess,
                     (; x=[3.0, 4.0], y=[4.0, 5.0]))
    @test [(h.name, h.caption) for h in d2.highlights] ==
          [(h.name, h.caption) for h in d.highlights]
    inventory2 = Dict(def.name => def for def in d2.stan.definitions)
    @test all(h -> h.definition === inventory2[h.name], d2.highlights)
    @test all(h -> [def.signature for def in h.closure] ==
                         [def.signature for def in
                          StanBlocks.stan_definition_closure(d2.stan, h.definition)],
              d2.highlights)

    @test_throws ErrorException brm_descriptor(
        DescriptorFragments.builder, DescriptorFragments.df;
        mod=DescriptorFragments, highlights=(:not_in_this_model,))
    @test_throws ErrorException brm_descriptor(
        DescriptorFragments.builder, DescriptorFragments.df;
        mod=DescriptorFragments,
        highlights=(:descriptor_fragment_outer, :descriptor_fragment_outer))
end

@testset "schema propagation — Stan inputs carry their dataframe column" begin
    zs_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + zscale(x)
        y ~ Normal(mu, sigma)
    end
    d = brm_descriptor(zs_builder, df; mod=@__MODULE__)

    @test :x in d.columns          # a predictor, reached through zscale(...)
    @test :y in d.columns          # the response — a `~` op, not a DataColumn op
    @test brm_columns(d) === d.columns

    # The formula is DERIVED from the parsed declaration, so it cannot drift
    # from the model that runs — unlike a hand-kept `formula_src` string.
    @test occursin("zscale", d.formula)
    @test d.formula == sprint(show, d.plan.parent)

    byname = Dict(i.name => i for i in d.inputs)
    @test byname[:y].column === :y
    @test byname[:y].transform === nothing
    @test byname[:y].observed

    # The z-scaled predictor lands under a transformed key that still knows the
    # raw column it came from and how it was transformed.
    zs = [i for i in d.inputs if i.transform === :zscale]
    @test length(zs) == 1
    @test only(zs).column === :x

    # Sizes are `derived` -> a form must not ask for them.
    @test byname[:y_n].derived
    @test byname[:y_n].column === nothing
end

@testset "generative semantics — BRM roles and labels" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__)
    byname = Dict(o.name => o for o in d.outputs)

    @test byname[:sigma].role === :parameter
    @test byname[:sigma].declaration.family === :exponential
    @test byname[:sigma].logical === :sigma

    # The population block and its internals both resolve to the popefs
    # declaration, but only the returned block output carries its logical
    # target. The linear predictor comes from the formula, not a `~`.
    @test byname[:pop_mu].role === :population_effect
    @test byname[:pop_mu].logical === :pop_mu
    @test byname[:pop_mu_beta_pop].role === :population_effect
    @test byname[:pop_mu_beta_pop].declaration.target === :pop_mu
    @test byname[:pop_mu_beta_pop].logical === nothing
    @test byname[:mu].role === :linear_predictor
    @test byname[:mu].logical === :mu
    @test byname[:mu].declaration === nothing
    @test brm_output(d, :mu; role=:linear_predictor).name === :mu
    @test brm_output_coordinates(
        d, :mu, ["mu.1", "mu.2", "y_gen.1"];
        role=:linear_predictor) == [1, 2]

    @test byname[:r_mu_g].role === :random_effect
    @test byname[:r_mu_g_xi].role === :random_effect

    # Predictive draws and pointwise log-liks are tagged from StanBlocks'
    # `source` link, never by parsing the `_gen` suffix.
    @test byname[:y_gen].role === :posterior_predictive
    @test byname[:y_gen].source === :y
    @test byname[:y_likelihood].role === :pointwise_loglik

    # Coefficient labels replace `pop_mu_beta_pop.N` guessing.
    @test byname[:pop_mu_beta_pop].labels == [:Intercept, :x]
    @test byname[:mu].labels === nothing
end

@testset "population coordinates — public logical address and LHS link" begin
    linked_df = (; weight=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                   y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])
    linked_builder = @brm begin
        log(Vc) ~ 1 + weight
        y ~ Normal(log(Vc), 0.2)
    end
    d = brm_descriptor(linked_builder, linked_df; mod=@__MODULE__, name=:linked)
    names = ["y_gen.1", "pop_log_Vc_beta_pop.1",
             "pop_log_Vc_beta_pop.2", "y_likelihood.1"]

    intercept = brm_population_effect_coordinates(
        d, :Vc, names; coefficient=:Intercept)
    slope = brm_population_effect_coordinates(
        d, :Vc, names; coefficient=:weight)

    @test intercept.logical === :Vc
    @test intercept.coefficient === :Intercept
    @test intercept.output.role === :population_effect
    @test intercept.output.name === :pop_log_Vc_beta_pop
    @test intercept.coordinates == [2]
    @test slope.coordinates == [3]
    @test intercept.link === log
    @test intercept.inverse_link === exp

    # Same answer as the former consumer workaround, but without constructing
    # the compiler-owned carrier/coordinate spellings or guessing output order.
    workaround = findall(==("pop_log_Vc_beta_pop.1"), names)
    @test intercept.coordinates == workaround

    @test_throws "available labels are (:Intercept, :weight)" begin
        brm_population_effect_coordinates(d, :Vc, names; coefficient=:missing)
    end
    @test_throws "0 formula declarations" begin
        brm_population_effect_coordinates(d, :log_Vc, names)
    end
    @test_throws "2 coefficient labels but resolves to 1 constrained coordinates" begin
        brm_population_effect_coordinates(
            d, :Vc, ["pop_log_Vc_beta_pop.1"])
    end

    identity_link = brm_population_effect_coordinates(
        brm_descriptor(hier_builder, df; mod=@__MODULE__), :mu,
        ["pop_mu_beta_pop.1", "pop_mu_beta_pop.2"])
    @test identity_link.link === identity
    @test identity_link.inverse_link === identity
end

@testset "categorical population coordinates — response-qualified contrasts" begin
    categorical_df = (;
        indication=[1, 2, 3, 1, 2, 3],
        y=zeros(6),
    )
    categorical_builder = @brm begin
        log(Vc) ~ 1 + indication
        log(k10) ~ 1 + indication
        y ~ Normal(log(Vc) + log(k10), 1.0)
    end
    d = brm_descriptor(
        categorical_builder, categorical_df;
        mod=@__MODULE__, name=:categorical_coordinates)
    names = [
        "cat_log_k10_indication_beta.1",
        "cat_log_Vc_indication_beta.1",
        "cat_log_Vc_indication_beta.2",
        "cat_log_k10_indication_beta.2",
    ]

    vc = brm_population_effect_coordinates(
        d, :Vc, names; coefficient=:indication)
    k10 = brm_population_effect_coordinates(
        d, :k10, names; coefficient=:indication)

    @test vc.logical === :Vc
    @test vc.coefficient === :indication
    @test vc.predictor === :indication
    @test vc.output.name === :cat_log_Vc_indication_beta
    @test vc.coordinates == [2, 3]
    @test vc.reference_level == 1
    @test vc.nonreference_levels == [2, 3]
    @test vc.contrasts == [
        (; nonreference_level=2, reference_level=1, coordinate=2),
        (; nonreference_level=3, reference_level=1, coordinate=3),
    ]
    @test vc.link === log
    @test vc.inverse_link === exp
    @test k10.output.name === :cat_log_k10_indication_beta
    @test k10.coordinates == [1, 4]

    # The private-name workaround finds the same posterior columns, but only
    # the descriptor result identifies their predictor, fitted reference, and
    # ordered non-reference levels.
    workaround = findall(
        name -> startswith(name, "cat_log_Vc_indication_beta."), names)
    @test vc.coordinates == workaround

    @test_throws "available labels are (:Intercept, :indication)" begin
        brm_population_effect_coordinates(
            d, :Vc, names; coefficient=:missing)
    end
    @test_throws "owns 2 treatment contrasts but resolves to 1" begin
        brm_population_effect_coordinates(
            d, :Vc, ["cat_log_Vc_indication_beta.1"];
            coefficient=:indication)
    end

    reffed_builder = @brm begin
        mu ~ 0 + factor(indication; ref=3)
        y ~ Normal(mu, 1.0)
    end
    reffed = brm_descriptor(
        reffed_builder, categorical_df;
        mod=@__MODULE__, name=:categorical_reference)
    reffed_result = brm_population_effect_coordinates(
        reffed, :mu,
        ["cat_mu_indication__ref_3_beta.1",
         "cat_mu_indication__ref_3_beta.2"];
        coefficient=:indication)
    @test reffed_result.predictor === :indication
    @test reffed_result.reference_level == 3
    @test reffed_result.nonreference_levels == [2, 1]
    @test reffed_result.coordinates == [1, 2]

    one_level_df = (; indication=fill(1, 4), y=zeros(4))
    one_level_builder = @brm begin
        mu ~ 0 + indication
        y ~ Normal(mu, 1.0)
    end
    one_level = brm_descriptor(
        one_level_builder, one_level_df;
        mod=@__MODULE__, name=:categorical_one_level)
    one_level_result = brm_population_effect_coordinates(
        one_level, :mu, String[]; coefficient=:indication)
    @test one_level_result.reference_level == 1
    @test isempty(one_level_result.nonreference_levels)
    @test isempty(one_level_result.coordinates)
    @test isempty(one_level_result.contrasts)
end

@testset "term coordinates — monotonic simplex and HSGP internals" begin
    n = 12
    term_df = (;
        y=zeros(n),
        vessel_bottle=repeat([0.0, 1.0, 0.0], 4),
        vessel_bottle_20=repeat([0.0, 0.0, 1.0], 4),
        vessel_tablet=repeat([1.0, 0.0, 0.0], 4),
        vessel_tablet_20=repeat([0.0, 1.0, 0.0], 4),
        op_diet=repeat(1:4, 3),
        op_log_dose=collect(range(-1.0, 1.0; length=n)),
    )
    term_builder = @brm begin
        y ~ Normal(log_F, 1.0)
        log_F ~ 0 + vessel_bottle + vessel_bottle_20 + vessel_tablet +
                    vessel_tablet_20 + mo(op_diet) + op_log_dose +
                    hsgp(op_log_dose; k=5)
        effect(log_F, :) ~ Normal(0.0, 0.5)
        effect(log_F, op_log_dose) ~ Normal(0.0, 0.6676)
        length_scale(:, hsgp(op_log_dose)) ~ LogNormal(0.0, 1.0)
        sd(:, hsgp(op_log_dose)) ~ LogNormal(0.0, 1.0)
    end
    d = brm_descriptor(term_builder, term_df;
                       mod=@__MODULE__, name=:joint_log_F, highlights=())
    @test popcoefnames(d.plan.parent, :log_F) == [
        :vessel_bottle, :vessel_bottle_20, :vessel_tablet,
        :vessel_tablet_20, :mo_op_diet, :op_log_dose,
    ]

    prob = brm_execute(d, :fit)
    names = StanBlocks.BridgeStan.param_names(
        prob.model; include_tp=false, include_gq=false)
    simplex = brm_term_coordinates(
        d, :log_F, names; term=:mo_op_diet, parameter=:simplex)
    rho = brm_term_coordinates(
        d, :log_F, names;
        term=:hsgp_op_log_dose, parameter=:length_scale)
    amplitude = brm_term_coordinates(
        d, :log_F, names; term=:hsgp_op_log_dose, parameter=:sd)
    weights = brm_term_coordinates(
        d, :log_F, names;
        term=:hsgp_op_log_dose, parameter=:basis_weights)

    @test length(simplex.coordinates) == 3
    @test length(rho.coordinates) == 1
    @test length(amplitude.coordinates) == 1
    @test length(weights.coordinates) == 5
    @test simplex.output.declaration.target === :mo_op_diet
    @test weights.output.declaration.target === :hsgp_op_log_dose
    @test all(x -> x.link === identity && x.inverse_link === identity,
              (simplex, rho, amplitude, weights))

    # The numerical workaround and the public selector address the SAME draws.
    # What the selector removes is the consumer's dependency on these private
    # compiler spellings, not a mathematical difference in the posterior.
    workaround_simplex = findall(
        name -> startswith(name, "mo_op_diet_simplex_incr."), names)
    workaround_rho = findall(==("hsgp_op_log_dose_rho_iso"), names)
    workaround_amplitude = findall(==("hsgp_op_log_dose_sigma"), names)
    workaround_weights = findall(
        name -> startswith(name, "hsgp_op_log_dose_beta_raw."), names)
    @test simplex.coordinates == workaround_simplex
    @test rho.coordinates == workaround_rho
    @test amplitude.coordinates == workaround_amplitude
    @test weights.coordinates == workaround_weights

    @test_throws "available term labels" brm_term_coordinates(
        d, :log_F, names; term=:mo_missing, parameter=:simplex)
    @test_throws "available roles are (:simplex,)" brm_term_coordinates(
        d, :log_F, names; term=:mo_op_diet, parameter=:sd)
    @test_throws "owns 5 constrained coordinates but resolves to 2" begin
        partial_names = names[setdiff(eachindex(names), weights.coordinates[3:end])]
        brm_term_coordinates(
            d, :log_F, partial_names;
            term=:hsgp_op_log_dose, parameter=:basis_weights)
    end
    @test_throws "0 formula declarations" brm_term_coordinates(
        d, :missing, names; term=:mo_op_diet, parameter=:simplex)
end

@testset "semantic output query — role, not emitted name" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__)
    byname = Dict(o.name => o for o in d.outputs)

    # An OBSERVATION's twins carry its logical target. This is the case a
    # consumer previously could not reach semantically: `y_gen` is StanBlocks'
    # name for the predictive carrier of `y`, so anyone wanting that slice had
    # to filter `descriptor.stan.outputs` and hardcode the `_gen` suffix.
    @test byname[:y_gen].logical === :y
    @test byname[:y_likelihood].logical === :y
    @test byname[:z_gen].logical === :z

    # ...and the emitted name is exactly what is NOT part of the contract.
    @test brm_output(d, :y; role=:posterior_predictive).name === :y_gen
    @test brm_output(d, :y; role=:pointwise_loglik).name === :y_likelihood
    @test brm_output(d, :z; role=:posterior_predictive).name === :z_gen

    # Two carriers, one target: unqualified must FAIL rather than pick one by
    # descriptor order, and the message must name the roles to choose from.
    @test_throws ErrorException brm_output(d, :y)
    msg = try; brm_output(d, :y); catch e; sprint(showerror, e); end
    @test occursin("posterior_predictive", msg) && occursin("pointwise_loglik", msg)
    @test occursin("role=", msg)

    # A single-carrier target is unaffected by the new keyword, and a role that
    # does not apply to it fails closed naming the roles that do.
    @test brm_output(d, :sigma).name === :sigma
    @test brm_output(d, :sigma; role=:parameter).name === :sigma
    @test_throws ErrorException brm_output(d, :sigma; role=:posterior_predictive)
    miss = try
        brm_output(d, :sigma; role=:posterior_predictive)
    catch e; sprint(showerror, e); end
    @test occursin("parameter", miss)

    # The plural query is the discovery half: zero, one or many are all valid.
    predictive = brm_outputs(d; role=:posterior_predictive)
    @test Set(o.name for o in predictive) == Set([:y_gen, :z_gen])
    @test Set(o.logical for o in predictive) == Set([:y, :z])
    @test Set(o.name for o in brm_outputs(d; logical=:y)) == Set([:y_gen, :y_likelihood])
    @test isempty(brm_outputs(d; role=:no_such_role))
    # A collection filter means "any of these".
    @test length(brm_outputs(d; role=(:posterior_predictive, :pointwise_loglik))) == 4
    # An omitted filter does not constrain.
    @test length(brm_outputs(d)) == length(d.outputs)
    @test all(o -> o.kind === :parameter, brm_outputs(d; kind=:parameter))

    # Coordinates resolve through the same role, so a consumer never needs the
    # emitted name to slice BridgeStan's constrained vector.
    names = ["sigma", "y_gen.1", "y_gen.2", "z_gen.1"]
    @test brm_output_coordinates(d, :y, names; role=:posterior_predictive) == [2, 3]
    @test brm_output_coordinates(d, :sigma, names) == [1]
    @test_throws ErrorException brm_output_coordinates(d, :y, names)
end

@testset "execution — BRM -> StanBlocks -> BridgeStan" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__)

    src = brm_execute(d, :transpile)
    @test src isa AbstractString
    @test occursin("y_gen", src)
    @test src == BayesianRegressionModels.stan_code(d.plan)

    prob = brm_execute(d, :fit)
    n = StanBlocks.LogDensityProblems.dimension(prob)
    @test n > 0
    theta = 0.1 .* randn(n)
    lp = StanBlocks.LogDensityProblems.logdensity(prob, theta)
    @test isfinite(lp)

    out = brm_execute(d, :predict; problem=prob, draws=theta, seed=1234)
    @test haskey(out, :y_gen)
    @test length(out.y_gen) == length(df.y)

    ll = brm_execute(d, :pointwise_loglik; problem=prob, draws=theta, seed=1234)
    @test length(ll.y_likelihood) == length(df.y)
end

@testset "replay — the same declaration on genuinely new groups" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)
    new_df = (; x=[0.0, 1.0, 2.0, 3.0],
                g=[7, 7, 8, 8],
                y=[0.2, 0.4, 0.6, 0.8],
                z=[1.2, 1.4, 1.6, 1.8])
    d2 = brm_execute(d, :replay, new_df)
    @test d2 isa BRMDescriptor
    @test d2.plan.data[:x] == new_df.x
    @test d2.plan.data[:n_g] == 2          # the new cohort has two groups
    @test Symbol[o.name for o in d2.outputs] == Symbol[o.name for o in d.outputs]
end

@testset "reprocess is offered exactly when it is supported" begin
    flat_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    d = brm_descriptor(flat_builder, df; mod=@__MODULE__)
    @test :reprocess in Symbol[op.name for op in d.operations]

    d2 = brm_execute(d, :reprocess, (; x=[9.0, 10.0], y=[0.0, 0.0]))
    @test d2.plan.data[:x] == [9.0, 10.0]

    mm_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | mm(g1, g2; weights=(w1, w2)))
        y ~ Normal(mu, sigma)
    end
    mm_df = (;
        g1=["a", "a", "b"],
        g2=["b", "c", "c"],
        w1=[2.0, 1.0, 0.0],
        w2=[1.0, 1.0, 3.0],
        y=[0.1, 0.2, 0.3],
    )
    mm_d = brm_descriptor(mm_builder, mm_df; mod=@__MODULE__)
    @test :reprocess in Symbol[op.name for op in mm_d.operations]

    changed = merge(mm_df, (; w1=[9.0, 1.0, 1.0], w2=[1.0, 3.0, 1.0]))
    changed_d = brm_execute(mm_d, :reprocess, changed)
    idx_key, entry = only((k, e) for (k, e) in mm_d.plan.preproc
                          if e.kind === :multi_membership)
    @test changed_d.plan.data[idx_key] == mm_d.plan.data[idx_key]
    @test changed_d.plan.data[entry.const_.weight_key] ≈
          [0.9, 0.1, 0.25, 0.75, 0.5, 0.5]

    # Ordinary and multi-membership groups both carry frozen replay provenance.
    mixed_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | mm(g1, g2; weights=(w1, w2))) + (1 | g)
        y ~ Normal(mu, sigma)
    end
    mixed_df = merge(mm_df, (; g=[1, 1, 2]))
    mixed_d = brm_descriptor(mixed_builder, mixed_df; mod=@__MODULE__)
    @test :reprocess in Symbol[op.name for op in mixed_d.operations]
end

kernel_builder = @brm begin
    sigma_a ~ Exponential(1)
    sigma_p ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    loc ~ kernel(t, dose, dv, log_CL, log_V) do ts, dd, yy, lCL, lV
        CL = exp(lCL)
        V = exp(lV)
        mu = dd / V * exp(-(CL / V) * ts)
        yy ~ normal(mu, addprop(mu, sigma_a, sigma_p))
        mu
    end
end

kernel_schedule(n; subject=collect(1:n)) = (;
    t=[collect(1.0:(i + 2)) for i in 1:n],
    dose=fill(100.0, n),
    dv=[collect(1.0:(i + 2)) ./ 10 for i in 1:n],
    subject,
)

@testset "kernel(...) — a plate-nested observation" begin
    d = brm_descriptor(kernel_builder, kernel_schedule(3); mod=@__MODULE__, name=:pk)
    byname = Dict(o.name => o for o in d.outputs)

    # Kernel v2 owns BSV through ordinary formula declarations. The population
    # blocks and the shared `|p|` random-effect block therefore retain their
    # ordinary BRM roles; no anonymous kernel-owned L/om/z block exists.
    @test byname[:pop_log_CL_beta_pop].role === :population_effect
    @test byname[:pop_log_V_beta_pop].role === :population_effect
    @test byname[:b_p_subject_L].role === :random_effect
    @test byname[:b_p_subject_L].declaration.family === :ranef_correlated_draws
    @test byname[:b_p_subject_tau].role === :random_effect
    @test !any(o -> startswith(string(o.name), "kernel_L_") ||
                   startswith(string(o.name), "kernel_om_") ||
                   startswith(string(o.name), "kernel_z"), d.outputs)
    @test byname[:sigma_a].role === :parameter

    # The plate's group-block carriers — the transformed-parameter cell values
    # AND the collected return — all resolve to the plate DECLARATION, and each
    # named value additionally carries its own logical identity. No
    # compiler-owned suffix or descriptor order is parsed. The carrier's Stan
    # `kind` is a StanBlocks likelihood-reachability decision, not part of this
    # contract: the cell values `mu`/`CL`/`V` feed the in-cell `~` and so land
    # in transformed parameters, while the collected `loc` return feeds no
    # likelihood and lands in generated quantities — so this does NOT pin kind.
    plate_outputs = [o for o in d.outputs if o.role === :group_block]
    @test !isempty(plate_outputs)
    @test all(o -> o.declaration.target === :loc, plate_outputs)
    primary = brm_output(d, :loc)
    @test primary.logical === :loc
    @test primary.name in (o.name for o in plate_outputs)
    @test startswith(string(primary.name), "loc__pl_mem_")

    # Every value this cell names is addressable, including the scratch link
    # inverses. That is the deliberate width of the rule (decision `1tpze5q`):
    # `CL = exp(lCL)` is a value the author named, so it gets an address; the
    # descriptor does not try to guess which names are "primary".
    @test byname[:loc_CL].role === :group_block
    @test byname[:loc_CL].logical === :CL
    @test byname[:loc_V].role === :group_block
    @test byname[:loc_V].logical === :V
    @test brm_output(d, :CL).name === :loc_CL
    @test brm_output(d, :mu).logical === :mu       # the returned cell value

    @test :dv in d.columns && :subject in d.columns && :dose in d.columns

    # The ragged plate-sliced observation is recognised as conditioned-on, and
    # the model is fittable. Both were briefly wrong upstream — the traced-`~`
    # walk stopped at the `getfield` in `dv.mem[start:end]` — and BRM carried
    # workarounds for them until `44f58fa` fixed the walk. These now assert
    # PLAIN DELEGATION: if a future regression reintroduces that blind spot,
    # these fail rather than being silently absorbed.
    @test Dict(i.name => i for i in d.inputs)[:dv].observed

    ops = Symbol[op.name for op in d.operations]
    @test :fit in ops
    prob = brm_execute(d, :fit)
    n = StanBlocks.LogDensityProblems.dimension(prob)
    @test n > 0
    @test isfinite(StanBlocks.LogDensityProblems.logdensity(prob, 0.1 .* randn(n)))

    constrained_names = StanBlocks.BridgeStan.param_names(
        prob.model; include_tp=true, include_gq=true)
    loc_coordinates = brm_output_coordinates(d, :loc, constrained_names)
    @test length(loc_coordinates) == sum(length, kernel_schedule(3).t)
    @test all(startswith(string(primary.name) * "."),
              constrained_names[loc_coordinates])
    @test_throws ErrorException brm_output_coordinates(d, :loc, ["not_loc.1"])

    # A ragged observation left INSIDE the plate cell (`yy ~ normal(...)` above)
    # keeps the observed base through StanBlocks' compiler-owned loop. Its draw
    # is flat over the ragged backing memory, its log likelihood is aggregate per
    # subject, and both descriptor outputs carry the observed inclusive ends.
    # The local alias `yy` is deliberately absent from the public names: BRM
    # resolves through the upstream `source === :dv` link.
    draw = brm_output(d, :dv; role=:posterior_predictive)
    loglik = brm_output(d, :dv; role=:pointwise_loglik)
    @test draw.name === :dv_gen
    @test loglik.name === :dv_likelihood
    @test draw.source === :dv && loglik.source === :dv
    @test draw.logical === :dv && loglik.logical === :dv
    @test Set(o.name for o in brm_outputs(d; logical=:dv)) ==
          Set([:dv_gen, :dv_likelihood])

    ends = cumsum(length.(kernel_schedule(3).dv))
    @test draw.segments == ends
    @test loglik.segments == ends
    draw_coordinates = brm_output_coordinates(
        d, :dv, constrained_names; role=:posterior_predictive)
    loglik_coordinates = brm_output_coordinates(
        d, :dv, constrained_names; role=:pointwise_loglik)
    @test length(draw_coordinates) == last(ends)
    @test length(loglik_coordinates) == length(ends)

    # A varying-length plate member carries its OWN inclusive group ends, just
    # like the observation twins. These compose with the primary carrier's
    # coordinates without parsing the compiler-owned backing name.
    @test primary.segments == ends
    @test brm_output(d, :mu).segments == ends
    @test brm_output(d, :CL).segments === nothing

    # A builder-backed kernel can rebuild for genuinely new groups and can
    # reprocess the same fitted groups without changing parameter coordinates.
    @test :replay in ops
    @test :reprocess in ops
    @test brm_execute(d, :replay, kernel_schedule(5)) isa BRMDescriptor

    @test :predict in ops
    @test :pointwise_loglik in ops
    @test isempty(d.unpredictable)
end

# The equivalent top-level ragged spelling: return the kernel's ragged result,
# then apply the family outside the cell. It owns the same flat `dv_gen`,
# aggregate `dv_likelihood`, and per-subject `segments` contract as the in-cell
# form above.
toplevel_ragged_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, log_CL, log_V) do ts, dd, lCL, lV
        CL = exp(lCL)
        V = exp(lV)
        dd / V * exp(-(CL / V) * ts)
    end
    dv ~ Normal(pred, sigma)
end

@testset "top-level ragged observation — carrier and segments by role" begin
    sched = kernel_schedule(3)
    d = brm_descriptor(toplevel_ragged_builder, sched; mod=@__MODULE__, name=:pk_flat)

    draw = brm_output(d, :dv; role=:posterior_predictive)
    loglik = brm_output(d, :dv; role=:pointwise_loglik)
    @test draw.logical === :dv && loglik.logical === :dv
    @test draw.source === :dv && loglik.source === :dv
    @test draw.name !== loglik.name
    # Two carriers, so unqualified must refuse rather than pick.
    @test_throws ErrorException brm_output(d, :dv)
    @test Set(o.name for o in brm_outputs(d; logical=:dv)) ==
          Set([draw.name, loglik.name])

    # `segments` is the metadata BRM used to drop. Inclusive per-subject ends
    # on the flat predictive carrier; the aggregate likelihood has one entry
    # per subject, so it is NOT the same length.
    @test draw.segments == cumsum(length.(sched.t))
    @test last(draw.segments) == sum(length, sched.t)

    # ...and it composes directly with the coordinates, which is the whole
    # point: subject g's posterior columns without re-deriving boundaries.
    prob = brm_execute(d, :fit)
    constrained = StanBlocks.BridgeStan.param_names(
        prob.model; include_tp=true, include_gq=true)
    cols = brm_output_coordinates(d, :dv, constrained; role=:posterior_predictive)
    @test length(cols) == last(draw.segments)
    starts = [1; draw.segments[1:end-1] .+ 1]
    per_subject = [cols[s:e] for (s, e) in zip(starts, draw.segments)]
    @test length.(per_subject) == length.(sched.t)
    @test reduce(vcat, per_subject) == cols
end

scalar_kernel_builder = @brm begin
    sigma ~ Exponential(1)
    log_scale ~ 1 + (1 | p | subject)
    pred ~ kernel(dose, dv, log_scale) do dd, yy, ls
        mu = (dd / 10.0) * exp(ls)
        yy ~ normal(mu, sigma)
        mu
    end
end

scalar_schedule(n) = (; dose=fill(100.0, n), dv=collect(1.0:n) ./ 10,
                        subject=collect(1:n))

# A plate observation on a NON-RAGGED (scalar-per-subject) base. StanBlocks
# emits and describes the `<data_source>_gen` twin, so BRM must offer a
# prediction operation that resolves through the upstream `source` link.
@testset "non-ragged plate observation — :predict stays consistent" begin
    d = brm_descriptor(scalar_kernel_builder, scalar_schedule(4);
                       mod=@__MODULE__, name=:pk_scalar)
    ops = Symbol[op.name for op in d.operations]

    @test :fit in ops
    @test :predict in ops
    @test brm_output(d, :pred).name === :pred
    @test brm_operation(d, :predict).outputs == (:dv_gen,)

    # THE plate case that does have a predictive carrier, reached by role rather
    # than by the `_gen` suffix. `:predict` names `dv_gen` above; a consumer
    # should never have to read that name off an operation to slice a posterior.
    predictive = brm_output(d, :dv; role=:posterior_predictive)
    @test predictive.name === :dv_gen
    @test predictive.logical === :dv
    @test predictive.source === :dv
    @test brm_outputs(d; role=:posterior_predictive) == [predictive]
    @test brm_output_coordinates(d, :dv, ["sigma_a", "dv_gen.1", "dv_gen.2"];
                                 role=:posterior_predictive) == [2, 3]
    # Non-ragged, so no group boundaries — `segments` stays honest.
    @test predictive.segments === nothing
    @test isempty(d.unpredictable)
    prob = brm_execute(d, :fit)
    n = StanBlocks.LogDensityProblems.dimension(prob)
    out = brm_execute(d, :predict; problem=prob, draws=0.1 .* randn(n), seed=7)
    @test haskey(out, :dv_gen)

    @test :reprocess in ops
    @test brm_execute(d, :replay, scalar_schedule(5)) isa BRMDescriptor
end

# A named deterministic value INSIDE the cell — the noise-free location an
# in-cell observation is drawn around. It is owned by the plate declaration but
# never occurs in its return binding, so the return-binding rule cannot reach
# it. It has a binding of its own, and that is what makes it addressable — with
# no annotation on the term (decision `1tpze5q`).
qt_kernel_builder = @brm begin
    sigma    ~ Exponential(1)
    qt_sigma ~ Exponential(1)
    log_CL   ~ 1 + (1 | p | subject)
    qt_base  ~ 1 + (1 | p | subject)
    pk_loc ~ kernel(t, dose, dv, qt_y, qt_idx, log_CL, qt_base) do ts, dd, yy, qy, qidx, lCL, qbase
        conc = (dd / 10.0) .* exp(-exp(lCL) .* ts)
        # Reporter shape: a dense concentration workspace sized from the
        # largest read index, then projected onto a ragged observation axis.
        # Descriptor reflection must evaluate Stan's collection `max` with
        # Julia `maximum`; Base.max(::Vector) is not a valid call.
        dense_conc = rep_vector(qbase, max(qidx))
        qt_loc = dense_conc[qidx] .+ 2.0 .* conc[qidx]
        qy ~ normal(qt_loc, qt_sigma)
        yy ~ normal(conc, sigma)
        conc
    end
end

qt_schedule(n) = (;
    t=[collect(1.0:(i + 2)) for i in 1:n],
    dose=fill(100.0, n),
    dv=[collect(1.0:(i + 2)) ./ 10 for i in 1:n],
    qt_y=[collect(1.0:i) .+ 10.0 for i in 1:n],
    # Read indices deliberately skip the first two dense locations: the
    # observation-axis length (`length`) and dense workspace extent (`max`)
    # are different contracts, as in the reported joint PK/QT model.
    qt_idx=[collect(3:(i + 2)) for i in 1:n],
    subject=collect(1:n),
)

@testset "kernel(...) — named cell values are addressable, no annotation" begin
    nsub = 3
    d = brm_descriptor(qt_kernel_builder, qt_schedule(nsub);
                       mod=@__MODULE__, name=:pk_qt)

    # The reporter's target: address the fitted noise-free QT location by the
    # name the cell author bound, with no emitted name parsed in the consumer
    # and nothing added to the formula.
    published = brm_output(d, :qt_loc)
    @test published.logical === :qt_loc
    @test published.role === :group_block
    @test published.kind === :transformed_parameter
    @test published.declaration.target === :pk_loc     # still the plate's
    @test published.name !== :qt_loc                   # a compiler-owned carrier

    sched = qt_schedule(nsub)
    pk_ends = cumsum(length.(sched.t))
    qt_ends = cumsum(length.(sched.qt_idx))
    dense_ends = cumsum(maximum.(sched.qt_idx))
    @test published.segments == qt_ends

    # It is a PLATE member, resolved by the same rule as the collected return —
    # not a name matched by prefix or picked by descriptor order.
    plate_outputs = [o for o in d.outputs
                     if o.kind === :transformed_parameter && o.role === :group_block]
    @test published.name in (o.name for o in plate_outputs)

    # EVERY named cell value, not just one: the other local and the return.
    conc = brm_output(d, :conc)
    dense_conc = brm_output(d, :dense_conc)
    ret = brm_output(d, :pk_loc)
    @test conc.logical === :conc
    @test dense_conc.logical === :dense_conc
    @test ret.logical === :pk_loc
    @test conc.segments == pk_ends
    @test dense_conc.segments == dense_ends
    @test ret.segments == pk_ends
    @test Set([:qt_loc, :conc, :dense_conc, :pk_loc]) ⊆
          Set(o.logical for o in d.outputs if !isnothing(o.logical))

    # It compiles and the carrier is real. `:qt_loc` has one element per QT
    # observation, `:pk_loc` one per PK time — different axes, both correct.
    ops = Symbol[op.name for op in d.operations]
    @test :fit in ops
    prob = brm_execute(d, :fit)
    n = StanBlocks.LogDensityProblems.dimension(prob)
    @test isfinite(StanBlocks.LogDensityProblems.logdensity(prob, 0.1 .* randn(n)))

    constrained_names = StanBlocks.BridgeStan.param_names(
        prob.model; include_tp=true, include_gq=true)
    qt_cols = brm_output_coordinates(d, :qt_loc, constrained_names)
    @test length(qt_cols) == last(qt_ends)
    @test all(startswith(string(published.name) * "."), constrained_names[qt_cols])
    qt_starts = [1; qt_ends[1:end-1] .+ 1]
    qt_groups = [qt_cols[s:e] for (s, e) in zip(qt_starts, qt_ends)]
    @test length.(qt_groups) == length.(sched.qt_idx)
    @test reduce(vcat, qt_groups) == qt_cols

    # The published value is the NOISE-FREE location, not the predictive twin
    # drawn around it. They are different outputs at different Stan stages.
    draw = brm_output(d, :qt_y; role=:posterior_predictive)
    @test draw.name === :qt_y_gen
    @test draw.generative === :draw
    @test published.generative !== :draw
    @test isdisjoint(Set(qt_cols),
                     Set(brm_output_coordinates(d, :qt_y, constrained_names;
                                                role=:posterior_predictive)))

    # ...and it holds the value the author bound. The cell RETURNS `conc`, so
    # one program carries the same quantity through two independent carriers:
    # the collected return and the named cell value. Distinct names (asserted,
    # so an aliasing emitter fails here rather than passing vacuously),
    # identical numbers.
    @test ret.name !== conc.name
    tp_names = StanBlocks.BridgeStan.param_names(
        prob.model; include_tp=true, include_gq=false)
    theta = 0.1 .* randn(n)
    tp = StanBlocks.BridgeStan.param_constrain(
        prob.model, theta; include_tp=true, include_gq=false)
    @test tp[brm_output_coordinates(d, :pk_loc, tp_names)] ==
          tp[brm_output_coordinates(d, :conc, tp_names)]
    # The QT location is a real, distinct quantity in the same draw.
    @test all(isfinite, tp[brm_output_coordinates(d, :qt_loc, tp_names)])

    # Replay keeps every cell value addressable on new subjects.
    replay_sched = qt_schedule(5)
    replayed = brm_execute(d, :replay, replay_sched)
    @test brm_output(replayed, :qt_loc).logical === :qt_loc
    @test brm_output(replayed, :conc).logical === :conc
    @test brm_output(replayed, :dense_conc).logical === :dense_conc
    @test brm_output(replayed, :pk_loc).logical === :pk_loc
    @test brm_output(replayed, :qt_loc).segments ==
          cumsum(length.(replay_sched.qt_idx))
    @test brm_output(replayed, :conc).segments ==
          cumsum(length.(replay_sched.t))
    @test brm_output(replayed, :dense_conc).segments ==
          cumsum(maximum.(replay_sched.qt_idx))
    @test brm_output(replayed, :pk_loc).segments ==
          cumsum(length.(replay_sched.t))
end

@testset "two cells naming one value — ambiguity, not failure" begin
    # `mu` in a PK cell and a PD cell is ordinary. Their emitted bindings differ
    # (`pred_a_mu` / `pred_b_mu`), so both are claimed and the model BUILDS. The
    # BRM address `:mu` then names two quantities, which resolves through the
    # descriptor's existing one-logical-many-carriers contract — the same one an
    # observation's predictive and pointwise twins already use.
    collide = @brm begin
        sigma ~ Exponential(1)
        log_a ~ 1 + (1 | p | subject)
        log_b ~ 1 + (1 | q | subject)
        pred_a ~ kernel(dose, dv, log_a) do dd, yy, ls
            mu = (dd / 10.0) * exp(ls)
            yy ~ normal(mu, sigma)
            mu
        end
        pred_b ~ kernel(dose, log_b) do dd, ls
            mu = (dd / 20.0) * exp(ls)
            mu
        end
    end
    d = brm_descriptor(collide, scalar_schedule(4); mod=@__MODULE__, name=:collide)

    # Ambiguous singular query refuses, and NAMES the owners — `role=` cannot
    # separate these (both `:group_block`), so reporting only the role would
    # name a discriminator that does not discriminate.
    err = try; brm_output(d, :mu); catch e; e; end
    @test err isa ErrorException
    @test occursin("spans several emitted carriers", err.msg)
    @test occursin("pred_a", err.msg) && occursin("pred_b", err.msg)
    @test occursin("declaration.target", err.msg)
    @test !occursin("pass `role=` to select one", err.msg)

    # The plural discovery query returns both, selectable by owner.
    both = brm_outputs(d; logical=:mu)
    @test length(both) == 2
    @test Set(o.declaration.target for o in both) == Set([:pred_a, :pred_b])

    # Every unambiguous name around them still resolves directly.
    @test brm_output(d, :pred_a).logical === :pred_a
    @test brm_output(d, :pred_b).logical === :pred_b
    @test brm_output(d, :sigma).logical === :sigma
end

@testset "kernel(...) accepts no keywords" begin
    # A cell value shadowing an existing model binding never reaches BRM's own
    # address space: StanBlocks' plate tracing refuses it first. Asserting it
    # here keeps the descriptor's collision reasoning honest — if this ever
    # starts building, BRM owes that collision an answer.
    shadow = @brm begin
        sigma ~ Exponential(1)
        log_scale ~ 1 + (1 | p | subject)
        pred ~ kernel(dose, dv, log_scale) do dd, yy, ls
            sigma = (dd / 10.0) * exp(ls)
            yy ~ normal(sigma, 1.0)
            sigma
        end
    end
    @test_throws Exception brm_descriptor(shadow, scalar_schedule(4); mod=@__MODULE__)

    # `kernel(...)` takes NO keywords now that nothing needs annotating. An
    # unknown one is a typo or retired syntax, and silently ignoring it is how
    # retired v1 syntax passed a consumer's construction-time gate for eight
    # days (snag `by-and-n-eta-are-3625f645`). Caught at BRMI construction, so
    # no Stan toolchain is needed to gate on it.
    stray = @brm begin
        sigma ~ Exponential(1)
        log_scale ~ 1 + (1 | p | subject)
        pred ~ kernel(dose, dv, log_scale; outputs=(:mu,)) do dd, yy, ls
            mu = (dd / 10.0) * exp(ls)
            yy ~ normal(mu, sigma)
            mu
        end
    end
    err = try; stray(scalar_schedule(4)); catch e; e; end
    @test err isa ErrorException
    @test occursin("does not accept `outputs=`", err.msg)
    @test occursin("accepts no keywords", err.msg)

    # The retired keywords keep their own guidance rather than falling through
    # to the generic message.
    retired = @brm begin
        sigma ~ Exponential(1)
        log_scale ~ 1 + (1 | p | subject)
        pred ~ kernel(dose, dv, log_scale; n_eta=2) do dd, yy, ls
            mu = (dd / 10.0) * exp(ls)
            yy ~ normal(mu, sigma)
            mu
        end
    end
    err = try; retired(scalar_schedule(4)); catch e; e; end
    @test err isa ErrorException
    @test occursin("no longer accepts `n_eta=`", err.msg)
end

@testset "fail closed" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)

    # 1. An operation the model does not offer errors and NAMES the ones it does.
    err = try; brm_operation(d, :simulate); catch e; e; end
    @test err isa ErrorException
    @test occursin("does not offer operation `simulate`", err.msg)
    @test occursin("predict", err.msg)
    @test_throws ErrorException brm_execute(d, :simulate)

    # 2. Suppressing an operation that was never derived is loud — a caller
    #    doing that is holding exactly the stale list this type removes.
    @test_throws ErrorException brm_descriptor(
        hier_builder, df; mod=@__MODULE__, operations=Dict(:simulate => nothing))

    # 3. Retitling an operation that is not offered is loud for the same reason.
    @test_throws ErrorException brm_descriptor(
        hier_builder, df; mod=@__MODULE__, titles=Dict(:simulate => "nope"))

    # 4. A model with no predictive draw is not offered :predict at all,
    #    instead of offering one that would return nothing usable.
    prior_only = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
    end
    dp = brm_descriptor(prior_only, df; mod=@__MODULE__)
    @test :predict ∉ Symbol[op.name for op in dp.operations]
    @test :transpile in Symbol[op.name for op in dp.operations]
end

@testset "extension points" begin
    # (a) add an operation the derivation cannot know about
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                       operations=Dict(:summarise => (dd; kwargs...) -> length(dd.outputs)))
    @test :summarise in Symbol[op.name for op in d.operations]
    @test brm_operation(d, :summarise).origin === :override
    @test brm_execute(d, :summarise) == length(d.outputs)

    # (b) replace a derived operation's behaviour, keeping its identity
    d2 = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                        operations=Dict(:transpile => (dd; kwargs...) -> "STUB"))
    @test brm_execute(d2, :transpile) == "STUB"
    @test brm_operation(d2, :transpile).origin === :override

    # (c) suppress a derived operation — the button simply is not there
    d3 = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                        operations=Dict(:pointwise_loglik => nothing))
    @test :pointwise_loglik ∉ Symbol[op.name for op in d3.operations]
    @test :fit in Symbol[op.name for op in d3.operations]

    # (d) relabel
    d4 = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                        titles=Dict(:fit => "Run the sampler"))
    @test brm_operation(d4, :fit).title == "Run the sampler"
    @test brm_operation(d4, :fit).origin === :stan   # still the derived runner
end
