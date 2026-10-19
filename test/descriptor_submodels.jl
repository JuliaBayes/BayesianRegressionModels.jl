# test/descriptor_submodels.jl — consumer @slic submodels on the descriptor
# (`d.submodels`) and optional Stan-definition highlights (`brm_highlight`).
#
# Run: julia --project=test test/descriptor_submodels.jl

using Test
using BayesianRegressionModels
using StanBlocks

module ToySubmodels
using BayesianRegressionModels, StanBlocks
import BayesianRegressionModels: _sb_submodel_rhs!
using BayesianRegressionModels: getargs, name

StanBlocks.@deffun toy_shape(x::vector[n], loc::real, slope::real)::vector[n] =
    inv_logit((x .- loc) .* slope)
StanBlocks.@deffun toy_unused(x::real)::real = x + 1.0

# A consumer-authored submodel that owns its parameters and priors.
function saturating end
const _sb_saturating = StanBlocks.@slic begin
    loc ~ normal(0.0, 1.0)          # half-effect location
    slope ~ normal(0.0, 2.0; lower=0.0)
    return toy_shape(dose, loc, slope)
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(saturating), rhs) = begin
    (dose_term,) = getargs(rhs)
    push!(stmts, :($target ~ _sb_saturating(; dose=$(name(dose_term)))))
    :done
end

# A submodel that calls another one by name.
function scaled end
const _sb_inner = StanBlocks.@slic begin
    loc ~ normal(0.0, 1.0)
    return toy_shape(dose, loc, 1.0)
end
const _sb_outer = StanBlocks.@slic begin
    inner ~ _sb_inner(; dose=dose)
    scale ~ normal(0.0, 1.0; lower=0.0)
    return inner .* scale
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(scaled), rhs) = begin
    (dose_term,) = getargs(rhs)
    push!(stmts, :($target ~ _sb_outer(; dose=$(name(dose_term)))))
    :done
end

# A submodel spliced into the emitted body as a value, named and unnamed.
function valued end
function configured end
const _sb_valued = StanBlocks.@slic begin
    loc ~ normal(0.0, 1.0)
    return toy_shape(dose, loc, 2.0)
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(valued), rhs) = begin
    (dose_term,) = getargs(rhs)
    push!(stmts, :($target ~ $(_sb_valued)(; dose=$(name(dose_term)))))
    :done
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(configured), rhs) = begin
    (dose_term,) = getargs(rhs)
    variant = Base.merge(_sb_valued, quote
        loc ~ normal(1.0, 0.5)
    end)
    push!(stmts, :($target ~ $(variant)(; dose=$(name(dose_term)))))
    :done
end

# A named submodel function: its body is selected by dispatch at trace time.
function named end
StanBlocks.@slic named_shift(dose) = begin
    loc ~ normal(0.0, 1.0)
    return toy_shape(dose, loc, 0.5)
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(named), rhs) = begin
    (dose_term,) = getargs(rhs)
    push!(stmts, :($target ~ named_shift($(name(dose_term)))))
    :done
end

# A submodel with a deterministic `=` value next to its parameter.
function rated end
const _sb_rated = StanBlocks.@slic begin
    log_rate ~ normal(0.0, 1.0)
    rate = exp(log_rate)
    return toy_shape(dose, rate, 1.0)
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(rated), rhs) = begin
    (dose_term,) = getargs(rhs)
    push!(stmts, :($target ~ _sb_rated(; dose=$(name(dose_term)))))
    :done
end

# A submodel whose own parameter shares its name with a model-level prior.
function noisy end
const _sb_noisy = StanBlocks.@slic begin
    sigma ~ normal(0.0, 1.0; lower=0.0)
    return dose .* sigma
end
_sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(noisy), rhs) = begin
    (dose_term,) = getargs(rhs)
    push!(stmts, :($target ~ _sb_noisy(; dose=$(name(dose_term)))))
    :done
end

df = (; dose=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], x=[0.1, -0.2, 0.3, 0.0, 0.5, -0.4],
        g=[1, 1, 2, 2, 3, 3], y=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        z=[0.2, 0.0, 0.4, 0.1, 0.6, -0.1])

toy_builder = @brm begin
    sigma ~ Exponential(1)
    effect_scale ~ Exponential(1)
    effect ~ saturating(dose)
    mu ~ 1 + x + (1 | g)
    y ~ Normal(effect, sigma)
    z ~ Normal(mu, effect_scale)
end

nested_builder = @brm begin
    sigma ~ Exponential(1)
    effect ~ scaled(dose)
    y ~ Normal(effect, sigma)
end

named_builder = @brm begin
    sigma ~ Exponential(1)
    n ~ named(dose)
    y ~ Normal(n, sigma)
end

valued_builder = @brm begin
    sigma ~ Exponential(1)
    a ~ valued(dose)
    b ~ configured(dose)
    y ~ Normal(a, sigma)
    z ~ Normal(b, sigma)
end

# `inner` is the target of a top-level call AND of the call inside `scaled`.
address_builder = @brm begin
    sigma ~ Exponential(1)
    inner ~ valued(dose)
    effect ~ scaled(dose)
    y ~ Normal(inner, sigma)
    z ~ Normal(effect, sigma)
end

rated_builder = @brm begin
    sigma ~ Exponential(1)
    r ~ rated(dose)
    noisy_effect ~ noisy(dose)
    y ~ Normal(r, sigma)
    z ~ Normal(noisy_effect, sigma)
end
end

const TS = ToySubmodels

@testset "a consumer submodel is listed with its call, own priors and source" begin
    d = brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=(), name=:toy)
    @test d.submodels isa Tuple
    @test length(d.submodels) == 1
    s = only(d.submodels)
    @test s isa BRMSubmodel
    @test s.name === :_sb_saturating
    @test s.model isa StanBlocks.SlicModel
    @test s.model.mod === TS
    @test s.target === :effect
    @test s.path == ()
    @test s.parent === nothing
    @test s.trigger == "effect ~ saturating(dose)"
    @test s.declaration.target === :effect
    @test s.declaration.family === :_sb_saturating
    @test s.inputs == (dose=:dose,)
    @test [x.target for x in s.declarations] == [:loc, :slope]
    @test all(x -> x.role === :prior, s.declarations)
    @test [x.family for x in s.declarations] == [:normal, :normal]
    @test s.declarations[2].constraints == (lower=0.0,)
    # The submodel's own parameters, under its prefix. The sibling prior
    # `effect_scale` shares the prefix but is its own declaration, and the
    # returned `effect` carrier is not one of the submodel's outputs.
    @test [o.name for o in s.outputs] == [:effect_loc, :effect_slope]
    @test all(o -> o.kind === :parameter, s.outputs)
    names = Set(o.name for o in d.outputs)
    @test :effect_scale in names && :effect in names
    # Derived from the lowered value: no comment, Julia's printed form.
    @test s.source == "loc ~ normal(0.0, 1.0)\n" *
                      "slope ~ normal(0.0, 2.0; lower = 0.0)\n" *
                      "return toy_shape(dose, loc, slope)"
    # BRM's own population/random-effect submodels are not consumer submodels.
    @test !any(s -> s.model isa StanBlocks.SlicModel &&
                    s.model.mod === BayesianRegressionModels, d.submodels)
    @test occursin("submodels:  _sb_saturating (effect)", sprint(show, d))
    # The listing is presentation: identity is the emitted program's.
    sb = SBBRMI(TS.toy_builder(TS.df); mod=TS, total_groups=())
    @test brm_descriptor(sb).id == d.id
end

@testset "nested submodels are listed under their parent" begin
    d = brm_descriptor(TS.nested_builder, TS.df; mod=TS, total_groups=())
    @test [s.name for s in d.submodels] == [:_sb_outer, :_sb_inner]
    outer, inner = d.submodels
    @test outer.parent === nothing && outer.trigger == "effect ~ scaled(dose)"
    @test [x.target for x in outer.declarations] == [:inner, :scale]
    @test inner.parent == 1
    @test inner.target === :inner
    @test inner.path == (:effect,)
    @test inner.trigger === nothing
    @test inner.inputs == (dose=:dose,)
    @test [x.target for x in inner.declarations] == [:loc]
    # Longest prefix wins: the inner parameter is the inner submodel's.
    outer_names = Set(o.name for o in outer.outputs)
    inner_names = Set(o.name for o in inner.outputs)
    @test :effect_scale in outer_names
    @test :effect_inner_loc in inner_names
    @test isempty(intersect(outer_names, inner_names))
    @test outer.source == "inner ~ _sb_inner(; dose = dose)\n" *
                          "scale ~ normal(0.0, 1.0; lower = 0.0)\n" *
                          "return inner .* scale"
end

@testset "value-spliced submodels resolve their binding by content" begin
    d = brm_descriptor(TS.valued_builder, TS.df; mod=TS, total_groups=())
    @test [s.target for s in d.submodels] == [:a, :b]
    a, b = d.submodels
    @test a.name === :_sb_valued
    @test a.trigger == "a ~ valued(dose)"
    @test a.source == "loc ~ normal(0.0, 1.0)\nreturn toy_shape(dose, loc, 2.0)"
    # An unnamed merge configuration has no binding; its source is the
    # configured body that actually runs.
    @test b.name === nothing
    @test b.source == "loc ~ normal(1.0, 0.5)\nreturn toy_shape(dose, loc, 2.0)"
    @test [o.name for o in a.outputs] == [:a_loc]
    @test [o.name for o in b.outputs] == [:b_loc]
end

@testset "a named submodel function is listed without a body" begin
    d = brm_descriptor(TS.named_builder, TS.df; mod=TS, total_groups=())
    s = only(d.submodels)
    @test s.name === :named_shift
    @test s.model isa StanBlocks.SubmodelFn
    @test s.trigger == "n ~ named(dose)"
    @test s.declaration.arguments == (:dose,)
    @test s.inputs == NamedTuple()
    @test s.declarations == ()
    @test s.source === nothing
    @test [o.name for o in s.outputs] == [:n_loc]
end

@testset "models without consumer submodels list none" begin
    plain = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        y ~ Normal(mu, sigma)
    end
    d = brm_descriptor(plain, TS.df; mod=TS, total_groups=())
    @test d.submodels == ()
    @test !occursin("submodels:", sprint(show, d))
end

@testset "optional highlights" begin
    present = Set(def.name for def in
                  brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=()).stan.definitions)
    @test :toy_shape in present
    @test !(:toy_unused in present)

    d = brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=(),
        highlights=(
            brm_highlight(:toy_unused, "Never emitted"; optional=true),
            :toy_shape => "Saturating shape",
            brm_highlight("toy_shape_missing"; optional=true),
        ))
    @test [(h.name, h.caption) for h in d.highlights] == [(:toy_shape, "Saturating shape")]
    @test [(s.name, s.caption, s.optional) for s in d.highlight_specs] ==
          [(:toy_unused, "Never emitted", true),
           (:toy_shape, "Saturating shape", false),
           (:toy_shape_missing, nothing, true)]
    @test d.highlight_specs[1].signature === nothing
    @test d.highlight_specs[2].signature == only(d.highlights).definition.signature

    # An optional entry that IS emitted resolves like a required one.
    opt = brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=(),
        highlights=(brm_highlight(:toy_shape, "Shape"; optional=true),))
    @test only(opt.highlights).name === :toy_shape
    @test only(opt.highlights).caption == "Shape"
    @test only(opt.highlight_specs).optional

    # Required entries still fail closed; duplicates fail whatever the marker.
    @test_throws ErrorException brm_descriptor(TS.toy_builder, TS.df; mod=TS,
        total_groups=(), highlights=(brm_highlight(:toy_unused; optional=true),
                                     :toy_shape_missing))
    @test_throws ErrorException brm_descriptor(TS.toy_builder, TS.df; mod=TS,
        total_groups=(), highlights=(brm_highlight(:toy_unused),))
    @test_throws ErrorException brm_descriptor(TS.toy_builder, TS.df; mod=TS,
        total_groups=(), highlights=(:toy_shape,
                                     brm_highlight(:toy_shape; optional=true)))
    @test_throws ErrorException brm_descriptor(TS.toy_builder, TS.df; mod=TS,
        total_groups=(), highlights=(42,))

    # Highlights stay presentation metadata.
    @test d.id == brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=()).id
    # Required-only requests behave exactly as before.
    plain = brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=(),
                           highlights=(:toy_shape,))
    @test [h.name for h in plain.highlights] == [:toy_shape]
    @test [(s.name, s.optional) for s in plain.highlight_specs] == [(:toy_shape, false)]
end

@testset "replay re-resolves optional highlights against the new program" begin
    # The pointwise-likelihood helper exists only when a response is bound.
    helper = :jbroadcasted_normal_lpdfs
    prior_df = Base.structdiff(TS.df, NamedTuple{(:y, :z)})
    prior = brm_descriptor(TS.toy_builder, prior_df; mod=TS, total_groups=(),
        highlights=(brm_highlight(helper, "Pointwise likelihood"; optional=true),
                    :toy_shape))
    @test !(helper in (def.name for def in prior.stan.definitions))
    @test [h.name for h in prior.highlights] == [:toy_shape]

    fitted = brm_execute(prior, :replay, TS.df)
    @test helper in (def.name for def in fitted.stan.definitions)
    @test [(h.name, h.caption) for h in fitted.highlights] ==
          [(helper, "Pointwise likelihood"), (:toy_shape, nothing)]
    @test [s.optional for s in fitted.highlight_specs] == [true, false]

    # And back: replaying without the response skips it again instead of raising.
    again = brm_execute(fitted, :replay, prior_df)
    @test [h.name for h in again.highlights] == [:toy_shape]

    # Same-group reprocess keeps the resolved selection.
    re = brm_execute(fitted, :reprocess, TS.df)
    @test [h.name for h in re.highlights] == [helper, :toy_shape]
    @test [s.name for s in re.submodels] == [:_sb_saturating]
end

@testset "positional constructor without the new fields" begin
    d = brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=(),
                       highlights=(:toy_shape => "Shape",))
    manual = BRMDescriptor(d.id, d.name, d.formula, d.plan, d.stan, d.highlights,
                           d.inputs, d.outputs, d.operations, d.columns, d.unpredictable)
    @test manual.submodels == ()
    @test [(s.name, s.caption, s.signature, s.optional) for s in manual.highlight_specs] ==
          [(:toy_shape, "Shape", only(d.highlights).definition.signature, false)]
end

# The message of the error `f` throws.
thrown(f) = try
    f()
    nothing
catch err
    sprint(showerror, err)
end

@testset "a submodel's own values resolve through submodel=" begin
    d = brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=(), name=:toy)
    s = only(d.submodels)
    slope = brm_output(d, :slope; submodel=:effect)
    @test slope === s.outputs[2]
    @test brm_output(d, :loc; submodel=:effect) === s.outputs[1]
    # The same call by its full address and by its `d.submodels` entry.
    @test brm_output(d, :slope; submodel=(:effect,)) === slope
    @test brm_output(d, :slope; submodel=s) === slope
    @test brm_output(d, :slope; submodel=:effect, role=:parameter) === slope

    # Coordinates are the shared resolver's, in the names' own order.
    names = reverse([string(o.name) for o in d.outputs])
    @test brm_output_coordinates(d, :slope, names; submodel=:effect) ==
          brm_output_coordinates(slope, names) ==
          [findfirst(==(string(slope.name)), names)]

    @test brm_outputs(d; submodel=:effect) == collect(s.outputs)
    @test brm_outputs(d; submodel=:effect, logical=:slope) == [slope]
    @test brm_outputs(d; submodel=:effect, logical=(:loc, :slope)) == collect(s.outputs)
    @test brm_outputs(d; submodel=:effect, logical=:missing_name) == BRMOutput[]
    @test brm_outputs(d; submodel=:effect, role=:pointwise_loglik) == BRMOutput[]

    # Without the keyword nothing changes, and the miss names the keyword.
    msg = thrown(() -> brm_output(d, :slope))
    @test occursin("has no emitted posterior carrier", msg)
    @test occursin("pass `submodel=:effect` to address it", msg)
    msg = thrown(() -> brm_output_coordinates(d, :slope, names))
    @test occursin("pass `submodel=:effect` to address it", msg)
    @test brm_output(d, :effect_scale).name === :effect_scale

    # Only names the body binds: the sibling prior `effect_scale` is not the
    # submodel's `scale`.
    msg = thrown(() -> brm_output(d, :scale; submodel=:effect))
    @test occursin("binds no `scale`; it binds (:loc, :slope)", msg)
    msg = thrown(() -> brm_output(d, :slope; submodel=:effect, role=:pointwise_loglik))
    @test occursin("has no carrier with role `pointwise_loglik`", msg)
    msg = thrown(() -> brm_output_coordinates(d, :slope, ["sigma"]; submodel=:effect))
    @test occursin("of `submodel=:effect` resolves to emitted", msg)

    # Addresses that name no call, or not a call of this descriptor.
    msg = thrown(() -> brm_output(d, :slope; submodel=:nope))
    @test occursin("no consumer submodel call at `:nope`", msg)
    @test occursin("(path..., target): ((:effect,),)", msg)
    @test occursin("takes a call's target",
                   thrown(() -> brm_output(d, :slope; submodel="effect")))
    other = only(brm_descriptor(TS.toy_builder, TS.df; mod=TS, total_groups=()).submodels)
    @test occursin("is not an entry of model `toy`",
                   thrown(() -> brm_output(d, :slope; submodel=other)))
    none = @brm begin
        sigma ~ Exponential(1)
        y ~ Normal(0, sigma)
    end
    dn = brm_descriptor(none, TS.df; mod=TS, total_groups=())
    @test occursin("no consumer submodel call at `:effect`. It calls none.",
                   thrown(() -> brm_output(dn, :slope; submodel=:effect)))
end

@testset "nested calls and calls sharing a target" begin
    d = brm_descriptor(TS.nested_builder, TS.df; mod=TS, total_groups=())
    outer, inner = d.submodels
    loc = brm_output(d, :loc; submodel=(:effect, :inner))
    @test loc === only(inner.outputs)
    @test brm_output(d, :loc; submodel=:inner) === loc
    @test brm_output(d, :scale; submodel=:effect) in outer.outputs
    # The outer body binds `inner` too: the nested call's returned value.
    @test brm_output(d, :inner; submodel=:effect) in outer.outputs
    @test occursin("binds no `loc`; it binds (:inner, :scale)",
                   thrown(() -> brm_output(d, :loc; submodel=:effect)))

    d = brm_descriptor(TS.address_builder, TS.df; mod=TS, total_groups=())
    @test [(s.path, s.target) for s in d.submodels] ==
          [((), :inner), ((), :effect), ((:effect,), :inner)]
    msg = thrown(() -> brm_output(d, :loc; submodel=:inner))
    @test occursin("matches 2 consumer submodel calls ((:inner,), (:effect, :inner))", msg)
    top = brm_output(d, :loc; submodel=(:inner,))
    nested = brm_output(d, :loc; submodel=(:effect, :inner))
    @test top !== nested
    @test top in d.submodels[1].outputs && nested in d.submodels[3].outputs
    msg = thrown(() -> brm_output(d, :loc))
    @test occursin("bound inside 2 consumer submodel calls; pass " *
                   "`submodel=(:inner,)` or `submodel=(:effect, :inner)`", msg)
end

@testset "value-spliced and named submodels" begin
    d = brm_descriptor(TS.valued_builder, TS.df; mod=TS, total_groups=())
    a, b = d.submodels
    @test brm_output(d, :loc; submodel=:a) === only(a.outputs)
    @test brm_output(d, :loc; submodel=:b) === only(b.outputs)
    @test occursin("pass `submodel=:a` or `submodel=:b` to address one",
                   thrown(() -> brm_output(d, :loc)))

    # A named submodel function's body is chosen at trace time, so its bound
    # names are unknown; its owned outputs still resolve.
    d = brm_descriptor(TS.named_builder, TS.df; mod=TS, total_groups=())
    s = only(d.submodels)
    @test brm_output(d, :loc; submodel=:n) === only(s.outputs)
    @test occursin("emits no posterior carrier for `slope`",
                   thrown(() -> brm_output(d, :slope; submodel=:n)))
end

@testset "assigned values, and names the model also declares" begin
    d = brm_descriptor(TS.rated_builder, TS.df; mod=TS, total_groups=())
    r, noisy = d.submodels
    rate = brm_output(d, :rate; submodel=:r)
    log_rate = brm_output(d, :log_rate; submodel=:r)
    @test rate in r.outputs && log_rate in r.outputs && rate !== log_rate
    @test rate.kind !== log_rate.kind
    @test brm_outputs(d; submodel=:r, kind=log_rate.kind) == [log_rate]

    # A model-level `sigma` keeps its bare address; the submodel's own `sigma`
    # is reached only through its call.
    top = brm_output(d, :sigma)
    own = brm_output(d, :sigma; submodel=:noisy_effect)
    @test top.declaration.target === :sigma
    @test own === only(noisy.outputs)
    @test top !== own
    @test only(brm_outputs(d; logical=:sigma)) === top
end
