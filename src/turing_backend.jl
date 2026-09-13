# Core Turing-facing types and semantic validation. This file deliberately has
# no dependency on Turing, DynamicPPL, StanBlocks, SBBRMI, or emitted SLIC. The
# package extension supplies the executable model after Turing is loaded.

mutable struct _TuringGenericPlan{C,G,P,S,A,E,Y,L,MR,RM,OW,RF,J}
    context::C
    prepared::G
    predictors::P
    parameters::S
    assignments::A
    distribution::E
    response::Y
    response_name::Symbol
    lhs::L
    missing_response::MR
    response_modifier::RM
    observation_weight::OW
    response_fit::RF
    joint_r2d2::J
    source_ast::Any
end

function Base.getproperty(plan::_TuringGenericPlan, field::Symbol)
    field === :design && return only(getfield(plan, :predictors)).design
    field === :predictor && return only(getfield(plan, :predictors)).predictor
    field === :random_effects && return only(getfield(plan, :predictors)).random_effects
    field === :beta_location && return only(getfield(plan, :predictors)).beta_location
    field === :beta_scale && return only(getfield(plan, :predictors)).beta_scale
    getfield(plan, field)
end

struct _TuringPopulationComponent{P<:_BRMPopulationPredictor,
                                  D<:_BRMPopulationDesign,
                                  B<:AbstractVector,R<:Tuple}
    predictor::P
    design::D
    beta_location::B
    beta_scale::B
    random_effects::R
end

struct _TuringGenericPredictor{C,P,T,R}
    component::C
    priors::P
    terms::T
    r2d2::R
end
_TuringGenericPredictor(component, priors, terms) =
    _TuringGenericPredictor(component, priors, terms, nothing)
function Base.getproperty(p::_TuringGenericPredictor, field::Symbol)
    field in (:component, :priors, :terms, :r2d2) ? getfield(p, field) :
        getproperty(getfield(p, :component), field)
end

mutable struct _TuringMultiResponsePlan{N<:Tuple,P<:Tuple,O<:Tuple,J}
    responses::N
    plans::P
    owners::O
    joint_r2d2::J
    source_ast::Any
end
_TuringMultiResponsePlan(responses, plans, owners, joint_r2d2) =
    _TuringMultiResponsePlan(responses, plans, owners, joint_r2d2, nothing)

struct _TuringSharedGroupPlan{K,M,B}
    key::K
    members::M
    blocks::B
end

function _turing_shared_group_plans(predictors)
    keyed = Dict{Tuple,Vector{Any}}()
    order = Tuple[]
    for (component_index, component) in enumerate(predictors)
        for (block_index, block) in enumerate(component.random_effects)
            isnothing(block.id) && continue
            key = (block.id, block.group, block.by)
            haskey(keyed, key) || (keyed[key] = Any[]; push!(order, key))
            push!(keyed[key], (; predictor=component.predictor.name,
                component_index, block_index, block))
        end
    end
    Tuple(begin
        entries = keyed[key]
        reference = first(entries).block
        for entry in Iterators.drop(entries, 1)
            block = entry.block
            reference.levels == block.levels &&
                reference.strata == block.strata &&
                reference.group_strata == block.group_strata &&
                reference.centered == block.centered &&
                reference.lkj_eta == block.lkj_eta || error(
                    "Turing backend: shared random-effect ID `$(first(key))` " *
                    "has incompatible fitted geometry")
        end
        _TuringSharedGroupPlan(key,
            Tuple((; predictor=e.predictor,
                    component_index=e.component_index,
                    block_index=e.block_index) for e in entries),
            Tuple(e.block for e in entries))
    end for key in order if length(keyed[key]) > 1)
end

struct _TuringReplayState{G<:Tuple}
    resample_groups::G
end
_TuringReplayState() = _TuringReplayState(())

"""
    TuringBRMI(brmi; centered_groups=(), cv_groups=())

A [`BRMI`](@ref) lowered to the Turing backend. `plan` is the strict,
Stan-independent semantic plan; `model` is the concrete DynamicPPL model
provided by `BayesianRegressionModelsTuringExt` when Turing is loaded.

Groups named in `centered_groups` sample their model-scale effects directly;
all others use the default non-centered geometry. `cv_groups` is accepted only
to give a loud boundary: it controls emitted Stan artifact sizing and therefore
does not apply to the dynamic Turing model. Use
`reprocess(backend, new_data; resample_groups=...)` for new group populations.
"""
struct TuringBRMI{P<:BRMI,PL,M,R<:_TuringReplayState}
    parent::P
    plan::PL
    model::M
    replay::R
end
TuringBRMI(parent::BRMI, plan, model) =
    TuringBRMI(parent, plan, model, _TuringReplayState())

Base.parent(x::TuringBRMI) = x.parent
_turing_num_population_coefficients(plan::_TuringMultiResponsePlan) =
    sum(_turing_num_population_coefficients, plan.plans)
_turing_num_population_coefficients(plan::_TuringGenericPlan) =
    sum(component -> size(component.design.matrix, 2), plan.predictors)

_turing_num_observations(plan::_TuringMultiResponsePlan) =
    sum(_turing_num_observations, plan.plans)
_turing_num_observations(plan::_TuringGenericPlan) = length(plan.response)

Base.show(io::IO, x::TuringBRMI{<:BRMI,<:_TuringMultiResponsePlan}) = print(
    io, "TuringBRMI with ", _turing_num_population_coefficients(x.plan),
    " population coefficients across ", length(x.plan.responses),
    " responses and ", _turing_num_observations(x.plan), " observations")
Base.show(io::IO, x::TuringBRMI) = print(
    io, "TuringBRMI with ", _turing_num_population_coefficients(x.plan),
    " population coefficients and ", _turing_num_observations(x.plan),
    " observations")

# Implemented only by the Turing package extension. Keeping the generic here
# lets the core validate and materialise plans without loading Turing.
function _brm_turing_model end

"""
    turing_pointwise_loglikelihoods(backend::TuringBRMI, parameters)

Evaluate the direct-BRMI Turing model's observation sites at one constrained
parameter draw and return a response-named `NamedTuple` of rowwise log
likelihoods. Responses with modelled missing values retain their original row
axis, with `missing` at latent (non-observed) rows.

The Turing extension implements this by re-running DynamicPPL's pointwise
likelihood accumulator, so response modifiers and observation weights use the
same executable distributions as `backend.model`.
"""
function turing_pointwise_loglikelihoods end

"""
    turing_predictive_model(backend::TuringBRMI)

Return the direct-BRMI DynamicPPL model with every response site unobserved.
For chain-level prediction, call `Turing.predict([rng], backend, chain)` so
latent response values from partially missing fitted responses are removed
before DynamicPPL conditions the predictive model. Use this model directly
with `Turing.predict` only when the chain contains no response parameters, or
call [`turing_posterior_predictive`](@ref) for one constrained parameter draw.
"""
function turing_predictive_model end

"""
    turing_generated_quantities(backend::TuringBRMI, parameters)

Return the deterministic quantities produced by the fitted Turing model at one
constrained parameter draw. This is the response-aware BRM entry point to
DynamicPPL's `returned` evaluation.
"""
function turing_generated_quantities end

"""
    turing_posterior_predictive([rng], backend::TuringBRMI, parameters)

Draw every modeled response at one constrained parameter draw. The result is a
response-named `NamedTuple`, including one entry per response in a
multi-response model. Response parameters in `parameters` (for example latent
values fitted through `mi(y)`) are deliberately not conditioned, so every row
is regenerated.
"""
function turing_posterior_predictive end

"""Return the generated Turing model AST used by `backend.model`."""
turing_model_source(backend::TuringBRMI) = backend.plan.source_ast
turing_model_source(backend::TuringBRMI{<:BRMI,<:_TuringMultiResponsePlan}) =
    backend.plan.source_ast

function _turing_unobserved_response(response)
    T = nonmissingtype(eltype(response))
    unobserved = Vector{Union{Missing,T}}(undef, length(response))
    fill!(unobserved, missing)
end

function _turing_predictive_plan(plan::_TuringGenericPlan)
    _TuringGenericPlan(
        plan.context, plan.prepared, plan.predictors, plan.parameters,
        plan.assignments, plan.distribution,
        _turing_unobserved_response(plan.response), plan.response_name,
        plan.lhs, plan.missing_response, plan.response_modifier,
        plan.observation_weight, plan.response_fit, plan.joint_r2d2, plan.source_ast)
end

_turing_predictive_plan(plan::_TuringMultiResponsePlan) =
    _TuringMultiResponsePlan(
        plan.responses,
        Tuple(_turing_predictive_plan(child) for child in plan.plans),
        plan.owners, plan.joint_r2d2)

function _turing_replay_component(
        training::_TuringPopulationComponent,
        fresh::_TuringPopulationComponent,
        context::_BRMBackendContext, resample_groups)
    predictor = _brm_replay_population_predictor(training.predictor, context)
    random_effects = _turing_replay_random_effects(
        training.random_effects, fresh.random_effects, context,
        resample_groups)
    _TuringPopulationComponent(
        predictor, predictor.design, training.beta_location,
        training.beta_scale, random_effects)
end
function _turing_replay_component(
        training::_TuringGenericPredictor, fresh::_TuringGenericPredictor,
        context::_BRMBackendContext, resample_groups)
    _TuringGenericPredictor(
        _turing_replay_component(training.component, fresh.component,
                                 context, resample_groups),
        training.priors,
        Tuple(_brm_replay_term(old, new, context)
              for (old, new) in zip(training.terms, fresh.terms)),
        training.r2d2)
end

function _turing_replay_random_effects(
        training::Tuple, fresh::Tuple, context::_BRMBackendContext,
        resample_groups)
    length(training) == length(fresh) || error(
        "Turing backend: replay changed the random-effect block count")
    Tuple(map(zip(training, fresh)) do (old, new)
        _turing_same_random_effect_identity(old, new) || error(
            "Turing backend: replay changed random-effect block identity")
        _turing_block_is_resampled(old, resample_groups) ?
            _turing_resampled_random_effect_plan(old, new) :
            _brm_replay_random_effect_plan(old, context)
    end)
end

_turing_same_random_effect_identity(old, new) =
    old.predictor === new.predictor && old.id === new.id &&
    old.group === new.group && old.by === new.by
function _turing_same_random_effect_identity(
        old::_BRMMultiMembershipPlan, new::_BRMMultiMembershipPlan)
    old.predictor === new.predictor && old.groups == new.groups &&
    old.weight_sources == new.weight_sources && old.normalize == new.normalize
end

_turing_block_group_names(block::_BRMRandomEffectPlan) = Set((block.group,))
_turing_block_group_names(block::_BRMMultiMembershipPlan) = Set(block.groups)
_turing_block_is_resampled(block::_BRMRandomEffectPlan, groups) =
    block.group in groups
function _turing_block_is_resampled(block::_BRMMultiMembershipPlan, groups)
    required = Set(block.groups)
    selected = intersect(required, groups)
    isempty(selected) && return false
    selected == required || error(
        "Turing backend: multi-membership block `mm($(join(block.groups, ", ")))` " *
        "must be resampled as one shared population; select every membership " *
        "column $(collect(block.groups)), not $(sort!(collect(selected)))")
    true
end

function _turing_resampled_random_effect_plan(old::_BRMRandomEffectPlan,
                                              new::_BRMRandomEffectPlan)
    _BRMRandomEffectPlan(
        new.predictor, new.id, new.group, new.by, new.levels, new.strata,
        new.indices, new.stratum_indices, new.group_strata, new.columns,
        new.matrix, new.intercept_only, new.zero_correlation, old.centered,
        old.sd_prior, old.sd_family, old.sd_rate, old.lkj_eta)
end

function _turing_resampled_random_effect_plan(
        old::_BRMMultiMembershipPlan, new::_BRMMultiMembershipPlan)
    _BRMMultiMembershipPlan(
        new.predictor, nothing, new.group, nothing, new.levels, Any[],
        new.indices, Int[], Int[], new.columns, new.matrix,
        new.intercept_only, false, false, old.sd_prior, old.sd_family, old.sd_rate,
        old.lkj_eta, new.groups, new.weight_sources, new.weights, new.n_obs,
        new.n_memberships, new.normalize)
end

function _turing_replay_plan(
        training::_TuringGenericPlan, fresh::_TuringGenericPlan,
        resample_groups=Set{Symbol}())
    length(training.predictors) == length(fresh.predictors) || error(
        "Turing backend: replay changed the predictor count")
    predictors = Tuple(_turing_replay_component(old, new, fresh.context,
                                                resample_groups)
                       for (old, new) in zip(training.predictors,
                                             fresh.predictors))
    _TuringGenericPlan(
        fresh.context, fresh.prepared, predictors, training.parameters,
        fresh.assignments, fresh.distribution,
        fresh.response, fresh.response_name, fresh.lhs, fresh.missing_response,
        fresh.response_modifier, fresh.observation_weight,
        fresh.response_fit, training.joint_r2d2, fresh.source_ast)
end

function _turing_replay_plan(
        training::_TuringMultiResponsePlan,
        fresh::_TuringMultiResponsePlan, resample_groups=Set{Symbol}())
    training.responses == fresh.responses || error(
        "Turing backend: replay changed the response-name set")
    length(training.plans) == length(fresh.plans) || error(
        "Turing backend: replay changed the response-plan count")
    plans = Tuple(_turing_replay_plan(old, new, resample_groups)
                  for (old, new) in zip(training.plans, fresh.plans))
    _TuringMultiResponsePlan(training.responses, plans, training.owners,
                             training.joint_r2d2)
end


function _turing_random_effect_group_names(blocks)
    groups = Set{Symbol}()
    foreach(block -> union!(groups, _turing_block_group_names(block)), blocks)
    groups
end
function _turing_group_names(plan::_TuringGenericPlan)
    groups = Set{Symbol}()
    foreach(component -> union!(groups,
        _turing_random_effect_group_names(component.random_effects)),
        plan.predictors)
    groups
end
function _turing_group_names(plan::_TuringMultiResponsePlan)
    groups = Set{Symbol}()
    foreach(child -> union!(groups, _turing_group_names(child)), plan.plans)
    groups
end

_turing_centered_block_group_names(block::_BRMRandomEffectPlan) =
    block.centered ? Set((block.group,)) : Set{Symbol}()
_turing_centered_block_group_names(::_BRMMultiMembershipPlan) = Set{Symbol}()
function _turing_centered_random_effect_group_names(blocks)
    groups = Set{Symbol}()
    foreach(block -> union!(groups, _turing_centered_block_group_names(block)),
            blocks)
    groups
end
function _turing_centered_group_names(plan::_TuringGenericPlan)
    groups = Set{Symbol}()
    foreach(component -> union!(groups,
        _turing_centered_random_effect_group_names(component.random_effects)),
        plan.predictors)
    groups
end
function _turing_centered_group_names(plan::_TuringMultiResponsePlan)
    groups = Set{Symbol}()
    foreach(child -> union!(groups, _turing_centered_group_names(child)), plan.plans)
    groups
end

function _turing_collect_factor_schemas!(schemas, column)
    preprocess = column.preprocess
    isnothing(preprocess) && return schemas
    if preprocess.kind === :population_factor_dummy
        levels = collect(preprocess.const_.levels)
        existing = get(schemas, column.source, levels)
        existing == levels || error(
            "Turing backend: fitted categorical source `$(column.source)` " *
            "has inconsistent level schemas across predictors")
        schemas[column.source] = levels
    end
    foreach(dependency -> _turing_collect_factor_schemas!(schemas, dependency),
            preprocess.dependencies)
    schemas
end

function _turing_collect_factor_schemas!(schemas, design::_BRMPopulationDesign)
    foreach(column -> _turing_collect_factor_schemas!(schemas, column),
            design.columns)
    schemas
end
function _turing_collect_factor_schemas!(schemas, plan::_TuringGenericPlan)
    foreach(component -> _turing_collect_factor_schemas!(schemas,
        component.design), plan.predictors)
    schemas
end
function _turing_collect_factor_schemas!(schemas, plan::_TuringMultiResponsePlan)
    foreach(child -> _turing_collect_factor_schemas!(schemas, child), plan.plans)
    schemas
end

function _turing_replay_input(plan, new_data)
    schemas = _turing_collect_factor_schemas!(Dict{Symbol,Any}(), plan)
    isempty(schemas) && return new_data
    names = Tuple(propertynames(new_data))
    values = Tuple(getproperty(new_data, key) for key in names)
    prepared = NamedTuple{names}(values)
    for (source, levels) in schemas
        raw = _brm_df_column(new_data, source)
        raw_values = raw isa CA.CategoricalVector ? let raw_levels = CA.levels(raw)
            [raw_levels[code] for code in Int.(CA.levelcode.(raw))]
        end : collect(raw)
        unknown = unique(
            [value for value in raw_values if value ∉ levels])
        isempty(unknown) || error(
            "BRM replay: categorical predictor `$source` contains unseen " *
            "level(s) $(collect(unknown)); fitted levels are $levels")
        categorical = CA.categorical(raw_values; levels)
        prepared = merge(
            prepared, NamedTuple{(source,)}((categorical,)))
    end
    prepared
end

function _turing_direct_observations(brmi::BRMI)
    found = Any[]
    for (key, op_nc) in pairs(brmi.operations)
        op_nc isa NamedColumn || continue
        op = parent(op_nc)
        op isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(op, 2)
        isnothing(_brm_observation_name(lhs)) && continue
        push!(found, (; key, lhs, rhs))
    end
    isempty(found) && error(
        "Turing backend: direct execution requires at least one observed likelihood")
    Tuple(found)
end


function _turing_direct_observation(brmi::BRMI)
    found = _turing_direct_observations(brmi)
    length(found) == 1 || error(
        "Turing backend: requested one direct observation but found " *
        "$(length(found))")
    only(found)
end

_turing_collect_model_references!(_out, _x) = nothing
function _turing_collect_model_references!(out, x::NamedColumn)
    parent(x) isa DataColumn || push!(out, name(x))
    nothing
end
function _turing_collect_model_references!(out, x::ExprColumn)
    foreach(arg -> _turing_collect_model_references!(out, arg), getargs(x))
    foreach(value -> _turing_collect_model_references!(out, value),
            values(getkwargs(x)))
    nothing
end
function _turing_collect_model_references!(out, x::Union{Tuple,AbstractArray,NamedTuple})
    foreach(value -> _turing_collect_model_references!(out, value), x)
    nothing
end

function _turing_multi_model_operations(observations)
    out = Set{Symbol}(observation.key for observation in observations)
    foreach(observations) do observation
        _turing_collect_model_references!(out, observation.rhs)
    end
    Tuple(sort!(collect(out)))
end

function _turing_named_reference(x, role::AbstractString)
    x isa NamedColumn || error(
        "Turing backend: $role must be a direct named predictor; " *
        "got $(typeof(x))")
    name(x)
end

function _turing_with_ranef_prior(block::_BRMRandomEffectPlan,
                                  sd_prior, sd_family, sd_rate, lkj_eta)
    _BRMRandomEffectPlan(
        block.predictor, block.id, block.group, block.by, block.levels,
        block.strata, block.indices, block.stratum_indices, block.group_strata,
        block.columns, block.matrix, block.intercept_only,
        block.zero_correlation, block.centered, collect(sd_prior),
        collect(Int, sd_family),
        collect(Float64, sd_rate), Float64(lkj_eta))
end

function _turing_with_centering(block::_BRMRandomEffectPlan, centered::Bool)
    _BRMRandomEffectPlan(
        block.predictor, block.id, block.group, block.by, block.levels,
        block.strata, block.indices, block.stratum_indices, block.group_strata,
        block.columns, block.matrix, block.intercept_only,
        block.zero_correlation, centered, block.sd_prior, block.sd_family, block.sd_rate,
        block.lkj_eta)
end

function _turing_with_centering(block::_BRMMultiMembershipPlan,
                                centered::Bool)
    centered && error(
        "Turing backend: centered parameterization for `mm(...)` is not " *
        "supported; leave the shared multi-membership block non-centered")
    block
end

_turing_center_block(block::_BRMRandomEffectPlan, groups) =
    _turing_with_centering(block, block.group in groups)
function _turing_center_block(block::_BRMMultiMembershipPlan, groups)
    selected = intersect(Set(block.groups), groups)
    _turing_with_centering(block, !isempty(selected))
end

function _turing_center_component(component::_TuringPopulationComponent,
                                  groups::Set{Symbol})
    random_effects = Tuple(
        _turing_center_block(block, groups)
        for block in component.random_effects)
    _TuringPopulationComponent(
        component.predictor, component.design, component.beta_location,
        component.beta_scale, random_effects)
end
function _turing_center_component(component::_TuringGenericPredictor,
                                  groups::Set{Symbol})
    _TuringGenericPredictor(
        _turing_center_component(component.component, groups),
        component.priors, component.terms, component.r2d2)
end

function _turing_center_groups(plan::_TuringGenericPlan,
                               groups::Set{Symbol})
    predictors = Tuple(_turing_center_component(component, groups)
                       for component in plan.predictors)
    _TuringGenericPlan(
        plan.context, plan.prepared, predictors, plan.parameters,
        plan.assignments, plan.distribution,
        plan.response, plan.response_name, plan.lhs, plan.missing_response,
        plan.response_modifier, plan.observation_weight,
        plan.response_fit, plan.joint_r2d2, plan.source_ast)
end

_turing_center_groups(plan::_TuringMultiResponsePlan, groups::Set{Symbol}) =
    _TuringMultiResponsePlan(
        plan.responses,
        Tuple(_turing_center_groups(child, groups) for child in plan.plans),
        plan.owners, plan.joint_r2d2)

function _turing_apply_ranef_effect_priors(brmi::BRMI, components::Tuple)
    specs = ranef_effect_priors(brmi)
    isempty(specs) && return components
    any(component -> any(block -> !isnothing(block.by),
                         component.random_effects), components) && error(
        "Turing backend: random-effect `sd`/`cor` overrides for stratified " *
        "`gr(group; by=...)` blocks are not yet supported")
    margins = Dict{Tuple{Symbol,Symbol},Vector{NamedTuple}}()
    locations = Dict{Tuple{Int,Int},Tuple{Tuple{Symbol,Symbol},UnitRange{Int}}}()
    for (component_index, component) in enumerate(components)
        for (block_index, block) in enumerate(component.random_effects)
            isnothing(block.id) && continue
            key = (block.id, block.group)
            axis = get!(margins, key, NamedTuple[])
            first_index = length(axis) + 1
            append!(axis, ((; predictor=block.predictor,
                             coefficient=column.label)
                           for column in block.columns))
            locations[(component_index, block_index)] =
                (key, first_index:length(axis))
        end
    end
    resolved = _brm_resolve_ranef_effect_overrides(
        specs, margins; prefix="Turing backend")
    Tuple(map(enumerate(components)) do (component_index, component)
        blocks = Tuple(map(enumerate(component.random_effects)) do (block_index, block)
            location = get(locations, (component_index, block_index), nothing)
            isnothing(location) && return block
            key, range = location
            override = get(resolved, key, nothing)
            isnothing(override) && return block
            _turing_with_ranef_prior(
                block, override.sd_prior[range], override.sd_family[range], override.sd_rate[range],
                override.lkj_eta)
        end)
        _TuringPopulationComponent(
            component.predictor, component.design, component.beta_location,
            component.beta_scale, blocks)
    end)
end

function _turing_materialize_response_modifier(
        response_modifier, observation, response, context;
        support_kind::Symbol=:continuous)
    isnothing(response_modifier) && return nothing
    response_modifier.kind === :interval_censored ?
        _brm_materialize_interval_response(
            response_modifier, observation.key, response, context.data;
            support_kind, prefix="Turing backend") :
        _brm_materialize_bounded_response(
            response_modifier, observation.key, response, context.data;
            support_kind, prefix="Turing backend")
end

function _turing_has_categorical_basis(column)
    preprocess = column.preprocess
    isnothing(preprocess) && return false
    preprocess.kind === :population_factor_dummy && return true
    preprocess.kind === :interaction || return false
    any(_turing_has_categorical_basis, preprocess.dependencies)
end

function _turing_predictor_component(brmi::BRMI, context::_BRMBackendContext,
                                     predictor::Symbol;
                                     available_predictors=(predictor,),
                                     allow_group_terms::Bool=false,
                                     allow_random_slopes::Bool=false,
                                     allow_zero_correlation::Bool=false)
    random_effects = _brm_simple_random_effect_plans(
        brmi, predictor, context; required=true)
    if !allow_group_terms && !isempty(random_effects)
        error("Turing backend: random effects for predictor `$predictor` are " *
              "not yet supported by this likelihood plan")
    end
    if !allow_random_slopes && any(!block.intercept_only for block in random_effects)
        error("Turing backend: random slopes for predictor `$predictor` are " *
              "not yet supported by this likelihood plan")
    end
    if !allow_zero_correlation && any(block.zero_correlation
                                      for block in random_effects)
        error("Turing backend: zero-correlation `||` random effects for " *
              "predictor `$predictor` are not yet supported by this likelihood plan")
    end
    if any(block.zero_correlation &&
           any(_turing_has_categorical_basis, block.columns)
           for block in random_effects)
        error("Turing backend: categorical random slopes inside a " *
              "zero-correlation `||` block are not yet parity-safe; " *
              "use correlated `|` random effects")
    end
    predictor_plan = _brm_simple_population_predictor(
        brmi, predictor, context; required=true)
    design = predictor_plan.design
    k = size(design.matrix, 2)
    overrides = _brm_simple_population_effect_overrides(
        brmi, design; prefix="Turing backend", available_predictors)
    beta_location, beta_scale = _brm_materialize_normal_effect_priors(
        overrides, k; prefix="Turing backend")
    _TuringPopulationComponent(
        predictor_plan, design, beta_location, beta_scale, random_effects)
end

function _turing_generic_predictor_component(
        brmi::BRMI, context::_BRMBackendContext, predictor::Symbol;
        available_predictors=(predictor,), training=nothing)
    random_effects = _brm_simple_random_effect_plans(
        brmi, predictor, context; required=true)
    op = linear_predictor_op(brmi, predictor)
    lhs, rhs = getargs(op, 2)
    link_lhs_fn, name = _peel_lp_lhs(lhs)
    raw_terms = _brm_additive_terms(rhs)
    structured_terms = if isnothing(training)
        Tuple(term for term in raw_terms if _brm_prepares_term(term))
    else
        Tuple(term for term in raw_terms if term isa ExprColumn &&
            any(old -> old.callable === getf(term), training.terms))
    end
    prepared_terms = if isnothing(training)
        Tuple(_brm_prepare_term(term, predictor, context)
              for term in structured_terms)
    else
        length(structured_terms) == length(training.terms) || error(
            "Turing replay: prepared term set changed for `$predictor`")
        Tuple(_brm_replay_term(old, fresh, context)
              for (old, fresh) in zip(training.terms, structured_terms))
    end
    ordinary_terms = Tuple(term for term in raw_terms
                           if !(term in structured_terms))
    row_source = isempty(prepared_terms) ? nothing : begin
        source = first(prepared_terms).source
        candidate = source isa Tuple ? first(source) : source
        haskey(context.data, candidate) ? candidate :
            get(context.target_obs, name, nothing)
    end
    design = _brm_population_design(
        name, ordinary_terms, context.data, get(context.target_obs, name, nothing);
        required=true, row_source)
    predictor_plan = _BRMPopulationPredictor(
        name, link_lhs_fn, _brm_lp_emitted_name(name, link_lhs_fn), design)
    priors = _brm_simple_population_effect_overrides(
        brmi, design; prefix="Turing backend", available_predictors)
    isnothing(priors) && (priors = Any[nothing for _ in design.columns])
    defaults = zeros(Float64, length(priors)), ones(Float64, length(priors))
    component = _TuringPopulationComponent(
        predictor_plan, design, defaults..., random_effects)
    r2plan = _brm_whole_predictor_r2d2(
        brmi, design, priors; prefix="Turing backend", available_predictors)
    _TuringGenericPredictor(component, Tuple(priors), prepared_terms, r2plan)
end

function _brm_turing_single_plan(brmi::BRMI, observation;
                                 additional_model_operations=(), training=nothing)
    missing_response = _brm_missing_response_plan(
        observation.lhs; prefix="Turing backend")
    (observation.lhs isa Union{NamedColumn,JointResponseColumn} || !isnothing(missing_response)) || error(
        "Turing backend: response decorators other than `mi(response)` and " *
        "response links are not yet supported")
    rhs = observation.rhs
    rhs isa ExprColumn || error(
        "Turing backend: observed likelihood must be a distribution call")
    raw_response = observation.lhs isa JointResponseColumn ?
        _brm_joint_response_values(observation.lhs; prefix="Turing backend") :
        isnothing(missing_response) ? _brm_data_vec(
            observation.key, parent(parent(observation.lhs))) :
            missing_response.values
    raw_response = _brm_observation_rows(raw_response, _brm_distribution_shape(rhs))
    observation_weight = _brm_observation_weight_plan(
        rhs, observation.key, raw_response; prefix="Turing backend")
    if !isnothing(observation_weight)
        rhs = observation_weight.distribution
    end
    response_modifier = _brm_response_modifier_plan(rhs; prefix="Turing backend")
    (isnothing(missing_response) || isnothing(observation_weight)) || error(
        "Turing backend: `mi(response)` cannot yet be composed with " *
        "observation weights")
    (isnothing(missing_response) || isnothing(response_modifier)) || error(
        "Turing backend: `mi(response)` cannot yet be composed with response " *
        "modifiers")
    (isnothing(observation_weight) || isnothing(response_modifier) ||
     observation_weight.kind !== :analytic) || error(
        "Turing backend: analytic/precision weights cannot yet be composed " *
        "with response modifiers; use frequency/power objective weights or " *
        "an unmodified Normal observation")
    if !isnothing(response_modifier)
        response_modifier.kind in (:truncated, :censored, :interval_censored) || error(
            "Turing backend: response modifier `$(response_modifier.kind)` is " *
            "not executable because no lowering is registered for this " *
            "response modifier")
        rhs = response_modifier.base
        rhs isa ExprColumn || error(
            "Turing backend: bounded response base must be a distribution call")
    end
    prepared_response = _brm_prepare_response(
        observation.key, rhs, raw_response;
        training=isnothing(training) ? nothing : training.response_fit)
    rhs = prepared_response.distribution
    raw_response = prepared_response.response
    program = _brm_prepare_program(
        brmi; context=_brm_backend_context(brmi; retain_mm_sources=true))
    context = program.context
    context.data[observation.key] = raw_response
    if !isnothing(response_modifier)
        support_kind = nonmissingtype(eltype(raw_response)) <: Integer ?
            :discrete : :continuous
        response_modifier = _turing_materialize_response_modifier(
            response_modifier, observation, raw_response, context;
            support_kind)
    end
    referenced = Set{Symbol}()
    _turing_collect_model_references!(referenced, rhs)
    operation_by_name = Dict(operation.name => operation
                             for operation in program.operations)
    changed = true
    while changed
        changed = false
        for target in collect(referenced)
            operation = get(operation_by_name, target, nothing)
            isnothing(operation) && continue
            before = length(referenced)
            union!(referenced, operation.dependencies)
            changed |= length(referenced) != before
        end
    end
    predictor_names = Symbol[]
    parameter_names = Symbol[first(pair) for pair in prepared_response.parameters]
    priors = ExprColumn[last(pair) for pair in prepared_response.parameters]
    raw_assignments = Pair{Symbol,Any}[]
    for target in program.order
        target in keys(context.data) && continue
        prepared_operation = get(operation_by_name, target, nothing)
        standalone_parameter = !isnothing(prepared_operation) &&
            prepared_operation.role === :parameter &&
            first(getargs(prepared_operation.expression)) isa NamedColumn
        (target in referenced || standalone_parameter) || continue
        isnothing(prepared_operation) && error(
            "Turing backend: likelihood references `$target`, which has no declaration")
        operation = prepared_operation.expression
        if prepared_operation.role === :assignment
            _brm_is_prior_declaration(brmi, target) && continue
            push!(raw_assignments, target => last(getargs(operation)))
            continue
        end
        operation isa ExprColumn{typeof(~)} || error(
            "Turing backend: dependency `$target` is not an executable declaration")
        lhs, declaration = getargs(operation, 2)
        peeled = _peel_lp_lhs(lhs)
        has_structured_term = !isnothing(peeled) &&
            any(_brm_prepares_term, _brm_additive_terms(declaration))
        population = if !isnothing(peeled) && last(peeled) === target
            try
                _brm_simple_population_predictor(
                    brmi, target, context; required=true)
            catch
                nothing
            end
        end
        if !isnothing(peeled) && last(peeled) === target &&
           (has_structured_term || !isnothing(population))
            push!(predictor_names, target)
        elseif lhs isa NamedColumn && name(lhs) === target && declaration isa ExprColumn
            push!(parameter_names, target)
            push!(priors, declaration)
        else
            error("Turing backend: declaration for `$target` cannot be lowered " *
                  "as a population predictor or scalar prior")
        end
    end
    training_by_name = isnothing(training) ? Dict{Symbol,Any}() :
        Dict(component.predictor.name => component for component in training.predictors)
    components = Tuple(_turing_generic_predictor_component(
        brmi, context, target; available_predictors=Tuple(predictor_names),
        training=get(training_by_name, target, nothing)) for target in predictor_names)
    raw_components = _turing_apply_ranef_effect_priors(
        brmi, Tuple(component.component for component in components))
    components = Tuple(_TuringGenericPredictor(raw, old.priors, old.terms, old.r2d2)
                       for (raw, old) in zip(raw_components, components))
    claims = Pair{Symbol,Tuple}[]
    for component in components
        expressions = Any[prior for prior in component.priors if !isnothing(prior)]
        for block in component.random_effects
            append!(expressions, (prior for prior in block.sd_prior
                                  if !isnothing(prior)))
        end
        for term in component.terms
            append!(expressions, values(_brm_term_prior_expressions(term)))
        end
        push!(claims, component.predictor.name => Tuple(expressions))
    end
    program = _brm_with_prior_dependencies(program, claims)
    operation_by_name = Dict(operation.name => operation
                             for operation in program.operations)
    referenced = Set{Symbol}()
    _turing_collect_model_references!(referenced, rhs)
    changed = true
    while changed
        changed = false
        for target in collect(referenced)
            operation = get(operation_by_name, target, nothing)
            isnothing(operation) && continue
            before = length(referenced)
            union!(referenced, operation.dependencies)
            changed |= length(referenced) != before
        end
    end
    for target in program.order
        target in referenced || continue
        target in parameter_names && continue
        target in predictor_names && continue
        any(pair -> first(pair) === target, raw_assignments) && continue
        operation = get(operation_by_name, target, nothing)
        isnothing(operation) && continue
        operation.role === :parameter || continue
        lhs, prior = getargs(operation.expression, 2)
        lhs isa NamedColumn || continue
        push!(parameter_names, target)
        push!(priors, prior)
    end
    n = length(raw_response)
    all(component -> size(component.design.matrix, 1) == n, components) ||
        error("Turing backend: response and predictor row counts differ")
    axes = Dict{Symbol,Symbol}(key => :observation for key in keys(context.data))
    foreach(key -> axes[key] = :observation, predictor_names)
    foreach(zip(parameter_names, priors)) do (key, prior)
        axes[key] = _brm_parameter_reference_axis(prior)
    end
    prior_axes = copy(axes)
    foreach(key -> prior_axes[key] = :whole, keys(context.data))
    prepared_rhs = _brm_prepare_expr(rhs, axes)
    parameters = Tuple(_BRMPreparedParameter(
        name, _brm_prepare_expr(prior, prior_axes), :distribution, true)
        for (name, prior) in zip(parameter_names, priors))
    prepared_assignments = Any[]
    for (assignment_name, expression) in raw_assignments
        prepared_expression = _brm_prepare_expr(expression, axes)
        refs = _brm_prepared_references(prepared_expression)
        axis = any(ref -> get(axes, ref, :scalar) === :observation, refs) ?
               :observation : :scalar
        axes[assignment_name] = axis
        push!(prepared_assignments, _BRMPreparedAssignment(
            assignment_name, prepared_expression, refs))
    end
    # Reprepare the observation after assignment axes have been inferred.
    prepared_rhs = _brm_prepare_expr(rhs, axes)
    assignments = Tuple(prepared_assignments)
    source_ast = (; operations=brmi.operations, observation=prepared_rhs)
    prepared_model = _brm_prepare_model(brmi; program)
    joint_r2d2 = _brm_joint_r2d2_plans(brmi, components; prefix="Turing backend")
    _TuringGenericPlan(
        context, prepared_model, components, parameters, assignments,
        prepared_rhs, raw_response,
        observation.key, observation.lhs, missing_response, response_modifier,
        observation_weight, prepared_response.fit, joint_r2d2, source_ast)
end

_turing_parameter_sources(plan::_TuringGenericPlan) =
    (map(component -> component.predictor.name, plan.predictors)...,
     map(parameter -> parameter.name, plan.parameters)...)

function _turing_same_random_effect(left::_BRMRandomEffectPlan,
                                    right::_BRMRandomEffectPlan)
    left.predictor === right.predictor && left.id === right.id &&
    left.group === right.group && left.by === right.by &&
    left.levels == right.levels && left.indices == right.indices &&
    left.strata == right.strata &&
    left.stratum_indices == right.stratum_indices &&
    left.group_strata == right.group_strata && left.matrix == right.matrix &&
    left.intercept_only == right.intercept_only &&
    left.zero_correlation == right.zero_correlation &&
    left.centered == right.centered && isequal(left.sd_prior, right.sd_prior) &&
    left.lkj_eta == right.lkj_eta
end

function _turing_same_random_effect(left::_BRMMultiMembershipPlan,
                                    right::_BRMMultiMembershipPlan)
    _turing_same_random_effect_identity(left, right) &&
    left.levels == right.levels && left.indices == right.indices &&
    left.weights == right.weights && left.n_obs == right.n_obs &&
    left.n_memberships == right.n_memberships &&
    left.matrix == right.matrix &&
    left.intercept_only == right.intercept_only &&
    left.zero_correlation == right.zero_correlation &&
    left.centered == right.centered && isequal(left.sd_prior, right.sd_prior) &&
    left.lkj_eta == right.lkj_eta
end

_turing_same_random_effect(_left, _right) = false

function _turing_same_random_effects(left, right)
    length(left) == length(right) || return false
    all(pair -> _turing_same_random_effect(pair...), zip(left, right))
end

function _turing_same_component(left::_TuringPopulationComponent,
                                right::_TuringPopulationComponent)
    left.predictor.name === right.predictor.name &&
    left.predictor.link_lhs_fn === right.predictor.link_lhs_fn &&
    left.design.matrix == right.design.matrix &&
    left.design.fixed == right.design.fixed &&
    left.beta_location == right.beta_location &&
    left.beta_scale == right.beta_scale &&
    _turing_same_random_effects(left.random_effects, right.random_effects)
end

function _turing_shared_plans_compatible(left::_TuringGenericPlan,
                                          right::_TuringGenericPlan)
    _turing_parameter_sources(left) == _turing_parameter_sources(right) &&
    length(left.predictors) == length(right.predictors) &&
    all(pair -> _turing_same_component(pair[1].component, pair[2].component),
        zip(left.predictors, right.predictors))
end

function _turing_response_owners(responses, plans)
    owners = Int[]
    source_sets = [Set(_turing_parameter_sources(plan)) for plan in plans]
    for i in eachindex(plans)
        owner = i
        for j in 1:(i - 1)
            overlap = intersect(source_sets[i], source_sets[j])
            isempty(overlap) && continue
            source_sets[i] == source_sets[j] || continue
            _turing_shared_plans_compatible(plans[j], plans[i]) || continue
            owner = owners[j]
            break
        end
        push!(owners, owner)
    end
    Tuple(owners)
end

function _brm_turing_plan(brmi::BRMI; training=nothing)
    observations = _turing_direct_observations(brmi)
    length(observations) == 1 &&
        return _brm_turing_single_plan(brmi, only(observations); training)

    response_names = Tuple(observation.key for observation in observations)
    length(unique(response_names)) == length(response_names) || error(
        "Turing backend: multi-response observation names must be unique")
    model_operations = _turing_multi_model_operations(observations)
    training_by_response = isnothing(training) ? Dict{Symbol,Any}() :
        Dict(name => child for (name, child) in
             zip(training.responses, training.plans))
    plans = Tuple(_brm_turing_single_plan(
        brmi, observation; additional_model_operations=model_operations,
        training=get(training_by_response, observation.key, nothing))
        for observation in observations)
    owners = _turing_response_owners(response_names, plans)
    joint_r2d2 = _brm_merge_joint_r2d2(plans, owners; prefix="Turing backend")
    _TuringMultiResponsePlan(response_names, plans, owners, joint_r2d2)
end
