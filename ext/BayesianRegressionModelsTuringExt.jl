module BayesianRegressionModelsTuringExt

using BayesianRegressionModels
using Distributions: ContinuousMultivariateDistribution,
                     ContinuousUnivariateDistribution,
                     DiscreteUnivariateDistribution, Exponential, LKJCholesky,
                     MvNormal, Normal, Poisson, censored, logcdf,
                     mean, product_distribution, std, truncated
using LinearAlgebra: Diagonal, Symmetric, cholesky, dot
using LogExpFunctions: logistic, log1mexp
using InverseFunctions
import Distributions: logpdf
import Random: AbstractRNG, default_rng, rand
using Turing
import Distributions
import RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

const BRM = BayesianRegressionModels

include("turing_terms.jl")
include("turing_shapes.jl")
include("turing_r2d2.jl")

BRM._brm_turing_term_model(term, nobs, priors, _inputs) =
    BRM._brm_turing_term_model(term, nobs, priors)
_brm_turing_term_call_ast(_term, term, nobs, priors, inputs) =
    :(BRM.turing_term_model($term, $nobs, $priors, $inputs))
BRM.turing_term_model(term, nobs, priors, inputs) =
    BRM._brm_turing_term_model(term, nobs, priors, inputs)
BRM.turing_group_effect(block, sd_priors, residual_scale) =
    _brm_group_effect_model(block, sd_priors, residual_scale)
include("turing_gp.jl")
include("turing_structured.jl")

struct _BRMGroupedMvNormal{D<:ContinuousMultivariateDistribution} <:
       ContinuousMultivariateDistribution
    base::D
    n_groups::Int
end

Base.length(distribution::_BRMGroupedMvNormal) =
    length(distribution.base) * distribution.n_groups

function logpdf(distribution::_BRMGroupedMvNormal, values::AbstractVector)
    length(values) == length(distribution) || return oftype(sum(values), -Inf)
    n_terms = length(distribution.base)
    sum(1:distribution.n_groups) do group
        first_index = (group - 1) * n_terms + 1
        logpdf(distribution.base,
               @view values[first_index:(first_index + n_terms - 1)])
    end
end

rand(rng::AbstractRNG, distribution::_BRMGroupedMvNormal) =
    vcat((rand(rng, distribution.base)
          for _ in 1:distribution.n_groups)...)

struct _BRMStratifiedMvNormal{B,G<:AbstractVector{Int}} <:
       ContinuousMultivariateDistribution
    bases::B
    group_strata::G
end


Base.length(distribution::_BRMStratifiedMvNormal) =
    length(first(distribution.bases)) * length(distribution.group_strata)

function logpdf(distribution::_BRMStratifiedMvNormal,
                values::AbstractVector)
    length(values) == length(distribution) || return oftype(sum(values), -Inf)
    n_terms = length(first(distribution.bases))
    sum(eachindex(distribution.group_strata)) do group
        first_index = (group - 1) * n_terms + 1
        logpdf(distribution.bases[distribution.group_strata[group]],
               @view values[first_index:(first_index + n_terms - 1)])
    end
end

rand(rng::AbstractRNG, distribution::_BRMStratifiedMvNormal) =
    vcat((rand(rng, distribution.bases[stratum])
          for stratum in distribution.group_strata)...)

struct _BRMContinuousIntervalEvidence{D,U} <:
       ContinuousUnivariateDistribution
    base::D
    upper::U
end

struct _BRMDiscreteIntervalEvidence{D,U} <:
       DiscreteUnivariateDistribution
    base::D
    upper::U
end

struct _BRMObjectiveWeight{VF<:Distributions.VariateForm,
                           VS<:Distributions.ValueSupport,
                           D<:Distributions.Distribution{VF,VS},W} <:
       Distributions.Distribution{VF,VS}
    base::D
    weight::W
end
_BRMObjectiveWeight(base::Distributions.Distribution{VF,VS}, weight::W) where
    {VF,VS,W} = _BRMObjectiveWeight{VF,VS,typeof(base),W}(base, weight)


function _brm_interval_logmass(base, lower, upper)
    upper_logcdf = logcdf(base, upper)
    lower_logcdf = logcdf(base, lower)
    lower_logcdf < upper_logcdf || return oftype(upper_logcdf, -Inf)
    upper_logcdf + log1mexp(lower_logcdf - upper_logcdf)
end

logpdf(d::_BRMContinuousIntervalEvidence, lower::Real) =
    _brm_interval_logmass(d.base, lower, d.upper)
logpdf(d::_BRMDiscreteIntervalEvidence, lower::Real) =
    _brm_interval_logmass(d.base, lower, d.upper)
rand(rng::AbstractRNG, d::_BRMContinuousIntervalEvidence) = rand(rng, d.base)
rand(rng::AbstractRNG, d::_BRMDiscreteIntervalEvidence) = rand(rng, d.base)

Base.length(d::_BRMObjectiveWeight) = length(d.base)
Base.size(d::_BRMObjectiveWeight) = size(d.base)
logpdf(d::_BRMObjectiveWeight{Distributions.Univariate}, x::Real) =
    d.weight * logpdf(d.base, x)
Distributions._logpdf(d::_BRMObjectiveWeight{Distributions.ArrayLikeVariate{N}},
                      x::AbstractArray{<:Real,N}) where {N} =
    d.weight * logpdf(d.base, x)
rand(rng::AbstractRNG, d::_BRMObjectiveWeight) = rand(rng, d.base)

_brm_interval_evidence(base::ContinuousUnivariateDistribution, upper) =
    _BRMContinuousIntervalEvidence(base, upper)
_brm_interval_evidence(base::DiscreteUnivariateDistribution, upper) =
    _BRMDiscreteIntervalEvidence(base, upper)

include("turing_observations.jl")
include("turing_model_inputs.jl")

function _brm_callable_ast(callable, callables)
    if applicable(parentmodule, callable) && applicable(nameof, callable)
        module_ = parentmodule(callable)
        name = nameof(callable)
        if isdefined(module_, name) && getfield(module_, name) === callable
            return GlobalRef(module_, name)
        end
    end
    push!(callables, callable)
    :(callables[$(length(callables))])
end

function _brm_ast_call(callable, args, kwargs, callables)
    f = _brm_callable_ast(callable, callables)
    positional = map(x -> _brm_prepared_ast(x, callables), args)
    keywords = [Expr(:kw, key, _brm_prepared_ast(value, callables))
                for (key, value) in pairs(kwargs)]
    isempty(keywords) ? Expr(:call, f, positional...) :
        Expr(:call, f, Expr(:parameters, keywords...), positional...)
end

_brm_prepared_ast(x, _callables) = QuoteNode(x)
_brm_prepared_ast(x::Union{Number,AbstractString,Char}, _callables) = x
function _brm_prepared_ast(x::BRM._BRMPreparedRef, callables)
    value = _brm_reference_ast(x.name, callables)
    row = _brm_row_symbol(callables)
    x.axis === :observation_row ? :(view($value, $row, :)) :
        x.axis === :observation ? :($value[$row]) : value
end
_brm_prepared_ast(x::BRM._BRMPreparedExpr, callables) =
    _brm_ast_call(x.callable, x.args, x.kwargs, callables)
function _brm_prepared_ast(x::Tuple, callables)
    Expr(:tuple, map(value -> _brm_prepared_ast(value, callables), x)...)
end

const _brm_has_row_ref = BRM._brm_has_row_ref

# A callable with no Julia methods cannot execute in the emitted Turing
# program. In practice this is a Stan-only `@deffun` function or a `@lpxf`
# family base stub (`function f end` with no methods); the Turing backend runs
# ordinary Julia, so emitting a call to one only moves the failure from
# construction time to sampling time (or out-of-bounds row indexing before
# it). Walk a prepared expression tree and return the first such callee, or
# `nothing` when every call resolves to real Julia code.
_brm_turing_stan_only(x) = nothing
_brm_turing_stan_only(x::BRM._BRMPreparedExpr) =
    _brm_turing_stan_only_callable(x.callable) !== nothing ?
        _brm_turing_stan_only_callable(x.callable) :
        _brm_turing_stan_only_args(x.args, x.kwargs)
_brm_turing_stan_only_callable(callable::Function) =
    isempty(methods(callable)) ? callable : nothing
_brm_turing_stan_only_callable(_callable) = nothing
function _brm_turing_stan_only_args(args, kwargs)
    for arg in args
        found = _brm_turing_stan_only(arg)
        isnothing(found) || return found
    end
    for value in values(kwargs)
        found = _brm_turing_stan_only(value)
        isnothing(found) || return found
    end
    nothing
end
_brm_turing_stan_only(x::Tuple) = _brm_turing_stan_only_args(x, (;))

function _brm_group_prior_ast(block, callables)
    Expr(:tuple, map(block.sd_prior) do prior
        if isnothing(prior)
            nothing
        else
            prepared = BRM._brm_prepare_prior_expr(prior)
            prepared isa BRM._BRMPreparedExpr ?
                _brm_turing_prior_ast(prepared, callables) :
                _brm_prepared_ast(prepared, callables)
        end
    end...)
end
function _brm_prior_value_ast(value, callables)
    prepared = BRM._brm_prepare_prior_expr(value)
    prepared isa BRM._BRMPreparedExpr ?
        _brm_turing_prior_ast(prepared, callables) :
        _brm_prepared_ast(prepared, callables)
end
_brm_prior_value_ast(values::Tuple, callables) =
    Expr(:tuple, map(value -> _brm_prior_value_ast(value, callables), values)...)
function _brm_term_priors_ast(term, callables)
    expressions = BRM._brm_term_prior_expressions(term)
    names = keys(expressions)
    values = Expr(:tuple, map(value -> _brm_prior_value_ast(value, callables),
                             Base.values(expressions))...)
    :(NamedTuple{$(QuoteNode(names))}($values))
end
function _brm_term_inputs_ast(term, callables)
    names = Tuple(term.dependencies)
    values = map(name -> _brm_reference_ast(name, callables), names)
    :(NamedTuple{$(QuoteNode(names))}($(Expr(:tuple, values...))))
end

function _brm_generic_model_ast(plan::BRM._TuringGenericPlan)
    graph = (; plans=(plan,), joint_r2d2=plan.joint_r2d2)
    _brm_generic_response_graph_ast(graph; single=true)
end

const _BRM_GENERIC_MODEL_CACHE = Dict{Any,Tuple{Function,Expr}}()
const _BRM_GENERIC_MODEL_CACHE_LOCK = ReentrantLock()

function _brm_cached_generic_evaluator(lowered)
    key = _brm_generic_structure_key(lowered.definition)
    lock(_BRM_GENERIC_MODEL_CACHE_LOCK) do
        get!(_BRM_GENERIC_MODEL_CACHE, key) do
            (_brm_staged_turing_evaluator(lowered.definition), lowered.definition)
        end
    end
end

# Use DynamicPPL's own compiler for tilde/context semantics, then stage its
# evaluator without adding a new Julia method at runtime. Ordinary compiled
# callers and AD can call this body immediately, without a hot-loop world-age
# barrier. Retain the original @model AST for introspection.
function _brm_staged_turing_evaluator(definition)
    compiled = Turing.DynamicPPL.model(
        @__MODULE__, LineNumberNode(0), deepcopy(last(definition.args)), false)
    evaluator = only(node for node in compiled.args if Meta.isexpr(node, :function))
    RuntimeGeneratedFunctions.drop_expr(
        RuntimeGeneratedFunctions.RuntimeGeneratedFunction(
            @__MODULE__, @__MODULE__, evaluator))
end

function _brm_generic_response_graph_ast(multi; single::Bool=false)
    row = _brm_fresh_model_name(:i, _brm_model_binding_names(multi.plans))
    body = Expr(:block)
    node_statements = Dict{Symbol,Vector{Any}}()
    parameters = Dict{Symbol,Any}()
    predictors = Dict{Symbol,Tuple{Int,Int,Any}}()
    assignments = Dict{Symbol,Tuple{Int,Any}}()
    data_sources = Dict{Symbol,Int}()
    response_symbols = _brm_response_symbols(multi.plans; single)
    response_names = Set(plan.response_name for plan in multi.plans)
    for (pi, plan) in enumerate(multi.plans)
        foreach(key -> get!(data_sources, key, pi), keys(plan.context.data))
        foreach(p -> get!(parameters, p.name, p), plan.parameters)
        foreach(a -> get!(assignments, a.name, (pi, a)), plan.assignments)
        for (ci, component) in enumerate(plan.predictors)
            get!(predictors, component.predictor.name, (pi, ci, component))
        end
    end
    for name in union(response_names, keys(parameters), keys(predictors), keys(assignments))
        delete!(data_sources, name)
    end
    callables = _BRMTuringASTContext(Any[], row, data_sources)
    predictor_entries = Tuple(values(predictors))
    shared_groups = BRM._turing_shared_group_plans(
        Tuple(entry[3] for entry in predictor_entries))
    shared_members = Set((member.predictor, member.block_index)
        for shared in shared_groups for member in shared.members)
    shared_predictors = Set(member.predictor
        for shared in shared_groups for member in shared.members)
    predictor_bases = Dict(name => gensym(Symbol(:predictor_base_, name))
                          for name in shared_predictors)
    shared_barriers = [gensym(:shared_group) for _ in shared_groups]
    predictor_shared_nodes = Dict(name => Symbol[] for name in shared_predictors)
    residual_scales = Dict{Symbol,Any}()
    block_residual_scales = Dict{Tuple{Symbol,Int},Any}()
    joint_mappings = Dict{Symbol,Tuple{Int,Int,Any}}()
    for (joint_index, joint) in enumerate(multi.joint_r2d2)
        for (mapping_index, mapping) in enumerate(joint.predictors)
            joint_mappings[mapping.predictor] = (joint_index, mapping_index, joint)
        end
    end
    for parameter in values(parameters)
        prior = _brm_turing_parameter_ast(parameter, callables)
        node_statements[parameter.name] = Any[:($(parameter.name) ~ $prior)]
    end
    for (pi, assignment) in values(assignments)
        if _brm_has_row_ref(assignment.expression)
            stan_only = _brm_turing_stan_only(assignment.expression)
            isnothing(stan_only) || error(
                "Turing backend: assignment `$(assignment.name)` calls " *
                "`$(nameof(stan_only))`, which defines no Julia methods " *
                "(a Stan `@deffun` function has no Julia implementation). " *
                "Row-dependent assignments lower to row-wise Julia " *
                "comprehensions, so this call cannot execute. Express the " *
                "computation with row-wise Julia code or fit this model " *
                "with the Stan backend.")
        end
        value = _brm_prepared_ast(assignment.expression, callables)
        statement = if _brm_has_row_ref(assignment.expression)
            :($(assignment.name) = [$value for $row in eachindex($(response_symbols[pi]))])
        else
            :($(assignment.name) = $value)
        end
        node_statements[assignment.name] = Any[statement]
    end
    logical = Symbol[]
    for (name, (pi, ci, component)) in predictors
        beta = single ? (ci == 1 ? :beta_pop : Symbol(:beta_pop_, name)) :
               length(predictors) == 1 ? :beta_pop : Symbol(:beta_pop_, name)
        eta = Symbol(:eta_, name)
        push!(logical, name)
        prior_asts = map(component.priors) do prior
            if isnothing(prior)
                :($(_brm_callable_ast(Normal, callables))(0, 1))
            else
                _brm_turing_prior_ast(BRM._brm_prepare_expr(prior), callables)
            end
        end
        statements = get!(node_statements, name, Any[])
        prior_vector = Expr(:vect, prior_asts...)
        joint_mapping = get(joint_mappings, name, nothing)
        if !isnothing(joint_mapping)
            joint_index, mapping_index, joint = joint_mapping
            site = Symbol(:r2d2_joint_, joint_index)
            if mapping_index == 1
                design_exprs = Any[]
                coefficient_shares = Any[]
                margin_shares = Any[]
                fallback_exprs = Any[]
                for mapping in joint.predictors
                    mapped_plan_index = single ? 1 : mapping.plan_index
                    mapped = multi.plans[mapped_plan_index].predictors[
                        mapping.component_index]
                    push!(design_exprs, :(multi.plans[$mapped_plan_index].predictors[
                        $(mapping.component_index)].design.matrix))
                    push!(coefficient_shares, QuoteNode(mapping.coefficient_shares))
                    push!(margin_shares, QuoteNode(mapping.margin_shares))
                    mapped_priors = map(mapped.priors) do prior
                        isnothing(prior) ? begin
                            :($(_brm_callable_ast(Normal, callables))())
                        end : _brm_turing_prior_ast(
                            BRM._brm_prepare_expr(prior), callables)
                    end
                    push!(fallback_exprs, Expr(:vect, mapped_priors...))
                end
                r2_prior = _brm_turing_prior_ast(
                    BRM._brm_prepare_expr(joint.prior), callables)
                reference = isnothing(joint.reference_scale) ? nothing :
                    _brm_prepared_ast(
                        BRM._brm_prepare_expr(joint.reference_scale), callables)
                push!(statements, :($site ~ to_submodel(_brm_r2d2_joint(
                    $(Expr(:tuple, design_exprs...)),
                    $(Expr(:tuple, coefficient_shares...)),
                    $(Expr(:tuple, margin_shares...)), $r2_prior,
                    $(joint.alpha), $reference,
                    $(Expr(:tuple, fallback_exprs...))))))
            end
            push!(statements, :($beta = $site.betas[$mapping_index]))
            mapping = joint.predictors[mapping_index]
            block_residual_scales[(name, mapping.block_index)] =
                :($site.scales[$mapping_index])
            residual_scale = nothing
        elseif isempty(component.priors)
            residual_scale = nothing
        elseif isnothing(component.r2d2)
            priors = all(isnothing, component.priors) ?
                :(fill(Normal(), $(length(component.priors)))) : prior_vector
            push!(statements, :($beta ~ product_distribution($priors)))
            residual_scale = nothing
        else
            r2d2 = component.r2d2
            r2_prior = _brm_turing_prior_ast(
                BRM._brm_prepare_expr(r2d2.prior), callables)
            total_scale = isnothing(r2d2.total_scale) ? nothing :
                _brm_prepared_ast(BRM._brm_prepare_expr(r2d2.total_scale), callables)
            site = Symbol(:r2d2_, name)
            push!(statements, :($site ~ to_submodel(_brm_r2d2_population(
                multi.plans[$pi].predictors[$ci].design.matrix,
                $(QuoteNode(r2d2.share_indices)), $r2_prior,
                $(r2d2.alpha), $total_scale, $prior_vector))))
            push!(statements, :($beta = $site.beta))
            residual_scale = :($site.residual_scale)
        end
        residual_scales[name] = residual_scale
        has_offset = !all(iszero, component.design.fixed)
        term_only = isempty(component.priors) && !has_offset &&
            isempty(component.random_effects) && length(component.terms) == 1
        population = if isempty(component.priors)
            has_offset ? :(copy(multi.plans[$pi].predictors[$ci].design.fixed)) :
                :(zeros(length(multi.plans[$pi].response)))
        else
            product = :(multi.plans[$pi].predictors[$ci].design.matrix * $beta)
            has_offset ? :($product + multi.plans[$pi].predictors[$ci].design.fixed) : product
        end
        term_only || push!(statements, :($eta = $population))
        # Keep the established single-response summation order: saved draws
        # produce exactly the same predictor values for crossed group blocks.
        group_effect = single ? Symbol(:group_effect_, ci) : eta
        if single && !isempty(component.random_effects)
            push!(statements, :($group_effect = zeros(length(multi.plans[$pi].response))))
        end
        for (gi, _) in enumerate(component.random_effects)
            (name, gi) in shared_members && continue
            group = single ? Symbol(:group_, ci, :_, gi) :
                    Symbol(:group_, name, :_, gi)
            group_scale = get(block_residual_scales, (name, gi), residual_scale)
            priors = haskey(block_residual_scales, (name, gi)) ?
                Expr(:tuple, fill(nothing,
                    size(component.random_effects[gi].matrix, 2))...) :
                _brm_group_prior_ast(component.random_effects[gi], callables)
            push!(statements, :($group ~ to_submodel(BRM.turing_group_effect(
                multi.plans[$pi].predictors[$ci].random_effects[$gi], $priors,
                $group_scale))))
            push!(statements, :($group_effect = $group_effect + $group.effect))
        end
        if single && !isempty(component.random_effects)
            push!(statements, :($eta = $eta + $group_effect))
        end
        for term_index in eachindex(component.terms)
            term_site = Symbol(:term_, name, :_, term_index)
            term = component.terms[term_index]
            priors = _brm_term_priors_ast(term, callables)
            inputs = _brm_term_inputs_ast(term, callables)
            term_model = _brm_turing_term_call_ast(
                term,
                :(multi.plans[$pi].predictors[$ci].terms[$term_index]),
                :(length(multi.plans[$pi].response)),
                priors,
                inputs,
            )
            push!(statements,
                  :($term_site ~ to_submodel($term_model)))
            push!(statements, term_only ? :($eta = $term_site.effect) :
                :($eta = $eta + $term_site.effect))
        end
        inverse_link = InverseFunctions.inverse(component.predictor.link_lhs_fn)
        inverse_link_ast = _brm_callable_ast(inverse_link, callables)
        push!(statements, inverse_link === identity ? :($name = $eta) :
            :($name = $inverse_link_ast.($eta)))
        if name in shared_predictors
            finalizer = pop!(statements)
            base = predictor_bases[name]
            node_statements[base] = pop!(node_statements, name)
            node_statements[name] = Any[finalizer]
        end
    end
    for (shared_index, shared) in enumerate(shared_groups)
        site = Symbol(:shared_group_, shared_index)
        barrier = shared_barriers[shared_index]
        statements = get!(node_statements, barrier, Any[])
        block_exprs = Any[]
        prior_exprs = Any[]
        scale_exprs = Any[]
        for (member, block) in zip(shared.members, shared.blocks)
            pi, ci, _ = predictor_entries[member.component_index]
            push!(block_exprs, :(multi.plans[$pi].predictors[$ci].random_effects[
                $(member.block_index)]))
            scale_key = (member.predictor, member.block_index)
            if haskey(block_residual_scales, scale_key)
                append!(prior_exprs, fill(nothing, size(block.matrix, 2)))
            else
                append!(prior_exprs, map(block.sd_prior) do prior
                    isnothing(prior) ? nothing :
                        _brm_turing_prior_ast(BRM._brm_prepare_expr(prior), callables)
                end)
            end
            push!(scale_exprs, get(block_residual_scales,
                (member.predictor, member.block_index),
                residual_scales[member.predictor]))
        end
        push!(statements, :($site ~ to_submodel(_brm_shared_group_effect_model(
            $(Expr(:tuple, block_exprs...)), $(Expr(:tuple, prior_exprs...)),
            $(Expr(:tuple, scale_exprs...))))))
        for (effect_index, member) in enumerate(shared.members)
            eta = Symbol(:eta_, member.predictor)
            push!(statements, :($eta = $eta + $site.effects[$effect_index]))
            push!(predictor_shared_nodes[member.predictor], barrier)
        end
    end

    emitted = Set{Symbol}()
    if isempty(shared_groups)
        for plan in multi.plans, name in plan.prepared.order
            name in emitted && continue
            statements = get(node_statements, name, nothing)
            isnothing(statements) && continue
            append!(body.args, statements)
            push!(emitted, name)
        end
        for (name, statements) in node_statements
            name in emitted || append!(body.args, statements)
        end
    else
        operations = Dict{Symbol,Any}()
        ordered_names = Symbol[]
        for plan in multi.plans, operation in plan.prepared.program.operations
            haskey(operations, operation.name) && continue
            operations[operation.name] = operation
            push!(ordered_names, operation.name)
        end
        scheduled = Any[]
        for name in ordered_names
            operation = operations[name]
            if name in shared_predictors
                base = predictor_bases[name]
                push!(scheduled, BRM._BRMPreparedOperation(
                    base, :emitter, nothing, operation.dependencies))
                push!(scheduled, BRM._BRMPreparedOperation(
                    name, operation.role, operation.expression,
                    (base, predictor_shared_nodes[name]...)))
            else
                push!(scheduled, operation)
            end
        end
        for (shared_index, shared) in enumerate(shared_groups)
            barrier = shared_barriers[shared_index]
            dependencies = Tuple(predictor_bases[member.predictor]
                                 for member in shared.members)
            push!(scheduled, BRM._BRMPreparedOperation(
                barrier, :emitter, nothing, dependencies))
        end
        for name in BRM._brm_operation_order(scheduled)
            statements = get(node_statements, name, nothing)
            isnothing(statements) || append!(body.args, statements)
            push!(emitted, name)
        end
        for (name, statements) in node_statements
            name in emitted || append!(body.args, statements)
        end
    end
    value_names = Tuple((logical..., keys(parameters)..., keys(assignments)...))
    returned = Any[]
    for (pi, plan) in enumerate(multi.plans)
        y = response_symbols[pi]
        distribution = _brm_observation_ast(plan, pi, callables)
        push!(body.args, quote
            for $row in eachindex($y)
                $y[$row] ~ $distribution
            end
        end)
        output_names = unique((value_names..., :response))
        fields = [Expr(:kw, name, name === :response ? y : name)
                  for name in output_names]
        push!(returned, Expr(:tuple, Expr(:parameters, fields...)))
    end
    push!(body.args, single ? first(returned) :
        :(; responses=$(Expr(:tuple, returned...))))
    function_name = single ? :brm_model : :brm_multi_model
    inputs = _brm_model_inputs!(body, multi, Tuple(callables), response_symbols)
    signature = Expr(:call, function_name, keys(inputs)...)
    definition = Expr(:macrocall, GlobalRef(Turing, Symbol("@model")),
                      LineNumberNode(0),
                      Expr(:function, signature, body))
    (; definition, inputs)
end

_brm_generic_multi_model_ast(multi::BRM._TuringMultiResponsePlan) =
    _brm_generic_response_graph_ast(multi; single=false)

function _brm_generic_structure_key(definition::Expr)
    function_definition = last(definition.args)
    signature, body = function_definition.args
    # Every emitted literal and structural choice belongs to the cache key.
    # Exclude only the display name. Callable objects remain ordinary runtime
    # arguments, specialized by Julia, and need no name-based registry.
    (repr(signature.args[2:end]), repr(body))
end

function _brm_validate_turing_term_rows(term, nobs)
    rows = _brm_term_rows(term)
    isnothing(rows) && return nothing
    rows == nobs || error(
        "Turing backend: prepared `$(nameof(term.callable))` term has $rows rows, " *
        "but its predictor has $nobs observations")
    nothing
end

function _brm_validate_turing_term_rows(plan::BRM._TuringGenericPlan)
    nobs = length(plan.response)
    for component in plan.predictors, term in component.terms
        _brm_validate_turing_term_rows(term, nobs)
    end
    nothing
end

function _brm_validate_turing_term_rows(plan::BRM._TuringMultiResponsePlan)
    foreach(_brm_validate_turing_term_rows, plan.plans)
    nothing
end

function BRM._brm_turing_model(plan::BRM._TuringGenericPlan)
    _brm_validate_turing_term_rows(plan)
    lowered = _brm_generic_model_ast(plan)
    evaluator, definition = _brm_cached_generic_evaluator(lowered)
    model = Turing.DynamicPPL.Model{false}(evaluator, lowered.inputs)
    plan.source_ast = definition
    model
end
function _zero_correlation_scales(intercept_index, intercept_scale,
                                  slope_scales)
    # DynamicPPL can infer a sampled local as Union{Nothing,T}.  Allocate from
    # the realized slope type so Enzyme never receives an isbits-union array.
    scales = Vector{typeof(first(slope_scales))}(
        undef, length(slope_scales) + (intercept_index > 0))
    slope_index = 1
    for term_index in eachindex(scales)
        if term_index == intercept_index
            scales[term_index] = intercept_scale
        else
            scales[term_index] = slope_scales[slope_index]
            slope_index += 1
        end
    end
    scales
end

function _noncentered_group_coefficients(scales, z_flat, n_groups)
    n_terms = length(scales)
    transpose(reshape(scales, n_terms, 1) .*
              reshape(z_flat, n_terms, n_groups))
end

_brm_group_scale_distribution(prior) = _brm_constrained_kernel(
    isnothing(prior) ? Normal() : prior; lower=0)
_brm_group_scale_distributions(priors::Tuple) =
    collect(map(_brm_group_scale_distribution, priors))
_brm_has_group_prior_override(block) =
    any(!isnothing, block.sd_prior) || block.lkj_eta != 1.0
_brm_has_group_prior_override(blocks::Tuple) =
    any(_brm_has_group_prior_override, blocks)
_brm_has_group_geometry_override(block) =
    !isnothing(block.by) || block.centered || _brm_has_group_prior_override(block)
_brm_has_group_geometry_override(::BRM._BRMMultiMembershipPlan) = true
_brm_has_group_geometry_override(blocks::Tuple) =
    any(_brm_has_group_geometry_override, blocks)

function _brm_centered_coefficients_distribution(scales, L, n_groups)
    factor = Diagonal(scales) * Matrix(L.L)
    covariance = factor * transpose(factor)
    _BRMGroupedMvNormal(
        MvNormal(zeros(length(scales)), Symmetric(covariance)), n_groups)
end

function _brm_centered_diagonal_distribution(scales, n_groups)
    _BRMGroupedMvNormal(
        MvNormal(zeros(length(scales)), Diagonal(scales .^ 2)), n_groups)
end

_brm_centered_intercept_distribution(scale, n_groups) =
    MvNormal(zeros(n_groups), Diagonal(fill(scale^2, n_groups)))

function _brm_stratified_centered_distribution(frames, group_strata)
    n_terms = size(first(frames).factor, 1)
    bases = [MvNormal(
        zeros(n_terms),
        Symmetric(frame.factor * transpose(frame.factor))) for frame in frames]
    _BRMStratifiedMvNormal(bases, group_strata)
end

function _brm_stratified_noncentered_coefficients(
        frames, group_strata, z_flat, n_groups)
    n_terms = size(first(frames).factor, 1)
    z = reshape(z_flat, n_terms, n_groups)
    transpose(hcat((
        frames[group_strata[group]].factor * @view(z[:, group])
        for group in 1:n_groups)...))
end

_brm_stratified_centered_coefficients(values, n_terms, n_groups) =
    transpose(reshape(values, n_terms, n_groups))


Turing.@model function _brm_stratified_group_frame(
        n_terms, sd_priors, lkj_eta, residual_scale)
    L ~ LKJCholesky(n_terms, lkj_eta)
    tau = fill(residual_scale, n_terms)
    if isnothing(residual_scale)
        tau ~ product_distribution(
            _brm_group_scale_distributions(sd_priors))
    end
    factor = Diagonal(tau) * Matrix(L.L)
    (; L, tau, factor)
end

Turing.@model function _brm_random_intercept_effect(group_idx, n_groups)
    log_scale ~ Normal()
    z ~ product_distribution(fill(Normal(), n_groups))
    scale = exp(log_scale)
    values = scale .* z
    effect = values[group_idx]
    (; effect, scale, values)
end


Turing.@model function _brm_random_intercept_effect_prior(
        group_idx, n_groups, prior, residual_scale)
    scale = residual_scale
    if isnothing(residual_scale)
        scale ~ _brm_group_scale_distribution(prior)
    end
    z ~ product_distribution(fill(Normal(), n_groups))
    values = scale .* z
    effect = values[group_idx]
    (; effect, scale, values)
end


Turing.@model function _brm_centered_random_intercept_effect(
        group_idx, n_groups)
    log_scale ~ Normal()
    scale = exp(log_scale)
    values ~ _brm_centered_intercept_distribution(scale, n_groups)
    effect = values[group_idx]
    (; effect, scale, values)
end


Turing.@model function _brm_centered_random_intercept_effect_prior(
        group_idx, n_groups, prior, residual_scale)
    scale = residual_scale
    if isnothing(residual_scale)
        scale ~ _brm_group_scale_distribution(prior)
    end
    values ~ _brm_centered_intercept_distribution(scale, n_groups)
    effect = values[group_idx]
    (; effect, scale, values)
end


Turing.@model function _brm_correlated_group_effect(
        Z, group_idx, n_groups, sd_priors, lkj_eta, residual_scale)
    n_terms = size(Z, 2)
    L ~ LKJCholesky(n_terms, lkj_eta)
    tau = fill(residual_scale, n_terms)
    if isnothing(residual_scale)
        tau ~ product_distribution(
            _brm_group_scale_distributions(sd_priors))
    end
    z_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
    z = reshape(z_flat, n_terms, n_groups)
    coefficients = transpose(Diagonal(tau) * Matrix(L.L) * z)
    effect = vec(sum(Z .* coefficients[group_idx, :]; dims=2))
    (; effect, L, tau, coefficients)
end


function _brm_multi_membership_intercept(values, group_idx, weights,
                                          n_obs, n_memberships)
    effect = similar(values, n_obs)
    for observation in 1:n_obs
        first_index = (observation - 1) * n_memberships
        value = zero(eltype(effect))
        for membership in 1:n_memberships
            index = first_index + membership
            value += weights[index] * values[group_idx[index]]
        end
        effect[observation] = value
    end
    effect
end

function _brm_multi_membership_correlated(
        Z, coefficients, group_idx, weights, n_obs, n_memberships)
    effect = Vector{eltype(coefficients)}(undef, n_obs)
    for observation in 1:n_obs
        first_index = (observation - 1) * n_memberships
        value = zero(eltype(effect))
        for membership in 1:n_memberships
            index = first_index + membership
            member_value = zero(eltype(effect))
            for term in axes(Z, 2)
                member_value += Z[observation, term] *
                                coefficients[group_idx[index], term]
            end
            value += weights[index] * member_value
        end
        effect[observation] = value
    end
    effect
end

Turing.@model function _brm_multi_membership_intercept_effect(
        group_idx, weights, n_obs, n_memberships, n_groups, sd_prior,
        residual_scale)
    scale = residual_scale
    if isnothing(residual_scale)
        scale ~ _brm_group_scale_distribution(sd_prior)
    end
    z ~ product_distribution(fill(Normal(), n_groups))
    values = scale .* z
    effect = _brm_multi_membership_intercept(
        values, group_idx, weights, n_obs, n_memberships)
    (; effect, scale, values)
end

Turing.@model function _brm_multi_membership_correlated_effect(
        Z, group_idx, weights, n_obs, n_memberships, n_groups,
        sd_priors, lkj_eta, residual_scale)
    n_terms = size(Z, 2)
    L ~ LKJCholesky(n_terms, lkj_eta)
    tau = fill(residual_scale, n_terms)
    if isnothing(residual_scale)
        tau ~ product_distribution(_brm_group_scale_distributions(sd_priors))
    end
    z_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
    z = reshape(z_flat, n_terms, n_groups)
    coefficients = transpose(Diagonal(tau) * Matrix(L.L) * z)
    effect = _brm_multi_membership_correlated(
        Z, coefficients, group_idx, weights, n_obs, n_memberships)
    (; effect, L, tau, coefficients)
end


Turing.@model function _brm_centered_correlated_group_effect(
        Z, group_idx, n_groups, sd_priors, lkj_eta, residual_scale)
    n_terms = size(Z, 2)
    L ~ LKJCholesky(n_terms, lkj_eta)
    tau = fill(residual_scale, n_terms)
    if isnothing(residual_scale)
        tau ~ product_distribution(_brm_group_scale_distributions(sd_priors))
    end
    coefficients_flat ~ _brm_centered_coefficients_distribution(
        tau, L, n_groups)
    coefficients = transpose(reshape(coefficients_flat, n_terms, n_groups))
    effect = vec(sum(Z .* coefficients[group_idx, :]; dims=2))
    (; effect, L, tau, coefficients)
end


Turing.@model function _brm_zero_correlation_group_effect(
        Z, group_idx, n_groups, intercept_index, sd_priors, residual_scale)
    n_terms = size(Z, 2)
    n_slopes = n_terms - (intercept_index > 0)
    intercept_scale = residual_scale
    if intercept_index > 0 && isnothing(residual_scale)
        intercept_prior = sd_priors[intercept_index]
        if isnothing(intercept_prior)
            # Match the established random-intercept and Stan default geometry.
            log_intercept_scale ~ Normal()
            intercept_scale = exp(log_intercept_scale)
        else
            intercept_scale ~ _brm_group_scale_distribution(intercept_prior)
        end
    end
    if isnothing(residual_scale)
        tau_slopes ~ product_distribution(
            [_brm_group_scale_distribution(sd_priors[i]) for i in eachindex(sd_priors)
             if i != intercept_index])
    else
        tau_slopes = fill(residual_scale, n_slopes)
    end
    scales = _zero_correlation_scales(
        intercept_index, intercept_scale, tau_slopes)
    z_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
    coefficients = _noncentered_group_coefficients(
        scales, z_flat, n_groups)
    effect = vec(sum(Z .* coefficients[group_idx, :]; dims=2))
    (; effect, intercept_scale, tau_slopes, scales, coefficients)
end


Turing.@model function _brm_centered_zero_correlation_group_effect(
        Z, group_idx, n_groups, intercept_index, sd_priors, residual_scale)
    n_terms = size(Z, 2)
    n_slopes = n_terms - (intercept_index > 0)
    intercept_scale = residual_scale
    if intercept_index > 0 && isnothing(residual_scale)
        intercept_prior = sd_priors[intercept_index]
        if isnothing(intercept_prior)
            # Match the established random-intercept and Stan default geometry.
            log_intercept_scale ~ Normal()
            intercept_scale = exp(log_intercept_scale)
        else
            intercept_scale ~ _brm_group_scale_distribution(intercept_prior)
        end
    end
    if isnothing(residual_scale)
        tau_slopes ~ product_distribution(
            [_brm_group_scale_distribution(sd_priors[i]) for i in eachindex(sd_priors)
             if i != intercept_index])
    else
        tau_slopes = fill(residual_scale, n_slopes)
    end
    scales = _zero_correlation_scales(
        intercept_index, intercept_scale, tau_slopes)
    coefficients_flat ~ _brm_centered_diagonal_distribution(scales, n_groups)
    coefficients = transpose(reshape(coefficients_flat, n_terms, n_groups))
    effect = vec(sum(Z .* coefficients[group_idx, :]; dims=2))
    (; effect, intercept_scale, tau_slopes, scales, coefficients)
end


Turing.@model function _brm_stratified_group_effect(
        Z, group_idx, n_groups, group_strata, n_strata, sd_priors,
        lkj_eta, centered, residual_scale)
    n_terms = size(Z, 2)
    frame_model = _brm_stratified_group_frame(
        n_terms, sd_priors, lkj_eta, residual_scale)
    strata = Vector{Any}(undef, n_strata)
    for stratum in 1:n_strata
        strata[stratum] ~ to_submodel(frame_model)
    end
    coefficients = nothing
    if centered
        coefficients_flat ~ _brm_stratified_centered_distribution(
            strata, group_strata)
        coefficients = _brm_stratified_centered_coefficients(
            coefficients_flat, n_terms, n_groups)
    else
        z_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
        coefficients = _brm_stratified_noncentered_coefficients(
            strata, group_strata, z_flat, n_groups)
    end
    effect = vec(sum(Z .* coefficients[group_idx, :]; dims=2))
    (; effect, strata, coefficients)
end


function _brm_group_effect_model(block::BRM._BRMMultiMembershipPlan, sd_priors,
                                 residual_scale=nothing)
    block.intercept_only && return _brm_multi_membership_intercept_effect(
        block.indices, block.weights, block.n_obs, block.n_memberships,
        length(block.levels), only(sd_priors), residual_scale)
    _brm_multi_membership_correlated_effect(
        block.matrix, block.indices, block.weights, block.n_obs,
        block.n_memberships, length(block.levels), sd_priors,
        block.lkj_eta, residual_scale)
end

function _brm_group_effect_model(block, sd_priors, residual_scale=nothing)
    if !isnothing(block.by)
        return _brm_stratified_group_effect(
            block.matrix, block.indices, length(block.levels),
            block.group_strata, length(block.strata), sd_priors,
            block.lkj_eta, block.centered, residual_scale)
    end
    if block.intercept_only
        prior = only(sd_priors)
        if isnothing(prior) && isnothing(residual_scale)
            return (block.centered ? _brm_centered_random_intercept_effect :
                                     _brm_random_intercept_effect)(
                block.indices, length(block.levels))
        end
        if block.centered
            return _brm_centered_random_intercept_effect_prior(
                block.indices, length(block.levels), prior,
                residual_scale)
        end
        return _brm_random_intercept_effect_prior(
            block.indices, length(block.levels), prior, residual_scale)
    end
    if block.zero_correlation
        intercept_index = something(
            findfirst(column -> column.label === :Intercept, block.columns), 0)
        return (block.centered ?
            _brm_centered_zero_correlation_group_effect :
            _brm_zero_correlation_group_effect)(
            block.matrix, block.indices, length(block.levels), intercept_index,
            sd_priors, residual_scale)
    end
    (block.centered ? _brm_centered_correlated_group_effect :
                      _brm_correlated_group_effect)(
        block.matrix, block.indices, length(block.levels), sd_priors,
        block.lkj_eta, residual_scale)
end

_brm_group_effect_models(component) =
    Tuple(_brm_group_effect_model(block) for block in component.random_effects)
_brm_group_effect_models(blocks::Tuple) =
    Tuple(_brm_group_effect_model(block) for block in blocks)


_brm_repeat_scale(::Tuple{}, ::Tuple{}) = ()
function _brm_repeat_scale(blocks::Tuple, scales::Tuple)
    n_terms = size(first(blocks).matrix, 2)
    scale = first(scales)
    head = if scale isa AbstractVector
        length(scale) == n_terms || error(
            "Turing backend: shared group scale has $(length(scale)) margins; " *
            "expected $n_terms")
        Tuple(scale)
    else
        ntuple(_ -> scale, n_terms)
    end
    (head..., _brm_repeat_scale(Base.tail(blocks), Base.tail(scales))...)
end

_brm_free_group_priors(::Tuple{}, ::Tuple{}) = ()
function _brm_free_group_priors(priors::Tuple, residual_scales::Tuple)
    tail = _brm_free_group_priors(Base.tail(priors), Base.tail(residual_scales))
    isnothing(first(residual_scales)) ? (first(priors), tail...) : tail
end

_brm_shared_term_count(::Tuple{}) = 0
_brm_shared_term_count(matrices::Tuple) =
    size(first(matrices), 2) + _brm_shared_term_count(Base.tail(matrices))

function _brm_shared_scales(tau, residual_scales)
    isempty(tau) && return collect(residual_scales)
    scales = similar(tau, length(residual_scales))
    free = 1
    for i in eachindex(residual_scales)
        if isnothing(residual_scales[i])
            scales[i] = tau[free]
            free += 1
        else
            scales[i] = residual_scales[i]
        end
    end
    scales
end

function _brm_shared_effect(matrix, coefficients, group_idx, first_column)
    effect = Vector{eltype(coefficients)}(undef, size(matrix, 1))
    for observation in axes(matrix, 1)
        value = zero(eltype(effect))
        for term in axes(matrix, 2)
            value += matrix[observation, term] *
                coefficients[group_idx[observation], first_column + term - 1]
        end
        effect[observation] = value
    end
    effect
end


_brm_shared_effects(::Tuple{}, ::Tuple{}, coefficients, first_column) = ()
function _brm_shared_effects(matrices::Tuple, group_indices::Tuple, coefficients,
                             first_column)
    matrix = first(matrices)
    effect = _brm_shared_effect(
        matrix, coefficients, first(group_indices), first_column)
    (effect, _brm_shared_effects(Base.tail(matrices), Base.tail(group_indices),
                                 coefficients,
                                 first_column + size(matrix, 2))...)
end


Turing.@model function _brm_shared_correlated_group_effect(
        matrices, group_indices, n_groups, sd_priors, lkj_eta, residual_scales)
    n_terms = _brm_shared_term_count(matrices)
    L ~ LKJCholesky(n_terms, lkj_eta)
    free_priors = _brm_free_group_priors(sd_priors, residual_scales)
    tau = Float64[]
    if !isempty(free_priors)
        tau ~ product_distribution(_brm_group_scale_distributions(free_priors))
    end
    scales = _brm_shared_scales(tau, residual_scales)
    z_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
    z = reshape(z_flat, n_terms, n_groups)
    coefficients = transpose(Diagonal(scales) * Matrix(L.L) * z)
    effects = _brm_shared_effects(matrices, group_indices, coefficients, 1)
    (; effects, L, tau, scales, coefficients)
end


Turing.@model function _brm_centered_shared_correlated_group_effect(
        matrices, group_indices, n_groups, sd_priors, lkj_eta, residual_scales)
    n_terms = _brm_shared_term_count(matrices)
    L ~ LKJCholesky(n_terms, lkj_eta)
    free_priors = _brm_free_group_priors(sd_priors, residual_scales)
    tau = Float64[]
    if !isempty(free_priors)
        tau ~ product_distribution(_brm_group_scale_distributions(free_priors))
    end
    scales = _brm_shared_scales(tau, residual_scales)
    coefficients_flat ~ _brm_centered_coefficients_distribution(
        scales, L, n_groups)
    coefficients = transpose(reshape(coefficients_flat, n_terms, n_groups))
    effects = _brm_shared_effects(matrices, group_indices, coefficients, 1)
    (; effects, L, tau, scales, coefficients)
end


function _brm_shared_group_effect_model(blocks::Tuple, sd_priors::Tuple,
                                        residual_scales::Tuple)
    isempty(blocks) && error(
        "Turing backend: internal shared group must contain at least one block")
    first_block = first(blocks)
    matrices = map(block -> block.matrix, blocks)
    group_indices = map(block -> block.indices, blocks)
    margin_scales = _brm_repeat_scale(blocks, residual_scales)
    if !isnothing(first_block.by)
        return _brm_stratified_shared_group_effect(
            matrices, group_indices, length(first_block.levels),
            first_block.group_strata, length(first_block.strata), sd_priors,
            first_block.lkj_eta, first_block.centered, margin_scales)
    end
    model = first_block.centered ?
        _brm_centered_shared_correlated_group_effect :
        _brm_shared_correlated_group_effect
    model(matrices, group_indices, length(first_block.levels), sd_priors,
          first_block.lkj_eta, margin_scales)
end


Turing.@model function _brm_stratified_shared_group_effect(
        matrices, group_indices, n_groups, group_strata, n_strata, sd_priors,
        lkj_eta, centered, residual_scales)
    n_terms = _brm_shared_term_count(matrices)
    frame_model = _brm_shared_group_frame(
        n_terms, sd_priors, lkj_eta, residual_scales)
    strata = Vector{Any}(undef, n_strata)
    for stratum in 1:n_strata
        strata[stratum] ~ to_submodel(frame_model)
    end
    coefficients = nothing
    if centered
        coefficients_flat ~ _brm_stratified_centered_distribution(
            strata, group_strata)
        coefficients = _brm_stratified_centered_coefficients(
            coefficients_flat, n_terms, n_groups)
    else
        z_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
        coefficients = _brm_stratified_noncentered_coefficients(
            strata, group_strata, z_flat, n_groups)
    end
    effects = _brm_shared_effects(matrices, group_indices, coefficients, 1)
    (; effects, strata, coefficients)
end


Turing.@model function _brm_shared_group_frame(
        n_terms, sd_priors, lkj_eta, residual_scales)
    L ~ LKJCholesky(n_terms, lkj_eta)
    free_priors = _brm_free_group_priors(sd_priors, residual_scales)
    tau = Float64[]
    if !isempty(free_priors)
        tau ~ product_distribution(_brm_group_scale_distributions(free_priors))
    end
    scales = _brm_shared_scales(tau, residual_scales)
    factor = Diagonal(scales) * Matrix(L.L)
    (; L, tau, scales, factor)
end


function BRM._brm_turing_model(plan::BRM._TuringMultiResponsePlan)
    _brm_validate_turing_term_rows(plan)
    if all(child -> child isa BRM._TuringGenericPlan, plan.plans)
        lowered = _brm_generic_multi_model_ast(plan)
        evaluator, definition = _brm_cached_generic_evaluator(lowered)
        plan.source_ast = definition
        return Turing.DynamicPPL.Model{false}(evaluator, lowered.inputs)
    end
    error("Turing backend: internal non-generic multi-response plan")
end

function _brm_group_keyword_set(value, keyword::Symbol)
    value === nothing && return Set{Symbol}()
    message = "Turing backend: `$keyword` expects a Symbol or collection of Symbols"
    values = if value isa Symbol
        (value,)
    else
        try
            Tuple(value)
        catch
            error(message)
        end
    end
    all(item -> item isa Symbol, values) || error(message)
    Set{Symbol}(values)
end

function BRM.TuringBRMI(brmi::BRM.BRMI; centered_groups=(), cv_groups=())
    centered = _brm_group_keyword_set(centered_groups, :centered_groups)
    cv = _brm_group_keyword_set(cv_groups, :cv_groups)
    isempty(cv) || error(
        "Turing backend: `cv_groups` is a Stan artifact-sizing control and " *
        "does not apply to DynamicPPL execution. Use `reprocess(backend, " *
        "new_data; resample_groups=...)` to draw new group populations.")
    plan = BRM._brm_turing_plan(brmi)
    unknown = setdiff(centered, BRM._turing_group_names(plan))
    isempty(unknown) || error(
        "Turing backend: `centered_groups` names no random-effect block for " *
        "$(sort!(collect(unknown)))")
    plan = BRM._turing_center_groups(plan, centered)
    model = BRM._brm_turing_model(plan)
    BRM.TuringBRMI(brmi, plan, model)
end

"""
    reprocess(backend::TuringBRMI, new_data;
              freeze_constants=true, resample_groups=())

Rebuild a direct-BRMI Turing backend on `new_data`. Frozen replay retains the
training transform constants, categorical coordinates, effect priors, and
ordinary group coordinates; unseen levels fail loudly. Groups named in
`resample_groups` instead take their levels from `new_data`. At posterior
prediction time [`turing_posterior_predictive`](@ref) retains fitted group
scales/correlation factors and redraws only those groups' standardized effects.

`freeze_constants=false` has fresh-fit semantics and should be refitted rather
than evaluated with the old posterior draw.
"""
function BRM.reprocess(
        backend::BRM.TuringBRMI, new_data;
        freeze_constants::Bool=true, resample_groups=())
    groups = resample_groups === nothing ? () :
             resample_groups isa Symbol ? (resample_groups,) :
             Tuple(resample_groups)
    all(group -> group isa Symbol, groups) || error(
        "Turing backend: `resample_groups` expects a Symbol or collection of " *
        "Symbols")
    length(unique(groups)) == length(groups) || error(
        "Turing backend: `resample_groups` contains duplicate group names")
    available_groups = BRM._turing_group_names(backend.plan)
    unknown_groups = setdiff(Set(groups), available_groups)
    isempty(unknown_groups) || error(
        "Turing backend: `resample_groups` names no fitted random-effect " *
        "block for $(sort!(collect(unknown_groups)))")
    prepared_data = freeze_constants ?
                    BRM._turing_replay_input(backend.plan, new_data) : new_data
    rebound = BRM._brm_rebind_brmi(backend.parent, prepared_data)
    fresh = BRM._brm_turing_plan(
        rebound; training=freeze_constants ? backend.plan : nothing)
    fresh = BRM._turing_center_groups(
        fresh, BRM._turing_centered_group_names(backend.plan))
    plan = freeze_constants ? BRM._turing_replay_plan(
        backend.plan, fresh, Set{Symbol}(groups)) : fresh
    model = BRM._brm_turing_model(plan)
    replay = BRM._TuringReplayState(Tuple(groups))
    BRM.TuringBRMI(rebound, plan, model, replay)
end

_brm_pointwise_indices(plan::BRM._TuringGenericPlan) =
    isnothing(plan.missing_response) ? eachindex(plan.response) :
    plan.missing_response.observed_indices

function _brm_typed_loglikelihoods(values)
    isempty(values) && return Float64[]
    T = promote_type(map(typeof, values)...)
    T[values...]
end

function _brm_pointwise_response(plan, raw_values, first_index)
    indices = _brm_pointwise_indices(plan)
    last_index = first_index + length(indices) - 1
    segment = _brm_typed_loglikelihoods(raw_values[first_index:last_index])
    isnothing(plan.missing_response) && return segment, last_index + 1

    T = eltype(segment)
    aligned = Vector{Union{Missing,T}}(undef, length(plan.response))
    fill!(aligned, missing)
    aligned[indices] = segment
    aligned, last_index + 1
end

function _brm_named_pointwise(plan, raw_values)
    result, next_index = _brm_pointwise_response(plan, raw_values, 1)
    next_index == length(raw_values) + 1 || error(
        "Turing backend: DynamicPPL returned an unexpected number of " *
        "pointwise likelihood terms")
    response_name = BRM._turing_direct_observation(plan.context.parent).key
    NamedTuple{(response_name,)}((result,))
end

function _brm_named_pointwise(
        plan::BRM._TuringMultiResponsePlan, raw_values)
    results = Any[]
    next_index = 1
    for child in plan.plans
        result, next_index = _brm_pointwise_response(
            child, raw_values, next_index)
        push!(results, result)
    end
    next_index == length(raw_values) + 1 || error(
        "Turing backend: DynamicPPL returned an unexpected number of " *
        "pointwise likelihood terms")
    NamedTuple{plan.responses}(Tuple(results))
end

function BRM.turing_pointwise_loglikelihoods(
        backend::BRM.TuringBRMI, parameters)
    pointwise = Turing.DynamicPPL.pointwise_loglikelihoods(
        backend.model, Turing.DynamicPPL.InitFromParams(parameters))
    raw_values = values(pointwise)
    all(value -> value isa Number, raw_values) || error(
        "Turing backend: DynamicPPL returned a non-scalar pointwise " *
        "likelihood term for a rowwise BRM observation")
    _brm_named_pointwise(backend.plan, raw_values)
end

function BRM.turing_predictive_model(backend::BRM.TuringBRMI)
    BRM._brm_turing_model(BRM._turing_predictive_plan(backend.plan))
end

_brm_predictive_response_varnames(plan) =
    (Core.apply_type(Turing.DynamicPPL.VarName,
        only(_brm_response_symbols((plan,); single=true)))(),)
_brm_predictive_response_varnames(plan::BRM._TuringMultiResponsePlan) =
    Tuple(Core.apply_type(Turing.DynamicPPL.VarName, name)()
          for name in _brm_response_symbols(plan.plans))

_brm_initialized_chain_skeleton(value) = value
# DynamicPPL 0.41 can leave `#undef` entries in the skeleton of a
# heterogeneous array of prefixed submodels.  Prediction normally avoids the
# broken template through exact chain-key matches, but new-group replay removes
# one such key and therefore needs every remaining skeleton slot initialized.
function _brm_initialized_chain_skeleton(value::AbstractArray)
    initialized = Array{Any}(undef, size(value))
    for i in eachindex(value)
        initialized[i] = isassigned(value, i) ?
                         _brm_initialized_chain_skeleton(value[i]) : nothing
    end
    initialized
end
function _brm_initialized_chain_skeleton(
        value::Turing.DynamicPPL.VarNamedTuple)
    Turing.DynamicPPL.VarNamedTuple(
        map(_brm_initialized_chain_skeleton, value.data))
end

function _brm_initialized_predictive_chain(chain)
    flexi = Turing.FlexiChains
    flexi.FlexiChain{Turing.DynamicPPL.VarName}(
        flexi.niters(chain),
        flexi.nchains(chain),
        chain._data;
        structures=map(
            _brm_initialized_chain_skeleton, chain._structures),
        iter_indices=flexi.iter_indices(chain),
        chain_indices=flexi.chain_indices(chain),
        sampling_time=flexi.sampling_time(chain),
        last_sampler_state=flexi.last_sampler_state(chain),
    )
end

function _brm_predictive_chain(backend, chain)
    response_varnames = _brm_predictive_response_varnames(backend.plan)
    resampled_parameters = _brm_resampled_parameter_names(backend)
    retained = filter(collect(keys(chain))) do key
        hasfield(typeof(key), :name) || return true
        name = getfield(key, :name)
        name isa Turing.DynamicPPL.VarName || return true
        is_response = any(response_varnames) do response_varname
            Turing.DynamicPPL.subsumes(response_varname, name)
        end
        !is_response && string(name) ∉ resampled_parameters
    end
    predictive_chain = chain[retained]
    isempty(resampled_parameters) && return predictive_chain
    _brm_initialized_predictive_chain(predictive_chain)
end

function Turing.predict(
        rng::AbstractRNG, backend::BRM.TuringBRMI, chain;
        include_all::Bool=true)
    parameters = _brm_predictive_chain(backend, chain)
    Turing.predict(
        rng, BRM.turing_predictive_model(backend), parameters;
        include_all,
    )
end

Turing.predict(backend::BRM.TuringBRMI, chain; include_all::Bool=true) =
    Turing.predict(default_rng(), backend, chain; include_all)

BRM.turing_generated_quantities(backend::BRM.TuringBRMI, parameters) =
    Turing.DynamicPPL.returned(backend.model, parameters)

function _brm_complete_predictive_response(response)
    any(ismissing, response) && error(
        "Turing backend: posterior-predictive execution left an ungenerated " *
        "response value")
    collect(nonmissingtype(eltype(response)), response)
end

function _brm_named_predictive(plan, returned)
    response_name = BRM._turing_direct_observation(plan.context.parent).key
    response = _brm_complete_predictive_response(returned.response)
    NamedTuple{(response_name,)}((response,))
end

function _brm_named_predictive(
        plan::BRM._TuringMultiResponsePlan, returned)
    responses = Tuple(
        _brm_complete_predictive_response(child.response)
        for child in returned.responses)
    NamedTuple{plan.responses}(responses)
end

_brm_group_latent_name(block) = block.centered ?
    (block.intercept_only ? "values" : "coefficients_flat") :
    (block.intercept_only ? "z" : "z_flat")
_brm_multiple_group_model(plan::BRM._TuringGenericPlan) =
    any(component -> !isempty(component.random_effects), plan.predictors)

function _brm_resampled_latent_paths(plan::BRM._TuringGenericPlan, groups)
    paths = Set{String}()
    shared_groups = BRM._turing_shared_group_plans(plan.predictors)
    shared_members = Set((member.component_index, member.block_index)
        for shared in shared_groups for member in shared.members)
    for (shared_index, shared) in enumerate(shared_groups)
        any(block -> BRM._turing_block_is_resampled(block, groups),
            shared.blocks) || continue
        latent = first(shared.blocks).centered ? "coefficients_flat" : "z_flat"
        push!(paths, "shared_group_$(shared_index).$latent")
    end
    for (component_index, component) in enumerate(plan.predictors)
        for (block_index, block) in enumerate(component.random_effects)
            (component_index, block_index) in shared_members && continue
            BRM._turing_block_is_resampled(block, groups) || continue
            push!(paths, "group_$(component_index)_$(block_index)." *
                         _brm_group_latent_name(block))
        end
    end
    paths
end

function _brm_generic_multi_resampled_paths(plan, groups)
    paths = Set{String}()
    components = Any[]
    component_names = Set{Symbol}()
    for child in plan.plans, component in child.predictors
        component.predictor.name in component_names && continue
        push!(component_names, component.predictor.name)
        push!(components, component)
    end
    shared_groups = BRM._turing_shared_group_plans(Tuple(components))
    shared_members = Set((member.predictor, member.block_index)
        for shared in shared_groups for member in shared.members)
    for (shared_index, shared) in enumerate(shared_groups)
        any(block -> BRM._turing_block_is_resampled(block, groups),
            shared.blocks) || continue
        latent = first(shared.blocks).centered ? "coefficients_flat" : "z_flat"
        push!(paths, "shared_group_$(shared_index).$latent")
    end
    seen = Set{Tuple{Symbol,Int}}()
    for child in plan.plans, component in child.predictors
        for (block_index, block) in enumerate(component.random_effects)
            key = (component.predictor.name, block_index)
            key in seen && continue
            push!(seen, key)
            key in shared_members && continue
            BRM._turing_block_is_resampled(block, groups) || continue
            push!(paths, "group_$(component.predictor.name)_$(block_index)." *
                         _brm_group_latent_name(block))
        end
    end
    paths
end

function _brm_without_fields(parameters::NamedTuple, removed)
    kept = Tuple(name for name in keys(parameters) if name ∉ removed)
    NamedTuple{kept}(Tuple(getproperty(parameters, name) for name in kept))
end

function _brm_without_parameter_paths(parameters, removed)
    point_parameters = Turing.DynamicPPL.VarNamedTuple(parameters)
    kept = Dict{Turing.DynamicPPL.VarName,Any}()
    for variable in keys(point_parameters)
        string(variable) in removed && continue
        kept[variable] = point_parameters[variable]
    end
    kept
end
function _brm_without_named_parameter_paths(parameters::NamedTuple, removed,
                                             prefix::String)
    kept_names = Symbol[]
    kept_values = Any[]
    for name in keys(parameters)
        path = isempty(prefix) ? string(name) : prefix * "." * string(name)
        path in removed && continue
        value = getproperty(parameters, name)
        if value isa NamedTuple
            value = _brm_without_named_parameter_paths(value, removed, path)
        elseif value isa Turing.DynamicPPL.VarNamedTuple
            value = _brm_without_named_parameter_paths(value.data, removed, path)
        end
        push!(kept_names, name)
        push!(kept_values, value)
    end
    NamedTuple{Tuple(kept_names)}(Tuple(kept_values))
end

function _brm_resampled_parameters(backend::BRM.TuringBRMI, parameters)
    groups = Set{Symbol}(backend.replay.resample_groups)
    isempty(groups) && return parameters
    if backend.plan isa BRM._TuringMultiResponsePlan
        removed = all(child -> child isa BRM._TuringGenericPlan,
                      backend.plan.plans) ?
                  _brm_generic_multi_resampled_paths(backend.plan, groups) :
                  Set{String}()
        if isempty(removed) && !all(child -> child isa BRM._TuringGenericPlan,
                                    backend.plan.plans)
            for i in eachindex(backend.plan.plans)
                backend.plan.owners[i] == i || continue
                for path in _brm_resampled_latent_paths(
                        backend.plan.plans[i], groups)
                    push!(removed, "responses[$i].$path")
                end
            end
        end
        return parameters isa NamedTuple ?
            _brm_without_named_parameter_paths(parameters, removed, "") :
            _brm_without_parameter_paths(parameters, removed)
    end
    if backend.plan isa BRM._TuringGenericPlan
        removed = _brm_resampled_latent_paths(backend.plan, groups)
        return parameters isa NamedTuple ?
            _brm_without_named_parameter_paths(parameters, removed, "") :
            _brm_without_parameter_paths(parameters, removed)
    end
    parameters isa NamedTuple || error(
        "Turing backend: `resample_groups` posterior prediction currently " *
        "requires one constrained parameter draw as a NamedTuple")
    removed = Set(Symbol(path) for path in
                  _brm_resampled_latent_paths(backend.plan, groups))
    _brm_without_fields(parameters, removed)
end

function _brm_resampled_parameter_names(backend::BRM.TuringBRMI)
    groups = Set{Symbol}(backend.replay.resample_groups)
    isempty(groups) && return Set{String}()
    if backend.plan isa BRM._TuringMultiResponsePlan
        all(child -> child isa BRM._TuringGenericPlan,
            backend.plan.plans) &&
            return _brm_generic_multi_resampled_paths(backend.plan, groups)
        removed = Set{String}()
        for i in eachindex(backend.plan.plans)
            backend.plan.owners[i] == i || continue
            for name in _brm_resampled_latent_paths(
                    backend.plan.plans[i], groups)
                push!(removed, "responses[$i].$name")
            end
        end
        return removed
    end
    _brm_resampled_latent_paths(backend.plan, groups)
end

function BRM.turing_posterior_predictive(
        rng::AbstractRNG, backend::BRM.TuringBRMI, parameters)
    predictive = BRM.turing_predictive_model(backend)
    fixed_parameters = _brm_resampled_parameters(backend, parameters)
    fixed = Turing.fix(predictive, fixed_parameters)
    generative = Turing.unfix(
        fixed, _brm_predictive_response_varnames(backend.plan)...)
    draw = rand(rng, generative)
    returned = Turing.DynamicPPL.returned(generative, draw.data)
    _brm_named_predictive(backend.plan, returned)
end

BRM.turing_posterior_predictive(
    backend::BRM.TuringBRMI, parameters; rng=default_rng()) =
    BRM.turing_posterior_predictive(rng, backend, parameters)

end # module
