"""
    _BRMPreparedRef(name, axis)

A logical reference in a prepared model expression. `axis` is `:scalar`,
`:observation`, `:observation_row`, or `:whole`; it makes row indexing a property of the formula
reference rather than a guess based on the value's Julia container type.
"""
struct _BRMPreparedRef
    name::Symbol
    axis::Symbol
    function _BRMPreparedRef(name::Symbol, axis::Symbol)
        axis in (:scalar, :observation, :observation_row, :whole) || throw(ArgumentError(
            "prepared reference axis must be :scalar, :observation, :observation_row, or :whole"))
        new(name, axis)
    end
end

"""A callable expression retained from an `ExprColumn`, including keywords."""
struct _BRMPreparedExpr{F,A<:Tuple,K<:NamedTuple}
    callable::F
    args::A
    kwargs::K
end

_brm_has_row_ref(x) = false
_brm_has_row_ref(x::_BRMPreparedRef) =
    x.axis in (:observation, :observation_row)
_brm_has_row_ref(x::_BRMPreparedExpr) =
    any(_brm_has_row_ref, x.args) || any(_brm_has_row_ref, values(x.kwargs))

struct _BRMPreparedParameter{P,S}
    name::Symbol
    prior::P
    support::S
    normalized::Bool
end

struct _BRMPreparedPredictor{E,L,P,D}
    name::Symbol
    expression::E
    link::L
    term_priors::P
    dependencies::D
end

struct _BRMPreparedAssignment{E,D}
    name::Symbol
    expression::E
    dependencies::D
end

struct _BRMPreparedObservation{L,E,Y,M,W,MI}
    name::Symbol
    lhs::L
    distribution::E
    response::Y
    modifier::M
    weight::W
    missing_response::MI
end
_BRMPreparedObservation(name, lhs, distribution, response, modifier, weight) =
    _BRMPreparedObservation(name, lhs, distribution, response, modifier, weight,
        _brm_missing_response_plan(lhs; prefix="BRM preparation"))

"""
Backend-neutral semantic program. Backends may prepare only the operation
classes they consume; `source_operations` always retains the complete ordered
BRMI program so unsupported backend features are not discarded.
"""
struct _BRMPreparedModel{G,P,A,O,S}
    program::G
    parameters::P
    predictors::A
    assignments::O
    observations::S
end

function Base.getproperty(model::_BRMPreparedModel, field::Symbol)
    field === :context && return getfield(model, :program).context
    field === :order && return getfield(model, :program).order
    field === :source_operations &&
        return getfield(model, :program).context.parent.operations
    getfield(model, field)
end


"""
    _brm_prepare_model(brmi; program=_brm_prepare_program(brmi))

Prepare the backend-neutral ordered expression graph without deciding which
operations a backend can execute. This intentionally retains exotic terms and
joint responses in `source_operations`; backend-specific preparation can add
geometry while sharing the same identities and dependency order.
"""
function _brm_prepare_model(brmi::BRMI;
                            program=_brm_prepare_program(brmi),
                            additional_parameters=(), observation_overrides=Dict())
    context = program.context
    axes = Dict{Symbol,Symbol}(key => :observation for key in keys(context.data))
    parameters = Any[]
    predictors = Any[]
    assignments = Any[]
    observations = Any[]
    operations_by_name = Dict(operation.name => operation
                              for operation in program.operations)
    # Seed declaration shapes before preparing expressions. Implicit family
    # parameters (for example ordinal cutpoints) use the same preparation as
    # explicitly declared priors.
    raw_parameters = Pair{Symbol,Any}[key => prior for (key, prior) in additional_parameters]
    implicit_names = Set(first.(raw_parameters))
    for key in program.order
        operation = operations_by_name[key]
        if operation.role === :parameter && !(key in implicit_names)
            push!(raw_parameters, key => last(getargs(operation.expression)))
        elseif operation.role === :predictor
            axes[key] = :observation
        end
    end
    for (key, prior) in raw_parameters
        axes[key] = _brm_parameter_reference_axis(prior)
    end
    prior_axes = copy(axes)
    foreach(key -> prior_axes[key] = :whole, keys(context.data))
    for (key, prior) in raw_parameters
        push!(parameters, _BRMPreparedParameter(
            key, _brm_prepare_expr(prior, prior_axes), :distribution, true))
    end
    for operation_name in program.order
        prepared_operation = operations_by_name[operation_name]
        key = prepared_operation.name
        operation = prepared_operation.expression
        if prepared_operation.role === :assignment
            lhs, rhs = getargs(operation, 2)
            target = lhs isa NamedColumn ? name(lhs) : key
            prepared = _brm_prepare_expr(rhs, axes)
            refs = _brm_prepared_references(prepared)
            axes[target] = any(ref -> get(axes, ref, :scalar) in
                                      (:observation, :observation_row), refs) ?
                           :observation : :scalar
            push!(assignments, _BRMPreparedAssignment(
                target, prepared, refs))
            continue
        end
        operation isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(operation, 2)
        response_data_name = _brm_observation_name(lhs)
        if !isnothing(response_data_name)
            # An observation keeps its declaration identity. Joint outcomes
            # have a separate synthetic carrier for their materialized data.
            observation_name = key
            override = get(observation_overrides, observation_name, nothing)
            if !isnothing(override)
                push!(observations, _BRMPreparedObservation(
                    observation_name, lhs, _brm_prepare_expr(override.distribution, axes),
                    override.response, override.modifier, override.weight,
                    override.missing_response))
                continue
            end
            prepared = _brm_prepare_expr(rhs, axes)
            response = get(context.data, response_data_name, nothing)
            weight = isnothing(response) ? nothing :
                _brm_observation_weight_plan(rhs, observation_name, response;
                    prefix="BRM preparation")
            base = isnothing(weight) ? rhs : weight.distribution
            modifier = _brm_response_modifier_plan(base; prefix="BRM preparation")
            push!(observations, _BRMPreparedObservation(
                observation_name, lhs, prepared, response, modifier, weight))
            continue
        end
        if prepared_operation.role === :parameter
            continue
        end
        peeled = _peel_lp_lhs(lhs)
        isnothing(peeled) && continue
        target = last(peeled)
        axes[target] = :observation
        push!(predictors, _BRMPreparedPredictor(
            target, rhs, first(peeled),
            get(context.term_priors, target, Dict{Symbol,Dict{Symbol,Any}}()),
            prepared_operation.dependencies))
    end
    _BRMPreparedModel(
        program, Tuple(parameters), Tuple(predictors), Tuple(assignments), Tuple(observations))
end

_brm_prepared_operation(model::_BRMPreparedModel, key::Symbol) =
    only(operation for operation in model.program.operations if operation.name === key)

_brm_prepared_nodes(model::_BRMPreparedModel) =
    (model.parameters..., model.predictors..., model.assignments..., model.observations...)

# One dependency closure, including resolved prior claims, serves all lowering
# passes. Data references terminate naturally; no backend reclassifies them.
function _brm_reachable_operations(program::_BRMPreparedProgram, roots)
    byname = Dict(operation.name => operation for operation in program.operations)
    found = Set{Symbol}()
    function visit(key)
        key in found && return
        push!(found, key)
        operation = get(byname, key, nothing)
        isnothing(operation) || foreach(visit, operation.dependencies)
    end
    foreach(visit, roots)
    found
end

function _brm_prepared_references!(out::Set{Symbol}, x)
    x isa _BRMPreparedRef && (push!(out, x.name); return out)
    x isa _BRMPreparedExpr || return out
    foreach(arg -> _brm_prepared_references!(out, arg), x.args)
    foreach(arg -> _brm_prepared_references!(out, arg), values(x.kwargs))
    out
end
function _brm_prepared_references!(out::Set{Symbol}, values::Tuple)
    foreach(value -> _brm_prepared_references!(out, value), values)
    out
end
_brm_prepared_references(x) =
    Tuple(sort!(collect(_brm_prepared_references!(Set{Symbol}(), x))))

function _brm_prepare_expr(x, axes::AbstractDict{Symbol,Symbol})
    if x isa NamedColumn
        key = name(x)
        axis = get(axes, key, parent(x) isa DataColumn ? :observation : :scalar)
        if axis === :observation && parent(x) isa DataColumn &&
           parent(parent(x)) isa AbstractMatrix
            axis = :observation_row
        end
        return _BRMPreparedRef(key, axis)
    elseif x isa ExprColumn
        return _brm_prepare_call_expr(getf(x), x, axes)
    elseif x isa Tuple
        return map(value -> _brm_prepare_expr(value, axes), x)
    elseif x isa AbstractArray
        elements = Tuple(_brm_prepare_expr(value, axes) for value in x)
        vector = _BRMPreparedExpr(Base.vect, elements, (;))
        return ndims(x) == 1 ? vector :
            _BRMPreparedExpr(reshape, (vector, size(x)...), (;))
    end
    x
end

_brm_prepare_expr(x; axes=Dict{Symbol,Symbol}()) = _brm_prepare_expr(x, axes)

# A prior is constructed at its parameter site, outside observation loops.
# Named data used by that call is a whole mathematical value, as are sampled
# parameters and deterministic hyperparameters referenced by the prior.
function _brm_prepare_prior_expr(expression)
    references = _brm_operation_references!(Set{Symbol}(), expression)
    _brm_prepare_expr(expression, Dict(name => :whole for name in references))
end

function _brm_prepare_call_expr(callable, expression, axes)
    args = map(arg -> _brm_prepare_expr(arg, axes), getargs(expression))
    kwargs = map(value -> _brm_prepare_expr(value, axes), getkwargs(expression))
    _BRMPreparedExpr(callable, args, kwargs)
end

# A data vector used as a mathematical vector argument is one whole value.
# Matrices supply one vector per observation row; parameter references keep
# their resolved shape and expressions retain their explicit construction.
function _brm_prepare_vector_argument(argument, axes)
    if argument isa NamedColumn && parent(argument) isa DataColumn &&
       parent(parent(argument)) isa AbstractVector{<:Real}
        return _BRMPreparedRef(name(argument), :whole)
    end
    _brm_prepare_expr(argument, axes)
end

function _brm_prepare_call_expr(callable::Type{<:Multinomial}, expression, axes)
    raw = getargs(expression)
    args = Tuple(index == 2 ? _brm_prepare_vector_argument(argument, axes) :
                 _brm_prepare_expr(argument, axes) for (index, argument) in enumerate(raw))
    _BRMPreparedExpr(callable, args,
        map(value -> _brm_prepare_expr(value, axes), getkwargs(expression)))
end

function _brm_prepare_call_expr(callable::Type{<:MvNormal}, expression, axes)
    args = Tuple(if index > 1 && argument isa NamedColumn &&
                    parent(argument) isa DataColumn &&
                    parent(parent(argument)) isa AbstractMatrix
                     _BRMPreparedRef(name(argument), :whole)
                 else
                     _brm_prepare_vector_argument(argument, axes)
                 end for (index, argument) in enumerate(getargs(expression)))
    _BRMPreparedExpr(callable, args,
        map(value -> _brm_prepare_expr(value, axes), getkwargs(expression)))
end

# This evaluator is for preparation/replay diagnostics. Executable backends
# lower `_BRMPreparedExpr` to native code and do not call it in density loops.
_brm_eval_prepared_expr(x, _values, _row=nothing) = x
function _brm_eval_prepared_expr(x::_BRMPreparedRef, values, row=nothing)
    value = getproperty(values, x.name)
    x.axis === :observation_row ? view(value, row, :) :
        x.axis === :observation ? value[row] : value
end
function _brm_eval_prepared_expr(x::_BRMPreparedExpr, values, row=nothing)
    args = map(arg -> _brm_eval_prepared_expr(arg, values, row), x.args)
    kwargs = map(arg -> _brm_eval_prepared_expr(arg, values, row), x.kwargs)
    x.callable(args...; kwargs...)
end

struct _BRMPopulationComponent{P<:_BRMPopulationPredictor,
                                  D<:_BRMPopulationDesign,
                                  B<:AbstractVector,R<:Tuple}
    predictor::P
    design::D
    beta_location::B
    beta_scale::B
    random_effects::R
end

struct _BRMPreparedPredictorGeometry{C,P,T,R}
    component::C
    priors::P
    terms::T
    r2d2::R
end
_BRMPreparedPredictorGeometry(component, priors, terms) =
    _BRMPreparedPredictorGeometry(component, priors, terms, nothing)
function Base.getproperty(p::_BRMPreparedPredictorGeometry, field::Symbol)
    field in (:component, :priors, :terms, :r2d2) ? getfield(p, field) :
        getproperty(getfield(p, :component), field)
end

function _brm_prepare_predictor_geometry(
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
            "BRM replay: prepared term set changed for `$predictor`")
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
        required=true, row_source,
        implicit_intercept=name in _brm_threshold_located_predictors(brmi))
    predictor_plan = _BRMPopulationPredictor(
        name, link_lhs_fn, _brm_lp_emitted_name(name, link_lhs_fn), design)
    priors = _brm_simple_population_effect_overrides(
        brmi, design; prefix="BRM preparation", available_predictors)
    isnothing(priors) && (priors = Any[nothing for _ in design.columns])
    defaults = zeros(Float64, length(priors)), ones(Float64, length(priors))
    component = _BRMPopulationComponent(
        predictor_plan, design, defaults..., random_effects)
    r2plan = _brm_whole_predictor_r2d2(
        brmi, design, priors; prefix="BRM preparation", available_predictors)
    _BRMPreparedPredictorGeometry(component, Tuple(priors), prepared_terms, r2plan)
end
