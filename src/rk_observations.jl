# Keep the full statistical program and its coordinate declarations when a
# response is withheld. Selection changes ordinary emitted observation source;
# it never edits an already-lowered RKPPL plan or caller-owned data.
struct _RKHeldOutPlan{P}
    parent::P
    held_out::Set{Symbol}
end

# An unconditioned BRM program forward-simulates on the Stan route. Its
# declarations belong to generated quantities, not to an HMC coordinate pack.
# Retain the authoritative generative snapshot while emitting only its empty
# fitted density. This does not implement native generated draws.
struct _RKUnconditionedPlan{G}
    generative::G
    columns::Dict{Symbol,Any}
end
_rk_plan_summary(plan::_RKUnconditionedPlan) =
    "unconditioned program (zero fitted coordinates; " *
    string(length(plan.generative.declarations)) * " retained generative declarations)"
_rk_observed_names(::_RKUnconditionedPlan) = ()
_rk_emit_ast(::_RKUnconditionedPlan) =
    _RKEmittedProgram(Expr[], Expr(:block), Pair{Symbol,Any}[])

Base.@nospecializeinfer function _rk_unconditioned_plan(@nospecialize(brmi))
    # Bound formula observations, including kernel-cell inputs, use the
    # ordinary planner. Empty bound arrays still retain this fitted role.
    for node in values(brmi.operations)
        node isa NamedColumn || continue
        operation = parent(node)
        operation isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(operation, 2)
        _brm_observation_name(lhs) === nothing || return nothing
        rhs isa ExprColumn && getf(rhs) === kernel || continue
        args = getargs(rhs)
        isempty(args) && continue
        parts = _sb_kernel_lambda_parts(first(args))
        parts === nothing && continue
        params, body = parts
        for (param, column) in zip(params, args[2:end])
            _sb_cell_param_observed(body, param) || continue
            _brm_observation_name(column) === nothing || return nothing
        end
    end
    # The emitted declaration inventory also finds synthetic observations
    # (for example a modeled baseline). Endpoint absence alone must never
    # discard their densities or their inference coordinates.
    generated = generative_plan(SBBRMI(brmi))
    any(d -> d.role === :observation && d.data_source !== nothing,
        generated.declarations) && return nothing
    _RKUnconditionedPlan(generated, Dict{Symbol,Any}(generated.data))
end
function Base.getproperty(plan::_RKHeldOutPlan, field::Symbol)
    field in (:parent, :held_out) ? getfield(plan, field) :
        getproperty(getfield(plan, :parent), field)
end
_rk_plan_summary(plan::_RKHeldOutPlan) =
    _rk_plan_summary(plan.parent) * " (withheld: " *
    join(string.(sort!(collect(plan.held_out))), ", ") * ")"

# Compilation of preparation must not scale with the entire BRMI source type.
# This barrier applies only to planning; emitted numerical kernels retain their
# ordinary specialization and contain exactly the same source and data.
Base.@nospecializeinfer function _brm_rk_plan(@nospecialize(brmi::BRMI); held_out=())
    selected = _rk_held_out_selection(brmi, held_out)
    original = _brm_rk_unselected_plan(brmi)
    isempty(selected) ? original : _RKHeldOutPlan(original, selected)
end

function _rk_held_out_selection(brmi, held_out)
    request = _brm_held_out_request(held_out; prefix="RK backend")
    isempty(request.names) && return Set{Symbol}()
    aliases = Dict{Symbol,Set{Symbol}}()
    sources = Set{Symbol}()
    unbound = Symbol[]
    function record(target, column)
        source = _brm_observation_name(column)
        if source === nothing
            push!(unbound, target)
            return
        end
        push!(sources, source)
        for alias in (target, source)
            push!(get!(() -> Set{Symbol}(), aliases, alias), source)
        end
    end
    for (key, node) in pairs(brmi.operations)
        node isa NamedColumn || continue
        operation = parent(node)
        operation isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(operation, 2)
        if rhs isa ExprColumn && getf(rhs) === kernel
            args = getargs(rhs)
            isempty(args) && continue
            lam = first(args)
            parts = _sb_kernel_lambda_parts(lam)
            parts === nothing && continue
            params, body = parts
            for (param, column) in zip(params, args[2:end])
                _sb_cell_param_observed(body, param) && record(param, column)
            end
        elseif _brm_observation_name(lhs) !== nothing
            record(key, lhs)
        end
    end
    _brm_resolve_held_out(request, aliases, sources, unbound; prefix="RK backend")
end

_rk_emitted_observation_name(lhs::Symbol) = lhs
_rk_emitted_observation_name(lhs::Expr) =
    lhs.head === :ref ? _rk_emitted_observation_name(first(lhs.args)) : nothing
_rk_emitted_observation_name(_) = nothing

function _rk_source_symbols!(names, value)
    value isa Symbol && push!(names, value)
    value isa Expr && foreach(arg -> _rk_source_symbols!(names, arg), value.args)
    names
end
_rk_source_symbols!(names, values::AbstractVector) =
    (foreach(value -> _rk_source_symbols!(names, value), values); names)
function _rk_source_lhs!(names, lhs)
    name = _rk_emitted_observation_name(lhs)
    name === nothing || push!(names, name)
    Meta.isexpr(lhs, :tuple) && foreach(arg -> _rk_source_lhs!(names, arg), lhs.args)
    names
end
function _rk_source_outputs!(names, statement)
    statement isa Expr || return names
    if statement.head === :(=)
        _rk_source_lhs!(names, first(statement.args))
    elseif statement.head === :call && length(statement.args) == 3 &&
            first(statement.args) in (:~, :.~)
        _rk_source_lhs!(names, statement.args[2])
    elseif statement.head in (:macrocall, :for, :block)
        foreach(arg -> _rk_source_outputs!(names, arg), statement.args)
    end
    names
end
function _rk_source_observes(statement, conditioned)
    statement isa Expr || return false
    statement.head === :call && length(statement.args) == 3 &&
        first(statement.args) in (:~, :.~) &&
        _rk_emitted_observation_name(statement.args[2]) in conditioned && return true
    statement.head in (:macrocall, :for, :block) &&
        return any(arg -> _rk_source_observes(arg, conditioned), statement.args)
    false
end

# Fitted density follows the original observation graph. Independent priors
# never used by any response become generated draws on the Stan route. Resolve
# this before withholding, so a parameter used by a held-out response stays.
# A compound plate remains atomic; its declared block geometry is preserved.
function _rk_fitted_source(emitted, conditioned)
    isempty(conditioned) && return emitted
    statements = emitted.main.args
    outputs = [_rk_source_outputs!(Set{Symbol}(), statement) for statement in statements]
    references = [_rk_source_symbols!(Set{Symbol}(), statement) for statement in statements]
    kept = [_rk_source_observes(statement, conditioned) for statement in statements]
    needed = Set{Symbol}()
    for i in eachindex(statements)
        kept[i] && union!(needed, references[i])
    end
    changed = true
    while changed
        changed = false
        for i in eachindex(statements)
            (kept[i] || isempty(intersect(outputs[i], needed))) && continue
            kept[i] = true
            union!(needed, references[i])
            changed = true
        end
    end
    _RKEmittedProgram(emitted.defs, Expr(:block, statements[kept]...), emitted.bindings)
end
# Withholding changes the observation likelihood. Authored priors and their
# coordinates stay in the program, including priors used only by that response.
function _rk_active_observation_source(emitted, conditioned, held_out)
    statements = emitted.main.args
    withheld = [Meta.isexpr(statement, :call) && length(statement.args) == 3 &&
        statement.args[1] in (:~, :.~) &&
        _rk_emitted_observation_name(statement.args[2]) in conditioned &&
        _rk_emitted_observation_name(statement.args[2]) in held_out
        for statement in statements]
    Expr(:block, statements[.!withheld]...)
end
function _rk_emit_ast(plan::_RKHeldOutPlan; coordinates=nothing)
    emitted = _rk_emit_ast(plan.parent; coordinates)
    conditioned = _rk_observed_names(plan.parent)
    main = _rk_active_observation_source(emitted, conditioned, plan.held_out)
    _RKEmittedProgram(emitted.defs, main, emitted.bindings)
end

_rk_observed_names(plan::_RKStructuralPlan) = Tuple(unique(
    [name for response in plan.responses for name in
        (response.response, response.extra_responses...)]))
_rk_observed_names(plan::_RKValuePlan) = Tuple(o.name for o in plan.observations)
_rk_observed_names(plan::_RKKernelPlan) = (plan.kernel.data_columns[
    findfirst(==(plan.kernel.obs_response), plan.kernel.slice_params)],)
_rk_observed_names(plan::_RKHeldOutPlan) = Tuple(
    name for name in _rk_observed_names(plan.parent) if !(name in plan.held_out))
