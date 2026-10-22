# Population formula terms broadcast all positional and keyword inputs, as
# shared preparation does. Emitted ordinary functions state that same broadcast.

# A singleton used only by a broadcast recipe has its own input axis. Direct
# predictor and observation reads still require their original row alignment.
function _rk_callable_broadcast_columns(predictors, responses, derived, columns)
    inputs = Set{Symbol}()
    function visit(value)
        value isa Expr || return
        if value.head === :call && first(value.args) isa _RKDataCall
            foreach(arg -> _rk_source_symbols!(inputs, arg), value.args[2:end])
        end
        foreach(visit, value.args)
    end
    foreach(spec -> visit(spec.expression), derived)
    aligned = Set{Symbol}(column for predictor in predictors
        for term in predictor.terms for column in term.columns)
    for response in responses, field in fieldnames(typeof(response))
        value = getfield(response, field)
        value isa Symbol && push!(aligned, value)
        value isa AbstractVector{Symbol} && union!(aligned, value)
    end
    Set{Symbol}(key for key in setdiff(inputs, aligned)
        if haskey(columns, key) && length(columns[key]) == 1)
end

# A data preparation recipe is a named statistical kernel over its inputs.
# Data-only responses instead use direct definitions, which RKPPL evaluates
# once, at binding.
_rk_ast_data_preparation!(defs, taken, ::Val{recipe}, inputs...) where {recipe} =
    _rk_ast_statistical_call!(defs, taken, recipe, inputs...; kernel=true)
_rk_ast_data_preparation!(defs, taken, ::Val{:brm_flatten_response}, cells) =
    Expr(:call, :brm_flatten_cells, cells)
_rk_ast_data_preparation!(defs, taken, ::Val{:brm_gather_response}, raw, groups) =
    Expr(:ref, raw, Expr(:call, :brm_flatten_cells, groups))

# A fitted value (an anchor, a level set) that a program computes only from
# its fit-role inputs (`_rk_fit_twin!`) is its own named data-only definition,
# emitted once: `age_fit_mean = brm_covariate_mean(age_fit)`. A replay can then
# bind the fitted values in place of the training columns (`brm_fitted_state`).
# `fitted` holds the fit-role names and the definitions emitted so far.
_RKFittedValues(inputs) = (; inputs=Set{Symbol}(inputs), names=Dict{Expr,Symbol}())

_rk_fitted_argument(x::Symbol, inputs) = x in inputs
_rk_fitted_argument(x::Expr, _inputs) = false
_rk_fitted_argument(_x, _inputs) = true

_rk_ast_fitted_value!(statements, taken, ::Nothing, recipe, call) = call
function _rk_ast_fitted_value!(statements, taken, fitted, recipe::Symbol, call)
    Meta.isexpr(call, :call) || return call
    inputs = call.args[2:end]
    fit = findfirst(x -> x isa Symbol, inputs)
    fit === nothing && return call
    all(x -> _rk_fitted_argument(x, fitted.inputs), inputs) || return call
    get!(fitted.names, call) do
        stem = replace(string(recipe), r"^brm_(covariate_|fit_)?" => "")
        name = _rk_ast_fresh_name(string(inputs[fit], "_", stem), taken)
        push!(statements, Expr(:(=), name, call))
        name
    end
end

function _rk_ast_data_expr!(defs, statements, bindings, taken, value; fitted=nothing)
    value isa Expr || return value
    if value.head === :_rk_data_preparation
        inputs = map(arg -> _rk_ast_data_expr!(defs, statements, bindings, taken, arg;
            fitted), value.args[2:end])
        recipe = first(value.args)
        call = _rk_ast_data_preparation!(defs, taken, Val(recipe), inputs...)
        return _rk_ast_fitted_value!(statements, taken, fitted, recipe, call)
    end
    if value.head === :call && first(value.args) isa _RKDataCall
        recipe = first(value.args)
        inputs = map(arg -> _rk_ast_data_expr!(defs, statements, bindings, taken, arg;
            fitted), value.args[2:end])
        callee = _rk_value_callee!(bindings, recipe.callable, taken)
        params = [Symbol("input", i) for i in eachindex(inputs)]
        scalar_inputs = params
        call = Expr(:call, callee, scalar_inputs[1:recipe.npos]...)
        if !isempty(recipe.keywords)
            kws = [Expr(:kw, key, scalar_inputs[recipe.npos+i])
                for (i, key) in enumerate(recipe.keywords)]
            insert!(call.args, 2, Expr(:parameters, kws...))
        end
        if !any(recipe.vector_inputs)
            scalar = Dict(param => input for (param, input) in zip(params, inputs))
            substitute(x::Symbol) = get(scalar, x, x)
            substitute(x::Expr) = Expr(x.head, map(substitute, x.args)...)
            substitute(x) = x
            return substitute(call)
        end
        model = _rk_ast_fresh_name("brm_data_term", taken)
        result = _rk_ast_fresh_name("brm_data_value", taken)
        scalar = Expr(:->, Expr(:tuple, params...), call)
        broadcast = Expr(:call, :broadcast, scalar, params...)
        push!(defs, Expr(:function, Expr(:call, model, params...),
            Expr(:block, Expr(:return, broadcast))))
        push!(statements, Expr(:(=), result, Expr(:call, model, inputs...)))
        return result
    end
    Expr(value.head, map(arg -> _rk_ast_data_expr!(defs, statements, bindings, taken, arg;
        fitted), value.args)...)
end
