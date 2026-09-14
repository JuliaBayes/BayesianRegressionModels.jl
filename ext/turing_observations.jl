# Observation structure is known when the model is emitted. Compose the
# distribution AST once; the sampled body contains ordinary distribution calls.
_brm_bound_ast(::Nothing, _reference, _row) = nothing
_brm_bound_ast(::Real, reference, _row) = reference
_brm_bound_ast(::AbstractVector, reference, row) = :($reference[$row])

_brm_modifier_callable(::Val{:truncated}) = truncated
_brm_modifier_callable(::Val{:censored}) = censored

function _brm_modifier_ast(kind, base, modifier, reference, row)
    callable = _brm_modifier_callable(kind)
    wrapper = GlobalRef(parentmodule(callable), nameof(callable))
    bounds = [Expr(:kw, name, _brm_bound_ast(value, :($reference.$name), row))
              for name in (:lower, :upper)
              for value in (getproperty(modifier, name),) if !isnothing(value)]
    Expr(:call, wrapper, Expr(:parameters, bounds...), base)
end
function _brm_modifier_ast(::Val{:interval_censored}, base, modifier, reference, row)
    upper = _brm_bound_ast(modifier.upper, :($reference.upper), row)
    :(_brm_interval_evidence($base, $upper))
end

# Analytic weights mean Gaussian precision, unlike a likelihood power. Preserve
# a custom Normal factory's complete call and evaluate it only once.
_brm_precision_ast(_constructor, distribution, weight, callables) =
    _brm_precision_factory_ast(distribution, weight, callables)
function _brm_precision_factory_ast(distribution, weight, callables)
    base = _brm_prepared_ast(distribution, callables)
    quote
        let observation_distribution = $base::$(GlobalRef(Distributions, :Normal))
            Normal(mean(observation_distribution),
                   std(observation_distribution) / sqrt($weight))
        end
    end
end
function _brm_precision_ast(::Type{Normal}, distribution, weight, callables)
    length(distribution.args) <= 2 ||
        return _brm_precision_factory_ast(distribution, weight, callables)
    args = map(arg -> _brm_prepared_ast(arg, callables), distribution.args)
    location = isempty(args) ? 0 : first(args)
    scale = length(args) < 2 ? 1 : args[2]
    keywords = [Expr(:kw, key, _brm_prepared_ast(value, callables))
                for (key, value) in pairs(distribution.kwargs)]
    signature = Any[GlobalRef(Distributions, :Normal)]
    isempty(keywords) || push!(signature, Expr(:parameters, keywords...))
    append!(signature, (location, :($scale / sqrt($weight))))
    Expr(:call, signature...)
end

_brm_weight_ast(::Val{:frequency}, base, weight) =
    :(_BRMObjectiveWeight($base, $weight))
_brm_weight_ast(::Val{:power}, base, weight) =
    :(_BRMObjectiveWeight($base, $weight))

function _brm_observation_ast(plan, index, callables)
    row = _brm_row_symbol(callables)
    weight = plan.observation_weight
    weight_value = :(multi.plans[$index].observation_weight.values[$row])
    base = if !isnothing(weight) && weight.kind === :analytic
        _brm_precision_ast(plan.distribution.callable, plan.distribution,
                           weight_value, callables)
    else
        _brm_prepared_ast(plan.distribution, callables)
    end
    modifier = plan.response_modifier
    if !isnothing(modifier)
        base = _brm_modifier_ast(Val(modifier.kind), base, modifier,
                                 :(multi.plans[$index].response_modifier), row)
    end
    isnothing(weight) || weight.kind === :analytic ? base :
        _brm_weight_ast(Val(weight.kind), base, weight_value)
end
