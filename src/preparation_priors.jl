# Common composite-prior preparation. Concrete backends own declarations and
# parameterization translations; the prepared prior retains the Julia call.

_brm_prior_constant(value) = value
function _brm_prior_constant(value::ExprColumn)
    args = map(_brm_prior_constant, getargs(value))
    kwargs = map(_brm_prior_constant, getkwargs(value))
    if all(x -> x isa Real, args) && all(x -> x isa Real, values(kwargs))
        result = try getf(value)(args...; kwargs...) catch; nothing end
        result isa Real && return result
    end
    value
end

function _brm_horseshoe_scale(target, key, value; prefix="BRM preparation")
    resolved = _brm_prior_constant(value)
    resolved isa Bool && error(
        "$prefix: `$target ~ Horseshoe($key=...)` requires a numeric formula constant or model value")
    if resolved isa Real
        isfinite(resolved) && resolved > 0 || error(
            "$prefix: `$target ~ Horseshoe($key=...)` must be finite and strictly positive")
        return Float64(resolved)
    end
    resolved
end

function _brm_horseshoe_spec(target, args, kwargs; prefix="BRM preparation")
    isempty(args) || error(
        "$prefix: `$target ~ Horseshoe(...)` accepts no positional arguments; " *
        "use `local_scale=` and/or `global_scale=`")
    unknown = setdiff(keys(kwargs), (:local_scale, :global_scale))
    isempty(unknown) || error(
        "$prefix: `$target ~ Horseshoe(...)` accepts only `local_scale` and `global_scale` keywords, got $unknown")
    (; local_scale=_brm_horseshoe_scale(target, :local_scale,
                                      get(kwargs, :local_scale, 1.0); prefix),
       global_scale=_brm_horseshoe_scale(target, :global_scale,
                                       get(kwargs, :global_scale, 1.0); prefix))
end

# Shape/support traits describe distributions and composition, rather than a
# catalogue of constructors accepted by one backend.
_brm_distribution_shape(::Type{D}, _args) where {D<:Distribution} =
    (Distributions.variate_form(D), Distributions.value_support(D))
function _brm_distribution_shape(::Type{<:LocationScale}, args)
    length(args) == 3 || error("BRM preparation: LocationScale expects three arguments")
    _brm_distribution_shape(last(args))
end
_brm_distribution_shape(expression::ExprColumn) =
    _brm_distribution_shape(getf(expression), getargs(expression))
function _brm_distribution_shape(constructor, args)
    type = brm_distribution_type(constructor)
    isnothing(type) && return nothing
    type isa Type && type <: Distribution || error(
        "BRM preparation: brm_distribution_type($constructor) must return a Distribution type or nothing")
    _brm_distribution_shape(type, args)
end

function _brm_r2d2_prior(value, target)
    prior = isnothing(value) ? ExprColumn(Beta, 1.0, 1.0) :
            _as_expr_column(value)
    isnothing(prior) && error(
        "BRM preparation: R2 prior for `$target` must be a distribution call")
    _brm_r2d2_prior_normalize(getf(prior), prior)
end
_brm_r2d2_prior_normalize(_constructor, prior) = prior
function _brm_r2d2_prior_normalize(::Type{<:Beta}, prior)
    # Preserve the established constant Beta representation. Symbolic shapes
    # remain model references and are ordered with the other sampled values.
    args = map(x -> x isa Real ? Float64(x) : x, getargs(prior))
    if all(x -> x isa Real, args)
        Beta(args...)
    end
    ExprColumn(getf(prior), args...; getkwargs(prior)...)
end

struct _BRMR2D2Plan{P,A,T}
    prior::P
    share_indices::Vector{Int}
    alpha::A
    total_scale::T
end

struct _BRMJointR2D2Predictor
    plan_index::Int
    predictor::Symbol
    component_index::Int
    block_index::Int
    coefficient_shares::Vector{Int}
    margin_shares::Vector{Int}
end
_BRMJointR2D2Predictor(predictor, component_index, block_index,
                       coefficient_shares, margin_shares) =
    _BRMJointR2D2Predictor(0, predictor, component_index, block_index,
                           coefficient_shares, margin_shares)

struct _BRMJointR2D2Plan{P,R}
    id::Symbol
    prior::P
    alpha::Float64
    reference_scale::R
    n_shares::Int
    predictors::Vector{_BRMJointR2D2Predictor}
end

function _brm_merge_joint_r2d2(plans, owners=eachindex(plans); prefix="Turing backend")
    grouped = Dict{Symbol,Vector{Tuple{Int,Any}}}()
    for (pi, plan) in pairs(plans), allocation in plan.joint_r2d2
        owners[pi] == pi || continue
        push!(get!(grouped, allocation.id, Tuple{Int,Any}[]), (pi, allocation))
    end
    Tuple(begin
        first_plan = first(entries)[2]
        all(e -> isequal(e[2].prior, first_plan.prior) &&
                 e[2].alpha == first_plan.alpha &&
                 isequal(e[2].reference_scale, first_plan.reference_scale), entries) ||
            error("$prefix: shared joint R2D2 block `$(first_plan.id)` has inconsistent priors")
        cursor = 0
        mappings = _BRMJointR2D2Predictor[]
        # Rebase every child's local simplex positions into one global simplex.
        for (pi, allocation) in entries
            offset = cursor
            for mapping in allocation.predictors
                push!(mappings, _BRMJointR2D2Predictor(pi, mapping.predictor,
                    mapping.component_index, mapping.block_index,
                    [s == 0 ? 0 : s + offset for s in mapping.coefficient_shares],
                    [s + offset for s in mapping.margin_shares]))
            end
            cursor += allocation.n_shares
        end
        _BRMJointR2D2Plan(first_plan.id, first_plan.prior, first_plan.alpha,
                          first_plan.reference_scale, cursor, mappings)
    end for entries in values(grouped))
end

function _brm_joint_r2d2_plans(brmi::BRMI, components; prefix="BRM preparation")
    specs = [s for s in ranef_effect_priors(brmi)
             if s.class === :sd && s.family === r2d2 &&
                isnothing(s.predictor) && isnothing(s.coefficient) &&
                haskey(s.keywords, :include)]
    out = _BRMJointR2D2Plan[]
    for spec in specs
        members = spec.keywords.include isa Symbol ? (spec.keywords.include,) : Tuple(spec.keywords.include)
        :population in members || error("$prefix: joint R2D2 requires `include=:population`")
        prior = if haskey(spec.keywords, :R2)
            _brm_r2d2_prior(spec.keywords.R2, "sd(:, $(spec.id))")
        else
            m = _brm_numeric_constant(get(spec.keywords, :mean_R2, .5))
            p = _brm_numeric_constant(get(spec.keywords, :prec_R2, 2.))
            (isnothing(m) || isnothing(p)) && error("$prefix: joint R2D2 moments must be numeric")
            Beta(Float64(m*p), Float64((1-m)*p))
        end
        alpha = _brm_numeric_constant(get(spec.keywords, :alpha, get(spec.keywords, :concentration, 1.)))
        (isnothing(alpha) || alpha <= 0) && error("$prefix: joint R2D2 concentration must be positive")
        reference = get(spec.keywords, :reference_scale, nothing)
        isnothing(reference) && error("$prefix: joint R2D2 requires `reference_scale=`")
        matches = Tuple{Int,Int,Any}[]
        for (ci, component) in pairs(components), (bi, block) in pairs(component.random_effects)
            block.id === spec.id && push!(matches, (ci, bi, block))
        end
        isempty(matches) && error("$prefix: `sd(:, $(spec.id))` matches no random-effect block")
        cursor = 0
        margins = Dict{Int,Vector{Int}}()
        for (ci, _, block) in matches
            shares = collect(cursor+1:cursor+length(block.columns)); cursor += length(shares)
            margins[ci] = shares
        end
        mappings = _BRMJointR2D2Predictor[]
        for (ci, bi, _) in matches
            component = components[ci]
            isnothing(component.r2d2) || error("$prefix: predictor `$(component.predictor.name)` has both whole and joint R2D2")
            coeff = zeros(Int, length(component.design.columns))
            for j in eachindex(coeff)
                component.design.columns[j].label === :Intercept && continue
                !isnothing(component.priors[j]) && error("$prefix: explicit coefficient prior conflicts with joint R2D2")
                cursor += 1; coeff[j] = cursor
            end
            push!(mappings, _BRMJointR2D2Predictor(component.predictor.name, ci, bi, coeff, margins[ci]))
        end
        push!(out, _BRMJointR2D2Plan(spec.id, prior, Float64(alpha), reference, cursor, mappings))
    end
    Tuple(out)
end

function _brm_whole_predictor_r2d2(brmi::BRMI, design, coefficient_priors;
                                    prefix="BRM preparation",
                                    available_predictors=(design.target,))
    any(spec -> isnothing(spec.predictor), r2d2_priors(brmi)) &&
        length(available_predictors) != 1 && error(
            "$prefix: `effect(:, :) ~ r2d2(...)` is ambiguous; name a predictor")
    specs = [spec for spec in r2d2_priors(brmi)
             if isnothing(spec.predictor) || spec.predictor === design.target]
    isempty(specs) && return nothing
    length(specs) == 1 || error("$prefix: duplicate `r2d2` statement for `$(design.target)`")
    spec = only(specs)
    getf(spec.expression) === r2d2 || error("$prefix: malformed R2D2 prior")
    isempty(spec.arguments) || error("$prefix: `r2d2(...)` takes keyword arguments only")
    all(k -> k in (:R2, :tau_bsv, :alpha), keys(spec.keywords)) || error(
        "$prefix: unknown `r2d2` keyword")
    prior = _brm_r2d2_prior(get(spec.keywords, :R2, nothing), design.target)
    alpha = _brm_numeric_constant(get(spec.keywords, :alpha, 1.0))
    isnothing(alpha) && error("$prefix: R2D2 `alpha` must be a numeric constant")
    isfinite(alpha) && alpha > 0 || error("$prefix: R2D2 `alpha` must be positive")
    total = get(spec.keywords, :tau_bsv, nothing)
    if !isnothing(total)
        total = _brm_numeric_constant(total)
        isnothing(total) && error("$prefix: R2D2 `tau_bsv` must be a numeric constant")
        isfinite(total) && total > 0 || error("$prefix: R2D2 `tau_bsv` must be positive")
    end
    shares = zeros(Int, length(design.columns))
    next = 0
    for i in eachindex(shares)
        design.columns[i].label === :Intercept && continue
        !isnothing(coefficient_priors[i]) && continue
        next += 1
        shares[i] = next
    end
    _BRMR2D2Plan(prior, shares, Float64(alpha), total)
end

function _brm_lkj_covariance_factor_spec(target::Symbol, op::ExprColumn;
                                          prefix="BRM preparation")
    args = getargs(op)
    length(args) == 1 || error(
        "$prefix: `$target ~ LKJCovarianceFactor(K; ...)` needs exactly one " *
        "dimension argument, got $(length(args))")
    K = only(args)
    K isa Integer && !(K isa Bool) && K >= 1 || error(
        "$prefix: `$target ~ LKJCovarianceFactor(K; ...)` needs an integer " *
        "dimension >= 1, got $(repr(K))")

    kw = getkwargs(op)
    unknown = Symbol[k for k in keys(kw) if !(k in (:scale_prior, :shape))]
    isempty(unknown) || error(
        "$prefix: `$target ~ LKJCovarianceFactor(...)` accepts only " *
        "`scale_prior` and `shape`, got $unknown")

    scale_prior = get(kw, :scale_prior, ExprColumn(Exponential, 1.0))
    scale_prior isa ExprColumn || error(
        "$prefix: `scale_prior` for `$target` must be a positive-support " *
        "distribution call, got $(typeof(scale_prior))")
    # Keep the callable distribution expression; positive declaration support
    # is a property of the covariance scale, not a restriction to one family.
    # Each backend preserves the ordinary prior kernel on that support.

    shape = _brm_prior_constant(get(kw, :shape, 1.0))
    if shape isa Real
        !(shape isa Bool) && isfinite(shape) && shape > 0 || error(
            "$prefix: LKJ `shape` for `$target` must be finite and strictly " *
            "positive, got $(repr(shape))")
        shape = Float64(shape)
    end
    (; K=Int(K), scale_prior, shape)
end
