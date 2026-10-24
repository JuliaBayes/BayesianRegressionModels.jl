# The fitted `center` / `zscale` anchors. One source serves both the Julia
# helpers below (SBBRMI and BRM's own preparation call them) and the
# `brm_covariate_mean` / `brm_covariate_sd` kernels an RKBRMI emits
# (`_BRM_STATISTICAL_VALUES`), so ReactiveKernels sees the anchors' own
# arithmetic and guards rather than an opaque call, and can translate them
# (jax_source's host-side `prepare_data`). Hence the bodies keep to what RK
# translates: index loops over `values`, `throw` guards, no closures, no
# `break` and no early `return`. `name`, `transform` and `make_error` are the helpers'
# arguments; an emitted kernel fixes them (`_brm_fit_kernel_body`).

_brm_substitute(x::Symbol, pairs::Pair...) = begin
    for (from, to) in pairs
        x === from && return to
    end
    x
end
_brm_substitute(x::Expr, pairs::Pair...) =
    Expr(x.head, map(arg -> _brm_substitute(arg, pairs...), x.args)...)
_brm_substitute(x, ::Pair...) = x

const _BRM_FIT_MEAN_BODY = quote
    for i in 1:length(values)
        isfinite(values[i]) || throw(make_error(
            "`$transform($name)` requires finite real training values"))
    end
    isempty(values) && throw(make_error(
        "`$transform($name)` requires at least one training value"))
    fitted_mean = float(values[1])
    for count in 2:length(values)
        # Dividing before subtracting avoids overflow for finite values near
        # `floatmax`, including samples spanning both signs.
        fitted_mean += float(values[count]) / count - fitted_mean / count
    end
    isfinite(fitted_mean) || throw(make_error(
        "`$transform($name)` produced a non-finite fitted mean"))
    fitted_mean
end

# One deviation's step of the scaled sum of squares: `scaled_squares *
# magnitude_scale^2` is the sum of squared deviations so far, accumulated
# without squaring a deviation near `floatmax`.
const _BRM_SCALED_SUMSQ_STEP = quote
    magnitude = abs(deviation)
    if !iszero(magnitude)
        if magnitude_scale < magnitude
            ratio = magnitude_scale / magnitude
            scaled_squares = one(magnitude) + scaled_squares * ratio * ratio
            magnitude_scale = magnitude
        else
            ratio = magnitude / magnitude_scale
            scaled_squares = scaled_squares + ratio * ratio
        end
    end
end

# Leaves `fitted_mean` and `fitted_scale` (the sample SD) bound.
const _BRM_FIT_ZSCALE_BODY = quote
    length(values) >= 2 || throw(make_error(
        "`zscale($name)` requires at least two training values for sample SD"))
    fitted_mean = $(_brm_substitute(_BRM_FIT_MEAN_BODY, :transform => QuoteNode(:zscale)))
    magnitude_scale = zero(fitted_mean)
    scaled_squares = zero(fitted_mean)
    restore_scale = one(fitted_mean)
    # A non-finite centered deviation switches to normalized values below,
    # which discards these sums. (No `break`: RK traces recipe loops, and a
    # traced loop admits none.)
    centered_overflow = false
    for i in 1:length(values)
        deviation = float(values[i]) - fitted_mean
        if !isfinite(deviation)
            centered_overflow = true
        end
        if !centered_overflow
            $_BRM_SCALED_SUMSQ_STEP
        end
    end
    if centered_overflow
        value_scale = abs(float(values[1]))
        for i in 2:length(values)
            value_scale = max(value_scale, abs(float(values[i])))
        end
        isfinite(value_scale) && value_scale > zero(value_scale) ||
            throw(make_error(
                "`zscale($name)` could not scale its finite training values"))
        normalized_mean = fitted_mean / value_scale
        restore_scale = value_scale
        magnitude_scale = zero(normalized_mean)
        scaled_squares = zero(normalized_mean)
        for i in 1:length(values)
            deviation = float(values[i]) / value_scale - normalized_mean
            $_BRM_SCALED_SUMSQ_STEP
        end
    end
    iszero(magnitude_scale) && throw(make_error(
        "`zscale($name)` requires nonzero sample variance"))
    fitted_scale = (magnitude_scale *
        sqrt(scaled_squares / (length(values) - 1))) * restore_scale
    isfinite(fitted_scale) && fitted_scale > zero(fitted_scale) ||
        throw(make_error(
            "`zscale($name)` produced a non-finite or zero sample SD"))
end

@eval function _brm_fit_mean_numeric(values::AbstractVector, name::Symbol,
                                     transform::Symbol, make_error)
    Base.require_one_based_indexing(values)
    all(value -> value isa Real, values) ||
        throw(make_error("`$transform($name)` requires finite real training values"))
    $_BRM_FIT_MEAN_BODY
end

@eval function _brm_fit_zscale_numeric(values::AbstractVector, name::Symbol,
                                       make_error)
    Base.require_one_based_indexing(values)
    all(value -> value isa Real, values) ||
        throw(make_error("`zscale($name)` requires finite real training values"))
    $_BRM_FIT_ZSCALE_BODY
    (; mean=fitted_mean, scale=fitted_scale)
end

# An emitted anchor kernel's body: the shared source with the helpers'
# arguments fixed and each message, whose parts are then all literal, folded
# into one string, so the emitted program reads as authored.
_brm_fit_kernel_body(body, value...; name=:predictor, transform=:zscale) =
    Expr(:block, _brm_fold_strings(_brm_substitute(body,
        :name => QuoteNode(name), :transform => QuoteNode(transform),
        :make_error => :ArgumentError)).args..., value...)

_brm_fold_strings(x) = x
function _brm_fold_strings(x::Expr)
    args = map(_brm_fold_strings, x.args)
    x.head === :string && all(arg -> arg isa String || arg isa QuoteNode, args) &&
        return join(arg isa QuoteNode ? string(arg.value) : arg for arg in args)
    Expr(x.head, args...)
end
