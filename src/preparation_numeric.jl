function _brm_fit_mean_numeric(values::AbstractVector, name::Symbol,
                               transform::Symbol, make_error)
    all(value -> value isa Real && isfinite(value), values) ||
        throw(make_error("`$transform($name)` requires finite real training values"))
    isempty(values) && throw(make_error(
        "`$transform($name)` requires at least one training value"))
    fitted_mean = float(first(values))
    for (offset, value) in enumerate(Iterators.drop(values, 1))
        count = offset + 1
        # Dividing before subtracting avoids overflow for finite values near
        # `floatmax`, including samples spanning both signs.
        fitted_mean += float(value) / count - fitted_mean / count
    end
    isfinite(fitted_mean) || throw(make_error(
        "`$transform($name)` produced a non-finite fitted mean"))
    fitted_mean
end

@inline function _brm_scaled_sumsq_numeric(
        magnitude_scale, scaled_squares, deviation)
    magnitude = abs(deviation)
    iszero(magnitude) && return magnitude_scale, scaled_squares
    if magnitude_scale < magnitude
        ratio = magnitude_scale / magnitude
        return magnitude, one(magnitude) + scaled_squares * ratio * ratio
    end
    ratio = magnitude / magnitude_scale
    magnitude_scale, scaled_squares + ratio * ratio
end

function _brm_fit_zscale_numeric(values::AbstractVector, name::Symbol,
                                 make_error)
    length(values) >= 2 || throw(make_error(
        "`zscale($name)` requires at least two training values for sample SD"))
    fitted_mean = _brm_fit_mean_numeric(values, name, :zscale, make_error)

    magnitude_scale = zero(fitted_mean)
    scaled_squares = zero(fitted_mean)
    restore_scale = one(fitted_mean)
    centered_overflow = false
    for value in values
        deviation = float(value) - fitted_mean
        if !isfinite(deviation)
            centered_overflow = true
            break
        end
        magnitude_scale, scaled_squares = _brm_scaled_sumsq_numeric(
            magnitude_scale, scaled_squares, deviation)
    end
    if centered_overflow
        value_scale = maximum(value -> abs(float(value)), values)
        isfinite(value_scale) && value_scale > zero(value_scale) ||
            throw(make_error(
                "`zscale($name)` could not scale its finite training values"))
        normalized_mean = fitted_mean / value_scale
        restore_scale = value_scale
        magnitude_scale = zero(normalized_mean)
        scaled_squares = zero(normalized_mean)
        for value in values
            deviation = float(value) / value_scale - normalized_mean
            magnitude_scale, scaled_squares = _brm_scaled_sumsq_numeric(
                magnitude_scale, scaled_squares, deviation)
        end
    end
    iszero(magnitude_scale) && throw(make_error(
        "`zscale($name)` requires nonzero sample variance"))
    fitted_scale = (magnitude_scale *
        sqrt(scaled_squares / (length(values) - 1))) * restore_scale
    isfinite(fitted_scale) && fitted_scale > zero(fitted_scale) ||
        throw(make_error(
            "`zscale($name)` produced a non-finite or zero sample SD"))
    (; mean=fitted_mean, scale=fitted_scale)
end
