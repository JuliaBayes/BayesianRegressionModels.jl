struct _BRMPreprocessResult{V,E<:_BRMPreprocEntry}
    values::V
    entry::E
end

function _brm_apply_fitted_levels(levels, raw::AbstractVector; prefix="BRM replay")
    values = raw isa CA.CategoricalVector ? CA.unwrap.(raw) : raw
    lookup = Dict(level => index for (index, level) in enumerate(levels))
    indices = Vector{Int}(undef, length(values))
    for (row, level) in enumerate(values)
        haskey(lookup, level) || error(
            "$prefix: value `$level` is not a training level for this factor " *
            "(training levels: $(collect(levels))). The trained model has no " *
            "parameter for an unseen level. Re-fit with `freeze_constants=false` to " *
            "re-derive levels from the new data, or drop unseen categories first.")
        indices[row] = lookup[level]
    end
    indices
end

_brm_replay_preprocess(entry::_BRMPreprocEntry, input; freeze::Bool=true,
                       prefix="BRM replay") =
    _brm_replay_preprocess(Val(entry.kind), entry, input, freeze, prefix)

function _brm_replay_preprocess(::Union{Val{:zscale},Val{:standardize}},
                                entry, raw, freeze, _prefix)
    values = collect(Float64, raw)
    fit = freeze ? entry.const_ : _brm_fit_zscale(values)
    _BRMPreprocessResult((primary=_brm_apply_zscale(fit, values),),
        _BRMPreprocEntry(entry.kind, fit, entry.raw_ref, false))
end

function _brm_replay_preprocess(::Val{:center}, entry, raw, freeze, _prefix)
    values = collect(Float64, raw)
    fit = freeze ? entry.const_ : _brm_fit_center(values)
    _BRMPreprocessResult((primary=_brm_apply_center(fit, values),),
        _BRMPreprocEntry(:center, fit, entry.raw_ref, false))
end

function _brm_replay_preprocess(::Val{:protect}, entry, raw, _freeze, _prefix)
    _BRMPreprocessResult((primary=collect(Float64, raw),), entry)
end

function _brm_replay_preprocess(::Val{:interaction}, entry, inputs::Tuple,
                                _freeze, prefix)
    left, right = inputs
    length(left) == length(right) || error(
        "$prefix operand lengths mismatch " *
        "($(length(left)) vs $(length(right)))")
    _BRMPreprocessResult((primary=collect(Float64, left .* right),), entry)
end

function _brm_replay_preprocess(::Val{:population_factor_dummy}, entry,
                                raw::AbstractVector, freeze, prefix)
    ref = entry.const_.ref
    ref isa AbstractString &&
        return _brm_replay_string_factor_dummy(entry, raw, freeze, prefix)
    recoded = if ref == 1
        raw
    else
        raw isa AbstractVector{<:Integer} || error(
            "$prefix: `factor($(entry.raw_ref); ref=$ref)` requires " *
            "integer-coded categorical data")
        Int[value == ref ? 1 : value == 1 ? ref : value for value in raw]
    end
    levels = freeze ? entry.const_.levels : _brm_fit_levels(recoded)
    length(levels) == entry.const_.n_levels || error(
        "$prefix: categorical population predictor `$(entry.raw_ref)` has " *
        "$(length(levels)) levels, but the fitted interaction design has " *
        "$(entry.const_.n_levels). Preserve the fitted level count or rebuild " *
        "the model.")
    indices = _brm_apply_fitted_levels(levels, recoded; prefix)
    values = Float64[index == entry.const_.level ? 1.0 : 0.0 for index in indices]
    fit = (; levels, level=entry.const_.level,
           n_levels=entry.const_.n_levels, ref)
    _BRMPreprocessResult((primary=values,), _BRMPreprocEntry(
        :population_factor_dummy, fit, entry.raw_ref, true))
end

# String mirror of the integer `factor(...; ref=...)` replay above. Fit time
# recodes string levels to sorted integer codes with the reference first, so
# replay maps the raw strings back onto the frozen (or, with
# `freeze=false`, re-derived) level set and applies the same code swap.
# Unseen levels fail loudly under `freeze=true`, exactly as for integers. A
# re-derived level set that drops the reference keeps the codes as they are:
# there is no string analogue of the integer path's phantom recoded value.
function _brm_replay_string_factor_dummy(entry, raw::AbstractVector,
                                          freeze, prefix)
    values = _brm_factor_values(raw)
    levels = freeze ? entry.const_.levels : _brm_fit_levels(values)
    length(levels) == entry.const_.n_levels || error(
        "$prefix: categorical population predictor `$(entry.raw_ref)` has " *
        "$(length(levels)) levels, but the fitted interaction design has " *
        "$(entry.const_.n_levels). Preserve the fitted level count or rebuild " *
        "the model.")
    ref_code = findfirst(==(entry.const_.ref), levels)
    codes = _brm_apply_fitted_levels(levels, values; prefix)
    indices = isnothing(ref_code) ? codes :
        Int[code == ref_code ? 1 : code == 1 ? ref_code : code for code in codes]
    dummy = Float64[index == entry.const_.level ? 1.0 : 0.0 for index in indices]
    fit = (; levels, level=entry.const_.level,
           n_levels=entry.const_.n_levels, ref=entry.const_.ref)
    _BRMPreprocessResult((primary=dummy,), _BRMPreprocEntry(
        :population_factor_dummy, fit, entry.raw_ref, true))
end

function _brm_replay_preprocess(::Val{:spline}, entry, raw, freeze, _prefix)
    values = collect(Float64, raw)
    old_fit = entry.const_.fit
    fit = freeze ? old_fit : _brm_fit_spline(values; k=old_fit.k)
    Xnull, Zpen = _brm_apply_spline(fit, values)
    updated = (; fit, zpen_key=entry.const_.zpen_key)
    _BRMPreprocessResult((primary=Xnull, penalty=Zpen),
        _BRMPreprocEntry(:spline, updated, entry.raw_ref, false))
end

function _brm_replay_preprocess(::Val{:tensor_spline}, entry, axes::Tuple,
                                freeze, prefix)
    length(axes) == 2 || error("$prefix: `t2` needs exactly two margins")
    old_fit = entry.const_.fit
    fit = freeze ? old_fit : _brm_fit_t2(axes[1], axes[2]; k=old_fit.k)
    Xfixed, Zrr, Zrn, Znr = _brm_apply_t2(fit, axes[1], axes[2])
    updated = (; fit, zrr_key=entry.const_.zrr_key,
               zrn_key=entry.const_.zrn_key, znr_key=entry.const_.znr_key)
    _BRMPreprocessResult((primary=Xfixed, rr=Zrr, rn=Zrn, nr=Znr),
        _BRMPreprocEntry(:tensor_spline, updated, entry.raw_ref, false))
end

function _brm_replay_preprocess(::Val{:gp}, entry, axes::Tuple,
                                _freeze, _prefix)
    _BRMPreprocessResult((primary=_brm_gp_matrix(axes),),
        _BRMPreprocEntry(:gp, entry.const_, entry.raw_ref, false))
end

function _brm_check_hsgp_domain(fits, axes; prefix="BRM replay")
    isnothing(fits) && return nothing
    for axis in eachindex(axes)
        center, half_width = fits[axis]
        lower, upper = center - half_width, center + half_width
        all(value -> lower <= value <= upper, axes[axis]) || error(
            "$prefix: `hsgp(...; domain=...)` axis $axis contains training " *
            "values outside its fixed domain ($lower, $upper)")
    end
    nothing
end

# The SB reprocess input is the replay axes plus, for a `rows=` term, the
# replay mask (decision 1fk9b29).
_brm_replay_preprocess(kind::Val{:hsgp}, entry, axes::Tuple, freeze, prefix) =
    _brm_replay_preprocess(kind, entry, (; axes, mask=nothing), freeze, prefix)

function _brm_replay_preprocess(::Val{:hsgp}, entry, input::NamedTuple,
                                freeze, prefix)
    (; axes, mask) = input
    covariance = get(entry.const_, :cov, :exp_quad)
    covariance === :periodic && return _brm_replay_periodic_hsgp(
        entry, axes, mask, prefix)
    K = entry.const_.K
    anchor = get(entry.const_, :anchor, nothing)
    extrapolate = get(entry.const_, :extrapolate, :error)
    domain_fits = get(entry.const_, :domain_fits, nothing)
    fitted_axes = _brm_hsgp_masked_axes(axes, mask)
    fresh = !freeze && isnothing(domain_fits)
    fits = !isnothing(domain_fits) ? domain_fits :
           freeze ? entry.const_.fits :
           _brm_fit_hsgp(fitted_axes, K, entry.const_.c; anchor)
    # A fresh fit re-derives the clamp range; a frozen or fixed-domain fit
    # keeps the one it was fitted with.
    support = fresh || !haskey(entry.const_, :support) ?
        _brm_hsgp_support(fitted_axes, fits) : entry.const_.support
    evaluated = fresh ? axes :
        _brm_hsgp_replay_axes(axes, fits, support, extrapolate, mask; prefix,
                              fixed_domain=!isnothing(domain_fits))
    orthogonal_to = get(entry.const_, :orthogonal_to, nothing)
    iso = entry.const_.iso
    basis = _brm_hsgp_basis_state(evaluated, K, :exp_quad, iso, nothing;
        fits, orthogonal=orthogonal_to, anchor, mask)
    # Only a clamping record keeps its support; others stay as they were fitted.
    updated = haskey(entry.const_, :support) ?
        merge(entry.const_, (; fits, support)) : merge(entry.const_, (; fits))
    _BRMPreprocessResult(
        (primary=basis.PHI, omega2=basis.omega2, rho_lower=basis.rho_lower),
        _BRMPreprocEntry(:hsgp, updated, entry.raw_ref, false))
end

function _brm_replay_periodic_hsgp(entry, axes::Tuple, mask, prefix)
    length(axes) == 1 || error(
        "$prefix: periodic `hsgp` expects exactly one axis")
    K = entry.const_.K
    basis = _brm_hsgp_basis_state(axes, (K,), :periodic, true,
        entry.const_.period; anchor=get(entry.const_, :anchor, nothing), mask)
    values = (primary=basis.PHI, harmonics=basis.harmonics,
              rho_lower=basis.rho_lower)
    _BRMPreprocessResult(values,
        _BRMPreprocEntry(:hsgp, entry.const_, entry.raw_ref, false))
end
