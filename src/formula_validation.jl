_brm_interval_literal(x::Real) = Float64(x)
_brm_interval_literal(x::NamedColumn) = begin
    parent(x) isa MissingColumn && name(x) in (:pi, :π) || error(
        "CircularVonMises: `interval` must be a compile-time numeric pair; " *
        "got formula symbol `$(name(x))`")
    Float64(pi)
end
function _brm_interval_literal(x::ExprColumn)
    args = getargs(x)
    length(args) == 1 && getf(x) === (-) && return -_brm_interval_literal(args[1])
    length(args) == 1 && getf(x) === (+) && return _brm_interval_literal(args[1])
    error("CircularVonMises: `interval` must be a compile-time numeric pair; got $(x)")
end
_brm_interval_literal(x) = error(
    "CircularVonMises: `interval` must be a compile-time numeric pair; got $(typeof(x))")

function _brm_circular_interval(kwargs)
    interval = get(kwargs, :interval, (-Float64(pi), Float64(pi)))
    interval isa Tuple && length(interval) == 2 || error(
        "CircularVonMises: `interval` must be a 2-tuple `(lo, hi)`, got $(repr(interval))")
    lo, hi = map(_brm_interval_literal, interval)
    all(isfinite, (lo, hi)) && lo < hi || error(
        "CircularVonMises: `interval` endpoints must be finite with lo < hi, got $(repr((lo, hi)))")
    isapprox(hi - lo, 2 * Float64(pi); rtol=8eps(Float64), atol=8eps(Float64)) || error(
        "CircularVonMises: `interval` must have length 2pi, got $(hi - lo)")
    (lo, hi)
end

function _check_term_kwargs(::Type{<:CircularVonMises}, kwargs)
    unknown = filter(!=(:interval), keys(kwargs))
    isempty(unknown) || error(
        "CircularVonMises: unsupported keyword(s): $(join(unknown, ", ")); " *
        "the only supported keyword is `interval`")
    _brm_circular_interval(kwargs)
    nothing
end

function _check_term_kwargs(::Type{<:Ordinal}, kwargs)
    allowed = (:discrimination, :per_threshold)
    unknown = filter(k -> k ∉ allowed, keys(kwargs))
    isempty(unknown) || error(
        "Ordinal: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `discrimination` and `per_threshold`")
    nothing
end

function _check_term_kwargs(::typeof(gp), kw)
    allowed = (:cov, :iso, :jitter, :period)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "gp: exact GP accepts only `cov`, `iso`, `jitter`, and `period`; unsupported keyword(s): $(join(unknown, ", ")). " *
        "Use `hsgp(...; k=..., c=..., by=...)` for the Hilbert-space approximation.")
    cov = _brm_gp_cov(kw, :gp)
    _brm_gp_period(kw, :gp, cov)
    _brm_gp_iso(kw, :gp)
    jitter = get(kw, :jitter, 1e-9)
    jitter isa Real && isfinite(jitter) && jitter > 0 || error(
        "gp: `jitter` must be a finite positive real, got $(repr(jitter))")
    nothing
end

function _check_term_kwargs(::typeof(hsgp), kw)
    allowed = (:cov, :iso, :k, :c, :by, :domain, :orthogonal_to, :period)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "hsgp: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `cov`, `iso`, `k`, `c`, `by`, `domain`, " *
        "`orthogonal_to`, and `period`")
    cov = _brm_gp_cov(kw, :hsgp)
    _brm_gp_period(kw, :hsgp, cov)
    _brm_gp_iso(kw, :hsgp)
    nothing
end

function _check_term_kwargs(::typeof(t2), kw)
    allowed = (:k, :basis, :full)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "t2: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `k`, `basis`, and `full`")
    _brm_t2_options(kw)
    nothing
end
