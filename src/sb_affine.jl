# Generic affine distribution composition. The transformed density, CDFs and
# RNG all use the same retained base call; no location/scale argument positions
# are guessed from a base family's arity.
StanBlocks.@deffun begin
    @lpxf brm_affine_lpdf(value::real, family, location, scale, args...)::real =
        density(family, (value - location) / scale, args...) - log(scale)
    brm_affine_lpdf(value::anything[n], family, location, scale, args...)::real =
        sum(brm_affine_lpdfs(value, family, location, scale, args...))
    brm_affine_lpdfs(value::real, family, location, scale, args...)::real =
        brm_affine_lpdf(value, family, location, scale, args...)
    brm_affine_lpdfs(value::anything[n], family, location, scale, args...)::vector[n] =
        jbroadcasted(brm_affine_lpdf, value, family, location, scale, args...)
    brm_affine_lcdf(value::real, family, location, scale, args...)::real =
        logcdf(family, (value - location) / scale, args...)
    brm_affine_lccdf(value::real, family, location, scale, args...)::real =
        logccdf(family, (value - location) / scale, args...)
    brm_affine_rng(family, location, scale, args...)::real =
        location + scale * predictive(family, args...)
    brm_affine_cell_rng(dummy, family, location, scale, args...)::real =
        brm_affine_rng(family, location, scale, args...)
    brm_affine_rng(vector[n], family, location, scale, args...)::vector[n] =
        jbroadcasted(brm_affine_cell_rng, rep_vector(0., n), family, location, scale, args...)
end

function _sb_affine_call(loc, scale, base::ExprColumn, lower_arg)
    shape = _brm_distribution_shape(base)
    isnothing(shape) || shape == (Distributions.Univariate, Distributions.Continuous) || error(
        "sbimpl: LocationScale requires a continuous univariate base distribution")
    base_constructor = getf(base)
    call = _sb_stan_distribution_call(base_constructor,
        map(lower_arg, getargs(base)), map(lower_arg, getkwargs(base)))
    any(arg -> Meta.isexpr(arg, :parameters), call.args[2:end]) && error(
        "sbimpl: an affine base translation must produce positional distribution arguments")
    base_name, args = first(call.args), Tuple(call.args[2:end])
    _sb_affine_call(base_constructor, lower_arg(loc), lower_arg(scale), args, base_name)
end
function _sb_affine_call(_constructor, loc, scale, args, base_name)
    family = base_name isa Symbol ? getfield(StanBlocks, base_name) : base_name
    Expr(:call, brm_affine, family, loc, scale, args...)
end
# Preserve the established native Student-t program and its parameter identities.
_sb_affine_call(::Type{<:TDist}, loc, scale, args, _base_name) =
    Expr(:call, :student_t, first(args), loc, scale)

function _sb_location_scale_parts(args)
    length(args) == 3 || error("sbimpl: LocationScale expects location, scale, and a base call")
    loc, scale, raw = args
    base = _as_expr_column(raw)
    isnothing(base) && error("sbimpl: LocationScale base must be a distribution call")
    scale isa Real && !(isfinite(scale) && scale > 0) && error(
        "sbimpl: LocationScale scale must be finite and strictly positive")
    loc, scale, base
end

function _sb_prior_bound_keywords(target, constructor, kwargs)
    unknown = Symbol[k for k in keys(kwargs) if !(k in (:lower, :upper))]
    isempty(unknown) || error(
        "sbimpl: `$target ~ $constructor(...)` accepts only `lower` and " *
        "`upper` declaration-bound keywords, got $unknown")
    bounds = Dict{Symbol,Any}()
    for key in (:lower, :upper)
        haskey(kwargs, key) || continue
        value = kwargs[key] isa Expr ? kwargs[key] : _sb_effect_prior_arg(kwargs[key])
        if value isa Real
            !(value isa Bool) && isfinite(value) || error(
                "sbimpl: `$target ~ $constructor(...; $key=...)` requires a " *
                "finite bound, got $(repr(value))")
            value = Float64(value)
        end
        bounds[key] = value
    end
    haskey(bounds, :lower) && haskey(bounds, :upper) &&
        bounds[:lower] isa Real && bounds[:upper] isa Real &&
        bounds[:lower] >= bounds[:upper] && error(
            "sbimpl: bounded scalar prior `$target` requires `lower < upper`, " *
            "got $(bounds[:lower]) >= $(bounds[:upper])")
    Expr(:parameters, (Expr(:kw, key, bounds[key]) for key in (:lower, :upper)
                       if haskey(bounds, key))...)
end
