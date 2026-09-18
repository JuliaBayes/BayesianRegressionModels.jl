# Post-fit data preparation belongs beside the semantic descriptor. Plotting
# consumers should not reconstruct carrier names, spectral priors, or links.

function _brm_check_draw_names(draws::AbstractMatrix, names)
    size(draws, 2) == length(names) || throw(DimensionMismatch(
        "BRM diagnostics expects draws × coordinates; $(size(draws, 2)) columns " *
        "do not match $(length(names)) coordinate names"))
    length(unique(names)) == length(names) || error(
        "BRM diagnostics requires unique coordinate names")
    size(draws, 1) > 0 || error("BRM diagnostics requires at least one draw")
    nothing
end

"""
    brm_output_draws(descriptor, constrained_draws, constrained_names;
                     logical, role=nothing)

Extract a model's logical output as a draws × output-elements matrix. Input
rows are draws, not the coordinates × draws orientation returned by WarmupHMC.
Obtain `constrained_names` and values with matching transformed-parameter /
generated-quantity settings. Resolution uses [`brm_output_coordinates`](@ref),
so this works for named predictors, parameters, and predictive outputs without
parsing emitted names. It does not simulate or transform the supplied values.
"""
function brm_output_draws(d::BRMDescriptor, draws::AbstractMatrix, names;
                          logical::Symbol, role=nothing)
    _brm_check_draw_names(draws, names)
    indices = brm_output_coordinates(d, logical, names; role)
    Matrix(@view draws[:, indices])
end

"""
    brm_predictive_draws(descriptor, unconstrained_draws; problem, seed)

Run the descriptor's native predictive operation once per posterior draw and
return one draws × response-elements matrix per logical response. Input rows
are draws. This simulates new response values conditional on fitted parameter
draws; it does not fit a model. `seed` is passed to the first draw, with successive
integer seeds for subsequent draws. Multiple responses share each execution.
"""
function brm_predictive_draws(d::BRMDescriptor, draws::AbstractMatrix;
                              problem, seed::Integer)
    size(draws, 1) > 0 || error("BRM prediction requires at least one draw")
    outputs = brm_outputs(d; role=:posterior_predictive)
    isempty(outputs) && error("This BRM descriptor has no predictive outputs")
    logical = [isnothing(o.logical) ? o.name : o.logical for o in outputs]
    length(unique(logical)) == length(logical) || error(
        "BRM prediction has ambiguous logical response names")
    values = [brm_execute(d, :predict; problem, draws=collect(q),
                         seed=Base.checked_add(seed, i - 1))
              for (i, q) in enumerate(eachrow(draws))]
    matrices = map(outputs) do output
        rows = [vec(collect(_brm_diagnostic_vector(getproperty(v, output.name))))
                for v in values]
        all(row -> length(row) == length(first(rows)), rows) || error(
            "Predictive output $(output.name) changed shape between draws")
        permutedims(reduce(hcat, rows))
    end
    NamedTuple{Tuple(logical)}(Tuple(matrices))
end

_brm_diagnostic_vector(x::Number) = (x,)
_brm_diagnostic_vector(x::AbstractArray) = x

"""
    hsgp_coordinate_draws(descriptor, constrained_draws, constrained_names;
                          predictor, term, centeredness=nothing,
                          basis_gradients=nothing)

Read one ungrouped squared-exponential HSGP from its descriptor, returning
draws × basis matrices `coordinates`, `sampled_coordinates`, `unit_weights`,
`physical_weights`, and `log_scales`, plus physical `length_scales` (draws ×
axes), `marginal_sd`, and sampled/display `centeredness` vectors. Input rows are
draws. `centeredness=nothing` preserves the fitted coordinates; a scalar or
per-basis vector transforms the SAME draws to `u = s^c z`. No new fit is run.

When provided, `basis_gradients` must be draws × basis gradients of the **log
density with respect to the sampled basis coordinates**, at these same draws.
The returned `gradients` are with respect to the displayed coordinates, holding
hyperparameters fixed. This is not a transport of hyperparameter gradients or
a warmup loss history. `finite` records representable displayed position/gradient
pairs; non-finite values are never silently discarded.

WarmupHMC's finalized online draws are already in the original model frame:
constrain those draws normally, and pass the learned centeredness only as the
display transformation. Do not apply a second sampler back-transform.
"""
function hsgp_coordinate_draws(d::BRMDescriptor, draws::AbstractMatrix, names;
                               predictor::Symbol, term::Symbol,
                               centeredness=nothing, basis_gradients=nothing)
    _brm_check_draw_names(draws, names)
    entry = only(e for e in _brm_term_coordinate_entries(d.plan.parent, predictor)
                 if e.term === term)
    getf(entry.value) === hsgp || error("$term is not an HSGP term")
    kw = getkwargs(entry.value)
    !haskey(kw, :by) && _brm_gp_cov(kw, :hsgp) === :exp_quad || error(
        "HSGP coordinate diagnostics currently require an ungrouped exp_quad term")
    weights = brm_term_coordinates(d, predictor, names; term, parameter=:basis_weights)
    owner = weights.output.declaration
    rho = _adaptive_same_hsgp_owner(brm_term_coordinates(
        d, predictor, names; term, parameter=:length_scale), owner, "length scale")
    sigma = _adaptive_same_hsgp_owner(brm_term_coordinates(
        d, predictor, names; term, parameter=:sd), owner, "marginal SD")
    sampled = Matrix(@view draws[:, weights.coordinates])
    rhos = Matrix(@view draws[:, rho.coordinates])
    sigmas = collect(@view draws[:, only(sigma.coordinates)])
    all(x -> isfinite(x) && x > 0, rhos) &&
        all(x -> isfinite(x) && x > 0, sigmas) || error(
        "HSGP coordinate diagnostics need finite positive constrained hyperparameters")
    all(isfinite, sampled) || error("HSGP sampled coordinates must be finite")
    omega_key = get(owner.keywords, :omega2, nothing)
    omega_key isa Symbol || error("HSGP declaration has no spectral-frequency binding")
    omega2 = d.plan.data[omega_key]
    size(omega2) == (size(sampled, 2), size(rhos, 2)) || throw(DimensionMismatch(
        "HSGP spectral metadata does not match its descriptor coordinates"))
    source = _brm_hsgp_centeredness(kw, size(sampled, 2))
    target = isnothing(centeredness) ? copy(source) :
        _brm_hsgp_centeredness((; centeredness), size(sampled, 2))
    logs = [log(sigmas[i]) + sum(
                0.5 * (log(rhos[i, a]) + 0.9189385332046727) -
                0.25 * rhos[i, a]^2 * omega2[b, a] for a in axes(rhos, 2))
            for i in axes(sampled, 1), b in axes(sampled, 2)]
    transformed = hsgp_transform_draws(sampled, logs; from=source, to=target,
                                       gradients=basis_gradients)
    (; predictor, term, transformed.coordinates, sampled_coordinates=sampled,
       unit_weights=hsgp_transform_draws(sampled, logs; from=source, to=0.0).coordinates,
       physical_weights=hsgp_transform_draws(sampled, logs; from=source, to=1.0).coordinates,
       log_scales=logs, length_scales=rhos, marginal_sd=sigmas,
       sampled_centeredness=source, centeredness=target,
       transformed.gradients, transformed.finite)
end

_brm_hsgp_rescale(value, log_scale, exponent) =
    iszero(exponent) ? value : value * exp(exponent * log_scale)

"""
    hsgp_transform_draws(coordinates, log_scales; from=0.0, to, gradients=nothing)

Transform saved HSGP basis coordinates between centering frames without a model
handle. Both matrices have draws × basis shape. `from` and `to` are scalar or
per-basis centeredness values. Optional gradients are with respect to the input
basis coordinates at fixed hyperparameters; returned gradients are with respect
to the output coordinates. Returns `coordinates`, `gradients`, and a `finite`
mask. This is a post-fit change of coordinates, not new sampling or loss fitting.
"""
function hsgp_transform_draws(coordinates::AbstractMatrix, log_scales::AbstractMatrix;
                               from=0.0, to, gradients=nothing)
    size(coordinates) == size(log_scales) || throw(DimensionMismatch(
        "HSGP coordinates and log scales must have matching draws × basis shape"))
    all(isfinite, coordinates) || error("HSGP input coordinates must be finite")
    source = _brm_hsgp_centeredness((; centeredness=from), size(coordinates, 2))
    target = _brm_hsgp_centeredness((; centeredness=to), size(coordinates, 2))
    delta = target .- source
    result = [_brm_hsgp_rescale(coordinates[i, b], log_scales[i, b], delta[b])
              for i in axes(coordinates, 1), b in axes(coordinates, 2)]
    transformed_gradients = if isnothing(gradients)
        nothing
    else
        size(gradients) == size(coordinates) || throw(DimensionMismatch(
            "HSGP gradients must match the coordinates' draws × basis shape"))
        all(isfinite, gradients) || error("Input basis gradients must be finite")
        [_brm_hsgp_rescale(gradients[i, b], log_scales[i, b], -delta[b])
         for i in axes(coordinates, 1), b in axes(coordinates, 2)]
    end
    finite = isfinite.(result)
    isnothing(transformed_gradients) || (finite .&= isfinite.(transformed_gradients))
    (; coordinates=result, gradients=transformed_gradients, finite)
end

# Methods load only with AlgebraOfVega. In particular, ordinary BRM fitting
# never acquires a Makie or HTMXObjects dependency through these entry points.
function brm_posteriorplot end
function brm_ppcplot end
function brm_pairplot end
function brm_centerednessplot end
function brm_centering_lossplot end
function brm_gradientplot end
function brm_predictionsplot end
function brm_comparisonsplot end
function brm_slopesplot end
