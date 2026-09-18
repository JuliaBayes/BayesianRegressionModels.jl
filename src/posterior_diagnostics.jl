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
    hsgp_boundary_check(d::BRMDescriptor, draws::AbstractMatrix, names;
                        predictor::Symbol, term::Symbol)

Post-fit verdict on whether a fitted `hsgp` term's length scale lives where
the Hilbert-space approximation still represents the kernel. An HSGP over a
domain of half-width `L` fitted on data extending to `extent` keeps a margin
`L - extent` past the data; once posterior `rho` approaches that margin the
prior covariance silently distorts (boundary eats edge-point variance), while
the model still transpiles, samples, and returns finite draws. This check
reports the distortion ratio per group and flags at `1.0` — it reports, never
fails: a flagged term fitted fine, it just may not be the GP you asked for.

Returns `(; predictor, term, n_groups, margin, floor, mean_rho, ratios,
flagged, floor_binding)`:

- `margin` — the fitted past-data margin `(c-1)/c * L`, from the frozen basis
  fit (for an auto-fitted domain `L = c * extent`, so this is exact);
- `mean_rho` / `ratios` — per-group posterior-mean `rho` and
  `mean_rho / margin` (`n_groups == 1` for a bare term, even a grouped one,
  whose hyper value is scalar-shared; one entry per group for a
  hyper-driven term, with `rho` reconstructed as `exp(eta)` from the
  hyper-LP coefficients, never sliced from a transformed parameter);
- `flagged` — whether any ratio reaches `1.0`;
- `floor_binding` — per-group posterior probability the hyper value sits
  below the approximation-validity floor (`mean(rho < floor)`): structural
  `0.0` for a default-floor bare term, measured otherwise. In particular a
  hyper-driven term's sub-floor mass is likelihood-flat prior reported as
  posterior — this number says how much.

`draws` is draws × coordinates with `names` its constrained axis (the same
orientation every resolver here takes); carriers resolve through
[`brm_term_coordinates`](@ref), so a response-free prior program works with
`include_gq=true` names exactly as a fit does.

v1 covers one-dimensional auto-fitted `exp_quad` terms, grouped or not, bare
or hyper-driven. Periodic, multi-axis, anisotropic, model-derived-axis, and
explicit-`domain=` terms refuse loudly: their boundary geometry needs its own
margin rule, and a wrong margin here would be worse than none.
"""
function hsgp_boundary_check(d::BRMDescriptor, draws::AbstractMatrix, names;
                             predictor::Symbol, term::Symbol)
    _brm_check_draw_names(draws, names)
    entries = [e for e in _brm_term_coordinate_entries(d.plan.parent, predictor)
               if e.term === term]
    length(entries) == 1 || error(
        "HSGP boundary check: term `$term` occurs $(length(entries)) times " *
        "on logical predictor `$predictor`; expected exactly one.")
    entry = only(entries)
    getf(entry.value) === hsgp || error(
        "HSGP boundary check: term `$term` is not an `hsgp` term")
    t = entry.value
    kw = getkwargs(t)
    _brm_gp_cov(kw, :hsgp) === :exp_quad || error(
        "HSGP boundary check: term `$term` uses a non-`exp_quad` covariance; " *
        "v1 covers `exp_quad` only")
    length(getargs(t)) == 1 || error(
        "HSGP boundary check: term `$term` spans " *
        "$(length(getargs(t))) axes; v1 covers one-dimensional terms only")
    _sb_gp_iso(kw, :hsgp) || error(
        "HSGP boundary check: term `$term` is anisotropic; v1 covers " *
        "isotropic terms only")
    for a in getargs(t)
        inner = _sb_named_inner(:hsgp, a)
        (inner isa NamedColumn && parent(inner) isa DataColumn) || error(
            "HSGP boundary check: term `$term` has a model-derived axis; " *
            "v1 covers raw-data axes only")
    end
    # Owner join, mirroring `brm_term_coordinates`: formula term to logical
    # term output to owning declaration.
    link = entry.link
    emitted_lp = _sb_lp_emitted_name(predictor, link)
    owner_labels = _brm_term_owner_labels(getf(t), t, emitted_lp)
    owners = BRMOutput[]
    for label in owner_labels
        append!(owners, BRMOutput[
            o for o in d.outputs
            if o.logical === label && !isnothing(o.declaration) &&
               _brm_term_owner_matches(d.plan, getf(t), t, o)
        ])
        isempty(owners) || break
    end
    length(owners) == 1 || error(
        "HSGP boundary check: term `$term` has $(length(owners)) logical " *
        "output owners; expected exactly one.")
    owner = only(owners).declaration
    # The latent/model-derived shape carries an `x` keyword instead of a `PHI`
    # binding (see `_brm_term_owner_matches`); without frozen fits there is no
    # margin to check against.
    phi = get(owner.keywords, :PHI, nothing)
    phi isa Symbol || error(
        "HSGP boundary check: term `$term` has no frozen basis fit; v1 " *
        "covers raw-data auto-fitted terms only")
    preproc = get(d.plan.preproc, phi, nothing)
    if isnothing(preproc) || preproc.kind !== :hsgp
        error("HSGP boundary check: term `$term` resolves to no fitted " *
              "`:hsgp` preprocessing record at `$phi`. Re-reflect the model " *
              "that produced the posterior draws.")
    end
    const_ = preproc.const_
    get(const_, :cov, :exp_quad) === :exp_quad || error(
        "HSGP boundary check: term `$term` is periodic; v1 covers " *
        "`exp_quad` only")
    const_.iso || error(
        "HSGP boundary check: term `$term` is anisotropic; v1 covers " *
        "isotropic terms only")
    length(const_.fits) == 1 || error(
        "HSGP boundary check: term `$term` spans " *
        "$(length(const_.fits)) fitted axes; v1 covers one-dimensional " *
        "terms only")
    isnothing(const_.domain_fits) || error(
        "HSGP boundary check: term `$term` uses an explicit `domain=`; the " *
        "boundary margin needs the fitted data extent, which v1 does not " *
        "retain — auto-fitted domains only")
    _, L = only(const_.fits)
    c = only(const_.c)
    (c isa Real && c > 1 && L > 0) || error(
        "HSGP boundary check: term `$term` has an unusable frozen fit " *
        "(c=$(repr(c)), L=$(repr(L))).")
    margin = (c - 1) / c * L
    floor_key = const_.rho_lower_key
    floor = get(d.plan.data, floor_key, nothing)
    floor isa Real || error(
        "HSGP boundary check: term `$term` has no validity floor at " *
        "`$floor_key`. Re-reflect the model that produced the posterior draws.")
    # Hyper values reconstruct from coefficients (the B3 roles), never from a
    # transformed-parameter carrier — the same eta the emitter builds.
    plans = _brm_term_hyper_plans(d.plan, predictor, t)
    rho_plan = _sb_hyper_plan_for(plans, :length_scale)
    rho_draws = if isnothing(rho_plan)
        rho = brm_term_coordinates(d, predictor, names; term,
                                   parameter=:length_scale)
        length(rho.coordinates) == 1 || error(
            "HSGP boundary check: term `$term` resolves `:length_scale` to " *
            "$(length(rho.coordinates)) coordinates; expected one.")
        draws_mat = Matrix(@view draws[:, rho.coordinates])
        all(x -> isfinite(x) && x > 0, draws_mat) || error(
            "HSGP boundary check: term `$term` needs finite positive " *
            "constrained length scales")
        draws_mat
    else
        _brm_hyper_eta_rho_draws(d, draws, names, predictor, term, rho_plan)
    end
    n_groups = size(rho_draws, 2)
    mean_rho = vec(sum(rho_draws; dims=1) ./ size(rho_draws, 1))
    ratios = mean_rho ./ margin
    binding = vec(sum(rho_draws .< floor; dims=1) ./ size(rho_draws, 1))
    (; predictor, term, n_groups, margin, floor,
       mean_rho, ratios, flagged=any(ratios .>= 1.0), floor_binding=binding)
end

# Per-draw per-group hyper values from the sampled hyper-LP coefficients,
# following the emitter's eta construction (`_sb_hyper_param_stmts!`): full
# `beta + sd*z` when a ranef is present, the bare intercept when it is not.
# B1 guarantees at least one of the two.
function _brm_hyper_eta_rho_draws(d, draws, names, predictor, term, plan)
    at(role) = brm_term_coordinates(d, predictor, names; term,
                                    parameter=role).coordinates
    if isempty(plan.ranefs)
        beta = vec(Matrix(@view draws[:, at(:length_scale_intercept)]))
        all(isfinite, beta) || error(
            "HSGP boundary check: term `$term` needs finite hyper-LP " *
            "intercept draws")
        return reshape(exp.(beta), :, 1)
    end
    z = Matrix(@view draws[:, at(:length_scale_ranef_z)])
    sd = vec(Matrix(@view draws[:, at(:length_scale_ranef_sd)]))
    all(isfinite, z) && all(isfinite, sd) && all(>(0), sd) || error(
        "HSGP boundary check: term `$term` needs finite hyper-LP " *
        "deviation draws and a finite positive group scale")
    if plan.intercept
        beta = vec(Matrix(@view draws[:, at(:length_scale_intercept)]))
        all(isfinite, beta) || error(
            "HSGP boundary check: term `$term` needs finite hyper-LP " *
            "intercept draws")
        return exp.(beta .+ sd .* z)
    end
    exp.(sd .* z)
end

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
