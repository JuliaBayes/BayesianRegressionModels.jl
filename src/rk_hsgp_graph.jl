# Numerical HSGP source is authored graph data. Fixed domain fits are formula
# constants; automatic fits are computed from the current bound raw axes. A
# model-derived axis always has fixed fits and is read from its graph value.
# Basis and spectral arithmetic never cross an ordinary BRM helper boundary.
function _rk_ast_graph_plate(arguments, parameters, body)
    body = Meta.isexpr(body, :block) ? deepcopy(body) : Expr(:block, body)
    last = findlast(x -> !(x isa LineNumberNode), body.args)
    Meta.isexpr(body.args[last], :return) ||
        (body.args[last] = Expr(:return, body.args[last]))
    Expr(:do, Expr(:call, Expr(:., :ReactiveKernels, QuoteNode(:plate)), arguments...),
        Expr(:->, Expr(:tuple, parameters...), body))
end

# Axis inputs of a squared-exponential HSGP graph: a one-dimensional basis
# reads `axis`; per-axis suffixes, tuples and one-column frequency matrices
# appear only for a tensor basis.
_rk_hsgp_per_axis(stem, D) = D == 1 ? [Symbol(stem)] : [Symbol(stem, :_, j) for j in 1:D]
_rk_hsgp_modes(options) = options.k isa Tuple ? options.k : (options.k,)
_rk_hsgp_axis_inputs(term) = _rk_hsgp_per_axis(:axis, length(term.columns))

# A `rows=` term also reads its bound 0/1 mask (decision 1fk9b29): the formal
# follows the axes in every graph that fits or evaluates the basis.
_rk_hsgp_mask(term) = get(term.options, :mask, nothing)
_rk_hsgp_mask_formals(term) = _rk_hsgp_mask(term) === nothing ? () : (:mask,)
_rk_hsgp_mask_actuals(term) = _rk_hsgp_mask(term) === nothing ? () : (_rk_hsgp_mask(term),)

# An automatic interval reads fit-role copies of the axes (and mask) planned by
# `_rk_plan_hsgp_term!`: the axes themselves when the model is built, the
# training axes on a frozen `reprocess`. They follow the axes and mask.
_rk_hsgp_fit_axes(term) = get(term.options, :fit_axes, nothing)
_rk_hsgp_fit_axis_inputs(term) = _rk_hsgp_fit_axes(term) === nothing ?
    _rk_hsgp_axis_inputs(term) : _rk_hsgp_per_axis(:axis_fit, length(term.columns))
_rk_hsgp_fit_mask_input(term) = _rk_hsgp_fit_axes(term) === nothing ? :mask : :mask_fit
_rk_hsgp_fit_formals(term) = _rk_hsgp_fit_axes(term) === nothing ? () :
    (_rk_hsgp_fit_axis_inputs(term)...,
     (_rk_hsgp_mask(term) === nothing ? () : (:mask_fit,))...)
_rk_hsgp_fit_actuals(term) = _rk_hsgp_fit_axes(term) === nothing ? () :
    (_rk_hsgp_fit_axes(term)...,
     (_rk_hsgp_mask(term) === nothing ? () : (term.options.fit_mask,))...)

# Each axis's half-width, and its center when the basis reads it: formula
# constants for a fixed domain, otherwise computed from the current axis --
# over the masked rows only, widened to contain an anchor -- the law of
# `_brm_fit_hsgp`.
function _rk_ast_hsgp_fits!(body, options, inputs; centered=true, mask=:mask)
    D = length(inputs)
    C = options.c isa Tuple ? options.c : (options.c,)
    centers, widths = _rk_hsgp_per_axis(:center, D), _rk_hsgp_per_axis(:width, D)
    fits = options.fixed_fits
    masked = get(options, :mask, nothing) !== nothing
    anchor = get(options, :anchor, nothing)
    for j in 1:D
        x, center, width = inputs[j], centers[j], widths[j]
        (centered || fits === nothing) && push!(body.args, Expr(:(=), center,
            fits !== nothing ? fits[j][1] :
            masked ? :(sum($x .* $mask) / sum($mask)) : :(sum($x) / length($x))))
        if fits === nothing
            deviation = masked ? :(maximum(abs.($x .- $center) .* $mask)) :
                :(maximum(abs.($x .- $center)))
            anchor === nothing ||
                (deviation = :(max($deviation, abs($(anchor[j]) - $center))))
            push!(body.args, Expr(:(=), width, :($(C[j]) * $deviation)))
        else
            push!(body.args, Expr(:(=), width, fits[j][2]))
        end
    end
    centers, widths
end

# The basis matrix and squared frequencies of one basis structure (modes,
# boundary factor, fits, orthogonalization); terms with the same structure
# share the definition. Its caller destructures both results.
function _rk_ast_hsgp_basis_graph!(definitions, term, taken)
    options = term.options
    K = _rk_hsgp_modes(options)
    D, B = length(K), prod(K)
    inputs = _rk_hsgp_axis_inputs(term)
    fit_inputs = _rk_hsgp_fit_axis_inputs(term)
    fit_mask = _rk_hsgp_fit_mask_input(term)
    body = Expr(:block)
    centers, widths = _rk_ast_hsgp_fits!(body, options, fit_inputs; mask=fit_mask)
    frequencies = D == 1 ? [:omega2] : _rk_hsgp_per_axis(:frequency, D)
    for j in 1:D
        # These are the tensor basis coordinates in the same Julia/Stan
        # column-major order; no prepared numerical basis is shipped.
        modes = D == 1 ? :(1:$B) : Expr(:vect, [I[j] for I in CartesianIndices(K)]...)
        frequency = _rk_ast_graph_plate([modes], [:mode],
            :((mode * pi / (2 * $(widths[j])))^2))
        push!(body.args, Expr(:(=), frequencies[j], frequency))
    end
    # Each row cell zips the axes; its mode cells close over the row's axis
    # values and the graph's whole `omega2`, widths and centers. An anchored
    # term subtracts the basis row at its anchor; a masked one zeroes the rows
    # outside its mask (decision 1fk9b29).
    anchor = get(options, :anchor, nothing)
    masked = _rk_hsgp_mask(term) !== nothing
    if D == 1
        cell = :(sin(sqrt(omega2[b]) * (x - center + width)) / sqrt(width))
        row_values = [:x]
        row_inputs = Any[:axis]
    else
        push!(body.args, :(widths = $(Expr(:tuple, widths...))))
        push!(body.args, :(centers = $(Expr(:tuple, centers...))))
        push!(body.args, :(omega2 = hcat($(frequencies...))))
        row_values = [Symbol(:x_, j) for j in 1:D]
        cell = Expr(:call, :*, [:(sin(sqrt(omega2[b, $j]) *
            ($(row_values[j]) - centers[$j] + widths[$j])) / sqrt(widths[$j]))
            for j in 1:D]...)
        row_inputs = Any[inputs...]
    end
    # `extrapolate=:clamp` evaluates the basis at each value clamped into the
    # training data range of its axis (a no-op on the training rows).
    if get(options, :extrapolate, :error) === :clamp && _rk_hsgp_fit_axes(term) !== nothing
        clamped = Dict{Symbol,Any}()
        for j in 1:D
            lower, upper = Symbol(:support_lower_, j), Symbol(:support_upper_, j)
            x_fit = fit_inputs[j]
            push!(body.args, masked ?
                :($lower = minimum(ifelse.($fit_mask .> 0, $x_fit, Inf))) :
                :($lower = minimum($x_fit)))
            push!(body.args, masked ?
                :($upper = maximum(ifelse.($fit_mask .> 0, $x_fit, -Inf))) :
                :($upper = maximum($x_fit)))
            clamped[row_values[j]] = :(clamp($(row_values[j]), $lower, $upper))
        end
        cell = _rk_ast_hsgp_substitute(cell, clamped)
    end
    inner = _rk_ast_graph_plate([:(1:$B)], [:b], cell)
    row_value = :values
    if anchor !== nothing
        anchored = _rk_ast_hsgp_substitute(cell,
            Dict(zip(row_values, collect(Any, anchor))))
        push!(body.args, :(anchor_values = $(_rk_ast_graph_plate([:(1:$B)], [:b], anchored))))
        row_value = :(values .- anchor_values)
    end
    if masked
        push!(row_inputs, :mask)
        push!(row_values, :m)
        row_value = :(m .* $row_value)
    end
    outer = _rk_ast_graph_plate(row_inputs, row_values,
        Expr(:block, :(values = $inner), row_value))
    push!(body.args, :(basis_rows = $outer))
    if get(options, :orthogonal, nothing) === :linear
        # `orthogonal_to=:linear` (one axis, by preparation): center every
        # basis column and project out the centered axis, column by column,
        # from the current axis values — the law of SB's in-graph
        # `brm_hsgp_orthogonalize_linear`, including its degenerate-axis guard.
        x = only(inputs)
        push!(body.args, :(raw_basis = stack(basis_rows; dims=1)))
        if _rk_hsgp_fit_axes(term) === nothing
            push!(body.args, :(axis_centered = $x .- sum($x) / length($x)))
            push!(body.args, :(axis_ss = sum(axis_centered .^ 2)))
            column = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
                    phi = raw_basis[:, b]
                    centered = phi .- sum(phi) / length(phi)
                    axis_ss > 1e-12 ?
                        centered .- axis_centered .*
                            (sum(axis_centered .* centered) / axis_ss) : centered
                end))
        else
            # The column means and projections are fitted on the basis at the
            # fit-role axis and applied to the rows: the same arithmetic on
            # the training rows.
            x_fit = only(fit_inputs)
            fit_rows = _rk_ast_graph_plate(Any[x_fit], [:x],
                Expr(:block, :(values = $inner), :values))
            push!(body.args, :(fit_basis = stack($fit_rows; dims=1)))
            push!(body.args, :(axis_mean = sum($x_fit) / length($x_fit)))
            push!(body.args, :(fit_axis_centered = $x_fit .- axis_mean))
            push!(body.args, :(axis_ss = sum(fit_axis_centered .^ 2)))
            push!(body.args, :(axis_centered = $x .- axis_mean))
            column = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
                    fit_phi = fit_basis[:, b]
                    phi_mean = sum(fit_phi) / length(fit_phi)
                    fit_centered = fit_phi .- phi_mean
                    centered = raw_basis[:, b] .- phi_mean
                    axis_ss > 1e-12 ?
                        centered .- axis_centered .*
                            (sum(fit_axis_centered .* fit_centered) / axis_ss) : centered
                end))
        end
        push!(body.args, :(basis_columns = $column))
        push!(body.args, :(PHI = stack(basis_columns; dims=2)))
    elseif options.fixed_fits !== nothing && !options.latent
        # Only an explicit domain admits an axis with no rows (decision
        # 1fk9b29): its basis is then `0 x B`, which `stack` cannot size. The
        # basis reads data and formula constants only, so both arms are
        # inactive under AD.
        push!(body.args, :(PHI = isempty(basis_rows) ? zeros(0, $B) :
            stack(basis_rows; dims=1)))
    else
        push!(body.args, :(PHI = stack(basis_rows; dims=1)))
    end
    push!(body.args, Expr(:return, Expr(:tuple, :PHI, :omega2)))
    _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_basis_graph",
        [inputs; _rk_hsgp_mask_formals(term)...; _rk_hsgp_fit_formals(term)...],
        body; kernel=true)
end

# The length-scale validity floor `(4L/pi) * sqrt(log(100) / (k^2 - 1))` per
# axis (the largest for an isotropic length scale, `0.0` for a single mode),
# emitted only where a prior or hyper-predictor reads it. Automatic fits read
# the current axes through a shared graph; a fixed domain's floor is constant.
function _rk_ast_hsgp_floor!(definitions, statements, term, taken)
    options = term.options
    K = _rk_hsgp_modes(options)
    D = length(K)
    all(==(1), K) && return 0.0
    body = Expr(:block)
    fitted = _rk_hsgp_fit_axes(term) !== nothing
    inputs = _rk_hsgp_fit_axis_inputs(term)
    _, widths = _rk_ast_hsgp_fits!(body, options, inputs; centered=false,
        mask=_rk_hsgp_fit_mask_input(term))
    floors = [K[j] == 1 ? 0.0 : :((4 * $(widths[j]) / pi) *
        sqrt(log($(_BRM_HSGP_WEIGHT_THRESHOLD)) / ($(K[j])^2 - 1))) for j in 1:D]
    value = D == 1 ? only(floors) : options.iso ? Expr(:call, :max, floors...) :
        Expr(:vect, floors...)
    name = _rk_ast_fresh_name(string(options.id, "_rho_floor"), taken)
    if options.fixed_fits === nothing
        push!(body.args, :(rho_floor = $value), :(return rho_floor))
        # The floor reads only the fitted interval: the fit-role axes (and
        # mask) when the interval is fitted in-graph.
        formals = fitted ? collect(_rk_hsgp_fit_formals(term)) :
            [inputs; _rk_hsgp_mask_formals(term)...]
        actuals = fitted ? _rk_hsgp_fit_actuals(term) :
            (term.columns..., _rk_hsgp_mask_actuals(term)...)
        entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_rho_floor_graph",
            formals, body; kernel=true)
        push!(statements, Expr(:(=), name, Expr(:call, entry, actuals...)))
    else
        fits = Dict(widths[j] => options.fixed_fits[j][2] for j in 1:D)
        push!(statements, Expr(:(=), name, _rk_ast_hsgp_substitute(value, fits)))
    end
    name
end

_rk_ast_hsgp_substitute(value::Symbol, values) = get(values, value, value)
_rk_ast_hsgp_substitute(value::Expr, values) =
    Expr(value.head, (_rk_ast_hsgp_substitute(a, values) for a in value.args)...)
_rk_ast_hsgp_substitute(value, values) = value

# The spectral-weighted basis product. `inputs` pairs each graph argument
# with its value: a squared-exponential term passes its axes and the graph
# composes its basis; a periodic term passes its prepared `PHI` and `omega2`.
function _rk_ast_hsgp_value_graph!(definitions, term, taken, inputs, sigma, rho, z;
        group_index=nothing)
    options = term.options
    D = length(term.columns)
    rho_by_group = any(p -> p.hyper === :length_scale, options.hyper_plans)
    sigma_by_group = any(p -> p.hyper === :sd, options.hyper_plans)
    # A one-dimensional basis has one frequency per mode, not a one-column matrix.
    frequency(j) = D == 1 ? :(omega2[b]) : :(omega2[b, $j])
    exponent_parts = options.iso ? [:(rho^2 * $(frequency(j))) for j in 1:D] :
        [:(rho[$j]^2 * $(frequency(j))) for j in 1:D]
    exponent = D == 1 ? only(exponent_parts) : Expr(:call, :+, exponent_parts...)
    scale = options.iso ? :(sigma * (rho * sqrt(2pi))^($D / 2)) :
        Expr(:call, :*, :sigma, [:(sqrt(rho[$j] * sqrt(2pi))) for j in 1:D]...)
    weight_cell = :($scale * exp(-0.25 * $exponent))
    # Mode cells close over `omega2` and the scales, read under their names.
    weight_plate(sigma_value, rho_value) = _rk_ast_graph_plate([:(axes(omega2, 1))], [:b],
        _rk_ast_hsgp_substitute(weight_cell, Dict(:sigma => sigma_value, :rho => rho_value)))
    formals = first.(inputs)
    arguments = [formals..., :sigma, :rho, :z]
    body = Expr(:block)
    if get(options, :cov, :exp_quad) !== :periodic
        basis = _rk_ast_hsgp_basis_graph!(definitions, term, taken)
        push!(body.args, Expr(:(=), Expr(:tuple, :PHI, :omega2),
            Expr(:call, basis, formals...)))
    end
    if group_index === nothing
        # Hyper-predicted scales arrive as one-element vectors.
        sigma_value, rho_value = :sigma, :rho
        sigma_by_group && (push!(body.args, :(sigma_value = sigma[1])); sigma_value = :sigma_value)
        rho_by_group && (push!(body.args, :(rho_value = rho[1])); rho_value = :rho_value)
        push!(body.args, :(weights = $(weight_plate(sigma_value, rho_value))),
            :(value = PHI * (weights .* z)), :(return value))
    else
        push!(arguments, :group_index)
        if !sigma_by_group && !rho_by_group
            push!(body.args, :(weights = $(weight_plate(:sigma, :rho))),
                :(scaled_z = z .* transpose(weights)))
        else
            frequencies = D == 1 ? :(transpose(omega2)) : :(transpose(vec(sum(omega2; dims=2))))
            push!(body.args, :(frequencies = $frequencies),
                :(scale = sigma .* (rho .* sqrt(2pi)).^($D / 2)),
                :(exponent = (rho .^ 2) .* frequencies),
                :(spectra = scale .* exp.(-0.25 .* exponent)),
                :(scaled_z = z .* spectra))
        end
        # The same row-wise contraction as the scalar sum: gather the
        # original group's mode weights, multiply by the row's basis,
        # then reduce only the mode axis.
        push!(body.args, :(rows = scaled_z[group_index,:]),
            :(value = vec(sum(PHI .* rows; dims=2))), Expr(:return,:value))
    end
    # Terms with the same spectral shape share one explicit numerical graph.
    entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_spectral_graph",
        arguments, body; kernel=true)
    values = [last.(inputs)..., sigma, rho, z]
    group_index === nothing || push!(values, group_index)
    Expr(:call, entry, values...)
end
