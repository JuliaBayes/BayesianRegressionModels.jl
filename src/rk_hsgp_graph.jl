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

# Each axis's half-width, and its center when the basis reads it: formula
# constants for a fixed domain, otherwise computed from the current axis.
function _rk_ast_hsgp_fits!(body, options, inputs; centered=true)
    D = length(inputs)
    C = options.c isa Tuple ? options.c : (options.c,)
    centers, widths = _rk_hsgp_per_axis(:center, D), _rk_hsgp_per_axis(:width, D)
    fits = options.fixed_fits
    for j in 1:D
        x, center, width = inputs[j], centers[j], widths[j]
        (centered || fits === nothing) && push!(body.args, Expr(:(=), center,
            fits === nothing ? :(sum($x) / length($x)) : fits[j][1]))
        push!(body.args, Expr(:(=), width, fits === nothing ?
            :($(C[j]) * maximum(abs.($x .- $center))) : fits[j][2]))
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
    body = Expr(:block)
    centers, widths = _rk_ast_hsgp_fits!(body, options, inputs)
    frequencies = D == 1 ? [:omega2] : _rk_hsgp_per_axis(:frequency, D)
    for j in 1:D
        # These are the tensor basis coordinates in the same Julia/Stan
        # column-major order; no prepared numerical basis is shipped.
        modes = D == 1 ? :(1:$B) : Expr(:vect, [I[j] for I in CartesianIndices(K)]...)
        frequency = _rk_ast_graph_plate([modes, :(Ref($(widths[j])))], [:mode, :width],
            :((mode * pi / (2 * width))^2))
        push!(body.args, Expr(:(=), frequencies[j], frequency))
    end
    if D == 1
        cell = :(sin(sqrt(omega2[b]) * (x - center + width)) / sqrt(width))
        inner = _rk_ast_graph_plate([:(1:$B), :(Ref(x)), :(Ref(omega2)), :(Ref(width)),
            :(Ref(center))], [:b, :x, :omega2, :width, :center], cell)
        outer = _rk_ast_graph_plate([:axis, :(Ref(omega2)), :(Ref(width)), :(Ref(center))],
            [:x, :omega2, :width, :center], Expr(:block, :(values = $inner), :values))
    else
        push!(body.args, :(widths = $(Expr(:tuple, widths...))))
        push!(body.args, :(centers = $(Expr(:tuple, centers...))))
        push!(body.args, :(omega2 = hcat($(frequencies...))))
        row_values = [Symbol(:x_, j) for j in 1:D]
        factors = [:(sin(sqrt(frequencies[b, $j]) *
            (xs[$j] - centers[$j] + widths[$j])) / sqrt(widths[$j])) for j in 1:D]
        inner = _rk_ast_graph_plate([:(1:$B), :(Ref(xs)), :(Ref(frequencies)),
            :(Ref(widths)), :(Ref(centers))], [:b, :xs, :frequencies, :widths, :centers],
            Expr(:call, :*, factors...))
        outer = _rk_ast_graph_plate([inputs..., :(Ref(omega2)), :(Ref(widths)),
                :(Ref(centers))], [row_values..., :frequencies, :widths, :centers],
            Expr(:block, :(xs = $(Expr(:tuple, row_values...))), :(values = $inner), :values))
    end
    push!(body.args, :(basis_rows = $outer))
    if get(options, :orthogonal, nothing) === :linear
        # `orthogonal_to=:linear` (one axis, by preparation): center every
        # basis column and project out the centered axis, column by column,
        # from the current axis values — the law of SB's in-graph
        # `brm_hsgp_orthogonalize_linear`, including its degenerate-axis guard.
        x = only(inputs)
        push!(body.args, :(raw_basis = stack(basis_rows; dims=1)))
        push!(body.args, :(axis_centered = $x .- sum($x) / length($x)))
        push!(body.args, :(axis_ss = sum(axis_centered .^ 2)))
        column = _rk_ast_graph_plate([:(1:$B), :(Ref(raw_basis)),
                :(Ref(axis_centered)), :(Ref(axis_ss))],
            [:b, :raw_basis, :axis_centered, :axis_ss], Base.remove_linenums!(quote
                phi = raw_basis[:, b]
                centered = phi .- sum(phi) / length(phi)
                axis_ss > 1e-12 ?
                    centered .- axis_centered .*
                        (sum(axis_centered .* centered) / axis_ss) : centered
            end))
        push!(body.args, :(basis_columns = $column))
        push!(body.args, :(PHI = stack(basis_columns; dims=2)))
    else
        push!(body.args, :(PHI = stack(basis_rows; dims=1)))
    end
    push!(body.args, Expr(:return, Expr(:tuple, :PHI, :omega2)))
    _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_basis_graph", inputs, body;
        kernel=true)
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
    inputs = _rk_hsgp_axis_inputs(term)
    _, widths = _rk_ast_hsgp_fits!(body, options, inputs; centered=false)
    floors = [K[j] == 1 ? 0.0 : :((4 * $(widths[j]) / pi) *
        sqrt(log($(_BRM_HSGP_WEIGHT_THRESHOLD)) / ($(K[j])^2 - 1))) for j in 1:D]
    value = D == 1 ? only(floors) : options.iso ? Expr(:call, :max, floors...) :
        Expr(:vect, floors...)
    name = _rk_ast_fresh_name(string(options.id, "_rho_floor"), taken)
    if options.fixed_fits === nothing
        push!(body.args, :(rho_floor = $value), :(return rho_floor))
        entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_rho_floor_graph",
            inputs, body; kernel=true)
        push!(statements, Expr(:(=), name, Expr(:call, entry, term.columns...)))
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
    weight_plate(sigma_value, rho_value) = _rk_ast_graph_plate([:(axes(omega2, 1)),
            :(Ref(omega2)), :(Ref($sigma_value)), :(Ref($rho_value))],
        [:b, :omega2, :sigma, :rho], weight_cell)
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
