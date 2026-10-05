# Numerical HSGP source is authored graph data. Fixed domain fits are formula
# constants; automatic fits are computed from the current bound raw axes. A
# model-derived axis always has fixed fits and is read from its graph value.
# Basis and spectral arithmetic never cross an ordinary BRM helper boundary.
function _rk_ast_graph_definition(name, arguments, body)
    Expr(:macrocall, Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
        LineNumberNode(0), Expr(:(=), Expr(:call, name, arguments...), body))
end

function _rk_ast_graph_plate(arguments, parameters, body)
    body = Meta.isexpr(body, :block) ? deepcopy(body) : Expr(:block, body)
    last = findlast(x -> !(x isa LineNumberNode), body.args)
    Meta.isexpr(body.args[last], :return) ||
        (body.args[last] = Expr(:return, body.args[last]))
    Expr(:do, Expr(:call, Expr(:., :ReactiveKernels, QuoteNode(:plate)), arguments...),
        Expr(:->, Expr(:tuple, parameters...), body))
end

function _rk_ast_hsgp_basis_graph!(definitions, term, taken, PHI, omega2, floor)
    options = term.options
    K = options.k isa Tuple ? options.k : (options.k,)
    C = options.c isa Tuple ? options.c : (options.c,)
    D, B = length(K), prod(K)
    entry = _rk_ast_fresh_name(string(options.id, "_basis_graph"), taken)
    inputs = [Symbol(:axis_, j) for j in 1:D]
    centers = [Symbol(:center_, j) for j in 1:D]
    widths = [Symbol(:width_, j) for j in 1:D]
    frequencies = [Symbol(:frequency_, j) for j in 1:D]
    floors = [Symbol(:floor_, j) for j in 1:D]
    body = Expr(:block)
    fits = options.fixed_fits
    for j in 1:D
        x, center, width = inputs[j], centers[j], widths[j]
        push!(body.args, Expr(:(=), center, fits === nothing ?
            :(sum($x) / length($x)) : fits[j][1]))
        push!(body.args, Expr(:(=), width, fits === nothing ?
            :($(C[j]) * maximum(abs.($x .- $center))) : fits[j][2]))
        # These are the tensor basis coordinates in the same Julia/Stan
        # column-major order; no prepared numerical basis is shipped.
        modes = Expr(:vect, [I[j] for I in CartesianIndices(K)]...)
        frequency = _rk_ast_graph_plate([modes, :(Ref($width))], [:mode, :width],
            :((mode * pi / (2 * width))^2))
        push!(body.args, Expr(:(=), frequencies[j], frequency))
        lower = K[j] == 1 ? 0.0 :
            :((4 * $width / pi) * sqrt(log($(_BRM_HSGP_WEIGHT_THRESHOLD)) / ($(K[j])^2 - 1)))
        push!(body.args, Expr(:(=), floors[j], lower))
    end
    push!(body.args, :(widths = $(Expr(:tuple, widths...))))
    push!(body.args, :(centers = $(Expr(:tuple, centers...))))
    push!(body.args, :(omega2 = hcat($(frequencies...))))
    row_values = [Symbol(:x_, j) for j in 1:D]
    factors = [:(sin(sqrt(frequencies[b, $j]) *
        (xs[$j] - centers[$j] + widths[$j])) / sqrt(widths[$j])) for j in 1:D]
    product = length(factors) == 1 ? only(factors) : Expr(:call, :*, factors...)
    inner = _rk_ast_graph_plate([:(1:$B), :(Ref(xs)), :(Ref(frequencies)),
        :(Ref(widths)), :(Ref(centers))], [:b, :xs, :frequencies, :widths, :centers], product)
    outer = _rk_ast_graph_plate([inputs... , :(Ref(omega2)), :(Ref(widths)), :(Ref(centers))],
        [row_values..., :frequencies, :widths, :centers],
        Expr(:block, :(xs = $(Expr(:tuple, row_values...))), :(values = $inner), :values))
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
    push!(body.args, :(rho_floor = $(options.iso ?
        Expr(:call, :max, floors...) : Expr(:vect, floors...))))
    push!(body.args, :(basis_matrix() = PHI))
    push!(body.args, :(squared_frequencies() = omega2))
    push!(body.args, :(length_scale_floor() = rho_floor))
    push!(definitions, _rk_ast_graph_definition(entry, inputs, body))
    # The PPL definition surface accepts a positional function-shaped graph.
    # Keep object endpoint syntax inside that explicit graph adapter.
    stmts = Expr[]
    for (target, method) in ((PHI, :basis_matrix),
            (omega2, :squared_frequencies), (floor, :length_scale_floor))
        reader = _rk_ast_fresh_name(string(entry, "_", method), taken)
        owner = Expr(:call, entry, inputs...)
        value = Expr(:call, Expr(:., owner, QuoteNode(method)))
        push!(definitions, _rk_ast_graph_definition(reader, inputs,
            Expr(:block, Expr(:(=), :value, value), Expr(:return, :value))))
        push!(stmts, Expr(:(=), target, Expr(:call, reader, term.columns...)))
    end
    stmts
end

function _rk_ast_hsgp_value_graph!(definitions, term, taken,
        PHI, omega2, sigma, rho, z; group_index=nothing)
    options = term.options
    D = length(term.columns)
    entry = _rk_ast_fresh_name(string(options.id, "_spectral_graph"), taken)
    rho_by_group = any(p -> p.hyper === :length_scale, options.hyper_plans)
    sigma_by_group = any(p -> p.hyper === :sd, options.hyper_plans)
    cell_rho = group_index === nothing && rho_by_group ? :(rho[1]) : :rho
    cell_sigma = group_index === nothing && sigma_by_group ? :(sigma[1]) : :sigma
    exponent_parts = [:(r^2 * omega2[b, $j]) for j in 1:D]
    options.iso || (exponent_parts = [:(r[$j]^2 * omega2[b, $j]) for j in 1:D])
    exponent = D == 1 ? only(exponent_parts) : Expr(:call, :+, exponent_parts...)
    scale = options.iso ? :(s * (r * sqrt(2pi))^($D / 2)) :
        Expr(:call, :*, :s, [:(sqrt(r[$j] * sqrt(2pi))) for j in 1:D]...)
    weight_cell = :($scale * exp(-0.25 * $exponent))
    weight_plate = _rk_ast_graph_plate([:(axes(omega2,1)), :(Ref(omega2)), :(Ref(s)), :(Ref(r))],
        [:b, :omega2, :s, :r], weight_cell)
    arguments = [:PHI, :omega2, :sigma, :rho, :z]
    if group_index === nothing
        body = quote
            s = $cell_sigma
            r = $cell_rho
            weights = $weight_plate
            value = PHI * (weights .* z)
            return value
        end
    else
        push!(arguments, :group_index)
        body = if !sigma_by_group && !rho_by_group
            quote
                s = sigma
                r = rho
                weights = $weight_plate
                scaled_z = z .* transpose(weights)
            end
        else
            quote
                frequencies = transpose(vec(sum(omega2; dims=2)))
                scale = sigma .* (rho .* sqrt(2pi)).^($D / 2)
                exponent = (rho .^ 2) .* frequencies
                spectra = scale .* exp.(-0.25 .* exponent)
                scaled_z = z .* spectra
            end
        end
        # The same row-wise contraction as the scalar sum: gather the
        # original group's mode weights, multiply by the row's basis,
        # then reduce only the mode axis.
        push!(body.args, :(rows = scaled_z[group_index,:]),
            :(value = vec(sum(PHI .* rows; dims=2))), Expr(:return,:value))
    end
    push!(definitions, _rk_ast_graph_definition(entry, arguments, body))
    # Retain the reusable statistical value declaration, with its numerical
    # child graph explicit in the same emitted namespace.
    wrapper = _rk_ast_fresh_name("brm_hsgp_summand", taken)
    push!(definitions, Expr(:(=), Expr(:call, wrapper, arguments...),
        Expr(:block, Expr(:call, entry, arguments...))))
    values = [PHI, omega2, sigma, rho, z]
    group_index === nothing || push!(values, group_index)
    Expr(:call, wrapper, values...)
end
