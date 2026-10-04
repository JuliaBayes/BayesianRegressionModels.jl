# General prepared group fields: BRM owns prior declarations and block algebra;
# the consumer's existing native effect owns how each block enters the term.
# One term's fitted statistical inputs cross through an ordinary data port.
# Capturing its arrays in a callable would hide them from the AD interface.
function brm_structured_effect(block, prepared_inputs, field_index)
    prepared = only(prepared_inputs)
    _brm_native_structured_effect(prepared, block, prepared.state.fields[field_index])
end

# The statistical reader applies each group's weights to the same fitted
# Hilbert basis. Shared hypers remain scalars; authored hyper predictors are
# vectors with one value per declared group, and rho floors act per group.
function brm_hsgp_grouped(PHI, omega2, z, group_index, rho, sigma)
    values = similar(z, size(PHI,1))
    if rho isa Number && sigma isa Number
        spectrum = brm_hsgp_sqrt_spd(omega2, sigma, rho)
        for i in axes(PHI,1)
            value = zero(eltype(z))
            for j in axes(PHI,2)
                value += PHI[i,j] * z[group_index[i],j] * spectrum[j]
            end
            values[i] = value
        end
        return values
    end
    spectra = similar(z)
    for g in axes(z,1)
        spectrum = brm_hsgp_sqrt_spd(omega2,
            sigma isa Number ? sigma : sigma[g], rho isa Number ? rho : rho[g])
        for j in axes(z,2)
            spectra[g,j] = spectrum[j]
        end
    end
    for i in axes(PHI,1)
        value = zero(eltype(z))
        for j in axes(PHI,2)
            value += PHI[i,j] * z[group_index[i],j] * spectra[group_index[i],j]
        end
        values[i] = value
    end
    values
end

function _rk_ast_hsgp_hyper!(stmts, options, hyper, floors, taken, bindings, G)
    id = options.id
    prior = hyper === :length_scale ? options.rho_prior : options.sigma_prior
    stated = hyper === :length_scale ? options.rho_stated : options.sigma_stated
    plans = options.hyper_plans
    position = findfirst(p -> p.hyper === hyper, plans)
    stem = hyper === :length_scale ? "rho" : "sigma"
    if position === nothing
        value = _rk_ast_fresh_name(string(id, "_", stem), taken)
        expression = _rk_ast_positive_prior(prior, bindings, taken)
        hyper === :length_scale && options.rho_truncated &&
            (expression = Expr(:call, :restricted, expression, floors, Inf))
        push!(stmts, Expr(:call, :~, value, expression))
        return value
    end
    plan = plans[position]
    parts = Any[]
    if plan.intercept
        beta = _rk_ast_fresh_name(string(id, "_", stem, "_Intercept"), taken)
        expression = stated ? _rk_value_expr!(bindings, _brm_prepare_expr(prior), taken) :
            Expr(:call, :Normal, 0, 1)
        push!(stmts, Expr(:call, :~, beta, expression))
        push!(parts, beta)
    elseif stated
        error("RK backend: HSGP hyper prior needs an intercept in its authored predictor")
    end
    if !isempty(plan.ranefs)
        sd = _rk_ast_fresh_name(string(id, "_", stem, "_sd"), taken)
        z = _rk_ast_fresh_name(string(id, "_", stem, "_z"), taken)
        push!(stmts, Expr(:call, :~, sd,
            Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
        push!(stmts, Expr(:call, :.~, Expr(:ref, z, Expr(:call, :(:), 1, G)),
            _rk_ast_dotted(:Normal, 0, 1)))
        push!(parts, Expr(:call, :.*, sd, z))
    end
    eta = length(parts) == 1 ? only(parts) : Expr(:call, :.+, parts...)
    value = _rk_ast_fresh_name(string(id, "_", stem, "_value"), taken)
    expression = _rk_ast_dotted(:exp, eta)
    hyper === :length_scale && (expression = _rk_ast_dotted(:max, expression, floors))
    push!(stmts, Expr(:(=), value, expression))
    value
end

function _rk_ast_hsgp_grouped(definitions, term, PHI, lambda, floors, taken, bindings)
    options = term.options
    G = get(options, :n_groups, 1)
    stmts = Expr[]
    rho = _rk_ast_hsgp_hyper!(stmts, options, :length_scale, floors, taken, bindings, G)
    sigma = _rk_ast_hsgp_hyper!(stmts, options, :sd, floors, taken, bindings, G)
    z = _rk_ast_fresh_name(string(options.id, "_z"), taken)
    if haskey(options, :group_index)
        push!(stmts, Expr(:call, :.~, Expr(:ref, z, Expr(:call, :(:), 1, G),
            Expr(:call, :axes, PHI, 2)), _rk_ast_dotted(:Normal, 0, 1)))
        push!(stmts, Expr(:call, :~, options.id,
            _rk_ast_hsgp_value_graph!(definitions, term, taken,
                PHI, lambda, sigma, rho, z; group_index=options.group_index)))
    else
        push!(stmts, Expr(:call, :.~, Expr(:ref, z, Expr(:call, :axes, PHI, 2)),
            _rk_ast_dotted(:Normal, 0, 1)))
        push!(stmts, Expr(:call, :~, options.id,
            _rk_ast_hsgp_value_graph!(definitions, term, taken,
                PHI, lambda, sigma, rho, z)))
    end
    stmts
end
function _rk_ast_structured_block(field, taken, bindings)
    K, G = field.n_per_group, length(field.levels)
    K isa Integer && K > 0 || error("RK backend: structured field needs a positive width")
    G > 0 || error("RK backend: structured field has no declared group levels")
    block = _rk_ast_fresh_name(string("b_", field.name, "_", field.source), taken)
    z = _rk_ast_fresh_name(string(block, "_z"), taken)
    axes = (Expr(:call, :(:), 1, G), Expr(:call, :(:), 1, K))
    stmts = Expr[]
    if field.prior === :correlated_normal
        tau = _rk_ast_fresh_name(string(block, "_sd"), taken)
        push!(stmts, Expr(:call, :.~, Expr(:ref, tau, last(axes)),
            _rk_ast_dotted(:restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
        push!(stmts, Expr(:call, :.~, Expr(:ref, z, axes...), _rk_ast_dotted(:Normal, 0, 1)))
        value = if K == 1
            Expr(:call, :.*, z, Expr(:ref, tau, 1))
        else
            L = _rk_ast_fresh_name(string(block, "_L"), taken)
            push!(stmts, Expr(:call, :~, L, Expr(:call, :LKJCholesky, K, 1.0)))
            Expr(:call, :*, z, Expr(:call, :transpose, Expr(:call, :.*, tau, L)))
        end
        push!(stmts, Expr(:(=), block, value))
    else
        prior = field.prior === :iid_normal ? ExprColumn(Normal, 0, 1) :
            _brm_structured_prior_expression(field)
        expression = _rk_value_expr!(bindings, _brm_prepare_expr(prior), taken)
        if field.prior isa NamedTuple &&
                (haskey(field.prior, :lower) || haskey(field.prior, :upper))
            expression = Expr(:call, :truncated, expression,
                get(field.prior, :lower, -Inf), get(field.prior, :upper, Inf))
        end
        push!(stmts, Expr(:call, :.~, Expr(:ref, z, axes...),
            _rk_ast_dotted(expression.args[1], expression.args[2:end]...)))
        push!(stmts, Expr(:(=), block, z))
    end
    block, stmts
end

function _rk_ast_structured_term(term, blocks, taken, bindings)
    prepared = term.options.prepared
    stmts = Expr[]
    parts = Symbol[]
    for (field_index, field) in enumerate(prepared.state.fields)
        key = (field.name, field.source)
        signature = (field.n_per_group, field.levels, field.prior)
        block = if haskey(blocks, key)
            old = blocks[key]
            isequal(old.signature, signature) || error(
                "RK backend: structured field `$(field.name)` over `$(field.source)` " *
                "has conflicting widths, levels or priors")
            old.name
        else
            name, body = _rk_ast_structured_block(field, taken, bindings)
            append!(stmts, body)
            blocks[key] = (; name, signature)
            name
        end
        callee = _rk_value_callee!(bindings, brm_structured_effect, taken)
        value = _rk_ast_fresh_name(string(term.options.id, "_", field.name), taken)
        push!(stmts, Expr(:(=), value, Expr(:call, callee, block,
            term.options.prepared_data, field_index)))
        push!(parts, value)
    end
    isempty(parts) && error("RK backend: structured term has no prepared fields")
    value = length(parts) == 1 ? only(parts) : Expr(:call, :.+, parts...)
    push!(stmts, Expr(:(=), term.options.id, value))
    stmts
end
