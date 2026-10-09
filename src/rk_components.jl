# Statistical components as self-contained RKPPL submodels. Each component
# allocates its own parameters and returns its post-processed value, the
# shape SBBRMI emits with `popefs`, `ranef_*_draws`, `_sb_mo` and `_sb_hsgp`.
# Data preparation and the combination of component values stay in the main
# block. Single-expression algebra is written inline rather than as a submodel.

const _RK_KERNEL_MACRO = Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel")))

# Install one definition per distinct signature and body. Components with
# identical configuration share a definition; a different configuration (or
# a collision with an authored name) receives a fresh name.
function _rk_ast_shared_definition!(definitions, taken, base, arguments, body;
        kernel=false, head=:(=))
    arguments = collect(Any, arguments)
    for definition in definitions
        kernel == Meta.isexpr(definition, :macrocall) || continue
        inner = kernel ? last(definition.args) : definition
        Meta.isexpr(inner, head, 2) || continue
        call = first(inner.args)
        Meta.isexpr(call, :call) || continue
        isequal(call.args[2:end], arguments) || continue
        isequal(last(inner.args), body) || continue
        return first(call.args)
    end
    callee = _rk_ast_fresh_name(string(base), taken)
    definition = Expr(head, Expr(:call, callee, arguments...), body)
    kernel && (definition = Expr(:macrocall, _RK_KERNEL_MACRO, LineNumberNode(0), definition))
    push!(definitions, definition)
    callee
end

_rk_ast_component_body(statements...) = Expr(:block, statements...)

# A component call samples its submodel under `name`; its locals are scoped
# there (`name.beta`, `name.sd`, ...).
_rk_ast_component_call(name, callee, arguments...) =
    Expr(:call, :~, name, Expr(:call, callee, arguments...))

# Caller values in authored priors are explicit component inputs. Otherwise a
# caller named `tau` or `sigma` would be captured by the component's own draw.
# Call heads retain their complete emitted module bindings.
function _rk_ast_component_prior_inputs(priors, occupied)
    reads = Set{Symbol}()
    function collect_reads(value)
        value isa Symbol && push!(reads, value)
        value isa Expr || return
        args = value.head in (:call, :kw) ? value.args[2:end] : value.args
        foreach(collect_reads, args)
    end
    foreach(collect_reads, priors)
    values = sort!(collect(reads); by=string)
    used = union(Set{Symbol}(occupied), reads)
    inputs = Symbol[]
    for j in eachindex(values)
        push!(inputs, _rk_ast_fresh_name("prior_input_$j", used))
    end
    replacements = Dict(zip(values, inputs))
    function replace_reads(value)
        value isa Symbol && return get(replacements, value, value)
        value isa Expr || return value
        value.head in (:call, :kw) && return Expr(value.head, first(value.args),
            map(replace_reads, value.args[2:end])...)
        Expr(value.head, map(replace_reads, value.args)...)
    end
    map(replace_reads, priors), inputs, values
end

# Scalar population coefficient families admitted on RKPPL's design-matrix
# route, with their positional argument roles.
const _RK_POPULATION_FAMILIES = (
    Normal=(:loc, :scale), Cauchy=(:loc, :scale), Laplace=(:loc, :scale),
    Logistic=(:loc, :scale), StudentT=(:nu, :loc, :scale), Flat=(),
    Uniform=(:lower, :upper))

_rk_ast_population_family(prior) = prior !== nothing &&
    haskey(_RK_POPULATION_FAMILIES, first(prior)) &&
    length(last(prior)) == length(getproperty(_RK_POPULATION_FAMILIES, first(prior)))

# One argument shared by every coefficient stays scalar; otherwise each
# column states its own value.
function _rk_ast_population_argument(values)
    all(isequal(first(values)), values) ? first(values) : Expr(:vect, values...)
end

"""Population effects over design columns: coefficients allocated inside, `X * beta_pop` returned.

Predictors with identical design columns share one design matrix; `design` is
the base name of a new one. `designs` records each matrix with the statements
that name it, for `_rk_ast_name_shared_designs!`."""
function _rk_ast_population_component!(definitions, statements, taken, name, design,
        columns, priors, designs)
    matrix = Expr(:call, :hcat, columns...)
    shared = findfirst(record -> isequal(last(record.assignment.args), matrix), designs)
    if shared === nothing
        design = _rk_ast_fresh_name(design, taken)
        assignment = Expr(:(=), design, matrix)
        push!(statements, assignment)
        designs[design] = (; columns, assignment, calls=Expr[])
    else
        design = shared
    end
    product = Expr(:call, :*, :X, :beta_pop)
    families = unique(first.(priors))
    if length(families) > 1
        # Coefficients with different prior families keep one statement each,
        # as SBBRMI's configured generic population block does.
        coefficients = [Symbol(:beta_pop_, j) for j in eachindex(priors)]
        body = Expr(:block)
        for (coefficient, (family, args)) in zip(coefficients, priors)
            push!(body.args, Expr(:call, :~, coefficient, Expr(:call, family, args...)))
        end
        push!(body.args, Expr(:(=), :beta_pop, Expr(:vect, coefficients...)))
        push!(body.args, Expr(:return, product))
        callee = _rk_ast_shared_definition!(definitions, taken,
            "brm_mixed_population_effects", (:X,), body)
        call = _rk_ast_component_call(name, callee, design)
        push!(designs[design].calls, call)
        push!(statements, call)
        return name
    end
    family = only(families)
    roles = getproperty(_RK_POPULATION_FAMILIES, family)
    body = _rk_ast_component_body(
        Expr(:call, :.~, Expr(:ref, :beta_pop, Expr(:call, :(:), 1, :ncoef)),
            _rk_ast_dotted(family, roles...)),
        Expr(:return, product))
    base = family === :Normal ? "brm_population_effects" :
        string("brm_population_effects_", lowercase(string(family)))
    callee = _rk_ast_shared_definition!(definitions, taken, base, (:X, :ncoef, roles...), body)
    arguments = [_rk_ast_population_argument([last(prior)[j] for prior in priors])
        for j in eachindex(roles)]
    call = _rk_ast_component_call(name, callee, design, length(columns), arguments...)
    push!(designs[design].calls, call)
    push!(statements, call)
    name
end

# A design column's part of a shared matrix name: the intercept's column of
# ones is `Intercept`, as its coefficient is.
_rk_ast_design_label(column::Symbol) = string(column)
function _rk_ast_design_label(column::Expr)
    Meta.isexpr(column, :call) && first(column.args) === :ones && return "Intercept"
    error("RK backend: internal: design column `$column` has no name")
end

# A design matrix read by one predictor keeps that predictor's name. One read
# by several belongs to none of them: it is named for its columns instead.
# The recorded statements are this emission's own, not yet published.
function _rk_ast_name_shared_designs!(designs, taken)
    for design in sort!(collect(keys(designs)); by=string)
        record = designs[design]
        length(record.calls) > 1 || continue
        shared = _rk_ast_fresh_name(
            join(("X", map(_rk_ast_design_label, record.columns)...), "_"), taken)
        record.assignment.args[1] = shared
        foreach(call -> call.args[3].args[2] = shared, record.calls)
    end
    designs
end

# A monotonic effect owns its increment simplex and, when it has one, its
# coefficient. A coefficient-free term returns only its contrast.
function _rk_ast_monotonic_component!(definitions, statements, taken, name, index,
        alpha, prior)
    contrast = :(cumsum(vcat(0.0, simplex_incr))[c])
    simplex = Expr(:call, :~, :simplex_incr, Expr(:call, :Dirichlet, :alpha))
    if prior === nothing
        body = _rk_ast_component_body(simplex, Expr(:return, contrast))
        callee = _rk_ast_shared_definition!(definitions, taken,
            "brm_monotonic_value", (:c, :alpha), body)
        push!(statements, _rk_ast_component_call(name, callee, index, Expr(:vect, alpha...)))
        return name
    end
    family, args = prior
    roles = haskey(_RK_POPULATION_FAMILIES, family) &&
        length(getproperty(_RK_POPULATION_FAMILIES, family)) == length(args) ?
        getproperty(_RK_POPULATION_FAMILIES, family) :
        Tuple(Symbol(:arg_, j) for j in eachindex(args))
    body = _rk_ast_component_body(simplex,
        Expr(:call, :~, :beta, Expr(:call, family, roles...)),
        Expr(:return, Expr(:call, :.*, :beta, contrast)))
    base = family === :Normal ? "brm_monotonic_effect" :
        string("brm_monotonic_effect_", lowercase(string(family)))
    callee = _rk_ast_shared_definition!(definitions, taken, base, (:c, :alpha, roles...), body)
    push!(statements, _rk_ast_component_call(name, callee, index,
        Expr(:vect, alpha...), args...))
    name
end

# One shared positive prior broadcasts over the margins; distinct priors are
# stated per margin and collected into the same `tau` vector.
function _rk_ast_group_scales(priors, index)
    if all(isequal(first(priors)), priors)
        prior = first(priors)
        return Expr[Expr(:call, :.~, Expr(:ref, :tau, index),
            _rk_ast_dotted(prior.args[1], prior.args[2:end]...))]
    end
    scales = [Symbol(:tau_, j) for j in eachindex(priors)]
    statements = Expr[Expr(:call, :~, scale, prior) for (scale, prior) in zip(scales, priors)]
    push!(statements, Expr(:(=), :tau, Expr(:vect, scales...)))
    statements
end

# Group-level effects allocate their scales, correlation factor and
# standardized draws, and return the non-centered levels-by-margins matrix.
# `priors` holds one positive prior per margin; with `scale_value` the scales
# are instead a supplied value (shared R2D2M2 budgets). A Student-t block
# (`nu`, a constant or graph value) also draws one mixing weight per level,
# `w ~ InverseGamma(nu/2, nu/2)`, and scales that level's row by `sqrt(w)`:
# the same scale mixture as StanBlocks' `_sb_student_t_ranef`.
function _rk_ast_group_component!(definitions, statements, taken, name, group, K, eta,
        priors; scale_value=nothing, nu=nothing)
    priors, prior_inputs, prior_values = _rk_ast_component_prior_inputs(priors,
        (:g, :K, :eta, :tau, :L, :z, :w, :nu, (Symbol(:tau_, j) for j in 1:K)...))
    index = K > 1 ? Expr(:call, :(:), 1, :K) : Expr(:call, :(:), 1, 1)
    body = Expr(:block)
    scale_value === nothing && append!(body.args, _rk_ast_group_scales(priors, index))
    K > 1 && push!(body.args, Expr(:call, :~, :L, Expr(:call, :LKJCholesky, :K, :eta)))
    push!(body.args, Expr(:call, :.~, Expr(:ref, :z, Expr(:call, :levels, :g), index),
        _rk_ast_dotted(:Normal, 0, 1)))
    value = K > 1 ?
        Expr(:call, :*, :z, Expr(:call, :transpose, Expr(:call, :.*, :tau, :L))) :
        Expr(:call, :.*, :z, Expr(:ref, :tau, 1))
    if nu !== nothing
        half = Expr(:call, :/, :nu, 2)
        push!(body.args, Expr(:call, :.~, Expr(:ref, :w, Expr(:call, :levels, :g)),
            _rk_ast_dotted(:InverseGamma, half, half)))
        value = Expr(:call, :.*, _rk_ast_dotted(:sqrt, :w), value)
    end
    push!(body.args, Expr(:return, value))
    arguments = Any[:g]
    K > 1 && push!(arguments, :K, :eta)
    scale_value === nothing || push!(arguments, :tau)
    nu === nothing || push!(arguments, :nu)
    append!(arguments, prior_inputs)
    base = string(scale_value === nothing ? "brm_" : "brm_scaled_",
        nu === nothing ? "" : "student_t_",
        K > 1 ? "correlated_group_effects" : "group_effects")
    callee = _rk_ast_shared_definition!(definitions, taken, base, arguments, body)
    values = Any[group]
    K > 1 && push!(values, K, eta)
    scale_value === nothing || push!(values, scale_value)
    nu === nothing || push!(values, nu)
    append!(values, prior_values)
    push!(statements, _rk_ast_component_call(name, callee, values...))
    name
end

# Stratified group-level effects: one scale vector (and factor) per stratum.
function _rk_ast_stratified_group_component!(definitions, statements, taken, name,
        group, stratum, K, eta)
    index = Expr(:call, :(:), 1, K)
    body = Expr(:block, Expr(:call, :.~, Expr(:ref, :tau, Expr(:call, :levels, :s), index),
        _rk_ast_dotted(:restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
    if K > 1
        cell = Expr(:call, :~, Expr(:ref, :L, :k), Expr(:call, :LKJCholesky, K, eta))
        push!(body.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
            Expr(:for, Expr(:(=), :k, Expr(:call, :levels, :s)), Expr(:block, cell))))
    end
    push!(body.args, Expr(:call, :.~, Expr(:ref, :z, Expr(:call, :levels, :g), index),
        _rk_ast_dotted(:Normal, 0, 1)))
    scale = Expr(:ref, :tau, Expr(:ref, :s, :i), :(:))
    raw = Expr(:ref, :z, Expr(:ref, :g, :i), :(:))
    value = K == 1 ? Expr(:call, :.*, scale, raw) :
        Expr(:call, :*, Expr(:call, :.*, scale, Expr(:ref, :L, Expr(:ref, :s, :i))), raw)
    cell = Expr(:(=), Expr(:ref, :b, :i, index), value)
    push!(body.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :g)), Expr(:block, cell))))
    push!(body.args, Expr(:return, :b))
    callee = _rk_ast_shared_definition!(definitions, taken,
        "brm_stratified_group_effects", (:g, :s), body)
    push!(statements, _rk_ast_component_call(name, callee, group, stratum))
    name
end

# An HSGP effect allocates its length scale(s), marginal scale and basis
# weights and returns the spectral-weighted basis product. Basis, frequencies
# and validity floors are passed in from the main block. A model-derived
# basis supplies its formula extent, since its matrix is a graph value.
function _rk_ast_hsgp_component!(definitions, statements, taken, name, PHI, omega2,
        floors, rho_priors, sigma_prior, value; truncated=true, nbasis=nothing,
        base="brm_hsgp_effect")
    priors, prior_inputs, prior_values = _rk_ast_component_prior_inputs(
        [rho_priors; sigma_prior],
        (:PHI, :omega2, :rho_floor, :nbasis, :rho_iso, :rho, :sigma, :beta_raw,
            (Symbol(:rho_, j) for j in eachindex(rho_priors))...))
    rho_priors, sigma_prior = priors[1:end-1], last(priors)
    body = Expr(:block)
    floor_argument = truncated ? (:rho_floor,) : ()
    if length(rho_priors) == 1
        prior = only(rho_priors)
        truncated && (prior = Expr(:call, :restricted, prior, :rho_floor, Inf))
        push!(body.args, Expr(:call, :~, :rho_iso, prior))
    else
        rhos = Symbol[]
        for (j, prior) in enumerate(rho_priors)
            rho = Symbol(:rho_, j)
            push!(rhos, rho)
            truncated && (prior = Expr(:call, :restricted, prior,
                Expr(:ref, :rho_floor, j), Inf))
            push!(body.args, Expr(:call, :~, rho, prior))
        end
        push!(body.args, Expr(:(=), :rho, Expr(:vect, rhos...)))
    end
    push!(body.args, Expr(:call, :~, :sigma, sigma_prior))
    extent = nbasis === nothing ? Expr(:call, :axes, :PHI, 2) :
        Expr(:call, :(:), 1, :nbasis)
    push!(body.args, Expr(:call, :.~, Expr(:ref, :beta_raw, extent),
        _rk_ast_dotted(:Normal, 0, 1)))
    push!(body.args, Expr(:return, value))
    callee = _rk_ast_shared_definition!(definitions, taken, base,
        (:PHI, :omega2, floor_argument..., (nbasis === nothing ? () : (:nbasis,))...,
            prior_inputs...), body)
    push!(statements, _rk_ast_component_call(name, callee, PHI, omega2,
        (truncated ? (floors,) : ())..., (nbasis === nothing ? () : (nbasis,))...,
        prior_values...))
    name
end
