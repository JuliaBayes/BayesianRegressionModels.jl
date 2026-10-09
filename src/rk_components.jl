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
the base name of a new one. `designs` records each matrix with the component
calls that read it, their priors and their coefficients' coordinate records
(`coefficients`, one `(; predictor, coefficient, paired)` per column), for
`_rk_ast_join_shared_designs!`."""
function _rk_ast_population_component!(definitions, statements, taken, name, design,
        columns, priors, designs, coefficients)
    matrix = Expr(:call, :hcat, columns...)
    shared = findfirst(record -> isequal(last(record.assignment.args), matrix), designs)
    if shared === nothing
        design = _rk_ast_fresh_name(design, taken)
        assignment = Expr(:(=), design, matrix)
        push!(statements, assignment)
        designs[design] = (; columns, assignment, entries=Any[])
    else
        design = shared
    end
    entry(call, mixed) = push!(designs[design].entries,
        (; name, call, priors, coefficients, mixed))
    product = Expr(:call, :*, :X, :beta_pop)
    families = unique(first.(priors))
    if length(families) > 1
        # Coefficients with different prior families keep one statement each,
        # as SBBRMI's configured generic population block does.
        declarations = [Symbol(:beta_pop_, j) for j in eachindex(priors)]
        body = Expr(:block)
        for (declaration, (family, args)) in zip(declarations, priors)
            push!(body.args, Expr(:call, :~, declaration, Expr(:call, family, args...)))
        end
        push!(body.args, Expr(:(=), :beta_pop, Expr(:vect, declarations...)))
        push!(body.args, Expr(:return, product))
        callee = _rk_ast_shared_definition!(definitions, taken,
            "brm_mixed_population_effects", (:X,), body)
        call = _rk_ast_component_call(name, callee, design)
        entry(call, true)
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
    entry(call, false)
    push!(statements, call)
    name
end

# A population component's coefficient records: `<name>.beta_pop` indexed by
# column, or one `<name>.beta_pop_<j>` declaration per column for mixed families.
function _rk_ast_population_records!(coordinates, entry)
    for (j, c) in enumerate(entry.coefficients)
        c.paired || continue
        _rk_coordinate_record!(coordinates, (; kind=:population,
            declaration=Symbol(entry.name, entry.mixed ? ".beta_pop_$j" : ".beta_pop"),
            index=entry.mixed ? () : (j,), predictor=c.predictor, coefficient=c.coefficient))
    end
end

# A prior argument of a joint population block: one value shared by every
# coefficient of every predictor, or a coefficients x predictors literal matrix
# (`hcat` of one column per predictor). `nothing` when a value is not a literal.
function _rk_ast_joint_population_argument(values::AbstractMatrix)
    all(isequal(first(values)), values) && return first(values)
    all(v -> v isa Real, values) || return nothing
    Expr(:call, :hcat, (Expr(:vect, values[:, t]...) for t in axes(values, 2))...)
end

# A design column's part of a shared matrix name: the intercept's column of
# ones is `Intercept`, as its coefficient is.
_rk_ast_design_label(column::Symbol) = string(column)
function _rk_ast_design_label(column::Expr)
    Meta.isexpr(column, :call) && first(column.args) === :ones && return "Intercept"
    error("RK backend: internal: design column `$column` has no name")
end

# A design matrix read by one predictor keeps that predictor's name and its
# own population component. One read by several belongs to none of them: it is
# named for its columns, and the predictors' coefficients become one joint
# block over it (user decision `1pj2p5j`): `pop_<columns>` owns a coefficients
# x predictors matrix `beta_pop`, computes `X * beta_pop` once, and each
# predictor reads its column, `pop_<target> = pop_<columns>[:, t]`. Its
# coordinates are `pop_<columns>.beta_pop.<j>.<t>`. The recorded statements are
# this emission's own, not yet published.
function _rk_ast_join_shared_designs!(definitions, statements, designs, taken, coordinates)
    for design in sort!(collect(keys(designs)); by=string)
        record = designs[design]
        entries = record.entries
        if length(entries) == 1
            _rk_ast_population_records!(coordinates, only(entries))
            continue
        end
        labels = map(_rk_ast_design_label, record.columns)
        shared = _rk_ast_fresh_name(join(("X", labels...), "_"), taken)
        record.assignment.args[1] = shared
        foreach(entry -> entry.call.args[3].args[2] = shared, entries)
        if !_rk_ast_joint_population!(definitions, statements, taken, coordinates,
                shared, labels, entries)
            foreach(entry -> _rk_ast_population_records!(coordinates, entry), entries)
        end
    end
    designs
end

# The joint block of `_rk_ast_join_shared_designs!`. Its coefficients share one
# prior family, whose arguments may differ per coefficient and predictor. When
# they need families or argument values a two-axis declaration cannot state
# (mixed families, non-literal varying arguments, per-element `Uniform`
# bounds), the predictors keep their own components and return `false`.
function _rk_ast_joint_population!(definitions, statements, taken, coordinates,
        design, labels, entries)
    any(entry -> entry.mixed, entries) && return false
    families = unique(first(prior) for entry in entries for prior in entry.priors)
    length(families) == 1 || return false
    family = only(families)
    roles = getproperty(_RK_POPULATION_FAMILIES, family)
    ncoef, ntargets = length(first(entries).priors), length(entries)
    arguments = map(eachindex(roles)) do r
        _rk_ast_joint_population_argument(
            [last(entry.priors[j])[r] for j in 1:ncoef, entry in entries])
    end
    any(isnothing, arguments) && return false
    family === :Uniform && any(a -> a isa Expr, arguments) && return false
    body = _rk_ast_component_body(
        Expr(:call, :.~, Expr(:ref, :beta_pop, Expr(:call, :(:), 1, :ncoef),
                Expr(:call, :(:), 1, :ntargets)),
            _rk_ast_dotted(family, roles...)),
        Expr(:return, Expr(:call, :*, :X, :beta_pop)))
    base = family === :Normal ? "brm_joint_population_effects" :
        string("brm_joint_population_effects_", lowercase(string(family)))
    callee = _rk_ast_shared_definition!(definitions, taken, base,
        (:X, :ncoef, :ntargets, roles...), body)
    joint = _rk_ast_fresh_name(join(("pop", labels...), "_"), taken)
    first_call = findfirst(s -> s === first(entries).call, statements)
    insert!(statements, first_call,
        _rk_ast_component_call(joint, callee, design, ncoef, ntargets, arguments...))
    for (t, entry) in enumerate(entries)
        entry.call.head = :(=)
        entry.call.args = Any[entry.name, Expr(:ref, joint, :(:), t)]
        for (j, c) in enumerate(entry.coefficients)
            c.paired && _rk_coordinate_record!(coordinates, (; kind=:population,
                declaration=Symbol(joint, ".beta_pop"), index=(j, t),
                predictor=c.predictor, coefficient=c.coefficient))
        end
    end
    true
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

# Several predictors' monotonic effects of one column, as one component: one
# increment simplex per predictor (the columns of `simplex_incr`), one
# cumulative table, one gather by the codes `c`, and the coefficients `beta`
# when the effects have them. Returns the call statement and, per predictor,
# its coefficient's `(declaration, index)` or `nothing`. One shared prior
# family broadcasts over `beta[1:ntargets]`; mixed families give one
# statement per predictor. A shared Dirichlet concentration is one vector;
# per-predictor concentrations are the columns of an increments x predictors
# matrix (`Dirichlet.(eachcol(alpha))`).
function _rk_ast_joint_monotonic_component!(definitions, taken, name, index, alphas, priors)
    ntargets = length(priors)
    shared = all(isequal(first(alphas)), alphas)
    law = shared ? Expr(:call, :Dirichlet, :alpha) :
        Expr(:., :Dirichlet, Expr(:tuple, Expr(:call, :eachcol, :alpha)))
    simplex = Expr(:call, :.~, Expr(:call, :eachcol, Expr(:ref, :simplex_incr,
            Expr(:call, :(:), 1, :nincr), Expr(:call, :(:), 1, :ntargets))), law)
    table = :((cumsum(vcat(zeros(1, ntargets), simplex_incr); dims=1))[c, :])
    alpha = shared ? Expr(:vect, first(alphas)...) :
        Expr(:vcat, (Expr(:row, (alphas[t][i] for t in 1:ntargets)...)
            for i in eachindex(first(alphas)))...)
    suffix = shared ? "" : "_by_column"
    inputs = (index, alpha, length(first(alphas)), ntargets)
    if all(isnothing, priors)
        body = _rk_ast_component_body(simplex, Expr(:return, table))
        callee = _rk_ast_shared_definition!(definitions, taken,
            "brm_joint_monotonic_values" * suffix, (:c, :alpha, :nincr, :ntargets), body)
        return _rk_ast_component_call(name, callee, inputs...), fill(nothing, ntargets)
    end
    any(isnothing, priors) && error("RK backend: internal: joint monotonic block " *
        "`$name` mixes effects with and without a coefficient")
    families = unique(first.(priors))
    effect = Expr(:return, Expr(:call, :.*, table, Expr(Symbol("'"), :beta)))
    if length(families) == 1
        family = only(families)
        roles = haskey(_RK_POPULATION_FAMILIES, family) &&
            length(getproperty(_RK_POPULATION_FAMILIES, family)) == length(last(first(priors))) ?
            getproperty(_RK_POPULATION_FAMILIES, family) :
            Tuple(Symbol(:arg_, j) for j in eachindex(last(first(priors))))
        body = _rk_ast_component_body(simplex,
            Expr(:call, :.~, Expr(:ref, :beta, Expr(:call, :(:), 1, :ntargets)),
                _rk_ast_dotted(family, roles...)),
            effect)
        base = (family === :Normal ? "brm_joint_monotonic_effects" :
            string("brm_joint_monotonic_effects_", lowercase(string(family)))) * suffix
        callee = _rk_ast_shared_definition!(definitions, taken, base,
            (:c, :alpha, :nincr, :ntargets, roles...), body)
        arguments = [_rk_ast_population_argument([last(prior)[j] for prior in priors])
            for j in eachindex(roles)]
        betas = [(Symbol(name, ".beta"), (t,)) for t in 1:ntargets]
        return _rk_ast_component_call(name, callee, inputs..., arguments...), betas
    end
    # Coefficients with different prior families keep one statement each.
    coefficients = [Symbol(:beta_, t) for t in 1:ntargets]
    body = _rk_ast_component_body(simplex,
        (Expr(:call, :~, coefficient, Expr(:call, family, args...))
            for (coefficient, (family, args)) in zip(coefficients, priors))...,
        Expr(:(=), :beta, Expr(:vect, coefficients...)),
        effect)
    callee = _rk_ast_shared_definition!(definitions, taken,
        "brm_joint_monotonic_mixed_effects" * suffix, (:c, :alpha, :nincr, :ntargets), body)
    betas = [(Symbol(name, ".", coefficient), ()) for coefficient in coefficients]
    _rk_ast_component_call(name, callee, inputs...), betas
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
# weights and returns the spectral-weighted basis product. `inputs` pairs its
# data arguments with their main-block values: the axes of a
# squared-exponential term, or a periodic term's prepared basis and
# frequencies. The validity floor is passed in where a prior reads it.
function _rk_ast_hsgp_component!(definitions, statements, taken, name, inputs,
        floors, rho_priors, sigma_prior, value, extent; truncated=true,
        base="brm_hsgp_effect")
    formals = first.(inputs)
    priors, prior_inputs, prior_values = _rk_ast_component_prior_inputs(
        [rho_priors; sigma_prior],
        (formals..., :rho_floor, :rho_iso, :rho, :sigma, :beta_raw,
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
    push!(body.args, Expr(:call, :.~, Expr(:ref, :beta_raw, extent),
        _rk_ast_dotted(:Normal, 0, 1)))
    push!(body.args, Expr(:return, value))
    callee = _rk_ast_shared_definition!(definitions, taken, base,
        (formals..., floor_argument..., prior_inputs...), body)
    push!(statements, _rk_ast_component_call(name, callee, last.(inputs)...,
        (truncated ? (floors,) : ())..., prior_values...))
    name
end
