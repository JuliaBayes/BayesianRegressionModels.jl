# Ordinary model values beside regression formulas. Regression geometry and
# priors use the same planner/emitter as the GLM route; calls cross as values,
# with their exact callable captured in the build's private module.
struct _RKValuePlan
    regression::_RKStructuralPlan
    assignments::Tuple
    observations::Tuple
    columns::Dict{Symbol,Any}
end

_rk_value_invlogit(x) = logistic(x)
brm_invprobit(x) = 0.5erfc(-x / sqrt(2))
brm_invcloglog(x) = -expm1(-exp(x))

function _rk_value_link!(bindings, link, lhs, taken)
    link === :identity && return lhs
    head = link === :log ? :exp : link === :logit ? :logistic :
        link === :probit ? :brm_invprobit :
        link === :cloglog ? :brm_invcloglog : error("RK backend: unknown link `$link`")
    _rk_ast_dotted(head, lhs)
end

# Arrays, rather than the retired structural varying/smooth summands, let
# a named predictor be read by ordinary Julia functions on its own axis.
function _rk_value_level_indices(labels, source)
    levels = _rk_grouping_levels(source)
    Int[findfirst(isequal(label), levels) for label in labels]
end

_rk_value_dummy(values, level) = Float64.(isequal.(values, level))

function _rk_ast_positive_prior(prior, bindings, taken; default=:HalfNormal)
    prior === nothing && return default === :LogNormal ?
        Expr(:call, :LogNormal, 0, 1) :
        Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)
    prepared = _brm_prepare_expr(prior)
    if prepared.callable === truncated
        all(key -> key in (:lower, :upper), keys(prepared.kwargs)) || error(
            "RK backend: a truncated positive prior accepts only lower/upper bounds")
        args = prepared.args
        if length(args) == 1
            lower = get(prepared.kwargs, :lower, -Inf)
            upper = get(prepared.kwargs, :upper, Inf)
        elseif length(args) == 3 && isempty(prepared.kwargs)
            lower, upper = args[2:3]
        else
            error("RK backend: a truncated positive prior needs a base law and lower/upper bounds")
        end
        lower === nothing && (lower = -Inf)
        upper === nothing && (upper = Inf)
        return Expr(:call, :truncated,
            _rk_value_expr!(bindings, first(args), taken),
            _rk_value_expr!(bindings, lower, taken),
            _rk_value_expr!(bindings, upper, taken))
    end
    expression = _rk_value_expr!(bindings, prepared, taken)
    family = nameof(getf(prior))
    family in (:Exponential, :Gamma, :InverseGamma, :LogNormal, :Weibull,
        :HalfNormal, :HalfCauchy, :truncated) && return expression
    family === :Uniform && first(prepared.args) isa Real &&
        first(prepared.args) >= 0 && return expression
    # A BRM scale declaration constrains support without renormalizing its
    # authored family. Explicit `truncated` above retains its own normalizer.
    Expr(:call, :restricted, expression, 0.0, Inf)
end

function _rk_ast_value_bucket(definitions, bucket, draws, effects, taken, bindings)
    grouping = bucket.grouping
    K = length(bucket.margins)
    group = first(grouping.columns)
    stmts = Expr[]
    if grouping.form === :mm
        group = _rk_ast_fresh_name(string(draws, "_groups"), taken)
        push!(stmts, Expr(:(=), group, Expr(:call, :vcat, grouping.columns...)))
    end
    tau = _rk_ast_fresh_name(string(draws, "_sd"), taken)
    z = _rk_ast_fresh_name(string(draws, "_z"), taken)
    index = Expr(:call, :(:), 1, K)
    if grouping.form === :gr
        stratum = grouping.by
        L = _rk_ast_fresh_name(string(draws, "_L"), taken)
        push!(stmts, Expr(:call, :.~, Expr(:ref, tau,
            Expr(:call, :levels, stratum), index),
            _rk_ast_dotted(:restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
        if K > 1
            cell = Expr(:call, :~, Expr(:ref, L, :k),
                Expr(:call, :LKJCholesky, K, bucket.lkj_eta))
            push!(stmts, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
                Expr(:for, Expr(:(=), :k, Expr(:call, :levels, stratum)),
                    Expr(:block, cell))))
        end
        push!(stmts, Expr(:call, :.~, Expr(:ref, z,
            Expr(:call, :levels, group), index), _rk_ast_dotted(:Normal, 0, 1)))
        scale = Expr(:ref, tau, Expr(:ref, stratum, :i), :(:))
        raw = Expr(:ref, z, Expr(:ref, group, :i), :(:))
        value = K == 1 ? Expr(:call, :.*, scale, raw) :
            Expr(:call, :*, Expr(:call, :.*, scale,
                Expr(:ref, L, Expr(:ref, stratum, :i))), raw)
        cell = Expr(:(=), Expr(:ref, draws, :i, index), value)
        push!(stmts, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
            Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, group)), Expr(:block, cell))))
    else
        # Stan's ordinary unnamed intercept and multi-membership intercept
        # families sample log_scale ~ Normal(0,1). Shared-ID, slope and
        # stratified families keep their half-normal scale default.
        default = bucket.kind === :intercept1 ? :LogNormal : :HalfNormal
        if all(isequal(first(bucket.sd_priors)), bucket.sd_priors)
            prior = _rk_ast_positive_prior(first(bucket.sd_priors), bindings, taken; default)
            push!(stmts, Expr(:call, :.~, Expr(:ref, tau, index),
                _rk_ast_dotted(prior.args[1], prior.args[2:end]...)))
        else
            scales = Symbol[]
            for (j, prior) in enumerate(bucket.sd_priors)
                scale = _rk_ast_fresh_name(string(tau, "_", j), taken)
                push!(scales, scale)
                push!(stmts, Expr(:call, :~, scale,
                    _rk_ast_positive_prior(prior, bindings, taken)))
            end
            push!(stmts, Expr(:(=), tau, Expr(:vect, scales...)))
        end
        push!(stmts, Expr(:call, :.~, Expr(:ref, z,
            Expr(:call, :levels, group), index), _rk_ast_dotted(:Normal, 0, 1)))
        value = if K == 1
            _rk_ast_statistical_call!(definitions, taken,
                :brm_scaled_random_coefficients, tau, z)
        else
            L = _rk_ast_fresh_name(string(draws, "_L"), taken)
            push!(stmts, Expr(:call, :~, L, Expr(:call, :LKJCholesky, K, bucket.lkj_eta)))
            _rk_ast_statistical_call!(definitions, taken,
                :brm_correlated_random_coefficients, tau, L, z)
        end
        push!(stmts, Expr(:call, :~, draws, value))
    end
    indices = Dict{Symbol,Symbol}()
    if grouping.form !== :gr
        callee = :brm_level_indices
        for col in grouping.columns
            idx = _rk_ast_fresh_name(string(draws, "_index_", col), taken)
            push!(stmts, Expr(:(=), idx,
                Expr(:call, callee, col, group)))
            indices[col] = idx
        end
    end
    gather_margin(col, margin) =
        Expr(:call, :brm_ranef_column, draws, indices[col], margin)
    for (target, margins) in bucket.slices
        summands = Any[]
        for margin in margins
            coef = if grouping.form === :gr
                Expr(:ref, draws, :(:), margin)
            elseif grouping.form === :mm
                members = Any[]
                for (j, col) in enumerate(grouping.columns)
                    gather = gather_margin(col, margin)
                    grouping.weights === nothing ||
                        (gather = Expr(:call, :.*, grouping.weights[j], gather))
                    push!(members, gather)
                end
                result = Expr(:call, :.+, members...)
                if grouping.normalize
                    denom = grouping.weights === nothing ? length(members) :
                        Expr(:call, :.+, grouping.weights...)
                    result = Expr(:call, :./, result, denom)
                end
                result
            else
                gather_margin(group, margin)
            end
            recipe = bucket.margins[margin].z
            if recipe.kind !== :ones
                value = if recipe.kind === :dummy
                    callee = :brm_dummy
                    Expr(:call, callee, recipe.column, recipe.level)
                else
                    recipe.column
                end
                coef = Expr(:call, :.*, coef, value)
            end
            push!(summands, coef)
        end
        value = length(summands) == 1 ? only(summands) : Expr(:call, :.+, summands...)
        push!(stmts, Expr(:(=), effects[(target, bucket.group, bucket.id)], value))
    end
    stmts
end

function _rk_ast_value_spline(term, taken)
    X = _rk_ast_fresh_name(string(term.options.id, "_X"), taken)
    nblocks = term.options.kind === :t2 ? 3 : 1
    Z = [_rk_ast_fresh_name(string(term.options.id, "_Z", j), taken) for j in 1:nblocks]
    k = term.options.k
    kval = k isa Tuple ? Expr(:tuple, k...) : k
    basis = term.options.kind === :t2 ? :brm_t2_basis : :brm_tps_basis
    call = Expr(:call, basis, term.columns..., kval)
    b = _rk_ast_fresh_name(string(term.options.id, "_fixed"), taken)
    stmts = Expr[Expr(:(=), Expr(:tuple, X, Z...), call),
        Expr(:call, :.~, Expr(:ref, b, Expr(:call, :axes, X, 2)),
            _rk_ast_dotted(:Flat))]
    parts = Any[Expr(:call, :*, X, b)]
    for (j, block) in enumerate(Z)
        sd = _rk_ast_fresh_name(string(term.options.id, "_sd", j), taken)
        raw = _rk_ast_fresh_name(string(term.options.id, "_raw", j), taken)
        push!(stmts, Expr(:call, :~, sd,
            Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
        push!(stmts, Expr(:call, :.~, Expr(:ref, raw, Expr(:call, :axes, block, 2)),
            _rk_ast_dotted(:Normal, 0, 1)))
        push!(parts, Expr(:call, :*, block, Expr(:call, :.*, sd, raw)))
    end
    push!(stmts, Expr(:(=), term.options.id, Expr(:call, :.+, parts...)))
    stmts
end

function _rk_ast_value_hsgp(definitions, term, taken, bindings)
    options = term.options
    PHI = _rk_ast_fresh_name(string(options.id, "_PHI"), taken)
    lambda = _rk_ast_fresh_name(string(options.id, "_lambda"), taken)
    periodic = get(options, :cov, :exp_quad) === :periodic
    k = options.k isa Tuple ? Expr(:tuple, options.k...) : options.k
    floors = _rk_ast_fresh_name(string(options.id, "_floors"), taken)
    call = if periodic
        Expr(:call, :brm_hsgp_periodic_basis, only(term.columns), k, options.period)
    else
        c = options.c isa Tuple ? Expr(:tuple, options.c...) : options.c
        Expr(:call, :brm_hsgp_basis, Expr(:tuple, term.columns...), k, c, options.iso)
    end
    stmts = Expr[Expr(:(=), Expr(:tuple, PHI, lambda, floors), call)]
    if haskey(options, :group_index) || !isempty(get(options, :hyper_plans, ()))
        append!(stmts, _rk_ast_hsgp_grouped(definitions, term, PHI, lambda, floors, taken, bindings))
        return stmts
    end
    rho_value = if periodic || options.iso
        rho = _rk_ast_fresh_name(string(options.id, "_rho"), taken)
        prior = _rk_ast_positive_prior(options.rho_prior, bindings, taken)
        options.rho_truncated && (prior = Expr(:call, :restricted, prior, floors, Inf))
        push!(stmts, Expr(:call, :~, rho, prior))
        rho
    else
        rhos = Symbol[]
        for j in eachindex(term.columns)
            rho = _rk_ast_fresh_name(string(options.id, "_rho", j), taken)
            push!(rhos, rho)
            prior = _rk_ast_positive_prior(options.rho_prior, bindings, taken)
            options.rho_truncated && (prior = Expr(:call, :restricted, prior, Expr(:ref, floors, j), Inf))
            push!(stmts, Expr(:call, :~, rho, prior))
        end
        Expr(:vect, rhos...)
    end
    sigma = _rk_ast_fresh_name(string(options.id, "_sigma"), taken)
    z = _rk_ast_fresh_name(string(options.id, "_z"), taken)
    push!(stmts, Expr(:call, :~, sigma,
        _rk_ast_positive_prior(options.sigma_prior, bindings, taken)))
    push!(stmts, Expr(:call, :.~, Expr(:ref, z, Expr(:call, :axes, PHI, 2)),
        _rk_ast_dotted(:Normal, 0, 1)))
    model = periodic ? :brm_periodic_hsgp_summand : :brm_hsgp_summand
    push!(stmts, Expr(:call, :~, options.id,
        _rk_ast_statistical_call!(definitions, taken, model,
            PHI, lambda, sigma, rho_value, z)))
    stmts
end

_rk_plan_summary(plan::_RKValuePlan) = string(
    _rk_num_coefficients(plan.regression), " population coefficients and ",
    length(plan.observations), " value-based responses")

function _rk_needs_value_plan(program, observations)
    assignments = Set(op.name for op in program.operations if op.role === :assignment)
    predictors = Set(op.name for op in program.operations if op.role === :predictor)
    for op in program.operations
        op.role === :assignment || continue
        any(in(predictors), op.dependencies) && return true
        expression = _brm_prepare_expr(last(getargs(op.expression)))
        _rk_has_value_call(expression) && return true
    end
    for observation in observations
        rhs = observation.rhs
        rhs isa ExprColumn || continue
        args = getargs(rhs)
        isempty(args) && continue
        # A scalar family can consume constants or sampled values without a
        # regression formula. It uses the same ordinary value observation as
        # an authored array reader; no population intercept is synthesized.
        family = rhs
        while family isa ExprColumn && getf(family) in
                (censored, truncated, interval_censored) && !isempty(getargs(family))
            family = first(getargs(family))
        end
        head = family isa ExprColumn ? getf(family) : nothing
        # Caller-owned scalar RHS constructors use the ordinary value/source
        # protocol. Their sampled parents need no synthetic formula predictor.
        if head !== nothing && head !== LocationScale &&
                !(head isa Type && head <: Distribution) &&
                !(head in _RK_ASSIGNMENT_CALLABLES)
            return true
        end
        if head === LocationScale || (head isa Type && head <: UnivariateDistribution)
            reachable = _brm_reachable_operations(program,
                _brm_prepared_references(_brm_prepare_expr(family)))
            isempty(intersect(predictors, reachable)) && return true
        end
        # A formula response may put a shape before its location (Weibull,
        # Student-t, binomial trials). An assignment in that slot does not
        # make the formula location an arbitrary whole-array reader.
        any(arg -> arg isa NamedColumn && name(arg) in predictors, args) && continue
        first(args) isa NamedColumn && name(first(args)) in assignments && return true
    end
    false
end

function _rk_ast_value_distribution(distribution, bindings, taken)
    # BRM authors Julia's LocationScale/TDist composition; RKPPL authors the
    # same family in Stan argument order as StudentT(nu, location, scale).
    if distribution.callable === LocationScale
        isempty(distribution.kwargs) && length(distribution.args) == 3 || error(
            "RK backend: a Student-t response needs LocationScale(mu, scale, TDist(nu))")
        location, scale, base = distribution.args
        base isa _BRMPreparedExpr && base.callable === TDist &&
            isempty(base.kwargs) && length(base.args) == 1 || error(
            "RK backend: a LocationScale response must wrap TDist(nu)")
        return _rk_ast_dotted(:StudentT,
            _rk_value_expr!(bindings, only(base.args), taken),
            _rk_value_expr!(bindings, location, taken),
            _rk_value_expr!(bindings, scale, taken))
    end
    callee = _rk_value_callee!(bindings, distribution.callable, taken)
    args = map(arg -> _rk_value_expr!(bindings, arg, taken), distribution.args)
    _rk_ast_dotted(callee, args...)
end

_rk_has_value_call(_) = false
function _rk_has_value_call(expression::_BRMPreparedExpr)
    expression.callable in _RK_ASSIGNMENT_CALLABLES || return true
    any(_rk_has_value_call, expression.args) ||
        any(_rk_has_value_call, values(expression.kwargs))
end

function _rk_predictor_components(brmi, context, predictor_order, columns,
        derived, taken, parameters)
    predictors = _RKPredictorSpec[]
    priors = _RKPopulationPrior[]
    r2d2_priors = _RKR2D2Prior[]
    horseshoe_priors = _RKHorseshoePrior[]
    r2d2_vectors = _RKVectorParameter[]
    buckets, lookup = _rk_plan_ranef_buckets(
        brmi, context, predictor_order, columns, taken, derived)
    me_sources = Set{Symbol}()
    matched_defaults = Set{Int}()
    for target in predictor_order
        spec, term_priors, r2d2, hs = _rk_plan_predictor(
            brmi, context, target, Tuple(predictor_order), columns, derived,
            taken, lookup, me_sources; tolerant_default=true, matched_defaults)
        push!(predictors, spec)
        append!(priors, term_priors)
        append!(horseshoe_priors, hs)
        # Structured latent coefficients use dedicated prior cells rather
        # than design columns, but still own a whole-coefficient default.
        if !isempty(term_priors) || !isempty(hs) || r2d2 !== nothing
            for (index, prior) in enumerate(effect_priors(brmi))
                prior.predictor === _EFFECT_COLON &&
                    prior.coefficient === _EFFECT_COLON &&
                    push!(matched_defaults, index)
            end
        end
        r2d2 === nothing && continue
        push!(r2d2_priors, r2d2.prior)
        append!(parameters, r2d2.scalars)
        push!(r2d2_vectors, r2d2.phi)
    end
    _brm_validate_population_effect_defaults(brmi, matched_defaults)
    vectors = [_rk_plan_monotonic_vectors!(predictors); r2d2_vectors]
    (; predictors, priors, r2d2_priors, horseshoe_priors, buckets, vectors)
end

function _brm_rk_value_plan(brmi, program, observations; kernels=(), submodels=())
    context = program.context
    prepared = _brm_prepare_model(brmi; program)
    roots = Set{Symbol}(observation.key for observation in observations)
    routes = (kernels..., submodels...)
    for route in routes
        push!(roots, route.name)
        union!(roots, route.globals)
    end
    union!(roots, (parameter.name for parameter in prepared.parameters))
    referenced = _brm_reachable_operations(program, roots)
    ordinary_assignments = Tuple(a for a in prepared.assignments if a.name in referenced)
    operations = Dict(a.name => a for a in (ordinary_assignments..., routes...))
    assignments = Tuple(operations[name] for name in program.order if haskey(operations, name))
    parameter_names = Set{Symbol}(p.name for p in prepared.parameters)
    assignment_names = Set{Symbol}(a.name for a in assignments)
    consts = Dict{Symbol,Float64}(a.name => Float64(a.expression)
        for a in ordinary_assignments if a.expression isa Number)
    parameters = _rk_plan_parameters!(prepared, context.data, consts,
        Dict{Symbol,Symbol}(), parameter_names, assignment_names)
    vectors = _rk_plan_vector_parameters!(prepared, consts)
    predictor_order = Symbol[op.name for op in program.operations
        if op.role === :predictor && op.name in referenced &&
            !any(route -> route.name === op.name, routes)]
    columns = Dict{Symbol,AbstractVector}()
    derived = _RKDerivedSpec[]
    taken = union(Set(predictor_order), parameter_names, assignment_names,
        Set(v.name for v in vectors), Set(keys(context.data)))
    components = _rk_predictor_components(brmi, context, predictor_order,
        columns, derived, taken, parameters)
    append!(vectors, components.vectors)
    # Regression columns each keep their own row axis. The PPL binder checks
    # their consumers; neither a subject nor a secondary axis is resized to y.
    value_columns = Dict{Symbol,Any}(columns)
    for route in routes
        merge!(value_columns, route.columns)
    end
    for key in referenced
        haskey(context.data, key) || continue
        haskey(value_columns, key) || (value_columns[key] = context.data[key])
    end
    obs = Tuple(o for o in prepared.observations if o.name in roots)
    obs = map(obs) do o
        o.missing_response === nothing || error(
            "RK backend: value-based response `$(o.name)` needs explicit observed values")
        o.weight === nothing || error(
            "RK backend: value-based response `$(o.name)` weights need an authored response")
        layout = isempty(kernels) ?
            (; values=o.response, rows=nothing, lengths=nothing) :
            _rk_kernel_observed_layout(o, kernels)
        modifier = _rk_kernel_response_modifier!(value_columns, taken, o, layout)
        if modifier !== nothing
            bounds = (modifier.lower, modifier.upper)
            if all(b -> b === nothing || b isa Real ||
                    (b isa NamedColumn && parent(b) isa DataColumn), bounds)
                # The same response/row attribution as structural observations.
                materialize = modifier.kind === :interval_censored ?
                    _brm_materialize_interval_response : _brm_materialize_bounded_response
                materialize(modifier, o.name,
                    layout.values, context.data; prefix="RK backend")
            end
        end
        value_columns[o.name] = layout.values
        _BRMPreparedObservation(o.name, o.lhs, o.distribution, o.response,
            modifier, o.weight, o.missing_response)
    end
    regression = _RKStructuralPlan(_RKLikelihoodSpec[], components.predictors,
        components.priors, parameters, _RKAssignmentSpec[], derived, columns,
        0, components.buckets, vectors, components.r2d2_priors,
        components.horseshoe_priors)
    _RKValuePlan(regression, assignments, obs, value_columns)
end

function _rk_emit_value_assignment!(defs, statements, bindings, taken,
        assignment::_BRMPreparedAssignment)
    push!(statements, Expr(:(=), assignment.name,
        _rk_value_expr!(bindings, assignment.expression, taken)))
end

function _rk_value_callee!(bindings, callable, taken)
    # Surface-owned heads use the same canonical names as the GLM emitter.
    if callable in _RK_ASSIGNMENT_CALLABLES || callable isa Type{<:Distribution}
        return nameof(callable)
    end
    index = findfirst(pair -> last(pair) === callable, bindings)
    index === nothing || return first(bindings[index])
    name = _rk_ast_fresh_name("brm_value_function", taken)
    push!(bindings, name => callable)
    name
end

_rk_value_expr!(bindings, value, taken) = value
_rk_value_expr!(bindings, value::_BRMPreparedRef, taken) = value.name
_rk_value_expr!(bindings, values::Tuple, taken) =
    Expr(:tuple, (_rk_value_expr!(bindings, value, taken) for value in values)...)
function _rk_value_expr!(bindings, expression::_BRMPreparedExpr, taken)
    args = map(arg -> _rk_value_expr!(bindings, arg, taken), expression.args)
    expression.callable === getindex && return Expr(:ref, args...)
    expression.callable === Base.vect && return Expr(:vect, args...)
    # BRM expression arithmetic is elementwise. Ordinary RKPPL source must
    # state that explicitly, while reductions and authored whole-array calls
    # keep their own Julia semantics and exact callable bindings.
    if isempty(expression.kwargs)
        if haskey(_RK_DERIVED_BINOPS, expression.callable)
            return Expr(:call, _RK_DERIVED_BINOPS[expression.callable], args...)
        elseif haskey(_RK_DERIVED_CMP, expression.callable)
            return Expr(:call, _RK_DERIVED_CMP[expression.callable], args...)
        elseif haskey(_RK_DERIVED_MATH, expression.callable)
            return _rk_ast_dotted(_RK_DERIVED_MATH[expression.callable], args...)
        end
    end
    callee = _rk_value_callee!(bindings, expression.callable, taken)
    call = Expr(:call, callee, args...)
    if !isempty(expression.kwargs)
        kws = (Expr(:kw, key, _rk_value_expr!(bindings, value, taken))
            for (key, value) in pairs(expression.kwargs))
        insert!(call.args, 2, Expr(:parameters, kws...))
    end
    call
end

function _rk_emit_ast(plan::_RKValuePlan)
    reserved = Set{Symbol}(keys(plan.columns))
    union!(reserved, (a.name for a in plan.assignments))
    regression = _rk_emit_ast(plan.regression, false; values=true, reserved)
    stmts = copy(regression.main.args)
    defs = copy(regression.defs)
    bindings = copy(regression.bindings)
    taken = Set{Symbol}(keys(plan.columns))
    union!(taken, first.(bindings), (a.name for a in plan.assignments),
        (p.name for p in plan.regression.parameters),
        (p.name for p in plan.regression.predictors))
    for assignment in plan.assignments
        _rk_emit_value_assignment!(defs, stmts, bindings, taken, assignment)
    end
    for observation in plan.observations
        modifier = observation.modifier
        distribution = modifier === nothing ? observation.distribution :
            _brm_prepare_expr(modifier.base)
        distribution isa _BRMPreparedExpr || error(
            "RK backend: response `$(observation.name)` needs a distribution call")
        isempty(distribution.kwargs) || error(
            "RK backend: response `$(observation.name)` distribution keywords are unsupported")
        _rk_emit_observation_source!(defs, stmts, bindings, taken,
            observation, distribution) && continue
        base = _rk_ast_value_distribution(distribution, bindings, taken)
        if modifier !== nothing
            lower = modifier.lower === nothing ? -Inf :
                _rk_value_expr!(bindings, _brm_prepare_expr(modifier.lower), taken)
            upper = modifier.upper === nothing ? Inf :
                _rk_value_expr!(bindings, _brm_prepare_expr(modifier.upper), taken)
            base = _rk_ast_response_modifier(base, modifier.kind, lower, upper)
        end
        push!(stmts, Expr(:call, :.~, observation.name, base))
    end
    _rk_fitted_source(_rk_source_program(defs, Expr(:block, stmts...), bindings),
        _rk_observed_names(plan))
end
