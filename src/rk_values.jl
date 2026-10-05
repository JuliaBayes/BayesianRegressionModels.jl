# Ordinary model values beside regression formulas. Regression geometry and
# priors use the same planner/emitter as the GLM route; calls cross as values,
# with their exact callable captured in the build's private module.
struct _RKValuePlan
    regression::_RKStructuralPlan
    assignments::Tuple
    observations::Tuple
    columns::Dict{Symbol,Any}
    completions::Tuple
end

_RKValuePlan(regression, assignments, observations, columns) =
    _RKValuePlan(regression, assignments, observations, columns, ())

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

# Native prior source spells declaration bounds as a support restriction,
# preserving the original family kernel. Mathematical truncation keeps its
# own normalized wrapper. This also applies when a prior addresses a log
# hyper-intercept rather than a shared positive scale.
function _rk_ast_declared_prior(prior, bindings, taken)
    prepared = _brm_prepare_expr(prior)
    prepared.callable === truncated &&
        return _rk_ast_truncated_prior(prepared, bindings, taken)
    !any(key -> key in (:lower, :upper), keys(prepared.kwargs)) &&
        return _rk_value_expr!(bindings, prepared, taken)
    base = _BRMPreparedExpr(prepared.callable, prepared.args,
        (; (key => value for (key, value) in pairs(prepared.kwargs)
            if !(key in (:lower, :upper)))...))
    lower = _rk_value_expr!(bindings, get(prepared.kwargs, :lower, -Inf), taken)
    upper = _rk_value_expr!(bindings, get(prepared.kwargs, :upper, Inf), taken)
    Expr(:call, :restricted, _rk_value_expr!(bindings, base, taken), lower, upper)
end

function _rk_ast_truncated_prior(prepared, bindings, taken)
    all(key -> key in (:lower, :upper), keys(prepared.kwargs)) || error(
        "RK backend: a truncated prior accepts only lower/upper bounds")
    args = prepared.args
    if length(args) == 1
        lower = get(prepared.kwargs, :lower, -Inf)
        upper = get(prepared.kwargs, :upper, Inf)
    elseif length(args) == 3 && isempty(prepared.kwargs)
        lower, upper = args[2:3]
    else
        error("RK backend: a truncated prior needs a base law and lower/upper bounds")
    end
    lower === nothing && (lower = -Inf)
    upper === nothing && (upper = Inf)
    Expr(:call, :truncated,
        _rk_value_expr!(bindings, first(args), taken),
        _rk_value_expr!(bindings, lower, taken),
        _rk_value_expr!(bindings, upper, taken))
end

function _rk_ast_positive_prior(prior, bindings, taken; default=:HalfNormal)
    prior === nothing && return default === :LogNormal ?
        Expr(:call, :LogNormal, 0, 1) :
        Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)
    prepared = _brm_prepare_expr(prior)
    prepared.callable === truncated &&
        return _rk_ast_truncated_prior(prepared, bindings, taken)
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

# One varying-effect block, as StanBlocks' `ranef_*_draws` submodels: the
# scales (unless an R2D2 budget derives them and passes them in), standard
# innovations and correlation factor, returning the levels x K draws.
# `scale_priors` is one shared law or a vector of per-margin laws.
function _rk_ast_varying_draws!(definitions, taken, K, eta, rows;
        group=nothing, scale_priors=nothing, scales=nothing)
    block = _rk_block_body(rows, scale_priors)
    axis = group === nothing ? rows :
        Expr(:call, :levels, _rk_block_argument!(block, :g, group))
    index = Expr(:call, :(:), 1, K)
    sd = scales === nothing ? _rk_block_local!(block, :sd) :
        _rk_block_argument!(block, :sd, scales)
    if scale_priors isa AbstractVector
        parts = [_rk_block_local!(block, string(sd, "_", j)) for j in eachindex(scale_priors)]
        for (part, prior) in zip(parts, scale_priors)
            push!(block.statements, Expr(:call, :~, part, prior))
        end
        push!(block.statements, Expr(:(=), sd, Expr(:vect, parts...)))
    elseif scale_priors !== nothing
        push!(block.statements, Expr(:call, :.~, Expr(:ref, sd, index),
            _rk_ast_dotted(scale_priors.args[1], scale_priors.args[2:end]...)))
    end
    z = _rk_block_local!(block, :z)
    push!(block.statements, Expr(:call, :.~, Expr(:ref, z, axis, index),
        _rk_ast_dotted(:Normal, 0, 1)))
    value = if K == 1
        Expr(:call, :.*, z, Expr(:ref, sd, 1))
    else
        L = _rk_block_local!(block, :L)
        push!(block.statements, Expr(:call, :~, L, Expr(:call, :LKJCholesky, K, eta)))
        Expr(:call, :*, z, Expr(:call, :transpose, Expr(:call, :.*, sd, L)))
    end
    base = K == 1 ? "brm_varying_draws" : "brm_correlated_draws"
    _rk_ast_block_call!(definitions, taken, scales === nothing ? base : base * "_r2d2",
        block, value)
end

# `gr(g, by=s)`: per-stratum scales and correlation factors, one row of draws
# per observation (StanBlocks' `ranef_correlated_by_draws`).
function _rk_ast_stratified_draws!(definitions, taken, K, eta, group, stratum)
    block = _rk_block_body()
    g = _rk_block_argument!(block, :g, group)
    s = _rk_block_argument!(block, :s, stratum)
    sd, z, b = _rk_block_local!(block, :sd), _rk_block_local!(block, :z),
        _rk_block_local!(block, :b)
    i = _rk_block_local!(block, :i)
    index = Expr(:call, :(:), 1, K)
    push!(block.statements, Expr(:call, :.~, Expr(:ref, sd,
        Expr(:call, :levels, s), index),
        _rk_ast_dotted(:restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
    L = nothing
    if K > 1
        L, k = _rk_block_local!(block, :L), _rk_block_local!(block, :k)
        cell = Expr(:call, :~, Expr(:ref, L, k), Expr(:call, :LKJCholesky, K, eta))
        push!(block.statements, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
            Expr(:for, Expr(:(=), k, Expr(:call, :levels, s)), Expr(:block, cell))))
    end
    push!(block.statements, Expr(:call, :.~, Expr(:ref, z,
        Expr(:call, :levels, g), index), _rk_ast_dotted(:Normal, 0, 1)))
    scale = Expr(:ref, sd, Expr(:ref, s, i), :(:))
    raw = Expr(:ref, z, Expr(:ref, g, i), :(:))
    value = K == 1 ? Expr(:call, :.*, scale, raw) :
        Expr(:call, :*, Expr(:call, :.*, scale, Expr(:ref, L, Expr(:ref, s, i))), raw)
    cell = Expr(:(=), Expr(:ref, b, i, index), value)
    push!(block.statements, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), i, Expr(:call, :eachindex, g)), Expr(:block, cell))))
    _rk_ast_block_call!(definitions, taken, "brm_stratified_draws", block, b)
end

function _rk_ast_value_bucket(definitions, bucket, draws, effects, taken, bindings;
        predictors=(), population_priors=Dict())
    grouping = bucket.grouping
    K = length(bucket.margins)
    group = first(grouping.columns)
    stmts = Expr[]
    if grouping.form === :mm
        group = _rk_ast_fresh_name(string(draws, "_groups"), taken)
        push!(stmts, Expr(:(=), group, Expr(:call, :vcat, grouping.columns...)))
    end
    call = if grouping.form === :gr
        _rk_ast_stratified_draws!(definitions, taken, K, bucket.lkj_eta,
            group, grouping.by)
    elseif bucket.decomposition !== nothing
        # The shared budget stays a caller-level graph: its scales also feed
        # population priors. The block receives its derived scale vector.
        tau = _rk_ast_fresh_name(string(draws, "_sd"), taken)
        _rk_ast_ranef_r2d2!(definitions, stmts, bucket, tau, bindings, taken,
            predictors, population_priors)
        _rk_ast_varying_draws!(definitions, taken, K, bucket.lkj_eta, nothing;
            group, scales=tau)
    else
        # Stan's ordinary unnamed intercept and multi-membership intercept
        # families sample log_scale ~ Normal(0,1). Shared-ID, slope and
        # stratified families keep their half-normal scale default.
        default = bucket.kind === :intercept1 ? :LogNormal : :HalfNormal
        scale_priors = if all(isequal(first(bucket.sd_priors)), bucket.sd_priors)
            _rk_ast_positive_prior(first(bucket.sd_priors), bindings, taken; default)
        else
            [_rk_ast_positive_prior(prior, bindings, taken) for prior in bucket.sd_priors]
        end
        _rk_ast_varying_draws!(definitions, taken, K, bucket.lkj_eta, nothing;
            group, scale_priors)
    end
    push!(stmts, Expr(:call, :~, draws, call))
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

# A penalized smooth, as StanBlocks' `_sb_s_generic`/`_sb_t2_generic`: the
# fitted bases stay caller-level values; one block allocates the unpenalized
# coefficients, smoothing scales and standardized penalized coefficients.
function _rk_ast_value_spline(definitions, term, taken)
    X = _rk_ast_fresh_name(string(term.options.id, "_X"), taken)
    nblocks = term.options.kind === :t2 ? 3 : 1
    Z = [_rk_ast_fresh_name(string(term.options.id, "_Z", j), taken) for j in 1:nblocks]
    k = term.options.k
    kval = k isa Tuple ? Expr(:tuple, k...) : k
    basis = term.options.kind === :t2 ? :brm_t2_basis : :brm_tps_basis
    stmts = Expr[Expr(:(=), Expr(:tuple, X, Z...),
        Expr(:call, basis, term.columns..., kval))]
    block = _rk_block_body()
    fixed_basis = _rk_block_argument!(block, :X, X)
    penalized = [_rk_block_argument!(block, "Z$j", Zj) for (j, Zj) in enumerate(Z)]
    fixed = _rk_block_local!(block, :fixed)
    push!(block.statements, Expr(:call, :.~,
        Expr(:ref, fixed, Expr(:call, :axes, fixed_basis, 2)), _rk_ast_dotted(:Flat)))
    parts = Any[Expr(:call, :*, fixed_basis, fixed)]
    for (j, basis) in enumerate(penalized)
        sd, raw = _rk_block_local!(block, "sd$j"), _rk_block_local!(block, "raw$j")
        push!(block.statements, Expr(:call, :~, sd,
            Expr(:call, :restricted, Expr(:call, :Normal, 0, 1), 0.0, Inf)))
        push!(block.statements, Expr(:call, :.~,
            Expr(:ref, raw, Expr(:call, :axes, basis, 2)), _rk_ast_dotted(:Normal, 0, 1)))
        push!(parts, Expr(:call, :*, basis, Expr(:call, :.*, sd, raw)))
    end
    push!(stmts, Expr(:call, :~, term.options.id, _rk_ast_block_call!(definitions, taken,
        term.options.kind === :t2 ? "brm_t2_smooth" : "brm_smooth", block,
        Expr(:call, :.+, parts...))))
    stmts
end

function _rk_ast_value_hsgp(definitions, term, taken, bindings)
    options = term.options
    PHI = _rk_ast_fresh_name(string(options.id, "_PHI"), taken)
    lambda = _rk_ast_fresh_name(string(options.id, "_lambda"), taken)
    periodic = get(options, :cov, :exp_quad) === :periodic
    k = options.k isa Tuple ? Expr(:tuple, options.k...) : options.k
    floors = _rk_ast_fresh_name(string(options.id, "_floors"), taken)
    stmts = if periodic
        call =
        Expr(:call, :brm_hsgp_periodic_basis, only(term.columns), k, options.period)
        Expr[Expr(:(=), Expr(:tuple, PHI, lambda, floors), call)]
    else
        _rk_ast_hsgp_basis_graph!(definitions, term, taken, PHI, lambda, floors)
    end
    if haskey(options, :group_index) || !isempty(get(options, :hyper_plans, ()))
        push!(stmts, Expr(:call, :~, options.id,
            _rk_ast_hsgp_grouped(definitions, term, PHI, lambda, floors, taken, bindings)))
        return stmts
    end
    # One block, as StanBlocks' `_sb_hsgp*`: the hyperparameters and basis
    # weights with their priors, returning the HSGP summand. The fitted basis
    # and frequencies stay caller-level graph values, like Stan's data.
    shared = periodic || options.iso
    rho_priors = [_rk_ast_positive_prior(options.rho_prior, bindings, taken)
        for _ in 1:(shared ? 1 : length(term.columns))]
    sigma_prior = _rk_ast_positive_prior(options.sigma_prior, bindings, taken)
    block = _rk_block_body(rho_priors, sigma_prior)
    P = _rk_block_argument!(block, :PHI, PHI)
    omega2 = _rk_block_argument!(block, periodic ? :harmonics : :omega2, lambda)
    floor = options.rho_truncated ? _rk_block_argument!(block, :floor, floors) : nothing
    rhos = map(enumerate(rho_priors)) do (j, prior)
        rho = _rk_block_local!(block, shared ? "rho" : "rho$j")
        options.rho_truncated && (prior = Expr(:call, :restricted, prior,
            shared ? floor : Expr(:ref, floor, j), Inf))
        push!(block.statements, Expr(:call, :~, rho, prior))
        rho
    end
    rho = shared ? only(rhos) : Expr(:vect, rhos...)
    sigma, z = _rk_block_local!(block, :sigma), _rk_block_local!(block, :z)
    push!(block.statements, Expr(:call, :~, sigma, sigma_prior))
    # A basis over a model-derived axis is a graph value, so binding cannot
    # read its column extent; that extent is the formula's basis count.
    extent = get(options, :latent, false) ?
        Expr(:call, :(:), 1, prod(options.k)) : Expr(:call, :axes, P, 2)
    push!(block.statements, Expr(:call, :.~, Expr(:ref, z, extent),
        _rk_ast_dotted(:Normal, 0, 1)))
    value = _rk_block_local!(block, :value)
    push!(block.statements, Expr(:(=), value, periodic ?
        Expr(:call, :*, P, Expr(:call, :.*,
            Expr(:call, :brm_hsgp_periodic_sqrt_spd, omega2, sigma, rho), z)) :
        _rk_ast_hsgp_value_graph!(definitions, term, taken, P, omega2, sigma, rho, z)))
    push!(stmts, Expr(:call, :~, options.id, _rk_ast_block_call!(definitions, taken,
        periodic ? "brm_periodic_hsgp" : "brm_hsgp", block, value)))
    stmts
end

_rk_plan_summary(plan::_RKValuePlan) = string(
    _rk_num_coefficients(plan.regression), " population coefficients and ",
    length(plan.observations), " value-based responses")

function _rk_needs_value_plan(program, observations)
    for observation in observations
        plan = _brm_missing_response_plan(observation.lhs; prefix="RK backend")
        plan === nothing && continue
        _rk_mi_downstream(program, observation.key, plan.source) && return true
    end
    assignments = Set(op.name for op in program.operations if op.role === :assignment)
    predictors = Set(op.name for op in program.operations if op.role === :predictor)
    value_parents = union(assignments, predictors)
    any(op -> op.role === :predictor && any(in(value_parents), op.dependencies),
        program.operations) && return true
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
                (weighted, censored, truncated, interval_censored) && !isempty(getargs(family))
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
    for predictor in predictors, term in predictor.terms
        source = haskey(term.options, :zero_source) ? term.options.zero_source :
            term.kind in (:monotonic, :monotonic_summand) ? term.options.source : nothing
        source === nothing || (columns[source] = context.data[source])
    end
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
    referenced = _brm_reachable_operations(program, roots)
    prepared = _BRMPreparedModel(prepared.program,
        Tuple(p for p in prepared.parameters if p.name in referenced),
        prepared.predictors, prepared.assignments, prepared.observations)
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
    completions = _RKMissingValueSpec[]
    obs = map(obs) do original
        o = _rk_prepare_missing_value!(value_columns, taken, original, program, completions, derived)
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
    _RKValuePlan(regression, assignments, obs, value_columns, Tuple(completions))
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
    observations = Dict(completion.source => completion.observed for completion in plan.completions)
    stmts = map(statement -> _rk_observed_anchor_source(statement, observations), stmts)
    defs = map(definition -> _rk_observed_anchor_source(definition, observations), defs)
    taken = Set{Symbol}(keys(plan.columns))
    union!(taken, first.(bindings), (a.name for a in plan.assignments),
        (p.name for p in plan.regression.parameters),
        (p.name for p in plan.regression.predictors))
    for assignment in plan.assignments
        _rk_emit_value_assignment!(defs, stmts, bindings, taken, assignment)
    end
    for completion in plan.completions
        _rk_emit_missing_value!(defs, stmts, bindings, taken, completion)
    end
    # Keep authored deterministic values as transparent graph computations.
    # The surface's affine likelihood lowering may otherwise absorb their
    # expressions into a location, dropping the named value and its math.
    graph_values = Set(p.name for p in plan.regression.predictors)
    union!(graph_values, (a.name for a in plan.assignments
        if a isa _BRMPreparedAssignment))
    ports = Set{Symbol}(keys(plan.columns))
    foreach(statement -> _rk_source_outputs!(ports, statement), stmts)
    for index in eachindex(stmts)
        statement = stmts[index]
        Meta.isexpr(statement, :(=), 2) || continue
        target, expression = statement.args
        target in graph_values || continue
        inputs = sort!(collect(intersect(
            _rk_source_symbols!(Set{Symbol}(), expression), ports)); by=string)
        callee = _rk_ast_fresh_name(string("brm_value_", target), taken)
        definition = Expr(:(=), Expr(:call, callee, inputs...),
            Expr(:block, Expr(:(=), target, expression)))
        push!(defs, Expr(:macrocall,
            Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))),
            LineNumberNode(0), definition))
        stmts[index] = Expr(:(=), target, Expr(:call, callee, inputs...))
    end
    for observation in plan.observations
        modifier = observation.modifier
        distribution = modifier === nothing ? observation.distribution :
            _brm_prepare_expr(modifier.base)
        distribution isa _BRMPreparedExpr || error(
            "RK backend: response `$(observation.name)` needs a distribution call")
        isempty(distribution.kwargs) || error(
            "RK backend: response `$(observation.name)` distribution keywords are unsupported")
        distribution = _rk_align_kernel_observation_arguments!(defs, stmts, bindings, taken,
            plan, observation, distribution)
        _rk_emit_observation_source!(defs, stmts, bindings, taken,
            observation, distribution, plan.columns[observation.name]) && continue
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
    computed = Set{Symbol}()
    foreach(statement -> Meta.isexpr(statement, :(=), 2) &&
        _rk_source_outputs!(computed, statement), stmts)
    stmts = _rk_order_value_statements(stmts, setdiff(Set(keys(plan.columns)), computed), defs)
    _rk_fitted_source(_rk_source_program(defs, Expr(:block, stmts...), bindings),
        _rk_observed_names(plan))
end
