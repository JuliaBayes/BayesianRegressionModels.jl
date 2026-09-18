# BRM-side `@rkppl` AST emission (phase 2 U3 retarget). Pure Julia: builds
# the `begin ... end` block `Expr` that the thin layer lowers via
# `lower_rkppl(ast, data_names)` — no ReactiveKernels dependency, so this
# file is committed-testable without the PPL.
#
# Returns `nothing` for the inexpressible subset (offset-only predictors,
# predictor/data name overlap), which the extension routes through the
# legacy direct serializer.

function _rk_lower_assignment_expr(node, name::Symbol)
    node isa Number && return node
    node isa _BRMPreparedRef && return node.name
    node isa _BRMPreparedExpr || error(
        "RK backend: internal: assignment `$name` holds an unlowerable node")
    isempty(node.kwargs) || error(
        "RK backend: internal: assignment `$name` carries keywords " *
        "(slice 1 admits pure positional calls)")
    node.callable isa Function || error(
        "RK backend: internal: assignment `$name` calls a non-function " *
        "callable")
    lowered = map(node.args) do arg
        _rk_lower_assignment_expr(arg, name)
    end
    Expr(:call, nameof(node.callable), lowered...)
end

function _rk_ast_expressible(plan::_RKStructuralPlan)
    columns = plan.columns
    for predictor in plan.predictors
        haskey(columns, predictor.name) && return false
        any(term -> term.kind !== :offset, predictor.terms) || return false
    end
    true
end

function _rk_ast_coef_name(base::String, taken::Set{Symbol})
    name = Symbol(base)
    while name in taken
        name = Symbol(string(name), "_")
    end
    push!(taken, name)
    name
end

function _rk_ast_affine(predictor::_RKPredictorSpec, coefs::Dict{Int,Symbol})
    summands = Any[]
    for (index, term) in enumerate(predictor.terms)
        if term.kind === :intercept
            push!(summands, coefs[index])
        elseif term.kind === :continuous
            push!(summands, Expr(:call, :*, coefs[index], only(term.columns)))
        elseif term.kind === :factor
            col = only(term.columns)
            ref = term.options.ref
            index_expr = ref == 1 ? col :
                Expr(:call, :treatment, col, ref)
            push!(summands, Expr(:ref, coefs[index], index_expr))
        elseif term.kind === :offset
            push!(summands, only(term.columns))
        end
    end
    length(summands) == 1 ? only(summands) :
        Expr(:call, :+, summands...)
end

function _rk_ast_response_dist(response::_RKLikelihoodSpec)
    predictor = response.predictor
    base = if response.family === :gaussian
        Expr(:call, :Normal, predictor, response.scale)
    elseif response.family === :bernoulli_logit
        # Triple 2 and triple 3 both lower to the T2 shape: the affine
        # value feeds logistic either way.
        Expr(:call, :Bernoulli,
            Expr(:call, :logistic, predictor))
    elseif response.family === :poisson_log
        Expr(:call, :Poisson, Expr(:call, :exp, predictor))
    end
    evidence = response.evidence
    # Missing sides emit as ∓Inf floats; the thin layer normalizes them
    # back to nothing at bind.
    dist = if evidence.kind === :truncated
        lo = evidence.lower === nothing ? -Inf : evidence.lower
        hi = evidence.upper === nothing ? Inf : evidence.upper
        Expr(:call, :truncated, base, lo, hi)
    elseif evidence.kind === :censored
        lo = evidence.lower === nothing ? -Inf : evidence.lower
        hi = evidence.upper === nothing ? Inf : evidence.upper
        Expr(:call, :censored, base, lo, hi)
    elseif evidence.kind === :interval_censored
        Expr(:call, :interval_censored, base, evidence.upper)
    else
        base
    end
    response.weights === nothing ? dist :
        Expr(:call, :weighted, dist, response.weights)
end

function _rk_ast_sampled(parameter::_RKSampledParameter)
    name = parameter.name
    family, override = parameter.family, parameter.support_override
    if override === :positive
        head = family === :Cauchy ? :HalfCauchy : :HalfNormal
        return Expr(:call, :~, name,
            Expr(:call, head, parameter.args[2]))
    end
    family === :Flat && return Expr(:call, :~, name, Expr(:call, :Flat))
    Expr(:call, :~, name, Expr(:call, family, parameter.args...))
end

function _rk_emit_ast(plan::_RKStructuralPlan)
    _rk_ast_expressible(plan) || return nothing
    taken = union(Set(keys(plan.columns)),
        Set(p.name for p in plan.parameters),
        Set(a.name for a in plan.assignments),
        Set(p.name for p in plan.predictors),
        Set(d.name for d in plan.derived))
    priors = Dict((p.predictor, p.addressee) => (p.location, p.scale)
        for p in plan.population_priors)
    stmts = Expr[]
    for derived in plan.derived
        push!(stmts, Expr(:(=), derived.name, derived.expression))
    end
    for predictor in plan.predictors
        coefs = Dict{Int,Symbol}()
        counter = 0
        for (index, term) in enumerate(predictor.terms)
            term.kind === :offset && continue
            counter += 1
            coef = _rk_ast_coef_name(
                string(predictor.name, "_b", counter), taken)
            coefs[index] = coef
            key = (predictor.name, term.addressee)
            haskey(priors, key) || error(
                "RK backend: internal: no population prior for " *
                "`$(predictor.name)` addressee `$(term.addressee)`")
            location, scale = priors[key]
            push!(stmts, Expr(:call, :~,
                coef, Expr(:call, :Normal, location, scale)))
        end
        push!(stmts, Expr(:(=), predictor.name,
            _rk_ast_affine(predictor, coefs)))
    end
    for parameter in plan.parameters
        push!(stmts, _rk_ast_sampled(parameter))
    end
    for assignment in plan.assignments
        push!(stmts, Expr(:(=), assignment.name,
            _rk_lower_assignment_expr(assignment.expression, assignment.name)))
    end
    for response in plan.responses
        push!(stmts, Expr(:call, :~,
            response.response, _rk_ast_response_dist(response)))
    end
    Expr(:block, stmts...)
end
