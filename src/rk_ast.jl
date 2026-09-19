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
            push!(summands, Expr(:call, :.*, coefs[index], only(term.columns)))
        elseif term.kind === :factor
            # Factor use is always bare `c[g]`; the LevelMap (full cover
            # or subset) rides the broadcast prior, and unmapped rows
            # contribute 0.
            push!(summands, Expr(:ref, coefs[index], only(term.columns)))
        elseif term.kind === :offset
            push!(summands, only(term.columns))
        end
    end
    length(summands) == 1 ? only(summands) :
        Expr(:call, :.+, summands...)
end

function _rk_ast_dotted(head::Symbol, args...)
    Expr(:., head, Expr(:tuple, args...))
end

# The `levels(g)[S]` subset literal dropping position `p` of `K`: edge
# drops spell as explicit literal ranges, middle drops as literal index
# lists (no `end` — the plan knows `K`, so the literal is exact).
function _rk_ast_subset_literal(p::Int, K::Int)
    p == 1 && return Expr(:call, :(:), 2, K)
    p == K && return Expr(:call, :(:), 1, K - 1)
    Expr(:vect, [1:p-1; p+1:K]...)
end

# A factor coefficient's broadcast prior: `c[levels(g)] .~ Normal.(...)`
# full-rank, `c[levels(g)[S]] .~ Normal.(...)` for a reference subset.
# Always stated (factors have no default prior); the scalar location and
# scale broadcast over the LevelMap block.
function _rk_ast_factor_prior(coef::Symbol, col::Symbol,
        options::NamedTuple, K::Int, location::Float64, scale::Float64)
    index = if options.coding === :fullrank
        Expr(:call, :levels, col)
    else
        Expr(:ref, Expr(:call, :levels, col),
            _rk_ast_subset_literal(options.drop, K))
    end
    Expr(:call, :.~, Expr(:ref, coef, index),
        _rk_ast_dotted(:Normal, location, scale))
end

function _rk_ast_response_dist(response::_RKLikelihoodSpec)
    predictor = response.predictor
    base = if response.family === :gaussian
        _rk_ast_dotted(:Normal, predictor, response.scale)
    elseif response.family === :bernoulli_logit
        # Triple 2 and triple 3 both lower to the T2 shape: the affine
        # value feeds logistic either way.
        _rk_ast_dotted(:Bernoulli,
            _rk_ast_dotted(:logistic, predictor))
    elseif response.family === :poisson_log
        _rk_ast_dotted(:Poisson, _rk_ast_dotted(:exp, predictor))
    elseif response.family === :binomial_logit
        # Both triples lower to one spelling: `Binomial.(n,
        # logistic.(p))` with a column or literal `n`.
        _rk_ast_dotted(:Binomial, response.trials,
            _rk_ast_dotted(:logistic, predictor))
    elseif response.family === :nb2_log
        _rk_ast_dotted(:NegativeBinomial2,
            _rk_ast_dotted(:exp, predictor), response.scale)
    elseif response.family === :gamma_log
        # Mean-shape form: the plan pins both alpha positions identical,
        # so the same value emits twice.
        _rk_ast_dotted(:Gamma, response.scale, Expr(:call, :./,
            _rk_ast_dotted(:exp, predictor), response.scale))
    end
    evidence = response.evidence
    # Missing sides emit as ∓Inf floats; the thin layer normalizes them
    # back to nothing at bind.
    dist = if evidence.kind === :truncated
        lo = evidence.lower === nothing ? -Inf : evidence.lower
        hi = evidence.upper === nothing ? Inf : evidence.upper
        _rk_ast_dotted(:truncated, base, lo, hi)
    elseif evidence.kind === :censored
        lo = evidence.lower === nothing ? -Inf : evidence.lower
        hi = evidence.upper === nothing ? Inf : evidence.upper
        _rk_ast_dotted(:censored, base, lo, hi)
    elseif evidence.kind === :interval_censored
        _rk_ast_dotted(:interval_censored, base, evidence.upper)
    else
        base
    end
    response.weights === nothing ? dist :
        _rk_ast_dotted(:weighted, dist, response.weights)
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
            if term.kind === :factor
                col = only(term.columns)
                K = length(_rk_grouping_levels(plan.columns[col]))
                push!(stmts, _rk_ast_factor_prior(
                    coef, col, term.options, K, location, scale))
            else
                push!(stmts, Expr(:call, :~,
                    coef, Expr(:call, :Normal, location, scale)))
            end
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
        push!(stmts, Expr(:call, :.~,
            response.response, _rk_ast_response_dist(response)))
    end
    Expr(:block, stmts...)
end
