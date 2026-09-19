# BRM-side `@rkppl` AST emission (phase 2 U3 retarget). Pure Julia: builds
# the `begin ... end` block `Expr` that the thin layer lowers via
# `lower_rkppl(ast, data_names)` — no ReactiveKernels dependency, so this
# file is committed-testable without the PPL.
#
# Total over slice-1 plans: offset-only predictors emit a bare data
# affine (`mu = z`, no coefficients, no priors), and a predictor sharing
# its name with a data column is alpha-renamed (the single program
# namespace cannot hold both bindings). The AST is the sole emission
# path; the extension holds no fallback serializer.

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
        elseif term.kind === :ranef_gather
            id = term.options.bucket_id
            group = only(term.columns)
            push!(summands, id === nothing ? Expr(:call, :ranef, group) :
                Expr(:call, :ranef, QuoteNode(id), group))
        elseif term.kind === :offset
            push!(summands, only(term.columns))
        elseif term.kind === :spline
            # Direct summand, always inline: the thin layer fails an
            # assigned-then-used `spline(...)` closed (no gather alias).
            push!(summands,
                Expr(:call, :spline, QuoteNode(term.options.id)))
        end
    end
    length(summands) == 1 ? only(summands) :
        Expr(:call, :.+, summands...)
end

# A spline declaration: `spline_basis(:id, axes...; k=k)` — kind is
# inferred thin-layer-side from the axis count (1 → `:tps`, 2 → `:t2`),
# so BRM states only the literal `k` (`Int` for `s`, `(Int, Int)` for
# `t2`). Shape-verified against `Meta.parse` of the surface spelling.
function _rk_ast_spline_basis(term)
    options = term.options
    kval = options.k isa Tuple ? Expr(:tuple, options.k...) : options.k
    Expr(:call, :spline_basis,
        Expr(:parameters, Expr(:kw, :k, kval)),
        QuoteNode(options.id), term.columns...)
end

function _rk_ast_spline_ids(plan::_RKStructuralPlan)
    ids = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :spline || continue
        push!(ids, term.options.id)
    end
    ids
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

# The inverse-link spelling (`Bernoulli.(logistic.(η))`,
# `Poisson.(exp.(η))`, …) is what the `@rkppl` surface takes; the thin
# layer recovers the link-native HAVE from it — the lowered
# `LikelihoodSpec` carries the same family+link the link-faithful
# direct serializer produced, on all six slice-1 families (verified
# behaviorally against `lower_rkppl`, not assumed).
function _rk_ast_response_dist(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol})
    predictor = get(rename, response.predictor, response.predictor)
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
    elseif response.family === :categorical_logit
        # Reference-coded: K−1 non-reference etas, class 1 the implicit
        # zero reference (class order follows predictor order).
        extras = [get(rename, p, p) for p in response.extra_predictors]
        _rk_ast_dotted(:CategoricalLogit, predictor, extras...)
    elseif response.family === :ordered_logit
        # Cutpoints are implicit surface-side (`y_cutpoints`).
        _rk_ast_dotted(:OrderedLogistic, predictor)
    elseif response.family === :ordinal
        # Plain ordinal only: discrimination/per-threshold fail closed at
        # plan (the surface spells three positionals only); thresholds
        # are implicit surface-side (`y_thresholds`).
        structure = response.ordinal_structure === :cumulative ?
            :Cumulative : :StoppingRatio
        linktag = response.link === :logit ? :LogitLink :
            response.link === :probit ? :ProbitLink : :CloglogLink
        _rk_ast_dotted(:Ordinal, Expr(:call, structure),
            Expr(:call, linktag), predictor)
    elseif response.family === :multinomial
        # Lead count column (LHS) + trials + simplex + tail count columns.
        _rk_ast_dotted(:Multinomial, response.trials, predictor,
            response.count_columns...)
    elseif response.family === :categorical
        _rk_ast_dotted(:Categorical, predictor)
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

function _rk_ast_bucket_margin(z::_RKRanefZRecipe)
    z.kind === :ones && return 1
    z.kind === :column && return z.column
    Expr(:call, :dummy, z.column, z.level)
end

# One `ranef_bucket(...) do ... end` block. The do-block shape matches the
# parser's exactly (committed tests compare against `Meta.parse`), so the
# thin layer lowers it like hand-written surface. `eta` rides iff
# `:correlated` (peer rule: K=1 plain buckets take no eta).
function _rk_ast_bucket(bucket::_RKRanefBucket, rename::Dict{Symbol,Symbol})
    lines = Any[]
    for (predictor, cols) in bucket.slices
        elements = Any[_rk_ast_bucket_margin(m.z)
            for m in bucket.margins[cols]]
        push!(lines, Expr(:call, :(=>),
            get(rename, predictor, predictor), Expr(:vect, elements...)))
    end
    call = if bucket.id === nothing
        args = Any[:ranef_bucket, bucket.group]
        bucket.kind === :correlated && insert!(args, 2,
            Expr(:parameters, Expr(:kw, :eta, bucket.lkj_eta)))
        Expr(:call, args...)
    else
        Expr(:call, :ranef_bucket,
            Expr(:parameters, Expr(:kw, :eta, bucket.lkj_eta)),
            QuoteNode(bucket.id), bucket.group)
    end
    Expr(:do, call, Expr(:(->), Expr(:tuple), Expr(:block, lines...)))
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

# Simplex vector parameters emit as `s ~ Dirichlet([...])` (frozen
# concentration vector); threshold vectors are implicit surface-side
# (cutpoints/thresholds), so they emit nothing here.
function _rk_ast_vector_parameter(parameter::_RKVectorParameter)
    parameter.family === :simplex_dirichlet || return nothing
    alpha = only(parameter.args)
    Expr(:call, :~, parameter.name,
        Expr(:call, :Dirichlet, Expr(:vect, alpha...)))
end

function _rk_emit_ast(plan::_RKStructuralPlan)
    taken = union(Set(keys(plan.columns)),
        Set(p.name for p in plan.parameters),
        Set(a.name for a in plan.assignments),
        Set(p.name for p in plan.predictors),
        Set(d.name for d in plan.derived),
        Set(v.name for v in plan.vector_parameters),
        _rk_ast_spline_ids(plan))
    # A predictor sharing its name with a data column cannot keep it:
    # the program has one namespace, so the affine (definition and
    # response uses) is alpha-renamed. Unreachable via `@brm`
    # (observation discovery claims `P ~ …` as a likelihood whenever `P`
    # is observed data), but programmatic plans can still overlap.
    rename = Dict{Symbol,Symbol}()
    for predictor in plan.predictors
        haskey(plan.columns, predictor.name) || continue
        fresh = Symbol(string(predictor.name), "_")
        while fresh in taken
            fresh = Symbol(string(fresh), "_")
        end
        push!(taken, fresh)
        rename[predictor.name] = fresh
    end
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
            (term.kind === :offset || term.kind === :ranef_gather ||
                term.kind === :spline) && continue
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
        for term in predictor.terms
            term.kind === :spline || continue
            push!(stmts, _rk_ast_spline_basis(term))
        end
        push!(stmts, Expr(:(=), get(rename, predictor.name, predictor.name),
            _rk_ast_affine(predictor, coefs)))
    end
    for bucket in plan.ranef_buckets
        push!(stmts, _rk_ast_bucket(bucket, rename))
    end
    for parameter in plan.parameters
        push!(stmts, _rk_ast_sampled(parameter))
    end
    for vector_parameter in plan.vector_parameters
        stmt = _rk_ast_vector_parameter(vector_parameter)
        stmt === nothing || push!(stmts, stmt)
    end
    for assignment in plan.assignments
        push!(stmts, Expr(:(=), assignment.name,
            _rk_lower_assignment_expr(assignment.expression, assignment.name)))
    end
    for response in plan.responses
        push!(stmts, Expr(:call, :.~,
            response.response, _rk_ast_response_dist(response, rename)))
    end
    Expr(:block, stmts...)
end
