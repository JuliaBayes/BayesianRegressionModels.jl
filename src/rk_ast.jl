# BRM-side `@rkppl` AST emission (phase 2 U3 retarget). Pure Julia: builds
# the `_RKEmittedProgram` (surface-spelling submodel `defs` + the `main`
# `begin ... end` block `Expr`) that the thin layer lowers via
# `lower_rkppl(main, data_names; mod)` after evaluating the defs through
# `@rkppl` — no ReactiveKernels dependency, so this file is
# committed-testable without the PPL.
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

# Thin-layer `_ns` mirror: a submodel local `nm` under use-site LHS
# `lhs` expands to `lhs_nm`. The emitter predicts expanded names with
# this to reserve them in `taken` (no silent merge on collision).
_rk_ast_ns(lhs::Symbol, nm::Symbol) = Symbol(lhs, :_, nm)

# Mint a submodel-local spelling: the candidate must avoid `taken`
# itself (a local shadows any same-named free outer reference in the
# body — arguments fail loud, free names would clobber silently), must
# avoid the def's own argument names (the thin layer rejects arg/local
# overlap), and its expansion must avoid `taken` (no merge with an
# existing program name). Bumps the LOCAL spelling; the expanded name
# follows. Expanded names match the old flat spellings exactly except
# in adversarial edges (data holding a bare `bN` column, or a renamed
# predictor LHS whose prefix differs).
function _rk_ast_mint_local(base::String, lhs::Symbol, taken::Set{Symbol},
        argset::Set{Symbol})
    candidate = Symbol(base)
    while candidate in taken || candidate in argset ||
            _rk_ast_ns(lhs, candidate) in taken
        candidate = Symbol(string(candidate), "_")
    end
    push!(taken, _rk_ast_ns(lhs, candidate))
    candidate
end

_rk_ast_popefs_name(predictor::Symbol) = Symbol(:popefs_, predictor)

# Sorted distinct data columns the affine reads: the submodel's explicit
# inputs (SB passes X explicitly). Outer parameters/atoms (factor coefs,
# spline/hsgp atoms, gp latents, ranef draws) stay free references.
function _rk_ast_popefs_args(predictor::_RKPredictorSpec)
    cols = Set{Symbol}()
    for term in predictor.terms
        (term.kind === :continuous || term.kind === :factor ||
            term.kind === :offset || term.kind === :ranef_gather ||
            term.kind === :monotonic ||
            term.kind === :monotonic_summand) || continue
        union!(cols, term.columns)
    end
    sort!(collect(cols))
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
        elseif term.kind === :monotonic
            # Free-beta monotonic column: `b .* mo(idx, s)` is the only
            # `mo()` shape the thin layer lowers.
            push!(summands, Expr(:call, :.*, coefs[index],
                Expr(:call, :mo, only(term.columns),
                    term.options.increments)))
        elseif term.kind === :monotonic_summand
            # Beta-free direct summand, always inline like `spline(...)`.
            push!(summands, Expr(:call, :mo1, only(term.columns),
                term.options.increments))
        elseif term.kind === :offset
            push!(summands, only(term.columns))
        elseif term.kind === :spline
            # Direct summand, always inline: the thin layer fails an
            # assigned-then-used `spline(...)` closed (no gather alias).
            push!(summands,
                Expr(:call, :spline, QuoteNode(term.options.id)))
        elseif term.kind === :hsgp
            # Direct summand, always inline: the thin layer fails an
            # assigned-then-used `hsgp(...)` closed (no gather alias).
            push!(summands,
                Expr(:call, :hsgp, QuoteNode(term.options.id)))
        elseif term.kind === :gp
            push!(summands, term.options.f)
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

# A hsgp declaration: `hsgp_basis(:id, axes...; k=k, c=c, iso=iso)` —
# `k`/`c` scalars for one axis, per-axis tuples otherwise (the thin
# layer broadcasts scalars). Shape-verified against `Meta.parse` of
# the surface spelling.
function _rk_ast_hsgp_basis(term)
    options = term.options
    kval = options.k isa Tuple ? Expr(:tuple, options.k...) : options.k
    cval = options.c isa Tuple ? Expr(:tuple, options.c...) : options.c
    Expr(:call, :hsgp_basis,
        Expr(:parameters, Expr(:kw, :k, kval), Expr(:kw, :c, cval),
            Expr(:kw, :iso, options.iso)),
        QuoteNode(options.id), term.columns...)
end

function _rk_ast_hsgp_ids(plan::_RKStructuralPlan)
    ids = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :hsgp || continue
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

# The scale-slot formal role per family (Stan/SB-literal where one
# exists): `normal_id_glm(eta, sigma)` reads like its Stan fused form.
_rk_ast_glm_scale_role(family::Symbol) =
    family === :nb2_log ? :phi :
    family === :gamma_log ? :alpha :
    family === :beta_logit ? :kappa : :sigma

_rk_ast_glm_uses_scale(family::Symbol) =
    family === :gaussian || family === :nb2_log ||
    family === :gamma_log || family === :beta_logit

# The scale-slot body spelling inside a stream def. A direct scale
# (outer name, literal, or the plan-forbidden nothing) passes through
# the role formal — the call carries the value, so literal and outer
# scales share one def. A distributional scale predictor inverts its
# link on the role formal exactly like a location predictor (`exp.` for
# log), so the response always reads the constrained vector; the link
# is structural and joins the def name (see `_rk_ast_glm_name`).
# Returns `(formal, body)`.
function _rk_ast_glm_scale_leaf(response::_RKLikelihoodSpec,
        predictor_link::Dict{Symbol,Symbol})
    name = response.scale_predictor
    formal = _rk_ast_glm_scale_role(response.family)
    name === nothing && return formal, formal
    response.scale === nothing || error(
        "RK backend: internal: response `$(response.response)` carries " *
        "both a scalar scale and a scale predictor")
    link = predictor_link[name]
    body = link === :identity ? formal :
        link === :log ? _rk_ast_dotted(:exp, formal) :
        link === :logit ? _rk_ast_dotted(:logistic, formal) :
        error("RK backend: internal: scale predictor `$name` has link `$link`")
    formal, body
end

# The inverse-link spelling (`Bernoulli.(logistic.(η))`,
# `Poisson.(exp.(η))`, …) is what the `@rkppl` surface takes; the thin
# layer recovers the link-native HAVE from it — the lowered
# `LikelihoodSpec` carries the same family+link the link-faithful
# direct serializer produced, on all six slice-1 families (verified
# behaviorally against `lower_rkppl`, not assumed).
#
# `leaf` maps each role to its stream-def BODY spelling: `:predictor`
# (`:eta`, or `:p` for simplex responses), `:scale` (the role formal,
# possibly link-inverted), `:trials`/`:weights`/`:lower`/`:upper`
# (formals; the call carries columns or literals so defs share across
# values), `:extra_predictors` (renamed free refs — categorical tails
# are per-response defs). Evidence and weights STRUCTURE (which
# wrapper, whether weighted) still read from `response`.
function _rk_ast_response_dist(response::_RKLikelihoodSpec,
        leaf::Dict{Symbol,Any})
    predictor = leaf[:predictor]
    base = if response.family === :gaussian
        _rk_ast_dotted(:Normal, predictor, leaf[:scale])
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
        _rk_ast_dotted(:Binomial, leaf[:trials],
            _rk_ast_dotted(:logistic, predictor))
    elseif response.family === :bernoulli_probit
        _rk_ast_dotted(:Bernoulli,
            _rk_ast_dotted(:probit, predictor))
    elseif response.family === :bernoulli_cloglog
        _rk_ast_dotted(:Bernoulli,
            _rk_ast_dotted(:cloglog, predictor))
    elseif response.family === :binomial_probit
        _rk_ast_dotted(:Binomial, leaf[:trials],
            _rk_ast_dotted(:probit, predictor))
    elseif response.family === :binomial_cloglog
        _rk_ast_dotted(:Binomial, leaf[:trials],
            _rk_ast_dotted(:cloglog, predictor))
    elseif response.family === :beta_logit
        # Mean-concentration form: the plan pins mu (the predictor
        # itself) and kappa identical in both positions, so the same
        # values emit twice. `probit`/`cloglog` are thin-layer link
        # words (peel-and-discard, like `logistic`/`exp`); the AST
        # never calls them.
        mu_log = _rk_ast_dotted(:logistic, predictor)
        kappa = leaf[:scale]
        _rk_ast_dotted(:Beta,
            Expr(:call, :.*, mu_log, kappa),
            Expr(:call, :.*, Expr(:call, :.-, 1, mu_log), kappa))
    elseif response.family === :nb2_log
        _rk_ast_dotted(:NegativeBinomial2,
            _rk_ast_dotted(:exp, predictor),
            leaf[:scale])
    elseif response.family === :gamma_log
        # Mean-shape form: the plan pins both alpha positions identical,
        # so the same value emits twice.
        shape = leaf[:scale]
        _rk_ast_dotted(:Gamma, shape, Expr(:call, :./,
            _rk_ast_dotted(:exp, predictor), shape))
    elseif response.family === :categorical_logit
        # Reference-coded: K−1 non-reference etas, class 1 the implicit
        # zero reference (class order follows predictor order).
        _rk_ast_dotted(:CategoricalLogit, predictor,
            leaf[:extra_predictors]...)
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
        _rk_ast_dotted(:Multinomial, leaf[:trials], predictor,
            response.count_columns...)
    elseif response.family === :categorical
        _rk_ast_dotted(:Categorical, predictor)
    end
    evidence = response.evidence
    dist = if evidence.kind === :truncated
        _rk_ast_dotted(:truncated, base, leaf[:lower], leaf[:upper])
    elseif evidence.kind === :censored
        _rk_ast_dotted(:censored, base, leaf[:lower], leaf[:upper])
    elseif evidence.kind === :interval_censored
        _rk_ast_dotted(:interval_censored, base, leaf[:upper])
    else
        base
    end
    response.weights === nothing ? dist :
        _rk_ast_dotted(:weighted, dist, leaf[:weights])
end

# The shared stream-submodel lattice name for a response: mechanical
# `<family>_glm` (+ ordinal structure/link, + distributional-scale
# link, + evidence, + weights), except the Gaussian identity plain
# case, which takes SB's fused name `normal_id_glm` verbatim for
# cross-backend grep-ability. Every structural body input joins the
# name; value inputs (columns, literals, outer names) ride arguments,
# so same name means same body and defs dedupe by name. Per-response
# `glm_<response>` names (see below) prefix with `glm_` while lattice
# names suffix with it, so the two classes cannot collide.
function _rk_ast_glm_name(response::_RKLikelihoodSpec,
        predictor_link::Dict{Symbol,Symbol})
    family = response.family
    if family === :gaussian && response.link === :identity &&
            response.scale_predictor === nothing &&
            response.evidence.kind === :none && response.weights === nothing
        return :normal_id_glm
    end
    parts = Any[family]
    family === :ordinal &&
        push!(parts, response.ordinal_structure, response.link)
    if response.scale_predictor !== nothing
        push!(parts, :dist, predictor_link[response.scale_predictor])
    end
    response.evidence.kind !== :none && push!(parts, response.evidence.kind)
    response.weights !== nothing && push!(parts, :weighted)
    push!(parts, :glm)
    Symbol(join(parts, "_"))
end

# Categorical-logit tails and multinomial counts vary in number per
# response, so no fixed-arity shared def fits: those responses get a
# per-response `glm_<response>` def (same body scheme, bespoke name).
_rk_ast_glm_defname(response::_RKLikelihoodSpec,
        predictor_link::Dict{Symbol,Symbol}) =
    (response.family === :categorical_logit ||
        response.family === :multinomial) ?
    Symbol(:glm_, response.response) :
    _rk_ast_glm_name(response, predictor_link)

# Build the stream-submodel definition + use-site call for a response.
# Formal order is fixed — location, scale role, `n`, `weights`,
# `lower`, `upper` (absent roles skipped) — so shared defs agree by
# construction. Location is `:eta` (pre-link by construction — the body
# applies the inverse link) or `:p` for simplex responses. Missing
# evidence sides pass ∓Inf floats; the thin layer normalizes them back
# to nothing at bind. Returns `(defname, def, call)`.
function _rk_ast_glm_parts(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol},
        taken::Set{Symbol})
    family = response.family
    loc_formal =
        (family === :multinomial || family === :categorical) ? :p : :eta
    formals = Symbol[loc_formal]
    callargs = Any[get(rename, response.predictor, response.predictor)]
    leaf = Dict{Symbol,Any}(:predictor => loc_formal)
    if _rk_ast_glm_uses_scale(family)
        formal, body = _rk_ast_glm_scale_leaf(response, predictor_link)
        leaf[:scale] = body
        push!(formals, formal)
        push!(callargs, response.scale_predictor === nothing ?
            response.scale : get(rename, response.scale_predictor,
                response.scale_predictor))
    end
    if family === :binomial_logit || family === :binomial_probit ||
            family === :binomial_cloglog || family === :multinomial
        leaf[:trials] = :n
        push!(formals, :n)
        push!(callargs, response.trials)
    end
    if response.weights !== nothing
        leaf[:weights] = :weights
        push!(formals, :weights)
        push!(callargs, response.weights)
    end
    kind = response.evidence.kind
    if kind === :truncated || kind === :censored
        leaf[:lower] = :lower
        leaf[:upper] = :upper
        push!(formals, :lower, :upper)
        lower = response.evidence.lower
        upper = response.evidence.upper
        push!(callargs, lower === nothing ? -Inf : lower,
            upper === nothing ? Inf : upper)
    elseif kind === :interval_censored
        leaf[:upper] = :upper
        push!(formals, :upper)
        push!(callargs, response.evidence.upper)
    end
    if family === :categorical_logit
        leaf[:extra_predictors] =
            [get(rename, p, p) for p in response.extra_predictors]
    end
    dist = _rk_ast_response_dist(response, leaf)
    # The response slot: `slot` unless taken (a free tail/count ref with
    # the same spelling would clobber under substitution).
    slot = :slot
    while slot in taken
        slot = Symbol(string(slot), "_")
    end
    defname = _rk_ast_glm_defname(response, predictor_link)
    def = Expr(:(=), Expr(:call, defname, formals...),
        Expr(:block, Expr(:call, :.~, slot, dist), slot))
    call = Expr(:call, :~, response.response,
        Expr(:call, defname, callargs...))
    defname, def, call
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

# A GP latent's `@plate` block: `z[i] ~ Normal(0, 1)` over the using
# response's index (length `n_obs`, like every column). The macrocall
# carries a synthetic line node; the surface reads only `args[3]`.
function _rk_ast_plate(name::Symbol, range::Symbol)
    cell = Expr(:call, :~,
        Expr(:ref, name, :i), Expr(:call, :Normal, 0.0, 1.0))
    loop = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, range)),
        Expr(:block, cell))
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(0), loop)
end

# `gp_chol_latent(gp_exp_quad_cov(x, sigma, rho, jitter), z)`: arg order
# is (locations, sigma, rho, jitter) per the thin-layer contract.
function _rk_ast_gp_latent(term)
    options = term.options
    Expr(:call, :gp_chol_latent,
        Expr(:call, :gp_exp_quad_cov, only(term.columns),
            options.sigma, options.rho, options.jitter),
        options.z)
end

function _rk_ast_gp_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :gp || continue
        options = term.options
        push!(names, options.rho, options.sigma, options.z, options.f)
    end
    names
end

function _rk_emit_ast(plan::_RKStructuralPlan)
    taken = union(Set(keys(plan.columns)),
        Set(p.name for p in plan.parameters),
        Set(a.name for a in plan.assignments),
        Set(p.name for p in plan.predictors),
        Set(d.name for d in plan.derived),
        Set(v.name for v in plan.vector_parameters),
        _rk_ast_spline_ids(plan),
        _rk_ast_hsgp_ids(plan),
        _rk_ast_gp_names(plan))
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
    response_for = Dict{Symbol,Symbol}()
    for response in plan.responses
        haskey(response_for, response.predictor) ||
            (response_for[response.predictor] = response.response)
    end
    defs = Expr[]
    seen_glm = Dict{Symbol,Expr}()
    stmts = Expr[]
    for derived in plan.derived
        push!(stmts, Expr(:(=), derived.name, derived.expression))
    end
    for predictor in plan.predictors
        lhs = get(rename, predictor.name, predictor.name)
        # Scalar-coefficient terms (intercept/continuous/free-beta
        # monotonic) become submodel locals inside a per-predictor
        # `popefs_<pred>` latent submodel; factor terms keep top-level
        # broadcast priors (a `c[levels(g)]` LHS is not a bare Symbol
        # and cannot sit in a submodel body). Counter order is
        # unchanged, so expanded names match the old flat spellings.
        argcols = _rk_ast_popefs_args(predictor)
        argset = Set(argcols)
        coefs = Dict{Int,Symbol}()
        scalar_stmts = Expr[]
        counter = 0
        for (index, term) in enumerate(predictor.terms)
            (term.kind === :offset || term.kind === :ranef_gather ||
                term.kind === :spline || term.kind === :hsgp ||
                term.kind === :gp ||
                term.kind === :monotonic_summand) && continue
            counter += 1
            key = (predictor.name, term.addressee)
            haskey(priors, key) || error(
                "RK backend: internal: no population prior for " *
                "`$(predictor.name)` addressee `$(term.addressee)`")
            location, scale = priors[key]
            if term.kind === :factor
                col = only(term.columns)
                K = length(_rk_grouping_levels(plan.columns[col]))
                coef = _rk_ast_coef_name(
                    string(predictor.name, "_b", counter), taken)
                coefs[index] = coef
                push!(stmts, _rk_ast_factor_prior(
                    coef, col, term.options, K, location, scale))
            else
                local_coef = _rk_ast_mint_local(
                    string("b", counter), lhs, taken, argset)
                coefs[index] = local_coef
                push!(scalar_stmts, Expr(:call, :~,
                    local_coef, Expr(:call, :Normal, location, scale)))
            end
        end
        for term in predictor.terms
            term.kind === :spline || continue
            push!(stmts, _rk_ast_spline_basis(term))
        end
        for term in predictor.terms
            term.kind === :hsgp || continue
            push!(stmts, _rk_ast_hsgp_basis(term))
        end
        for term in predictor.terms
            term.kind === :gp || continue
            options = term.options
            push!(stmts, _rk_ast_sampled(options.rho_param))
            push!(stmts, _rk_ast_sampled(options.sigma_param))
            response = get(response_for, predictor.name, nothing)
            isnothing(response) && error(
                "RK backend: internal: gp predictor `$(predictor.name)` " *
                "feeds no response")
            push!(stmts, _rk_ast_plate(options.z, response))
            push!(stmts, Expr(:(=), options.f, _rk_ast_gp_latent(term)))
        end
        if isempty(scalar_stmts)
            # No scalar coefficients (offset-only, gp-only,
            # factor-only): nothing repeated, nothing to name — the
            # affine stays inline exactly as before.
            push!(stmts, Expr(:(=), lhs, _rk_ast_affine(predictor, coefs)))
        else
            defname = _rk_ast_popefs_name(predictor.name)
            body = Expr(:block, scalar_stmts...,
                _rk_ast_affine(predictor, coefs))
            push!(defs, Expr(:(=),
                Expr(:call, defname, argcols...), body))
            push!(stmts, Expr(:call, :~, lhs,
                Expr(:call, defname, argcols...)))
        end
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
    predictor_link = Dict(spec.name => spec.link for spec in plan.predictors)
    for response in plan.responses
        defname, def, call = _rk_ast_glm_parts(
            response, rename, predictor_link, taken)
        if haskey(seen_glm, defname)
            seen_glm[defname] == def || error(
                "RK backend: internal: glm lattice collision on " *
                "`$defname` (same name, different body)")
        else
            seen_glm[defname] = def
            push!(defs, def)
        end
        push!(stmts, call)
    end
    _RKEmittedProgram(defs, Expr(:block, stmts...))
end
