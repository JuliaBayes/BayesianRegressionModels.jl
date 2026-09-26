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
# `lhs` expands to `lhs_nm`. The emitter predicts expanded names to
# reserve them in `taken` (no silent merge on collision); on collision
# the predictor LHS is alpha-renamed (the def's canonical locals never
# move, so the def stays shared).
_rk_ast_ns(lhs::Symbol, nm::Symbol) = Symbol(lhs, :_, nm)

# The lattice word per affine term kind. The latent-def name is a pure
# function of the ordered skeleton (`popefs_normal_i_c`), so same name
# means same body across all programs: identical skeletons share one
# def, different skeletons cannot collide. Single letters for the GLM
# core, short words elsewhere; `rid` marks a ranef gather carrying an
# explicit bucket id (absent-vs-present is structural — different call
# arity — so it joins the name while the id itself rides an argument).
function _rk_ast_popefs_word(term::_RKTermSpec)
    kind = term.kind
    kind === :intercept && return "i"
    kind === :continuous && return "c"
    kind === :factor && return "f"
    kind === :offset && return "o"
    kind === :spline && return "s"
    kind === :hsgp && return "h"
    kind === :gp && return "gp"
    kind === :ar && return "ar"
    kind === :me && return "me"
    kind === :monotonic && return "mo"
    kind === :monotonic_summand && return "mo1"
    kind === :dar && return "dar"
    if kind === :ranef_gather
        return term.options.bucket_id === nothing ? "r" : "rid"
    end
    error("RK backend: internal: no popefs lattice word for `$kind`")
end

# The shared latent-def name for a predictor: family + ordered skeleton
# words, plus the stated scalar-slot pattern (`_s1_3`) when an R2D2
# predictor states priors on only some slots (unstated slots join the
# simplex with no statement, so the body differs). Fully-stated bodies
# — R2D2 or not — share one name.
function _rk_ast_popefs_lattice(predictor::_RKPredictorSpec,
        stated::Vector{Int}, nscalar::Int)
    parts = Any["popefs", "normal",
        (_rk_ast_popefs_word(t) for t in predictor.terms)...]
    if !isempty(stated) && length(stated) != nscalar
        push!(parts, "s" * join(sort!(copy(stated)), "_"))
    end
    Symbol(join(parts, "_"))
end

# Canonical slot assignment (a pure skeleton function): every column
# slot gets its own `x` formal in term order (sharing a formal across
# slots would bake a per-model dedup pattern into the body), every
# outer reference an `f` formal (`dar` takes two), and every scalar
# coefficient a `b` local. Scalar numbers follow the historical
# counter — factor terms consume a number without taking a local, so
# `1 + factor(g) + x` numbers its locals `b1, b3` exactly as before
# (expanded posterior names are unchanged). Returns the counts plus
# the per-term-index maps (`reff` holds a tuple for `dar`).
function _rk_ast_popefs_slots(predictor::_RKPredictorSpec)
    colf = Dict{Int,Symbol}()
    reff = Dict{Int,Any}()
    number = Dict{Int,Int}()
    nx = 0
    nf = 0
    nb = 0
    nscalar = 0
    for (index, term) in enumerate(predictor.terms)
        kind = term.kind
        if kind === :continuous || kind === :factor || kind === :offset ||
                kind === :monotonic || kind === :monotonic_summand
            nx += 1
            colf[index] = Symbol(:x, nx)
        end
        if kind === :factor || kind === :monotonic ||
                kind === :monotonic_summand || kind === :spline ||
                kind === :hsgp || kind === :gp || kind === :ar ||
                kind === :me
            nf += 1
            reff[index] = Symbol(:f, nf)
        elseif kind === :dar
            reff[index] = (Symbol(:f, nf + 1), Symbol(:f, nf + 2))
            nf += 2
        elseif kind === :ranef_gather
            # Varying spelling: the group column is unused in the body
            # (the effect arrives fully formed); every gather takes one
            # outer-reference formal bound to its slice effect.
            nf += 1
            reff[index] = Symbol(:f, nf)
        end
        if !(kind === :offset || kind === :ranef_gather ||
                kind === :spline || kind === :hsgp || kind === :gp ||
                kind === :dar || kind === :monotonic_summand)
            nb += 1
            number[index] = nb
            kind === :factor || (nscalar += 1)
        end
    end
    (; nx, nf, nb, nscalar, number, colf, reff)
end

function _rk_ast_affine(predictor::_RKPredictorSpec, coefs::Dict{Int,Symbol},
        colref::Dict{Int}, refref::Dict{Int})
    summands = Any[]
    for (index, term) in enumerate(predictor.terms)
        if term.kind === :intercept
            push!(summands, coefs[index])
        elseif term.kind === :continuous
            push!(summands, Expr(:call, :.*, coefs[index], colref[index]))
        elseif term.kind === :factor
            # Factor use is always bare `c[g]`; the LevelMap (full cover
            # or subset) rides the broadcast prior, and unmapped rows
            # contribute 0.
            push!(summands, Expr(:ref, refref[index], colref[index]))
        elseif term.kind === :ranef_gather
            push!(summands, refref[index])
        elseif term.kind === :monotonic
            # Free-beta monotonic column: `b .* mo(idx, s)` is the only
            # `mo()` shape the thin layer lowers.
            push!(summands, Expr(:call, :.*, coefs[index],
                Expr(:call, :mo, colref[index], refref[index])))
        elseif term.kind === :monotonic_summand
            # Beta-free direct summand, always inline like `spline(...)`.
            push!(summands,
                Expr(:call, :mo1, colref[index], refref[index]))
        elseif term.kind === :dar
            # Beta-free trajectory summand, always inline like `mo1(...)`
            # (the surface takes no axis — T is n_obs by construction).
            push!(summands, Expr(:call, :dar,
                refref[index][1], refref[index][2]))
        elseif term.kind === :offset
            push!(summands, colref[index])
        elseif term.kind === :spline
            # Direct summand, always inline: the thin layer fails an
            # assigned-then-used `spline(...)` closed (no gather alias).
            push!(summands, Expr(:call, :spline, refref[index]))
        elseif term.kind === :hsgp
            # Direct summand, always inline: the thin layer fails an
            # assigned-then-used `hsgp(...)` closed (no gather alias).
            push!(summands, Expr(:call, :hsgp, refref[index]))
        elseif term.kind === :gp
            push!(summands, refref[index])
        elseif term.kind === :ar
            # Scaled scan summand: the thin layer classifies
            # `coef .* state` as a ScanSummandTerm (SB's `ar` latent
            # path with its free beta).
            push!(summands, Expr(:call, :.*, coefs[index], refref[index]))
        elseif term.kind === :me
            # Scaled latent summand: the thin layer classifies
            # `coef .* latent` as a ContinuousTerm over the plate
            # vector (SB's `me` true covariate with its free beta).
            push!(summands, Expr(:call, :.*, coefs[index], refref[index]))
        end
    end
    length(summands) == 1 ? only(summands) :
        Expr(:call, :.+, summands...)
end

# An R2D2 declaration: `r2d2(mu, R2, phi[, tau])` — positional
# predictor + R2/phi parameter names, plus the tau sampled-parameter
# name or data literal (BRM always states tau; the thin layer only
# synthesizes it when omitted). Shape-verified against `Meta.parse`
# of the surface spelling.
function _rk_ast_r2d2_decl(r2d2::_RKR2D2Prior, lhs::Symbol)
    Expr(:call, :r2d2, lhs, r2d2.r2, r2d2.phi, r2d2.tau)
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

_rk_ast_response_uses_scale(family::Symbol) =
    family === :gaussian || family === :nb2_log ||
    family === :gamma_log || family === :beta_logit ||
    family === :student_t

# The scale-slot body spelling inside a bare response statement. A
# direct scale (outer name, literal, or the plan-forbidden nothing)
# passes through inline. A distributional scale predictor inverts its
# link on the predictor name exactly like a location predictor (`exp.`
# for log), so the response always reads the constrained vector.
function _rk_ast_response_scale(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol})
    name = response.scale_predictor
    name === nothing && return response.scale
    response.scale === nothing || error(
        "RK backend: internal: response `$(response.response)` carries " *
        "both a scalar scale and a scale predictor")
    actual = get(rename, name, name)
    link = predictor_link[name]
    link === :identity && return actual
    link === :log && return _rk_ast_dotted(:exp, actual)
    link === :logit && return _rk_ast_dotted(:logistic, actual)
    error("RK backend: internal: scale predictor `$name` has link `$link`")
end

# The inverse-link spelling (`Bernoulli.(logistic.(η))`,
# `Poisson.(exp.(η))`, …) is what the `@rkppl` surface takes; the thin
# layer recovers the link-native HAVE from it — the lowered
# `LikelihoodSpec` carries the same family+link the link-faithful
# direct serializer produced, on all six slice-1 families (verified
# behaviorally against `lower_rkppl`, not assumed).
#
# `leaf` maps each role to its INLINE spelling: `:predictor` (the
# predictor name, possibly renamed), `:scale` (the scale value or
# link-inverted scale predictor), `:nu` (the Student-t degrees of
# freedom, literal or name, inline), `:trials`/`:weights`/`:lower`/
# `:upper` (columns or literals inline),
# `:extra_predictors`/`:count_columns` (tail predictors / tail count
# columns inline). Evidence and weights STRUCTURE (which wrapper,
# whether weighted) still read from `response`.
#
# With `fused_heads`, the six families the thin layer desugars
# pre-spine (`BernoulliLogit.(eta)`, `PoissonLog.(eta)`,
# `BinomialLogit.(n, mu)`, `NegativeBinomial2Log.(eta, phi)`,
# `GammaLog.(alpha, eta)`, `BetaLogit.(mu, kappa)`) emit the fused
# spelling; the desugar rewrites each to exactly the decomposed twin
# below (same roles, same order), and evidence/weights wrappers
# recurse, so the lowered plan is identical by construction. Default
# on at the Digest-2 pin; `false` remains the decomposed-twin pin.
#
# Eligible canonical GLMs use the stronger object form:
# `y ~ NormalIDGLM(X, alpha, beta, sigma)` and its two no-scale heads.
# They replace the whole predictor spine (the thin side prepends alpha
# and owns the layout), so they apply only when the planned predictor is
# exactly intercept + numeric continuous columns with ordinary Normal
# population priors. The pinned Digest-2 eligibility keeps every other
# shape — intercept-only, factor/derived/nonlinear terms, transformed
# predictors, evidence/weights, modeled scales, and R2D2 — on the
# decomposed-predic path above.
struct _RKGLMObjectSpec
    response::Symbol
    head::Symbol
    x::Symbol
    alpha::Symbol
    beta::Symbol
    scale::Any
    columns::Vector{Symbol}
    alpha_prior::Tuple{Float64,Float64}
    beta_priors::Vector{Tuple{Float64,Float64}}
end

function _rk_ast_fresh_name(base::String, taken::Set{Symbol})
    name = Symbol(base)
    tail = ""
    while name in taken
        tail *= "_"
        name = Symbol(base, tail)
    end
    push!(taken, name)
    name
end

function _rk_ast_glm_object_prior(priors::Dict{Tuple{Symbol,Symbol},
        Tuple{Float64,Float64}}, predictor::Symbol, addressee::Symbol)
    get(priors, (predictor, addressee), nothing)
end

function _rk_ast_glm_object_spec(response::_RKLikelihoodSpec,
        plan::_RKStructuralPlan, taken::Set{Symbol},
        priors::Dict{Tuple{Symbol,Symbol},Tuple{Float64,Float64}})
    head = response.family === :gaussian ? :NormalIDGLM :
        response.family === :bernoulli_logit ? :BernoulliLogitGLM :
        response.family === :poisson_log ? :PoissonLogGLM : return nothing
    (response.evidence.kind === :none && response.weights === nothing &&
        response.trials === nothing && response.scale_predictor === nothing &&
        isempty(response.extra_predictors) &&
        isempty(response.count_columns)) || return nothing
    count(r -> r.predictor === response.predictor, plan.responses) == 1 ||
        return nothing
    index = findfirst(p -> p.name === response.predictor, plan.predictors)
    index === nothing && return nothing
    predictor = plan.predictors[index]
    wanted_link = response.family === :gaussian ? :identity :
        response.family === :bernoulli_logit ? :identity : :log
    predictor.link === wanted_link || return nothing
    all(t -> t.kind === :intercept || t.kind === :continuous,
        predictor.terms) || return nothing
    intercept_index = findfirst(t -> t.kind === :intercept, predictor.terms)
    continuous = filter(t -> t.kind === :continuous, predictor.terms)
    (intercept_index !== nothing && !isempty(continuous)) || return nothing
    all(t -> length(t.columns) == 1 && haskey(plan.columns, only(t.columns)),
        continuous) || return nothing
    all(t -> (length(t.columns) == 1 &&
        let values = plan.columns[only(t.columns)]
            all(v -> v isa Real && !(v isa Bool), values)
        end), continuous) || return nothing
    intercept = predictor.terms[intercept_index]
    alpha_prior = _rk_ast_glm_object_prior(
        priors, predictor.name, intercept.addressee)
    alpha_prior isa Tuple || return nothing
    beta_priors = Tuple{Float64,Float64}[]
    for term in continuous
        prior = _rk_ast_glm_object_prior(priors, predictor.name,
            term.addressee)
        prior isa Tuple || return nothing
        push!(beta_priors, prior)
    end
    _RKGLMObjectSpec(
        response.response, head,
        _rk_ast_fresh_name(string(response.response, "_X"), taken),
        _rk_ast_fresh_name(string(predictor.name, "_alpha"), taken),
        _rk_ast_fresh_name(string(predictor.name, "_beta"), taken),
        response.scale, Symbol[only(t.columns) for t in continuous],
        alpha_prior, beta_priors)
end

function _rk_ast_glm_object_stmts(spec::_RKGLMObjectSpec)
    alpha_loc, alpha_scale = spec.alpha_prior
    locs = Tuple{Float64,Float64}[p for p in spec.beta_priors]
    locarg = all(p -> p[1] == first(locs)[1], locs) ? first(locs)[1] :
        Expr(:vect, (p[1] for p in locs)...)
    scalearg = all(p -> p[2] == first(locs)[2], locs) ? first(locs)[2] :
        Expr(:vect, (p[2] for p in locs)...)
    stmts = Expr[
        Expr(:(=), spec.x, Expr(:call, :hcat, spec.columns...)),
        Expr(:call, :~, spec.alpha,
            Expr(:call, :Normal, alpha_loc, alpha_scale)),
        Expr(:call, :.~, Expr(:ref, spec.beta,
                Expr(:call, :axes, spec.x, 2)),
            _rk_ast_dotted(:Normal, locarg, scalearg)),
        Expr(:call, :~, spec.response,
            Expr(:call, spec.head, spec.x, spec.alpha, spec.beta,
                (spec.head === :NormalIDGLM ? (spec.scale,) : ())...)),
    ]
    stmts
end

function _rk_ast_response_dist(response::_RKLikelihoodSpec,
        leaf::Dict{Symbol,Any}, fused_heads::Bool,
        wrap_location::Bool=true)
    predictor = leaf[:predictor]
    # Mixture components reuse these branches per component: predictor
    # locations wrap (the decomposed twin — fused heads desugar
    # pre-spine on whole responses only, so components always spell
    # the wrapper form), while param/literal locations ride the
    # constrained scale bare. Single-family responses always wrap.
    base = if response.family === :gaussian
        _rk_ast_dotted(:Normal, predictor, leaf[:scale])
    elseif response.family === :bernoulli_logit
        # Triple 2 and triple 3 both lower to the T2 shape: the affine
        # value feeds logistic either way.
        fused_heads && wrap_location ?
            _rk_ast_dotted(:BernoulliLogit, predictor) :
            wrap_location ? _rk_ast_dotted(:Bernoulli,
                _rk_ast_dotted(:logistic, predictor)) :
            _rk_ast_dotted(:Bernoulli, predictor)
    elseif response.family === :poisson_log
        fused_heads && wrap_location ?
            _rk_ast_dotted(:PoissonLog, predictor) :
            wrap_location ? _rk_ast_dotted(:Poisson,
                _rk_ast_dotted(:exp, predictor)) :
            _rk_ast_dotted(:Poisson, predictor)
    elseif response.family === :binomial_logit
        # Both triples lower to one spelling: `Binomial.(n,
        # logistic.(p))` with a column or literal `n`.
        fused_heads && wrap_location ?
            _rk_ast_dotted(:BinomialLogit, leaf[:trials], predictor) :
            wrap_location ? _rk_ast_dotted(:Binomial, leaf[:trials],
                _rk_ast_dotted(:logistic, predictor)) :
            _rk_ast_dotted(:Binomial, leaf[:trials], predictor)
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
        mu = wrap_location ? _rk_ast_dotted(:logistic, predictor) : predictor
        kappa = leaf[:scale]
        fused_heads && wrap_location ?
            _rk_ast_dotted(:BetaLogit, predictor, kappa) :
            _rk_ast_dotted(:Beta,
                Expr(:call, :.*, mu, kappa),
                Expr(:call, :.*, Expr(:call, :.-, 1, mu), kappa))
    elseif response.family === :nb2_log
        fused_heads && wrap_location ?
            _rk_ast_dotted(:NegativeBinomial2Log, predictor, leaf[:scale]) :
            wrap_location ? _rk_ast_dotted(:NegativeBinomial2,
                _rk_ast_dotted(:exp, predictor),
                leaf[:scale]) :
            _rk_ast_dotted(:NegativeBinomial2, predictor, leaf[:scale])
    elseif response.family === :gamma_log
        # Mean-shape form: the plan pins both alpha positions identical,
        # so the same value emits twice.
        shape = leaf[:scale]
        loc = wrap_location ? _rk_ast_dotted(:exp, predictor) : predictor
        fused_heads && wrap_location ?
            _rk_ast_dotted(:GammaLog, shape, predictor) :
            _rk_ast_dotted(:Gamma, shape, Expr(:call, :./, loc, shape))
    elseif response.family === :student_t
        # Dedicated single head (thin-layer decision, pair fam-student):
        # the plan's `LocationScale(mu, s, TDist(nu))` maps to
        # `StudentT.(nu, mu, sigma)` by arg reorder (Stan
        # `student_t(nu, mu, sigma)` order), the same class of
        # normalization as the existing spelling maps. No
        # `LocationScale` twin: the Normal single-head precedent
        # governs (no link wrap to bridge).
        _rk_ast_dotted(:StudentT, leaf[:nu], predictor, leaf[:scale])
    elseif response.family === :categorical_logit
        # Reference-coded: K−1 non-reference etas, class 1 the implicit
        # zero reference (class order follows predictor order).
        _rk_ast_dotted(:CategoricalLogit, predictor,
            leaf[:extra_predictors]...)
    elseif response.family === :ordered_logit
        # Cutpoints are implicit surface-side (`y_cutpoints`).
        _rk_ast_dotted(:OrderedLogistic, predictor)
    elseif response.family === :ordinal
        # The surface spells three positionals only; discrimination and
        # per-threshold design ride plan-level via the extension, and
        # thresholds are implicit surface-side (`y_thresholds`).
        structure = response.ordinal_structure === :cumulative ?
            :Cumulative : :StoppingRatio
        linktag = response.link === :logit ? :LogitLink :
            response.link === :probit ? :ProbitLink : :CloglogLink
        _rk_ast_dotted(:Ordinal, Expr(:call, structure),
            Expr(:call, linktag), predictor)
    elseif response.family === :multinomial
        # Lead count column (LHS) + trials + simplex + tail count columns.
        _rk_ast_dotted(:Multinomial, leaf[:trials], predictor,
            leaf[:count_columns]...)
    elseif response.family === :categorical
        _rk_ast_dotted(:Categorical, predictor)
    elseif response.family === :mixture
        _rk_ast_mixture_dist(response, leaf)
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

# The class count behind a leveled response: the planned `n_levels`
# when set, else the tail/count arity implies it (K−2 tails, K−1 tail
# counts). A mismatch is an internal error.
function _rk_ast_response_levels(response::_RKLikelihoodSpec)
    family = response.family
    implied = family === :categorical_logit ?
        length(response.extra_predictors) + 2 :
        length(response.count_columns) + 1
    n = response.n_levels
    n === nothing && return implied
    n == implied || error(
        "RK backend: internal: response `$(response.response)` plans " *
        "$n levels but carries $implied")
    n
end

# One mixture component's inline spelling: a synthesized single-family
# spec (the dist branches read family/evidence/weights from the spec —
# evidence/weights are always none here — and every role from the leaf,
# so the spec's own predictor slot is unread) plus that leaf plus the
# wrap flag (predictor locations wrap, params/literals ride bare).
function _rk_ast_mixture_leaves(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol})
    map(response.mixture_components) do comp
        wrap = comp.location_kind === :predictor
        loc = wrap ? get(rename, comp.location::Symbol, comp.location) :
            comp.location
        cspec = _RKLikelihoodSpec(comp.family, comp.link, response.response,
            response.response, comp.scale, comp.scale_predictor, nothing,
            _RKResponseEvidence(:none, nothing, nothing), response.label,
            response.trials, nothing, nothing, Symbol[], Symbol[], nothing,
            nothing, Symbol[], nothing, Symbol[], nothing,
            _RKMixtureComponent[], nothing, nothing)
        cleaf = Dict{Symbol,Any}(:predictor => loc)
        if _rk_ast_response_uses_scale(comp.family)
            cleaf[:scale] =
                _rk_ast_response_scale(cspec, rename, predictor_link)
        end
        comp.family === :binomial_logit &&
            (cleaf[:trials] = response.trials)
        (cspec, cleaf, wrap)
    end
end

# A finite mixture over the planned components: each component emits
# its decomposed twin (fused heads desugar pre-spine on whole
# responses only — components always spell the wrapper form), with
# predictor locations link-wrapped and param/literal locations bare.
# Literal weights inline; a Dirichlet simplex name rides bare.
function _rk_ast_mixture_dist(response::_RKLikelihoodSpec,
        leaf::Dict{Symbol,Any})
    comp_exprs = map(leaf[:mixture]) do (cspec, cleaf, wrap)
        _rk_ast_response_dist(cspec, cleaf, false, wrap)
    end
    weights = response.mixture_weights
    weights_expr = weights isa Vector ? Expr(:vect, weights...) :
        weights isa Symbol ? weights :
        error("RK backend: internal: response `$(response.response)` " *
              "mixture has no planned weights")
    _rk_ast_dotted(:MixtureModel, Expr(:vect, comp_exprs...), weights_expr)
end

# Build the bare response statement for a response: `resp .~ dist`
# with every role spelled inline (predictor and scale-predictor names
# renamed like any other use-site). This is exactly the statement a
# one-shot stream submodel would expand to (the old `*_glm` shells took
# a precomputed predictor while borrowing Stan/SB's fused
# design-matrix-plus-coefficients name, so they were renamed out of
# existence); emitting it directly keeps the lowered program identical
# while the surface stays honest. Missing evidence sides pass ∓Inf
# floats; the thin layer normalizes them back to nothing at bind.
function _rk_ast_response_stmt(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol},
        fused_heads::Bool)
    family = response.family
    leaf = Dict{Symbol,Any}(
        :predictor => get(rename, response.predictor, response.predictor))
    if _rk_ast_response_uses_scale(family)
        leaf[:scale] =
            _rk_ast_response_scale(response, rename, predictor_link)
    end
    if family === :student_t
        # Scalar-only like the scale slot (sampled/assignment names pass
        # through; only predictor names alpha-rename).
        response.nu === nothing && error(
            "RK backend: internal: response `$(response.response)` plans " *
            "Student-t without degrees of freedom")
        leaf[:nu] = response.nu
    end
    if family === :binomial_logit || family === :binomial_probit ||
            family === :binomial_cloglog || family === :multinomial
        leaf[:trials] = response.trials
    end
    if response.weights !== nothing
        leaf[:weights] = response.weights
    end
    kind = response.evidence.kind
    if kind === :truncated || kind === :censored
        lower = response.evidence.lower
        upper = response.evidence.upper
        leaf[:lower] = lower === nothing ? -Inf : lower
        leaf[:upper] = upper === nothing ? Inf : upper
    elseif kind === :interval_censored
        leaf[:upper] = response.evidence.upper
    end
    if family === :categorical_logit
        _rk_ast_response_levels(response)
        leaf[:extra_predictors] =
            Any[get(rename, p, p) for p in response.extra_predictors]
    elseif family === :multinomial
        _rk_ast_response_levels(response)
        leaf[:count_columns] = response.count_columns
    elseif family === :mixture
        leaf[:mixture] =
            _rk_ast_mixture_leaves(response, rename, predictor_link)
    end
    Expr(:call, :.~, response.response,
        _rk_ast_response_dist(response, leaf, fused_heads))
end

function _rk_ast_bucket_margin(z::_RKRanefZRecipe)
    z.kind === :ones && return 1
    z.kind === :column && return z.column
    Expr(:call, :dummy, z.column, z.level)
end

# Varying-name pre-pass (uniform split form): one `ranef_draws_<suffix>`
# per bucket plus one `ranef_<target>_<suffix>` per slice. The `ranef_`
# prefix cannot collide with the thin layer's implicit `r_<target>_<group>`
# term labels; `taken` dedups the rest. Rename-independent (targets use
# original predictor names), so names pre-mint before predictor emission.
function _rk_ast_ranef_names!(plan::_RKStructuralPlan, taken::Set{Symbol})
    draws = Dict{Int,Symbol}()
    effects = Dict{Tuple{Symbol,Symbol,Union{Symbol,Nothing}},Symbol}()
    for (bi, bucket) in enumerate(plan.ranef_buckets)
        suffix = bucket.id === nothing ? string(bucket.group) :
            string(bucket.id, "_", bucket.group)
        draws[bi] = _rk_ast_coef_name("ranef_draws_" * suffix, taken)
        for (target, _) in bucket.slices
            effect = _rk_ast_coef_name(
                "ranef_" * string(target) * "_" * suffix, taken)
            effects[(target, bucket.group, bucket.id)] = effect
        end
    end
    draws, effects
end

# One bucket's varying statements (uniform split form): a draws
# statement plus one slice per target predictor. Shapes match the
# parser's exactly (committed tests compare against `Meta.parse`), so
# the thin layer lowers them like hand-written surface. `eta` rides
# iff `:correlated` (peer rule: K=1 plain buckets take no eta).
function _rk_ast_bucket_stmts(bucket::_RKRanefBucket, draws::Symbol,
        effects::Dict{Tuple{Symbol,Symbol,Union{Symbol,Nothing}},Symbol})
    margins = Any[_rk_ast_bucket_margin(m.z) for m in bucket.margins]
    call = Expr(:call, :varying_draws, bucket.group, Expr(:vect, margins...))
    if bucket.kind === :correlated
        insert!(call.args, 2,
            Expr(:parameters, Expr(:kw, :eta, bucket.lkj_eta)))
    end
    stmts = Expr[Expr(:call, :~, draws, call)]
    for (target, cols) in bucket.slices
        idx = length(cols) == 1 ? first(cols) :
            Expr(:call, :(:), first(cols), last(cols))
        effect = effects[(target, bucket.group, bucket.id)]
        push!(stmts, Expr(:call, :~, effect,
            Expr(:call, :varying_slice, draws, idx)))
    end
    stmts
end

function _rk_ast_sampled(parameter::_RKSampledParameter)
    name = parameter.name
    family, override = parameter.family, parameter.support_override
    if override === :positive
        head = family === :Cauchy ? :HalfCauchy : :HalfNormal
        return Expr(:call, :~, name,
            Expr(:call, head, parameter.args[2]))
    end
    if override === :interval
        # Unit-interval truncated-Normal (dar persistence): the thin-layer
        # screen takes `truncated(Normal(mu, s), 0, 1)` exactly.
        return Expr(:call, :~, name,
            Expr(:call, :truncated,
                Expr(:call, :Normal,
                    parameter.args[1], parameter.args[2]),
                0, 1))
    end
    family === :Flat && return Expr(:call, :~, name, Expr(:call, :Flat))
    if family === :LKJCovarianceFactor
        # SB's covariance-factor declaration, decomposed thin-side into
        # `<stem>_scales` / `<stem>_L_corr`; K/θ/η positional.
        K, theta, eta = parameter.args
        return Expr(:call, :~, name,
            Expr(:call, :LKJCovarianceFactor, K,
                Expr(:call, :Exponential, theta), eta))
    end
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

# A `@plate` block: `name[i] ~ Normal(loc, scale)` over the range
# column's index (length `n_obs`, like every column). GP latents take
# the standardized default; `me` latents take the shared-scalar args.
# The macrocall carries a synthetic line node; the surface reads only
# `args[3]`.
function _rk_ast_plate(name::Symbol, range::Symbol,
        loc::Float64=0.0, scale::Float64=1.0)
    cell = Expr(:call, :~,
        Expr(:ref, name, :i), Expr(:call, :Normal, loc, scale))
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

# An AR(1) latent path: the sampled `phi_raw ~ Normal(0, 1)`, the
# non-centered `@scan` block the thin layer folds through RK-core
# `scan(...)`, and the `phi = tanh(phi_raw)` stationarity map. The
# loop bound `T` is the thin-layer data-length name (binds `n_obs`);
# the seed + innovation shape is SB's `ar1_recurse` verbatim
# (`u[1] = eps[1]`, `u[t] = phi*u[t-1] + eps[t]`). Shape-verified
# against `Meta.parse` of the surface spelling.
function _rk_ast_ar_preamble(term)
    options = term.options
    state, phi, phi_raw, eps =
        options.state, options.phi, options.phi_raw, options.eps
    setup = Expr(:call, :~,
        Expr(:ref, state, 1), Expr(:call, :Normal, 0.0, 1.0))
    innov = Expr(:call, :~,
        eps, Expr(:call, :Normal, 0.0, 1.0))
    carry = Expr(:(=), Expr(:ref, state, :t),
        Expr(:call, :+,
            Expr(:call, :*, phi,
                Expr(:ref, state, Expr(:call, :-, :t, 1))),
            eps))
    loop = Expr(:for, Expr(:(=), :t, Expr(:call, :(:), 2, :T)),
        Expr(:block, innov, carry))
    scan = Expr(:macrocall, Symbol("@scan"), LineNumberNode(0),
        Expr(:block, setup, loop))
    Any[Expr(:call, :~, phi_raw, Expr(:call, :Normal, 0.0, 1.0)),
        scan,
        Expr(:(=), phi, Expr(:call, :tanh, phi_raw))]
end

function _rk_ast_ar_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :ar || continue
        options = term.options
        push!(names, options.state, options.phi, options.phi_raw,
            options.eps)
    end
    names
end

function _rk_ast_dar_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :dar || continue
        options = term.options
        push!(names, options.beta, options.sigma)
    end
    names
end

# A joint correlated-outcomes response emits the plain-`~` vector form
# directly in main (row-grouped, never broadcast — no stream-submodel
# def): `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`.
function _rk_ast_joint_response(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol})
    outcomes = [response.response; response.extra_responses...]
    means = [get(rename, response.predictor, response.predictor);
        [get(rename, p, p) for p in response.extra_predictors]...]
    stem = response.factor
    stem === nothing && error(
        "RK backend: internal: joint response `$(response.label)` has " *
        "no factor stem")
    length(outcomes) == length(means) || error(
        "RK backend: internal: joint response `$(response.label)` has " *
        "$(length(outcomes)) outcomes but $(length(means)) means")
    Expr(:call, :~, Expr(:vect, outcomes...),
        Expr(:call, :MvNormalCholesky, Expr(:vect, means...), stem))
end

function _rk_ast_me_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :me || continue
        push!(names, term.options.latent)
    end
    names
end

function _rk_emit_ast(plan::_RKStructuralPlan, fused_heads::Bool=true)
    taken = union(Set(keys(plan.columns)),
        Set(p.name for p in plan.parameters),
        Set(a.name for a in plan.assignments),
        Set(p.name for p in plan.predictors),
        Set(d.name for d in plan.derived),
        Set(v.name for v in plan.vector_parameters),
        _rk_ast_spline_ids(plan),
        _rk_ast_hsgp_ids(plan),
        _rk_ast_gp_names(plan),
        _rk_ast_dar_names(plan),
        _rk_ast_ar_names(plan),
        _rk_ast_me_names(plan))
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
    # Reserve X/alpha/beta before any other generated name; an eligible
    # GLM consumes its predictor entirely, so no affine names follow.
    glm_objects = Dict{Symbol,_RKGLMObjectSpec}()
    glm_object_predictors = Set{Symbol}()
    if fused_heads
        for response in plan.responses
            spec = _rk_ast_glm_object_spec(response, plan, taken, priors)
            if spec !== nothing
                glm_objects[response.response] = spec
                push!(glm_object_predictors, response.predictor)
            end
        end
    end
    ranef_draws, ranef_effects = _rk_ast_ranef_names!(plan, taken)
    r2d2s = Dict(rp.predictor => rp for rp in plan.r2d2_priors)
    response_for = Dict{Symbol,Symbol}()
    for response in plan.responses
        haskey(response_for, response.predictor) ||
            (response_for[response.predictor] = response.response)
    end
    defs = Expr[]
    seen = Dict{Symbol,Expr}()
    stmts = Expr[]
    for derived in plan.derived
        push!(stmts, Expr(:(=), derived.name, derived.expression))
    end
    # Modeled ordinal scales skip the AST: a discrimination predictor has
    # no response use-site (the surface spells `Ordinal` with three
    # positionals only), so its affine and priors would lower to dead
    # posterior dimensions — the extension translates it plan-level
    # instead. Same predictor-first rule as the planner: a discrimination
    # symbol naming a predictor is a scale (column discriminations match
    # no predictor and need no AST change).
    scales = Set{Symbol}(response.discrimination
        for response in plan.responses if response.discrimination isa Symbol)
    for predictor in plan.predictors
        predictor.name in scales && continue
        predictor.name in glm_object_predictors && continue
        lhs = get(rename, predictor.name, predictor.name)
        r2d2 = get(r2d2s, predictor.name, nothing)
        # Scalar-coefficient terms (intercept/continuous/free-beta
        # monotonic, plus the `ar`/`me` scaling coefs) become canonical
        # `b` locals inside a SHARED lattice-named latent submodel; the
        # name is a pure function of the ordered term skeleton, so
        # identical skeletons share one def across all programs.
        # Factor terms keep top-level broadcast priors (a `c[levels(g)]`
        # LHS is not a bare Symbol and cannot sit in a submodel body)
        # with program-global coefficient names; the affine references
        # them through an `f` formal. Every other body input — data
        # columns (`x`), outer references (`f`), prior locations/scales
        # (`loc`/`s`) — rides a formal in fixed order, so the body
        # carries no baked values. An R2D2 predictor states a prior
        # ONLY for explicit-Normal columns (share-0 overrides); the
        # rest join the simplex with no statement (their scales derive
        # at bind), and the stated-slot pattern joins the def name.
        # Without a scalar statement the affine inlines, so its scalar
        # coefficients need program-global names; with one the submodel
        # path namespaces them.
        flat_scalars = r2d2 !== nothing && !any(
            t -> (t.kind === :intercept || t.kind === :continuous ||
                  t.kind === :monotonic) &&
                haskey(r2d2.overrides, t.addressee),
            predictor.terms)
        slots = _rk_ast_popefs_slots(predictor)
        coefs = Dict{Int,Symbol}()
        factorcoef = Dict{Int,Symbol}()
        colactual = Dict{Int,Any}()
        refactual = Dict{Int,Any}()
        scalar_stmts = Expr[]
        stated = Int[]
        stateloc = Dict{Int,Tuple{Float64,Float64}}()
        for (index, term) in enumerate(predictor.terms)
            kind = term.kind
            if haskey(slots.colf, index)
                colactual[index] = only(term.columns)
            end
            if kind === :monotonic || kind === :monotonic_summand
                refactual[index] = term.options.increments
            elseif kind === :spline || kind === :hsgp
                refactual[index] = QuoteNode(term.options.id)
            elseif kind === :gp
                refactual[index] = term.options.f
            elseif kind === :ar
                refactual[index] = term.options.state
            elseif kind === :me
                refactual[index] = term.options.latent
            elseif kind === :dar
                refactual[index] =
                    (term.options.beta, term.options.sigma)
            elseif kind === :ranef_gather
                refactual[index] = ranef_effects[(predictor.name,
                    term.options.bucket_group, term.options.bucket_id)]
            end
            (kind === :offset || kind === :ranef_gather ||
                kind === :spline || kind === :hsgp ||
                kind === :gp || kind === :dar ||
                kind === :monotonic_summand) && continue
            slot = slots.number[index]
            override = if r2d2 === nothing
                key = (predictor.name, term.addressee)
                haskey(priors, key) || error(
                    "RK backend: internal: no population prior for " *
                    "`$(predictor.name)` addressee `$(term.addressee)`")
                priors[key]
            else
                get(r2d2.overrides, term.addressee, nothing)
            end
            if kind === :factor
                col = only(term.columns)
                K = length(_rk_grouping_levels(plan.columns[col]))
                coef = _rk_ast_coef_name(
                    string(predictor.name, "_b", slot), taken)
                factorcoef[index] = coef
                refactual[index] = coef
                override === nothing || push!(stmts,
                    _rk_ast_factor_prior(coef, col, term.options, K,
                        override[1], override[2]))
            else
                if flat_scalars
                    coefs[index] = _rk_ast_coef_name(
                        string(predictor.name, "_b", slot), taken)
                else
                    local_coef = Symbol(:b, slot)
                    coefs[index] = local_coef
                    if override !== nothing
                        push!(stated, slot)
                        stateloc[slot] = override
                        push!(scalar_stmts, Expr(:call, :~, local_coef,
                            Expr(:call, :Normal, Symbol(:loc, slot),
                                Symbol(:s, slot))))
                    end
                end
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
            term.kind === :dar || continue
            options = term.options
            push!(stmts, _rk_ast_sampled(options.beta_param))
            push!(stmts, _rk_ast_sampled(options.sigma_param))
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
        for term in predictor.terms
            term.kind === :ar || continue
            append!(stmts, _rk_ast_ar_preamble(term))
        end
        for term in predictor.terms
            term.kind === :me || continue
            options = term.options
            push!(stmts, _rk_ast_plate(options.latent,
                only(term.columns), options.loc, options.scale))
        end
        if isempty(scalar_stmts)
            # No scalar statements (offset-only, gp-only,
            # factor-only — or an override-free R2D2 predictor, whose
            # coefficients all join the simplex): the affine stays
            # inline over program-global names exactly as before.
            push!(stmts, Expr(:(=), lhs,
                _rk_ast_affine(predictor, coefs, colactual, refactual)))
        else
            # Canonical use-site: columns, then outer references, then
            # prior locations/scales — the callarg order mirrors the
            # formal order slot by slot (term order, like the
            # assignment).
            formals = Symbol[]
            callargs = Any[]
            for (index, term) in enumerate(predictor.terms)
                haskey(slots.colf, index) || continue
                push!(formals, slots.colf[index])
                push!(callargs, colactual[index])
            end
            for (index, term) in enumerate(predictor.terms)
                haskey(slots.reff, index) || continue
                formal = slots.reff[index]
                if formal isa Tuple
                    append!(formals, formal)
                    append!(callargs, refactual[index])
                else
                    push!(formals, formal)
                    push!(callargs, refactual[index])
                end
            end
            for slot in sort!(stated)
                push!(formals, Symbol(:loc, slot), Symbol(:s, slot))
                loc, scale = stateloc[slot]
                push!(callargs, loc, scale)
            end
            # Expanded locals must avoid `taken`; on collision the LHS
            # is alpha-renamed (the def's canonical locals never move,
            # so the def stays shared).
            localslots = sort!([slots.number[index]
                for (index, term) in enumerate(predictor.terms)
                if term.kind === :intercept || term.kind === :continuous ||
                    term.kind === :monotonic || term.kind === :ar ||
                    term.kind === :me])
            if any(slot -> _rk_ast_ns(lhs, Symbol(:b, slot)) in taken,
                    localslots)
                fresh = Symbol(string(lhs), "_")
                while fresh in taken || any(slot ->
                        _rk_ast_ns(fresh, Symbol(:b, slot)) in taken,
                        localslots)
                    fresh = Symbol(string(fresh), "_")
                end
                push!(taken, fresh)
                rename[predictor.name] = fresh
                lhs = fresh
            end
            for slot in localslots
                push!(taken, _rk_ast_ns(lhs, Symbol(:b, slot)))
            end
            defname =
                _rk_ast_popefs_lattice(predictor, stated, slots.nscalar)
            body = Expr(:block, scalar_stmts...,
                _rk_ast_affine(predictor, coefs, slots.colf, slots.reff))
            def = Expr(:(=), Expr(:call, defname, formals...), body)
            if haskey(seen, defname)
                seen[defname] == def || error(
                    "RK backend: internal: submodel lattice collision " *
                    "on `$defname` (same name, different body)")
            else
                seen[defname] = def
                push!(defs, def)
            end
            push!(stmts, Expr(:call, :~, lhs,
                Expr(:call, defname, callargs...)))
        end
        r2d2 === nothing || push!(stmts, _rk_ast_r2d2_decl(r2d2, lhs))
    end
    for (bi, bucket) in enumerate(plan.ranef_buckets)
        append!(stmts,
            _rk_ast_bucket_stmts(bucket, ranef_draws[bi], ranef_effects))
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
        if response.family === :mvnormal_cholesky
            push!(stmts, _rk_ast_joint_response(response, rename))
            continue
        end
        object = get(glm_objects, response.response, nothing)
        if object !== nothing
            append!(stmts, _rk_ast_glm_object_stmts(object))
            continue
        end
        push!(stmts,
            _rk_ast_response_stmt(response, rename, predictor_link,
                fused_heads))
    end
    _RKEmittedProgram(defs, Expr(:block, stmts...))
end
