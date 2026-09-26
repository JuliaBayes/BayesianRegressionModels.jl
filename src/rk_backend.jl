# Core RK-facing structural plan and slice-1 admission. This file deliberately
# has no dependency on ReactiveKernels or the thin RK-PPL layer: the BRM-side
# structural plan uses BRM-owned structs, and the package extension translates
# them to the thin-layer contract at the boundary (agreed contract text v1+v2,
# co-designed with ReactiveKernels:brm; the thin layer never imports BRM).
#
# Slice-1 admission (user-resolved D3/D4): population GLMs —
# Gaussian/Bernoulli-logit/Poisson-log + frequency/power weights + response
# evidence on Gaussian/Poisson — density+gradient contract — plus the
# random-effects draws regime (non-centered `(x|g)` / `|ID|` buckets
# mirroring SB, peer Stage-A surface). Distributional responses
# (`log(sigma) ~ ...` + `Normal(mu, sigma)`, NB2/Gamma analogues) plan
# with the second predictor in the scale/shape slot; execution needs
# thin-layer vector-scale support. Predictor terms admit raw columns
# plus derived columns (provisional lowering): `&` interactions,
# `center`/`zscale`/`standardize`, and pure numeric data expressions lower
# to thin-layer dotted definitions computed in-graph from raw columns; only
# raw columns cross the boundary. Everything else fails closed with the
# admitted spelling named.

const _RK_ADMITTED_TRIPLES = Set{Tuple{Symbol,Symbol,Symbol}}([
    (:gaussian, :identity, :identity),
    (:bernoulli_logit, :logit, :identity),
    (:bernoulli_logit, :logit, :logit),
    (:poisson_log, :log, :log),
    (:binomial_logit, :logit, :logit),
    (:nb2_log, :log, :log),
    (:gamma_log, :log, :log),
    (:categorical_logit, :logit, :identity),
    (:ordered_logit, :logit, :identity),
    (:ordinal, :logit, :identity),
    (:ordinal, :probit, :identity),
    (:ordinal, :cloglog, :identity),
    # Slice 2 group A (links + Beta): planner-side triples key the @brm
    # link spelling; the AST moves the link into the response wrapper so
    # the thin layer sees its Identity-predlink triples (Binomial
    # precedent).
    (:bernoulli_probit, :probit, :probit),
    (:bernoulli_cloglog, :cloglog, :cloglog),
    (:binomial_probit, :probit, :probit),
    (:binomial_cloglog, :cloglog, :cloglog),
    (:beta_logit, :logit, :logit),
    # Group B (robust): location-scale Student-t over an identity
    # predictor; nu rides its own plan slot (literal / sampled /
    # assignment — no modeled-nu predictor).
    (:student_t, :identity, :identity),
])
# Slice-2 families: no weights or evidence (no driving case — the thin
# layer admits neither on the new triples, so the planner fails closed).
const _RK_SLICE2_FAMILIES = Set{Symbol}([
    :bernoulli_probit, :bernoulli_cloglog, :binomial_probit,
    :binomial_cloglog, :beta_logit, :student_t,
])
# Leveled simplex responses (multinomial/categorical) name a simplex
# vector parameter instead of a linear predictor, so they skip the
# triple (the thin-layer scan-state precedent) and validate on the
# simplex path.
const _RK_SIMPLEX_FAMILIES = Set{Symbol}([:multinomial, :categorical])
const _RK_ORDINAL_LINKS = Dict{Symbol,Symbol}(
    :LogitLink => :logit, :ProbitLink => :probit, :CloglogLink => :cloglog)
const _RK_SLICE1_PRIOR_ARITY = Dict{Symbol,Int}(
    :Normal => 2, :Cauchy => 2, :Exponential => 1, :Gamma => 2,
    :LogNormal => 2, :Beta => 2, :InverseGamma => 2, :Flat => 0)
const _RK_ASSIGNMENT_CALLABLES = Set{Any}([+, -, *, /, ^, log, log10, log1p,
    exp, expm1, sqrt, abs, sum, mean, std, var, minimum, maximum, length])
const _RK_ASSIGNMENT_REDUCTIONS = Set{Any}(
    [sum, mean, std, var, minimum, maximum, length])

# Provisional derived lowering: BRM scalar data-expression heads admitted in
# predictor terms, mapped to the thin-layer dotted vocabulary (mirrors the
# thin-layer ELEMENTWISE_OPS/ELEMENTWISE_FNS/REDUCTION_FNS allowlists).
const _RK_DERIVED_BINOPS = Dict{Function,Symbol}(
    (+) => :.+, (-) => :.-, (*) => :.*, (/) => :./, (^) => :.^, (%) => :.%)
const _RK_DERIVED_MATH = Dict{Function,Symbol}(
    log => :log, log10 => :log10, log1p => :log1p, exp => :exp,
    expm1 => :expm1, sqrt => :sqrt, abs => :abs)
const _RK_DERIVED_CMP = Dict{Function,Symbol}(
    (==) => :.==, (!=) => :.!=, (<) => :.<, (>) => :.>,
    (<=) => :.<=, (>=) => :.>=)
const _RK_DERIVED_REDNAME = Dict{Function,Symbol}(
    sum => :sum, mean => :mean, std => :std, var => :var,
    minimum => :minimum, maximum => :maximum, length => :length)

struct _RKDerivedSpec
    name::Symbol
    expression::Expr # dotted thin-layer body (VectorAssignmentSpec vocabulary)
    label::Symbol
end

struct _RKResponseEvidence
    kind::Symbol # :none | :truncated | :censored | :interval_censored
    lower::Union{Nothing,Float64,Symbol}
    upper::Union{Nothing,Float64,Symbol}
end

# One mixture component: the per-family classification of a single
# `MixtureModel` distribution call. Locations name a linear predictor, a
# sampled scalar parameter, or a numeric literal (params/literals ride the
# constrained scale — no link inversion); scales reuse the scalar-or-
# predictor scale slot; trials ride the shared response-level field.
struct _RKMixtureComponent
    family::Symbol # :gaussian | :bernoulli_logit | :poisson_log |
                   # :binomial_logit | :nb2_log | :gamma_log | :beta_logit
    link::Symbol   # effective link: :identity | :logit | :log
    location::Union{Symbol,Float64}
    location_kind::Symbol # :predictor | :param | :literal
    scale::Union{Nothing,Symbol,Float64}
    scale_predictor::Union{Nothing,Symbol}
end

struct _RKLikelihoodSpec
    family::Symbol # :gaussian | :bernoulli_logit | :poisson_log |
                   # :binomial_logit | :nb2_log | :gamma_log |
                   # :categorical_logit | :ordered_logit | :ordinal |
                   # :multinomial | :categorical | slice-2 group A:
                   # :bernoulli_probit | :bernoulli_cloglog |
                   # :binomial_probit | :binomial_cloglog | :beta_logit |
                   # group B: :student_t |
                   # longtail: :mvnormal_cholesky (joint correlated outcomes)
                   # | :mixture (finite MixtureModel response)
    link::Symbol   # effective link: :identity | :logit | :log |
                   # :probit | :cloglog
    response::Symbol
    predictor::Symbol # leveled simplex responses name their simplex
                      # vector parameter here (no linear predictor)
    scale::Union{Nothing,Symbol,Float64}
    # Distributional scale/shape: a second linear predictor feeding the
    # scale slot (Gaussian sigma, NB2 phi, Gamma alpha). Exactly one of
    # `scale` / `scale_predictor` is non-nothing; the predictor's link
    # inverts at the AST use-site. Execution needs thin-layer
    # vector-scale support (tracked by the vector-scale decision on this
    # lane's queue); the thin-layer lowering fails closed until then.
    scale_predictor::Union{Nothing,Symbol}
    weights::Union{Nothing,Symbol}
    evidence::_RKResponseEvidence
    label::Symbol
    trials::Union{Nothing,Symbol,Int} # binomial/multinomial: data col or literal
    # Leveled trailing fields (thin-layer LikelihoodSpec mirror); every
    # other family leaves them at defaults.
    n_levels::Union{Nothing,Int}
    thresholds::Union{Nothing,Symbol} # ordered/ordinal cutpoint param
    extra_predictors::Vector{Symbol}  # categorical-logit tail (class 3..K)
    count_columns::Vector{Symbol}     # multinomial tail count columns
    ordinal_structure::Union{Nothing,Symbol} # :cumulative | :stopping
    # Reserved for the ordinal-extras surface ask (always defaults until
    # the surface spells discrimination/per-threshold).
    discrimination::Union{Nothing,Float64,Symbol} # ordinal literal or column
    threshold_columns::Vector{Symbol} # ordinal per-threshold design columns
    threshold_coefs::Union{Nothing,Symbol} # per-threshold coef vector param
    # Joint correlated-outcomes trailing fields (thin-layer
    # LikelihoodSpec mirror); every other family leaves them at defaults.
    extra_responses::Vector{Symbol} # joint tail outcome columns (K >= 2)
    factor::Union{Nothing,Symbol} # joint LKJ stem (`L ~ LKJCovarianceFactor`)
    # Finite-mixture trailing fields; every other family leaves them at
    # defaults. Binomial-mixture trials ride the shared `trials` field.
    mixture_components::Vector{_RKMixtureComponent}
    mixture_weights::Union{Nothing,Vector{Float64},Symbol} # frozen
        # literal or Dirichlet simplex-param name
    # Student-t trailing field (thin-layer LikelihoodSpec mirror once the
    # nu slot lands there); every other family leaves it at default.
    nu::Union{Nothing,Float64,Symbol} # literal or sampled/assignment name
end

struct _RKTermSpec
    kind::Symbol # :intercept | :continuous | :factor | :offset |
                # :ranef_gather | :spline | :gp | :hsgp | :ar |
                # :monotonic | :monotonic_summand | :dar | :me
    columns::Vector{Symbol}
    # factor: (coding=:fullrank, levels=:observed) over every observed level,
    # or (coding=:subset, drop::Int, levels=:observed) over every observed
    # level but the `drop`-th (thin-layer `levels(g)` sort order).
    options::NamedTuple
    addressee::Symbol
    label::Symbol
end

struct _RKPredictorSpec
    name::Symbol
    link::Symbol
    terms::Vector{_RKTermSpec}
    label::Symbol
end

struct _RKPopulationPrior
    predictor::Symbol
    addressee::Symbol
    location::Float64
    scale::Float64
end

# Flat whole-predictor R2D2 variance decomposition (SB
# `effect(lp, :) ~ r2d2(...)` mirror; the `sd(...) ~ r2d2(...)`
# R2D2M2/ICC grammar belongs to the hierarchical lane, never here).
# One per predictor at most; a predictor carrying one has NO
# `_RKPopulationPrior` rows (coverage moves here, like the thin-layer
# `R2D2Prior`). `r2`/`phi` name the Beta/Dirichlet sampled parameters
# (planned alongside, in `parameters`/`vector_parameters`); `tau` is a
# sampled half-Normal name or the data `tau_bsv` literal; `overrides`
# maps explicit-Normal addressees to (location, scale) — those columns
# keep their own scale and leave the simplex (share 0).
struct _RKR2D2Prior
    predictor::Symbol
    r2::Symbol
    phi::Symbol
    tau::Union{Symbol,Float64}
    overrides::Dict{Symbol,Tuple{Float64,Float64}}
end

# Per-coefficient structured Horseshoe (SB `effect(lp, coef) ~ Horseshoe(...)`
# mirror, SB-literal per-coefficient tau). One per horseshoe addressee; an
# addressee carrying one has NO `_RKPopulationPrior` row (its prior mass
# moves here, like the thin-layer `HorseshoePrior`). Scales are validated
# positive-finite literals (the shared `_brm_horseshoe_spec` rule).
struct _RKHorseshoePrior
    predictor::Symbol
    addressee::Symbol
    local_scale::Float64
    global_scale::Float64
end

struct _RKSampledParameter
    name::Symbol
    family::Symbol # scalar prior family, or :LKJCovarianceFactor for a
                   # joint factor stem (args `(K::Int, theta, eta)`; the
                   # thin layer derives `<stem>_scales`/`<stem>_L_corr`)
    args::Tuple # Number literals or Symbol param/assignment refs, positional
    support_override::Union{Nothing,Symbol}
    label::Symbol
end

struct _RKAssignmentSpec
    name::Symbol
    expression::Any # scalar _BRMPreparedExpr over folded refs
    label::Symbol
end

struct _RKRanefZRecipe
    kind::Symbol # :ones | :column | :dummy
    column::Symbol # :none for :ones
    level::Union{Nothing,Int,String} # dummy only: raw level value
end

struct _RKRanefMargin
    predictor::Symbol
    coefficient::Symbol # :Intercept | column | <column>_dummy_<value>
    z::_RKRanefZRecipe
end

struct _RKRanefBucket
    id::Union{Nothing,Symbol}
    group::Symbol # bound grouping column (raw; crossed strings if categorical)
    kind::Symbol # :intercept1 | :slope1 | :correlated
    margins::Vector{_RKRanefMargin}
    slices::Vector{Tuple{Symbol,UnitRange{Int}}}
    lkj_eta::Float64 # correlated only; NaN otherwise
    label::Symbol # :bucket_<suffix>
end

struct _RKVectorParameter
    name::Symbol
    family::Symbol # :ordered_normal | :vector_normal | :simplex_dirichlet
    args::Tuple # literals: (location, scale) / (concentration-vector,)
    size::Union{Nothing,Int}
    label::Symbol
end

struct _RKStructuralPlan
    responses::Vector{_RKLikelihoodSpec}
    predictors::Vector{_RKPredictorSpec}
    population_priors::Vector{_RKPopulationPrior}
    parameters::Vector{_RKSampledParameter}
    assignments::Vector{_RKAssignmentSpec}
    derived::Vector{_RKDerivedSpec}
    columns::Dict{Symbol,AbstractVector}
    n_obs::Int
    ranef_buckets::Vector{_RKRanefBucket}
    vector_parameters::Vector{_RKVectorParameter}
    r2d2_priors::Vector{_RKR2D2Prior}
    horseshoe_priors::Vector{_RKHorseshoePrior}
end

# A submodel-bearing emitted program: `defs` are surface-spelling
# `sm(args...) = begin ... end` definitions (the extension evaluates
# each through `@rkppl` in a fresh module per lowering); `main` is the
# `begin ... end` block lowered via `lower_rkppl(main, data_names;
# mod)`. Kernel plans carry no defs (a single plate, no top-level
# repeated structure — `plate() do` bodies are not expansion sites).
struct _RKEmittedProgram
    defs::Vector{Expr}
    main::Expr
end

"""
    RKBRMI(brmi; kwargs...)

A [`BRMI`](@ref) lowered to the ReactiveKernels backend. `plan` is the strict,
RK-independent structural plan; `model` is the executable thin-layer program
provided by `BayesianRegressionModelsReactiveKernelsExt` when ReactiveKernels
is loaded. Implemented only by that extension; the generic here lets the core
validate and materialise plans without loading RK.
"""
struct RKBRMI{P<:BRMI,PL,M}
    parent::P
    plan::PL
    model::M
end

Base.parent(x::RKBRMI) = x.parent
structure_of(x::RKBRMI) = structure_of(parent(x))
priors_of(x::RKBRMI) = priors_of(parent(x))

# Thin-layer `levels(g)` order, mirrored exactly (sort of observed values;
# `CategoricalValue`/non-`String` rows string-normalize, as in the thin
# layer's `_grouping_levels`). Every position below (subset drops,
# coefficient counts) is a position in THIS order.
function _rk_grouping_levels(col::AbstractVector)
    v = first(col)
    if v isa CA.CategoricalValue ||
            (v isa AbstractString && !isa(v, String))
        return sort!(unique!(string.(col)))
    end
    return sort(unique(col))
end

function _rk_num_coefficients(plan::_RKStructuralPlan)
    total = 0
    for predictor in plan.predictors, term in predictor.terms
        if term.kind === :intercept || term.kind === :continuous ||
                term.kind === :ar || term.kind === :monotonic ||
                term.kind === :me
            total += 1
        elseif term.kind === :factor
            width = length(_rk_grouping_levels(
                plan.columns[only(term.columns)]))
            total += term.options.coding === :fullrank ? width : width - 1
        end
    end
    total
end

_rk_plan_summary(plan::_RKStructuralPlan) = string(
    _rk_num_coefficients(plan), " population coefficients and ",
    plan.n_obs, " observations")
# `_rk_plan_summary(::_RKKernelPlan)` is defined with that type, below.
Base.show(io::IO, x::RKBRMI) = print(io, "RKBRMI with ", _rk_plan_summary(x.plan))

# Implemented only by the ReactiveKernels package extension. Keeping the
# generic here lets the core validate and materialise plans without loading RK.
function _brm_rk_model end

"""
    rk_logdensity_problem(backend::RKBRMI; ad_backend, u0) -> problem

A `LogDensityProblems`-compatible density over the backend's packed
unconstrained coordinates (order 1: value + gradient). `ad_backend` is a
`DifferentiationInterface` AD type (e.g. reverse-mode `AutoEnzyme`);
`u0` is a length-consistent exemplar (default: zeros). Implemented only by
the `BayesianRegressionModelsReactiveKernelsExt` package extension.
"""
function rk_logdensity_problem end

"""
    rk_restore_draws(backend::RKBRMI, U::AbstractMatrix) -> NamedTuple

Restore named constrained parameters from an unconstrained draws matrix `U`
(`dimension` rows × draws columns, e.g. sampler output): coefficient
predictors map to `(size × draws)` matrices, sampled parameters to
length-`draws` vectors. Implemented only by the
`BayesianRegressionModelsReactiveKernelsExt` package extension.
"""
function rk_restore_draws end

const _RK_ADMITTED_SPELLINGS =
    "`y ~ Normal(mu, s)` + `mu ~ ...`, `y ~ BernoulliLogit(eta)` (or " *
    "`Bernoulli(logistic(eta))`) + `eta ~ ...`, `y ~ Bernoulli(p)` + " *
    "`logit(p) ~ ...`, `y ~ Poisson(mu)` + `log(mu) ~ ...`, " *
    "`H ~ Binomial(n, p)` + `logit(p) ~ ...` (or `Binomial(n, " *
    "logistic(eta))` + `eta ~ ...`), `y ~ NegativeBinomial2(mu, phi)` + " *
    "`log(mu) ~ ...`, `y ~ Gamma(alpha, mu/alpha)` + `log(mu) ~ ...`, " *
    "`y ~ CategoricalLogit(eta_2, ..., eta_K)` + identity `eta_j ~ ...`, " *
    "`y ~ OrderedLogistic(eta)` + `eta ~ ...`, `y ~ Ordinal(structure, " *
    "link, eta)` + `eta ~ 0 + ...`, `obs ~ Multinomial(N, s)` + " *
    "`s ~ Dirichlet(...)`, `y ~ Categorical(s)` + `s ~ Dirichlet(...)`, " *
    "slice-2 group A: `y ~ Bernoulli(p)` / `H ~ Binomial(n, p)` + " *
    "`probit(p)` / `cloglog(p) ~ ...`, `y ~ Beta(mu*kappa, " *
    "(1-mu)*kappa)` + `logit(mu) ~ ...`, group B: `y ~ LocationScale(mu, " *
    "s, TDist(nu))` + `mu ~ ...`, `[y1, y2] ~ " *
    "MvNormalCholesky([mu1, mu2], L)` + `L ~ LKJCovarianceFactor(K; " *
    "...)` + identity `mu_j ~ ...`, or `y ~ MixtureModel([D1, ..., " *
    "DK], w)` (K >= 1 same-family Normal/Bernoulli/Poisson/Binomial/" *
    "NegativeBinomial2/Gamma/Beta components + literal/Dirichlet weights)"

function _rk_predictor_link(brmi::BRMI, target::Symbol)
    prefix = "RK backend"
    op = linear_predictor_op(brmi, target)
    lhs, _ = getargs(op, 2)
    link_fn, _ = _peel_lp_lhs(lhs)
    link_fn === identity && return :identity
    link_fn isa Function || error(
        "$prefix: predictor `$target` has an uninterpretable link; " *
        "admitted links are identity, logit, log, probit, and cloglog")
    name = nameof(link_fn)
    name === :logit && return :logit
    name === :log && return :log
    name === :probit && return :probit
    name === :cloglog && return :cloglog
    error("$prefix: predictor `$target` uses link `$name`; admitted links " *
          "are identity, logit, log, probit, and cloglog")
end

_rk_is_predictor_ref(arg, candidates::AbstractVector{Symbol}) =
    arg isa NamedColumn && name(arg) in candidates
_rk_is_predictor_ref(arg, predictor::Symbol) =
    arg isa NamedColumn && name(arg) === predictor

# How location-slot messages spell the admitted predictor(s): one candidate
# reads exactly as before; two name the choice.
_rk_lp_phrase(candidates::AbstractVector{Symbol}) =
    length(candidates) == 1 ? "the linear predictor `$(only(candidates))`" :
    "one of the referenced linear predictors ($(join(candidates, ", ")))"

# Same, for the Bernoulli/Binomial probability slot (bare or `logistic` wrap).
_rk_probability_phrase(candidates::AbstractVector{Symbol}) =
    length(candidates) == 1 ?
    "the linear predictor `$(only(candidates))` or " *
    "`logistic($(only(candidates)))`" :
    "one of the referenced linear predictors " *
    "($(join(candidates, ", "))) or the `logistic` wrap of one of them"

# The location predictor: the candidate the location slot names.
function _rk_location_arg(arg, candidates::Vector{Symbol}, response::Symbol,
        slot::String, tail::String)
    prefix = "RK backend"
    arg isa NamedColumn && name(arg) in candidates && return name(arg)
    error("$prefix: response `$response` $slot must be " *
          "$(_rk_lp_phrase(candidates)) $tail")
end

function _rk_strip_logistic(arg, candidates::AbstractVector{Symbol})
    # `Bernoulli(logistic(eta))` lowers identically to `BernoulliLogit(eta)`;
    # the emitter strips the wrapper so the contract sees one spelling.
    # Returns the wrapped candidate's name (the location predictor).
    arg isa ExprColumn && getf(arg) === logistic || return nothing
    args = getargs(arg)
    length(args) == 1 && _rk_is_predictor_ref(only(args), candidates) ||
        return nothing
    isempty(getkwargs(arg)) || return nothing
    name(only(args))
end

# Returns `(scale, scale_predictor)` with exactly one side non-nothing: a
# scalar (parameter/assignment/literal) or a distributional bare-predictor
# reference. A predictor shadows a same-named data column here, matching
# the location slot (a data column can never be a scale, so the LP reading
# is the only useful one).
function _rk_scale_argument(arg, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, response::Symbol, what::String,
        candidates::Vector{Symbol})
    prefix = "RK backend"
    arg isa NamedColumn && name(arg) in candidates &&
        return (nothing, name(arg))
    arg isa Number && return (_rk_positive_literal(arg, response, what), nothing)
    if arg isa NamedColumn
        parent(arg) isa DataColumn && error(
            "$prefix: response `$response` $what cannot be a data column; " *
            "slice 1 admits a sampled parameter, a scalar assignment, or " *
            "a positive numeric literal")
        kind, value = _rk_resolve_use_ref(name(arg), consts, aliases,
            parameters, assignments, "response `$response` $what")
        kind === :number &&
            return (_rk_positive_literal(value, response, what), nothing)
        return (value, nothing)
    end
    error("$prefix: response `$response` $what must be a sampled parameter, " *
          "a positive numeric literal, or the second linear predictor " *
          "(`log(sigma) ~ ...` + bare `sigma`; deterministic wrappers " *
          "such as `exp(...)` spell as an LP link instead)")
end

# Binomial trials: an integer data column (values validated once columns
# cross) or a non-negative integer literal. Folded constants ride the
# literal path; sampled parameters, live assignments, and expressions
# cannot — the thin layer takes `ColumnRef | Int` only.
function _rk_trials_argument(arg, response::Symbol,
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol})
    prefix = "RK backend"
    arg isa Number && return _rk_trials_literal(arg, response)
    if arg isa NamedColumn
        parent(arg) isa DataColumn && return name(arg)
        kind, value = _rk_resolve_use_ref(name(arg), consts, aliases,
            parameters, assignments, "response `$response` trials")
        kind === :number && return _rk_trials_literal(value, response)
        error("$prefix: response `$response` trials must be an integer " *
              "data column or a non-negative integer literal, not a " *
              "sampled parameter or assignment")
    end
    error("$prefix: response `$response` trials must be an integer data " *
          "column or a non-negative integer literal")
end

function _rk_trials_literal(x::Number, response::Symbol)
    prefix = "RK backend"
    isfinite(Float64(x)) && x == round(x) && x >= 0 || error(
        "$prefix: response `$response` trials literal must be a " *
        "non-negative integer")
    Int(x)
end

# Student-t degrees of freedom: the `LocationScale` base must be a
# `TDist(nu)` call whose nu is a positive literal, a sampled parameter,
# or a scalar assignment. A linear predictor in the nu slot is a
# modeled-nu response (no driving case); a data column can never be a
# scalar. Returns the literal value or the resolved name.
function _rk_nu_argument(base, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, response::Symbol,
        candidates::Vector{Symbol})
    prefix = "RK backend"
    base isa ExprColumn && getf(base) === TDist || error(
        "$prefix: response `$response` `LocationScale` base must be " *
        "`TDist(nu)`; write `LocationScale(mu, s, TDist(nu))` with a " *
        "`mu ~ ...` predictor")
    bargs = getargs(base)
    length(bargs) == 1 || error(
        "$prefix: response `$response` `LocationScale` base `TDist` " *
        "needs one argument; write `TDist(nu)`")
    isempty(getkwargs(base)) || error(
        "$prefix: response `$response` `LocationScale` base `TDist` " *
        "takes no keywords")
    arg = only(bargs)
    arg isa Number && return _rk_positive_literal(arg, response,
        "degrees of freedom")
    if arg isa NamedColumn
        name(arg) in candidates && error(
            "$prefix: response `$response` degrees of freedom cannot be " *
            "the linear predictor `$(name(arg))`; modeled nu is out of " *
            "slice — write a sampled parameter, a positive literal, or " *
            "a scalar assignment (if a same-named parameter exists, " *
            "rename one of them)")
        parent(arg) isa DataColumn && error(
            "$prefix: response `$response` degrees of freedom cannot be " *
            "a data column; slice 1 admits a sampled parameter, a " *
            "scalar assignment, or a positive numeric literal")
        kind, value = _rk_resolve_use_ref(name(arg), consts, aliases,
            parameters, assignments, "response `$response` degrees of freedom")
        kind === :number && return _rk_positive_literal(value, response,
            "degrees of freedom")
        return value
    end
    error("$prefix: response `$response` degrees of freedom must be a " *
          "sampled parameter, a positive numeric literal, or a scalar " *
          "assignment")
end

# Gamma mean-shape form: `Gamma(alpha, mu/alpha)` with the SAME alpha in
# both positions (same name, or equal literals) — mirrors the thin
# layer's double-alpha identity rule. The shape may be a scalar or a
# distributional bare-predictor reference; the numerator names the location
# predictor. Returns `(shape, shape_predictor, location)`.
function _rk_gamma_shape_args(args, candidates::Vector{Symbol},
        response::Symbol, parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol})
    prefix = "RK backend"
    # One candidate renders exactly the old messages; two spell the location
    # as a metavariable (it is identified by the numerator check below).
    loc = length(candidates) == 1 ? string(only(candidates)) : "<location>"
    length(args) == 2 || error(
        "$prefix: response `$response` `Gamma` needs `(shape, scale)`; " *
        "write `Gamma(alpha, $loc/alpha)` with a `log($loc)` " *
        "predictor")
    shape, shape_predictor = _rk_scale_argument(args[1], parameters,
        assignments, consts, aliases, response, "shape", candidates)
    scale = args[2]
    scale isa ExprColumn && getf(scale) === (/) || error(
        "$prefix: response `$response` `Gamma` scale must be " *
        "`$loc/alpha` with the same shape in both positions")
    isempty(getkwargs(scale)) || error(
        "$prefix: response `$response` `Gamma` scale takes no keywords")
    sargs = getargs(scale)
    length(sargs) == 2 && _rk_is_predictor_ref(sargs[1], candidates) || error(
        "$prefix: response `$response` `Gamma` scale must be " *
        "`$loc/alpha` with $(_rk_lp_phrase(candidates)) itself")
    _rk_gamma_same_alpha(args[1], sargs[2]) || error(
        "$prefix: response `$response` `Gamma` shape and scale divisor " *
        "must be identical (`Gamma(alpha, $loc/alpha)`); distinct " *
        "shapes are out of slice 1")
    shape, shape_predictor, name(sargs[1])
end

function _rk_gamma_same_alpha(shape_arg, divisor_arg)
    shape_arg isa Number && divisor_arg isa Number &&
        return shape_arg == divisor_arg
    shape_arg isa NamedColumn && divisor_arg isa NamedColumn &&
        return name(shape_arg) == name(divisor_arg) &&
            typeof(parent(shape_arg)) === typeof(parent(divisor_arg))
    false
end

# Beta mean-concentration form: `Beta(mu*kappa, (1-mu)*kappa)` with the
# SAME mu (the linear predictor itself) and the SAME kappa in both
# positions — mirrors the thin layer's structural-match rule. Either
# multiplication order admits; anything else fails closed. Returns the
# concentration value.
function _rk_beta_shape_args(args, predictor::Symbol, response::Symbol,
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol})
    prefix = "RK backend"
    length(args) == 2 || error(
        "$prefix: response `$response` `Beta` needs `(a, b)`; write " *
        "`Beta($predictor*kappa, (1-$predictor)*kappa)` with a " *
        "`logit($predictor)` predictor")
    main_factors = _rk_beta_split_product(
        args[1], predictor, response, "first")
    comp_factors = _rk_beta_split_product(
        args[2], predictor, response, "second")
    main_mu, main_kappa = _rk_beta_orient_factors(main_factors,
        f -> _rk_is_predictor_ref(f, predictor), predictor, response,
        "first", "`$predictor*kappa`")
    comp_mu, comp_kappa = _rk_beta_orient_factors(comp_factors,
        f -> _rk_beta_is_complement(f, predictor), predictor, response,
        "second", "`(1-$predictor)*kappa`")
    _rk_gamma_same_alpha(main_kappa, comp_kappa) || error(
        "$prefix: response `$response` `Beta` concentration must be " *
        "identical in both positions (`Beta($predictor*kappa, " *
        "(1-$predictor)*kappa)`); distinct concentrations are out of " *
        "slice 2")
    # Scalar-only: Beta-kappa predictors are deferred (decision 005dq0u),
    # so the concentration never resolves against predictor candidates.
    concentration, concentration_predictor = _rk_scale_argument(
        main_kappa, parameters, assignments, consts,
        aliases, response, "concentration", Symbol[])
    concentration_predictor === nothing || error(
        "$prefix: response `$response` `Beta` concentration cannot be a " *
        "linear predictor (predictor-fed concentration is not admitted)")
    concentration
end

# Split one `Beta` position into its two factors: a bare product, no
# keywords. Orientation (which factor is mu-side) happens in
# `_rk_beta_orient_factors`, so either multiplication order admits.
function _rk_beta_split_product(position, predictor::Symbol,
        response::Symbol, which::String)
    position isa ExprColumn && getf(position) === (*) ||
        error("RK backend: response `$response` `Beta` $which argument " *
              "must be a product (`$predictor*kappa` / " *
              "`(1-$predictor)*kappa`)")
    isempty(getkwargs(position)) ||
        error("RK backend: response `$response` `Beta` $which argument " *
              "takes no keywords")
    pargs = getargs(position)
    length(pargs) == 2 ||
        error("RK backend: response `$response` `Beta` $which argument " *
              "must be a two-factor product")
    return pargs[1], pargs[2]
end

# Orient `(mu-side, kappa-side)`: the factor passing `mu_test` is mu.
function _rk_beta_orient_factors(factors, mu_test, predictor::Symbol,
        response::Symbol, which::String, form::String)
    f1, f2 = factors
    mu_test(f1) && return f1, f2
    mu_test(f2) && return f2, f1
    error("RK backend: response `$response` `Beta` $which argument must " *
          "be $form")
end

_rk_beta_is_complement(factor, predictor::Symbol) =
    factor isa ExprColumn && getf(factor) === (-) &&
    let cargs = getargs(factor)
        length(cargs) == 2 && cargs[1] isa Number && cargs[1] == 1 &&
            _rk_is_predictor_ref(cargs[2], predictor)
    end

function _rk_positive_literal(x::Number, response::Symbol, what::String)
    prefix = "RK backend"
    value = Float64(x)
    isfinite(value) && value > 0 || error(
        "$prefix: response `$response` $what must be finite and positive")
    value
end

# Classifies the peeled distribution call against the response's
# referenced predictors (one, or two for a distributional response). The
# location predictor is identified positionally per family; a second
# candidate in the scale/shape slot becomes `scale_predictor`. Leveled
# families return the same shape (location is the lead predictor; the
# categorical-logit tail arrives via `extra`). Returns
# `(; family, link, scale, scale_predictor, trials, location)` plus a `nu`
# key on the Student-t arm only; the caller rejects unclaimed candidates
# and a scale slot naming the location.
function _rk_classify_response(rhs::ExprColumn, candidates::Vector{Symbol},
        predictor_link::Dict{Symbol,Symbol}, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, response::Symbol,
        extra::Vector{Symbol} = Symbol[],
        extra_links::Vector{Symbol} = Symbol[])
    prefix = "RK backend"
    head = getf(rhs)
    args = getargs(rhs)
    # Ordinal carries its SB keywords (`discrimination`, `per_threshold`);
    # the arm below admits exactly those two.
    head !== Ordinal && (isempty(getkwargs(rhs)) || error(
        "$prefix: response `$response` distribution keywords are out of " *
        "slice 1; admitted spellings: $_RK_ADMITTED_SPELLINGS"))
    if head === Normal
        length(args) == 2 || error(
            "$prefix: response `$response` `Normal` needs `(location, scale)`")
        location = _rk_location_arg(args[1], candidates, response, "location",
            "itself, not a deterministic transform; write the transform " *
            "into the predictor formula")
        plink = predictor_link[location]
        scale, scale_predictor = _rk_scale_argument(args[2], parameters,
            assignments, consts, aliases, response, "scale", candidates)
        triple = (:gaussian, plink, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `Normal` with a " *
            "$plink-link predictor; slice 1 admits " *
            "$_RK_ADMITTED_SPELLINGS")
        return (; family=:gaussian, link=plink, scale, scale_predictor,
            trials=nothing, location)
    elseif head === BernoulliLogit
        length(args) == 1 || error(
            "$prefix: response `$response` `BernoulliLogit` needs one argument")
        location = _rk_location_arg(only(args), candidates, response,
            "argument", "itself")
        # NOTE: triple (:bernoulli_logit, :logit, :logit) exists for the
        # plain-`Bernoulli` head only; a `BernoulliLogit` head on a
        # logit-link predictor would apply the link twice.
        plink = predictor_link[location]
        plink === :identity || error(
            "$prefix: response `$response` applies `BernoulliLogit` on top " *
            "of a $plink-link predictor (double link); use an " *
            "identity-link predictor")
        return (; family=:bernoulli_logit, link=:logit, scale=nothing,
            scale_predictor=nothing, trials=nothing, location)
    elseif head === Bernoulli
        length(args) == 1 || error(
            "$prefix: response `$response` `Bernoulli` needs one argument")
        arg = only(args)
        if _rk_is_predictor_ref(arg, candidates)
            location = name(arg)
            plink = predictor_link[location]
            # The triple table is the admission key: plain `Bernoulli(p)`
            # over a logit/probit/cloglog predictor (slice-2 group A adds
            # the latter two). Identity stays closed — the identity
            # spelling wraps `logistic` (stripped below).
            for family in (:bernoulli_logit, :bernoulli_probit,
                    :bernoulli_cloglog)
                (family, plink, plink) in
                    _RK_ADMITTED_TRIPLES || continue
                return (; family, link=plink, scale=nothing,
                    scale_predictor=nothing, trials=nothing, location)
            end
            error(
                "$prefix: response `$response` pairs plain `Bernoulli` with " *
                "a $plink-link predictor; write `BernoulliLogit` " *
                "with an identity predictor or `Bernoulli(p)` with a " *
                "`logit(p)`/`probit(p)`/`cloglog(p)` predictor")
        end
        stripped = _rk_strip_logistic(arg, candidates)
        stripped === nothing || predictor_link[stripped] === :identity || error(
            "$prefix: response `$response` applies `logistic` on top of a " *
            "$(predictor_link[stripped])-link predictor (double link); use an " *
            "identity-link predictor")
        stripped === nothing && error(
            "$prefix: response `$response` probability must be " *
            "$(_rk_probability_phrase(candidates)); admitted " *
            "spellings: $_RK_ADMITTED_SPELLINGS")
        return (; family=:bernoulli_logit, link=:logit, scale=nothing,
            scale_predictor=nothing, trials=nothing, location=stripped)
    elseif head === Poisson
        length(args) == 1 || error(
            "$prefix: response `$response` `Poisson` needs one argument")
        location = _rk_location_arg(only(args), candidates, response, "rate",
            "itself; write `Poisson(mu)` with a `log(mu)` predictor " *
            "(slice 1 has no `Poisson(exp(..))` spelling)")
        plink = predictor_link[location]
        triple = (:poisson_log, plink, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `Poisson` with a " *
            "$plink-link predictor; write `Poisson(mu)` with a " *
            "`log(mu)` predictor")
        return (; family=:poisson_log, link=plink, scale=nothing,
            scale_predictor=nothing, trials=nothing, location)
    elseif head === Binomial
        length(args) == 2 || error(
            "$prefix: response `$response` `Binomial` needs `(trials, " *
            "probability)`")
        trials = _rk_trials_argument(args[1], response, parameters,
            assignments, consts, aliases)
        arg = args[2]
        if _rk_is_predictor_ref(arg, candidates)
            location = name(arg)
            plink = predictor_link[location]
            # Triple-table admission like `Bernoulli` above (slice-2
            # group A adds probit/cloglog).
            for family in (:binomial_logit, :binomial_probit,
                    :binomial_cloglog)
                (family, plink, plink) in
                    _RK_ADMITTED_TRIPLES || continue
                return (; family, link=plink, scale=nothing,
                    scale_predictor=nothing, trials, location)
            end
            error(
                "$prefix: response `$response` pairs `Binomial` with a " *
                "$plink-link predictor; write `Binomial(n, p)` " *
                "with a `logit(p)`/`probit(p)`/`cloglog(p)` predictor or " *
                "`Binomial(n, logistic(eta))` with an identity predictor")
        end
        stripped = _rk_strip_logistic(arg, candidates)
        stripped === nothing || predictor_link[stripped] === :identity || error(
            "$prefix: response `$response` applies `logistic` on top of a " *
            "$(predictor_link[stripped])-link predictor (double link); use an " *
            "identity-link predictor")
        stripped === nothing && error(
            "$prefix: response `$response` probability must be " *
            "$(_rk_probability_phrase(candidates)); admitted " *
            "spellings: $_RK_ADMITTED_SPELLINGS")
        return (; family=:binomial_logit, link=:logit, scale=nothing,
            scale_predictor=nothing, trials, location=stripped)
    elseif head === NegativeBinomial2
        length(args) == 2 || error(
            "$prefix: response `$response` `NegativeBinomial2` needs " *
            "`(mean, dispersion)`; write `NegativeBinomial2(mu, phi)` " *
            "with a `log(mu)` predictor")
        location = _rk_location_arg(args[1], candidates, response, "mean",
            "itself; write `NegativeBinomial2(mu, phi)` with a `log(mu)` " *
            "predictor (slice 1 has no `NegativeBinomial2(exp(..))` spelling)")
        plink = predictor_link[location]
        dispersion, dispersion_predictor = _rk_scale_argument(args[2],
            parameters, assignments, consts, aliases, response, "dispersion",
            candidates)
        triple = (:nb2_log, plink, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `NegativeBinomial2` with " *
            "a $plink-link predictor; write " *
            "`NegativeBinomial2(mu, phi)` with a `log(mu)` predictor")
        return (; family=:nb2_log, link=plink, scale=dispersion,
            scale_predictor=dispersion_predictor, trials=nothing, location)
    elseif head === BinomialLogit
        error("$prefix: response `$response` `BinomialLogit` is out of " *
              "slice 1; write `Binomial(n, p)` with a `logit(p)` " *
              "predictor or `Binomial(n, logistic(eta))` with an " *
              "identity predictor")
    elseif head === Gamma
        shape, shape_predictor, location =
            _rk_gamma_shape_args(args, candidates, response, parameters,
                assignments, consts, aliases)
        plink = predictor_link[location]
        triple = (:gamma_log, plink, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `Gamma` with a " *
            "$plink-link predictor; write `Gamma(alpha, " *
            "mu/alpha)` with a `log(mu)` predictor")
        return (; family=:gamma_log, link=plink, scale=shape,
            scale_predictor=shape_predictor, trials=nothing, location)
    elseif head === OrderedLogistic
        length(args) == 1 || error(
            "$prefix: response `$response` `OrderedLogistic` needs one " *
            "argument; write `OrderedLogistic(eta)` with an `eta ~ ...` " *
            "predictor")
        predictor = only(candidates)
        _rk_is_predictor_ref(only(args), predictor) || error(
            "$prefix: response `$response` location must be the linear " *
            "predictor `$predictor` itself")
        plink = predictor_link[predictor]
        triple = (:ordered_logit, :logit, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `OrderedLogistic` with " *
            "a $plink-link predictor; write `OrderedLogistic(eta)` " *
            "with an identity-link predictor")
        return (; family=:ordered_logit, link=:logit, scale=nothing,
            scale_predictor=nothing, trials=nothing, location=predictor)
    elseif head === Ordinal
        length(args) == 3 || error(
            "$prefix: response `$response` `Ordinal` needs `(structure, " *
            "link, eta)`; write `Ordinal(Cumulative(), LogitLink(), eta)` " *
            "with an `eta ~ 0 + ...` predictor")
        structure = _brm_ordinal_tag(args[1], OrdinalStructure; prefix)
        link_tag = _brm_ordinal_tag(args[2], OrdinalLink; prefix)
        link = _RK_ORDINAL_LINKS[nameof(typeof(link_tag))]
        # The location is whichever candidate the eta slot names (a
        # modeled discrimination adds a second candidate).
        loc = args[3]
        loc isa NamedColumn && name(loc) in candidates || error(
            "$prefix: response `$response` `Ordinal` location must be " *
            "$(_rk_lp_phrase(candidates)) itself")
        location = name(loc)
        _brm_ordinal_has_fixed_intercept(args[3]) && error(
            "$prefix: `Ordinal($response)` cannot include a fixed intercept " *
            "in `eta`; the estimated thresholds already supply the location. " *
            "Use `eta ~ 0 + ...`.")
        plink = predictor_link[location]
        triple = (:ordinal, link, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `Ordinal` with a " *
            "$plink-link predictor; write `Ordinal(structure, " *
            "link, eta)` with an identity-link predictor")
        return (; family=:ordinal, link, scale=nothing,
            scale_predictor=nothing, trials=nothing, location)
    elseif head === CategoricalLogit
        lead = only(candidates)
        preds = [lead; extra...]
        length(args) == length(preds) || error(
            "$prefix: response `$response` `CategoricalLogit` arguments " *
            "must be its resolved non-reference predictors " *
            "($(join(preds, ", ")))")
        for (arg, owned) in zip(args, preds)
            _rk_is_predictor_ref(arg, owned) || error(
                "$prefix: response `$response` `CategoricalLogit` argument " *
                "must be the linear predictor `$owned` itself (class order " *
                "follows argument order)")
        end
        for (owned, plink) in
                zip(preds, [predictor_link[lead]; extra_links...])
            triple = (:categorical_logit, :logit, plink)
            triple in _RK_ADMITTED_TRIPLES || error(
                "$prefix: response `$response` pairs `CategoricalLogit` " *
                "with a $plink-link predictor `$owned`; write identity-link " *
                "predictors (`eta_j ~ ...`)")
        end
        return (; family=:categorical_logit, link=:logit, scale=nothing,
            scale_predictor=nothing, trials=nothing, location=lead)
    elseif head === Beta
        # Beta-kappa predictors are deferred (decision 005dq0u): one
        # location predictor only — a second LP fails closed here with a
        # plain error, not an uninterpretable `only` throw.
        length(candidates) == 1 || error(
            "$prefix: response `$response` `Beta` takes one location " *
            "predictor; predictor-fed concentration is not admitted")
        predictor = only(candidates)
        concentration = _rk_beta_shape_args(args, predictor, response,
            parameters, assignments, consts, aliases)
        plink = predictor_link[predictor]
        triple = (:beta_logit, plink, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `Beta` with a " *
            "$plink-link predictor; write `Beta(mu*kappa, " *
            "(1-mu)*kappa)` with a `logit(mu)` predictor (slice 2 " *
            "admits a logit mu link only)")
        return (; family=:beta_logit, link=plink, scale=concentration,
            scale_predictor=nothing, trials=nothing, location=predictor)
    elseif head === LocationScale
        length(args) == 3 || error(
            "$prefix: response `$response` `LocationScale` needs " *
            "`(location, scale, base)`; write `LocationScale(mu, s, " *
            "TDist(nu))` with a `mu ~ ...` predictor")
        location = _rk_location_arg(args[1], candidates, response, "location",
            "itself, not a deterministic transform; write the transform " *
            "into the predictor formula")
        plink = predictor_link[location]
        scale, scale_predictor = _rk_scale_argument(args[2], parameters,
            assignments, consts, aliases, response, "scale", candidates)
        nu = _rk_nu_argument(args[3], parameters, assignments, consts,
            aliases, response, candidates)
        triple = (:student_t, plink, plink)
        triple in _RK_ADMITTED_TRIPLES || error(
            "$prefix: response `$response` pairs `LocationScale` with a " *
            "$plink-link predictor; write `LocationScale(mu, s, " *
            "TDist(nu))` with an identity-link predictor")
        return (; family=:student_t, link=plink, scale, scale_predictor,
            trials=nothing, location, nu)
    elseif head === MvNormalCholesky
        # Joint correlated-outcomes response (SB
        # `[y1..yK] ~ MvNormalCholesky([mu1..muK], L)`): Phase 4 resolved
        # the means positionally (`candidates` holds the lead, `extra`
        # the tail), so only the factor stem and the identity links are
        # left here. The stem's LKJ shape and width gate in Phase 6
        # (linkage needs the planned parameters).
        length(args) == 2 || error(
            "$prefix: response `$response` `MvNormalCholesky(means, " *
            "factor)` needs exactly two arguments")
        means, factor = args
        means isa AbstractVector || error(
            "$prefix: response `$response` `MvNormalCholesky` means must " *
            "use vector syntax `[mu1, mu2, ...]`, got $(typeof(means))")
        factor isa NamedColumn || error(
            "$prefix: response `$response` `MvNormalCholesky` factor must " *
            "name an `LKJCovarianceFactor` declaration, got " *
            "$(typeof(factor))")
        stem = name(factor)
        stem in parameters || error(
            "$prefix: response `$response` joint factor `$stem` must be " *
            "a sampled parameter, not observed data; declare it with " *
            "`$stem ~ LKJCovarianceFactor(K; ...)`")
        location = first(candidates)
        linked = Tuple{Symbol,Symbol}[(location, predictor_link[location]);
            ((p, l) for (p, l) in zip(extra, extra_links))...]
        bad = [p for (p, l) in linked if l !== :identity]
        isempty(bad) || error(
            "$prefix: response `$response` joint mean predictor(s) " *
            "$(join(bad, ", ")) must be identity-link (joint means " *
            "enter the MvNormal directly)")
        return (; family=:mvnormal_cholesky, link=:identity, scale=nothing,
            scale_predictor=nothing, trials=nothing, location, factor=stem)
    end
    head_name = head isa Function ? nameof(head) :
        head isa Type ? nameof(head) : string(head)
    error("$prefix: response `$response` family `$head_name` is not " *
          "admitted; admitted spellings: $_RK_ADMITTED_SPELLINGS")
end

# ---- finite mixtures (MixtureModel responses) ----
#
# Each component classifies like a single-family response of its family
# (same arg shapes, same predictor-link rules), generalized only in the
# location slot: a linear predictor, a sampled scalar parameter, or a
# numeric literal. Params/literals ride the constrained scale — no link
# inversion — so the AST wraps predictor locations and inlines the rest
# bare. Components share one Julia head (SB's `allequal(Ts)` rule: a
# `BernoulliLogit`/`Bernoulli` mix is two RK spellings of one RK family
# but SB rejects it, so RK rejects it too — everything admitted here
# has an SB number).

# Component args are numeric-or-names all the way down (SB rejects
# Booleans post-lowering; the planner rejects them pre-lowering).
function _rk_mixture_no_bool!(node, k::Int, response::Symbol)
    prefix = "RK backend"
    node isa Bool && error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "arguments must be numeric, got a Boolean")
    node isa ExprColumn &&
        foreach(a -> _rk_mixture_no_bool!(a, k, response), getargs(node))
    (node isa AbstractVector || node isa Tuple) &&
        foreach(a -> _rk_mixture_no_bool!(a, k, response), node)
    nothing
end

# A mixture location: a referenced linear predictor, a sampled scalar
# parameter, or a numeric literal. Assignments are scale-only (the
# surface location slot takes predictor | param | literal). Returns
# `(kind, value)` with `value::Union{Symbol,Float64}`.
function _rk_mixture_location_arg(arg, candidates::Vector{Symbol},
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        response::Symbol, what::String)
    prefix = "RK backend"
    arg isa NamedColumn && name(arg) in candidates &&
        return (:predictor, name(arg))
    arg isa Bool && error(
        "$prefix: response `$response` $what must be numeric, got a Boolean")
    arg isa Number && return (:literal, Float64(arg))
    arg isa NamedColumn && parent(arg) isa DataColumn && error(
        "$prefix: response `$response` $what cannot be a data column; " *
        "write a linear predictor, a sampled parameter, or a numeric literal")
    if arg isa NamedColumn
        kind, value = _rk_resolve_use_ref(name(arg), consts, aliases,
            parameters, assignments, "response `$response` $what")
        kind === :param && return (:param, value)
        kind === :number && return (:literal, value)
        error("$prefix: response `$response` $what must be a linear " *
              "predictor, a sampled parameter, or a numeric literal " *
              "(assignments are scale-only)")
    end
    error("$prefix: response `$response` $what must be a linear " *
          "predictor, a sampled parameter, or a numeric literal")
end

# A mixture scalar slot (Bernoulli/Binomial probability once the
# predictor spellings are excluded): a sampled scalar parameter or a
# numeric literal. A bare predictor reaching here names a misused
# predictor (the predictor arms ran first), never a valid spelling.
function _rk_mixture_scalar_slot(arg, candidates::Vector{Symbol},
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        response::Symbol, what::String, tail::String)
    prefix = "RK backend"
    message = "$prefix: response `$response` $what must be $tail"
    arg isa Bool && error(
        "$prefix: response `$response` $what must be numeric, got a Boolean")
    arg isa Number && return (:literal, Float64(arg))
    arg isa NamedColumn && name(arg) in candidates && error(message)
    arg isa NamedColumn && parent(arg) isa DataColumn && error(message)
    arg isa NamedColumn || error(message)
    kind, value = _rk_resolve_use_ref(name(arg), consts, aliases,
        parameters, assignments, "response `$response` $what")
    kind === :param && return (:param, value)
    kind === :number && return (:literal, value)
    error(message)
end

function _rk_mixture_unit_literal(x::Number, response::Symbol, what::String)
    prefix = "RK backend"
    value = Float64(x)
    isfinite(value) && 0 <= value <= 1 || error(
        "$prefix: response `$response` $what must be a probability in [0, 1]")
    value
end

_rk_mixture_head_name(head) =
    head isa Function ? nameof(head) :
    head isa Type ? nameof(head) : string(head)

# First-referenced predictor order over the component args (deduped,
# possibly empty — all-scalar mixtures reference no predictor). Total:
# malformed shapes collect nothing and fail specifically in Phase 5.
# Weights never contribute (a predictor-named weight fails in weights
# validation, it never plans a predictor).
function _rk_mixture_refs(program, rhs::ExprColumn, response::Symbol)
    declared = Set{Symbol}(node.name for node in program.operations
        if node.role === :predictor)
    names = Symbol[]
    args = getargs(rhs)
    isempty(args) || _rk_mixture_walk_refs!(
        names, Set{Symbol}(), args[1], declared)
    names
end

function _rk_mixture_walk_refs!(names::Vector{Symbol}, seen::Set{Symbol},
        node, declared::Set{Symbol})
    if node isa NamedColumn
        name(node) in declared && name(node) ∉ seen &&
            (push!(names, name(node)); push!(seen, name(node)))
        return nothing
    elseif node isa ExprColumn
        for arg in getargs(node)
            _rk_mixture_walk_refs!(names, seen, arg, declared)
        end
        return nothing
    elseif node isa AbstractVector || node isa Tuple
        for el in node
            _rk_mixture_walk_refs!(names, seen, el, declared)
        end
        return nothing
    end
    nothing
end

function _rk_classify_mixture(rhs::ExprColumn, candidates::Vector{Symbol},
        predictor_link::Dict{Symbol,Symbol}, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol},
        vectors::Dict{Symbol,_RKVectorParameter}, response::Symbol)
    prefix = "RK backend"
    args = getargs(rhs)
    length(args) == 2 || error(
        "$prefix: response `$response` `MixtureModel` expects " *
        "`(components, weights)`, got $(length(args)) arguments")
    isempty(getkwargs(rhs)) || error(
        "$prefix: response `$response` `MixtureModel` takes no keywords")
    components, weights_arg = args
    components isa AbstractVector || error(
        "$prefix: response `$response` `MixtureModel` components must be " *
        "a vector of distribution calls, got $(typeof(components))")
    K = length(components)
    K >= 1 || error(
        "$prefix: response `$response` `MixtureModel` needs at least one " *
        "component")
    for (k, comp) in enumerate(components)
        comp isa ExprColumn || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "must be a distribution call, got $(typeof(comp))")
    end
    heads = map(getf, components)
    allequal(heads) || error(
        "$prefix: response `$response` `MixtureModel` components must " *
        "share one family (found " *
        "$(join(unique!(map(string, heads)), ", "))); heterogeneous " *
        "mixtures are not supported because Stan rejects out-of-support " *
        "values where Turing returns `-Inf`)")
    classified = map(enumerate(components)) do (k, comp)
        _rk_classify_mixture_component(comp, k, candidates, predictor_link,
            parameters, assignments, consts, aliases, response)
    end
    trials = _rk_mixture_shared_trials(
        components, first(classified).family, parameters, assignments,
        consts, aliases, response)
    weights = _rk_mixture_weights(weights_arg, K, response, vectors)
    anchor = _rk_mixture_anchor(classified, weights, response)
    (; family=:mixture, link=first(classified).link,
        components=classified, weights, trials, anchor,
        component_family=first(classified).family)
end

function _rk_classify_mixture_component(comp::ExprColumn, k::Int,
        candidates::Vector{Symbol}, predictor_link::Dict{Symbol,Symbol},
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        response::Symbol)
    prefix = "RK backend"
    head = getf(comp)
    cargs = getargs(comp)
    isempty(getkwargs(comp)) || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "(`$(_rk_mixture_head_name(head))`) takes no constructor keywords")
    _rk_mixture_no_bool!(comp, k, response)
    # A predictor location's link, checked per family (params/literals
    # ride the constrained scale — no link applies).
    link_of(kind, loc) =
        kind === :predictor ? predictor_link[loc] : nothing
    if head === Normal
        length(cargs) == 2 || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "`Normal` needs `(location, scale)`")
        lkind, loc = _rk_mixture_location_arg(cargs[1], candidates,
            parameters, assignments, consts, aliases, response,
            "component $k location")
        plink = link_of(lkind, loc)
        plink === nothing || plink === :identity || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "pairs `Normal` with a $plink-link predictor `$loc`; write " *
            "an identity-link predictor, a sampled parameter, or a literal")
        scale, scale_predictor = _rk_scale_argument(cargs[2], parameters,
            assignments, consts, aliases, response, "component $k scale",
            candidates)
        return _RKMixtureComponent(
            :gaussian, :identity, loc, lkind, scale, scale_predictor)
    elseif head === BernoulliLogit
        length(cargs) == 1 || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "`BernoulliLogit` needs one argument")
        lkind, loc = _rk_mixture_location_arg(only(cargs), candidates,
            parameters, assignments, consts, aliases, response,
            "component $k location")
        # Logit-scale positions never ride bare: a bare parameter or
        # literal under `BernoulliLogit` is logit-scale, but the
        # decomposed twin reads its argument as a probability (SB
        # computes `bernoulli_logit_lpmf` — logistic inside). Only
        # identity predictors admit (their linear value wraps).
        lkind === :predictor || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "`BernoulliLogit` needs an identity-link predictor " *
            "location (a bare parameter or literal is logit-scale, " *
            "not a probability); write `Bernoulli(p)` with a sampled " *
            "parameter or literal probability, or `BernoulliLogit(eta)` " *
            "with an identity predictor")
        plink = predictor_link[loc]
        plink === :identity || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "applies `BernoulliLogit` on top of a $plink-link predictor " *
            "`$loc` (double link); use an identity-link predictor")
        return _RKMixtureComponent(
            :bernoulli_logit, :logit, loc, lkind, nothing, nothing)
    elseif head === Bernoulli
        length(cargs) == 1 || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "`Bernoulli` needs one argument")
        arg = only(cargs)
        prob_tail = "a `logit(p)` predictor, `logistic(eta)` over an " *
            "identity-link predictor, a sampled parameter, or a " *
            "probability literal in [0, 1]"
        if _rk_is_predictor_ref(arg, candidates)
            loc = name(arg)
            plink = predictor_link[loc]
            plink === :logit || error(
                "$prefix: response `$response` `MixtureModel` component " *
                "$k pairs `Bernoulli` with a $plink-link predictor " *
                "`$loc`; mixture v1 admits $prob_tail")
            return _RKMixtureComponent(
                :bernoulli_logit, :logit, loc, :predictor, nothing, nothing)
        end
        stripped = _rk_strip_logistic(arg, candidates)
        if stripped !== nothing
            predictor_link[stripped] === :identity || error(
                "$prefix: response `$response` `MixtureModel` component " *
                "$k applies `logistic` on top of a " *
                "$(predictor_link[stripped])-link predictor `$stripped` " *
                "(double link); mixture v1 admits $prob_tail")
            return _RKMixtureComponent(:bernoulli_logit, :logit,
                stripped, :predictor, nothing, nothing)
        end
        lkind, loc = _rk_mixture_scalar_slot(arg, candidates, parameters,
            assignments, consts, aliases, response,
            "component $k probability", prob_tail)
        lkind === :literal &&
            _rk_mixture_unit_literal(loc, response, "component $k probability")
        return _RKMixtureComponent(
            :bernoulli_logit, :logit, loc, lkind, nothing, nothing)
    elseif head === Poisson
        length(cargs) == 1 || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "`Poisson` needs one argument")
        lkind, loc = _rk_mixture_location_arg(only(cargs), candidates,
            parameters, assignments, consts, aliases, response,
            "component $k rate")
        plink = link_of(lkind, loc)
        if plink !== nothing && plink !== :log
            error("$prefix: response `$response` `MixtureModel` component " *
                  "$k pairs `Poisson` with a $plink-link predictor " *
                  "`$loc`; write `Poisson(mu)` with a `log(mu)` " *
                  "predictor, a sampled parameter, or a positive literal")
        end
        lkind === :literal &&
            _rk_positive_literal(loc, response, "component $k rate")
        return _RKMixtureComponent(
            :poisson_log, :log, loc, lkind, nothing, nothing)
    elseif head === Binomial
        length(cargs) == 2 || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "`Binomial` needs `(trials, probability)`")
        arg = cargs[2]
        prob_tail = "a `logit(p)` predictor, `logistic(eta)` over an " *
            "identity-link predictor, a sampled parameter, or a " *
            "probability literal in [0, 1]"
        if _rk_is_predictor_ref(arg, candidates)
            loc = name(arg)
            plink = predictor_link[loc]
            plink === :logit || error(
                "$prefix: response `$response` `MixtureModel` component " *
                "$k pairs `Binomial` with a $plink-link predictor " *
                "`$loc`; mixture v1 admits $prob_tail")
            return _RKMixtureComponent(
                :binomial_logit, :logit, loc, :predictor, nothing, nothing)
        end
        stripped = _rk_strip_logistic(arg, candidates)
        if stripped !== nothing
            predictor_link[stripped] === :identity || error(
                "$prefix: response `$response` `MixtureModel` component " *
                "$k applies `logistic` on top of a " *
                "$(predictor_link[stripped])-link predictor `$stripped` " *
                "(double link); mixture v1 admits $prob_tail")
            return _RKMixtureComponent(:binomial_logit, :logit,
                stripped, :predictor, nothing, nothing)
        end
        lkind, loc = _rk_mixture_scalar_slot(arg, candidates, parameters,
            assignments, consts, aliases, response,
            "component $k probability", prob_tail)
        lkind === :literal &&
            _rk_mixture_unit_literal(loc, response, "component $k probability")
        return _RKMixtureComponent(
            :binomial_logit, :logit, loc, lkind, nothing, nothing)
    elseif head === BinomialLogit
        error("$prefix: response `$response` `MixtureModel` component $k " *
              "`BinomialLogit` is out of mixture v1; write `Binomial(n, " *
              "p)` with a `logit(p)` predictor, `Binomial(n, " *
              "logistic(eta))` with an identity predictor, or a sampled " *
              "parameter / literal probability")
    elseif head === NegativeBinomial2
        length(cargs) == 2 || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "`NegativeBinomial2` needs `(mean, dispersion)`; write " *
            "`NegativeBinomial2(mu, phi)` with a `log(mu)` predictor, a " *
            "sampled parameter, or a positive literal mean")
        lkind, loc = _rk_mixture_location_arg(cargs[1], candidates,
            parameters, assignments, consts, aliases, response,
            "component $k mean")
        plink = link_of(lkind, loc)
        plink === nothing || plink === :log || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "pairs `NegativeBinomial2` with a $plink-link predictor " *
            "`$loc`; write a `log(mu)` predictor, a sampled parameter, " *
            "or a positive literal mean")
        lkind === :literal &&
            _rk_positive_literal(loc, response, "component $k mean")
        dispersion, dispersion_predictor = _rk_scale_argument(cargs[2],
            parameters, assignments, consts, aliases, response,
            "component $k dispersion", candidates)
        return _RKMixtureComponent(:nb2_log, :log, loc, lkind,
            dispersion, dispersion_predictor)
    elseif head === Gamma
        shape, shape_predictor, lkind, loc =
            _rk_mixture_gamma_args(cargs, k, candidates, predictor_link,
                parameters, assignments, consts, aliases, response)
        return _RKMixtureComponent(
            :gamma_log, :log, loc, lkind, shape, shape_predictor)
    elseif head === Beta
        mu_kind, mu_value, concentration = _rk_mixture_beta_args(cargs, k,
            candidates, predictor_link, parameters, assignments, consts,
            aliases, response)
        return _RKMixtureComponent(:beta_logit, :logit, mu_value,
            mu_kind, concentration, nothing)
    end
    error("$prefix: response `$response` `MixtureModel` component $k " *
          "family `$(_rk_mixture_head_name(head))` is not admitted; " *
          "mixture v1 admits Normal, Bernoulli/BernoulliLogit, Poisson, " *
          "Binomial, NegativeBinomial2, Gamma, and Beta components")
end

# Gamma mean-shape form per mixture component: `Gamma(alpha, mu/alpha)`
# with the SAME alpha in both positions (mirrors the single-family
# double-alpha rule); the mean is a `log(mu)` predictor, a sampled
# parameter, or a positive literal.
function _rk_mixture_gamma_args(cargs, k::Int, candidates::Vector{Symbol},
        predictor_link::Dict{Symbol,Symbol}, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, response::Symbol)
    prefix = "RK backend"
    length(cargs) == 2 || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Gamma` needs `(shape, scale)`; write `Gamma(alpha, " *
        "mu/alpha)` with a `log(mu)` predictor, a sampled parameter, " *
        "or a positive literal mean")
    shape, shape_predictor = _rk_scale_argument(cargs[1], parameters,
        assignments, consts, aliases, response, "component $k shape",
        candidates)
    scale = cargs[2]
    scale isa ExprColumn && getf(scale) === (/) || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Gamma` scale must be `mu/alpha` with the same shape in both " *
        "positions")
    isempty(getkwargs(scale)) || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Gamma` scale takes no keywords")
    sargs = getargs(scale)
    length(sargs) == 2 || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Gamma` scale must be `mu/alpha` with the same shape in both " *
        "positions")
    lkind, loc = _rk_mixture_location_arg(sargs[1], candidates,
        parameters, assignments, consts, aliases, response,
        "component $k mean")
    if lkind === :predictor
        plink = predictor_link[loc]
        plink === :log || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "pairs `Gamma` with a $plink-link predictor `$loc`; write a " *
            "`log(mu)` predictor, a sampled parameter, or a positive " *
            "literal mean")
    elseif lkind === :literal
        _rk_positive_literal(loc, response, "component $k mean")
    end
    _rk_gamma_same_alpha(cargs[1], sargs[2]) || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Gamma` shape and scale divisor must be identical " *
        "(`Gamma(alpha, mu/alpha)`); distinct shapes are out of " *
        "mixture v1")
    shape, shape_predictor, lkind, loc
end

# Beta mean-concentration form per mixture component:
# `Beta(mu*kappa, (1-mu)*kappa)` with the SAME mu and the SAME kappa in
# both positions (mirrors the single-family structural-match rule); mu
# is a `logit(mu)` predictor, a sampled parameter, or a [0, 1] literal.
# The concentration stays scalar-only (decision 005dq0u, like
# single-family). Returns `(mu_kind, mu_value, concentration)`.
function _rk_mixture_beta_args(cargs, k::Int, candidates::Vector{Symbol},
        predictor_link::Dict{Symbol,Symbol}, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, response::Symbol)
    prefix = "RK backend"
    form = "`Beta(mu*kappa, (1-mu)*kappa)` with a `logit(mu)` " *
        "predictor, a sampled parameter, or a [0, 1] literal mean"
    length(cargs) == 2 || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Beta` needs `(a, b)`; write $form")
    found = _rk_mixture_beta_split(cargs, k, candidates, parameters,
        assignments, consts, aliases, response, form)
    found === nothing && error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Beta` arguments must be `mu*kappa` / `(1-mu)*kappa` with " *
        "the same mu and identical kappa; write $form")
    mu_kind, mu_value, kappa_raw = found
    concentration, concentration_predictor = _rk_scale_argument(
        kappa_raw, parameters, assignments, consts, aliases, response,
        "component $k concentration", Symbol[])
    concentration_predictor === nothing || error(
        "$prefix: response `$response` `MixtureModel` component $k " *
        "`Beta` concentration cannot be a linear predictor " *
        "(predictor-fed concentration is not admitted)")
    if mu_kind === :predictor
        plink = predictor_link[mu_value]
        plink === :logit || error(
            "$prefix: response `$response` `MixtureModel` component $k " *
            "pairs `Beta` with a $plink-link predictor `$mu_value`; " *
            "write $form (mixture v1 admits a logit mu link only)")
    elseif mu_kind === :literal
        _rk_mixture_unit_literal(mu_value, response, "component $k mean")
    end
    mu_kind, mu_value, concentration
end

# One consistent `(mu, kappa)` split of the two Beta positions (either
# multiplication order admits): each first-position factor is tried as
# mu, the second position must carry its `(1-mu)` complement with an
# identical kappa. Returns `(kind, value, kappa_raw)` or `nothing`.
function _rk_mixture_beta_split(cargs, k::Int, candidates::Vector{Symbol},
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        response::Symbol, form::String)
    factors = _rk_mixture_beta_factors(cargs[1])
    factors === nothing && return nothing
    f1, f2 = factors
    for (mu_raw, kappa_raw) in ((f1, f2), (f2, f1))
        key = _rk_mixture_mu_key(mu_raw, candidates, parameters,
            assignments, consts, aliases, response)
        key === nothing && continue
        other = _rk_mixture_beta_factors(cargs[2])
        other === nothing && continue
        g1, g2 = other
        kappa2 = if _rk_mixture_beta_complement(g1, key)
            g2
        elseif _rk_mixture_beta_complement(g2, key)
            g1
        else
            continue
        end
        _rk_gamma_same_alpha(kappa_raw, kappa2) || continue
        return (key[1], key[2], kappa_raw)
    end
    nothing
end

# A bare two-factor product, or `nothing` (the caller reports the form).
function _rk_mixture_beta_factors(position)
    position isa ExprColumn && getf(position) === (*) || return nothing
    isempty(getkwargs(position)) || return nothing
    pargs = getargs(position)
    length(pargs) == 2 || return nothing
    (pargs[1], pargs[2])
end

# A Beta mu candidate: a referenced predictor, a sampled parameter, or
# a numeric literal — `(kind, value)` or `nothing` (data columns and
# assignments never match; unknown names throw with attribution, as in
# the location slot).
function _rk_mixture_mu_key(f, candidates::Vector{Symbol},
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        response::Symbol)
    f isa NamedColumn && name(f) in candidates && return (:predictor, name(f))
    f isa Bool && return nothing
    f isa Number && return (:literal, Float64(f))
    f isa NamedColumn || return nothing
    parent(f) isa DataColumn && return nothing
    kind, value = _rk_resolve_use_ref(name(f), consts, aliases,
        parameters, assignments, "response `$response` mixture mean")
    kind === :param && return (:param, value)
    kind === :number && return (:literal, value)
    nothing
end

_rk_mixture_beta_complement(g, key) = begin
    g isa ExprColumn && getf(g) === (-) || return false
    cargs = getargs(g)
    length(cargs) == 2 || return false
    cargs[1] isa Number && cargs[1] == 1 || return false
    _rk_mixture_mu_equals(cargs[2], key)
end

function _rk_mixture_mu_equals(f, key)
    kind, value = key
    kind === :predictor && return _rk_is_predictor_ref(f, value)
    kind === :param && return f isa NamedColumn && name(f) === value
    kind === :literal && return f isa Number && !(f isa Bool) &&
        Float64(f) == value
    false
end

# Binomial-mixture trials: each component resolves independently (an
# integer data column or a non-negative integer literal), then all
# components must share one structurally identical expression (SB's
# rule: value-equal but structurally different counts could diverge on
# `reprocess` — share one column). Returns the shared trials.
function _rk_mixture_shared_trials(components, family::Symbol,
        parameters::Set{Symbol}, assignments::Set{Symbol},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        response::Symbol)
    family === :binomial_logit || return nothing
    prefix = "RK backend"
    raws = map(comp -> getargs(comp)[1], components)
    resolved = map(raws) do raw
        _rk_trials_argument(raw, response, parameters, assignments,
            consts, aliases)
    end
    for raw in Iterators.drop(raws, 1)
        _rk_mixture_same_trials(first(raws), raw) || error(
            "$prefix: response `$response` `MixtureModel` `Binomial` " *
            "components must share one identical trial-count expression; " *
            "value-equal but structurally different counts could " *
            "diverge on `reprocess` — share one column")
    end
    first(resolved)
end

function _rk_mixture_same_trials(a, b)
    a isa Number && b isa Number && return isequal(a, b)
    a isa NamedColumn && b isa NamedColumn &&
        parent(a) isa DataColumn && parent(b) isa DataColumn &&
        return name(a) == name(b)
    false
end

# Mixture weights: a numeric vector/tuple summing to 1 (frozen), or a
# Dirichlet-backed simplex parameter of matching width. Data-column
# weights are out of mixture v1 (they are not n_obs columns, so they
# cannot cross as data — the thin layer agrees).
function _rk_mixture_weights(arg, K::Int, response::Symbol,
        vectors::Dict{Symbol,_RKVectorParameter})
    prefix = "RK backend"
    if arg isa Union{AbstractVector,Tuple} &&
            all(w -> w isa Real && !(w isa Bool), arg)
        length(arg) == K || error(
            "$prefix: response `$response` `MixtureModel` has $K " *
            "components but $(length(arg)) weights")
        weights = Float64.(collect(arg))
        all(isfinite, weights) || error(
            "$prefix: response `$response` `MixtureModel` weights must " *
            "be finite")
        all(>=(0), weights) || error(
            "$prefix: response `$response` `MixtureModel` weights must " *
            "be nonnegative")
        total = sum(weights)
        isapprox(total, 1.0; atol=1e-8) || error(
            "$prefix: response `$response` `MixtureModel` weights must " *
            "sum to 1 (got $total)")
        return weights
    end
    arg isa NamedColumn && parent(arg) isa DataColumn && error(
        "$prefix: response `$response` `MixtureModel` data-column " *
        "weights are out of mixture v1; write a literal weight vector " *
        "or a Dirichlet-backed simplex parameter")
    if arg isa NamedColumn
        wname = name(arg)
        haskey(vectors, wname) && vectors[wname].family ===
                :simplex_dirichlet || error(
            "$prefix: response `$response` `MixtureModel` weights " *
            "`$wname` must be a numeric vector or a `~ Dirichlet(...)` " *
            "simplex parameter")
        size = vectors[wname].size
        size == K || error(
            "$prefix: response `$response` has $K mixture components " *
            "but simplex `$wname` has $size categories; sizes must agree")
        return wname
    end
    error("$prefix: response `$response` `MixtureModel` weights must be " *
          "a numeric vector of length $K or a Dirichlet-backed simplex " *
          "parameter, got $(typeof(arg))")
end

# The struct predictor anchor for a mixture response: the anchor is
# never resolved as a predictor (component slots ride dedicated
# fields), but the struct field is non-nullable, so predictor-less
# mixtures still anchor it. Order (mirrored thin-side, verbatim):
# first location predictor, else first scale predictor, else the
# Dirichlet weights name, else the first location/scale param (a scale
# assignment anchors only when nothing else names a free value). A
# fully-fixed mixture (zero free names) has no RK plan.
function _rk_mixture_anchor(classified::Vector{_RKMixtureComponent},
        weights::Union{Vector{Float64},Symbol}, response::Symbol)
    prefix = "RK backend"
    for comp in classified
        comp.location_kind === :predictor && return comp.location::Symbol
    end
    for comp in classified
        comp.scale_predictor !== nothing && return comp.scale_predictor
    end
    weights isa Symbol && return weights
    for comp in classified
        comp.location_kind === :param && return comp.location::Symbol
    end
    for comp in classified
        comp.scale isa Symbol && return comp.scale
    end
    error("$prefix: response `$response` `MixtureModel` is fully fixed " *
          "(every component argument and the weights are literals); a " *
          "mixture with no free parameters has no RK plan")
end

function _rk_evidence_bound(bound, data::AbstractDict, response::Symbol,
        side::String, columns::Dict{Symbol,AbstractVector},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        parameters::Set{Symbol}, assign_names::Set{Symbol})
    prefix = "RK backend"
    isnothing(bound) && return nothing
    bound isa Number && return _rk_evidence_literal(
        Float64(bound), response, side)
    bound isa NamedColumn || error(
        "RK backend: response `$response` $side bound must be a numeric " *
        "literal or a raw data column")
    parent(bound) isa DataColumn || return _rk_evidence_name_bound(
        name(bound), response, side, consts, aliases, parameters, assign_names)
    key = name(bound)
    raw = get(data, key, nothing)
    raw isa AbstractVector{<:Real} || error(
        "$prefix: response `$response` $side bound column `$key` must be a " *
        "real vector")
    columns[key] = raw
    key
end

function _rk_evidence_literal(value::Float64, response::Symbol, side::String)
    prefix = "RK backend"
    isnan(value) && error(
        "$prefix: response `$response` $side bound is NaN")
    # One-sided = omitted side: ±Inf normalizes to nothing (the thin layer
    # accepts finite literals only, and the omission is semantics-preserving).
    isinf(value) && return nothing
    value
end

function _rk_evidence_name_bound(name::Symbol, response::Symbol, side::String,
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        parameters::Set{Symbol}, assign_names::Set{Symbol})
    prefix = "RK backend"
    # `Inf`/`NaN` arrive as names (Julia globals, not literals); resolve them
    # before the use-ref walk so Inf omits and NaN fails with its own error.
    name === :Inf && return _rk_evidence_literal(Inf, response, side)
    name === :NaN && return _rk_evidence_literal(NaN, response, side)
    kind, value = _rk_resolve_use_ref(name, consts, aliases, parameters,
        assign_names, "response `$response` $side bound")
    kind === :number && return _rk_evidence_literal(value, response, side)
    error("$prefix: response `$response` $side bound must be a numeric " *
          "literal or a raw data column (parameter/assignment bounds are " *
          "out of slice 1)")
end

function _rk_plan_evidence(modifier, family::Symbol, data::AbstractDict,
        response::Symbol, columns::Dict{Symbol,AbstractVector},
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        parameters::Set{Symbol}, assign_names::Set{Symbol})
    prefix = "RK backend"
    isnothing(modifier) && return _RKResponseEvidence(:none, nothing, nothing)
    family in (:gaussian, :poisson_log) || error(
        "$prefix: response `$response` evidence ($(modifier.kind)) is out " *
        "of slice 1 (evidence is admitted on Gaussian and Poisson " *
        "responses only)")
    lower = _rk_evidence_bound(modifier.lower, data, response, "lower",
        columns, consts, aliases, parameters, assign_names)
    upper = _rk_evidence_bound(modifier.upper, data, response, "upper",
        columns, consts, aliases, parameters, assign_names)
    if modifier.kind === :interval_censored
        lower === nothing || error(
            "$prefix: response `$response` interval evidence takes no " *
            "lower bound (the response itself is the lower endpoint)")
        upper === nothing && error(
            "$prefix: response `$response` interval evidence requires an " *
            "upper bound")
    end
    _RKResponseEvidence(modifier.kind, lower, upper)
end

function _rk_bound_values(bound, columns::Dict{Symbol,AbstractVector},
        n_obs::Int)
    isnothing(bound) && return nothing
    bound isa Float64 && return fill(bound, n_obs)
    columns[bound]
end

function _rk_gate_evidence_values!(specs::AbstractVector,
        columns::Dict{Symbol,AbstractVector}, n_obs::Int)
    prefix = "RK backend"
    for spec in specs
        evidence = spec.evidence
        evidence.kind === :none && continue
        lower = _rk_bound_values(evidence.lower, columns, n_obs)
        upper = _rk_bound_values(evidence.upper, columns, n_obs)
        if evidence.kind === :interval_censored
            response = columns[spec.response]
            all(isfinite, response) || error(
                "$prefix: response `$(spec.response)` interval evidence " *
                "requires finite response values")
            all(response .< upper) || error(
                "$prefix: response `$(spec.response)` interval evidence " *
                "requires response < upper every row")
        else
            (isnothing(lower) || isnothing(upper)) && continue
            all(lower .< upper) || error(
                "$prefix: response `$(spec.response)` evidence requires " *
                "strict lower < upper every row")
        end
        # Mirrors the thin layer: the poisson.cdf(::Int) endpoint needs
        # integer-valued bounds (literals and columns alike).
        if spec.family === :poisson_log
            for (side, bound) in (("lower", lower), ("upper", upper))
                bound === nothing && continue
                all(v -> v == round(v), bound) || error(
                    "$prefix: response `$(spec.response)` Poisson evidence " *
                    "$side bound must be integer-valued")
            end
        end
    end
    nothing
end

# Factor columns cross as plain value vectors; a `CategoricalVector`
# crosses string-normalized (the thin layer's `levels(g)` is observed-only
# sort order, so declared-but-unobserved levels cannot cross). The
# normalization is load-bearing, not deletable preprocessing: the
# thin-layer `_declared_codes` encoder compares bound rows against
# stringified levels, and a non-string-leveled categorical compares
# all-false there (`categorical([1,2]) .== "1"` is `Bool[0,0]` —
# verified empirically 2026-09-20, todo 14pgdwz), which would silently
# zero dummy indicators and group codes alike. The substantive
# level/code computation already happens in-graph thin-side (bind
# derivation + the `_ppl_gidx` encoder); only this row/level type
# agreement stays Julia-side. (A `string()` surface word would move
# even this; no such word exists — a possible peer follow-up, not
# requested.)
function _rk_factor_crossed(raw::AbstractVector)
    raw isa CA.CategoricalVector ? string.(collect(raw)) : raw
end

# The thin-layer `levels(g)` position of `ref_value`, or a fail-closed
# error naming the observed levels.
function _rk_factor_ref_position(source::Symbol, crossed::AbstractVector,
        ref_value::Union{Integer,AbstractString}, target::Symbol)
    prefix = "RK backend"
    levels = _rk_grouping_levels(crossed)
    pos = findfirst(==(ref_value), levels)
    isnothing(pos) && error(
        "$prefix: predictor `$target` factor `$source` ref `$ref_value` " *
        "is not an observed level (levels: $(join(levels, ", ")))")
    pos, length(levels)
end

# ---- provisional derived lowering (interactions, zscale-family, data exprs) ----
#
# A derived column is a thin-layer dotted definition computed in-graph from
# raw columns (`int_x_z = x .* z`); only raw columns cross the boundary.
# Every derived definition is verified against the shared lowering's
# materialized values before the plan accepts it, so a mirror bug fails
# loudly here instead of silently wrong densities.

const _RK_DERIVED_BINOP_FN = Dict{Symbol,Function}(
    v => k for (k, v) in _RK_DERIVED_BINOPS)
const _RK_DERIVED_MATH_FN = Dict{Symbol,Function}(
    v => k for (k, v) in _RK_DERIVED_MATH)
const _RK_DERIVED_CMP_FN = Dict{Symbol,Function}(
    v => k for (k, v) in _RK_DERIVED_CMP)
const _RK_DERIVED_RED_FN = Dict{Symbol,Function}(
    v => k for (k, v) in _RK_DERIVED_REDNAME)

function _rk_derived_hint(value)
    value isa Symbol && return string(value)
    value isa Number && return replace(string(value), "." => "_", "-" => "m")
    value isa Expr || return "expr"
    head = value.head
    if head === :call && !isempty(value.args) && value.args[1] isa Symbol
        fn = string(value.args[1])
        fn = startswith(fn, ".") ? fn[2:end] : fn
        args = value.args[2:end]
        isempty(args) && return fn
        return fn * "_" * join(_rk_derived_hint.(args), "_")
    elseif head === :.
        length(value.args) == 2 && value.args[1] isa Symbol || return "dotted"
        tup = value.args[2]
        tup isa Expr && tup.head === :tuple || return string(value.args[1])
        return string(value.args[1]) * "_" *
            join(_rk_derived_hint.(tup.args), "_")
    end
    return "expr"
end

function _rk_mint_derived!(derived::Vector{_RKDerivedSpec},
        taken::Set{Symbol}, columns::Dict{Symbol,AbstractVector}, hint::String)
    base = "rkd_" * join(filter(!isempty, split(
        replace(lowercase(hint), r"[^a-z0-9]+" => "_"), "_")), "_")
    base == "rkd_" && (base = "rkd_expr")
    length(base) > 40 && (base = base[1:40])
    name = Symbol(base)
    counter = 1
    while name in taken || haskey(columns, name) ||
            any(d -> d.name === name, derived)
        counter += 1
        name = Symbol(base * "_" * string(counter))
    end
    push!(taken, name)
    name
end

# Push a derived definition, deduping by (name, expression). Same name with
# a different expression is an internal error: shared labels are a function
# of term structure, so a collision means the mirror drifted.
function _rk_push_derived!(derived::Vector{_RKDerivedSpec}, name::Symbol,
        expression::Expr, label::Symbol, target::Symbol)
    prefix = "RK backend"
    for existing in derived
        existing.name === name || continue
        existing.expression == expression && return name
        error("$prefix: internal: derived column `$name` in `$target` has " *
              "conflicting definitions")
    end
    push!(derived, _RKDerivedSpec(name, expression, label))
    name
end

# Cross every raw data column a dotted definition touches (bind needs them).
function _rk_cross_derived_refs!(expression, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector})
    if expression isa Symbol
        if haskey(data, expression) && !haskey(columns, expression)
            raw = data[expression]
            columns[expression] =
                raw isa CA.CategoricalVector ? collect(raw) : raw
        end
        return nothing
    end
    expression isa Expr || return nothing
    for arg in expression.args
        _rk_cross_derived_refs!(arg, data, columns)
    end
    nothing
end

# Evaluate a dotted definition BRM-side for verification against shared
# values. Resolves staged names through the derived registry.
function _rk_eval_dotted(value, data::AbstractDict,
        derived::Vector{_RKDerivedSpec}, memo::Dict{Symbol,Any})
    value isa Number && return value
    if value isa Symbol
        haskey(data, value) && return data[value]
        haskey(memo, value) && return memo[value]
        for spec in derived
            spec.name === value || continue
            result = _rk_eval_dotted(
                spec.expression, data, derived, memo)
            memo[value] = result
            return result
        end
        error("RK backend: internal: derived verification references " *
              "unknown name `$value`")
    end
    value isa Expr || error("RK backend: internal: cannot evaluate " *
                            "derived value `$(repr(value))`")
    if value.head === :call && !isempty(value.args)
        fn = value.args[1]
        fn isa Symbol || error("RK backend: internal: cannot evaluate " *
                               "derived call `$(repr(value))`")
        args = map(a -> _rk_eval_dotted(a, data, derived, memo),
            value.args[2:end])
        haskey(_RK_DERIVED_BINOP_FN, fn) &&
            return broadcast(_RK_DERIVED_BINOP_FN[fn], args...)
        haskey(_RK_DERIVED_CMP_FN, fn) &&
            return broadcast(_RK_DERIVED_CMP_FN[fn], args...)
        haskey(_RK_DERIVED_RED_FN, fn) && length(args) == 1 &&
            return _RK_DERIVED_RED_FN[fn](args[1])
        error("RK backend: internal: cannot evaluate derived call `$fn`")
    end
    if value.head === :. && length(value.args) == 2 &&
            value.args[1] isa Symbol
        fname = value.args[1]
        tup = value.args[2]
        haskey(_RK_DERIVED_MATH_FN, fname) && tup isa Expr &&
            tup.head === :tuple || error(
                "RK backend: internal: cannot evaluate derived call `$fname.`")
        args = map(a -> _rk_eval_dotted(a, data, derived, memo), tup.args)
        return broadcast(_RK_DERIVED_MATH_FN[fname], args...)
    end
    error("RK backend: internal: cannot evaluate derived " *
          "expression `$(repr(value))`")
end

function _rk_verify_derived_values!(expression, expected::AbstractVector,
        name::Symbol, target::Symbol, data::AbstractDict,
        derived::Vector{_RKDerivedSpec}; origin="shared lowering")
    prefix = "RK backend"
    got = _rk_eval_dotted(expression, data, derived, Dict{Symbol,Any}())
    got isa AbstractVector && length(got) == length(expected) &&
        all(isapprox.(Float64.(got), Float64.(expected);
            rtol=1e-9, atol=1e-12)) && return nothing
    error("$prefix: internal: derived column `$name` in `$target` " *
          "disagrees with $origin values")
end

# Lower a BRM scalar data-expression node to thin-layer dotted AST. Returns
# (value, is_vector): bare names and dotted forms are vector-valued,
# literals and reductions are scalar. Nested non-name reduction arguments
# are staged as their own derived definitions automatically.
function _rk_lower_data_expr(node, target::Symbol, origin::String,
        data::AbstractDict, columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol})
    prefix = "RK backend"
    node isa Number && return node, false
    if node isa NamedColumn
        source = name(node)
        parent(node) isa DataColumn && haskey(data, source) || error(
            "$prefix: predictor `$target` $origin references `$source`, " *
            "which is not a raw data column; slice 1 data expressions " *
            "take raw data columns only")
        raw = data[source]
        raw isa AbstractVector && eltype(raw) <: Real &&
            !(eltype(raw) <: Bool) &&
            !(raw isa CA.CategoricalVector) || error(
            "$prefix: predictor `$target` $origin column `$source` must " *
            "be a plain numeric vector for arithmetic; categorical " *
            "columns enter data expressions through `factor()` " *
            "comparisons in `&` interactions")
        return source, true
    end
    node isa ExprColumn || error(
        "$prefix: predictor `$target` $origin is not supported in slice 1")
    f = getf(node)
    isempty(getkwargs(node)) || error(
        "$prefix: predictor `$target` $origin call keywords are out of " *
        "slice 1")
    args = getargs(node)
    if _brm_is_term_head(f) || f === (&) || f === (|) || f === factor ||
            f === offset || f === (~) || f === zscale || f === center ||
            f === standardize
        head = f isa Function ? nameof(f) : string(f)
        error("$prefix: predictor `$target` $origin nests `$head`, which " *
              "is not admittable inside a data expression in slice 1")
    end
    if f isa Function && f in _RK_ASSIGNMENT_REDUCTIONS
        length(args) == 1 || error(
            "$prefix: predictor `$target` $origin reduction " *
            "`$(nameof(f))` takes exactly one argument")
        lowered, _ = _rk_lower_data_expr(only(args), target, origin,
            data, columns, derived, taken)
        lowered isa Symbol &&
            return Expr(:call, _RK_DERIVED_REDNAME[f], lowered), false
        staged = _rk_mint_derived!(derived, taken, columns,
            _rk_derived_hint(lowered))
        _rk_push_derived!(derived, staged, lowered, staged, target)
        _rk_cross_derived_refs!(lowered, data, columns)
        return Expr(:call, _RK_DERIVED_REDNAME[f], staged), false
    end
    if f isa Function && haskey(_RK_DERIVED_MATH, f)
        length(args) == 1 || error(
            "$prefix: predictor `$target` $origin `$(nameof(f))` takes " *
            "exactly one argument")
        lowered, _ = _rk_lower_data_expr(only(args), target, origin,
            data, columns, derived, taken)
        return Expr(:., _RK_DERIVED_MATH[f], Expr(:tuple, lowered)), true
    end
    if f isa Function && haskey(_RK_DERIVED_BINOPS, f)
        if length(args) == 1
            f === (-) || error(
                "$prefix: predictor `$target` $origin unary " *
                "`$(nameof(f))` is out of slice 1 (write `-1 * x`)")
            lowered, isvec = _rk_lower_data_expr(only(args), target,
                origin, data, columns, derived, taken)
            isvec || error(
                "$prefix: predictor `$target` $origin unary minus needs " *
                "a vector argument")
            return Expr(:call, :.*, -1, lowered), true
        end
        length(args) == 2 || error(
            "$prefix: predictor `$target` $origin `$(nameof(f))` takes " *
            "exactly two arguments")
        left, _ = _rk_lower_data_expr(args[1], target, origin,
            data, columns, derived, taken)
        right, _ = _rk_lower_data_expr(args[2], target, origin,
            data, columns, derived, taken)
        return Expr(:call, _RK_DERIVED_BINOPS[f], left, right), true
    end
    if f isa Function && haskey(_RK_DERIVED_CMP, f)
        length(args) == 2 || error(
            "$prefix: predictor `$target` $origin `$(nameof(f))` takes " *
            "exactly two arguments")
        left, _ = _rk_lower_data_expr(args[1], target, origin,
            data, columns, derived, taken)
        right, _ = _rk_lower_data_expr(args[2], target, origin,
            data, columns, derived, taken)
        return Expr(:call, _RK_DERIVED_CMP[f], left, right), true
    end
    head = f isa Function ? nameof(f) : string(f)
    error("$prefix: predictor `$target` $origin calls `$head`, which is " *
          "out of slice 1 (admitted: +, -, *, /, ^, %, comparisons, " *
          "log, log10, log1p, exp, expm1, sqrt, abs, sum, mean, std, " *
          "var, minimum, maximum, length)")
end

# Comparison atoms for one categorical operand: `(group .== value)` per
# level with its 0/1 values and `<source>_lvl_<k>` label (`k` the level's
# position). Full-rank over every level: reference dropping left with the
# treatment vocabulary. An unobserved declared level compares against its
# own level value (an all-zero column, not a crash).
function _rk_interaction_dummies(source::Symbol, values::AbstractVector,
        levels::AbstractVector)
    lookup = Dict(level => i for (i, level) in enumerate(levels))
    codes = Int[lookup[value] for value in values]
    map(enumerate(levels)) do (i, lvl)
        atom = Expr(:call, :.==, source, lvl)
        atom, Float64.(codes .== i), Symbol(source, :_lvl_, i), true
    end
end

# Lower one `&` operand to a list of (atom, values, label, categorical):
# bare continuous columns lower to their name, categorical operands to
# per-level comparisons, nested forms to staged derived names (defs emitted
# as a side effect). No terms or priors: operands feed the cross product.
function _rk_interaction_side_atoms(side, target::Symbol, origin::String,
        data::AbstractDict, columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol})
    prefix = "RK backend"
    if side isa NamedColumn
        source = name(side)
        parent(side) isa DataColumn && haskey(data, source) || error(
            "$prefix: predictor `$target` $origin operand `$source` is " *
            "not a raw data column")
        raw = data[source]
        raw isa AbstractVector || error(
            "$prefix: predictor `$target` $origin operand `$source` is " *
            "not a vector")
        if _brm_is_categorical_data(raw)
            _brm_is_string_categorical_data(raw) && error(
                "$prefix: predictor `$target` $origin over string " *
                "grouping column `$source` is out of slice 1 (mixed " *
                "interactions need in-graph level codes)")
            values = collect(raw)
            levels = raw isa CA.CategoricalVector ?
                collect(CA.levels(raw)) : sort!(unique(values))
            atoms = _rk_interaction_dummies(source, values, levels)
            return atoms, (Any[], length(levels))
        end
        raw isa AbstractVector{<:Real} && !(eltype(raw) <: Integer) ||
            error("$prefix: predictor `$target` $origin operand " *
                  "`$source` is neither continuous nor categorical")
        return Any[(source, raw, source, false)],
            (Any[(:col, source)], 0)
    end
    side isa ExprColumn || error(
        "$prefix: predictor `$target` $origin operand is not supported " *
        "in slice 1")
    f = getf(side)
    f === factor && error(
        "$prefix: predictor `$target` $origin `factor()` is not " *
        "admitted inside `&` operands (interaction coding is always " *
        "full-rank there); use the bare grouping column")
    if f === (&)
        nested, isp = _rk_interaction_columns(side, target, origin, data,
            columns, derived, taken)
        atoms = Any[(name, values, label, false)
                    for (name, values, label) in nested]
        return atoms, isp
    end
    if f === zscale || f === center || f === standardize
        sargs = getargs(side)
        length(sargs) == 1 || error(
            "$prefix: predictor `$target` $origin `$(nameof(f))` needs " *
            "exactly one argument")
        vname = _rk_staged_transform(f, only(sargs), target, origin,
            data, columns, derived, taken)
        atoms = Any[(vname, _rk_eval_dotted(
            vname, data, derived, Dict{Symbol,Any}()), vname, false)]
        return atoms, (Any[(:expr, _rk_gate_derived_expr(
            vname, derived, target))], 0)
    end
    lowered, isvec = _rk_lower_data_expr(
        side, target, origin, data, columns, derived, taken)
    isvec || error(
        "$prefix: predictor `$target` $origin operand is scalar; " *
        "slice 1 interactions take vector operands")
    if lowered isa Symbol
        haskey(data, lowered) || error(
            "$prefix: internal: staged interaction operand `$lowered` " *
            "is not bound")
        return Any[(lowered, data[lowered], lowered, false)],
            (Any[(:col, lowered)], 0)
    end
    lowered isa Expr || error(
        "$prefix: internal: interaction operand lowered to " *
        "`$(repr(lowered))`")
    staged = _rk_mint_derived!(
        derived, taken, columns, _rk_derived_hint(lowered))
    _rk_push_derived!(derived, staged, lowered, staged, target)
    _rk_cross_derived_refs!(lowered, data, columns)
    Any[(staged, _rk_eval_dotted(
        lowered, data, derived, Dict{Symbol,Any}()), staged, false)],
        (Any[(:expr, lowered)], 0)
end

# Shared `&` column builder used by top-level interaction terms and nested
# `&` operands alike. Returns (name, values, label) per crossed pair.
# Shared `_brm_population_columns` stays treatment-coded, so the full-rank
# cross decouples from it: labels mirror the shared
# `int_<left>_x_<right>` scheme (continuous operand first) and every pair
# verifies against its own sides' values.
function _rk_interaction_columns(term, target::Symbol, origin::String,
        data::AbstractDict, columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol})
    prefix = "RK backend"
    args = getargs(term)
    length(args) == 2 || error(
        "$prefix: predictor `$target` $origin `&` takes exactly two operands")
    left, lspine = _rk_interaction_side_atoms(args[1], target, origin,
        data, columns, derived, taken)
    right, rspine = _rk_interaction_side_atoms(args[2], target, origin,
        data, columns, derived, taken)
    spine = (vcat(lspine[1], rspine[1]), lspine[2] + rspine[2])
    specs = map([(l, r) for l in left for r in right]) do (
            (latom, lvalues, llabel, lcat),
            (ratom, rvalues, rlabel, rcat))
        defexpr = Expr(:call, :.*, latom, ratom)
        label = lcat && !rcat ? Symbol(:int_, rlabel, :_x_, llabel) :
            Symbol(:int_, llabel, :_x_, rlabel)
        _rk_verify_derived_values!(defexpr, lvalues .* rvalues, label,
            target, data, derived; origin="interaction side values")
        name = _rk_push_derived!(
            derived, label, defexpr, label, target)
        _rk_cross_derived_refs!(defexpr, data, columns)
        got = _rk_eval_dotted(
            defexpr, data, derived, Dict{Symbol,Any}())
        name, got, label
    end
    specs, spine
end

# A `&` term whose every leaf operand is categorical: its full-rank dummy
# cross partitions the rows, so it structurally spans the intercept.
function _rk_cross_leaf_categorical(side, data::AbstractDict)
    if side isa NamedColumn
        raw = get(data, name(side), nothing)
        return raw isa AbstractVector && _brm_is_categorical_data(raw)
    end
    side isa ExprColumn && getf(side) === (&) || return false
    args = getargs(side)
    length(args) == 2 || return false
    return all(a -> _rk_cross_leaf_categorical(a, data), args)
end

# Lower a `center`/`zscale`/`standardize` inner form to the bare name of
# its vector value (raw column or staged derived definition). Nested
# specials fail closed here, before shared materialization runs.
function _rk_transform_inner_name(f::Function, inner, target::Symbol,
        origin::String, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol})
    prefix = "RK backend"
    head = nameof(f)
    if inner isa NamedColumn
        source = name(inner)
        parent(inner) isa DataColumn && haskey(data, source) || error(
            "$prefix: predictor `$target` $origin `$head()` needs a raw " *
            "data column or data expression")
        raw = data[source]
        raw isa AbstractVector{<:Real} || error(
            "$prefix: predictor `$target` $origin `$head()` needs a " *
            "numeric vector")
        return source
    end
    inner isa ExprColumn || error(
        "$prefix: predictor `$target` $origin `$head()` needs a raw " *
        "data column or data expression")
    lowered, isvec = _rk_lower_data_expr(inner, target, origin,
        data, columns, derived, taken)
    isvec || error(
        "$prefix: predictor `$target` $origin `$head()` needs a " *
        "vector-valued inner form")
    lowered isa Symbol && return lowered
    lowered isa Expr || error(
        "$prefix: internal: `$head()` inner form lowered to " *
        "`$(repr(lowered))`")
    staged = _rk_mint_derived!(
        derived, taken, columns, _rk_derived_hint(lowered))
    _rk_push_derived!(derived, staged, lowered, staged, target)
    _rk_cross_derived_refs!(lowered, data, columns)
    staged
end

function _rk_transform_defexpr(f::Function, vname::Symbol)
    centered = Expr(:call, :.-, vname, Expr(:call, :mean, vname))
    f === center ? centered :
        Expr(:call, :./, centered, Expr(:call, :std, vname))
end

# Stage a `center`/`zscale`/`standardize` value as a derived definition and
# return its name (nested uses, e.g. `&` operands, mint their names).
function _rk_staged_transform(f::Function, inner, target::Symbol,
        origin::String, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol})
    vname = _rk_transform_inner_name(f, inner, target, origin,
        data, columns, derived, taken)
    defexpr = _rk_transform_defexpr(f, vname)
    staged = _rk_mint_derived!(derived, taken, columns,
        string(nameof(f)) * "_" * _rk_derived_hint(vname))
    _rk_push_derived!(derived, staged, defexpr, staged, target)
    _rk_cross_derived_refs!(defexpr, data, columns)
    staged
end

function _rk_term_specs(term, target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol},
        has_intercept::Bool, spines::Dict{Symbol,Any})
    prefix = "RK backend"
    term isa Integer && term == 1 && return _RKTermSpec[_RKTermSpec(
        :intercept, Symbol[], (;), :Intercept, :Intercept)]
    if term isa ExprColumn && getf(term) === offset
        args = getargs(term)
        length(args) == 1 || error(
            "$prefix: predictor `$target` `offset()` needs exactly one argument")
        isempty(getkwargs(term)) || error(
            "$prefix: predictor `$target` `offset()` takes no keywords")
        inner = only(args)
        if inner isa NamedColumn
            sources = _brm_data_expression_sources(inner)
            length(sources) == 1 || error(
                "$prefix: predictor `$target` slice 1 supports `offset()` " *
                "of a single raw data column or data expression")
            source = only(sources)
            raw = get(data, source, nothing)
            raw isa AbstractVector{<:Real} || error(
                "$prefix: predictor `$target` offset column `$source` must " *
                "be a real vector")
            columns[source] = raw
            return _RKTermSpec[_RKTermSpec(:offset, [source], (;), source,
                Symbol(:offset_, source))]
        end
        inner isa ExprColumn || error(
            "$prefix: predictor `$target` slice 1 supports `offset()` of " *
            "a single raw data column or data expression")
        lowered, isvec = _rk_lower_data_expr(inner, target,
            "`offset()` inner form", data, columns, derived, taken)
        isvec || error(
            "$prefix: predictor `$target` `offset()` inner form is " *
            "scalar; slice 1 offsets take vector forms")
        lowered isa Expr || error(
            "$prefix: internal: `offset()` inner form lowered to " *
            "`$(repr(lowered))`")
        fixed = _brm_population_fixed_term(term)
        fixedvalues = fixed.values
        staged = _rk_mint_derived!(
            derived, taken, columns, "offset_" * _rk_derived_hint(lowered))
        _rk_verify_derived_values!(lowered, fixedvalues, staged,
            target, data, derived)
        _rk_push_derived!(derived, staged, lowered, staged, target)
        _rk_cross_derived_refs!(lowered, data, columns)
        return _RKTermSpec[_RKTermSpec(:offset, [staged], (;), staged,
            Symbol(:offset_, staged))]
    end
    if term isa ExprColumn && getf(term) === factor
        args = getargs(term)
        length(args) == 1 || error(
            "$prefix: predictor `$target` `factor()` needs exactly one argument")
        inner = only(args)
        inner isa NamedColumn && parent(inner) isa DataColumn || error(
            "$prefix: predictor `$target` `factor()` needs a raw data column")
        kwargs = getkwargs(term)
        all(k -> k === :ref || k === :cmc, keys(kwargs)) || error(
            "$prefix: predictor `$target` `factor()` takes only `ref`/`cmc`")
        source = name(inner)
        raw = get(data, source, nothing)
        raw isa AbstractVector && _brm_is_categorical_data(raw) || error(
            "$prefix: predictor `$target` factor column `$source` must be " *
            "categorical (integer codes, strings, or a CategoricalVector)")
        cmc = get(kwargs, :cmc, true)
        cmc isa Bool || error(
            "$prefix: predictor `$target` `factor($source; cmc=...)` " *
            "expects `true` or `false`, got `$(repr(cmc))`")
        crossed = _rk_factor_crossed(raw)
        if has_intercept || !cmc
            # A subset of the observed levels: under an intercept this is
            # identified reference coding; with `cmc=false` and no
            # intercept it pins the reference level at zero (unmapped
            # rows contribute 0). `cmc` only switches intercept-free
            # coding, so it is inert under an intercept.
            ref_value = get(
                kwargs, :ref, first(_rk_grouping_levels(crossed)))
            ref_value isa Integer || ref_value isa AbstractString || error(
                "$prefix: predictor `$target` " *
                "`factor($source; ref=...)` ref must be an integer or " *
                "string level value")
            pos, K = _rk_factor_ref_position(
                source, crossed, ref_value, target)
            K == 1 && error(
                "$prefix: predictor `$target` factor `$source` has a " *
                "single observed level, so a reference subset is empty; " *
                "drop the term" * (has_intercept ? " or the intercept" :
                    " or use the bare column for its one cell mean"))
            options = (coding=:subset, drop=pos, levels=:observed)
        else
            haskey(kwargs, :ref) && error(
                "$prefix: predictor `$target` `factor($source; ref=...)` " *
                "under `0 +` is full-rank over every observed level, so " *
                "an explicit `ref` is meaningless (drop it, or set " *
                "`cmc=false` to pin the level at zero)")
            options = (coding=:fullrank, levels=:observed)
        end
        columns[source] = crossed
        return _RKTermSpec[_RKTermSpec(
            :factor, [source], options, source, source)]
    end
    if term isa NamedColumn
        backing = parent(term)
        backing isa DataColumn || error(
            "$prefix: predictor `$target` term `$(name(term))` is not a " *
            "data column; slice 1 admits raw data columns only")
        source = name(term)
        raw = get(data, source, nothing)
        raw isa AbstractVector || error(
            "$prefix: predictor `$target` column `$source` is not a vector")
        if _brm_is_categorical_data(raw)
            has_intercept && error(
                "$prefix: predictor `$target` bare factor `$source` " *
                "under an intercept is unidentified (full-rank covers " *
                "every row); name an explicit reference " *
                "(`factor($source; ref=...)`) or drop the intercept " *
                "(`0 + ...`)")
            crossed = _rk_factor_crossed(raw)
            columns[source] = crossed
            return _RKTermSpec[_RKTermSpec(
                :factor, [source],
                (coding=:fullrank, levels=:observed), source, source)]
        end
        raw isa AbstractVector{<:Real} && !(eltype(raw) <: Integer) || error(
            "$prefix: predictor `$target` column `$source` is neither a " *
            "continuous (real non-integer) nor a categorical column")
        columns[source] = raw
        return _RKTermSpec[_RKTermSpec(
            :continuous, [source], (;), source, source)]
    end
    if term isa ExprColumn && getf(term) === (&)
        specs, spine = _rk_interaction_columns(term, target,
            "`&` interaction", data, columns, derived, taken)
        for (dname, _, _) in specs
            spines[dname] = spine
        end
        return _RKTermSpec[_RKTermSpec(
            :continuous, [dname], (;), dlabel, dlabel)
            for (dname, _, dlabel) in specs]
    end
    if term isa ExprColumn &&
            (getf(term) === zscale || getf(term) === center ||
             getf(term) === standardize)
        f = getf(term)
        args = getargs(term)
        length(args) == 1 || error(
            "$prefix: predictor `$target` `$(nameof(f))()` needs exactly " *
            "one argument")
        isempty(getkwargs(term)) || error(
            "$prefix: predictor `$target` `$(nameof(f))()` takes no keywords")
        # Validate the inner form before consulting shared: nested
        # specials fail closed here, where shared would crash undecorated.
        vname = _rk_transform_inner_name(f, only(args), target,
            "`$(nameof(f))()` term", data, columns, derived, taken)
        shared = _brm_population_columns(term; cellmeans=false)
        (!isnothing(shared) && length(shared) == 1) || error(
            "$prefix: predictor `$target` `$(nameof(f))()` cannot be " *
            "coded by shared lowering")
        scol = only(shared)
        defexpr = _rk_transform_defexpr(f, vname)
        _rk_verify_derived_values!(defexpr, scol.values, scol.label,
            target, data, derived)
        dname = _rk_push_derived!(
            derived, scol.label, defexpr, scol.label, target)
        _rk_cross_derived_refs!(defexpr, data, columns)
        return _RKTermSpec[_RKTermSpec(
            :continuous, [dname], (;), scol.label, scol.label)]
    end
    if term isa ExprColumn
        _brm_is_term_head(getf(term)) && error(
            "$prefix: predictor `$target` term `$(nameof(getf(term)))` " *
            "is out of slice 1")
        # Lower before consulting shared: nested specials fail closed
        # here, where shared materialization would crash undecorated.
        lowered, isvec = _rk_lower_data_expr(term, target,
            "term `$term`", data, columns, derived, taken)
        isvec || error(
            "$prefix: predictor `$target` term `$term` is scalar; " *
            "slice 1 predictors take vector terms")
        lowered isa Expr || error(
            "$prefix: internal: term `$term` lowered to " *
            "`$(repr(lowered))`")
        shared = _brm_population_columns(term; cellmeans=false)
        (!isnothing(shared) && length(shared) == 1) || error(
            "$prefix: predictor `$target` term `$term` cannot be coded " *
            "by shared lowering")
        scol = only(shared)
        _rk_verify_derived_values!(lowered, scol.values, scol.label,
            target, data, derived)
        dname = _rk_push_derived!(
            derived, scol.label, lowered, scol.label, target)
        _rk_cross_derived_refs!(lowered, data, columns)
        return _RKTermSpec[_RKTermSpec(
            :continuous, [dname], (;), scol.label, scol.label)]
    end
    error("$prefix: predictor `$target` term `$term` is not supported in " *
          "slice 1 (admitted: `1`, continuous columns, integer/string/" *
          "categorical columns, `factor()`, `offset()`, `&` interactions, " *
          "`center`/`zscale`/`standardize`, pure numeric data expressions)")
end

# Design columns grouped by prior addressee (derived columns by label —
# each is its own coefficient; factor dummies by source — one prior
# per block), in first-seen order. Shared by the PopulationPrior path
# and the R2D2 override composition (same addressees both ways).
function _rk_design_addressee_groups(design, target::Symbol;
        prefix="RK backend")
    groups = Dict{Symbol,Vector{Int}}()
    order = Symbol[]
    for (i, column) in enumerate(design.columns)
        kind = isnothing(column.preprocess) ? nothing :
            column.preprocess.kind
        addressee = if kind in (:interaction, :zscale, :standardize,
                :center, :protect)
            column.label
        elseif kind === :population_factor_dummy || isnothing(kind)
            isnothing(column.source) ? column.label : column.source
        else
            error("$prefix: internal: design column `$(column.label)` in " *
                  "`$target` has unknown preprocess kind `$kind`")
        end
        if isnothing(column.source) && addressee !== :Intercept
            error("$prefix: internal: sourceless non-intercept column " *
                  "`$(column.label)` in `$target`")
        end
        haskey(groups, addressee) || push!(order, addressee)
        push!(get!(groups, addressee, Int[]), i)
    end
    groups, order
end

function _rk_population_priors(brmi::BRMI, design, target::Symbol,
        available::Tuple, factor_addressees::Set{Symbol},
        terms::Vector{_RKTermSpec}, derived::Vector{_RKDerivedSpec},
        r2d2::Union{Nothing,_BRMR2D2Plan}, hs_addressees::Set{Symbol})
    prefix = "RK backend"
    overrides = _brm_simple_population_effect_overrides(
        brmi, design; prefix, available_predictors=available)
    # The shared seam resolves (location, scale) without checking the family;
    # slice 1 admits Normal or Horseshoe population effects (Horseshoe cells
    # are owned by `_rk_horseshoe_priors`, which runs before this function).
    claimed = isnothing(overrides) ? () : overrides
    for expression in claimed
        isnothing(expression) && continue
        family = expression isa ExprColumn ? getf(expression) : nothing
        family === Horseshoe && continue
        family === Normal || error(
            "$prefix: predictor `$target` population-effect priors must " *
            "be `Normal(location, scale)` or `Horseshoe(...)` in slice 1")
        isempty(getkwargs(expression)) || error(
            "$prefix: predictor `$target` population-effect `Normal` " *
            "prior cannot have keywords in slice 1")
    end
    # An R2D2 predictor carries its prior mass in the R2D2Prior
    # (explicit Normals ride the overrides map) — no PopulationPrior
    # rows. The family validation above still applies.
    isnothing(r2d2) || return _RKPopulationPrior[]
    n = length(design.columns)
    stated = isnothing(overrides) ? fill(false, n) :
        Bool[!isnothing(cell) for cell in overrides]
    # Horseshoe cells carry no Normal (location, scale) — the shared
    # materializer would misread `Horseshoe()`'s empty args as (0, 1) —
    # so neutralize them before materializing (their rows are skipped
    # below; the values are never read).
    mat_overrides = isnothing(overrides) ? nothing :
        Any[cell isa ExprColumn && getf(cell) === Horseshoe ? nothing :
            cell for cell in overrides]
    location, scale = _brm_materialize_normal_effect_priors(mat_overrides, n;
        prefix)
    groups, order = _rk_design_addressee_groups(design, target; prefix)
    priors = _RKPopulationPrior[]
    for addressee in order
        # A Horseshoe addressee carries its prior mass in the
        # HorseshoePrior — no PopulationPrior row (R2D2 precedent).
        addressee in hs_addressees && continue
        idxs = groups[addressee]
        if addressee in factor_addressees
            all(stated[idxs]) || error(
                "$prefix: predictor `$target` factor `$addressee` " *
                "needs one explicit Normal prior on the whole block " *
                "(e.g. `effect($target, $addressee) ~ Normal(0, 2)`); " *
                "slice 1 has no default factor prior (the stated " *
                "prior sizes the thin-layer block)")
        end
        first_loc, first_scale = location[first(idxs)], scale[first(idxs)]
        all(i -> location[i] == first_loc && scale[i] == first_scale,
            idxs) || error(
            "$prefix: predictor `$target` addressee `$addressee` has " *
            "disagreeing population priors across its columns; slice 1 " *
            "needs one shared Normal per addressee (address the source " *
            "column, not individual levels)")
        push!(priors, _RKPopulationPrior(
            target, addressee, first_loc, first_scale))
    end
    known = Set(order)
    for term in terms
        term.kind === :continuous || continue
        term.addressee in known && continue
        any(d -> d.name === term.addressee, derived) || error(
            "$prefix: internal: addressee `$(term.addressee)` in " *
            "`$target` has no shared design column")
        # Full-rank-only interaction dummies (e.g. the reference level's)
        # have no shared column — shared stays treatment-coded — so they
        # take the emitter default. Explicit claims on these labels fail
        # in the shared seam (unaddressable there); a future slice could
        # bridge them, since the thin layer takes per-addressee priors
        # for every block.
        push!(known, term.addressee)
        push!(priors, _RKPopulationPrior(target, term.addressee, 0.0, 1.0))
    end
    for term in terms
        term.kind === :monotonic || continue
        # A design column already claimed this addressee: the raw data
        # holds a `<c>_idx` column the monotonic codes clobbered (mirrored
        # SB overwrite). Sharing one prior across the continuous column
        # and the mo beta would silently misprice one of them — fail loud.
        term.addressee in known && error(
            "$prefix: predictor `$target` monotonic index " *
            "`$(term.addressee)` collides with a population column; " *
            "rename the raw `<c>_idx` column")
        push!(known, term.addressee)
        push!(priors, _rk_mo_beta_prior(brmi, target, term.addressee))
    end
    # Latent `ar` columns are not design columns, so their betas resolve
    # through the dedicated cell (default Normal(0, 1), `:`-wide claims
    # via the shared claim engine); one prior per addressee.
    seen_ar = Set{Symbol}()
    for term in terms
        term.kind === :ar || continue
        term.addressee in seen_ar && continue
        push!(seen_ar, term.addressee)
        location, scale = _rk_ar_beta_prior(brmi, target, term.addressee)
        push!(priors, _RKPopulationPrior(target, term.addressee, location, scale))
    end
    # Latent `me` columns are not design columns, so their betas resolve
    # through the dedicated cell (default Normal(0, 1), `:`-wide claims
    # via the shared claim engine); one prior per addressee.
    seen_me = Set{Symbol}()
    for term in terms
        term.kind === :me || continue
        term.addressee in seen_me && continue
        push!(seen_me, term.addressee)
        push!(priors, _rk_me_beta_prior(brmi, target, term.addressee))
    end
    priors
end

# Structural identifiability over full-cover groups: a bare (full-rank)
# factor and a factor-only `&` cross each structurally span the
# intercept, so an intercept admits neither, and an intercept-free
# predictor admits at most one of them. Subsets never span.
function _rk_gate_cover_identified!(terms::Vector{_RKTermSpec},
        ordinary::Tuple, target::Symbol, data::AbstractDict,
        has_intercept::Bool)
    prefix = "RK backend"
    fullrank = Symbol[only(t.columns) for t in terms
        if t.kind === :factor && t.options.coding === :fullrank]
    purecross = [term for term in ordinary
        if term isa ExprColumn && getf(term) === (&) &&
            _rk_cross_leaf_categorical(term, data)]
    isempty(fullrank) && isempty(purecross) && return nothing
    who = join([["`$s`" for s in fullrank];
        ["`$t`" for t in purecross]], ", ")
    if has_intercept
        fixes = String["drop the intercept (`0 + ...`)"]
        isempty(fullrank) || pushfirst!(fixes,
            "name an explicit reference (`factor(g; ref=...)`)")
        error("$prefix: predictor `$target` combines an intercept with " *
              "full-cover group(s) $who — unidentified (each covers " *
              "every row); " * join(fixes, " or "))
    end
    length(fullrank) + length(purecross) >= 2 || return nothing
    error("$prefix: predictor `$target` has full-cover groups $who " *
          "without an intercept — mutually collinear (each covers every " *
          "row); keep one full-cover group and subset the rest " *
          "(`factor(...; ref=..., cmc=false)` pins a level at zero)")
end

# Co-occurrence gate canonical forms. A lowered dotted expression
# normalizes two ways: scaling-normalized (scalar multiplications and
# divisions stripped — equal forms denote the same vector up to a
# nonzero scalar factor) and affine-normalized (scalar additions
# stripped too — equal forms are affine cousins over the same base).
# Staged names resolve through the derived registry; predictor terms
# never reference assignments (NamedColumn needs DataColumn backing),
# so anything neither data nor staged is an internal error.
# Scalar-valued forms (numbers, reductions, all-scalar calls) collapse
# to `_RK_GATE_CONST`; unrecognized operators keep their structure, so
# normalization is total but never strips what it cannot prove scalar.
const _RK_GATE_CONST = :__rk_gate_const__

function _rk_gate_derived_expr(name::Symbol,
        derived::AbstractVector, target::Symbol)
    for spec in derived
        spec.name === name || continue
        spec.expression isa Expr && return spec.expression
        error("RK backend: internal: derived `$name` in `$target` " *
            "has no expression to compare")
    end
    error("RK backend: internal: derived `$name` in `$target` " *
        "is not staged")
end

function _rk_gate_norm(node, data::AbstractDict,
        derived::AbstractVector, affine::Bool)
    _rk_gate_norm_inner(
        node, data, derived, affine, Set{Symbol}())
end

function _rk_gate_norm_inner(node, data::AbstractDict,
        derived::AbstractVector, affine::Bool, visited::Set{Symbol})
    prefix = "RK backend"
    node isa Number && return _RK_GATE_CONST
    if node isa Symbol
        haskey(data, node) && return node
        node in visited && error(
            "$prefix: internal: staged cycle at `$node`")
        for spec in derived
            if spec.name === node
                push!(visited, node)
                return _rk_gate_norm_inner(spec.expression, data,
                    derived, affine, visited)
            end
        end
        error("$prefix: internal: name `$node` is neither data nor staged")
    end
    node isa Expr || error(
        "$prefix: internal: cannot normalize `$(repr(node))`")
    if node.head === :.
        # Dotted math `log.(x)`: all-scalar stays scalar, else keep.
        length(node.args) == 2 && node.args[1] isa Symbol ||
            error("$prefix: internal: cannot normalize `$(repr(node))`")
        tup = node.args[2]
        tup isa Expr && tup.head === :tuple ||
            error("$prefix: internal: cannot normalize `$(repr(node))`")
        parts = Any[_rk_gate_norm_inner(a, data, derived, affine, visited)
                    for a in tup.args]
        all(p -> p === _RK_GATE_CONST, parts) && return _RK_GATE_CONST
        return (:math, node.args[1], parts...)
    end
    node.head === :call || error(
        "$prefix: internal: cannot normalize `$(repr(node))`")
    fn, args = node.args[1], node.args[2:end]
    fn isa Symbol || error(
        "$prefix: internal: cannot normalize `$(repr(node))`")
    haskey(_RK_DERIVED_RED_FN, fn) && return _RK_GATE_CONST
    if fn === :.*
        parts = Any[_rk_gate_norm_inner(a, data, derived, affine, visited)
                    for a in args]
        rest = filter(p -> p !== _RK_GATE_CONST, parts)
        isempty(rest) && return _RK_GATE_CONST
        length(rest) == 1 && return only(rest)
        return (:prod, sort!(rest; by=repr)...)
    end
    if fn === :./
        length(args) == 2 || error(
            "$prefix: internal: cannot normalize `$(repr(node))`")
        num = _rk_gate_norm_inner(
            args[1], data, derived, affine, visited)
        den = _rk_gate_norm_inner(
            args[2], data, derived, affine, visited)
        den === _RK_GATE_CONST && return num
        num === _RK_GATE_CONST && return (:inv, den)
        return (:div, num, den)
    end
    if fn === :.+ || fn === :.-
        length(args) == 2 || error(
            "$prefix: internal: cannot normalize `$(repr(node))`")
        left = _rk_gate_norm_inner(
            args[1], data, derived, affine, visited)
        right = _rk_gate_norm_inner(
            args[2], data, derived, affine, visited)
        if affine
            right === _RK_GATE_CONST && return left
            left === _RK_GATE_CONST && return right
        end
        left === _RK_GATE_CONST && right === _RK_GATE_CONST &&
            return _RK_GATE_CONST
        return (fn, left, right)
    end
    if fn === :.^
        length(args) == 2 || error(
            "$prefix: internal: cannot normalize `$(repr(node))`")
        base = _rk_gate_norm_inner(
            args[1], data, derived, affine, visited)
        expo = _rk_gate_norm_inner(
            args[2], data, derived, affine, visited)
        base === _RK_GATE_CONST && expo === _RK_GATE_CONST &&
            return _RK_GATE_CONST
        args[2] isa Number && args[2] == 0 && return _RK_GATE_CONST
        args[2] isa Number && args[2] == 1 && return base
        return (:pow, base, expo)
    end
    parts = Any[_rk_gate_norm_inner(a, data, derived, affine, visited)
                for a in args]
    all(p -> p === _RK_GATE_CONST, parts) && return _RK_GATE_CONST
    (fn, parts...)
end

# A continuous-cat cross over levels that partition the rows sums
# exactly to its continuous spine: Σ_dummies = 1 rowwise (bit-exact),
# and nested crosses splice partial sums level by level, so the fold
# below holds for arbitrarily nested `&` terms. Single-leaf spines
# project to the leaf; multi-leaf spines fold left-deep over `.*`
# (normalization sorts products, so association order is free).
function _rk_gate_sumfold(spine::Vector{Any})
    exprs = Any[id[2] for id in spine]
    foldl((a, b) -> Expr(:call, :.*, a, b), exprs)
end

# Mains-plus-crosses co-occurrence: a continuous `&` cross whose
# spine sums exactly to a main effect (up to a scalar factor) is
# structurally singular — the full cross already spans the main.
# Same for two crosses sharing a spine sum, and (under an intercept)
# for affine-cousin spine sums, where the intercept closes the rank
# gap. Non-affine distinct spines and intercept-free affine cousins
# stay admitted. Zero-column and nesting-residual designs are out of
# this gate's scope.
function _rk_gate_cross_identified!(terms::Vector{_RKTermSpec},
        spines::Dict{Symbol,Any}, derived::Vector{_RKDerivedSpec},
        data::AbstractDict, target::Symbol, has_intercept::Bool)
    prefix = "RK backend"
    crosses = Tuple{Symbol,Any}[] # (representative column, spine)
    for spec in terms
        spec.kind === :continuous || continue
        dname = only(spec.columns)
        haskey(spines, dname) || continue
        any(c -> c[2] == spines[dname], crosses) && continue
        push!(crosses, (dname, spines[dname]))
    end
    usable = filter(
        c -> !isempty(c[2][1]) && c[2][2] >= 1, crosses)
    isempty(usable) && return nothing
    mains = Tuple{Symbol,Any}[] # (addressee, identity expression)
    for spec in terms
        spec.kind === :continuous || continue
        dname = only(spec.columns)
        haskey(spines, dname) && continue
        identity = haskey(data, dname) ? dname :
            _rk_gate_derived_expr(dname, derived, target)
        push!(mains, (spec.addressee, identity))
    end
    sums = [(dname, _rk_gate_sumfold(spine[1]))
            for (dname, spine) in usable]
    for (dname, sumfold) in sums
        for (addressee, main) in mains
            _rk_gate_norm(main, data, derived, false) ==
                _rk_gate_norm(sumfold, data, derived, false) && error(
                "$prefix: predictor `$target` interaction `$dname` sums " *
                "exactly to main effect `$addressee` — structurally " *
                "singular (drop the main effect — the full cross already " *
                "spans it — or the interaction)")
            has_intercept &&
                _rk_gate_norm(main, data, derived, true) ==
                _rk_gate_norm(sumfold, data, derived, true) && error(
                "$prefix: predictor `$target` interaction `$dname` is an " *
                "affine cousin of main effect `$addressee` — singular " *
                "with an intercept (drop the intercept, the main effect, " *
                "or the interaction)")
        end
    end
    for i in 1:length(sums), j in (i + 1):length(sums)
        first, second = sums[i], sums[j]
        _rk_gate_norm(first[2], data, derived, false) ==
            _rk_gate_norm(second[2], data, derived, false) && error(
            "$prefix: predictor `$target` interactions `$(first[1])` and " *
            "`$(second[1])` sum to the same vector — structurally " *
            "singular (drop one of the interactions)")
        has_intercept &&
            _rk_gate_norm(first[2], data, derived, true) ==
            _rk_gate_norm(second[2], data, derived, true) && error(
            "$prefix: predictor `$target` interactions `$(first[1])` " *
            "and `$(second[1])` are affine cousins — singular with an " *
            "intercept (drop the intercept or one of the interactions)")
    end
    nothing
end

# ---- random effects (draws regime; peer Stage-A surface) ----
#
# Buckets mirror SB's draws path (`_sb_collect_id_buckets`,
# `_sb_emit_id_buckets!`, `ranefcoefnames`): one shared non-centered block
# per (id, group), sliced per target predictor. Plain `(x|g)` blocks are
# degenerate single-slice buckets. Totals-matchable models emit draws too,
# per decision 0yl36fh (draws-always, interim — the user notes RK-side
# marginalization may supersede it later, which would be additive).

_rk_ranef_suffix(id::Nothing, group::Symbol) = group
_rk_ranef_suffix(id::Symbol, group::Symbol) = Symbol(id, :_, group)
_rk_ranef_bucket_label(id, group) = Symbol(:bucket_, _rk_ranef_suffix(id, group))
_rk_ranef_gather_label(target, id, group) =
    Symbol(:r_, target, :_, _rk_ranef_suffix(id, group))

function _rk_ranef_group_column!(columns::Dict{Symbol,AbstractVector},
        taken::Set{Symbol}, gname::Symbol, raw::AbstractVector, what::String)
    prefix = "RK backend"
    raw isa CA.CategoricalVector || begin
        columns[gname] = raw
        return gname
    end
    # Categorical groupings bind factor-crossed strings (todo 14pgdwz/P1,
    # DONE): the thin layer numbers them by bind-derived sort order (peer
    # `_declared_codes`, RK >= fd1af39) — no outside-model codes, so the
    # last REAL outside-model computation on the RK side is gone. SB
    # numbers `CA.levels` order instead, so two shapes fail closed: a
    # custom-ordered declaration would silently misnumber groups (fixed
    # by the peer P2 declared-levels surface spelling,
    # ReactiveKernels:brm todo 15a8se2), and distinct levels sharing one
    # string form would collapse (same rule as the slope dummies below).
    # Unobserved declared levels keep today's drop behavior on both
    # sides (no regression, same P2).
    crossed = _rk_factor_crossed(raw)
    mask = .!ismissing.(raw)
    obs_strs = Set(crossed[mask])
    obs_lvls = Set(CA.levelcode.(raw)[mask])
    length(obs_strs) == length(obs_lvls) || error(
        "$prefix: $what grouping `$gname` has distinct levels with the " *
        "same string form; the draws regime needs unambiguous levels")
    sb_order = filter(lv -> lv in obs_strs, string.(CA.levels(raw)))
    issorted(sb_order) || error(
        "$prefix: $what grouping `$gname` declares custom-ordered " *
        "levels ($(join(repr.(CA.levels(raw)), ", "))); the thin layer " *
        "numbers groupings by sorted crossed strings — reorder the " *
        "declaration to sorted order or await the peer declared-levels " *
        "surface spelling (ReactiveKernels:brm P2 15a8se2)")
    columns[gname] = crossed
    gname
end

function _rk_ranef_dummies!(margins::Vector{_RKRanefMargin}, target::Symbol,
        term::NamedColumn, columns::Dict{Symbol,AbstractVector},
        what::String, first_level::Int)
    prefix = "RK backend"
    backing = parent(term)
    backing isa DataColumn || error(
        "$prefix: $what slope `$(name(term))` is not a raw data column")
    source = name(term)
    raw = parent(backing)
    eltype(raw) === Bool && error(
        "$prefix: $what slope `$source` is a Bool column; the draws " *
        "regime needs integer codes (recodify true/false as 1/0)")
    fitted = collect(_sb_fit_levels(raw))
    first_level == 2 && length(fitted) < 2 && error(
        "$prefix: $what slope `$source` has a single observed level, so " *
        "a treatment contrast is empty (SB drops such slopes silently; " *
        "the RK path needs an explicit slope — drop it or code it " *
        "intercept-free for its one cell mean)")
    keep = if raw isa CA.CategoricalVector
        # The thin layer derives levels from bound data, so only observed
        # levels cross (SB keeps unobserved declared levels as prior-only
        # coefficients; those columns contribute nothing observable).
        observed = Set{Int}(_brm_apply_fitted_levels(fitted, raw))
        [p for p in first_level:length(fitted) if p in observed]
    else
        collect(first_level:length(fitted))
    end
    isempty(keep) && return 0
    kept_values = raw isa CA.CategoricalVector ?
        string.(fitted[keep]) : fitted[keep]
    if raw isa CA.CategoricalVector
        length(unique(kept_values)) == length(kept_values) || error(
            "$prefix: $what slope `$source` has distinct levels with the " *
            "same string form; the draws regime needs unambiguous levels")
    end
    columns[source] = _rk_factor_crossed(raw)
    for k in kept_values
        push!(margins, _RKRanefMargin(target,
            Symbol(string(source) * "_dummy_" * string(k)),
            _RKRanefZRecipe(:dummy, source, k)))
    end
    length(kept_values)
end

function _rk_ranef_recipes!(margins::Vector{_RKRanefMargin}, lowered::Vector{Any},
        target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector}, derived::Vector{_RKDerivedSpec},
        taken::Set{Symbol}, what::String)
    prefix = "RK backend"
    admitted = "`1`, continuous columns, integer/string/categorical " *
        "columns, continuous `&` interactions"
    for t in lowered
        if t isa Integer
            t == 1 || error(
                "$prefix: $what has unsupported integer random-effect " *
                "term `$t` (admitted: `1`)")
            push!(margins, _RKRanefMargin(target, :Intercept,
                _RKRanefZRecipe(:ones, :none, nothing)))
        elseif t isa _SBCellMeansTerm
            _rk_ranef_dummies!(margins, target, t.term, columns, what, 1)
        elseif t isa ExprColumn && getf(t) === (&)
            # Same `&` recipe as the population path (D5a in-graph):
            # each crossed pair is a derived `.*` definition the thin
            # layer computes from raw columns, and the margin gathers
            # it like any continuous Z column.
            _rk_gate_ranef_interaction!(t, target, what, data)
            specs, _ = _rk_interaction_columns(t, target,
                "random-effect `&` interaction", data, columns,
                derived, taken)
            for (dname, _, dlabel) in specs
                push!(margins, _RKRanefMargin(target, dlabel,
                    _RKRanefZRecipe(:column, dname, nothing)))
            end
        elseif t isa NamedColumn
            backing = parent(t)
            backing isa DataColumn || error(
                "$prefix: $what slope `$(name(t))` is not a raw data " *
                "column (admitted: $admitted)")
            source = name(t)
            raw = parent(backing)
            # SB's ranef rule admits integer/categorical Z columns; string
            # columns ride along via the thin layer's exact-match dummies
            # (SB itself chokes materializing them).
            if !isnothing(_sb_cat_levels(t)) ||
                    (raw isa AbstractVector && eltype(raw) <: AbstractString)
                _rk_ranef_dummies!(margins, target, t, columns, what, 2)
            else
                raw isa AbstractVector{<:Real} &&
                    !(eltype(raw) <: Integer) || error(
                    "$prefix: $what slope `$source` must be a real " *
                    "non-integer vector (admitted: $admitted)")
                columns[source] = raw
                push!(margins, _RKRanefMargin(target, source,
                    _RKRanefZRecipe(:column, source, nothing)))
            end
        else
            head = t isa ExprColumn ? "`$(nameof(getf(t)))`" : "`$t`"
            error("$prefix: $what term $head is not in the draws regime " *
                "(admitted: $admitted; categorical `&` operands, " *
                "`offset()`, and transformed slopes are deferred)")
        end
    end
    margins
end

function _rk_gate_ranef_factor!(effects, target::Symbol, what::String)
    prefix = "RK backend"
    for t in effects
        t isa ExprColumn && getf(t) === factor || continue
        error("$prefix: $what `factor(...)` slopes are not in the draws " *
            "regime (the shared slope surface takes bare columns; use a " *
            "bare categorical column)")
    end
end

# Ranef `&` takes continuous leaves only. SB lowers ranef crosses
# treatment-coded (`_sb_interaction_cols!`: `2:n_levels` loops), while
# the shared `_rk_interaction_columns` recipe expands categorical
# leaves full-rank; admitting those would mismatch SB's Z in both
# count and values. Categorical crosses stay fail-closed here until a
# treatment-coded cross recipe exists — precompute numeric columns
# outside the formula. Non-leaf data expressions pass through: the
# shared recipe validates them, and neither side admits categorical
# columns inside arithmetic.
function _rk_gate_ranef_interaction!(t, target::Symbol, what::String,
        data::AbstractDict)
    prefix = "RK backend"
    if t isa ExprColumn && getf(t) === (&)
        for a in getargs(t)
            _rk_gate_ranef_interaction!(a, target, what, data)
        end
        return nothing
    end
    if t isa ExprColumn && getf(t) === factor
        error("$prefix: $what `factor(...)` inside `&` is not in the " *
            "draws regime (random-effect crosses take continuous " *
            "columns; precompute numeric columns outside the formula)")
    end
    if t isa ExprColumn && (getf(t) === zscale || getf(t) === center ||
            getf(t) === standardize)
        args = getargs(t)
        length(args) == 1 || error(
            "$prefix: $what `$(nameof(getf(t)))` inside `&` needs " *
            "exactly one argument")
        _rk_gate_ranef_interaction!(only(args), target, what, data)
        return nothing
    end
    t isa NamedColumn || return nothing
    source = name(t)
    raw = get(data, source, nothing)
    raw isa AbstractVector && _brm_is_categorical_data(raw) && error(
        "$prefix: $what `&` operand `$source` is categorical, which " *
        "is not in the draws regime (SB codes ranef crosses " *
        "treatment-coded; precompute numeric columns outside the " *
        "formula)")
    return nothing
end

function _rk_lower_ranef_bucket(context, id::Union{Nothing,Symbol},
        gname::Symbol, targets::Vector{Any},
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol},
        derived::Vector{_RKDerivedSpec})
    prefix = "RK backend"
    data = context.data
    raw = get(data, gname, nothing)
    bound = _rk_ranef_group_column!(columns, taken, gname, raw,
        id === nothing ? "predictor `$(first(targets)[1])` random effect" :
            "`|$id|` random-effect block")
    margins = _RKRanefMargin[]
    slices = Tuple{Symbol,UnitRange{Int}}[]
    cursor = 0
    for (target, d) in targets
        what = id === nothing ? "predictor `$target` random effect" :
            "`|$id|` random-effect block for predictor `$target`"
        _rk_gate_ranef_factor!(d.effects, target, what)
        lowered = _sb_ranef_lowered_terms(collect(Any, d.effects))
        before = length(margins)
        _rk_ranef_recipes!(margins, lowered, target, data, columns,
            derived, taken, what)
        ncols = length(margins) - before
        if ncols == 0
            # Degenerate blocks span no coefficients: plain ones are a
            # no-op, ID targets error (both mirror SB).
            id === nothing && continue
            error("$prefix: $what spans no coefficients (every slope " *
                "degenerated; mirrors SB: an ID bucket target needs at " *
                "least one column)")
        end
        push!(slices, (target, (cursor+1):(cursor+ncols)))
        cursor += ncols
    end
    isempty(margins) && return nothing
    kind = if id !== nothing || length(margins) > 1
        :correlated
    elseif margins[1].z.kind === :ones
        :intercept1
    else
        :slope1
    end
    eta = kind === :correlated ? 1.0 : NaN
    _RKRanefBucket(id, bound, kind, margins, slices, eta,
        _rk_ranef_bucket_label(id, bound))
end

function _rk_plan_ranef_buckets(brmi::BRMI, context,
        predictor_order::Vector{Symbol},
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol},
        derived::Vector{_RKDerivedSpec})
    prefix = "RK backend"
    buckets = _RKRanefBucket[]
    lookup = Dict{Tuple{Symbol,Symbol,Union{Nothing,Symbol}},
        Union{_RKRanefBucket,Nothing}}()
    declarations =
        [d for d in context.group_declarations if d.predictor in predictor_order]
    isempty(declarations) && return buckets, lookup
    for d in declarations
        what = "predictor `$(d.predictor)` random effect"
        d.uncorrelated && error(
            "$prefix: $what with `||` is not in the draws regime " *
            "(admitted: `(effects | group)`, `(effects | ID | group)`)")
        d.descriptor isa MultiMembershipTerm && error(
            "$prefix: $what with `mm(...)` is not in the draws regime " *
            "(admitted: `(effects | group)`, `(effects | ID | group)`)")
        d.descriptor isa Tuple && error(
            "$prefix: $what with `gr(...; by=...)` is not in the draws " *
            "regime (admitted: `(effects | group)`, " *
            "`(effects | ID | group)`)")
        isempty(d.effects) && error(
            "$prefix: $what has no terms after dropping `0` (mirrors SB)")
    end
    if !isempty(ranef_effect_priors(brmi))
        error("$prefix: `sd(...)`/`cor(...)` random-effect priors are not " *
            "in the draws regime (buckets take LKJ(1.0) + half-normal " *
            "scales); drop the statements")
    end
    plain_keys = Tuple{Symbol,Symbol}[]
    plain_decls = Dict{Tuple{Symbol,Symbol},Any}()
    id_keys = Tuple{Symbol,Symbol}[]
    id_decls = Dict{Tuple{Symbol,Symbol},Vector{Any}}()
    id_groups = Dict{Symbol,Symbol}()
    for d in declarations
        desc = d.descriptor
        desc isa NamedColumn || error(
            "$prefix: internal: unexpected group descriptor " *
            "`$(typeof(desc))`")
        gcol = desc
        parent(gcol) isa DataColumn || error(
            "$prefix: predictor `$(d.predictor)` group `$(name(gcol))` " *
            "must be a raw data column")
        gname = name(gcol)
        get(context.data, gname, nothing) isa AbstractVector || error(
            "$prefix: predictor `$(d.predictor)` grouping column " *
            "`$gname` must be a vector")
        if isnothing(d.id)
            key = (d.predictor, gname)
            haskey(plain_decls, key) && error(
                "$prefix: predictor `$(d.predictor)` repeats " *
                "random-effect block group `$gname` (mirrors SB: merge " *
                "the declarations into one `(effects | $gname)`)")
            push!(plain_keys, key)
            plain_decls[key] = d
        else
            haskey(id_groups, d.id) && id_groups[d.id] != gname && error(
                "$prefix: `|$(d.id)|` sees conflicting grouping factors " *
                "(`$(id_groups[d.id])` vs `$gname`) (mirrors SB)")
            id_groups[d.id] = gname
            key = (d.id, gname)
            prior = get(id_decls, key, Any[])
            any(p -> p.predictor === d.predictor, prior) && error(
                "$prefix: predictor `$(d.predictor)` repeats " *
                "random-effect block ID `$(d.id)`, group `$gname` " *
                "(mirrors SB: merge the declarations)")
            haskey(id_decls, key) || push!(id_keys, key)
            push!(get!(id_decls, key, Any[]), d)
        end
    end
    for key in plain_keys
        target, gname = key
        bucket = _rk_lower_ranef_bucket(context, nothing, gname,
            Any[(target, plain_decls[key])], columns, taken, derived)
        # A `nothing` bucket is a fully degenerate plain block: record it
        # so the predictor attaches no gather (vs a missing key, which is
        # an internal error).
        lookup[(target, gname, nothing)] = bucket
        isnothing(bucket) || push!(buckets, bucket)
    end
    for key in id_keys
        id, gname = key
        decls = id_decls[key]
        bucket = _rk_lower_ranef_bucket(context, id, gname,
            Any[(d.predictor, d) for d in decls], columns, taken, derived)
        for d in decls
            lookup[(d.predictor, gname, id)] = bucket
        end
        isnothing(bucket) || push!(buckets, bucket)
    end
    buckets, lookup
end

# Offset-only predictors (SB-admitted) carry no coefficient columns,
# so the shared geometry — which requires at least one — cannot build
# them. The empty-column design builds directly instead: the
# fixed-offset vector still materializes and validates its row axis,
# and the prior/r2d2 seam runs unchanged over zero columns (a stated
# effect or r2d2 prior keeps its fail-closed error).
function _rk_plan_offset_only_predictor(brmi::BRMI, context, target::Symbol,
        ordinary::Tuple, available::Tuple, link::Symbol,
        terms::Vector{_RKTermSpec}, derived::Vector{_RKDerivedSpec})
    prefix = "RK backend"
    isempty(terms) && error(
        "$prefix: internal: predictor `$target` has no term specs")
    raw_sources = [column for term in terms for column in term.columns
        if haskey(context.data, column)]
    row_source = isempty(raw_sources) ?
        get(context.target_obs, target, nothing) : first(raw_sources)
    isnothing(row_source) && error(
        "$prefix: predictor `$target` has no data column from which to " *
        "determine its row axis")
    design = _brm_population_design(target, ordinary, context.data,
        get(context.target_obs, target, nothing);
        required=true, row_source,
        implicit_intercept=target in _brm_threshold_located_predictors(brmi))
    hs_priors = _rk_horseshoe_priors(brmi, design, target, available,
        terms, nothing)
    priors = _rk_population_priors(brmi, design, target, available,
        Set{Symbol}(), terms, derived, nothing,
        Set{Symbol}(p.addressee for p in hs_priors))
    r2d2 = _brm_whole_predictor_r2d2(brmi, design, (); prefix,
        available_predictors=available)
    isnothing(r2d2) || error(
        "$prefix: predictor `$target` `r2d2` decomposes nothing (no " *
        "coefficient columns); drop the `r2d2` statement")
    _RKPredictorSpec(target, link, terms, target), priors, nothing, hs_priors
end

# ---- spline smooth terms (s/t2; mirrors `_sb_s_generic`/`_sb_t2_generic`) ----
#
# A `:spline` term carries its thin-layer declaration in `options`:
# `(; id, kind, k)` with `kind in (:tps, :t2)` and `k` an `Int` (`s`)
# or an `(Int, Int)` tuple (`t2`). The thin layer owns every parameter
# (`b_<id>_fixed`, `b_<id>_raw`/`_<block>_raw`, `sd_<id>`) plus the
# materialized `<id>_<block>_<j>` columns: BRM ships raw axes + the
# declaration, and the basis fit stays in-graph host-side (user-GO'd
# in-graph contract, decision `1cj6p76`). One id per smooth occurrence
# (exactly-one-use linkage), minted below with numeric stems on collision.
# One smooth-id domain for every basis-declaring smooth (`s`/`t2`/`hsgp`):
# ids must not collide with each other, user names (`taken`), or raw
# columns — numeric stems on collision.
function _rk_mint_generated!(taken::Set{Symbol},
        columns::Dict{Symbol,AbstractVector}, base::String)
    name = Symbol(base)
    serial = 2
    while name in taken || haskey(columns, name)
        name = Symbol(base * "_" * string(serial))
        serial += 1
    end
    push!(taken, name)
    name
end

function _rk_mint_smooth_id!(taken::Set{Symbol},
        columns::Dict{Symbol,AbstractVector}, base::String)
    _rk_mint_generated!(taken, columns, base)
end

# `sd(...)` smoothing-scale overrides are sequenced: the thin-layer
# spline surface takes default half-normal scales and fails overrides
# closed, so BRM rejects them here with RK attribution instead of
# emitting a default the formula did not ask for.
function _rk_gate_spline_term_priors!(brmi::BRMI, target::Symbol,
        spline_raw::AbstractVector)
    prefix = "RK backend"
    isempty(spline_raw) && return nothing
    per_target = get(_brm_resolve_term_priors(brmi), target, Dict())
    for t in spline_raw
        key = _brm_prepared_term_key(t)
        isempty(get(per_target, key, Dict())) || error(
            "$prefix: predictor `$target` smoothing-scale priors on " *
            "`$key` are out of slice 1 (the thin-layer spline surface " *
            "takes default half-normal scales; `sd(...)` overrides are " *
            "sequenced)")
    end
    nothing
end

function _rk_plan_spline_axis!(axis::Symbol, target::Symbol, head::String,
        data::AbstractDict, columns::Dict{Symbol,AbstractVector})
    prefix = "RK backend"
    raw = get(data, axis, nothing)
    raw isa AbstractVector && eltype(raw) <: Real &&
        !(eltype(raw) <: Bool) &&
        !(raw isa CA.CategoricalVector) || error(
        "$prefix: predictor `$target` `$head` axis `$axis` must be a " *
        "plain numeric vector")
    columns[axis] = raw
    nothing
end

function _rk_plan_spline_term!(prepared::_BRMPreparedTerm{typeof(s)},
        target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol})
    axis = prepared.source
    _rk_plan_spline_axis!(axis, target, "s", data, columns)
    id = _rk_mint_smooth_id!(taken, columns, "s_" * string(axis))
    _RKTermSpec(:spline, [axis],
        (; id, kind=:tps, k=prepared.state.fit.k), id, id)
end

function _rk_plan_spline_term!(prepared::_BRMPreparedTerm{typeof(t2)},
        target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol})
    first_axis, second_axis = prepared.source
    _rk_plan_spline_axis!(first_axis, target, "t2", data, columns)
    _rk_plan_spline_axis!(second_axis, target, "t2", data, columns)
    base = "t2_" * string(first_axis) * "_" * string(second_axis)
    id = _rk_mint_smooth_id!(taken, columns, base)
    _RKTermSpec(:spline, [first_axis, second_axis],
        (; id, kind=:t2, k=prepared.state.fit.k), id, id)
end

# ---- exact-GP latent terms (iso single-axis; mirrors `_sb_gp`) ----
#
# A `:gp` term carries its thin-layer names + hyper priors in `options`:
# `(; rho, sigma, z, f, jitter, rho_param, sigma_param)`. The hypers are
# `_RKSampledParameter`s emitted by the AST preamble (before the predictor
# affine that uses `f`), NOT entries of `plan.parameters` — the parameters
# loop emits after predictors, which would violate topo order. The AST is
# the sole emission path; overlap alpha-renames and GP composes with it
# (preamble names never collide with the affine).

function _rk_mint_gp_stem!(taken::Set{Symbol},
        columns::Dict{Symbol,AbstractVector})
    stem = ""
    names(stem) = (Symbol(:rho_gp, stem), Symbol(:sigma_gp, stem),
                   Symbol(:z_gp, stem), Symbol(:f_gp, stem))
    while any(nm -> nm in taken || haskey(columns, nm), names(stem))
        stem = stem == "" ? "2" : string(parse(Int, stem) + 1)
    end
    for nm in names(stem)
        push!(taken, nm)
    end
    stem
end

const _RK_GP_HYPER_ADMITTED = "LogNormal, InverseGamma, Gamma, Exponential, " *
    "or zero-location Normal"

function _rk_gp_hyper_prior(prior, role::String, target::Symbol, axis::Symbol)
    prefix = "RK backend"
    where = "predictor `$target` `gp($axis)` $role"
    prior isa ExprColumn || error(
        "$prefix: $where prior is not a distribution call")
    f = getf(prior)
    f isa Type || error(
        "$prefix: $where prior is out of slice 1 (admitted: " *
        "$_RK_GP_HYPER_ADMITTED)")
    name = nameof(f)
    isempty(getkwargs(prior)) || error(
        "$prefix: $where prior cannot have keywords in slice 1")
    args = getargs(prior)
    if name === :Normal
        length(args) == 2 || error(
            "$prefix: $where prior `Normal` needs 2 " *
            "arguments, got $(length(args))")
        location, scale = args
        location isa Number && location == 0 || error(
            "$prefix: $where `Normal` prior must have " *
            "location 0 in slice 1 (a positive scale takes a half-Normal)")
        scale isa Number && isfinite(Float64(scale)) || error(
            "$prefix: $where prior hyperparameters must be " *
            "finite literals")
        return (:Normal, (0.0, Float64(scale)), :positive)
    end
    (name === :LogNormal || name === :InverseGamma || name === :Gamma ||
        name === :Exponential) || error(
        "$prefix: $where prior `$name` is out of slice 1 " *
        "(admitted: $_RK_GP_HYPER_ADMITTED)")
    expected = _RK_SLICE1_PRIOR_ARITY[name]
    length(args) == expected || error(
        "$prefix: $where prior `$name` needs $expected " *
        "argument(s), got $(length(args))")
    resolved = map(args) do arg
        arg isa Number && isfinite(Float64(arg)) || error(
            "$prefix: $where prior hyperparameters must be " *
            "finite literals")
        Float64(arg)
    end
    (name, Tuple(resolved), nothing)
end

function _rk_plan_gp_term!(prepared::_BRMPreparedTerm{typeof(gp)},
        target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol})
    prefix = "RK backend"
    state = prepared.state
    state.cov === :exp_quad || error(
        "$prefix: predictor `$target` `gp(...; cov=$(repr(state.cov)))` " *
        "is out of slice 1 (the thin-layer surface is exp_quad; " *
        "periodic is sequenced)")
    (state.iso && length(prepared.source) == 1) || error(
        "$prefix: predictor `$target` anisotropic or multi-axis `gp(...)` " *
        "is out of slice 1 (the thin-layer surface is iso single-axis; " *
        "sequenced)")
    axis = only(prepared.source)
    _rk_plan_spline_axis!(axis, target, "gp", data, columns)
    stem = _rk_mint_gp_stem!(taken, columns)
    rho = Symbol(:rho_gp, stem)
    sigma = Symbol(:sigma_gp, stem)
    z = Symbol(:z_gp, stem)
    f = Symbol(:f_gp, stem)
    rho_family, rho_args, rho_support = _rk_gp_hyper_prior(
        state.rho_prior, "length-scale", target, axis)
    sig_family, sig_args, sig_support = _rk_gp_hyper_prior(
        state.sigma_prior, "marginal-scale", target, axis)
    _RKTermSpec(:gp, [axis],
        (; rho, sigma, z, f, jitter=Float64(state.jitter),
         rho_param=_RKSampledParameter(
             rho, rho_family, rho_args, rho_support, rho),
         sigma_param=_RKSampledParameter(
             sigma, sig_family, sig_args, sig_support, sigma)),
        f, f)
end

# ---- HSGP smooth terms (mirrors `_sb_hsgp`/`_sb_hsgp_aniso`) ----
#
# A `:hsgp` term carries its thin-layer declaration in `options`:
# `(; id, k, c, iso)` with `k`/`c` scalars (one axis) or per-axis
# tuples (variadic axes), normalized from the prepared state's
# per-axis tuples. The thin layer owns every parameter
# (`beta_raw_<id>`, `rho_<id>`/`rho_<id>_1..d`, `sigma_<id>`) and
# evaluates the basis in-graph from the raw axes: BRM ships the
# recipe, never materialized `PHI`/`omega2` (user-GO'd in-graph
# contract, decision `02e64eo`). One id per smooth occurrence
# (exactly-one-use linkage), minted with numeric stems on collision.

# `length_scale(...)`/`sd(...)` hyper overrides are sequenced: the
# thin-layer hsgp surface is self-priored with `LogNormal(0, 1)`
# defaults (the floor-zeroing override surface is a peer follow-up),
# so BRM rejects them here with RK attribution instead of emitting a
# default the formula did not ask for.
function _rk_gate_hsgp_term_priors!(brmi::BRMI, target::Symbol,
        hsgp_raw::AbstractVector)
    prefix = "RK backend"
    isempty(hsgp_raw) && return nothing
    per_target = get(_brm_resolve_term_priors(brmi), target, Dict())
    for t in hsgp_raw
        key = _brm_prepared_term_key(t)
        isempty(get(per_target, key, Dict())) || error(
            "$prefix: predictor `$target` hyper priors on " *
            "`$key` are out of slice 1 (the thin-layer hsgp surface " *
            "is self-priored with LogNormal(0, 1) defaults; " *
            "`length_scale(...)`/`sd(...)` overrides are sequenced)")
    end
    nothing
end

function _rk_plan_hsgp_term!(prepared::_BRMPreparedTerm{typeof(hsgp)},
        target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol})
    prefix = "RK backend"
    state = prepared.state
    state.latent && error(
        "$prefix: predictor `$target` model-derived `hsgp(...)` axis " *
        "is out of slice 1 (the thin-layer surface binds raw data " *
        "columns; latent axes are sequenced)")
    state.cov === :exp_quad || error(
        "$prefix: predictor `$target` `hsgp(...; cov=$(repr(state.cov)))` " *
        "is out of slice 1 (the thin-layer surface is exp_quad; " *
        "periodic is sequenced)")
    isnothing(state.by) || error(
        "$prefix: predictor `$target` grouped `hsgp(...; by=...)` " *
        "is out of slice 1 (the thin-layer surface is ungrouped; " *
        "`by=` weights are sequenced)")
    any(!iszero, state.centeredness) && error(
        "$prefix: predictor `$target` partially-centered `hsgp(...)` " *
        "is out of slice 1 (the thin-layer surface is non-centered; " *
        "partial centering is sequenced)")
    state.explicit_domain && error(
        "$prefix: predictor `$target` `hsgp(...; domain=...)` " *
        "is out of slice 1 (the thin-layer surface fits the boundary " *
        "from raw columns; explicit domains are sequenced)")
    state.orthogonal === nothing || error(
        "$prefix: predictor `$target` `hsgp(...; orthogonal_to=:linear)` " *
        "is out of slice 1 (the thin-layer surface takes the raw " *
        "tensor-product basis; orthogonalization is sequenced)")
    axes = Tuple(prepared.source)
    for axis in axes
        _rk_plan_spline_axis!(axis, target, "hsgp", data, columns)
    end
    base = "hsgp_" * join(string.(axes), "_")
    id = _rk_mint_smooth_id!(taken, columns, base)
    k = length(state.K) == 1 ? only(state.K) : state.K
    c = length(state.c) == 1 ? only(state.c) : state.c
    _RKTermSpec(:hsgp, collect(axes), (; id, k, c, iso=state.iso), id, id)
end

# ---- AR(1) latent-path terms (mirrors `_sb_ar1`) ----
#
# An `:ar` term carries its thin-layer scan names in `options`:
# `(; state, phi, phi_raw, eps)`. The state is the `@scan` carried
# array, `phi = tanh(phi_raw)` the stationarity map over the sampled
# `phi_raw ~ Normal(0, 1)`, and `eps` the per-step innovation local.
# The thin layer owns the innovations (`_ppl_scan_z_<state>` under a
# `Normal(0, 1)` plate-vector prior) and folds the state through
# RK-core `scan(...)` (`u[1] = eps[1]`,
# `u[t] = phi*u[t-1] + eps[t]` — SB's `ar1_recurse` verbatim); BRM
# ships the time axis (a length probe, as in SB) + the declaration,
# and the path takes a free population beta through the AST's generic
# coef path (`coef .* state`, classified thin-layer-side as a
# `ScanSummandTerm`). Names are SB-deterministic
# (`ar_<predictor>_<time>`): a duplicate `ar` term re-derives the
# names SB's own declarations would collide on, so it fails closed
# exactly as SB does (mo precedent) — never a serialized second
# exchangeable path. The time VALUES never cross (rows are already in
# time order, as in SB); the loop bound is the thin-layer data-length
# name (`T`, which binds `n_obs`).
function _rk_plan_ar_term!(prepared::_BRMPreparedTerm{typeof(ar)},
        target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol})
    prefix = "RK backend"
    source = prepared.source
    raw = get(data, source, nothing)
    raw isa AbstractVector && eltype(raw) <: Real &&
        !(eltype(raw) <: Bool) &&
        !(raw isa CA.CategoricalVector) || error(
        "$prefix: predictor `$target` `ar` time axis `$source` must be a " *
        "plain numeric vector")
    columns[source] = raw
    # SB's emitted Stan column name (`ar_<predictor>_<time>`, `sbimpl.jl`
    # walker): the beta's deterministic address label and scan-state
    # stem. (Explicit `effect(...)` addresses speak SB's `popcoefnames`
    # vocabulary, the un-namespaced `ar_<time>` — see
    # `_rk_gate_ar_effect_priors!`.)
    base = "ar_" * string(target) * "_" * string(source)
    addressee = Symbol(base)
    state, phi, phi_raw, eps = Symbol(base), Symbol(:phi_, base),
        Symbol(:phi_raw_, base), Symbol(:eps_, base)
    for nm in (state, phi, phi_raw, eps)
        (nm in taken || haskey(columns, nm)) && error(
            "$prefix: predictor `$target` `ar` latent name `$nm` is " *
            "already taken — duplicate `ar($source)` terms collide " *
            "exactly as SB's deterministic names do (rename the " *
            "colliding parameter, data column, or term)")
    end
    for nm in (state, phi, phi_raw, eps)
        push!(taken, nm)
    end
    _RKTermSpec(:ar, [source], (; state, phi, phi_raw, eps),
        addressee, addressee)
end

# An `ar` latent path needs a sibling population coefficient: the
# thin-layer surface pairs every scan summand with an intercept or
# coefficient (`mu = a .+ b .* u`) and fails a scan-only predictor
# closed. Gate it here with RK attribution. (Address collisions with
# fellow terms need no separate check: every fellow addressee a
# deterministic `ar_*` name could equal is already in `taken`/bound
# columns when the term plans, so the taken check in
# `_rk_plan_ar_term!` fires first.)
function _rk_gate_ar_sibling!(terms::Vector{_RKTermSpec}, target::Symbol)
    prefix = "RK backend"
    any(t -> t.kind === :ar, terms) || return nothing
    any(t -> t.kind === :intercept || t.kind === :continuous ||
        t.kind === :factor, terms) || error(
        "$prefix: predictor `$target` carries an `ar(...)` latent path " *
        "with no sibling population coefficient — the thin-layer scan " *
        "surface pairs every scan summand with an intercept or " *
        "coefficient (`mu = a .+ b .* u`); add an intercept or a " *
        "continuous/factor term")
    nothing
end

# Explicit `effect(...)` addresses on the `ar` latent column are
# sequenced: the shared seam resolves design columns only, so it would
# misreport the (SB-valid) label as "not a population coefficient".
# Reject it here with RK attribution instead. Runs before predictor
# geometry (which invokes the shared seam); labels derive
# syntactically, leniently — malformed terms stay shared prep's error.
# The label is SB's `popcoefnames` vocabulary (`ar_<time>`, WITHOUT
# the predictor namespace the emitted Stan column carries): SB
# resolves `effect(mu, ar_x)` and rejects `effect(mu, ar_mu_x)`
# ("not a population coefficient"), and the RK side matches both —
# the namespaced spelling falls through to the shared seam below.
function _rk_gate_ar_effect_priors!(brmi::BRMI, target::Symbol,
        ar_raw::AbstractVector)
    prefix = "RK backend"
    isempty(ar_raw) && return nothing
    addresses = Set{Symbol}()
    for t in ar_raw
        t isa ExprColumn || continue
        args = getargs(t)
        length(args) == 1 || continue
        axis = only(args)
        axis isa NamedColumn || continue
        push!(addresses, Symbol(:ar_, name(axis)))
    end
    isempty(addresses) && return nothing
    for spec in effect_priors(brmi)
        spec.coefficient in addresses || continue
        (spec.predictor === _EFFECT_COLON || spec.predictor === target) ||
            continue
        error("$prefix: predictor `$target` explicit priors on the `ar` " *
            "latent column `$(spec.coefficient)` are out of slice 1 " *
            "(address the predictor with `effect($target, :)`, or take " *
            "the Normal(0, 1) default; per-column `effect(...)` " *
            "addresses on latent columns are sequenced)")
    end
    nothing
end

# The `ar` beta's prior: the shared seam resolves design columns only,
# and the latent column is not one — so `:`-wide statements claim this
# cell through the SAME claim engine (tie semantics identical to the
# design cells), defaulting to Normal(0, 1) (SB's `popefs` default).
# Runs after the shared seam (which owns spec validation); explicit
# ar addresses never reach here (gated above).
function _rk_ar_beta_prior(brmi::BRMI, target::Symbol, addressee::Symbol)
    prefix = "RK backend"
    cell = Ref{Any}(nothing)
    for spec in effect_priors(brmi)
        (spec.predictor === _EFFECT_COLON || spec.predictor === target) ||
            continue
        spec.coefficient === _EFFECT_COLON || continue
        _brm_claim_effect_prior!(() -> cell[], v -> (cell[] = v), spec,
            "`$target`'s `$addressee` column"; prefix)
    end
    held = cell[]
    isnothing(held) && return (0.0, 1.0)
    expression = held.expression
    expression isa ExprColumn && getf(expression) === Normal || error(
        "$prefix: predictor `$target` population-effect priors must " *
        "be `Normal(location, scale)` in slice 1")
    isempty(getkwargs(expression)) || error(
        "$prefix: predictor `$target` population-effect `Normal` " *
        "prior cannot have keywords in slice 1")
    raw_location, raw_scale = _brm_normal_effect_args(expression; prefix)
    location = _brm_numeric_constant(raw_location)
    scale = _brm_numeric_constant(raw_scale)
    isnothing(location) && error(
        "$prefix: population-effect Normal location must be a numeric constant")
    isnothing(scale) && error(
        "$prefix: population-effect Normal scale must be a numeric constant")
    isfinite(location) || error(
        "$prefix: population-effect Normal location must be finite")
    isfinite(scale) && scale > 0 || error(
        "$prefix: population-effect Normal scale must be finite and positive")
    location, scale
end

# ---- monotonic terms (mo/mo1; mirrors `_sb_mo`) ----
#
# A `:monotonic` term carries `(; increments, alpha, source)`: the
# increment-simplex vector parameter name (SB's `<mo_c>_simplex_incr`),
# its frozen literal concentration, and the raw ordinal column. A
# `:monotonic_summand` (`mo1`) carries the same and is beta-free
# (self-addressed: no population prior). The thin layer owns the
# stick-breaking geometry + Dirichlet density + level-gather recipe;
# BRM ships the bound `<c>_idx` codes (SB's `<c>_idx`) + the declaration.
# One increments name per monotonic occurrence (exactly-one-use linkage),
# minted with numeric stems on collision. SB suffixes `mo`/`mo1` contrasts
# per occurrence (snag mo-term-in-sever-fe459870), so the same column in two
# predictors plans two independent increments vectors here; only a repeat
# within ONE predictor fails closed in `_rk_gate_monotonic_unique!` (RK mints
# one monotonic label per (predictor, source)). `mo(c)` + `mo1(c)` coexist
# (separate contrasts, separate simplexes — SB-accepted, parity-held).
function _rk_plan_monotonic_core!(head::Symbol,
        prepared::_BRMPreparedTerm, target::Symbol,
        columns::Dict{Symbol,AbstractVector}, taken::Set{Symbol})
    prefix = "RK backend"
    source = prepared.source
    K = length(prepared.state.levels)
    # K=1 degenerates caller-side (`mo` vanishes like SB's dropped column;
    # `mo1` zeros like SB's scalar `0.0` summand), before the simplex prior
    # is even consulted — SB returns before `_sb_mo_prior_plan` too.
    K < 2 && return nothing
    idx_name = Symbol(source, :_idx)
    # SB's unconditional `data[<c>_idx] = idx` overwrite, mirrored: a raw
    # column literally named `<c>_idx` is clobbered in both backends
    # (reproduced garbage, not diverged garbage).
    columns[idx_name] = prepared.state.idx
    alpha = prepared.state.alpha
    if isnothing(alpha)
        constructor = getf(prepared.state.simplex_prior)
        T = _as_distribution_type(constructor)
        if isnothing(T) || !(T <: Dirichlet)
            error("$prefix: predictor `$target` `$head($source)` simplex " *
                "prior is not Dirichlet; slice 1 admits `Dirichlet` " *
                "concentrations only (the thin-layer increments are " *
                "Dirichlet-sampled)")
        end
        error("$prefix: predictor `$target` `$head($source)` Dirichlet " *
            "concentration is not a literal vector; slice 1 admits " *
            "literal concentrations only (sampled/data concentrations " *
            "fail closed thin-layer-side)")
    end
    alpha_vec = collect(Float64, alpha)
    length(alpha_vec) == K - 1 || error(
        "$prefix: internal: `$head($source)` concentration has " *
        "$(length(alpha_vec)) entries for $K levels")
    increments = _rk_mint_generated!(
        taken, columns, string(head, "_", source, "_simplex_incr"))
    (; source, idx_name, increments, alpha=alpha_vec)
end

function _rk_plan_mo_term!(prepared::_BRMPreparedTerm{typeof(mo)},
        target::Symbol, columns::Dict{Symbol,AbstractVector},
        taken::Set{Symbol})
    core = _rk_plan_monotonic_core!(
        :mo, prepared, target, columns, taken)
    # Single-level factor: 0 increments — the free-beta monotonic effect is
    # identically 0, so contribute NO term (SB returns no column: no beta,
    # no `simplex[0]`). A lone `0 + mo(c)` is re-inflated to a zeros offset
    # after the geometry loop (SB's scalar `0.0`, vector-shaped).
    isnothing(core) && return nothing
    _RKTermSpec(:monotonic, [core.idx_name],
        (; increments=core.increments, alpha=core.alpha, source=core.source),
        core.idx_name, Symbol(:mo_, target, :_, core.source))
end

function _rk_plan_mo1_term!(prepared::_BRMPreparedTerm{typeof(mo1)},
        target::Symbol, columns::Dict{Symbol,AbstractVector},
        taken::Set{Symbol})
    core = _rk_plan_monotonic_core!(
        :mo1, prepared, target, columns, taken)
    label = Symbol(:mo1_, target, :_, prepared.source)
    if isnothing(core)
        # Single-level factor: SB contributes a scalar `0.0` summand. A
        # zeros offset is the vector-shaped twin: it keeps the summand
        # list non-empty with identical values, without a `simplex[0]`.
        zero = _rk_mint_generated!(
            taken, columns, "mo1_$(prepared.source)_zero")
        columns[zero] = zeros(length(prepared.state.idx))
        return _RKTermSpec(:offset, [zero], (;), zero, Symbol(:offset_, zero))
    end
    _RKTermSpec(:monotonic_summand, [core.idx_name],
        (; increments=core.increments, alpha=core.alpha, source=core.source),
        label, label)
end

# The `mo` free beta is a population coefficient the shared design does not
# materialize (its contrast needs the increment simplex), so the shared seam
# cannot resolve its prior: default Normal(0, 1) (SB's `std_normal()`), with
# whole-predictor `effect(lp, :)` / `effect(:, :)` claims fanning out onto it
# through the shared precedence engine — SB applies `:` to the mo column too
# (verified against emitted Stan). Generated-name `effect(mu, mo_c)` claims
# stay unaddressable (they fail in the shared seam, like generated
# interaction labels): the colon is the mainline spelling.
function _rk_mo_beta_prior(brmi::BRMI, target::Symbol, addressee::Symbol)
    prefix = "RK backend"
    won = Ref{Any}(nothing)
    for spec in effect_priors(brmi)
        spec.coefficient === _EFFECT_COLON || continue
        spec.predictor === _EFFECT_COLON || spec.predictor === target ||
            continue
        _brm_claim_effect_prior!(() -> won[], v -> (won[] = v), spec,
            "`$target`'s monotonic `$addressee` column"; prefix)
    end
    expression = isnothing(won[]) ? nothing : won[].expression
    if !isnothing(expression)
        expression isa ExprColumn && getf(expression) === Normal || error(
            "$prefix: predictor `$target` population-effect priors must " *
            "be `Normal(location, scale)` in slice 1")
        isempty(getkwargs(expression)) || error(
            "$prefix: predictor `$target` population-effect `Normal` " *
            "prior cannot have keywords in slice 1")
    end
    location, scale = _brm_materialize_normal_effect_priors(
        Any[expression], 1; prefix)
    _RKPopulationPrior(target, addressee, location[1], scale[1])
end

# Per-coefficient structured Horseshoe (SB `_sb_horseshoe_overrides`
# mirror): each Horseshoe-addressed population column becomes an
# `_RKHorseshoePrior` with validated literal scales. Runs BEFORE
# `_rk_population_priors` (which skips these addressees) and before
# `_rk_plan_r2d2_prior` (an r2d2 + Horseshoe combination fails closed
# here, ahead of the R2D2 override composition that would otherwise
# misread the Horseshoe cells as Normals).
function _rk_horseshoe_priors(brmi::BRMI, design, target::Symbol,
        available::Tuple, terms::Vector{_RKTermSpec},
        r2d2::Union{Nothing,_BRMR2D2Plan})
    prefix = "RK backend"
    overrides = _brm_simple_population_effect_overrides(
        brmi, design; prefix, available_predictors=available)
    isnothing(overrides) && return _RKHorseshoePrior[]
    n = length(design.columns)
    length(overrides) == n || error(
        "$prefix: internal effect-prior alignment error: " *
        "$(length(overrides)) priors for $n population columns")
    hs_cells = [i for i in eachindex(overrides)
        if overrides[i] isa ExprColumn &&
            getf(overrides[i]) === Horseshoe]
    isempty(hs_cells) && return _RKHorseshoePrior[]
    isnothing(r2d2) || error(
        "$prefix: predictor `$target` combines " *
        "`effect($target, :) ~ r2d2(...)` with `~ Horseshoe(...)`; one " *
        "predictor takes one structured prior in slice 1 (drop one of them)")
    # The thin flat slice covers intercept/continuous/offset predictors
    # only (every other term kind fails thin-side); gate it here with
    # BRM-side attribution.
    for term in terms
        term.kind in (:intercept, :continuous, :offset) || error(
            "$prefix: predictor `$target` carries `~ Horseshoe(...)` " *
            "with a `$(term.kind)` term; structured Horseshoe covers " *
            "intercept/continuous/offset predictors in slice 1")
    end
    groups, order = _rk_design_addressee_groups(design, target; prefix)
    labels = Symbol[c.label for c in design.columns]
    priors = _RKHorseshoePrior[]
    for addressee in order
        idxs = groups[addressee]
        hits = [i for i in idxs if i in hs_cells]
        isempty(hits) && continue
        # (Factor addressees are unreachable: the flatness gate above
        # already fails any Horseshoe predictor containing a factor
        # term, and factor addressees derive from factor terms.)
        length(hits) == length(idxs) || error(
            "$prefix: predictor `$target` addressee `$addressee` mixes " *
            "Horseshoe columns with non-Horseshoe columns; slice 1 " *
            "needs one prior per addressee (address the source column, " *
            "not individual levels)")
        specs = map(hits) do i
            _brm_horseshoe_spec("effect($target, $(labels[i]))",
                getargs(overrides[i]), getkwargs(overrides[i]); prefix)
        end
        for (i, spec) in zip(hits, specs)
            spec.local_scale isa Real || error(
                "$prefix: `effect($target, $(labels[i])) ~ Horseshoe(...)` " *
                "`local_scale` must be a numeric constant in slice 1")
            spec.global_scale isa Real || error(
                "$prefix: `effect($target, $(labels[i])) ~ Horseshoe(...)` " *
                "`global_scale` must be a numeric constant in slice 1")
        end
        first_local = Float64(specs[1].local_scale)
        first_global = Float64(specs[1].global_scale)
        all(s -> Float64(s.local_scale) == first_local &&
                Float64(s.global_scale) == first_global, specs) || error(
            "$prefix: predictor `$target` addressee `$addressee` has " *
            "disagreeing Horseshoe scales across its columns; slice 1 " *
            "needs one shared Horseshoe per addressee (address the source " *
            "column, not individual levels)")
        push!(priors,
            _RKHorseshoePrior(target, addressee, first_local, first_global))
    end
    priors
end

# Flat whole-predictor R2D2 (SB `_sb_emit_r2d2_params!` /
# `_sb_emit_r2d2_popefs!` mirror): the shared `_BRMR2D2Plan` (R2 prior,
# share indices, alpha, tau literal-or-nothing) becomes an
# `_RKR2D2Prior` plus its Beta-R2 / Dirichlet-phi / optional
# half-Normal-tau sampled parameters (SB spellings
# `r2d2_<target>_R2/_phi/_tau_bsv`, minted). Only the share COUNT
# crosses the boundary (as the Dirichlet length) — the thin layer
# recomputes the share composition at bind from the same
# override/no-override structure, so the numbering itself need not
# agree. Returns `(; prior, scalars, phi)`.
function _rk_plan_r2d2_prior(brmi::BRMI, design, r2plan::_BRMR2D2Plan,
        target::Symbol, available::Tuple, terms::Vector{_RKTermSpec},
        taken::Set{Symbol}, columns::Dict{Symbol,AbstractVector})
    prefix = "RK backend"
    for term in terms
        # The mo contrast is parameter-derived, so no data variance
        # exists (the thin layer rejects it too); gp latents have no
        # R2D2 term rule thin-layer-side. mo1/dar/spline/hsgp summands
        # are skipped by the bind-time share composition, like offsets
        # and ranef gathers.
        term.kind === :monotonic && error(
            "$prefix: predictor `$target` combines `mo` with `r2d2`; " *
            "the mo contrast is parameter-derived, so no data variance " *
            "exists to decompose (drop one of them)")
        term.kind === :gp && error(
            "$prefix: predictor `$target` combines `gp` with `r2d2`; " *
            "gp latents have no R2D2 term rule in slice 1")
    end
    n = length(design.columns)
    cell_overrides = _brm_simple_population_effect_overrides(
        brmi, design; prefix, available_predictors=available)
    stated = isnothing(cell_overrides) ? fill(false, n) :
        Bool[!isnothing(cell) for cell in cell_overrides]
    n_shares = count(!iszero, r2plan.share_indices)
    if n_shares == 0
        # SB mirror (`_sb_r2d2_overrides`): zero shares is a legitimate
        # no-op only with no non-intercept columns at all — but SB's
        # tau-only no-op has no thin-layer form (the peer rejects a
        # decomposition over nothing, and a bare tau would be an unused
        # parameter), so both shapes fail closed here.
        labels = Symbol[c.label for c in design.columns]
        if all(l -> l === :Intercept, labels)
            error("$prefix: predictor `$target` `r2d2` decomposes " *
                "nothing (no non-intercept coefficient columns); drop " *
                "the `r2d2` statement")
        end
        excluded = Symbol[labels[i] for i in eachindex(labels)
            if labels[i] !== :Intercept && stated[i]]
        error("$prefix: predictor `$target` `r2d2` has nothing to " *
            "allocate: every non-intercept population column " *
            "($(join(excluded, ", "))) carries its own explicit " *
            "`Normal` prior, and an explicitly prioried column leaves " *
            "the Dirichlet allocation (SB mirror); drop those " *
            "per-column statements or drop the `r2d2` statement")
    end
    prior = r2plan.prior
    ab = if prior isa Beta
        (Float64(prior.α), Float64(prior.β))
    elseif prior isa ExprColumn && getf(prior) === Beta &&
            length(getargs(prior)) == 2
        a = _brm_numeric_constant(getargs(prior)[1])
        b = _brm_numeric_constant(getargs(prior)[2])
        (isnothing(a) || isnothing(b)) && error(
            "$prefix: predictor `$target` R2D2 `R2` Beta shapes must be " *
            "numeric constants in slice 1")
        (Float64(a), Float64(b))
    else
        error("$prefix: predictor `$target` R2D2 `R2` prior must be " *
            "`Beta(a, b)` with numeric shapes in slice 1")
    end
    (all(isfinite, ab) && all(b -> b > 0, ab)) || error(
        "$prefix: predictor `$target` R2D2 `R2` Beta shapes must be " *
        "finite and positive")
    scalars = _RKSampledParameter[]
    r2_name = _rk_mint_generated!(taken, columns, "r2d2_$(target)_R2")
    push!(scalars,
        _RKSampledParameter(r2_name, :Beta, ab, nothing, r2_name))
    tau = if isnothing(r2plan.total_scale)
        # SB's honest default: a sampled half-standard-normal (SB
        # `std_normal(; lower=0.)`), not a fabricated constant.
        tau_name = _rk_mint_generated!(
            taken, columns, "r2d2_$(target)_tau_bsv")
        push!(scalars, _RKSampledParameter(
            tau_name, :Normal, (0.0, 1.0), :positive, tau_name))
        tau_name
    else
        Float64(r2plan.total_scale)
    end
    phi_name = _rk_mint_generated!(taken, columns, "r2d2_$(target)_phi")
    phi = _RKVectorParameter(phi_name, :simplex_dirichlet,
        (fill(r2plan.alpha, n_shares),), n_shares, phi_name)
    location, scale = _brm_materialize_normal_effect_priors(
        cell_overrides, n; prefix)
    groups, order = _rk_design_addressee_groups(design, target; prefix)
    overrides = Dict{Symbol,Tuple{Float64,Float64}}()
    for addressee in order
        idxs = groups[addressee]
        any_stated = any(stated[idxs])
        any_stated || continue
        all(stated[idxs]) || error(
            "$prefix: predictor `$target` addressee `$addressee` mixes " *
            "explicit-Normal columns with simplex columns; the R2D2 " *
            "composition is per-block (state the whole block or none " *
            "of it)")
        first_loc, first_scale = location[first(idxs)], scale[first(idxs)]
        all(i -> location[i] == first_loc && scale[i] == first_scale,
            idxs) || error(
            "$prefix: predictor `$target` addressee `$addressee` has " *
            "disagreeing population priors across its columns; slice 1 " *
            "needs one shared Normal per addressee (address the source " *
            "column, not individual levels)")
        overrides[addressee] = (first_loc, first_scale)
    end
    for term in terms
        # An unstated factor joins the simplex under a FULL-cover
        # LevelMap thin-layer-side — for a subset-coded (treatment /
        # reference) block that changes the coding, so subset blocks
        # must ride share 0 with an explicit Normal (which keeps the
        # subset, like the PopulationPrior path).
        term.kind === :factor || continue
        term.options.coding === :subset || continue
        haskey(overrides, term.addressee) && continue
        error("$prefix: predictor `$target` factor `$(term.addressee)` " *
            "is subset-coded but carries no explicit Normal prior; an " *
            "unstated factor joins the R2D2 simplex under full cover, " *
            "which changes the coding — state " *
            "`effect($target, $(term.addressee)) ~ Normal(0, s)` " *
            "(share-0 override) or use full-rank coding")
    end
    (; prior=_RKR2D2Prior(target, r2_name, phi_name, tau, overrides),
        scalars, phi)
end

# RK single-label rule (see the section header): a repeat (head, source)
# pair within ONE predictor fails closed, ahead of population priors (which
# would otherwise misattribute the second `mo(c)` as an index collision).
# Cross-predictor repeats plan — SB suffixes contrasts per occurrence and the
# increments mint below is model-wide, so each predictor owns its own vector.
function _rk_gate_monotonic_unique!(prefix::String, where::String,
        terms::AbstractVector)
    seen = Set{Tuple{Symbol,Symbol}}()
    for term in terms
        (term.kind === :monotonic ||
            term.kind === :monotonic_summand) || continue
        key = (term.kind, term.options.source)
        key in seen || (push!(seen, key); continue)
        head = term.kind === :monotonic ? "mo" : "mo1"
        error("$prefix: $where two `$head($(term.options.source))` terms; " *
            "RK addresses one `$head` increments vector per predictor, so " *
            "the second is out of slice 1 (drop it)")
    end
    nothing
end

# ---- differenced-AR terms (dar; mirrors `_sb_dar1`) ----
#
# A `:dar` term carries `(; beta, sigma, source, beta_param, sigma_param)`:
# the persistence/scale sampled-scalar names (SB's `<dar_mu_t>_beta/_sigma`),
# the raw time axis, and their `_RKSampledParameter`s (options-carried like
# gp hypers; the AST preamble emits them). Beta-free direct summand
# (self-addressed: no population prior). The thin layer owns the scan-tier
# trajectory + z[T-1] innovations (sized from n_obs); BRM ships the bound
# time axis (SB binds it too — values never enter the path, only its
# length, which the Phase-6 gate checks) + the declaration. T<2 (n_obs=1:
# SB's path is identically 0, the thin layer fails z[0] at layout) emits a
# zeros offset instead — the mo1-K=1 twin. One dar summand per predictor
# (thin-layer v1 state scoping) and at least one estimated coefficient
# (no coefficient-free dar predictor in v1) fail closed in
# `_rk_gate_dar_admitted!`.
function _rk_dar_beta_prior(prior, target::Symbol, source::Symbol)
    prefix = "RK backend"
    where = "predictor `$target` `dar($source)` persistence"
    prior isa ExprColumn || error(
        "$prefix: $where prior is not a distribution call")
    f = getf(prior)
    f isa Type || error(
        "$prefix: $where prior is out of slice 1 (admitted: Normal)")
    nameof(f) === :Normal || error(
        "$prefix: $where prior `$(nameof(f))` is out of slice 1 (the " *
        "thin-layer persistence is truncated-Normal on [0, 1]; admitted: " *
        "Normal)")
    isempty(getkwargs(prior)) || error(
        "$prefix: $where prior cannot have keywords in slice 1")
    args = getargs(prior)
    length(args) == 2 || error(
        "$prefix: $where prior `Normal` needs 2 arguments, got " *
        "$(length(args))")
    resolved = map(args) do arg
        arg isa Number && isfinite(Float64(arg)) || error(
            "$prefix: $where prior hyperparameters must be finite literals")
        Float64(arg)
    end
    (:Normal, (resolved[1], resolved[2]), :interval)
end

function _rk_dar_sigma_prior(prior, target::Symbol, source::Symbol)
    prefix = "RK backend"
    where = "predictor `$target` `dar($source)` scale"
    prior isa ExprColumn || error(
        "$prefix: $where prior is not a distribution call")
    f = getf(prior)
    f isa Type || error(
        "$prefix: $where prior is out of slice 1 (admitted: Normal)")
    nameof(f) === :Normal || error(
        "$prefix: $where prior `$(nameof(f))` is out of slice 1 (the " *
        "thin-layer scale is HalfNormal; admitted: Normal)")
    isempty(getkwargs(prior)) || error(
        "$prefix: $where prior cannot have keywords in slice 1")
    args = getargs(prior)
    length(args) == 2 || error(
        "$prefix: $where prior `Normal` needs 2 arguments, got " *
        "$(length(args))")
    location, scale = args
    location isa Number && location == 0 || error(
        "$prefix: $where `Normal` prior must have location 0 in " *
        "slice 1 (a positive scale takes a half-Normal)")
    scale isa Number && isfinite(Float64(scale)) || error(
        "$prefix: $where prior hyperparameters must be finite literals")
    (:Normal, (0.0, Float64(scale)), :positive)
end

function _rk_plan_dar_term!(prepared::_BRMPreparedTerm{typeof(dar)},
        target::Symbol, columns::Dict{Symbol,AbstractVector},
        taken::Set{Symbol})
    source = prepared.source
    # SB binds the axis under its own name; the values never enter the path
    # (only its length, Phase-6-gated), so a second use of the raw column
    # reads Float64-converted time in both backends.
    columns[source] = prepared.state.time
    if length(prepared.state.time) < 2
        # Single observation: SB's path is identically 0 (no innovations),
        # while the thin layer fails z[0] at layout — so emit the zeros
        # offset twin (mo1-K=1 shape) and never touch the surface.
        zero = _rk_mint_generated!(taken, columns, "dar_$(source)_zero")
        columns[zero] = zeros(length(prepared.state.time))
        return _RKTermSpec(:offset, [zero], (;), zero, Symbol(:offset_, zero))
    end
    beta_family, beta_args, beta_support = _rk_dar_beta_prior(
        prepared.state.ar_prior, target, source)
    sigma_family, sigma_args, sigma_support = _rk_dar_sigma_prior(
        prepared.state.sd_prior, target, source)
    stem = "dar_$(target)_$(source)"
    beta = _rk_mint_generated!(taken, columns, stem * "_beta")
    sigma = _rk_mint_generated!(taken, columns, stem * "_sigma")
    label = Symbol(stem)
    _RKTermSpec(:dar, Symbol[],
        (; beta, sigma, source,
         beta_param=_RKSampledParameter(
             beta, beta_family, beta_args, beta_support, beta),
         sigma_param=_RKSampledParameter(
             sigma, sigma_family, sigma_args, sigma_support, sigma)),
        label, label)
end

# Thin-layer v1 admission for dar summands (see the section header): at most
# one per predictor (state scoping), and at least one estimated coefficient
# (the surface fails coefficient-free dar predictors closed — latent-only
# shapes stay out until the peer admits them, spline-only precedent aside).
function _rk_gate_dar_admitted!(prefix::String, target::Symbol,
        terms::Vector{_RKTermSpec})
    dar_count = count(t -> t.kind === :dar, terms)
    dar_count == 0 && return nothing
    dar_count == 1 || error(
        "$prefix: predictor `$target` carries $dar_count `dar()` terms; " *
        "the thin layer splices one dar summand per predictor in v1 " *
        "(multi-trajectory predictors are sequenced)")
    any(t -> t.kind === :intercept || t.kind === :continuous ||
        t.kind === :factor || t.kind === :monotonic, terms) || error(
        "$prefix: predictor `$target` `dar()` has no estimated " *
        "coefficients — add an intercept or coefficient (the thin layer " *
        "admits no coefficient-free dar predictor in v1)")
    nothing
end

# One `:simplex_dirichlet` vector parameter per monotonic term (SB: one
# increment simplex per contrast).
function _rk_plan_monotonic_vectors!(predictor_specs::AbstractVector)
    specs = _RKVectorParameter[]
    for spec in predictor_specs, term in spec.terms
        (term.kind === :monotonic ||
            term.kind === :monotonic_summand) || continue
        push!(specs, _RKVectorParameter(term.options.increments,
            :simplex_dirichlet, (term.options.alpha,),
            length(term.options.alpha), term.options.increments))
    end
    specs
end

# ---- measurement-error terms (me; mirrors `_sb_me`) ----
#
# An `:me` term carries `(; latent, loc, scale, sd)`: the plate-vector
# latent name (SB's un-namespaced `me_<x>` — the beta addressee, the
# `popcoefnames` label, and the thin-layer `PlateParameter` alike), the
# shared-scalar Normal args (default `(0, 1)`, overridable via
# `latent(<lp|:>, me(x)) ~ Normal(...)`), and the scalar observation
# error `sd` (shared prep gates finite `sd > 0`; SB never takes vector
# error sizes either). `columns` holds the observed source column (bound
# as data, so Phase 6 checks its length/missingness/finiteness).
#
# The thin layer owns the plate (`x[i]`-ranged `@plate` with the shared
# args) and lowers `coef .* latent` to a `ContinuousTerm` over the
# plate; the observation likelihood rides a synthetic gaussian-identity
# plan response (`_rk_plan_me_observations!`, emitted bare like the
# main responses). SB shares ONE
# latent per model across collectors, so a second `me(x)` term fails
# closed here (dedup is sequenced with the hierarchical row) — never a
# serialized second exchangeable latent double-counting the evidence.
function _rk_plan_me_term!(prepared::_BRMPreparedTerm{typeof(me)},
        target::Symbol, columns::Dict{Symbol,AbstractVector},
        taken::Set{Symbol}, me_sources::Set{Symbol})
    prefix = "RK backend"
    source = prepared.source
    source in me_sources && error(
        "$prefix: predictor `$target` carries a second `me($source)` " *
        "term; SB shares one latent true covariate per model, so the " *
        "second is out of slice 1 (use one `me($source)` term)")
    latent = Symbol(:me_, source)
    (latent in taken || haskey(columns, latent)) && error(
        "$prefix: predictor `$target` `me` latent name `$latent` is " *
        "already taken — rename the colliding parameter, data column, " *
        "or term")
    push!(me_sources, source)
    push!(taken, latent)
    columns[source] = prepared.state.x_obs
    loc, scale = _rk_me_latent_args(
        prepared.state.latent_prior, target, source)
    _RKTermSpec(:me, [source], (; latent, loc, scale, sd=prepared.state.sd_x),
        latent, latent)
end

# The `me` latent's Normal args: shared prep resolves the
# `latent(<lp|:>, me(x))` override (defaulting to `Normal(0, 1)`) into
# `latent_prior`; slice 1 materializes Normal-only (SB's arbitrary-prior
# merge is sequenced), with numeric-constant args like every other
# slice-1 Normal.
function _rk_me_latent_args(prior, target::Symbol, source::Symbol)
    prefix = "RK backend"
    prior isa ExprColumn && getf(prior) === Normal || error(
        "$prefix: predictor `$target` `me($source)` latent prior must " *
        "be `Normal(location, scale)` in slice 1")
    isempty(getkwargs(prior)) || error(
        "$prefix: predictor `$target` `me($source)` latent `Normal` " *
        "prior cannot have keywords in slice 1")
    raw_location, raw_scale = _brm_normal_effect_args(prior; prefix)
    location = _brm_numeric_constant(raw_location)
    scale = _brm_numeric_constant(raw_scale)
    isnothing(location) && error(
        "$prefix: predictor `$target` `me($source)` latent Normal " *
        "location must be a numeric constant")
    isnothing(scale) && error(
        "$prefix: predictor `$target` `me($source)` latent Normal " *
        "scale must be a numeric constant")
    isfinite(location) || error(
        "$prefix: predictor `$target` `me($source)` latent Normal " *
        "location must be finite")
    isfinite(scale) && scale > 0 || error(
        "$prefix: predictor `$target` `me($source)` latent Normal " *
        "scale must be finite and positive")
    location, scale
end

# Explicit `effect(...)` addresses on the `me` latent column are
# sequenced: the shared seam resolves design columns only, so it would
# misreport the (SB-valid) `me_<x>` label as "not a population
# coefficient". Reject it here with RK attribution instead. Runs before
# predictor geometry (which invokes the shared seam); labels derive
# syntactically, leniently — malformed terms stay shared prep's error.
function _rk_gate_me_effect_priors!(brmi::BRMI, target::Symbol,
        me_raw::AbstractVector)
    prefix = "RK backend"
    isempty(me_raw) && return nothing
    addresses = Set{Symbol}()
    for t in me_raw
        t isa ExprColumn || continue
        args = getargs(t)
        length(args) == 2 || continue
        axis = first(args)
        axis isa NamedColumn || continue
        push!(addresses, Symbol(:me_, name(axis)))
    end
    isempty(addresses) && return nothing
    for spec in effect_priors(brmi)
        spec.coefficient in addresses || continue
        (spec.predictor === _EFFECT_COLON || spec.predictor === target) ||
            continue
        error("$prefix: predictor `$target` explicit priors on the `me` " *
            "latent column `$(spec.coefficient)` are out of slice 1 " *
            "(address the predictor with `effect($target, :)`, or take " *
            "the Normal(0, 1) default; per-column `effect(...)` " *
            "addresses on latent columns are sequenced)")
    end
    nothing
end

# The `me` beta's prior: the latent column is not a design column, so
# `:`-wide statements claim this cell through the SAME claim engine
# (tie semantics identical to the design cells), defaulting to
# Normal(0, 1) (SB's `popefs` default). Runs after the shared seam
# (which owns spec validation); explicit me addresses never reach here
# (gated above).
function _rk_me_beta_prior(brmi::BRMI, target::Symbol, addressee::Symbol)
    prefix = "RK backend"
    won = Ref{Any}(nothing)
    for spec in effect_priors(brmi)
        spec.coefficient === _EFFECT_COLON || continue
        spec.predictor === _EFFECT_COLON || spec.predictor === target ||
            continue
        _brm_claim_effect_prior!(() -> won[], v -> (won[] = v), spec,
            "`$target`'s measurement-error `$addressee` column"; prefix)
    end
    expression = isnothing(won[]) ? nothing : won[].expression
    if !isnothing(expression)
        expression isa ExprColumn && getf(expression) === Normal || error(
            "$prefix: predictor `$target` population-effect priors must " *
            "be `Normal(location, scale)` in slice 1")
        isempty(getkwargs(expression)) || error(
            "$prefix: predictor `$target` population-effect `Normal` " *
            "prior cannot have keywords in slice 1")
    end
    location, scale = _brm_materialize_normal_effect_priors(
        Any[expression], 1; prefix)
    _RKPopulationPrior(target, addressee, location[1], scale[1])
end

# One synthetic gaussian-identity observation per `:me` term: the SB
# `x_obs ~ Normal(x_true, sd)` likelihood (scalar constant `sd`) the
# thin layer lowers through its plate-mean path. Runs after the formula
# responses (Phase 5) so the observation order is deterministic;
# `_rk_gate_crossed_columns!` already bounds the source column.
function _rk_plan_me_observations!(response_specs::Vector{_RKLikelihoodSpec},
        predictor_specs::AbstractVector)
    for spec in predictor_specs, term in spec.terms
        term.kind === :me || continue
        source = only(term.columns)
        push!(response_specs, _RKLikelihoodSpec(:gaussian, :identity,
            source, term.options.latent, term.options.sd, nothing,
            nothing, _RKResponseEvidence(:none, nothing, nothing),
            source, nothing, nothing, nothing, Symbol[], Symbol[],
            nothing, nothing, Symbol[], nothing, Symbol[], nothing,
            _RKMixtureComponent[], nothing, nothing))
    end
    nothing
end

function _rk_plan_predictor(brmi::BRMI, context, target::Symbol,
        available::Tuple, columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol},
        ranef_buckets::Dict{Tuple{Symbol,Symbol,Union{Nothing,Symbol}},
            Union{_RKRanefBucket,Nothing}},
        me_sources::Set{Symbol})
    prefix = "RK backend"
    op = linear_predictor_op(brmi, target)
    _, rhs = getargs(op, 2)
    link = _rk_predictor_link(brmi, target)
    raw_terms = _brm_additive_terms(rhs)
    structured = filter(t -> _brm_prepares_term(t), raw_terms)
    spline_raw = filter(t -> t isa ExprColumn &&
        (getf(t) === s || getf(t) === t2), structured)
    gp_raw = filter(t -> t isa ExprColumn && getf(t) === gp, structured)
    hsgp_raw = filter(t -> t isa ExprColumn && getf(t) === hsgp, structured)
    mo_raw = filter(t -> t isa ExprColumn &&
        (getf(t) === mo || getf(t) === mo1), structured)
    dar_raw = filter(t -> t isa ExprColumn && getf(t) === dar, structured)
    ar_raw = filter(t -> t isa ExprColumn && getf(t) === ar, structured)
    me_raw = filter(t -> t isa ExprColumn && getf(t) === me, structured)
    other_structured = filter(
        t -> !(t in spline_raw) && !(t in gp_raw) && !(t in hsgp_raw) &&
            !(t in mo_raw) && !(t in dar_raw) && !(t in ar_raw) &&
            !(t in me_raw),
        structured)
    isempty(other_structured) || error(
        "$prefix: predictor `$target` structured term(s) " *
        "$(join(unique!(string.(getf.(filter(t -> t isa ExprColumn, other_structured)))), ", ")) " *
        "are out of slice 1 (population GLMs only)")
    _rk_gate_spline_term_priors!(brmi, target, spline_raw)
    _rk_gate_hsgp_term_priors!(brmi, target, hsgp_raw)
    _rk_gate_ar_effect_priors!(brmi, target, ar_raw)
    _rk_gate_me_effect_priors!(brmi, target, me_raw)
    grouped = filter(t -> _brm_is_grouped_term(t), raw_terms)
    ordinary = Tuple(t for t in raw_terms if !(t in structured) && !(t in grouped))
    isempty(ordinary) && isempty(spline_raw) && isempty(gp_raw) &&
        isempty(hsgp_raw) && isempty(mo_raw) && isempty(dar_raw) &&
        isempty(ar_raw) && isempty(me_raw) && error(
        "$prefix: predictor `$target` has no terms")
    has_intercept = any(t -> t isa Integer && t == 1, ordinary)
    # Classify before building geometry: fail fast on unknown terms with RK
    # attribution, before shared machinery can throw undecorated errors.
    # One term can lower to several specs (multi-column interactions).
    terms = _RKTermSpec[]
    spines = Dict{Symbol,Any}()
    for term in ordinary
        append!(terms, _rk_term_specs(term, target, context.data,
            columns, derived, taken, has_intercept, spines))
    end
    for term in grouped
        decl = _brm_group_declaration(target, term)
        decl.descriptor isa NamedColumn || error(
            "$prefix: internal: unexpected group descriptor " *
            "`$(typeof(decl.descriptor))`")
        gname = name(decl.descriptor)
        key = (target, gname, decl.id)
        haskey(ranef_buckets, key) || error(
            "$prefix: internal: no draws-regime bucket for predictor " *
            "`$target` term `$term`")
        bucket = ranef_buckets[key]
        # Degenerate plain block: no gather (mirrors SB's no-op).
        bucket === nothing && continue
        gather = _rk_ranef_gather_label(target, bucket.id, bucket.group)
        push!(terms, _RKTermSpec(:ranef_gather, [bucket.group],
            (bucket_id=bucket.id, bucket_group=bucket.group), gather, gather))
    end
    _rk_gate_cover_identified!(
        terms, ordinary, target, context.data, has_intercept)
    _rk_gate_cross_identified!(
        terms, spines, derived, context.data, target, has_intercept)
    any(t -> t.kind !== :offset, terms) || !isempty(spline_raw) ||
        !isempty(gp_raw) || !isempty(hsgp_raw) || !isempty(mo_raw) ||
        !isempty(dar_raw) || !isempty(ar_raw) || !isempty(me_raw) ||
        return _rk_plan_offset_only_predictor(
        brmi, context, target, ordinary, available, link, terms, derived)
    geometry = _brm_prepare_predictor_geometry(
        brmi, context, target; available_predictors=available)
    for prepared in geometry.terms
        if prepared.callable === gp
            push!(terms, _rk_plan_gp_term!(
                prepared, target, context.data, columns, taken))
        elseif prepared.callable === s || prepared.callable === t2
            push!(terms, _rk_plan_spline_term!(
                prepared, target, context.data, columns, taken))
        elseif prepared.callable === hsgp
            push!(terms, _rk_plan_hsgp_term!(
                prepared, target, context.data, columns, taken))
        elseif prepared.callable === mo
            spec = _rk_plan_mo_term!(
                prepared, target, columns, taken)
            spec === nothing || push!(terms, spec)
        elseif prepared.callable === mo1
            push!(terms, _rk_plan_mo1_term!(
                prepared, target, columns, taken))
        elseif prepared.callable === dar
            push!(terms, _rk_plan_dar_term!(
                prepared, target, columns, taken))
        elseif prepared.callable === ar
            push!(terms, _rk_plan_ar_term!(
                prepared, target, context.data, columns, taken))
        elseif prepared.callable === me
            push!(terms, _rk_plan_me_term!(
                prepared, target, columns, taken, me_sources))
        else
            error("$prefix: internal: unexpected structured term " *
                "survived pre-check in `$target`")
        end
    end
    if isempty(terms)
        # Every term degenerated (K=1 monotonic under `0 +`; SB emits
        # scalar `0.0`): a zeros offset is the vector-shaped twin. Only
        # K=1 `mo` drops terms, so a monotonic prepared term must exist.
        first_mo = nothing
        for prepared in geometry.terms
            if prepared.callable === mo || prepared.callable === mo1
                first_mo = prepared
                break
            end
        end
        isnothing(first_mo) && error(
            "$prefix: internal: predictor `$target` planned no terms")
        zero = _rk_mint_generated!(
            taken, columns, "mo_$(first_mo.source)_zero")
        columns[zero] = zeros(length(first_mo.state.idx))
        push!(terms, _RKTermSpec(:offset, [zero], (;), zero,
            Symbol(:offset_, zero)))
    end
    _rk_gate_monotonic_unique!(prefix, "predictor `$target` carries", terms)
    _rk_gate_dar_admitted!(prefix, target, terms)
    _rk_gate_ar_sibling!(terms, target)
    design = geometry.component.design
    factor_addressees = Set{Symbol}(t.addressee
        for t in terms if t.kind === :factor)
    r2plan = geometry.r2d2
    hs_priors = _rk_horseshoe_priors(brmi, design, target, available,
        terms, r2plan)
    priors = _rk_population_priors(brmi, design, target, available,
        factor_addressees, terms, derived, r2plan,
        Set{Symbol}(p.addressee for p in hs_priors))
    r2d2 = isnothing(r2plan) ? nothing :
        _rk_plan_r2d2_prior(brmi, design, r2plan, target, available,
            terms, taken, columns)
    _RKPredictorSpec(target, link, terms, target), priors, r2d2, hs_priors
end

function _rk_resolve_use_ref(name::Symbol, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, parameters::Set{Symbol},
        assign_names::Set{Symbol}, origin::String)
    prefix = "RK backend"
    name in parameters && return (:param, name)
    haskey(consts, name) && return (:number, consts[name])
    visiting = Set{Symbol}()
    current = name
    while haskey(aliases, current)
        current in visiting && error(
            "$prefix: $origin has a cyclic assignment reference through " *
            "`$current`; break the cycle")
        push!(visiting, current)
        current = aliases[current]
        current in parameters && return (:param, current)
        haskey(consts, current) && return (:number, consts[current])
    end
    current in assign_names && return (:assignment, current)
    current in visiting || current == name || error(
        "$prefix: $origin references unknown name `$current`")
    error("$prefix: $origin references unknown name `$name`")
end

function _rk_walk_assignment_expr!(node, name::Symbol,
        data::AbstractDict, parameters::Set{Symbol}, assign_names::Set{Symbol})
    prefix = "RK backend"
    node isa Number && return nothing
    if node isa _BRMPreparedRef
        ref = node.name
        if ref in parameters
            node.axis === :scalar || error(
                "$prefix: assignment `$name` references non-scalar " *
                "parameter `$ref`; slice 1 admits scalar parameters only")
            return nothing
        end
        ref in assign_names && return nothing # scalarity by the callee's walk
        haskey(data, ref) && error(
            "$prefix: assignment `$name` references data column `$ref` " *
            "outside a reduction; assignments are scalar in slice 1 " *
            "(precompute the column)")
        error("$prefix: assignment `$name` references unknown name `$ref`")
    end
    node isa _BRMPreparedExpr || error(
        "$prefix: assignment `$name` is not a scalar expression; slice 1 " *
        "admits pure scalar calls over parameters, assignments, and " *
        "whole-column reductions")
    callable = node.callable
    callable in _RK_ASSIGNMENT_CALLABLES || error(
        "$prefix: assignment `$name` calls `$callable`; slice 1 admits " *
        "{+,-,*,/,^,log,log10,log1p,exp,expm1,sqrt,abs,sum,mean,std,var," *
        "minimum,maximum,length} only")
    isempty(node.kwargs) || error(
        "$prefix: assignment `$name` call keywords are out of slice 1")
    if callable in _RK_ASSIGNMENT_REDUCTIONS
        # Mirrors the thin layer: a reduction takes exactly one bare raw
        # column (no nesting, no scalars) or the plan fails validation.
        length(node.args) == 1 || error(
            "$prefix: assignment `$name` reduction `$callable` takes " *
            "exactly one whole column")
        arg = only(node.args)
        arg isa _BRMPreparedRef && haskey(data, arg.name) &&
            data[arg.name] isa AbstractVector || error(
            "$prefix: assignment `$name` reduction `$callable` takes " *
            "exactly one whole column")
        return nothing
    end
    for arg in node.args
        _rk_walk_assignment_expr!(arg, name, data, parameters, assign_names)
    end
    nothing
end

function _rk_rewrite_assignment_refs!(node, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, parameters::Set{Symbol},
        assign_names::Set{Symbol}, data::AbstractDict, origin::String)
    node isa Number && return node
    if node isa _BRMPreparedRef
        # Data refs pass through (the walk validates their positions);
        # params/consts first so a data collision still resolves loudly
        # at the name-hygiene gate rather than silently shadowing.
        haskey(data, node.name) && node.name ∉ parameters &&
            !haskey(consts, node.name) && !haskey(aliases, node.name) &&
            node.name ∉ assign_names && return node
        kind, value = _rk_resolve_use_ref(
            node.name, consts, aliases, parameters, assign_names, origin)
        kind === :number && return value
        return _BRMPreparedRef(value, :scalar)
    end
    node isa _BRMPreparedExpr || return node
    callable = node.callable
    args = map(node.args) do arg
        _rk_rewrite_assignment_refs!(
            arg, consts, aliases, parameters, assign_names, data, origin)
    end
    kwargs = map(node.kwargs) do value
        _rk_rewrite_assignment_refs!(
            value, consts, aliases, parameters, assign_names, data, origin)
    end
    _BRMPreparedExpr(callable, args, kwargs)
end

function _rk_plan_parameters!(prepared, data::AbstractDict,
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        parameters::Set{Symbol}, assign_names::Set{Symbol})
    prefix = "RK backend"
    specs = _RKSampledParameter[]
    for parameter in prepared.parameters
        prior = parameter.prior
        prior isa _BRMPreparedExpr || error(
            "$prefix: parameter `$(parameter.name)` prior is not a " *
            "distribution call")
        callable = prior.callable
        # Dirichlet simplexes plan as vector parameters (below), not here.
        callable === Dirichlet && continue
        # LKJ covariance-factor stems plan as stem declarations (the
        # thin layer derives `<stem>_scales`/`<stem>_L_corr` itself).
        if callable === LKJCovarianceFactor
            push!(specs, _rk_lkj_factor_parameter(parameter.name, prior,
                consts, aliases, parameters, assign_names))
            continue
        end
        family, args, support_override = if callable === truncated
            # Keyword bounds are validated inside the half-normal gate.
            _rk_half_normal_prior(prior, parameter.name)
        else
            callable isa Type || error(
                "$prefix: parameter `$(parameter.name)` prior " *
                "`$(string(callable))` is out of slice 1 (admitted: " *
                "$(join(_RK_SLICE1_PRIOR_ARITY_KEYS, ", ")))")
            name = nameof(callable)
            haskey(_RK_SLICE1_PRIOR_ARITY, name) || error(
                "$prefix: parameter `$(parameter.name)` prior `$name` is " *
                "out of slice 1 (admitted: " *
                "$(join(_RK_SLICE1_PRIOR_ARITY_KEYS, ", ")))")
            isempty(prior.kwargs) || error(
                "$prefix: parameter `$(parameter.name)` prior keywords " *
                "are out of slice 1")
            (name, prior.args, nothing)
        end
        expected = _RK_SLICE1_PRIOR_ARITY[family]
        length(args) == expected || error(
            "$prefix: parameter `$(parameter.name)` prior `$family` needs " *
            "$expected argument(s), got $(length(args))")
        resolved = Any[]
        for arg in args
            if arg isa Number
                value = Float64(arg)
                isfinite(value) || error(
                    "$prefix: parameter `$(parameter.name)` prior " *
                    "hyperparameters must be finite")
                push!(resolved, value)
            elseif arg isa _BRMPreparedRef
                kind, value = _rk_resolve_use_ref(arg.name, consts, aliases,
                    parameters, assign_names,
                    "parameter `$(parameter.name)` prior")
                kind === :number && (push!(resolved, value); continue)
                push!(resolved, value)
            else
                error("$prefix: parameter `$(parameter.name)` prior " *
                      "hyperparameters must be literals or scalar " *
                      "references, not expressions (precompute into an " *
                      "assignment)")
            end
        end
        # Mirrors the thin layer: :positive adds log(2), exact only at
        # location 0 — nonzero literals and references fail closed here
        # with BRM attribution instead of silently wrong densities.
        if support_override === :positive
            location = first(resolved)
            location isa Number && location == 0 || error(
                "$prefix: parameter `$(parameter.name)` half-normal " *
                "location must be the literal 0 (slice 1 supports " *
                "zero-location half-normals only)")
        end
        push!(specs, _RKSampledParameter(
            parameter.name, family, Tuple(resolved), support_override,
            parameter.name))
    end
    specs
end

const _RK_SLICE1_PRIOR_ARITY_KEYS =
    Tuple(sort!(collect(keys(_RK_SLICE1_PRIOR_ARITY))))

# Dirichlet simplex parameters (`s ~ Dirichlet(alpha)` /
# `s ~ Dirichlet(K, a)`): the shared-simplex source for multinomial and
# categorical responses. Concentrations are frozen hyperparameters
# (thin-layer literal-only rule): a numeric vector literal, a symmetric
# integer dimension with a numeric (or folded-const) concentration, and
# nothing else.
function _rk_dirichlet_alpha(args, name::Symbol,
        consts::Dict{Symbol,Float64})
    prefix = "RK backend"
    if length(args) == 1
        arg = only(args)
        arg isa _BRMPreparedExpr && arg.callable === Base.vect ||
            error("$prefix: parameter `$name` `Dirichlet` needs a numeric " *
                  "concentration vector literal (`Dirichlet([1.0, 2.0])`) " *
                  "or symmetric `Dirichlet(K, a)`")
        isempty(arg.kwargs) || error(
            "$prefix: parameter `$name` `Dirichlet` takes no keywords")
        isempty(arg.args) && error(
            "$prefix: parameter `$name` `Dirichlet` concentration is empty")
        alpha = Float64[]
        for value in arg.args
            value isa Number || error(
                "$prefix: parameter `$name` `Dirichlet` concentration " *
                "must be a numeric literal vector (concentrations are " *
                "frozen hyperparameters)")
            push!(alpha, Float64(value))
        end
        all(isfinite, alpha) && all(>(0), alpha) || error(
            "$prefix: parameter `$name` `Dirichlet` concentration must be " *
            "finite and strictly positive")
        return alpha
    elseif length(args) == 2
        dimension, concentration = args
        dimension isa Integer && dimension >= 1 || error(
            "$prefix: parameter `$name` symmetric `Dirichlet(K, a)` needs " *
            "a positive integer dimension")
        level = if concentration isa Number
            Float64(concentration)
        elseif concentration isa _BRMPreparedRef &&
                haskey(consts, concentration.name)
            consts[concentration.name]
        else
            error("$prefix: parameter `$name` symmetric `Dirichlet(K, a)` " *
                  "needs a numeric concentration literal")
        end
        isfinite(level) && level > 0 || error(
            "$prefix: parameter `$name` symmetric `Dirichlet(K, a)` needs " *
            "a finite strictly positive concentration")
        return fill(level, Int(dimension))
    end
    error("$prefix: parameter `$name` `Dirichlet` takes a concentration " *
          "vector `Dirichlet(alpha)` or symmetric `Dirichlet(K, a)`, got " *
          "$(length(args)) arguments")
end

# LKJ covariance-factor stem (`L ~ LKJCovarianceFactor(K; scale_prior,
# shape)`): SB's factor-first declaration. The scale prior is
# Exponential-only in this slice (SB's default; sampled-θ
# hyperparameters ride the scalar-prior shape) and the shape is a
# literal hyperparameter (the thin-layer contract); anything else fails
# closed. Width agreement with the joint response gates in Phase 6.
function _rk_lkj_factor_parameter(name::Symbol, prior::_BRMPreparedExpr,
        consts::Dict{Symbol,Float64}, aliases::Dict{Symbol,Symbol},
        parameters::Set{Symbol}, assign_names::Set{Symbol})
    prefix = "RK backend"
    length(prior.args) == 1 || error(
        "$prefix: `$name ~ LKJCovarianceFactor(K; ...)` needs exactly " *
        "one dimension argument, got $(length(prior.args))")
    K = only(prior.args)
    K isa Integer && !(K isa Bool) && K >= 1 || error(
        "$prefix: `$name ~ LKJCovarianceFactor(K; ...)` needs an " *
        "integer dimension >= 1, got $(repr(K))")
    unknown = Symbol[k for k in keys(prior.kwargs)
        if !(k in (:scale_prior, :shape))]
    isempty(unknown) || error(
        "$prefix: `$name ~ LKJCovarianceFactor(...)` accepts only " *
        "`scale_prior` and `shape`, got $unknown")
    scale_expr = get(prior.kwargs, :scale_prior, nothing)
    theta = if isnothing(scale_expr)
        1.0
    else
        scale_expr isa _BRMPreparedExpr &&
            scale_expr.callable === Exponential &&
            isempty(scale_expr.kwargs) &&
            length(scale_expr.args) == 1 || error(
            "$prefix: parameter `$name` joint-factor scale prior is " *
            "`Exponential(θ)` in this slice (SB's default)")
        raw = only(scale_expr.args)
        if raw isa Number
            value = Float64(raw)
            isfinite(value) && value > 0 || error(
                "$prefix: parameter `$name` joint-factor `Exponential` " *
                "scale must be finite and positive")
            value
        elseif raw isa _BRMPreparedRef
            _, resolved = _rk_resolve_use_ref(raw.name, consts, aliases,
                parameters, assign_names,
                "parameter `$name` joint-factor scale")
            resolved isa Number &&
                (!isfinite(resolved) || resolved <= 0) && error(
                "$prefix: parameter `$name` joint-factor `Exponential` " *
                "scale must be finite and positive")
            resolved
        else
            error("$prefix: parameter `$name` joint-factor `Exponential` " *
                  "scale must be a literal or a scalar reference, not an " *
                  "expression (precompute into an assignment)")
        end
    end
    shape_raw = get(prior.kwargs, :shape, 1.0)
    shape_raw isa Number && !(shape_raw isa Bool) || error(
        "$prefix: parameter `$name` LKJ `shape` must be a finite " *
        "positive literal (a hyperparameter)")
    shape = Float64(shape_raw)
    isfinite(shape) && shape > 0 || error(
        "$prefix: parameter `$name` LKJ `shape` must be finite and " *
        "strictly positive, got $(repr(shape_raw))")
    _RKSampledParameter(name, :LKJCovarianceFactor, (Int(K), theta, shape),
        nothing, name)
end

function _rk_plan_vector_parameters!(prepared,
        consts::Dict{Symbol,Float64})
    prefix = "RK backend"
    specs = _RKVectorParameter[]
    for parameter in prepared.parameters
        prior = parameter.prior
        prior isa _BRMPreparedExpr || continue
        prior.callable === Dirichlet || continue
        isempty(prior.kwargs) || error(
            "$prefix: parameter `$(parameter.name)` `Dirichlet` takes no " *
            "keywords")
        alpha = _rk_dirichlet_alpha(prior.args, parameter.name, consts)
        push!(specs, _RKVectorParameter(parameter.name, :simplex_dirichlet,
            (alpha,), length(alpha), parameter.name))
    end
    specs
end

function _rk_half_normal_prior(prior::_BRMPreparedExpr, name::Symbol)
    prefix = "RK backend"
    spelled = "`truncated(Normal(location, scale), 0[, Inf])` or the " *
              "keyword form with `lower=0`"
    args, kwargs = prior.args, prior.kwargs
    lower, upper = if length(args) == 3 && isempty(kwargs)
        args[2], args[3]
    elseif length(args) == 1 &&
            all(k -> k === :lower || k === :upper, keys(kwargs))
        get(kwargs, :lower, nothing), get(kwargs, :upper, nothing)
    else
        error("$prefix: parameter `$name` truncated prior must be " *
              "$spelled for a half-Normal; other truncated priors are " *
              "out of slice 1")
    end
    inner = args[1]
    inner isa _BRMPreparedExpr && inner.callable === Normal || error(
        "$prefix: parameter `$name` truncated prior must wrap `Normal` " *
        "for a half-Normal; other truncated priors are out of slice 1")
    lower isa Number && lower == 0 || error(
        "$prefix: parameter `$name` truncated prior must have lower " *
        "bound 0 for a half-Normal")
    # `Inf` arrives as a name (Julia global), not a literal — same as
    # evidence bounds.
    upper_is_inf = upper isa Number && upper == Inf ||
        upper isa _BRMPreparedRef && upper.name === :Inf
    (isnothing(upper) || upper_is_inf) || error(
        "$prefix: parameter `$name` truncated prior must have upper " *
        "bound Inf (or omit it) for a half-Normal")
    isempty(inner.kwargs) || error(
        "$prefix: parameter `$name` prior keywords are out of slice 1")
    (:Normal, inner.args, :positive)
end

function _rk_fold_assignment_consts!(kept, data::AbstractDict,
        parameters::Set{Symbol})
    prefix = "RK backend"
    consts = Dict{Symbol,Float64}()
    aliases = Dict{Symbol,Symbol}()
    exprs = Dict{Symbol,Any}()
    for assignment in kept
        expression = assignment.expression
        name = assignment.name
        if expression isa Number
            consts[name] = Float64(expression)
        elseif expression isa _BRMPreparedRef
            target = expression.name
            target in parameters && (aliases[name] = target; continue)
            haskey(data, target) && error(
                "$prefix: assignment `$name` aliases data column " *
                "`$target`; reference the column directly")
            aliases[name] = target
        elseif expression isa _BRMPreparedExpr
            exprs[name] = expression
        else
            error("$prefix: assignment `$name` is not a scalar " *
                  "expression; slice 1 admits pure scalar calls over " *
                  "parameters, assignments, and whole-column reductions")
        end
    end
    consts, aliases, exprs
end

function _rk_plan_assignments!(kept, exprs::Dict{Symbol,Any},
        data::AbstractDict, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, parameters::Set{Symbol},
        assign_names::Set{Symbol})
    specs = _RKAssignmentSpec[]
    for assignment in kept
        name = assignment.name
        haskey(exprs, name) || continue
        rewritten = _rk_rewrite_assignment_refs!(exprs[name], consts,
            aliases, parameters, assign_names, data, "assignment `$name`")
        _rk_walk_assignment_expr!(rewritten, name, data,
            parameters, assign_names)
        push!(specs, _RKAssignmentSpec(name, rewritten, name))
    end
    specs
end

function _rk_gate_acyclic!(parameters::AbstractVector,
        assignments::AbstractVector)
    prefix = "RK backend"
    deps = Dict{Symbol,Vector{Symbol}}()
    for parameter in parameters
        deps[parameter.name] =
            Symbol[arg for arg in parameter.args if arg isa Symbol]
    end
    for assignment in assignments
        deps[assignment.name] = Symbol[
            ref for ref in _brm_prepared_references(assignment.expression)]
    end
    color = Dict{Symbol,Symbol}()
    function visit(node, stack)
        color[node] = :gray
        for dep in get(deps, node, Symbol[])
            haskey(deps, dep) || continue # data and literals are leaves
            get(color, dep, :white) === :gray && error(
                "$prefix: cyclic reference through `$(join(push!(copy(stack), dep), " -> "))`; " *
                "break the cycle")
            get(color, dep, :white) === :white && visit(dep, push!(copy(stack), dep))
        end
        color[node] = :black
        nothing
    end
    for node in keys(deps)
        get(color, node, :white) === :white && visit(node, Symbol[node])
    end
    nothing
end

function _rk_gate_response_values!(family::Symbol, values::AbstractVector,
        response::Symbol)
    prefix = "RK backend"
    any(ismissing, values) && error(
        "$prefix: response `$response` has missing values; slice 1 has no " *
        "missingness machinery (modelled `mi` fails closed)")
    if family === :gaussian || family === :student_t
        eltype(values) <: Real || error(
            "$prefix: response `$response` must be real-valued")
        all(isfinite, values) || error(
            "$prefix: response `$response` must be finite")
    elseif family === :bernoulli_logit
        # Mirrors the thin layer: Bool or 0/1 integers (float 0.0/1.0 fails
        # validation there, so it fails here with BRM-side attribution).
        (eltype(values) <: Integer &&
         all(x -> x == 0 || x == 1, values)) || error(
            "$prefix: response `$response` must be Bool or 0/1 integers")
    elseif family === :bernoulli_probit || family === :bernoulli_cloglog
        (eltype(values) <: Integer &&
         all(x -> x == 0 || x == 1, values)) || error(
            "$prefix: response `$response` must be Bool or 0/1 integers")
    elseif family === :poisson_log
        (eltype(values) <: Integer && all(>=(0), values)) || error(
            "$prefix: response `$response` must hold non-negative integers")
    elseif family === :binomial_logit
        # Rowwise y <= n is checked once trials cross (below): trials may
        # be a column or a literal, and neither is visible here.
        (eltype(values) <: Integer && all(>=(0), values)) || error(
            "$prefix: response `$response` must hold non-negative integers")
    elseif family === :binomial_probit || family === :binomial_cloglog
        (eltype(values) <: Integer && all(>=(0), values)) || error(
            "$prefix: response `$response` must hold non-negative integers")
    elseif family === :nb2_log
        (eltype(values) <: Integer && all(>=(0), values)) || error(
            "$prefix: response `$response` must hold non-negative integers")
    elseif family === :gamma_log
        # Mirrors the thin layer: strictly positive (y = 0 fails
        # validation there, so it fails here with BRM-side attribution).
        (eltype(values) <: Real && all(>(0), values)) || error(
            "$prefix: response `$response` must hold strictly positive values")
    elseif family === :categorical_logit || family === :ordinal ||
            family === :categorical
        # Recoded 1..K by construction (`_rk_leveled_levels`); assert the
        # thin-layer bind rule (integer 1..K, exact coverage, no gaps).
        (eltype(values) <: Integer && eltype(values) !== Bool) || error(
            "$prefix: response `$response` must hold integers 1..K " *
            "(recoded levels)")
        K = isempty(values) ? 0 : maximum(values)
        all(x -> x >= 1, values) && sort(unique(values)) == collect(1:K) ||
            error("$prefix: response `$response` must cover every level " *
                  "1..$K exactly (recoded levels have no gaps)")
    elseif family === :ordered_logit
        # Raw-integer coding (SB compat); contiguity is planned in
        # `_rk_ordered_levels` — re-assert the eltype half of the bind rule.
        (eltype(values) <: Integer && eltype(values) !== Bool) || error(
            "$prefix: response `$response` `OrderedLogistic` expects " *
            "integer outcome data, got $(eltype(values))")
    elseif family === :beta_logit
        # Mirrors the thin layer: strictly inside (0, 1).
        (eltype(values) <: Real && all(x -> 0 < x < 1, values)) || error(
            "$prefix: response `$response` must hold values strictly " *
            "inside (0, 1)")
    elseif family === :mvnormal_cholesky
        # Mirrors the thin layer: one raw numeric column per outcome
        # (complete aligned rows pack in the peel; the finite check
        # re-asserts the bind rule here with BRM attribution).
        eltype(values) <: Real || error(
            "$prefix: response `$response` must be real-valued")
        all(isfinite, values) || error(
            "$prefix: response `$response` must be finite")
    end
    values
end

# Binomial trials validation once columns cross: integer trials, no
# missing (covered by the crossed-columns gate for columns; literals
# are validated at classification), and y <= n every row.
function _rk_gate_trials_values!(specs::AbstractVector,
        columns::Dict{Symbol,AbstractVector}, n_obs::Int)
    prefix = "RK backend"
    for spec in specs
        spec.family in (:binomial_logit, :binomial_probit,
            :binomial_cloglog) ||
            (spec.family === :mixture && spec.trials !== nothing) || continue
        trials = spec.trials
        n = if trials isa Int
            fill(trials, n_obs)
        else
            raw = columns[trials]
            eltype(raw) <: Integer || error(
                "$prefix: response `$(spec.response)` trials column " *
                "`$trials` must hold integers")
            raw
        end
        all(>=(0), n) || error(
            "$prefix: response `$(spec.response)` trials must be " *
            "non-negative every row")
        y = columns[spec.response]
        all(y .<= n) || error(
            "$prefix: response `$(spec.response)` exceeds its trials " *
            "(`$trials`) on some row; Binomial needs y <= n every row")
    end
    nothing
end

# Multinomial trials validation once columns cross: integer trials and
# row sums meeting trials every row.
function _rk_gate_multinomial_trials!(specs::AbstractVector,
        columns::Dict{Symbol,AbstractVector}, n_obs::Int)
    prefix = "RK backend"
    for spec in specs
        spec.family === :multinomial || continue
        trials = spec.trials
        n = if trials isa Int
            fill(trials, n_obs)
        else
            raw = columns[trials]
            eltype(raw) <: Integer || error(
                "$prefix: response `$(spec.response)` trials column " *
                "`$trials` must hold integers")
            raw
        end
        all(>=(0), n) || error(
            "$prefix: response `$(spec.response)` trials must be " *
            "non-negative every row")
        counts = [columns[c] for c in [spec.response; spec.count_columns...]]
        sums = [sum(row) for row in zip(counts...)]
        bad = findfirst(i -> sums[i] != n[i], 1:length(sums))
        isnothing(bad) || error(
            "$prefix: response `$(spec.response)` count rows must sum to " *
            "their trials (`$trials`) every row; first mismatch at row $bad")
    end
    nothing
end

function _rk_peel_observation(brmi::BRMI, observation)
    prefix = "RK backend"
    missing_response = _brm_missing_response_plan(observation.lhs; prefix)
    isnothing(missing_response) || error(
        "$prefix: response `$(observation.key)` uses `mi()` (modelled " *
        "missingness); slice 1 has no missingness machinery")
    observation.lhs isa JointResponseColumn &&
        return _rk_peel_joint_observation(observation)
    observation.lhs isa NamedColumn || error(
        "$prefix: response `$(observation.key)` is not a plain response " *
        "column (slice 1 admits plain response columns and `[y1, ...]` " *
        "joint responses only)")
    parent(observation.lhs) isa DataColumn || error(
        "$prefix: response `$(observation.key)` carries a response " *
        "decorator or link; slice 1 admits plain response columns only")
    rhs = observation.rhs
    rhs isa ExprColumn || error(
        "$prefix: response `$(observation.key)` likelihood must be a " *
        "distribution call")
    raw_response = _brm_data_vec(
        observation.key, parent(parent(observation.lhs)))
    raw_response = _brm_observation_rows(
        raw_response, _brm_distribution_shape(rhs))
    weight_plan = _brm_observation_weight_plan(
        rhs, observation.key, raw_response; prefix)
    if !isnothing(weight_plan)
        weight_plan.kind in (:frequency, :power) || error(
            "$prefix: response `$(observation.key)` weights are " *
            "$(weight_plan.kind); slice 1 admits frequency/power " *
            "objective weights only")
        rhs = weight_plan.distribution
    end
    modifier = _brm_response_modifier_plan(rhs; prefix)
    if !isnothing(modifier)
        modifier.kind in (:truncated, :censored, :interval_censored) || error(
            "$prefix: response `$(observation.key)` modifier " *
            "`$(modifier.kind)` is out of slice 1")
        rhs = modifier.base
        rhs isa ExprColumn || error(
            "$prefix: response `$(observation.key)` bounded base must be " *
            "a distribution call")
    end
    (; key=observation.key, rhs, raw_response, weight_plan, modifier,
        joint_outcomes=Symbol[])
end

# Joint correlated-outcomes peel (SB `[y1..yK] ~ MvNormalCholesky(...)`):
# the vector LHS takes the explicit joint family only (row weights and
# bounded evidence change the head, so they fail here with attribution),
# and the outcomes pack to complete aligned row vectors (missingness
# fails at packing — the joint density never drops or factorizes
# missing outcome patterns). `raw_response` is the row-vector
# collection (its length is the observation axis); Phase 5 unpacks one
# raw numeric column per outcome.
function _rk_peel_joint_observation(observation)
    prefix = "RK backend"
    outcomes = collect(Symbol, joint_response_names(observation.lhs))
    rhs = observation.rhs
    rhs isa ExprColumn || error(
        "$prefix: response `$(observation.key)` likelihood must be a " *
        "distribution call")
    head = getf(rhs)
    head === MvNormalCholesky || error(
        "$prefix: vector response $outcomes supports the explicit " *
        "joint family `MvNormalCholesky(means, factor)`; got " *
        "`$(head isa Function || head isa Type ? nameof(head) : head)`" *
        (head === weighted ?
            " (row weights on a joint density are out of slice)" :
            head === truncated || head === censored ||
                head === interval_censored ?
            " (bounded joint responses are out of slice)" : ""))
    rows = _brm_joint_response_values(observation.lhs; prefix)
    (; key=observation.key, rhs, raw_response=rows, weight_plan=nothing,
        modifier=nothing, joint_outcomes=outcomes)
end

function _rk_referenced_predictors(program, rhs, response::Symbol)
    prefix = "RK backend"
    referenced = _brm_reachable_operations(
        program, _brm_prepared_references(_brm_prepare_expr(rhs)))
    names = Symbol[node.name for node in program.operations
                   if node.role === :predictor && node.name in referenced]
    isempty(names) && error(
        "$prefix: response `$response` does not reference a linear " *
        "predictor; slice 1 lowers likelihoods of a declared linear " *
        "predictor (`mu ~ 1 + x`)")
    length(names) > 2 && error(
        "$prefix: response `$response` references several linear " *
        "predictors ($(join(names, ", "))); a response feeds at most " *
        "two (location plus a scale/shape predictor)")
    names
end

# CategoricalLogit references K-1 predictors positionally (class order
# 2..K follows argument order). Each argument must name a declared
# linear predictor; returns the names in order (possibly empty — the
# K=1-or-mismatch shape resolves against observed levels in Phase 5).
function _rk_categorical_refs(program, rhs::ExprColumn, response::Symbol)
    prefix = "RK backend"
    declared = Set{Symbol}(node.name for node in program.operations
        if node.role === :predictor)
    names = Symbol[]
    for arg in getargs(rhs)
        arg isa NamedColumn && name(arg) in declared || error(
            "$prefix: response `$response` `CategoricalLogit` argument " *
            "`$(arg isa NamedColumn ? name(arg) : arg)` is not a declared " *
            "linear predictor; write one `eta_j ~ ...` predictor per " *
            "non-reference class")
        push!(names, name(arg))
    end
    length(unique(names)) == length(names) || error(
        "$prefix: response `$response` `CategoricalLogit` repeats a " *
        "predictor ($(join(names, ", "))) — one linear predictor per " *
        "non-reference class")
    names
end

# Joint correlated-outcomes means resolve positionally (outcome order):
# one declared linear predictor per outcome. Scalar, data-backed, and
# expression means are SB-only — the thin layer takes one identity-link
# predictor per outcome, so anything else fails closed here.
function _rk_joint_refs(program, rhs::ExprColumn, response::Symbol,
        outcomes::Vector{Symbol})
    prefix = "RK backend"
    isempty(outcomes) && error(
        "$prefix: response `$response` `MvNormalCholesky` is joint-only; " *
        "write `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)` with a " *
        "vector response")
    args = getargs(rhs)
    length(args) == 2 || error(
        "$prefix: response `$response` `MvNormalCholesky(means, factor)` " *
        "needs exactly two arguments")
    means = args[1]
    means isa AbstractVector || error(
        "$prefix: response `$response` `MvNormalCholesky` means must use " *
        "vector syntax `[mu1, mu2, ...]`, got $(typeof(means))")
    K = length(outcomes)
    length(means) == K || error(
        "$prefix: response `$response` has $K outcomes but " *
        "`MvNormalCholesky` received $(length(means)) means (one mean " *
        "per outcome, in outcome order)")
    declared = Set{Symbol}(node.name for node in program.operations
        if node.role === :predictor)
    names = Symbol[]
    for m in means
        m isa NamedColumn && name(m) in declared || error(
            "$prefix: response `$response` joint mean " *
            "`$(m isa NamedColumn ? name(m) : m)` is not a " *
            "declared linear predictor; the RK slice admits one " *
            "identity-link predictor per outcome (`mu_j ~ ...`)")
        push!(names, name(m))
    end
    length(unique(names)) == length(names) || error(
        "$prefix: response `$response` joint means repeat a predictor " *
        "($(join(names, ", "))) — one linear predictor per outcome)")
    names
end

# ---- leveled responses (categorical / ordinal / multinomial) ----
const _RK_LEVELED_FAMILIES =
    Set{Symbol}([:categorical_logit, :ordered_logit, :ordinal,
        :multinomial, :categorical])

# Recoded 1..K levels for a leveled vector response (SB's
# `_brm_response_levels`: sort(unique) order, reference = level 1).
function _rk_leveled_levels(response::Symbol, raw::AbstractVector)
    prefix = "RK backend"
    raw isa AbstractVector || error(
        "$prefix: response `$response` must be an observed vector")
    prepared = _brm_response_levels(response, raw; prefix)
    K = prepared.fit.n_levels
    K >= 1 || error(
        "$prefix: response `$response` has no observed levels")
    (prepared.response, K, prepared.fit.levels)
end

# OrderedLogistic keeps SB's raw-integer coding (K = max(y)): no recode,
# since recoding would change the cutpoint count and the posterior. Gaps
# fail closed — the thin layer needs exact 1..K coverage.
function _rk_ordered_levels(response::Symbol, raw::AbstractVector)
    prefix = "RK backend"
    (eltype(raw) <: Integer && eltype(raw) !== Bool) || error(
        "$prefix: response `$response` `OrderedLogistic` expects integer " *
        "outcome data, got $(eltype(raw))")
    any(x -> x < 1, raw) && error(
        "$prefix: response `$response` `OrderedLogistic` expects positive " *
        "integer outcome data")
    K = isempty(raw) ? 0 : maximum(raw)
    K >= 1 || error(
        "$prefix: response `$response` has no observed levels")
    sort(unique(raw)) == collect(1:K) || error(
        "$prefix: response `$response` holds non-contiguous levels " *
        "($(join(sort(unique(raw)), ", "))); SB accepts gappy " *
        "`OrderedLogistic` levels, but RK needs recoded contiguous 1..K — " *
        "recode the response or drop empty levels")
    (Int.(raw), K)
end

# The shared-simplex source of a multinomial/categorical response: a
# `Dirichlet`-sampled vector parameter (data columns and other
# parameters fail closed — the thin layer takes a simplex latent only).
function _rk_simplex_source(arg, response::Symbol,
        vectors::Dict{Symbol,_RKVectorParameter})
    prefix = "RK backend"
    arg isa NamedColumn || error(
        "$prefix: response `$response` probabilities must be a " *
        "`Dirichlet`-sampled parameter (`s ~ Dirichlet(...)`)")
    parent(arg) isa DataColumn && error(
        "$prefix: response `$response` probabilities cannot be a data " *
        "column; declare a shared simplex (`s ~ Dirichlet(...)`) — fixed " *
        "probability vectors are out of slice 1")
    sname = name(arg)
    haskey(vectors, sname) || error(
        "$prefix: response `$response` probabilities `$(sname)` must be a " *
        "`Dirichlet`-sampled parameter (`$(sname) ~ Dirichlet(...)`)")
    (sname, vectors[sname].size)
end

# Ordinal extras: `discrimination` (a positive literal, a raw data
# column, or a `log()` linear predictor — the modeled scale, positive by
# construction via `exp`) and `per_threshold` (a tuple of raw numeric
# design columns, StoppingRatio only). These are thin-layer IR fields
# (`LikelihoodSpec` discrimination/threshold_columns/threshold_coefs):
# the AST spells the plain three-positional ordinal and the extension
# carries the extras plan-level (the surface takes no extras).
function _rk_ordinal_discrimination(raw, response::Symbol,
        predictor_specs::AbstractVector)
    prefix = "RK backend"
    raw === nothing && return (nothing, Symbol[])
    if raw isa Number
        value = Float64(raw)
        isfinite(value) && value > 0 || error(
            "$prefix: response `$response` ordinal discrimination must be " *
            "finite and strictly positive, got $(repr(raw))")
        return (value, Symbol[])
    end
    raw isa NamedColumn || error(
        "$prefix: response `$response` ordinal discrimination must be a " *
        "positive literal, a raw data column, or a `log()` linear " *
        "predictor, got $(repr(raw))")
    sname = name(raw)
    known = findfirst(spec -> spec.name === sname, predictor_specs)
    if known !== nothing
        # Predictor-first, mirroring the thin layer (a name that is both
        # a predictor and a data column fails closed at hygiene).
        spec = predictor_specs[known]
        spec.link === :log || error(
            "$prefix: response `$response` discrimination predictor " *
            "`$sname` must be a `log()` linear predictor (a modeled scale " *
            "is positive by construction via `exp`); got a $(spec.link)-link " *
            "predictor — write `log($sname) ~ ...`")
        bad = unique!([term.kind for term in spec.terms if
            !(term.kind in (:intercept, :continuous, :factor, :offset))])
        isempty(bad) || error(
            "$prefix: response `$response` discrimination predictor " *
            "`$sname` uses $(join(bad, ", ")) terms; a modeled scale " *
            "admits population terms only (intercept, continuous, " *
            "factor, offset)")
        return (sname, Symbol[])
    end
    parent(raw) isa DataColumn || error(
        "$prefix: response `$response` ordinal discrimination `$sname` is " *
        "neither a declared `log()` linear predictor nor a raw data " *
        "column (sampled parameters and assignments are not admitted — " *
        "the thin layer takes literals, data columns, and log-link " *
        "predictors only)")
    values = parent(parent(raw))
    values isa AbstractVector{<:Real} &&
        all(x -> isfinite(x) && x > 0, values) || error(
        "$prefix: response `$response` ordinal discrimination data " *
        "`$sname` must contain only finite positive values")
    (sname, [sname])
end

function _rk_ordinal_threshold_columns(raw, response::Symbol, n_obs::Int)
    prefix = "RK backend"
    raw isa Tuple || error(
        "$prefix: response `$response` `per_threshold` expects a tuple " *
        "of raw numeric columns, for example `per_threshold=(treat,)`")
    names = Symbol[]
    for term in raw
        term isa NamedColumn && parent(term) isa DataColumn || error(
            "$prefix: response `$response` `per_threshold` currently " *
            "accepts only raw numeric data columns; got $(typeof(term))")
        key = name(term)
        values = parent(parent(term))
        values isa AbstractVector{<:Real} || error(
            "$prefix: response `$response` threshold predictor `$key` " *
            "must be numeric, got $(typeof(values))")
        length(values) == n_obs || error(
            "$prefix: response `$response` threshold predictor `$key` " *
            "has $(length(values)) rows; outcome `$response` has $n_obs")
        all(isfinite, values) || error(
            "$prefix: response `$response` threshold predictor `$key` " *
            "contains non-finite values")
        push!(names, key)
    end
    length(unique(names)) == length(names) || error(
        "$prefix: response `$response` threshold columns repeat a column " *
        "($(join(names, ", "))); the thin layer takes distinct design " *
        "columns)")
    names
end

function _rk_ordinal_extras(rhs::ExprColumn, response::Symbol,
        structure::Symbol, K::Int, n_obs::Int,
        predictor_specs::AbstractVector)
    prefix = "RK backend"
    kwargs = getkwargs(rhs)
    for key in keys(kwargs)
        key === :discrimination || key === :per_threshold || error(
            "$prefix: response `$response` `Ordinal` takes only " *
            "`discrimination` and `per_threshold` keywords, got `$key`")
    end
    discrimination, crossed = _rk_ordinal_discrimination(
        get(kwargs, :discrimination, nothing), response, predictor_specs)
    threshold_columns = _rk_ordinal_threshold_columns(
        get(kwargs, :per_threshold, ()), response, n_obs)
    structure === :cumulative && !isempty(threshold_columns) && error(
        "$prefix: response `$response` `per_threshold` is currently " *
        "supported for `StoppingRatio()` only; unrestricted cumulative " *
        "category-specific effects can make cumulative probabilities " *
        "non-monotone")
    threshold_coefs = nothing
    coef_implicit = _RKVectorParameter[]
    if !isempty(threshold_columns)
        threshold_coefs = Symbol(response, :_threshold_beta)
        push!(coef_implicit, _RKVectorParameter(threshold_coefs,
            :vector_normal, (0.0, 1.0),
            (K - 1) * length(threshold_columns), threshold_coefs))
    end
    (discrimination, threshold_columns, threshold_coefs,
        [crossed; threshold_columns], coef_implicit)
end

# Defaults for non-leveled families (the thin-layer leaves these at
# defaults too).
function _rk_unleveled(entry, predictor::Symbol)
    (; predictor, n_levels=nothing, thresholds=nothing,
        extra_predictors=Symbol[], count_columns=Symbol[],
        ordinal_structure=nothing, discrimination=nothing,
        threshold_columns=Symbol[], threshold_coefs=nothing,
        response_values=entry.raw_response, cross_columns=Symbol[])
end

# The joint factor stem off a classified `MvNormalCholesky` RHS (the
# classify arm guarantees the two-argument shape with a named factor).
_rk_joint_stem(rhs::ExprColumn) = name(getargs(rhs)[2])

# Joint correlated-outcomes response spec: the lead outcome takes
# `response`, the tail `extra_responses`, the tail means ride the
# reused `extra_predictors`, and the LKJ stem links explicitly. Each
# outcome crosses as its own raw numeric column (the thin layer binds
# K columns, row-aligned by the uniform-`n_obs` rule).
function _rk_plan_joint_response!(entry, link::Symbol, predictor::Symbol,
        extra::Vector{Symbol}, evidence::_RKResponseEvidence,
        columns::Dict{Symbol,AbstractVector})
    prefix = "RK backend"
    outcomes = entry.joint_outcomes
    K = length(outcomes)
    K >= 2 || error(
        "$prefix: internal: joint response `$(entry.key)` has fewer " *
        "than two outcomes")
    length(extra) == K - 1 || error(
        "$prefix: internal: joint response `$(entry.key)` has $K " *
        "outcomes but $(1 + length(extra)) mean predictors")
    stem = _rk_joint_stem(entry.rhs)
    for (i, outcome) in enumerate(outcomes)
        column = Float64[row[i] for row in entry.raw_response]
        columns[outcome] = _rk_gate_response_values!(
            :mvnormal_cholesky, column, outcome)
    end
    _RKLikelihoodSpec(:mvnormal_cholesky, link, first(outcomes), predictor,
        nothing, nothing, nothing, evidence, entry.key, nothing, nothing,
        nothing, extra, Symbol[], nothing, nothing, Symbol[], nothing,
        outcomes[2:end], stem, _RKMixtureComponent[], nothing, nothing)
end

# Joint responses link their LKJ factor stem explicitly (SB's
# factor-first contract): the stem must be a planned
# `LKJCovarianceFactor` declaration of matching width, feeding exactly
# one joint response. The thin layer derives `<stem>_scales` /
# `<stem>_L_corr` and enforces the same linkage — this gate gives BRM
# attribution instead of a thin-layer error.
function _rk_gate_joint_factors!(response_specs::AbstractVector,
        parameters::AbstractVector, assignment_names::Set{Symbol})
    prefix = "RK backend"
    stems = Dict{Symbol,Int}()
    for spec in parameters
        spec.family === :LKJCovarianceFactor || continue
        stems[spec.name] = spec.args[1]
        # A sampled scale hyperparameter θ must be scalar: a
        # vector-valued θ sails through the name table and fails
        # thin-side, so it fails here instead.
        theta = spec.args[2]
        theta isa Symbol || continue
        theta in assignment_names && continue
        any(p -> p.name === theta &&
            p.family !== :LKJCovarianceFactor, parameters) || error(
            "$prefix: parameter `$(spec.name)` joint-factor scale " *
            "`$theta` is not a scalar parameter or assignment " *
            "(sampled scales ride the scalar-prior shape)")
    end
    used = Set{Symbol}()
    for spec in response_specs
        spec.family === :mvnormal_cholesky || continue
        stem = spec.factor
        K = 1 + length(spec.extra_responses)
        haskey(stems, stem) || error(
            "$prefix: response `$(spec.label)` joint factor `$stem` " *
            "must name an `LKJCovarianceFactor` declaration " *
            "(`$stem ~ LKJCovarianceFactor($K; ...)` before the response)")
        stems[stem] == K || error(
            "$prefix: response `$(spec.label)` has $K ordered outcomes " *
            "but factor `$stem` has dimension $(stems[stem])")
        stem in used && error(
            "$prefix: factor `$stem` feeds two joint responses; one " *
            "factor per joint response (declare one stem per " *
            "`[..] ~ MvNormalCholesky(..)` statement)")
        push!(used, stem)
    end
    for stem in keys(stems)
        stem in used || error(
            "$prefix: parameter `$stem` is declared " *
            "`LKJCovarianceFactor`-sampled but no joint response uses " *
            "it; write `[y1, ...] ~ MvNormalCholesky([mu1, ...], $stem)`, " *
            "or drop the declaration")
    end
    nothing
end

# Leveled spec fields for one response. Appends implicit threshold
# vectors (cutpoints/thresholds/coefs) to `implicit`; multinomial count
# columns split in the caller (`response_values === nothing` marks it).
function _rk_plan_leveled!(entry, family::Symbol,
        predictor::Union{Symbol,Nothing}, extra::Vector{Symbol},
        implicit::Vector{_RKVectorParameter},
        vectors::Dict{Symbol,_RKVectorParameter}, data::AbstractDict,
        predictor_specs::AbstractVector)
    prefix = "RK backend"
    key, rhs, raw = entry.key, entry.rhs, entry.raw_response
    if family === :categorical_logit
        recoded, K, levels = _rk_leveled_levels(key, raw)
        n_nonref = 1 + length(extra)
        K == 1 + n_nonref || error(
            "$prefix: `CategoricalLogit($key)` observed $K outcome levels " *
            "but received $n_nonref non-reference predictors; expected " *
            "$(K - 1). Outcome level order is $(collect(levels)).")
        return (; predictor, n_levels=K, thresholds=nothing,
            extra_predictors=extra, count_columns=Symbol[],
            ordinal_structure=nothing, discrimination=nothing,
            threshold_columns=Symbol[], threshold_coefs=nothing,
            response_values=recoded, cross_columns=Symbol[])
    elseif family === :ordered_logit
        recoded, K = _rk_ordered_levels(key, raw)
        cut = Symbol(key, :_cutpoints)
        push!(implicit, _RKVectorParameter(cut, :ordered_normal, (0.0, 1.0),
            K - 1, cut))
        return (; predictor, n_levels=K, thresholds=cut,
            extra_predictors=Symbol[], count_columns=Symbol[],
            ordinal_structure=nothing, discrimination=nothing,
            threshold_columns=Symbol[], threshold_coefs=nothing,
            response_values=recoded, cross_columns=Symbol[])
    elseif family === :ordinal
        recoded, K, _ = _rk_leveled_levels(key, raw)
        structure = _brm_ordinal_tag(getargs(rhs)[1], OrdinalStructure;
            prefix) isa Cumulative ? :cumulative : :stopping
        (discrimination, threshold_columns, threshold_coefs, cross,
            coef_implicit) = _rk_ordinal_extras(rhs, key, structure, K,
            length(raw), predictor_specs)
        append!(implicit, coef_implicit)
        thresh = Symbol(key, :_thresholds)
        vfam = structure === :cumulative ? :ordered_normal : :vector_normal
        push!(implicit, _RKVectorParameter(thresh, vfam, (0.0, 1.0), K - 1,
            thresh))
        return (; predictor, n_levels=K, thresholds=thresh,
            extra_predictors=Symbol[], count_columns=Symbol[],
            ordinal_structure=structure, discrimination, threshold_columns,
            threshold_coefs, response_values=recoded,
            cross_columns=cross)
    elseif family === :categorical
        recoded, K, _ = _rk_leveled_levels(key, raw)
        simplex, ssize = _rk_simplex_source(getargs(rhs)[1], key, vectors)
        ssize == K || error(
            "$prefix: response `$key` has $K outcome levels but simplex " *
            "`$simplex` has $ssize categories; sizes must agree")
        return (; predictor=simplex, n_levels=K, thresholds=nothing,
            extra_predictors=Symbol[], count_columns=Symbol[],
            ordinal_structure=nothing, discrimination=nothing,
            threshold_columns=Symbol[], threshold_coefs=nothing,
            response_values=recoded, cross_columns=Symbol[])
    else # :multinomial
        simplex, ssize = _rk_simplex_source(getargs(rhs)[2], key, vectors)
        matrix = get(data, key, nothing)
        matrix isa AbstractMatrix || error(
            "$prefix: response `$key` `Multinomial` needs an n×K integer " *
            "count matrix, got $(typeof(matrix))")
        (eltype(matrix) <: Integer && eltype(matrix) !== Bool &&
            all(>=(0), matrix)) || error(
            "$prefix: response `$key` `Multinomial` count matrix must " *
            "hold non-negative integers")
        K = size(matrix, 2)
        K >= 1 || error(
            "$prefix: response `$key` `Multinomial` needs at least one " *
            "category")
        ssize == K || error(
            "$prefix: response `$key` has $K count columns but simplex " *
            "`$simplex` has $ssize categories; sizes must agree")
        tails = [Symbol(key, :_count_, k) for k in 2:K]
        return (; predictor=simplex, n_levels=K, thresholds=nothing,
            extra_predictors=Symbol[], count_columns=tails,
            ordinal_structure=nothing, discrimination=nothing,
            threshold_columns=Symbol[], threshold_coefs=nothing,
            response_values=nothing, cross_columns=Symbol[])
    end
end

# Split a multinomial n×K count matrix into the lead response column
# (category 1, under the response key) plus K−1 raw tail columns.
function _rk_split_multinomial_counts!(columns::Dict{Symbol,AbstractVector},
        key::Symbol, matrix::AbstractMatrix, tails::Vector{Symbol})
    prefix = "RK backend"
    size(matrix, 2) == 1 + length(tails) || error(
        "$prefix: internal: multinomial tail names disagree with $key")
    columns[key] = Int.(vec(matrix[:, 1]))
    for (k, name) in enumerate(tails)
        columns[name] = Int.(vec(matrix[:, k + 1]))
    end
    nothing
end

function _rk_gate_crossed_columns!(columns::Dict{Symbol,AbstractVector},
        n_obs::Int)
    prefix = "RK backend"
    for key in sort!(collect(keys(columns)))
        values = columns[key]
        length(values) == n_obs || error(
            "$prefix: column `$key` has $(length(values)) rows, expected " *
            "$n_obs (one observation axis in slice 1)")
        any(ismissing, values) && error(
            "$prefix: column `$key` has missing values; slice 1 has no " *
            "missingness machinery")
        eltype(values) <: Real || continue
        all(isfinite, values) || error(
            "$prefix: column `$key` must be finite")
    end
    nothing
end

function _rk_gate_name_hygiene!(predictor_specs::AbstractVector,
        parameters::AbstractVector, assignments::AbstractVector,
        derived::AbstractVector, columns::Dict{Symbol,AbstractVector},
        response_specs::AbstractVector, vector_parameters::AbstractVector)
    prefix = "RK backend"
    pnames = [spec.name for spec in predictor_specs]
    length(unique(pnames)) == length(pnames) || error(
        "$prefix: internal: duplicate predictor names")
    both = union(Set(spec.name for spec in parameters),
        Set(spec.name for spec in assignments))
    # Leveled names share the one name table (thin-layer rule): vector
    # parameters (Dirichlet simplexes + implicit thresholds/coefs)
    # collide with nothing; threshold references must resolve; generated
    # multinomial tail columns must exist and collide with nothing.
    vnames = [spec.name for spec in vector_parameters]
    length(unique(vnames)) == length(vnames) || error(
        "$prefix: internal: duplicate vector parameter names")
    for name in vnames
        name in both && error(
            "$prefix: vector parameter `$name` collides with a " *
            "parameter/assignment name; rename it")
        name in pnames && error(
            "$prefix: vector parameter `$name` collides with predictor " *
            "`$name`; rename it")
        haskey(columns, name) && error(
            "$prefix: vector parameter `$name` collides with raw column " *
            "`$name`; rename it")
    end
    dnames = [spec.name for spec in derived]
    for spec in response_specs
        for tname in (spec.thresholds, spec.threshold_coefs)
            tname === nothing && continue
            tname in vnames || error(
                "$prefix: internal: response `$(spec.response)` references " *
                "missing vector parameter `$tname`")
        end
        for cname in spec.count_columns
            haskey(columns, cname) || error(
                "$prefix: internal: multinomial tail column `$cname` missing")
            cname in both && error(
                "$prefix: generated count column `$cname` collides with a " *
                "parameter/assignment name; rename it")
            cname in pnames && error(
                "$prefix: generated count column `$cname` collides with " *
                "predictor `$cname`; rename the predictor")
            cname in vnames && error(
                "$prefix: generated count column `$cname` collides with " *
                "vector parameter `$cname`; rename the parameter")
            cname in dnames && error(
                "$prefix: generated count column `$cname` collides with " *
                "derived column `$cname`; rename the raw column")
        end
        for cname in spec.threshold_columns
            haskey(columns, cname) || error(
                "$prefix: internal: threshold column `$cname` missing")
        end
        d = spec.discrimination
        if d isa Symbol && d in pnames && haskey(columns, d)
            error("$prefix: discrimination predictor `$d` collides with " *
                  "raw column `$d`; rename one of them")
        end
    end
    col_overlap = sort!(filter(n -> haskey(columns, n), collect(both)))
    isempty(col_overlap) || error(
        "$prefix: parameter/assignment name(s) " *
        "$(join(col_overlap, ", ")) collide with raw columns; rename them")
    for pn in pnames
        pn in both && error(
            "$prefix: predictor `$pn` collides with a parameter/assignment " *
            "name; rename it")
        block = Symbol(string(pn) * "_coef")
        block in both && error(
            "$prefix: parameter/assignment `$block` collides with " *
            "predictor `$pn` coefficient block name; rename it")
    end
    length(unique(dnames)) == length(dnames) || error(
        "$prefix: internal: duplicate derived column names")
    for dn in dnames
        dn in both && error(
            "$prefix: generated derived column `$dn` collides with a " *
            "parameter/assignment name; rename the parameter/assignment")
        dn in pnames && error(
            "$prefix: generated derived column `$dn` collides with " *
            "predictor `$dn`; rename the predictor")
        dn in vnames && error(
            "$prefix: generated derived column `$dn` collides with " *
            "vector parameter `$dn`; rename the parameter")
        haskey(columns, dn) && error(
            "$prefix: generated derived column `$dn` collides with raw " *
            "column `$dn`; rename the raw column")
    end
    snames = Symbol[]
    for spec in predictor_specs, term in spec.terms
        term.kind === :spline || continue
        push!(snames, term.options.id)
    end
    length(unique(snames)) == length(snames) || error(
        "$prefix: internal: duplicate spline smooth ids")
    for sn in snames
        sn in both && error(
            "$prefix: internal: generated spline id `$sn` collides with " *
            "a parameter/assignment name")
        sn in pnames && error(
            "$prefix: internal: generated spline id `$sn` collides with " *
            "predictor `$sn`")
        haskey(columns, sn) && error(
            "$prefix: internal: generated spline id `$sn` collides with " *
            "raw column `$sn`")
    end
    gnames = Symbol[]
    for spec in predictor_specs, term in spec.terms
        term.kind === :gp || continue
        append!(gnames, (term.options.rho, term.options.sigma,
            term.options.z, term.options.f))
    end
    length(unique(gnames)) == length(gnames) || error(
        "$prefix: internal: duplicate gp latent names")
    for gn in gnames
        gn in both && error(
            "$prefix: internal: generated gp name `$gn` collides with a " *
            "parameter/assignment name")
        gn in pnames && error(
            "$prefix: internal: generated gp name `$gn` collides with " *
            "predictor `$gn`")
        haskey(columns, gn) && error(
            "$prefix: internal: generated gp name `$gn` collides with " *
            "raw column `$gn`")
    end
    darnames = Symbol[]
    for spec in predictor_specs, term in spec.terms
        term.kind === :dar || continue
        append!(darnames, (term.options.beta, term.options.sigma))
    end
    length(unique(darnames)) == length(darnames) || error(
        "$prefix: internal: duplicate dar trajectory names")
    for dn in darnames
        dn in both && error(
            "$prefix: internal: generated dar name `$dn` collides with a " *
            "parameter/assignment name")
        dn in pnames && error(
            "$prefix: internal: generated dar name `$dn` collides with " *
            "predictor `$dn`")
        haskey(columns, dn) && error(
            "$prefix: internal: generated dar name `$dn` collides with " *
            "raw column `$dn`")
    end
    hnames = Symbol[]
    for spec in predictor_specs, term in spec.terms
        term.kind === :hsgp || continue
        push!(hnames, term.options.id)
    end
    length(unique(hnames)) == length(hnames) || error(
        "$prefix: internal: duplicate hsgp smooth ids")
    for hn in hnames
        hn in both && error(
            "$prefix: internal: generated hsgp id `$hn` collides with " *
            "a parameter/assignment name")
        hn in pnames && error(
            "$prefix: internal: generated hsgp id `$hn` collides with " *
            "predictor `$hn`")
        haskey(columns, hn) && error(
            "$prefix: internal: generated hsgp id `$hn` collides with " *
            "raw column `$hn`")
    end
    anames = Symbol[]
    for spec in predictor_specs, term in spec.terms
        term.kind === :ar || continue
        append!(anames, (term.options.state, term.options.phi,
            term.options.phi_raw, term.options.eps))
    end
    length(unique(anames)) == length(anames) || error(
        "$prefix: internal: duplicate ar latent names")
    for an in anames
        an in both && error(
            "$prefix: internal: generated ar name `$an` collides with a " *
            "parameter/assignment name")
        an in pnames && error(
            "$prefix: internal: generated ar name `$an` collides with " *
            "predictor `$an`")
        haskey(columns, an) && error(
            "$prefix: internal: generated ar name `$an` collides with " *
            "raw column `$an`")
    end
    # LKJ factor stems reserve their two derived thin-layer bindings
    # (`<stem>_scales`, `<stem>_L_corr`); a model binding under either
    # spelling would collide at lowering — fail here with BRM
    # attribution instead (SB reserves the same two names).
    for spec in parameters
        spec.family === :LKJCovarianceFactor || continue
        for piece in (Symbol(spec.name, :_scales),
                Symbol(spec.name, :_L_corr))
            piece in both && error(
                "$prefix: `$(spec.name) ~ LKJCovarianceFactor(...)` " *
                "reserves emitted binding `$piece`, but the model also " *
                "declares that parameter/assignment; rename one of them")
            piece in pnames && error(
                "$prefix: `$(spec.name) ~ LKJCovarianceFactor(...)` " *
                "reserves emitted binding `$piece`, but the model also " *
                "declares predictor `$piece`; rename one of them")
            haskey(columns, piece) && error(
                "$prefix: `$(spec.name) ~ LKJCovarianceFactor(...)` " *
                "reserves emitted binding `$piece`, but the data also " *
                "binds raw column `$piece`; rename one of them")
            piece in vnames && error(
                "$prefix: `$(spec.name) ~ LKJCovarianceFactor(...)` " *
                "reserves emitted binding `$piece`, but the model also " *
                "declares vector parameter `$piece`; rename one of them")
            piece in dnames && error(
                "$prefix: `$(spec.name) ~ LKJCovarianceFactor(...)` " *
                "reserves emitted binding `$piece`, but the model also " *
                "declares derived column `$piece`; rename one of them")
        end
    end
    menames = Symbol[]
    for spec in predictor_specs, term in spec.terms
        term.kind === :me || continue
        push!(menames, term.options.latent)
    end
    length(unique(menames)) == length(menames) || error(
        "$prefix: internal: duplicate me latent names")
    for mn in menames
        mn in both && error(
            "$prefix: internal: generated me name `$mn` collides with a " *
            "parameter/assignment name")
        mn in pnames && error(
            "$prefix: internal: generated me name `$mn` collides with " *
            "predictor `$mn`")
        haskey(columns, mn) && error(
            "$prefix: internal: generated me name `$mn` collides with " *
            "raw column `$mn`")
    end
    for n in sort!(collect(Iterators.flatten(
            (pnames, both, dnames, vnames, snames, gnames, hnames, anames,
                darnames, menames,
                keys(columns)))))
        startswith(string(n), "_ppl_") && error(
            "$prefix: name `$n` uses the reserved `_ppl_` prefix; rename it")
    end
    nothing
end

# Every `<result> ~ kernel(...)` operation in the BRMI, as `(result, rhs)` pairs.
# The kernel RHS is an ExprColumn whose head is the `kernel` marker; its first
# positional arg is the do-block cell (a verbatim `:->` lambda), the rest are the
# per-subject positional args.
function _rk_kernel_ops(brmi::BRMI)
    ops = Tuple{Symbol,Any}[]
    for (key, op_nc) in pairs(brmi.operations)
        op_nc isa NamedColumn || continue
        op = parent(op_nc)
        op isa ExprColumn{typeof(~)} || continue
        _, rhs = getargs(op, 2)
        rhs isa ExprColumn && getf(rhs) === kernel || continue
        push!(ops, (key, rhs))
    end
    ops
end

# Panel-mode kernel(...) structural extraction (Phase 1a). The cell body is
# captured verbatim as a quoted `:->` lambda; RK does not trace SLIC (unlike
# SBBRMI's plate), so we PARSE it here into a real cell scope: local assignments
# feeding exactly one in-cell observation `<resp> ~ <Family>(...)`, plus the
# collected per-subject result. `obs_dist` is the raw distribution Expr; family/
# link/evidence classification is a later increment. v1 = panel mode (ranef-free,
# mirroring `_sb_kernel_doblock!`'s no-random-effects branch): every positional
# arg is a per-subject DATA column and the subject count is their common length;
# linear-predictor args (grouped kernels) and `ragged(...)` are out of panel v1.
struct _RKKernelSpec
    result::Symbol
    subject_count::Symbol            # bound-dims key for n subjects (kernel_nsub_<result>)
    timepoint_count::Union{Nothing,Symbol} # bound-dims key for T (kernel_T_<result>), or nothing
    n_timepoints::Union{Nothing,Int} # T = common inner length of vector slices (nothing if none)
    slice_params::Vector{Symbol}     # cell do-block params, in order
    data_columns::Vector{Symbol}     # per-subject data columns, parallel to params
    slice_kinds::Vector{Symbol}      # :vector (flat T-blocked) | :scalar (per-subject), parallel
    assignments::Vector{Pair{Symbol,Any}}  # cell-local name => raw cell Expr
    obs_response::Symbol             # the sliced observation param
    obs_dist::Any                    # raw distribution Expr (classified later)
    collected::Any                   # final cell expression = per-subject result
    n_subjects::Int
end

# Plate dims the extension binds (`bind_data(..., dims)`): the subjects
# key always, the timepoint key only when vector slices name one
# (all-scalar plates leave both timepoint fields `nothing`).
function _rk_kernel_bind_dims(kernel::_RKKernelSpec)
    dims = Dict{Symbol,Int}()
    dims[kernel.subject_count] = kernel.n_subjects
    if kernel.timepoint_count !== nothing && kernel.n_timepoints !== nothing
        dims[kernel.timepoint_count] = kernel.n_timepoints
    end
    dims
end

function _rk_kernel_spec(brmi::BRMI, result::Symbol, rhs)
    prefix = "RK backend"
    dcols = getargs(rhs)
    (!isempty(dcols) && first(dcols) isa Expr && first(dcols).head === :->) || error(
        "$prefix: kernel(...) `$result` is missing its inline do-block cell")
    lam = first(dcols)
    ptuple = lam.args[1]
    params = ptuple isa Symbol ? Symbol[ptuple] :
        (Meta.isexpr(ptuple, :tuple) && all(p -> p isa Symbol, ptuple.args) ?
            Symbol[ptuple.args...] :
            error("$prefix: kernel(...) `$result` cell params must be plain names"))
    body = lam.args[2]
    body_stmts = Meta.isexpr(body, :block) ?
        Any[s for s in body.args if !(s isa LineNumberNode)] : Any[body]

    posargs = collect(dcols[2:end])
    length(posargs) == length(params) || error(
        "$prefix: kernel(...) `$result` has $(length(params)) cell params but " *
        "$(length(posargs)) positional args")
    data_columns = Symbol[]
    slice_kinds = Symbol[]
    outer_lengths = Int[]
    vector_ts = Tuple{Symbol,Int}[]
    for c in posargs
        if c isa ExprColumn && getf(c) === ragged
            error("$prefix: kernel(...) `$result` uses `ragged(...)`; the RK " *
                  "backend's ragged/grouped kernel is out of Phase-1 panel mode " *
                  "(it needs the offsets representation and RK random effects). " *
                  "Panel kernels take per-subject data columns only.")
        end
        c isa NamedColumn || error(
            "$prefix: kernel(...) `$result` positional args must be per-subject " *
            "data columns in panel mode; got $(typeof(c))")
        parent(c) isa DataColumn || error(
            "$prefix: kernel(...) `$result` arg `$(name(c))` is a linear " *
            "predictor; RK panel mode (Phase 1, ranef-free) takes per-subject " *
            "data columns only — grouped kernels need RK random effects")
        col = parent(parent(c))
        push!(data_columns, name(c))
        push!(outer_lengths, length(col))
        # A Vector-of-Vector column is a per-subject VECTOR slice (flat T-blocked
        # at bind); a scalar-eltype column is a per-subject SCALAR slice. Vector
        # slices must be uniform-T (varying = ragged = out of Phase-1).
        if eltype(col) <: AbstractVector
            push!(slice_kinds, :vector)
            inner = unique(length(v) for v in col)
            length(inner) == 1 || error(
                "$prefix: kernel(...) `$result` per-subject vector column " *
                "`$(name(c))` has varying timepoint counts $(sort(collect(inner)))" *
                "; a ragged panel is out of Phase-1 (needs the offsets " *
                "representation) — panel v1 requires equal-length per-subject vectors")
            push!(vector_ts, (name(c), only(inner)))
        else
            push!(slice_kinds, :scalar)
        end
    end
    isempty(outer_lengths) && error(
        "$prefix: kernel(...) `$result` panel mode needs at least one " *
        "per-subject data column to derive the subject count from")
    length(unique(outer_lengths)) == 1 || error(
        "$prefix: kernel(...) `$result` per-subject columns disagree on the " *
        "subject count: $(collect(zip(data_columns, outer_lengths)))")
    nsub = first(outer_lengths)
    n_timepoints = if isempty(vector_ts)
        nothing
    else
        ts = unique(last.(vector_ts))
        length(ts) == 1 || error(
            "$prefix: kernel(...) `$result` per-subject vector columns disagree " *
            "on the timepoint count T: $(vector_ts)")
        first(ts)
    end

    assignments = Pair{Symbol,Any}[]
    obs = nothing
    collected = nothing
    for s in body_stmts
        if Meta.isexpr(s, :(=)) && s.args[1] isa Symbol
            push!(assignments, s.args[1] => s.args[2])
            collected = s.args[1]
        elseif Meta.isexpr(s, :call) && length(s.args) == 3 && s.args[1] === :~
            isnothing(obs) || error(
                "$prefix: kernel(...) `$result` cell has more than one `~` " *
                "observation; Phase-1 panel mode admits exactly one")
            obs = (response=s.args[2], dist=s.args[3])
        else
            collected = s
        end
    end
    isnothing(obs) && error(
        "$prefix: kernel(...) `$result` cell has no `~` observation; Phase-1 " *
        "panel mode needs exactly one in-cell likelihood")
    obs.response isa Symbol || error(
        "$prefix: kernel(...) `$result` observation LHS must be a plain cell " *
        "name; got $(obs.response)")
    isnothing(collected) && error(
        "$prefix: kernel(...) `$result` cell must end with a collected result " *
        "expression (the per-subject value bound to `$result`)")

    _RKKernelSpec(result, Symbol("kernel_nsub_", result),
        isnothing(n_timepoints) ? nothing : Symbol("kernel_T_", result),
        n_timepoints, params, data_columns, slice_kinds,
        assignments, obs.response, obs.dist, collected, nsub)
end

# In-cell observation classification (Phase-1a: Gaussian only). The cell obs
# `<resp> ~ <Family>(...)` is a raw quoted Expr in @brm likelihood vocabulary
# (the markers SBBRMI refuses in-cell). v1 admits `Normal(location, scale)` — the
# primary panel-PKPD residual — over the subject's timepoint vector; other
# families and the evidence/weights wrappers are follow-ups (fail closed).
struct _RKKernelObs
    response::Symbol
    family::Symbol
    location::Any    # cell-scope name/expr (e.g. a cell assignment)
    scale::Any       # cell-scope name or numeric literal
end

function _rk_classify_cell_obs(result::Symbol, response::Symbol, dist)
    prefix = "RK backend"
    (Meta.isexpr(dist, :call) && dist.args[1] === :Normal &&
        length(dist.args) == 3) || error(
        "$prefix: kernel(...) `$result` in-cell observation `$response ~ ...` " *
        "admits only `Normal(location, scale)` in Phase-1a; other families and " *
        "the evidence/weights wrappers are follow-ups")
    _RKKernelObs(response, :gaussian, dist.args[2], dist.args[3])
end

# Emit the panel kernel as an `@rkppl` subject-plate carrying a REAL cell
# subgraph (NOT the thin layer's desugar-to-flat): local assignments feeding one
# in-cell observation, then the collected per-subject result. The thin layer
# lowers `plate(cols...; subjects=N) do slices... <cell> end` by mapping the cell
# subgraph over N subjects (per-subject sliced HAVE ports; globals stay HAVE
# ports visible in-cell; the vector observation reduces over the subject's
# timepoints). Contract co-designed with peer `ReactiveKernels:brm`.
function _rk_emit_kernel_ast(spec::_RKKernelSpec)
    obs = _rk_classify_cell_obs(spec.result, spec.obs_response, spec.obs_dist)
    cell = Any[]
    for (nm, ex) in spec.assignments
        push!(cell, Expr(:(=), nm, ex))
    end
    # The in-cell observation is a VECTOR obs over the subject's timepoints and
    # MUST be emitted dotted (`yy .~ Normal.(mu, sigma)`): RK rejects a scalar `~`
    # over vectors (explicit-dots ruling) and the thin-layer desugar never invents
    # dots (peer `ReactiveKernels:brm` CellSpec scoping, 2026-09-19). One obs per
    # cell, reduced in-cell.
    push!(cell, Expr(:call, :.~, obs.response,
        _rk_ast_dotted(:Normal, obs.location, obs.scale)))
    push!(cell, spec.collected)
    plate_call = Expr(:call, :plate,
        Expr(:parameters, Expr(:kw, :subjects, spec.subject_count)),
        spec.data_columns...)
    Expr(:call, :~, spec.result,
        Expr(:do, plate_call,
            Expr(:->, Expr(:tuple, spec.slice_params...), Expr(:block, cell...))))
end

# A panel kernel model's plan. Deliberately a SEPARATE type from the GLM
# `_RKStructuralPlan` (which stays untouched — no regression risk on the GLM
# path): a panel kernel has no top-level observation (its likelihood is in the
# cell), so it carries globals-as-`parameters` + `assignments` + the flattened
# per-subject `columns` + the `_RKKernelSpec`. `columns` holds the thin-layer
# bind layout: a `:vector` slice is a flat contiguous T-block (subject order),
# a `:scalar` slice is length n_sub.
struct _RKKernelPlan
    kernel::_RKKernelSpec
    obs::_RKKernelObs        # classified in-cell observation (family/loc/scale)
    parameters::Vector{_RKSampledParameter}
    assignments::Vector{_RKAssignmentSpec}
    columns::Dict{Symbol,AbstractVector}
end

_rk_plan_summary(plan::_RKKernelPlan) = string(
    plan.kernel.n_subjects, "-subject kernel plate and ",
    length(plan.parameters), " parameters")

function _brm_rk_kernel_plan(brmi::BRMI)
    prefix = "RK backend"
    kops = _rk_kernel_ops(brmi)
    length(kops) == 1 || error(
        "$prefix: RK panel mode admits exactly one kernel(...) per model; " *
        "got $(length(kops))")
    result, rhs = only(kops)
    spec = _rk_kernel_spec(brmi, result, rhs)
    # Classify the in-cell family up front (Gaussian v1); an unadmitted family
    # fails at plan time, not only at emission. The plan carries the STRUCTURED
    # obs so the thin-layer reader consumes (family, loc, scale) without parsing
    # the raw Expr.
    obs = _rk_classify_cell_obs(spec.result, spec.obs_response, spec.obs_dist)
    # Globals (scalar params + top-level assignments) via the shared prepared-
    # model path — verified to extract cleanly for a kernel model.
    ctx = _brm_backend_context(brmi; retain_mm_sources=true)
    program = _brm_prepare_program(brmi; context=ctx)
    prepared = _brm_prepare_model(brmi; program,
        additional_parameters=(), observation_overrides=Dict{Symbol,Any}())
    roots = Set{Symbol}(p.name for p in prepared.parameters)
    referenced = _brm_reachable_operations(program, roots)
    kept = Tuple(node for node in prepared.assignments if node.name in referenced)
    parameter_names = Set{Symbol}(p.name for p in prepared.parameters)
    assignment_names = Set{Symbol}(a.name for a in kept)
    consts, aliases, exprs = _rk_fold_assignment_consts!(
        kept, ctx.data, parameter_names)
    parameters = _rk_plan_parameters!(prepared, ctx.data, consts, aliases,
        parameter_names, assignment_names)
    assignments = _rk_plan_assignments!(kept, exprs, ctx.data, consts, aliases,
        parameter_names, assignment_names)
    _rk_gate_acyclic!(parameters, assignments)
    # Flatten per-subject columns to the bind layout: :vector -> flat contiguous
    # T-block (subject order); :scalar -> length n_sub.
    columns = Dict{Symbol,AbstractVector}()
    posargs = collect(getargs(rhs)[2:end])
    for (i, c) in enumerate(posargs)
        col = parent(parent(c))
        flat = spec.slice_kinds[i] === :vector ? reduce(vcat, col) : collect(col)
        all(x -> x isa Real && isfinite(x), flat) || error(
            "$prefix: kernel(...) `$result` column `$(spec.data_columns[i])` " *
            "must be finite real values")
        columns[spec.data_columns[i]] = flat
    end
    _RKKernelPlan(spec, obs, parameters, assignments, columns)
end

# Emit a panel kernel model: globals (sampled params + assignments) as top-level
# `@rkppl` statements, then the subject-plate cell subgraph.
function _rk_emit_ast(plan::_RKKernelPlan)
    stmts = Expr[]
    for parameter in plan.parameters
        push!(stmts, _rk_ast_sampled(parameter))
    end
    for assignment in plan.assignments
        push!(stmts, Expr(:(=), assignment.name,
            _rk_lower_assignment_expr(assignment.expression, assignment.name)))
    end
    push!(stmts, _rk_emit_kernel_ast(plan.kernel))
    _RKEmittedProgram(Expr[], Expr(:block, stmts...))
end

# A modeled ordinal scale feeds nothing else: a discrimination
# predictor that also fills a location, scale/shape, or categorical-logit
# tail slot would need two links at once (the scale is `log` by
# construction, and the AST skips it — the extension translates it).
function _rk_gate_ordinal_scale_slots!(response_specs::AbstractVector,
        predictor_specs::AbstractVector)
    prefix = "RK backend"
    pnames = Set(spec.name for spec in predictor_specs)
    occupied = Set{Symbol}()
    for spec in response_specs
        spec.family in (:categorical, :multinomial) ||
            push!(occupied, spec.predictor)
        spec.scale_predictor === nothing ||
            push!(occupied, spec.scale_predictor)
        union!(occupied, spec.extra_predictors)
    end
    for spec in response_specs
        d = spec.discrimination
        (d isa Symbol && d in pnames && d in occupied) || continue
        error("$prefix: response `$(spec.response)` discrimination " *
              "predictor `$d` also feeds a location, scale/shape, or " *
              "categorical-logit slot; use a dedicated `log()` predictor " *
              "for the modeled scale")
    end
    nothing
end

"""
    _brm_rk_plan(brmi::BRMI)

Lower a data-bound [`BRMI`](@ref) to the backend-neutral RK structural plan.
Slice-1 admission (population GLMs plus the random-effects draws regime,
density+gradient contract) is enforced here with RK-attributed errors;
everything else fails closed. The package extension translates the
returned [`_RKStructuralPlan`](@ref) to the thin-layer contract at the
boundary.
"""
function _brm_rk_plan(brmi::BRMI)
    prefix = "RK backend"
    # A kernel(...) model routes to the panel-kernel planner (which admits the
    # ranef-free panel case and fails closed on the rest); it has no top-level
    # observation, so it must not enter the GLM flow below.
    isempty(_rk_kernel_ops(brmi)) || return _brm_rk_kernel_plan(brmi)
    observations = _brm_direct_observations(brmi; prefix)
    keys = Tuple(observation.key for observation in observations)
    length(unique(keys)) == length(keys) || error(
        "$prefix: multi-response observation names must be unique")
    program = _brm_prepare_program(
        brmi; context=_brm_backend_context(brmi; retain_mm_sources=true))
    context = program.context
    # Phase 1: peel weights/modifiers and materialize responses.
    peeled = map(observations) do observation
        _rk_peel_observation(brmi, observation)
    end
    # Phase 2: prepared model for parameters and assignments.
    overrides = Dict(entry.key => (;
        distribution=entry.rhs, response=entry.raw_response,
        modifier=entry.modifier, weight=entry.weight_plan,
        missing_response=nothing) for entry in peeled)
    prepared = _brm_prepare_model(brmi; program,
        additional_parameters=(), observation_overrides=overrides)
    roots = Set{Symbol}()
    for entry in peeled
        union!(roots, _brm_prepared_references(_brm_prepare_expr(entry.rhs)))
        # Truncation/censoring bounds reference assignments too; without them
        # a bound-only constant (e.g. `lo` in `truncated(.., lo, 2.0)`) is
        # pruned as dead and fails as "unknown name" at evidence time.
        modifier = entry.modifier
        if !isnothing(modifier)
            for bound in (modifier.lower, modifier.upper)
                isnothing(bound) && continue
                union!(roots,
                    _brm_prepared_references(_brm_prepare_expr(bound)))
            end
        end
    end
    union!(roots, Tuple(parameter.name for parameter in prepared.parameters))
    referenced = _brm_reachable_operations(program, roots)
    kept_assignments = Tuple(node for node in prepared.assignments
                             if node.name in referenced)
    # Phase 3: parameters and assignments (folding, cycles, uniqueness).
    parameter_names = Set{Symbol}(p.name for p in prepared.parameters)
    assignment_names = Set{Symbol}(a.name for a in kept_assignments)
    length(parameter_names) == length(prepared.parameters) || error(
        "$prefix: internal: duplicate parameter names")
    length(assignment_names) == length(kept_assignments) || error(
        "$prefix: internal: duplicate assignment names")
    overlap = intersect(parameter_names, assignment_names)
    isempty(overlap) || error(
        "$prefix: name(s) $(join(sort!(collect(overlap)), ", ")) are both " *
        "a sampled parameter and an assignment; names must be unique")
    consts, aliases, exprs = _rk_fold_assignment_consts!(
        kept_assignments, context.data, parameter_names)
    parameters = _rk_plan_parameters!(prepared, context.data, consts,
        aliases, parameter_names, assignment_names)
    vector_specs = _rk_plan_vector_parameters!(prepared, consts)
    vector_by_name = Dict{Symbol,_RKVectorParameter}(
        spec.name => spec for spec in vector_specs)
    assignments = _rk_plan_assignments!(kept_assignments, exprs,
        context.data, consts, aliases, parameter_names, assignment_names)
    _rk_gate_acyclic!(parameters, assignments)
    # Phase 4: discover and plan predictors (deduped, first-referenced order).
    # A distributional response references two (location + scale/shape); both
    # plan here so the scale predictor gets terms, priors, and validation.
    predictor_order = Symbol[]
    response_predictors = Dict{Symbol,Vector{Symbol}}()
    response_extra_predictors = Dict{Symbol,Vector{Symbol}}()
    for entry in peeled
        head = getf(entry.rhs)
        if head === Categorical || head === Multinomial
            # No linear predictor (the simplex path resolves in Phase 5).
            response_predictors[entry.key] = Symbol[]
            continue
        elseif head === CategoricalLogit
            preds = _rk_categorical_refs(program, entry.rhs, entry.key)
            response_predictors[entry.key] =
                isempty(preds) ? Symbol[] : [first(preds)]
            response_extra_predictors[entry.key] =
                isempty(preds) ? Symbol[] : preds[2:end]
            for target in preds
                target in predictor_order || push!(predictor_order, target)
            end
            continue
        elseif head === MvNormalCholesky
            # Joint means resolve positionally (outcome order); the lead
            # feeds the location slot, the tail the extra predictors.
            preds = _rk_joint_refs(program, entry.rhs, entry.key,
                entry.joint_outcomes)
            response_predictors[entry.key] = [first(preds)]
            response_extra_predictors[entry.key] = preds[2:end]
            for target in preds
                target in predictor_order || push!(predictor_order, target)
            end
            continue
        elseif head === MixtureModel
            # Component args reference predictors positionally per
            # component (possibly none — all-scalar mixtures plan with
            # zero predictors); the claim check in Phase 5 accounts
            # every referenced predictor to a component slot.
            preds = _rk_mixture_refs(program, entry.rhs, entry.key)
            response_predictors[entry.key] = preds
            for target in preds
                target in predictor_order || push!(predictor_order, target)
            end
            continue
        end
        names = _rk_referenced_predictors(program, entry.rhs, entry.key)
        response_predictors[entry.key] = names
        for target in names
            target in predictor_order || push!(predictor_order, target)
        end
    end
    available = Tuple(predictor_order)
    columns = Dict{Symbol,AbstractVector}()
    derived = _RKDerivedSpec[]
    taken = union(Set{Symbol}(predictor_order), parameter_names,
        assignment_names, Set{Symbol}(spec.name for spec in vector_specs))
    predictor_specs = _RKPredictorSpec[]
    prior_specs = _RKPopulationPrior[]
    r2d2_specs = _RKR2D2Prior[]
    hs_specs = _RKHorseshoePrior[]
    r2d2_vectors = _RKVectorParameter[]
    ranef_buckets, ranef_lookup = _rk_plan_ranef_buckets(
        brmi, context, predictor_order, columns, taken, derived)
    me_sources = Set{Symbol}()
    for target in predictor_order
        spec, priors, r2d2, hs_priors = _rk_plan_predictor(
            brmi, context, target, available, columns, derived, taken,
            ranef_lookup, me_sources)
        push!(predictor_specs, spec)
        append!(prior_specs, priors)
        append!(hs_specs, hs_priors)
        isnothing(r2d2) && continue
        push!(r2d2_specs, r2d2.prior)
        append!(parameters, r2d2.scalars)
        push!(r2d2_vectors, r2d2.phi)
    end
    mo_vectors = _rk_plan_monotonic_vectors!(predictor_specs)
    predictor_link = Dict(spec.name => spec.link for spec in predictor_specs)
    # Phase 5: response specs (triples need predictor links and name tables).
    response_specs = _RKLikelihoodSpec[]
    implicit_vectors = _RKVectorParameter[]
    for entry in peeled
        head = getf(entry.rhs)
        extra = get(response_extra_predictors, entry.key, Symbol[])
        candidates = response_predictors[entry.key]
        mixture_components = _RKMixtureComponent[]
        mixture_weights::Union{Nothing,Vector{Float64},Symbol} = nothing
        family, link, scale, scale_predictor, trials, predictor, nu = if head ===
                Categorical
            length(getargs(entry.rhs)) == 1 || error(
                "$prefix: response `$(entry.key)` `Categorical` needs " *
                "`Categorical(s)` with a `Dirichlet`-sampled `s`")
            (:categorical, :identity, nothing, nothing, nothing, nothing,
                nothing)
        elseif head === Multinomial
            length(getargs(entry.rhs)) == 2 || error(
                "$prefix: response `$(entry.key)` `Multinomial` needs " *
                "`(trials, probs)`; write `Multinomial(N, s)` with a " *
                "`Dirichlet`-sampled `s`")
            mtrials = _rk_trials_argument(getargs(entry.rhs)[1], entry.key,
                parameter_names, assignment_names, consts, aliases)
            (:multinomial, :identity, nothing, nothing, mtrials, nothing,
                nothing)
        elseif head === MixtureModel
            mclassified = _rk_classify_mixture(entry.rhs, candidates,
                predictor_link, parameter_names, assignment_names, consts,
                aliases, vector_by_name, entry.key)
            mixture_components = mclassified.components
            mixture_weights = mclassified.weights
            (mclassified.family, mclassified.link, nothing, nothing,
                mclassified.trials, mclassified.anchor, nothing)
        else
            if head === CategoricalLogit && isempty(candidates)
                # Zero-arg shape: K=1 (rejected per 0dteta6) or arity mismatch.
                levels = _brm_fit_levels(entry.raw_response)
                modal = length(levels)
                modal < 1 && error(
                    "$prefix: response `$(entry.key)` has no observed levels")
                modal == 1 && error(
                    "$prefix: response `$(entry.key)` has a single " *
                    "observed level; single-level `CategoricalLogit` is " *
                    "inexpressible (decision 0dteta6) — a categorical " *
                    "response needs at least two levels")
                error("$prefix: `CategoricalLogit($(entry.key))` observed " *
                      "$modal outcome levels but received 0 non-reference " *
                      "predictors; expected $(modal - 1). Outcome level " *
                      "order is $(collect(levels)).")
            end
            extra_links = [predictor_link[p] for p in extra]
            classified = _rk_classify_response(entry.rhs, candidates,
                predictor_link, parameter_names, assignment_names, consts,
                aliases, entry.key, extra, extra_links)
            (classified.family, classified.link, classified.scale,
                classified.scale_predictor, classified.trials,
                classified.location, get(classified, :nu, nothing))
        end
        leveled = family in _RK_LEVELED_FAMILIES ?
            _rk_plan_leveled!(entry, family, predictor, extra,
                implicit_vectors, vector_by_name, context.data,
                predictor_specs) :
            _rk_unleveled(entry, predictor)
        # Mixture v1 admits no response-level weights or bounded
        # evidence (no driving case), and no `gp` mixture predictors
        # (the AST plate ranges over one response per predictor).
        if family === :mixture
            entry.weight_plan === nothing || error(
                "$prefix: response `$(entry.key)` weights on a " *
                "`MixtureModel` response are out of mixture v1")
            entry.modifier === nothing || error(
                "$prefix: response `$(entry.key)` evidence " *
                "($(entry.modifier.kind)) on a `MixtureModel` response " *
                "is out of mixture v1")
            for pname in candidates
                spec = only(s for s in predictor_specs if s.name === pname)
                any(t -> t.kind === :gp, spec.terms) && error(
                    "$prefix: response `$(entry.key)` mixture predictor " *
                    "`$pname` carries a `gp` term; gp mixture predictors " *
                    "are out of mixture v1")
            end
        end
        # Every referenced predictor must feed a slot: the scale slot naming
        # the location is degenerate, and anything else unclaimed is a name
        # shadowed across slots (rename one of them). The categorical-logit
        # tail feeds `extra_predictors`, not a slot; an ordinal
        # discrimination predictor feeds the discrimination slot. Mixture
        # components share slots freely (sharing is unambiguous —
        # every slot is explicit), so anything referenced feeds at
        # least one. Runs after the leveled plan so the ordinal extras
        # are known.
        if family === :mixture
            claimed = Set{Symbol}()
            for comp in mixture_components
                comp.location_kind === :predictor &&
                    push!(claimed, comp.location::Symbol)
                comp.scale_predictor !== nothing &&
                    push!(claimed, comp.scale_predictor)
            end
            unclaimed = filter(name -> name ∉ claimed, candidates)
            isempty(unclaimed) || error(
                "$prefix: response `$(entry.key)` references linear " *
                "predictor(s) $(join(unclaimed, ", ")) outside the " *
                "mixture component slots; every referenced predictor " *
                "must feed one slot (if a name shadows a data column, " *
                "rename one of them)")
        elseif predictor !== nothing
            scale_predictor === nothing || scale_predictor !== predictor ||
                error("$prefix: response `$(entry.key)` feeds the location " *
                      "predictor `$predictor` into the scale/shape " *
                      "slot too; the two slots take distinct predictors")
            claimed = scale_predictor === nothing ?
                Set([predictor]) :
                Set([predictor, scale_predictor])
            union!(claimed, extra)
            disc = leveled.discrimination
            disc isa Symbol && haskey(predictor_link, disc) &&
                push!(claimed, disc)
            unclaimed = filter(name -> name ∉ claimed, candidates)
            isempty(unclaimed) || error(
                "$prefix: response `$(entry.key)` references linear " *
                "predictor(s) $(join(unclaimed, ", ")) outside the location " *
                "and scale/shape slots; every referenced predictor must feed " *
                "one slot (if a name shadows a data column, rename one of them)")
        end
        if trials isa Symbol
            raw = get(context.data, trials, nothing)
            raw isa AbstractVector || error(
                "$prefix: response `$(entry.key)` trials column `$trials` " *
                "is not a vector")
            columns[trials] = raw
        end
        weights = if isnothing(entry.weight_plan)
            nothing
        else
            # Slice-2 families carry no weights (no driving case; the
            # thin layer admits none on the new triples).
            family in _RK_SLICE2_FAMILIES && error(
                "$prefix: response `$(entry.key)` weights are out of " *
                "slice 2 for family `$family`; slice 2 admits weights " *
                "on slice-1 families only")
            source = entry.weight_plan.source
            columns[source] = entry.weight_plan.values
            source
        end
        evidence = _rk_plan_evidence(entry.modifier, family,
            context.data, entry.key, columns, consts, aliases, parameter_names,
            assignment_names)
        if family === :mvnormal_cholesky
            # Joint responses cross one raw column per outcome (not the
            # packed row vectors) and carry the factor stem explicitly.
            push!(response_specs, _rk_plan_joint_response!(entry, link,
                predictor, extra, evidence, columns))
            continue
        end
        for col in leveled.cross_columns
            raw = get(context.data, col, nothing)
            raw isa AbstractVector || error(
                "$prefix: response `$(entry.key)` leveled column `$col` " *
                "is not a vector")
            columns[col] = raw
        end
        if family === :multinomial
            _rk_split_multinomial_counts!(columns, entry.key,
                context.data[entry.key], leveled.count_columns)
        else
            # Mixture responses gate on the shared component family (the
            # same-family check ran at classification).
            gate_family = family === :mixture ?
                only(unique!(map(c -> c.family, mixture_components))) :
                family
            gated = _rk_gate_response_values!(gate_family,
                leveled.response_values, entry.key)
            columns[entry.key] = gated
        end
        push!(response_specs, _RKLikelihoodSpec(family, link, entry.key,
            leveled.predictor, scale, scale_predictor, weights, evidence,
            entry.key, trials, leveled.n_levels, leveled.thresholds,
            leveled.extra_predictors, leveled.count_columns,
            leveled.ordinal_structure, leveled.discrimination,
            leveled.threshold_columns, leveled.threshold_coefs, Symbol[],
            nothing, mixture_components, mixture_weights, nu))
    end
    # Measurement-error observations ride synthetic responses (SB's
    # `x_obs ~ Normal(x_true, sd)` likelihood per `me` term).
    _rk_plan_me_observations!(response_specs, predictor_specs)
    # A modeled ordinal scale feeds no other response slot (needs the
    # whole response table, so it runs after the loop).
    _rk_gate_ordinal_scale_slots!(response_specs, predictor_specs)
    # Phase 6: one observation axis, no missing, finite data, evidence
    # values, and name hygiene (mirrors thin-side validation, R8).
    n_obs = length(first(peeled).raw_response)
    n_obs > 0 || error(
        "$prefix: plan needs at least one observation, got none")
    for entry in Iterators.drop(peeled, 1)
        length(entry.raw_response) == n_obs || error(
            "$prefix: response `$(entry.key)` has " *
            "$(length(entry.raw_response)) rows, expected $n_obs (one " *
            "observation axis in slice 1)")
    end
    _rk_gate_crossed_columns!(columns, n_obs)
    _rk_gate_trials_values!(response_specs, columns, n_obs)
    _rk_gate_multinomial_trials!(response_specs, columns, n_obs)
    _rk_gate_evidence_values!(response_specs, columns, n_obs)
    # A declared simplex must back a multinomial/categorical response
    # or mixture weights — an unreferenced `s ~ Dirichlet(...)` does
    # nothing (and the longtail lane pins that shape closed), so it
    # fails here, not silently.
    used_simplex = Set{Symbol}(spec.predictor for spec in response_specs
        if spec.family in _RK_SIMPLEX_FAMILIES)
    for spec in response_specs
        spec.family === :mixture && spec.mixture_weights isa Symbol &&
            push!(used_simplex, spec.mixture_weights)
    end
    has_mixture = any(spec -> spec.family === :mixture, response_specs)
    for spec in vector_specs
        spec.family === :simplex_dirichlet || continue
        spec.name in used_simplex || error(
            "$prefix: parameter `$(spec.name)` is declared " *
            "`Dirichlet`-sampled but no multinomial/categorical " *
            "response" *
            (has_mixture ? " or mixture weights" : "") *
            " use$(has_mixture ? "" : "s") it; write `Multinomial(N, " *
            "$(spec.name))`, `Categorical($(spec.name))`" *
            (has_mixture ? ", or `MixtureModel(..., $(spec.name))`" : "") *
            ", or drop the declaration")
    end
    _rk_gate_joint_factors!(response_specs, parameters, assignment_names)
    _rk_gate_name_hygiene!(predictor_specs, parameters, assignments,
        derived, columns, response_specs,
        [vector_specs; implicit_vectors; mo_vectors; r2d2_vectors])
    _RKStructuralPlan(response_specs, predictor_specs, prior_specs,
        parameters, assignments, derived, columns, n_obs, ranef_buckets,
        [vector_specs; implicit_vectors; mo_vectors; r2d2_vectors],
        r2d2_specs, hs_specs)
end
