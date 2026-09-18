# Core RK-facing structural plan and slice-1 admission. This file deliberately
# has no dependency on ReactiveKernels or the thin RK-PPL layer: the BRM-side
# structural plan uses BRM-owned structs, and the package extension translates
# them to the thin-layer contract at the boundary (agreed contract text v1+v2,
# co-designed with ReactiveKernels:brm; the thin layer never imports BRM).
#
# Slice-1 admission (user-resolved D3/D4): population GLMs —
# Gaussian/Bernoulli-logit/Poisson-log + frequency/power weights + response
# evidence on Gaussian/Poisson — density+gradient contract. Predictor terms
# admit raw columns plus derived columns (provisional lowering):
# `&` interactions, `center`/`zscale`/`standardize`, and pure numeric data
# expressions lower to thin-layer dotted definitions computed in-graph from
# raw columns; only raw columns cross the boundary. Everything else fails
# closed with the admitted spelling named.

const _RK_SLICE1_TRIPLES = Set{Tuple{Symbol,Symbol,Symbol}}([
    (:gaussian, :identity, :identity),
    (:bernoulli_logit, :logit, :identity),
    (:bernoulli_logit, :logit, :logit),
    (:poisson_log, :log, :log),
])
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

struct _RKLikelihoodSpec
    family::Symbol # :gaussian | :bernoulli_logit | :poisson_log
    link::Symbol   # effective link: :identity | :logit | :log
    response::Symbol
    predictor::Symbol
    scale::Union{Nothing,Symbol,Float64}
    weights::Union{Nothing,Symbol}
    evidence::_RKResponseEvidence
    label::Symbol
end

struct _RKTermSpec
    kind::Symbol # :intercept | :continuous | :factor | :offset
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

struct _RKSampledParameter
    name::Symbol
    family::Symbol
    args::Tuple # Number literals or Symbol param/assignment refs, positional
    support_override::Union{Nothing,Symbol}
    label::Symbol
end

struct _RKAssignmentSpec
    name::Symbol
    expression::Any # scalar _BRMPreparedExpr over folded refs
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
        if term.kind === :intercept || term.kind === :continuous
            total += 1
        elseif term.kind === :factor
            width = length(_rk_grouping_levels(
                plan.columns[only(term.columns)]))
            total += term.options.coding === :fullrank ? width : width - 1
        end
    end
    total
end

Base.show(io::IO, x::RKBRMI) = print(io, "RKBRMI with ",
    _rk_num_coefficients(x.plan), " population coefficients and ",
    x.plan.n_obs, " observations")

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
    "`logit(p) ~ ...`, or `y ~ Poisson(mu)` + `log(mu) ~ ...`"

function _rk_predictor_link(brmi::BRMI, target::Symbol)
    prefix = "RK backend"
    op = linear_predictor_op(brmi, target)
    lhs, _ = getargs(op, 2)
    link_fn, _ = _peel_lp_lhs(lhs)
    link_fn === identity && return :identity
    link_fn isa Function || error(
        "$prefix: predictor `$target` has an uninterpretable link; slice 1 " *
        "admits identity, logit, and log links")
    name = nameof(link_fn)
    name === :logit && return :logit
    name === :log && return :log
    error("$prefix: predictor `$target` uses link `$name`; slice 1 admits " *
          "identity, logit, and log links")
end

_rk_is_predictor_ref(arg, predictor::Symbol) =
    arg isa NamedColumn && name(arg) == predictor

function _rk_strip_logistic(arg, predictor::Symbol)
    # `Bernoulli(logistic(eta))` lowers identically to `BernoulliLogit(eta)`;
    # the emitter strips the wrapper so the contract sees one spelling.
    arg isa ExprColumn && getf(arg) === logistic || return nothing
    args = getargs(arg)
    length(args) == 1 && _rk_is_predictor_ref(only(args), predictor) || return nothing
    isempty(getkwargs(arg)) || return nothing
    predictor
end

function _rk_scale_argument(arg, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, response::Symbol)
    prefix = "RK backend"
    arg isa Number && return _rk_positive_literal(arg, response, "scale")
    if arg isa NamedColumn
        parent(arg) isa DataColumn && error(
            "$prefix: response `$response` scale cannot be a data column; " *
            "slice 1 admits a sampled parameter, a scalar assignment, or " *
            "a positive numeric literal")
        kind, value = _rk_resolve_use_ref(name(arg), consts, aliases,
            parameters, assignments, "response `$response` scale")
        kind === :number && return _rk_positive_literal(value, response, "scale")
        return value
    end
    error("$prefix: response `$response` scale must be a sampled parameter " *
          "or a positive numeric literal")
end

function _rk_positive_literal(x::Number, response::Symbol, what::String)
    prefix = "RK backend"
    value = Float64(x)
    isfinite(value) && value > 0 || error(
        "$prefix: response `$response` $what must be finite and positive")
    value
end

function _rk_classify_response(rhs::ExprColumn, predictor::Symbol,
        predictor_link::Symbol, parameters::Set{Symbol},
        assignments::Set{Symbol}, consts::Dict{Symbol,Float64},
        aliases::Dict{Symbol,Symbol}, response::Symbol)
    prefix = "RK backend"
    head = getf(rhs)
    args = getargs(rhs)
    isempty(getkwargs(rhs)) || error(
        "$prefix: response `$response` distribution keywords are out of " *
        "slice 1; admitted spellings: $_RK_ADMITTED_SPELLINGS")
    if head === Normal
        length(args) == 2 || error(
            "$prefix: response `$response` `Normal` needs `(location, scale)`")
        _rk_is_predictor_ref(args[1], predictor) || error(
            "$prefix: response `$response` location must be the linear " *
            "predictor `$predictor` itself, not a deterministic transform; " *
            "write the transform into the predictor formula")
        scale = _rk_scale_argument(args[2], parameters, assignments, consts,
            aliases, response)
        triple = (:gaussian, predictor_link, predictor_link)
        triple in _RK_SLICE1_TRIPLES || error(
            "$prefix: response `$response` pairs `Normal` with a " *
            "$predictor_link-link predictor; slice 1 admits " *
            "$_RK_ADMITTED_SPELLINGS")
        return (:gaussian, predictor_link, scale)
    elseif head === BernoulliLogit
        length(args) == 1 || error(
            "$prefix: response `$response` `BernoulliLogit` needs one argument")
        _rk_is_predictor_ref(only(args), predictor) || error(
            "$prefix: response `$response` argument must be the linear " *
            "predictor `$predictor` itself")
        # NOTE: triple (:bernoulli_logit, :logit, :logit) exists for the
        # plain-`Bernoulli` head only; a `BernoulliLogit` head on a
        # logit-link predictor would apply the link twice.
        predictor_link === :identity || error(
            "$prefix: response `$response` applies `BernoulliLogit` on top " *
            "of a $predictor_link-link predictor (double link); use an " *
            "identity-link predictor")
        return (:bernoulli_logit, :logit, nothing)
    elseif head === Bernoulli
        length(args) == 1 || error(
            "$prefix: response `$response` `Bernoulli` needs one argument")
        arg = only(args)
        if _rk_is_predictor_ref(arg, predictor)
            triple = (:bernoulli_logit, predictor_link, predictor_link)
            triple in _RK_SLICE1_TRIPLES || error(
                "$prefix: response `$response` pairs plain `Bernoulli` with " *
                "a $predictor_link-link predictor; write `BernoulliLogit` " *
                "with an identity predictor or `Bernoulli(p)` with a " *
                "`logit(p)` predictor")
            return (:bernoulli_logit, predictor_link, nothing)
        end
        isnothing(_rk_strip_logistic(arg, predictor)) || predictor_link === :identity || error(
            "$prefix: response `$response` applies `logistic` on top of a " *
            "$predictor_link-link predictor (double link); use an " *
            "identity-link predictor")
        isnothing(_rk_strip_logistic(arg, predictor)) && error(
            "$prefix: response `$response` probability must be the linear " *
            "predictor `$predictor` or `logistic($predictor)`; admitted " *
            "spellings: $_RK_ADMITTED_SPELLINGS")
        return (:bernoulli_logit, :logit, nothing)
    elseif head === Poisson
        length(args) == 1 || error(
            "$prefix: response `$response` `Poisson` needs one argument")
        _rk_is_predictor_ref(only(args), predictor) || error(
            "$prefix: response `$response` rate must be the linear " *
            "predictor `$predictor` itself; write `Poisson(mu)` with a " *
            "`log(mu)` predictor (slice 1 has no `Poisson(exp(..))` spelling)")
        triple = (:poisson_log, predictor_link, predictor_link)
        triple in _RK_SLICE1_TRIPLES || error(
            "$prefix: response `$response` pairs `Poisson` with a " *
            "$predictor_link-link predictor; write `Poisson(mu)` with a " *
            "`log(mu)` predictor")
        return (:poisson_log, predictor_link, nothing)
    end
    head_name = head isa Function ? nameof(head) :
        head isa Type ? nameof(head) : string(head)
    error("$prefix: response `$response` family `$head_name` is out of " *
          "slice 1; admitted spellings: $_RK_ADMITTED_SPELLINGS")
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
        "$prefix: response `$response` evidence ($(modifier.kind)) on a " *
        "Bernoulli response is out of slice 1")
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
# sort order, so declared-but-unobserved levels cannot cross).
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

function _rk_population_priors(brmi::BRMI, design, target::Symbol,
        available::Tuple, factor_addressees::Set{Symbol},
        terms::Vector{_RKTermSpec}, derived::Vector{_RKDerivedSpec})
    prefix = "RK backend"
    overrides = _brm_simple_population_effect_overrides(
        brmi, design; prefix, available_predictors=available)
    # The shared seam resolves (location, scale) without checking the family;
    # slice 1 admits Normal-only population effects (NativePPL precedent).
    claimed = isnothing(overrides) ? () : overrides
    for expression in claimed
        isnothing(expression) && continue
        expression isa ExprColumn && getf(expression) === Normal || error(
            "$prefix: predictor `$target` population-effect priors must " *
            "be `Normal(location, scale)` in slice 1")
        isempty(getkwargs(expression)) || error(
            "$prefix: predictor `$target` population-effect `Normal` " *
            "prior cannot have keywords in slice 1")
    end
    n = length(design.columns)
    stated = isnothing(overrides) ? fill(false, n) :
        Bool[!isnothing(cell) for cell in overrides]
    location, scale = _brm_materialize_normal_effect_priors(overrides, n;
        prefix)
    groups = Dict{Symbol,Vector{Int}}()
    order = Symbol[]
    for (i, column) in enumerate(design.columns)
        kind = isnothing(column.preprocess) ? nothing :
            column.preprocess.kind
        # Derived columns group by label (each is its own coefficient);
        # factor dummies keep grouping by source (one prior per block).
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
    priors = _RKPopulationPrior[]
    for addressee in order
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

function _rk_plan_predictor(brmi::BRMI, context, target::Symbol,
        available::Tuple, columns::Dict{Symbol,AbstractVector},
        derived::Vector{_RKDerivedSpec}, taken::Set{Symbol})
    prefix = "RK backend"
    op = linear_predictor_op(brmi, target)
    _, rhs = getargs(op, 2)
    link = _rk_predictor_link(brmi, target)
    raw_terms = _brm_additive_terms(rhs)
    structured = filter(t -> _brm_prepares_term(t), raw_terms)
    isempty(structured) || error(
        "$prefix: predictor `$target` structured term(s) " *
        "$(join(unique!(string.(getf.(filter(t -> t isa ExprColumn, structured)))), ", ")) " *
        "are out of slice 1 (population GLMs only)")
    grouped = filter(t -> _brm_is_grouped_term(t), raw_terms)
    isempty(grouped) || error(
        "$prefix: predictor `$target` random-effect term(s) are out of " *
        "slice 1 (population GLMs only)")
    ordinary = Tuple(t for t in raw_terms if !(t in structured) && !(t in grouped))
    isempty(ordinary) && error(
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
    _rk_gate_cover_identified!(
        terms, ordinary, target, context.data, has_intercept)
    _rk_gate_cross_identified!(
        terms, spines, derived, context.data, target, has_intercept)
    geometry = _brm_prepare_predictor_geometry(
        brmi, context, target; available_predictors=available)
    isempty(geometry.terms) || error(
        "$prefix: internal: structured terms survived pre-check in `$target`")
    isempty(geometry.component.random_effects) || error(
        "$prefix: internal: random effects survived pre-check in `$target`")
    isnothing(geometry.r2d2) || error(
        "$prefix: predictor `$target` `r2d2` priors are out of slice 1")
    design = geometry.component.design
    factor_addressees = Set{Symbol}(t.addressee
        for t in terms if t.kind === :factor)
    priors = _rk_population_priors(brmi, design, target, available,
        factor_addressees, terms, derived)
    _RKPredictorSpec(target, link, terms, target), priors
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
    if family === :gaussian
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
    elseif family === :poisson_log
        (eltype(values) <: Integer && all(>=(0), values)) || error(
            "$prefix: response `$response` must hold non-negative integers")
    end
    values
end

function _rk_peel_observation(brmi::BRMI, observation)
    prefix = "RK backend"
    missing_response = _brm_missing_response_plan(observation.lhs; prefix)
    isnothing(missing_response) || error(
        "$prefix: response `$(observation.key)` uses `mi()` (modelled " *
        "missingness); slice 1 has no missingness machinery")
    observation.lhs isa NamedColumn || error(
        "$prefix: response `$(observation.key)` is not a plain response " *
        "column (joint/multivariate responses are out of slice 1)")
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
    (; key=observation.key, rhs, raw_response, weight_plan, modifier)
end

function _rk_referenced_predictor(program, rhs, response::Symbol)
    prefix = "RK backend"
    referenced = _brm_reachable_operations(
        program, _brm_prepared_references(_brm_prepare_expr(rhs)))
    names = Symbol[node.name for node in program.operations
                   if node.role === :predictor && node.name in referenced]
    isempty(names) && error(
        "$prefix: response `$response` does not reference a linear " *
        "predictor; slice 1 lowers likelihoods of a declared linear " *
        "predictor (`mu ~ 1 + x`)")
    length(names) == 1 || error(
        "$prefix: response `$response` references several linear " *
        "predictors ($(join(names, ", "))); distributional and " *
        "multi-predictor likelihoods are out of slice 1")
    only(names)
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
        derived::AbstractVector, columns::Dict{Symbol,AbstractVector})
    prefix = "RK backend"
    pnames = [spec.name for spec in predictor_specs]
    length(unique(pnames)) == length(pnames) || error(
        "$prefix: internal: duplicate predictor names")
    both = union(Set(spec.name for spec in parameters),
        Set(spec.name for spec in assignments))
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
    dnames = [spec.name for spec in derived]
    length(unique(dnames)) == length(dnames) || error(
        "$prefix: internal: duplicate derived column names")
    for dn in dnames
        dn in both && error(
            "$prefix: generated derived column `$dn` collides with a " *
            "parameter/assignment name; rename the parameter/assignment")
        dn in pnames && error(
            "$prefix: generated derived column `$dn` collides with " *
            "predictor `$dn`; rename the predictor")
        haskey(columns, dn) && error(
            "$prefix: generated derived column `$dn` collides with raw " *
            "column `$dn`; rename the raw column")
    end
    for n in sort!(collect(Iterators.flatten(
            (pnames, both, dnames, keys(columns)))))
        startswith(string(n), "_ppl_") && error(
            "$prefix: name `$n` uses the reserved `_ppl_` prefix; rename it")
    end
    nothing
end

"""
    _brm_rk_plan(brmi::BRMI)

Lower a data-bound [`BRMI`](@ref) to the backend-neutral RK structural plan.
Slice-1 admission (population GLMs, density+gradient contract) is enforced
here with RK-attributed errors; everything else fails closed. The package
extension translates the returned [`_RKStructuralPlan`](@ref) to the
thin-layer contract at the boundary.
"""
function _brm_rk_plan(brmi::BRMI)
    prefix = "RK backend"
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
    assignments = _rk_plan_assignments!(kept_assignments, exprs,
        context.data, consts, aliases, parameter_names, assignment_names)
    _rk_gate_acyclic!(parameters, assignments)
    # Phase 4: discover and plan predictors (deduped, first-referenced order).
    predictor_order = Symbol[]
    response_predictor = Dict{Symbol,Symbol}()
    for entry in peeled
        target = _rk_referenced_predictor(program, entry.rhs, entry.key)
        response_predictor[entry.key] = target
        target in predictor_order || push!(predictor_order, target)
    end
    available = Tuple(predictor_order)
    columns = Dict{Symbol,AbstractVector}()
    derived = _RKDerivedSpec[]
    taken = union(Set{Symbol}(predictor_order), parameter_names,
        assignment_names)
    predictor_specs = _RKPredictorSpec[]
    prior_specs = _RKPopulationPrior[]
    for target in predictor_order
        spec, priors = _rk_plan_predictor(
            brmi, context, target, available, columns, derived, taken)
        push!(predictor_specs, spec)
        append!(prior_specs, priors)
    end
    predictor_link = Dict(spec.name => spec.link for spec in predictor_specs)
    # Phase 5: response specs (triples need predictor links and name tables).
    response_specs = _RKLikelihoodSpec[]
    for entry in peeled
        predictor = response_predictor[entry.key]
        family, link, scale = _rk_classify_response(entry.rhs, predictor,
            predictor_link[predictor], parameter_names, assignment_names,
            consts, aliases, entry.key)
        weights = if isnothing(entry.weight_plan)
            nothing
        else
            source = entry.weight_plan.source
            columns[source] = entry.weight_plan.values
            source
        end
        evidence = _rk_plan_evidence(entry.modifier, family, context.data,
            entry.key, columns, consts, aliases, parameter_names,
            assignment_names)
        gated = _rk_gate_response_values!(family, entry.raw_response, entry.key)
        columns[entry.key] = gated
        push!(response_specs, _RKLikelihoodSpec(family, link, entry.key,
            predictor, scale, weights, evidence, entry.key))
    end
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
    _rk_gate_evidence_values!(response_specs, columns, n_obs)
    _rk_gate_name_hygiene!(
        predictor_specs, parameters, assignments, derived, columns)
    _RKStructuralPlan(response_specs, predictor_specs, prior_specs,
        parameters, assignments, derived, columns, n_obs)
end
