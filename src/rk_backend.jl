# Core RK-facing structural plan and slice-1 admission. This file deliberately
# has no dependency on ReactiveKernels or the thin RK-PPL layer: the BRM-side
# structural plan uses BRM-owned structs, and the package extension translates
# them to the thin-layer contract at the boundary (agreed contract text v1+v2,
# co-designed with ReactiveKernels:brm; the thin layer never imports BRM).
#
# Slice-1 admission (user-resolved D3/D4): population GLMs —
# Gaussian/Bernoulli-logit/Poisson-log + frequency/power weights + response
# evidence on Gaussian/Poisson — density+gradient contract. Everything else
# fails closed with the admitted spelling named.

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
    options::NamedTuple # factor: (contrasts=:treatment, ref::Int, levels=:observed)
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

function _rk_num_coefficients(plan::_RKStructuralPlan)
    total = 0
    for predictor in plan.predictors, term in predictor.terms
        if term.kind === :intercept || term.kind === :continuous
            total += 1
        elseif term.kind === :factor
            total += length(sort!(unique(plan.columns[only(term.columns)]))) - 1
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
    end
    nothing
end

function _rk_factor_options(source::Symbol, raw::AbstractVector,
        ref_value::Integer, target::Symbol)
    prefix = "RK backend"
    fit_levels = collect(_brm_fit_levels(raw))
    ref_value in fit_levels || error(
        "$prefix: predictor `$target` factor `$source` ref `$ref_value` is " *
        "not an observed level (levels: $(join(fit_levels, ", ")))")
    # Factor columns cross as plain value vectors so both sides sort the same
    # values; the thin layer indexes its sort-ordered levels from `ref`.
    crossed = raw isa CA.CategoricalVector ? collect(raw) : raw
    sort_levels = sort!(unique(crossed))
    ref_index = findfirst(==(ref_value), sort_levels)
    isnothing(ref_index) && error(
        "$prefix: predictor `$target` factor `$source` ref level " *
        "`$ref_value` is not observed; unused levels cannot be the " *
        "reference in slice 1 (drop them or set an explicit observed `ref`)")
    (contrasts=:treatment, ref=ref_index, levels=:observed), crossed
end

function _rk_term_spec(term, target::Symbol, data::AbstractDict,
        columns::Dict{Symbol,AbstractVector})
    prefix = "RK backend"
    term isa Integer && term == 1 && return _RKTermSpec(
        :intercept, Symbol[], (;), :Intercept, :Intercept)
    if term isa ExprColumn && getf(term) === offset
        args = getargs(term)
        length(args) == 1 || error(
            "$prefix: predictor `$target` `offset()` needs exactly one argument")
        isempty(getkwargs(term)) || error(
            "$prefix: predictor `$target` `offset()` takes no keywords")
        sources = _brm_data_expression_sources(only(args))
        length(sources) == 1 || error(
            "$prefix: predictor `$target` slice 1 supports `offset()` of a " *
            "single raw data column")
        source = only(sources)
        raw = get(data, source, nothing)
        raw isa AbstractVector{<:Real} || error(
            "$prefix: predictor `$target` offset column `$source` must be " *
            "a real vector")
        columns[source] = raw
        return _RKTermSpec(:offset, [source], (;), source,
            Symbol(:offset_, source))
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
            "categorical (integer codes or a CategoricalVector)")
        ref_value = get(kwargs, :ref, first(_brm_fit_levels(raw)))
        ref_value isa Integer || error(
            "$prefix: predictor `$target` `factor($source; ref=...)` ref " *
            "must be an integer level value")
        options, crossed = _rk_factor_options(source, raw, ref_value, target)
        columns[source] = crossed
        return _RKTermSpec(:factor, [source], options, source, source)
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
            options, crossed = _rk_factor_options(
                source, raw, first(_brm_fit_levels(raw)), target)
            columns[source] = crossed
            return _RKTermSpec(:factor, [source], options, source, source)
        end
        raw isa AbstractVector{<:Real} && !(eltype(raw) <: Integer) || error(
            "$prefix: predictor `$target` column `$source` is neither a " *
            "continuous (real non-integer) nor a categorical column")
        columns[source] = raw
        return _RKTermSpec(:continuous, [source], (;), source, source)
    end
    term isa ExprColumn && getf(term) === (&) && error(
        "$prefix: predictor `$target` interactions (`&`) are out of slice 1")
    term isa ExprColumn && _brm_is_term_head(getf(term)) && error(
        "$prefix: predictor `$target` term `$(nameof(getf(term)))` is out " *
        "of slice 1")
    error("$prefix: predictor `$target` term `$term` is not supported in " *
          "slice 1 (admitted: `1`, continuous columns, integer/categorical " *
          "columns, `factor()`, `offset()`)")
end

function _rk_population_priors(brmi::BRMI, design, target::Symbol,
        available::Tuple)
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
    location, scale = _brm_materialize_normal_effect_priors(overrides, n;
        prefix)
    groups = Dict{Symbol,Vector{Int}}()
    order = Symbol[]
    for (i, column) in enumerate(design.columns)
        addressee = isnothing(column.source) ? column.label : column.source
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
        first_loc, first_scale = location[first(idxs)], scale[first(idxs)]
        all(i -> location[i] == first_loc && scale[i] == first_scale,
            idxs) || error(
            "$prefix: predictor `$target` addressee `$addressee` has " *
            "disagreeing population priors across its columns; slice 1 " *
            "needs one shared Normal per addressee (address the source " *
            "column, not individual contrasts)")
        push!(priors, _RKPopulationPrior(
            target, addressee, first_loc, first_scale))
    end
    priors
end

function _rk_plan_predictor(brmi::BRMI, context, target::Symbol,
        available::Tuple, columns::Dict{Symbol,AbstractVector})
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
    cellmeans = _brm_cellmeans_block(ordinary; implicit_intercept=false)
    isnothing(cellmeans) || error(
        "$prefix: predictor `$target` uses cell-means coding (categorical " *
        "without an intercept); slice 1 needs treatment contrasts — add an " *
        "intercept or opt the categorical into treatment coding")
    # Classify before building geometry: fail fast on unknown terms with RK
    # attribution, before shared machinery can throw undecorated errors.
    terms = _RKTermSpec[
        _rk_term_spec(term, target, context.data, columns) for term in ordinary]
    geometry = _brm_prepare_predictor_geometry(
        brmi, context, target; available_predictors=available)
    isempty(geometry.terms) || error(
        "$prefix: internal: structured terms survived pre-check in `$target`")
    isempty(geometry.component.random_effects) || error(
        "$prefix: internal: random effects survived pre-check in `$target`")
    isnothing(geometry.r2d2) || error(
        "$prefix: predictor `$target` `r2d2` priors are out of slice 1")
    design = geometry.component.design
    priors = _rk_population_priors(brmi, design, target, available)
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
    (isnothing(upper) || (upper isa Number && upper == Inf)) || error(
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
        columns::Dict{Symbol,AbstractVector})
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
    for n in sort!(collect(Iterators.flatten(
            (pnames, both, keys(columns)))))
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
    predictor_specs = _RKPredictorSpec[]
    prior_specs = _RKPopulationPrior[]
    for target in predictor_order
        spec, priors = _rk_plan_predictor(
            brmi, context, target, available, columns)
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
    _rk_gate_name_hygiene!(predictor_specs, parameters, assignments, columns)
    _RKStructuralPlan(response_specs, predictor_specs, prior_specs,
        parameters, assignments, columns, n_obs)
end
