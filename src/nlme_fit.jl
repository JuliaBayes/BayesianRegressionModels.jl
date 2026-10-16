# NLME estimates read in the model's own names and natural units, and the
# one-call fit through NLMEEstimation's estimators (implemented by its
# extension). The NLME partition and the RK coordinate records remain the only
# sources of names and layout.

# The key of a random-effect block: its `|ID|`, or, without one, its predictor
# and grouping column, since every predictor owns its own unnamed block.
_brm_nlme_block_key(record) = record.id === nothing ?
    Symbol(first(first(record.margins)), "|", record.group) : record.id

# A record's natural value in the constrained values: one coefficient element,
# or a whole declaration.
function _brm_nlme_record_value(values, record)
    value = _brm_rk_declaration_value(values, record.declaration)
    index = get(record, :index, ())
    isempty(index) ? value : value[index...]
end

# Each record kind adds its entries to `population` (predictor => coefficient
# pairs), `scalars`, or `other` (the constrained values of declarations without
# a BRM address). Random-effect blocks are read from Ω separately.
_brm_nlme_named!(::Val{:ranef}, acc, record, values) = acc
function _brm_nlme_named!(::Val{:population}, acc, record, values)
    push!(get!(acc.population, record.predictor, Pair{Symbol,Any}[]),
        record.coefficient => _brm_nlme_record_value(values, record))
    acc
end
function _brm_nlme_named!(::Val{:population_block}, acc, record, values)
    # Level => value pairs, as `brm_start` takes a categorical column.
    contrasts = _brm_rk_declaration_value(values, record.declaration)
    push!(get!(acc.population, record.predictor, Pair{Symbol,Any}[]),
        record.coefficient => [level => x
            for (level, x) in zip(record.level_values, contrasts)])
    acc
end
function _brm_nlme_named!(::Val{:scalar}, acc, record, values)
    push!(acc.scalars, record.declaration => _brm_nlme_record_value(values, record))
    acc
end
function _brm_nlme_named!(::Val, acc, record, values)
    for root in sort!(collect(_rk_record_roots(record)))
        hasproperty(values, root) && !any(p -> first(p) === root, acc.other) &&
            push!(acc.other, root => getproperty(values, root))
    end
    acc
end

_brm_nlme_namedtuple(pairs) = NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))

function _brm_nlme_check_point(m::BRMNLMEModel, θ, σ, Ω)
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "brm_nlme_estimates: θ has length $(length(θ)), expected $(length(m.theta))"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "brm_nlme_estimates: σ has length $(length(σ)), expected $(length(m.sigma))"))
    all(isfinite, θ) && all(isfinite, σ) || _brm_nlme_error(
        "brm_nlme_estimates: θ and σ must be finite")
    if _brm_nlme_ragged(m.view)
        Ω isa AbstractVector{<:AbstractMatrix} && length(Ω) == length(m.view.family_sizes) &&
            all(size(Ω[f]) == (K, K) for (f, K) in enumerate(m.view.family_sizes)) ||
            throw(DimensionMismatch("brm_nlme_estimates: a nested model's Ω is one " *
                "covariance matrix per family, of sizes $(m.view.family_sizes)"))
    else
        nη = length(m.view.block_margins)
        Ω isa AbstractMatrix && size(Ω) == (nη, nη) || throw(DimensionMismatch(
            "brm_nlme_estimates: Ω must be the $nη × $nη natural-effect covariance"))
    end
    nothing
end

function _brm_nlme_block(Ω::AbstractMatrix, record)
    covariance = Matrix{Float64}(Ω)
    sd = sqrt.(diag(covariance))
    cor = covariance ./ (sd * sd')
    cor[diagind(cor)] .= 1 # exactly, as `brm_start` requires of a correlation
    (; group=record.group,
       margins=[(; predictor, coefficient) for (predictor, coefficient) in record.margins],
       covariance, sd, cor)
end

"""
    brm_nlme_estimates(m::BRMNLMEModel, θ, σ, Ω[, ebes])
    brm_nlme_estimates(m::BRMNLMEModel, result[, ebes])

Read an NLME point of `m` in the model's own names and natural units. `θ` and
`σ` are `m`'s unconstrained vectors (`m.theta`, `m.sigma` order) and `Ω` the
natural-effect covariance: one block-diagonal matrix for a flat model, one
matrix per family for a nested one. With NLMEEstimation loaded, `result` may
be a FOCE(I) or SAEM result; a FOCE(I) result carries its own EBEs. `ebes` is
an `empirical_bayes` / `posthoc` result for `m`.

Returns `(; point, eta)`:

- `point` — one named tuple keyed by the model's identifiers:
  - a predictor's population coefficients under the predictor's name, one
    entry per coefficient address (`point.log_CL.weight` for
    `effect(log_CL, weight)`), on the scale the formula gives them. A
    categorical column is one entry of `level => value` pairs, one per
    non-reference level, as [`brm_start`](@ref) takes it;
  - a sampled scalar under its name, constrained (`point.sigma`, not its log);
  - a random-effect block under its `|ID|` (`point.pk`), or
    `var"<predictor>|<group>"` for a block without one, as
    `(; group, margins, covariance, sd, cor)` on the η scale;
  - `other` — the constrained values of estimated or fixed declarations that
    have no BRM address (shared HSGP and imputation latents, simplexes,
    thresholds), under their program names; empty when there are none.
- `eta` — `nothing` without `ebes`; otherwise `(; subjects, names, values)`:
  the subjects' grouping levels, the margin names of
  `NLMEEstimation.parameter_names(m).eta`, and the natural EBEs in `m`'s
  native shape (one column, or one compact vector, per subject).

Predictor names, scalar names and block keys share one namespace; a model in
which two coincide is refused. No input is mutated.
"""
function brm_nlme_estimates(m::BRMNLMEModel, θ::AbstractVector, σ::AbstractVector, Ω,
        ebes=nothing)
    _brm_nlme_check_point(m, θ, σ, Ω)
    view, backend = m.view, m.backend
    u = copy(m.unit)
    u[m.theta] = θ
    u[m.sigma] = σ
    values = _rk_constrained_values(backend, u)
    acc = (; population=OrderedDict{Symbol,Vector{Pair{Symbol,Any}}}(),
        scalars=Pair{Symbol,Any}[], other=Pair{Symbol,Any}[])
    for record in _brm_nlme_records(backend.plan)
        _brm_nlme_named!(Val(record.kind), acc, record, values)
    end
    blocks = Pair{Symbol,Any}[_brm_nlme_block_key(record) => _brm_nlme_block(
            _brm_nlme_ragged(view) ? Ω[family] : Ω[indices, indices], record)
        for (; family, record, indices) in _brm_nlme_families(backend, view)]
    # Predictors and their coefficients in formula order; records arrive in
    # emission order.
    specs = _brm_nlme_regression(backend.plan).predictors
    order = Dict(p.name => i for (i, p) in enumerate(specs))
    terms = Dict(p.name => Dict(t.addressee => j for (j, t) in enumerate(p.terms))
        for p in specs)
    population = sort!(collect(acc.population); by=p -> get(order, first(p), typemax(Int)))
    for (predictor, coefficients) in population
        position = get(terms, predictor, Dict{Symbol,Int}())
        sort!(coefficients; by=c -> get(position, first(c), typemax(Int)))
    end
    entries = vcat(Pair{Symbol,Any}[p => _brm_nlme_namedtuple(c) for (p, c) in population],
        acc.scalars, blocks, Pair{Symbol,Any}[:other => _brm_nlme_namedtuple(acc.other)])
    keys = first.(entries)
    repeated = unique(k for k in keys if count(==(k), keys) > 1)
    isempty(repeated) || _brm_nlme_error(
        "brm_nlme_estimates: $(join(string.("`", repeated, "`"), ", ")) names more than " *
        "one of a predictor, a scalar, a random-effect block and `other`")
    eta = ebes === nothing ? nothing : _brm_nlme_named_eta(m, ebes)
    (; point=_brm_nlme_namedtuple(entries), eta)
end

function _brm_nlme_named_eta(m::BRMNLMEModel, ebes)
    names = _brm_nlme_parameter_names(m).eta
    (; subjects=m.view.levels[ebes.subjects],
       names=_brm_nlme_ragged(m.view) ? names[ebes.subjects] : names,
       values=copy(ebes.modes))
end

"""
    brm_nlme_fit(x, estimator; start=nothing, inits=nothing, ebes=true,
        ad_backend=nothing, shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), kwargs...)

Fit an NLME model through the RK backend with one of NLMEEstimation's own
estimators and read the result in the model's names and natural units.
Requires NLMEEstimation and Enzyme to be loaded.

- `x` — a `@brm` model ([`BRMI`](@ref)), its [`RKBRMI`](@ref), or a prepared
  [`BRMNLMEModel`](@ref). For a BRMI or RKBRMI the model is prepared with
  `ad_backend` (default reverse-mode `AutoEnzyme`) and the `shared_point`,
  `subject_group` and `nested_groups` of [`brm_nlme_model`](@ref); passing
  those to a prepared model is refused.
- `estimator` — `NLMEEstimation.saem`, `NLMEEstimation.focei`, or
  `NLMEEstimation.empirical_bayes` (equivalently `posthoc`, which estimates
  the subject effects at the start's population values). `kwargs` pass
  through to it (`rng`, `options`, `kernel`, `subjects`, `method`, …); FOCE
  without interaction is `focei` with `FOCEOptions(interaction=false)`.
- `start` — `nothing` for the prior-median start of
  [`brm_nlme_start`](@ref)`(m)`, a full RK packed point `u` for
  `brm_nlme_start(m, u)`, or a start returned by `brm_nlme_start`.
- `inits` — initial values keyed by prior address, as [`brm_start`](@ref)
  takes them; every coordinate they do not address keeps its prior-median
  start. Give either `start` or `inits`.
- `ebes` — after `saem`, run `empirical_bayes` at the fitted population point
  (warm-started from SAEM's final subject states) so the result reports EBEs.
  SAEM's own `eta` are chain states, not EBEs. `focei` always returns its EBEs.

Returns `(; estimates, result, ebes, start, model)`: `estimates` as
[`brm_nlme_estimates`](@ref) returns them, NLMEEstimation's raw `result`, the
EBE result used (or `nothing`), the start, and the prepared model.
"""
function brm_nlme_fit end
