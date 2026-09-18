# Conditional effects: bambi `interpret`-style predictions, comparisons, and
# slopes over a fitted SBBRMI.
#
# bambi's `plot_predictions` / `plot_comparisons` / `plot_slopes` notebooks
# evaluate a fitted model on a *grid* — focal predictors varied, everything
# else held at typical values — and summarize the posterior of the response
# mean there. BRM had the pieces (frozen `reprocess`, descriptor `:predict`,
# logical output resolution) but no entry point that composes them, and no
# public path at all to the response *mean* on new data: `:predict` returns
# only StanBlocks `:draw` generated quantities, while the linear-predictor
# transformed parameters it also computes are discarded by output selection.
# This file closes both gaps without touching emission:
#
# * `brm_prediction_grid` builds the grid from the fitted data record;
# * `brm_conditional_draws` evaluates response means (`target=:mean`, exact
#   transformed-parameter evaluation on the reprocessed problem — no RNG, no
#   seed) or posterior-predictive draws (`target=:predictive`) on it;
# * `brm_contrast_draws` / `brm_slope_draws` derive group comparisons and
#   finite-difference slopes from conditional draws;
# * `brm_summarize_draws` reduces draws to mean/sd/median/HDI rows for tables
#   and plots.
#
# Response means are outcome-anchored: the outcome's family decides which
# formula argument carries the mean and how it maps to the response scale,
# and the argument must be a bare linear predictor or sampled parameter,
# one supported unary wrapper around exactly one (`exp(mu)`), a number, or
# a data column. Anything more entangled — a second linear predictor inside
# the mean argument, an offset outside it, a chained outcome, an unmapped
# family — fails closed with the scale=:link / target=:predictive
# alternative named. A link-transformed observation LHS such as `log(y) ~
# Normal(mu, s)` would likewise be refused (E[y] = exp(mu + s^2/2) needs the
# residual scale, so `exp.(mu)` would be a median masquerading as a mean),
# but no BRM spelling lowers one today — the model itself fails to build —
# so that guard is defense-in-depth only.
#
# Declaration links need no handling here: an LHS-linked declaration
# `log(mu) ~ 1 + x` emits BOTH the linear form (`log_mu`) and the public
# predictor (`mu = exp(log_mu)`), and the `:linear_predictor` output
# resolves to the public one — the value the likelihood sees. So the
# evaluated carrier is already what the outcome's mean argument denotes,
# whether the declaration is bare or linked. `scale=:link` returns that
# carrier as-is (exactly what `brm_output_draws` with
# `role=:linear_predictor` yields on fitted draws); `scale=:response`
# applies only the formula-level wrapper and the family map on top.

# ---- prediction grids ------------------------------------------------------

_brm_grid_pairs(focal::Symbol) = [focal => nothing]
_brm_grid_pairs(focal::AbstractVector{Symbol}) = [f => nothing for f in focal]
_brm_grid_pairs(focal::AbstractVector{<:Pair}) =
    [Symbol(k) => v for (k, v) in focal]
_brm_grid_pairs(focal::AbstractDict) =
    [Symbol(k) => v for (k, v) in sort!(collect(pairs(focal)); by=first)]
_brm_grid_pairs(focal) = error(
    "BRM prediction: `focal` must be a Symbol, a vector of Symbols, a " *
    "dict, or a vector of pairs (got $(typeof(focal))).")

_brm_grid_values(::Nothing) = nothing
_brm_grid_values(v::Real) = [v]
_brm_grid_values(v::AbstractVector) = collect(v)
_brm_grid_values(v::Tuple) = collect(v)
_brm_grid_values(v) = error(
    "BRM prediction: focal values must be a scalar or a vector (got " *
    "$(typeof(v))).")

function _brm_grid_typical(values::AbstractVector, column::Symbol)
    present = [v for v in values if !ismissing(v)]
    isempty(present) && error(
        "BRM prediction: column `$column` has no non-missing training values " *
        "to hold at a typical value; pass it explicitly via `fixed=`.")
    tel = nonmissingtype(eltype(values))
    if tel <: Real && !(tel <: Bool) && !(tel <: Integer)
        return sum(present) / length(present)
    end
    best, best_count = present[1], 0
    counts = Dict{Any,Int}()
    for v in present
        counts[v] = get(counts, v, 0) + 1
    end
    for v in present
        if counts[v] > best_count
            best, best_count = v, counts[v]
        end
    end
    best
end

function _brm_grid_default_values(column::Symbol, values::AbstractVector,
                                  n::Integer)
    present = [v for v in values if !ismissing(v)]
    isempty(present) && error(
        "BRM prediction: focal `$column` has no non-missing training values; " *
        "pass explicit values.")
    tel = nonmissingtype(eltype(values))
    if tel <: Real && !(tel <: Bool) && !(tel <: Integer)
        lo, hi = extrema(present)
        return collect(range(lo, hi; length=n))
    elseif tel <: Integer && !(tel <: Bool)
        lo, hi = extrema(present)
        grid = unique!(round.(tel, range(lo, hi; length=n)))
        return sort!(collect(grid))
    else
        return unique(present)
    end
end

function _brm_grid_check_values(column::Symbol, values::AbstractVector,
                                training::AbstractVector)
    isempty(values) && error(
        "BRM prediction: focal `$column` needs at least one value.")
    any(ismissing, values) && error(
        "BRM prediction: focal `$column` values must not contain `missing`.")
    tel = nonmissingtype(eltype(training))
    if tel <: Bool
        all(v -> v isa Bool, values) || error(
            "BRM prediction: focal `$column` is Bool-typed; pass Bool values.")
    elseif tel <: Integer
        all(v -> v isa Integer, values) || error(
            "BRM prediction: focal `$column` is integer-typed; pass integer " *
            "values.")
    elseif tel <: Real
        all(v -> v isa Real && isfinite(v), values) || error(
            "BRM prediction: focal `$column` values must be finite real numbers.")
    else
        for v in values
            v in training || error(
                "BRM prediction: focal `$column` value $(repr(v)) was not " *
                "observed in training; conditional grids stay on fitted levels.")
        end
    end
    values
end

function _brm_grid_fixed(fixed::NamedTuple)
    Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(fixed))
end
_brm_grid_fixed(fixed::AbstractDict) =
    Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(fixed))
_brm_grid_fixed(fixed::AbstractVector{<:Pair}) =
    Dict{Symbol,Any}(Symbol(k) => v for (k, v) in fixed)
_brm_grid_fixed(fixed) = error(
    "BRM prediction: `fixed=` must be a NamedTuple, dict, or vector of " *
    "pairs (got $(typeof(fixed))).")

function _brm_grid_response_fill(brmi::BRMI, response::Symbol)
    values = column_data(brmi, response)
    isnothing(values) && error(
        "BRM prediction: response `$response` is not data-backed; " *
        "conditional grids need observed response columns.")
    values isa AbstractVector || error(
        "BRM prediction: response `$response` is not a vector column; joint " *
        "and ragged responses have no v1 conditional grid.")
    first_present = findfirst(v -> !ismissing(v), values)
    isnothing(first_present) && error(
        "BRM prediction: response `$response` has no non-missing training " *
        "value to fill the grid with; pass it explicitly via `fixed=`.")
    values[first_present]
end

"""
    brm_prediction_grid(d::BRMDescriptor; focal, n=25, fixed=(;)) -> NamedTuple

Build the evaluation grid conditional predictions are computed on: the
`focal` predictor(s) varied, every other model column held at one typical
value. This is the data half of bambi's `conditional` argument; pair it
with [`brm_conditional_draws`](@ref), which reprocesses the fitted model
onto the grid and evaluates it there.

- `focal` — a `Symbol`, a vector of `Symbol`s, a dict, or a vector of
  pairs. Plain symbols take default values: `n` points spanning the
  observed range for continuous predictors, the fitted levels for
  integer/Bool/categorical ones. A `name => values` pair (or dict entry)
  uses explicit `values` (a scalar or vector); a dict is read in sorted
  key order, a vector of pairs in the order given. Several focals expand
  to the full factorial, first focal fastest.
- `n` — default grid length for continuous focals (`n >= 2`).
- `fixed=` — `name => value` overrides for non-focal columns (a typical
  is otherwise the mean for continuous columns, the most frequent
  training value — first-seen wins ties — for everything else).

Response columns are filled with their first non-missing training value.
That fill is never evaluated: means and predictive draws condition on
fitted parameter draws, not on it. It only has to satisfy the Stan data
constraints, which a genuinely observed value does by construction.

The returned `NamedTuple` of equal-length vectors feeds
`brm_execute(d, :reprocess, grid)` directly. Responses cannot be focal;
unseen categorical levels, unknown columns, and `missing` fills fail
closed. Integer focals take integer values (auto grids round to the
fitted integer set); joint and ragged responses have no v1 grid.
"""
function brm_prediction_grid(d::BRMDescriptor; focal, n::Integer=25,
                             fixed=(;))
    n >= 2 || error("BRM prediction: `n` must be at least 2 (got $n).")
    pairs = _brm_grid_pairs(focal)
    isempty(pairs) && error("BRM prediction: `focal` must name at least one column.")
    names = [p.first for p in pairs]
    length(unique(names)) == length(names) || error(
        "BRM prediction: duplicate focal columns in `$names`.")
    brmi = d.plan.parent
    schema = Set{Symbol}(d.columns)
    for name in names
        name in schema || error(
            "BRM prediction: focal `$name` is not a model column. Model " *
            "columns are $(Tuple(d.columns)).")
    end
    scalar_responses = Symbol[]
    for outcome in outcomes(brmi)
        outcome.response isa Symbol || error(
            "BRM prediction: joint response `$(outcome.response)` has no v1 " *
            "conditional grid; condition one scalar outcome at a time.")
        push!(scalar_responses, outcome.response)
    end
    for name in names
        name in scalar_responses && error(
            "BRM prediction: response `$name` cannot be focal; condition on " *
            "predictors, not on the response.")
    end
    fixed_map = _brm_grid_fixed(fixed)
    for name in keys(fixed_map)
        name in schema || error(
            "BRM prediction: `fixed=` column `$name` is not a model column. " *
            "Model columns are $(Tuple(d.columns)).")
        name in names && error(
            "BRM prediction: `fixed=` column `$name` is also focal.")
    end
    training = Dict{Symbol,Any}()
    for column in d.columns
        values = column_data(brmi, column)
        isnothing(values) && error(
            "BRM prediction: model column `$column` is not data-backed in " *
            "the fitted model; conditional grids need observed columns.")
        values isa AbstractVector || error(
            "BRM prediction: model column `$column` is not a vector column; " *
            "ragged schedules have no v1 conditional grid.")
        eltype(values) <: AbstractArray && error(
            "BRM prediction: model column `$column` holds non-scalar cells; " *
            "ragged schedules have no v1 conditional grid.")
        isempty(values) && error(
            "BRM prediction: model column `$column` has no training rows.")
        training[column] = values
    end
    focal_values = map(pairs) do (name, given)
        values = isnothing(given) ?
            _brm_grid_default_values(name, training[name], n) :
            _brm_grid_check_values(name, _brm_grid_values(given),
                                   training[name])
        name => collect(values)
    end
    combos = vec(collect(Iterators.product(
        (last(p) for p in focal_values)...)))
    n_grid = length(combos)
    columns = map(d.columns) do column
        values = if any(p -> p.first === column, focal_values)
            axis = findfirst(p -> p.first === column, focal_values)
            [combo[axis] for combo in combos]
        elseif haskey(fixed_map, column)
            value = fixed_map[column]
            ismissing(value) && error(
                "BRM prediction: `fixed=` value for `$column` must not be " *
                "`missing`.")
            fill(value, n_grid)
        elseif column in scalar_responses
            fill(_brm_grid_response_fill(brmi, column), n_grid)
        else
            fill(_brm_grid_typical(training[column], column), n_grid)
        end
        column => values
    end
    NamedTuple{Tuple(d.columns)}(Tuple(last(c) for c in columns))
end

# ---- outcome-anchored response means ----------------------------------------
#
# Which formula argument carries the response mean is a per-family fact, and
# each such argument must be a bare linear predictor or one supported unary
# wrapper around exactly one — verified on the raw formula argument, not on
# the lossy introspection classification (which keeps `link_fn`/`link_lp`
# but drops inline offsets such as `exp(mu + log(exposure))`).

_brm_mean_link_ok(::typeof(identity)) = true
_brm_mean_link_ok(::typeof(exp)) = true
_brm_mean_link_ok(_) = false

# Strict mean-argument classification. Returns one of
#   (; role=:linear_predictor, link_fn, link_lp)
#   (; role=:parameter, link_fn, link_param)
#   (; role=:constant, value)
#   (; role=:data, name)
# or `nothing` when the argument has no response-mean reading. A `~`-backed
# name is a linear predictor only if the descriptor carries an
# `:linear_predictor` output for it — `linear_predictors` also lists prior
# statements, so it cannot decide this. A `~`-backed name that is also
# data-backed is a chained outcome, which v1 refuses (its grid values would
# be the arbitrary response fill); any other `~`-backed name is a sampled
# parameter, which evaluates through its parameter carrier but never joins
# the logical default. `=`-backed names are refused: assignment bodies need
# their own evaluator.
function _brm_mean_argument(value, brmi::BRMI, lp_names)
    value isa Number && return (; role=:constant, value)
    if value isa NamedColumn
        payload = parent(value)
        payload isa DataColumn && return (; role=:data, name=name(value))
        if payload isa ExprColumn && getf(payload) === (~)
            if name(value) in lp_names
                return (; role=:linear_predictor, link_fn=identity,
                          link_lp=name(value))
            end
            isnothing(column_data(brmi, name(value))) || return nothing
            return (; role=:parameter, link_fn=identity,
                      link_param=name(value))
        end
        return nothing
    end
    if value isa ExprColumn
        args = getargs(value)
        length(args) == 1 || return nothing
        inner = _brm_mean_argument(only(args), brmi, lp_names)
        isnothing(inner) && return nothing
        inner.role === :linear_predictor || inner.role === :parameter ||
            return nothing
        inner.link_fn === identity || return nothing
        _brm_mean_link_ok(getf(value)) || return nothing
        if inner.role === :linear_predictor
            return (; role=:linear_predictor, link_fn=getf(value),
                      link_lp=inner.link_lp)
        end
        return (; role=:parameter, link_fn=getf(value),
                  link_param=inner.link_param)
    end
    nothing
end

_brm_observation_link_ok(lhs) =
    _data_backed_or_nothing(lhs) !== nothing
function _brm_observation_link_ok(lhs::ExprColumn)
    (getf(lhs) === mi || getf(lhs) === ragged) || return false
    args = getargs(lhs)
    isempty(args) && return false
    _brm_observation_link_ok(first(args))
end

# `~` outcomes `outcomes()` skips: link-transformed (`log(y)`), decorated
# (`mi(y)`), or otherwise non-data LHSs. Named in response-selection
# errors so a skipped LHS never reads as "no such response".
function _brm_hidden_lhs(brmi::BRMI)
    hidden = Tuple{Symbol,String}[]
    for (key, value) in pairs(brmi.operations)
        op = _named_op(value)
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rh = getargs(op, 2)
        isnothing(_as_expr_column(rh)) && continue
        isnothing(_observed_lhs_or_nothing(lhs)) || continue
        desc = lhs isa ExprColumn ? string(getf(lhs), "(…)") : "non-data LHS"
        push!(hidden, (key, desc))
    end
    hidden
end

function _brm_hidden_hint(brmi::BRMI)
    hidden = _brm_hidden_lhs(brmi)
    isempty(hidden) && return ""
    " Unrecognized observation LHSs (link-transformed or decorated; " *
    "response means cannot anchor on them — use `scale=:link` or " *
    "`target=:predictive`): " *
    join(("$k: $v" for (k, v) in hidden), ", ") * "."
end

function _brm_mean_outcome(brmi::BRMI, response::Symbol, lp_names)
    found = [o for o in outcomes(brmi) if o.response === response]
    isempty(found) && error(
        "BRM prediction: `$response` is not a scalar observed response. " *
        "Observed responses are " *
        "$(Tuple(o.response for o in outcomes(brmi)))." *
        _brm_hidden_hint(brmi))
    length(found) == 1 || error(
        "BRM prediction: response `$response` matches $(length(found)) " *
        "outcomes; conditional means need exactly one.")
    outcome = only(found)
    operations = brmi.operations
    haskey(operations, response) || error(
        "BRM prediction: response `$response` has no formula entry.")
    op = _named_op(operations[response])
    isnothing(op) && error(
        "BRM prediction: response `$response` is not a `~` outcome.")
    getf(op) === (~) || error(
        "BRM prediction: response `$response` is not a `~` outcome.")
    lhs, rh = getargs(op, 2)
    _brm_observation_link_ok(lhs) || error(
        "BRM prediction: response `$response` has a link-transformed " *
        "observation LHS, whose response mean needs the residual scale — " *
        "`exp.(mu)` would be a median, not a mean. Use `scale=:link` for " *
        "the linear predictor or `target=:predictive` for response draws.")
    raw = getargs(rh)
    classified = [_brm_mean_argument(a, brmi, lp_names) for a in raw]
    (; response, family=outcome.family, args=classified, raw)
end

# Families with a v1 response-mean map. Probed before any Stan execution
# so an unmapped family fails fast, without a compile.
const _BRM_MEAN_FAMILIES =
    (Normal, Bernoulli, BernoulliLogit, BinomialLogit, Poisson,
     NegativeBinomial2, Gamma, Exponential)
_brm_mean_family_mapped(family) =
    family isa Type && any(T -> family <: T, _BRM_MEAN_FAMILIES)

function _brm_validate_mean_args(outcome, grid)
    _brm_mean_family_mapped(outcome.family) ||
        _brm_mean_formula(outcome.family, ())
    for classified in outcome.args
        isnothing(classified) && error(
            "BRM prediction: response `$(outcome.response)` has a family " *
            "argument with no response-mean reading (a bare linear " *
            "predictor or sampled parameter, one supported `exp(...)` " *
            "wrapper around exactly one, a number, or a data column is " *
            "required). Use `scale=:link` for the linear predictor or " *
            "`target=:predictive` for response draws.")
        if classified.role === :data
            haskey(grid, classified.name) || error(
                "BRM prediction: response `$(outcome.response)` needs " *
                "data column `$(classified.name)`, which the grid does " *
                "not carry.")
        end
    end
    _brm_mean_formula(outcome.family, fill(1.0, length(outcome.args)))
    nothing
end

# Per-family response mean from evaluated mean arguments. Each method
# validates its arity; anything without a method fails closed in the
# fallback. `Normal`/`Bernoulli`/`Poisson`/`NegativeBinomial2` take their
# mean directly (Julia parameterization, which Stan lowering preserves);
# `BernoulliLogit`/`BinomialLogit` apply the family logit link on top of
# the formula-level `link_fn`; `Gamma`/`Exponential` combine shape and
# scale. Truncated/censored/interval wrappers never reach here: their
# `family` is the wrapper function, not a `Type`.
_brm_mean_formula(::Type{<:Normal}, a) =
    length(a) == 2 ? a[1] : throw(ArgumentError(
        "BRM prediction: `Normal` response mean needs 2 family arguments " *
        "(got $(length(a)))."))
_brm_mean_formula(::Type{<:Bernoulli}, a) =
    length(a) == 1 ? a[1] : throw(ArgumentError(
        "BRM prediction: `Bernoulli` response mean needs 1 family argument " *
        "(got $(length(a)))."))
_brm_mean_formula(::Type{<:BernoulliLogit}, a) =
    length(a) == 1 ? logistic.(a[1]) : throw(ArgumentError(
        "BRM prediction: `BernoulliLogit` response mean needs 1 family " *
        "argument (got $(length(a)))."))
_brm_mean_formula(::Type{<:BinomialLogit}, a) =
    length(a) == 2 ? a[1] .* logistic.(a[2]) : throw(ArgumentError(
        "BRM prediction: `BinomialLogit` response mean needs 2 family " *
        "arguments `(trials, mu)` (got $(length(a)))."))
_brm_mean_formula(::Type{<:Poisson}, a) =
    length(a) == 1 ? a[1] : throw(ArgumentError(
        "BRM prediction: `Poisson` response mean needs 1 family argument " *
        "(got $(length(a)))."))
_brm_mean_formula(::Type{<:NegativeBinomial2}, a) =
    length(a) == 2 ? a[1] : throw(ArgumentError(
        "BRM prediction: `NegativeBinomial2` response mean needs 2 family " *
        "arguments `(mu, phi)` (got $(length(a)))."))
_brm_mean_formula(::Type{<:Gamma}, a) =
    length(a) == 1 ? a[1] :
    length(a) == 2 ? a[1] .* a[2] : throw(ArgumentError(
        "BRM prediction: `Gamma` response mean needs 1 or 2 family " *
        "arguments (got $(length(a)))."))
_brm_mean_formula(::Type{<:Exponential}, a) =
    isempty(a) ? 1.0 :
    length(a) == 1 ? a[1] : throw(ArgumentError(
        "BRM prediction: `Exponential` response mean needs 0 or 1 family " *
        "arguments (got $(length(a)))."))
_brm_mean_formula(family, _) = error(
    "BRM prediction: response means for family `$family` are not mapped " *
    "in v1 (mapped: `Normal`, `Bernoulli`, `BernoulliLogit`, " *
    "`BinomialLogit`, `Poisson`, `NegativeBinomial2`, `Gamma`, " *
    "`Exponential`). Use `scale=:link` for the linear predictor or " *
    "`target=:predictive` for response draws.")

# ---- conditional draws -----------------------------------------------------
#
# `target=:mean` evaluates the linear-predictor transformed parameters on
# the reprocessed problem with `BridgeStan.param_constrain` (`include_tp`,
# no GQ, no RNG): deterministic, exact, and valid for every term the Stan
# program computes — the same validity argument `:predict`-on-reprocess
# stands on, since `reprocess` reuses the transpiled model. Output
# selection stays descriptor-driven (`role=:linear_predictor`); no
# compiler-owned name is constructed or parsed here.

function _brm_check_conditional_grid(grid)
    grid isa NamedTuple || error(
        "BRM prediction: `grid` must be a `NamedTuple` of equal-length " *
        "vectors, as built by `brm_prediction_grid` (got " *
        "$(typeof(grid))).")
    isempty(grid) && error("BRM prediction: `grid` must not be empty.")
    lengths = map(collect(pairs(grid))) do (name, column)
        column isa AbstractVector || throw(DimensionMismatch(
            "BRM prediction: grid column `$name` is not a vector."))
        length(column)
    end
    all(==(first(lengths)), lengths) || throw(DimensionMismatch(
        "BRM prediction: grid columns have unequal lengths $lengths."))
    first(lengths) > 0 || error("BRM prediction: `grid` must not be empty.")
    first(lengths)
end

function _brm_constrain_grid_lps(prob, unc_draws::AbstractMatrix,
                                 selections)
    names = BridgeStan.param_names(prob.model; include_tp=true,
                                   include_gq=false)
    resolved = map(selections) do ((name, _), coordinates_fn)
        coordinates = coordinates_fn(names)
        name => coordinates
    end
    n_draws = size(unc_draws, 1)
    out = Dict{Symbol,Matrix{Float64}}()
    for (name, coordinates) in resolved
        out[name] = Matrix{Float64}(undef, n_draws, length(coordinates))
    end
    for (i, row) in enumerate(eachrow(unc_draws))
        constrained = BridgeStan.param_constrain(
            prob.model, collect(Float64, row); include_tp=true,
            include_gq=false)
        for (name, coordinates) in resolved
            out[name][i, :] .= @view constrained[coordinates]
        end
    end
    out
end

function _brm_evaluate_mean_arguments(measured, outcome, grid, n_grid,
                                       n_draws)
    # Arguments were validated by `_brm_validate_mean_args` before any Stan
    # execution; this evaluates the same record.
    map(outcome.args) do classified
        if classified.role === :linear_predictor
            draws = measured[classified.link_lp]
            return classified.link_fn.(draws)
        elseif classified.role === :parameter
            draws = measured[classified.link_param]
            return classified.link_fn.(draws)
        elseif classified.role === :constant
            return fill(Float64(classified.value), 1, n_grid)
        else
            column = collect(Float64, grid[classified.name])
            return repeat(reshape(column, 1, n_grid), n_draws, 1)
        end
    end
end

"""
    brm_conditional_draws(d::BRMDescriptor, unc_draws::AbstractMatrix, grid;
                          problem=nothing, target=:mean, logical=:auto,
                          response=:auto, scale=:response, seed=1,
                          focal=nothing) -> NamedTuple

Evaluate fitted posterior draws on a prediction grid — the engine behind
bambi-style conditional predictions. `grid` is a `NamedTuple` of
equal-length vectors as built by [`brm_prediction_grid`](@ref).
`unc_draws` holds unconstrained posterior draws as rows, in the fitted
model's unconstrained order.

The model is reprocessed onto the grid with frozen training constants
(`brm_execute(d, :reprocess, grid)`), so `zscale`/`factor`/spline/GP and
other Julia-side transforms rebuild exactly as out-of-sample prediction
requires, and unseen fitted levels fail loudly there — never silently.

- `target=:mean` (default) returns the posterior of the response mean:
  exact transformed-parameter evaluation on the reprocessed problem, so
  no RNG and no `seed` enters. `response` selects the outcome (`:auto`
  takes the sole scalar outcome); `scale=:response` maps the evaluated
  linear predictor(s) through the outcome family's mean map (`:link`
  returns the raw linear predictor instead — the public predictor value
  the likelihood sees (for an LHS-linked declaration such as `log(mu)`
  that is already response-mapped; exactly what `brm_output_draws` with
  `role=:linear_predictor` yields on fitted draws). `logical` selects
  which linear predictor `:link` returns (`:auto` takes the outcome's
  mean predictor, or the sole linear predictor); under `scale=:response`
  it must be `:auto` or one of the outcome's mean predictors. A mean fed
  by no linear predictor (data/parameter/constant-only arguments)
  records `logical=nothing`.
- `target=:predictive` returns posterior-predictive draws via the
  existing [`brm_predictive_draws`](@ref) machinery on the reprocessed
  descriptor; `seed` seeds the first draw exactly as there, and
  `response` selects the predictive output.

`problem` optionally supplies the reprocessed problem (as returned in a
previous result) to skip re-instantiation; it must be the problem for
THIS grid — a shape mismatch fails inside Stan, never silently. `focal`
is recorded verbatim as plot metadata (a `Symbol` or collection); it
changes nothing about the evaluation.

Returns `(; grid, draws, target, logical, response, scale, focal,
descriptor, problem)` with `draws` a draws × grid-elements matrix
(`logical` is `nothing` under `target=:predictive` and for response
means no linear predictor feeds, which have no linear predictor).
Response means are outcome-anchored (see the module header): unmapped
families, link-transformed observation LHSs, and entangled mean
arguments fail closed naming the `:link` / `:predictive` alternative.
"""
function brm_conditional_draws(d::BRMDescriptor, unc_draws::AbstractMatrix,
                               grid;
                               problem=nothing, target::Symbol=:mean,
                               logical::Union{Symbol,Nothing}=:auto,
                               response::Union{Symbol,Nothing}=:auto,
                               scale::Symbol=:response, seed::Integer=1,
                               focal=nothing)
    target in (:mean, :predictive) || throw(ArgumentError(
        "BRM prediction: `target` must be `:mean` or `:predictive` (got " *
        "`$target`)."))
    scale in (:response, :link) || throw(ArgumentError(
        "BRM prediction: `scale` must be `:response` or `:link` (got " *
        "`$scale`)."))
    size(unc_draws, 1) > 0 || error(
        "BRM prediction requires at least one draw")
    n_grid = _brm_check_conditional_grid(grid)
    if target === :predictive
        d2 = brm_execute(d, :reprocess, grid)
        # Fail fast on response selection, before any compile: mirror the
        # key rule `brm_predictive_draws` applies below (keep in lockstep
        # with posterior_diagnostics.jl).
        outs = brm_outputs(d2; role=:posterior_predictive)
        isempty(outs) && error(
            "This BRM descriptor has no predictive outputs")
        keys_available = Tuple(
            isnothing(o.logical) ? o.name : o.logical for o in outs)
        length(unique(keys_available)) == length(keys_available) || error(
            "BRM prediction has ambiguous logical response names")
        selected = if response === :auto
            length(keys_available) == 1 || error(
                "BRM prediction: `response=:auto` needs exactly one " *
                "predictive output; this model has $keys_available. Pass " *
                "`response=` explicitly.")
            only(keys_available)
        else
            response in keys_available || error(
                "BRM prediction: no predictive output for response " *
                "`$response`; available: $keys_available.")
            response
        end
        prob = isnothing(problem) ? brm_execute(d2, :instantiate) : problem
        predicted = brm_predictive_draws(d2, unc_draws; problem=prob, seed)
        draws = Matrix(getproperty(predicted, selected))
        size(draws, 2) == n_grid || throw(DimensionMismatch(
            "BRM prediction: predictive output `$selected` has " *
            "$(size(draws, 2)) elements for a $n_grid-row grid."))
        return (; grid, draws, target, logical=nothing, response=selected,
                  scale, focal, descriptor=d2, problem=prob)
    end
    scalar = [o.response for o in outcomes(d.plan.parent)
              if o.response isa Symbol]
    selected_response = if response === :auto
        length(scalar) == 1 || error(
            "BRM prediction: `response=:auto` needs exactly one scalar " *
            "outcome; this model has $(Tuple(scalar)). Pass `response=` " *
            "explicitly." * _brm_hidden_hint(d.plan.parent))
        only(scalar)
    else
        response in scalar || error(
            "BRM prediction: `$response` is not a scalar observed " *
            "response. Observed responses are $(Tuple(scalar))." *
            _brm_hidden_hint(d.plan.parent))
        response
    end
    lp_names = Tuple(
        o.logical for o in brm_outputs(d; role=:linear_predictor)
        if !isnothing(o.logical))
    outcome = _brm_mean_outcome(d.plan.parent, selected_response, lp_names)
    mean_lps = Symbol[c.link_lp for c in outcome.args
                      if !isnothing(c) && c.role === :linear_predictor]
    unique!(mean_lps)
    lps = [o.logical for o in brm_outputs(d; role=:linear_predictor)]
    # A data/parameter/constant-only mean has no LP to record: it still
    # computes, with `logical=nothing`. Anything else ambiguous errors.
    selected_logical::Union{Symbol,Nothing} = if logical !== :auto
        logical
    elseif scale === :response && isempty(mean_lps)
        nothing
    elseif scale === :response && length(mean_lps) == 1
        only(mean_lps)
    elseif scale === :response
        error(
            "BRM prediction: `logical=:auto` is ambiguous; candidate " *
            "linear predictors are $(Tuple(mean_lps)). Pass `logical=` " *
            "explicitly.")
    elseif length(lps) == 1
        only(lps)
    else
        error(
            "BRM prediction: `logical=:auto` is ambiguous; candidate " *
            "linear predictors are $(Tuple(lps)). Pass `logical=` " *
            "explicitly.")
    end
    if scale === :response && !isnothing(selected_logical) &&
            selected_logical ∉ mean_lps
        error(
            "BRM prediction: response means are outcome-anchored; logical " *
            "`$selected_logical` is not a mean predictor of response " *
            "`$selected_response` (mean predictors: $(Tuple(mean_lps))). " *
            "Drop `logical=` or use `scale=:link`.")
    end
    if scale === :link && isnothing(selected_logical)
        error(
            "BRM prediction: `scale=:link` needs a linear predictor; " *
            "candidate linear predictors are $(Tuple(lps)). Pass " *
            "`logical=` explicitly.")
    end
    needed = if scale === :response
        isnothing(selected_logical) ? mean_lps :
            union([selected_logical], mean_lps)
    else
        [selected_logical]
    end
    param_needs = scale === :response ? unique!(Symbol[
        c.link_param for c in outcome.args
        if !isnothing(c) && c.role === :parameter]) : Symbol[]
    if scale === :response
        _brm_validate_mean_args(outcome, grid)
    end
    for lp in needed
        brm_output(d, lp; role=:linear_predictor)
    end
    for name in param_needs
        brm_output(d, name)
    end
    d2 = brm_execute(d, :reprocess, grid)
    prob = isnothing(problem) ? brm_execute(d2, :instantiate) : problem
    # One `map` (one closure type): separate LP/parameter maps produce
    # distinct closure types that `append!` cannot merge.
    requests = vcat([(lp, true) for lp in needed],
                    [(name, false) for name in param_needs])
    selections = map(requests) do (key, is_lp)
        (key, is_lp ? :linear_predictor : :parameter) =>
            (names -> is_lp ? brm_output_coordinates(
                d2, key, names; role=:linear_predictor) :
                brm_output_coordinates(d2, key, names))
    end
    measured = _brm_constrain_grid_lps(prob, unc_draws, selections)
    for lp in needed
        size(measured[lp], 2) == n_grid || throw(DimensionMismatch(
            "BRM prediction: linear predictor `$lp` has " *
            "$(size(measured[lp], 2)) evaluated elements for a " *
            "$n_grid-row grid."))
    end
    for name in param_needs
        size(measured[name], 2) == 1 || error(
            "BRM prediction: sampled parameter `$name` is not scalar; " *
            "vector-valued parameters have no response-mean reading. Use " *
            "`scale=:link` for the linear predictor or " *
            "`target=:predictive` for response draws.")
    end
    if scale === :link
        draws = measured[selected_logical]
        return (; grid, draws, target, logical=selected_logical,
                  response=selected_response, scale, focal, descriptor=d2,
                  problem=prob)
    end
    evaluated = _brm_evaluate_mean_arguments(measured, outcome, grid, n_grid,
                                               size(unc_draws, 1))
    means = _brm_mean_formula(outcome.family, evaluated)
    draws = means isa Number ? fill(Float64(means), size(unc_draws, 1), n_grid) :
        Matrix(Float64.(means))
    if size(draws, 1) == 1 && size(unc_draws, 1) > 1
        draws = repeat(draws, size(unc_draws, 1), 1)
    end
    size(draws) == (size(unc_draws, 1), n_grid) || throw(DimensionMismatch(
        "BRM prediction: response mean for `$(outcome.response)` has shape " *
        "$(size(draws)); expected ($(size(unc_draws, 1)), $n_grid)."))
    (; grid, draws, target, logical=selected_logical,
       response=selected_response, scale, focal, descriptor=d2,
       problem=prob)
end

# ---- comparisons -----------------------------------------------------------

function _brm_contrast_pairs(pairs::Symbol, levels)
    pairs === :all && return [(a, b) for (i, a) in enumerate(levels)
                              for b in levels[i+1:end]]
    pairs === :reference && return [(level, levels[1])
                                    for level in levels[2:end]]
    pairs === :sequential && return [(levels[i+1], levels[i])
                                     for i in 1:length(levels)-1]
    throw(ArgumentError(
        "BRM prediction: `pairs` must be `:all`, `:reference`, " *
        "`:sequential`, or an explicit collection of level pairs (got " *
        "`$pairs`)."))
end
function _brm_contrast_pairs(pairs, levels)
    items = pairs isa Pair ? [pairs] : collect(pairs)
    map(items) do pair
        a, b = pair isa Pair ? (pair.first, pair.second) :
            pair isa Union{Tuple,AbstractVector} && length(pair) == 2 ?
            (pair[1], pair[2]) : error(
            "BRM prediction: explicit `pairs` entries must be pairs or " *
            "2-tuples (wrap a single pair in a vector; got " *
            "$(repr(pair))).")
        any(==(a), levels) || error(
            "BRM prediction: contrast level $(repr(a)) is not a level of " *
            "the contrast column (levels: $(Tuple(levels))).")
        any(==(b), levels) || error(
            "BRM prediction: contrast level $(repr(b)) is not a level of " *
            "the contrast column (levels: $(Tuple(levels))).")
        (a, b)
    end
end

"""
    brm_contrast_draws(cond; by, pairs=:all, how=:diff) -> NamedTuple

Compare conditional draws across the levels of one grid column — the
draws half of bambi's `comparisons`. `cond` is a
[`brm_conditional_draws`](@ref) result (or any `NamedTuple` with `grid`
and `draws`); `by` names the contrast column (required, like bambi's
`contrast`).

Every level must own the same number of grid rows with identical values
in every other column — the factorial shape
[`brm_prediction_grid`](@ref) builds. Anything else (unequal blocks,
misaligned rows, fewer than two levels) fails closed rather than
subtracting misaligned cells.

`pairs` selects level pairs `(A, B)`: `:all` (every unordered pair),
`:reference` (each level against the first), `:sequential` (each level
against its predecessor), or an explicit collection of pairs/tuples.
`how` is `:diff` (`A - B`) or `:ratio` (`A / B`).

Returns `(; draws, labels, pairs, by, how, subgrid, n_sub, target,
logical, scale)` with `draws` a draws × (`pairs` × `n_sub`) matrix in
pairs-major order, `labels` `"A vs B"` strings, and `subgrid` the shared
within-pair grid (the first level's block rows). `target`/`logical`/
`scale` are carried from `cond` when present, `nothing` otherwise.
"""
function brm_contrast_draws(cond::NamedTuple; by::Symbol,
                            pairs=:all, how::Symbol=:diff)
    haskey(cond, :grid) && haskey(cond, :draws) || error(
        "BRM prediction: `brm_contrast_draws` needs a NamedTuple with " *
        "`grid` and `draws` (as returned by `brm_conditional_draws`).")
    grid, draws = cond.grid, cond.draws
    grid isa NamedTuple || error(
        "BRM prediction: contrast `grid` must be a `NamedTuple`.")
    draws isa AbstractMatrix || throw(DimensionMismatch(
        "BRM prediction: contrast `draws` must be a matrix."))
    n_grid = _brm_check_conditional_grid(grid)
    size(draws, 2) == n_grid || throw(DimensionMismatch(
        "BRM prediction: contrast draws have $(size(draws, 2)) columns " *
        "for a $n_grid-row grid."))
    how in (:diff, :ratio) || throw(ArgumentError(
        "BRM prediction: `how` must be `:diff` or `:ratio` (got `$how`)."))
    haskey(grid, by) || error(
        "BRM prediction: contrast column `$by` is not in the grid. Grid " *
        "columns are $(Tuple(keys(grid))).")
    column = grid[by]
    levels = unique(column)
    length(levels) >= 2 || error(
        "BRM prediction: contrast column `$by` has fewer than two levels.")
    blocks = map(levels) do level
        findall(v -> (v == level) isa Bool && v == level, column)
    end
    counts = length.(blocks)
    all(==(first(counts)), counts) || error(
        "BRM prediction: levels of `$by` own unequal grid-row counts " *
        "$counts; build the grid factorial over `$by` so every level " *
        "shares the same within-level cells.")
    n_sub = first(counts)
    for name in keys(grid)
        name === by && continue
        first_block = grid[name][blocks[1]]
        for block in blocks[2:end]
            other = grid[name][block]
            all(zip(first_block, other)) do (a, b)
                (a == b) isa Bool && a == b
            end || error(
                "BRM prediction: levels of `$by` disagree on non-contrast " *
                "column `$name`; contrasts need identical within-level " *
                "cells in every level block.")
        end
    end
    selected = _brm_contrast_pairs(pairs, levels)
    isempty(selected) && error("BRM prediction: `pairs` selected no pairs.")
    by_index = Dict{Any,Int}(level => i for (i, level) in enumerate(levels))
    mats = map(selected) do (a, b)
        left = draws[:, blocks[by_index[a]]]
        right = draws[:, blocks[by_index[b]]]
        how === :diff ? left .- right : left ./ right
    end
    labels = ["$(a) vs $(b)" for (a, b) in selected]
    subgrid = map(grid) do values
        collect(values[blocks[1]])
    end
    (; draws=reduce(hcat, mats), labels, pairs=selected, by, how, subgrid,
       n_sub, target=get(cond, :target, nothing),
       logical=get(cond, :logical, nothing), scale=get(cond, :scale, nothing))
end

# ---- slopes ----------------------------------------------------------------

"""
    brm_slope_draws(d::BRMDescriptor, unc_draws::AbstractMatrix, grid;
                    wrt, eps=1e-4, logical=:auto, scale=:response,
                    focal=nothing) -> NamedTuple

Finite-difference slopes of the conditional response mean with respect
to grid column `wrt` — the draws half of bambi's `slopes`. Evaluates
[`brm_conditional_draws`](@ref) with `target=:mean` at `grid` perturbed
to `wrt ± eps/2` (central differences) and returns the divided
difference, a draws × grid-elements matrix.

Slopes are mean-only by construction: finite differences of
posterior-predictive RNG draws are RNG noise, not slopes, so there is no
`target` keyword. `eps` (default `1e-4`, bambi's default) must be finite
and positive; `wrt` must be a real-valued non-Bool column, and the
perturbed values must round-trip through its element type exactly
(integer columns need an integer-compatible `eps`, e.g. an even
integer).

Returns `(; grid, draws, wrt, eps, target=:mean, logical, response,
scale, focal)`. `logical`/`response`/`scale` pass through to the
underlying mean evaluation; `focal` is recorded verbatim as plot
metadata.
"""
function brm_slope_draws(d::BRMDescriptor, unc_draws::AbstractMatrix, grid;
                         wrt::Symbol, eps::Real=1e-4,
                         logical::Union{Symbol,Nothing}=:auto,
                         response::Union{Symbol,Nothing}=:auto,
                         scale::Symbol=:response, focal=nothing)
    isfinite(eps) && eps > 0 || throw(ArgumentError(
        "BRM prediction: `eps` must be finite and positive (got $eps)."))
    n_grid = _brm_check_conditional_grid(grid)
    haskey(grid, wrt) || error(
        "BRM prediction: slope column `$wrt` is not in the grid. Grid " *
        "columns are $(Tuple(keys(grid))).")
    base = grid[wrt]
    tel = nonmissingtype(eltype(base))
    tel <: Real && !(tel <: Bool) || error(
        "BRM prediction: slope column `$wrt` must be real-valued (got " *
        "element type $(eltype(base))).")
    any(ismissing, base) && error(
        "BRM prediction: slope column `$wrt` contains `missing`.")
    half = Float64(eps) / 2
    function _brm_perturb(sign)
        moved = Float64.(base) .+ sign * half
        if tel <: Integer
            all(isinteger, moved) || error(
                "BRM prediction: slope column `$wrt` is integer-typed but " *
                "`eps=$eps` perturbs it off the integers; pass an " *
                "integer-compatible `eps` (e.g. an even integer).")
            return tel.(moved)
        end
        tel.(moved)
    end
    under = _brm_perturb(-1)
    over = _brm_perturb(1)
    grid_under = (; grid..., wrt => under)
    grid_over = (; grid..., wrt => over)
    below = brm_conditional_draws(d, unc_draws, grid_under; target=:mean,
                                  logical, response, scale)
    above = brm_conditional_draws(d, unc_draws, grid_over; target=:mean,
                                  logical, response, scale)
    draws = (above.draws .- below.draws) ./ Float64(eps)
    (; grid, draws, wrt, eps=Float64(eps), target=:mean,
       logical=above.logical, response=above.response, scale, focal)
end

# ---- summaries ---------------------------------------------------------------

function _brm_check_summary_probs(probs)
    all(p -> p isa Real && isfinite(p) && 0 < p < 1, probs) || error(
        "BRM prediction: summary probabilities must lie strictly between " *
        "zero and one.")
    !isempty(probs) && issorted(probs; rev=true) || error(
        "BRM prediction: summary probabilities must be nonempty and " *
        "ordered outermost first.")
    collect(Float64, probs)
end

function _brm_hdi_interval(values::AbstractVector, prob::Real)
    ordered = sort!(collect(Float64, values))
    n = length(ordered)
    width = min(n - 1, floor(Int, prob * n))
    best, best_width = 1, Inf
    for i in 1:n-width
        gap = ordered[i+width] - ordered[i]
        if gap < best_width
            best, best_width = i, gap
        end
    end
    ordered[best], ordered[best+width]
end

"""
    brm_summarize_draws(draws::AbstractMatrix; probs=[0.95], how=:hdi)
        -> Vector{NamedTuple}

Reduce a draws × elements matrix (as returned by
[`brm_conditional_draws`](@ref), [`brm_contrast_draws`](@ref), or
[`brm_slope_draws`](@ref)) to one summary row per element: `(; element,
mean, sd, q50, lower_1, upper_1, …)` with one `lower_k`/`upper_k` pair
per entry of `probs` (nonempty, strictly between zero and one,
outermost first).

`how=:hdi` (default, bambi's default) reports shortest-posterior-mass
intervals; `how=:eti` reports equal-tailed quantiles. At least two
draws and finite values are required — a single draw has no interval
and a non-finite value has no summary.
"""
function brm_summarize_draws(draws::AbstractMatrix;
                             probs::AbstractVector=[0.95], how::Symbol=:hdi)
    how in (:hdi, :eti) || throw(ArgumentError(
        "BRM prediction: `how` must be `:hdi` or `:eti` (got `$how`)."))
    levels = _brm_check_summary_probs(probs)
    size(draws, 1) >= 2 || error(
        "BRM prediction: summaries need at least two draws (got " *
        "$(size(draws, 1))).")
    interval_names = Tuple(Iterators.flatten(
        (Symbol(:lower_, k), Symbol(:upper_, k)) for k in eachindex(levels)))
    map(axes(draws, 2)) do element
        values = collect(Float64, @view draws[:, element])
        all(isfinite, values) || error(
            "BRM prediction: non-finite values at element $element have " *
            "no summary.")
        bounds = Tuple(Iterators.flatten(
            if how === :hdi
                _brm_hdi_interval(values, p)
            else
                (quantile(values, (1-p)/2), quantile(values, (1+p)/2))
            end for p in levels))
        merge((; element, mean=sum(values) / length(values),
                 sd=std(values; corrected=true), q50=quantile(values, 0.5)),
              NamedTuple{interval_names}(bounds))
    end
end

