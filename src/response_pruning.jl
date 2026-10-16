# ==============================================================================
# response_pruning.jl — restrict a model to the responses it keeps.
#
# Omitting a response column already keeps that response's parts out of the
# sampler: whatever no remaining likelihood reaches is re-drawn in generated
# quantities. Pruning removes those parts from the formula itself, before any
# backend sees it, so emitted programs, descriptors and descriptions show only
# the kept model.
#
# The kept model is the dependency closure of the kept responses' operations
# over BRM's prepared program, plus the prior statements addressing what
# survives. A shared `|ID|` block that keeps `a` of its `K` margins keeps its
# per-margin scale priors. Its correlation prior is the formula's own spelling
# at width `a` by default (LKJ(a, eta), the model one would write without the
# pruned responses); `marginalize = true` takes the exact marginal of the full
# block instead: LKJ(K, eta) restricted to `a` margins is LKJ(a, eta + (K - a)/2).
# ==============================================================================

# The prior address of a prior statement (`effect(...)`, `sd(...)`,
# `cor(...)`, `term_*`): its raw slot Symbols, head first.
function _brm_prune_prior_address(@nospecialize(op))
    lhs, _ = getargs(op, 2)
    Tuple(getargs(lhs))
end

# The kernel operations of `brmi` with the responses each observes in its cell:
# one name set per observation statement, holding the cell-local formal and the
# data column bound to it.
function _brm_prune_kernel_observations(@nospecialize(brmi::BRMI), bound=Set{Symbol}())
    out = Dict{Symbol,Vector{Set{Symbol}}}()
    for (key, raw) in _brm_operation_entries(brmi)
        op = _as_expr_column(parent(raw))
        (isnothing(op) || getf(op) !== (~)) && continue
        args = getargs(op)
        length(args) == 2 || continue
        rhs = _as_expr_column(args[2])
        (isnothing(rhs) || getf(rhs) !== kernel) && continue
        dcols = getargs(rhs)
        isempty(dcols) && continue
        parts = _sb_kernel_lambda_parts(first(dcols))
        isnothing(parts) && continue
        params, body = parts
        observed = Set{Symbol}[]
        for statement in body
            Meta.isexpr(statement, :call, 3) && statement.args[1] === :~ || continue
            lhs = statement.args[2]
            lhs isa Symbol || continue
            i = findfirst(==(lhs), params)
            isnothing(i) && continue
            aliases = Set{Symbol}([lhs])
            column = i + 1 <= length(dcols) ? dcols[i + 1] : nothing
            column isa NamedColumn && push!(aliases, name(column))
            push!(observed, aliases)
            column isa NamedColumn && !(parent(column) isa MissingColumn) &&
                push!(bound, first(setdiff(aliases, Set([lhs]))))
        end
        isempty(observed) || (out[key] = observed)
    end
    out
end

# Names a kernel cell body reads from the model (its free names), which the
# prepared program does not list as dependencies.
function _brm_prune_kernel_globals(@nospecialize(op), names)
    rhs = _as_expr_column(getargs(op)[2])
    parts = _sb_kernel_lambda_parts(first(getargs(rhs)))
    isnothing(parts) && return Set{Symbol}()
    params, body = parts
    refs = Set{Symbol}()
    foreach(statement -> _brm_cell_value_refs!(refs, statement), body)
    intersect(setdiff(refs, Set(params)), names)
end

"""
    prune_responses(brmi::BRMI; keep = <the data-bound responses>, marginalize = false) -> BRMI

The model `brmi` restricted to the responses `keep`: their observation
statements, every statement and term their likelihoods depend on, and the prior
statements addressing what remains. Everything else — the other responses, the
predictors, parameters and random-effect margins only they read — is not in the
returned model at all, so its emitted programs, `brm_descriptor` and
`brm_description` show only the kept model. Without pruning, omitting a
response's column keeps its parts out of the sampler but re-draws them in
generated quantities.

`keep` is a response name or a collection of them: the formula name or the data
column of an observation, or, for one observed inside a `kernel(...)` cell, the
cell's formal. It defaults to the responses whose columns are bound in the data;
name the responses for a prior program, which binds none. It acts on the
formula, so every backend (`SBBRMI`, `RKBRMI`, …) builds the pruned model.

A shared `|ID|` block that keeps `a` of its `K` margins keeps their scale
priors. Its correlation prior is, by default, the formula's own at width `a`:
`LKJCholesky(a, eta)` with the authored (or default, 1) `eta`, the model one
would write without the pruned responses. `marginalize = true` instead keeps the
exact marginal prior of the full block, `LKJCholesky(a, eta + (K - a)/2)` —
the law the omitted-response program samples.

Refused: a name that is not a response; a kept `kernel(...)` cell that also
observes a pruned response (a cell body cannot lose a statement); a joint R2D2
budget that would lose terms; a stratified or multi-membership shared block that
loses margins.

```julia
pk_tumor = prune_responses(builder(data); keep = (:pk_y, :tumor_y))
sb = SBBRMI(pk_tumor)
```
"""
function prune_responses(brmi::BRMI; keep=nothing, marginalize::Bool=false)
    responses = isnothing(keep) ? _brm_prune_bound_responses(brmi) : keep
    _brm_prune_responses(brmi, responses; marginalize)
end

# The responses whose columns the data binds: top-level observations and
# observations inside kernel cells.
Base.@nospecializeinfer function _brm_prune_bound_responses(@nospecialize(brmi::BRMI))
    program = _brm_prepare_program(brmi)
    bound = Set{Symbol}(op.name for op in program.operations if op.role === :observation)
    _brm_prune_kernel_observations(brmi, bound)
    isempty(bound) && throw(ArgumentError(
        "prune_responses: no response column is bound in the data; name the " *
        "responses to keep (`keep = ...`)"))
    bound
end

Base.@nospecializeinfer function _brm_prune_responses(@nospecialize(brmi::BRMI), keep;
                                                     marginalize::Bool=false)
    wanted = Set{Symbol}(keep isa Symbol ? (keep,) : keep)
    isempty(wanted) && throw(ArgumentError(
        "prune_responses: `keep` names no response"))
    program = _brm_prepare_program(brmi)
    byname = Dict(op.name => op for op in program.operations)
    names = Set(keys(byname))
    kernels = _brm_prune_kernel_observations(brmi)

    # A kept response roots the closure at the operation that observes it. An
    # omitted response reads as a parameter; like SBBRMI's triage, a sampled
    # leaf that nothing else reads and whose law reads a model value is one.
    read = Set{Symbol}(dep for op in program.operations for dep in op.dependencies)
    response_op(op) = op.role === :observation ||
        op.role === :parameter && !(op.name in read) && !isempty(op.dependencies)
    roots = Set{Symbol}()
    for response in wanted
        hits = Symbol[]
        haskey(byname, response) && response_op(byname[response]) && push!(hits, response)
        for op in program.operations
            op.role === :observation || continue
            lhs = first(getargs(op.expression, 2))
            _brm_observation_name(lhs) === response && push!(hits, op.name)
        end
        for (kernel_op, observed) in kernels
            any(aliases -> response in aliases, observed) && push!(hits, kernel_op)
        end
        isempty(hits) && throw(ArgumentError(
            "prune_responses: `$response` is not a response of this model"))
        union!(roots, hits)
    end

    # Close over dependencies; kernel cells and kept priors add the names they
    # read, which the prepared program does not record.
    priors = [op for op in program.operations if op.role === :prior_modifier]
    reachable = Set{Symbol}()
    while true
        reachable = _brm_reachable_operations(program, roots)
        before = length(roots)
        for key in reachable
            haskey(kernels, key) && union!(roots,
                _brm_prune_kernel_globals(byname[key].expression, names))
        end
        length(roots) == before && break
    end

    # A kept kernel cell cannot drop an observation inside its body.
    for (kernel_op, observed) in kernels
        kernel_op in reachable || continue
        for aliases in observed
            isempty(intersect(aliases, wanted)) || continue
            throw(ArgumentError(
                "prune_responses: kernel `$kernel_op` is kept, but it also observes " *
                "$(join(sort!(collect(aliases)), " / ")) inside its cell, and pruning " *
                "cannot remove a statement from a cell body. Keep that response too."))
        end
    end

    kept = Set(op.name for op in program.operations if op.name in reachable)
    _brm_prune_assemble(brmi, program, kept, priors; marginalize)
end

# The pruned BRMI: kept operations in formula order, prior statements whose
# addresses survive, and the exact marginal correlation prior of every shared
# `|ID|` block that lost margins.
Base.@nospecializeinfer function _brm_prune_assemble(@nospecialize(brmi::BRMI), program, kept, priors;
                                                    marginalize::Bool)
    predictors = Set(op.name for op in program.operations
                     if op.role === :predictor && op.name in kept)
    declarations = program.context.group_declarations
    full = _sb_collect_id_buckets(declarations)
    reduced = _sb_collect_id_buckets(filter(d -> d.predictor in predictors, declarations))
    width(bucket) = length(_sb_id_bucket_margins(bucket))
    widths = Dict(key => (K=width(bucket), a=haskey(reduced, key) ? width(reduced[key]) : 0)
                  for (key, bucket) in full)
    narrowed = Dict(first(key) => w for (key, w) in widths if 0 < w.a < w.K)
    for (key, w) in widths
        0 < w.a < w.K || continue
        any(d -> d.predictor in predictors && d.raw_group isa MultiMembershipTerm ||
                 d.predictor in predictors && d.descriptor isa Tuple,
            filter(d -> !isnothing(d.id) && d.id === first(key), declarations)) &&
            throw(ArgumentError(
                "prune_responses: shared block `|$(first(key))|` keeps $(w.a) of its $(w.K) margins, " *
                "and its stratified or multi-membership structure has no exact " *
                "marginal implemented"))
    end
    present_ids = Set(first(key) for (key, w) in widths if w.a > 0)

    keep_prior = Dict{Symbol,Any}()
    cor_seen = Set{Symbol}()
    for op in priors
        address = _brm_prune_prior_address(op.expression)
        head = first(address)
        _, rhs = getargs(op.expression, 2)
        joint = _as_expr_column(rhs)
        joint_budget = !isnothing(joint) && getf(joint) === r2d2
        if head === :cor
            id = address[2]
            id in present_ids || continue
            push!(cor_seen, id)
            w = get(narrowed, id, nothing)
            if isnothing(w)
                keep_prior[op.name] = op.expression
            elseif w.a >= 2
                keep_prior[op.name] = _brm_prune_cor_prior(op.expression, w, marginalize)
            end
        elseif head in (:sd, :ranef)
            id = address[2]
            id in present_ids || continue
            margin = length(address) >= 3 && address[3] !== :(:)
            margin && !(address[3] in predictors) && continue
            # A per-margin prior (`sd(lp, ID) ~ r2d2(...)`, an ICC form) restricts
            # with its margin; a block-wide budget does not.
            joint_budget && !margin && haskey(narrowed, id) && throw(ArgumentError(
                "prune_responses: `$(join(address, ", "))` is a joint R2D2 budget over shared block " *
                "`|$id|`, which loses margins; restricting it to the kept margins has no " *
                "exact R2D2 form"))
            keep_prior[op.name] = op.expression
        else
            # `effect(lp, coef)` (slots `lp, coef`; also `r2d2`) and term priors
            # (`term_*, term, lp, ...`) address a predictor.
            lp = startswith(String(head), "term_") ?
                 (length(address) >= 3 ? address[3] : :(:)) : head
            if lp === :(:)
                joint_budget && length(predictors) < count(op -> op.role === :predictor,
                                                            program.operations) &&
                    throw(ArgumentError(
                        "prune_responses: `$(join(address, ", "))` is a joint R2D2 budget over every " *
                        "predictor, and pruning removes some; it has no exact restriction"))
                keep_prior[op.name] = op.expression
            elseif lp in predictors
                keep_prior[op.name] = op.expression
            end
        end
    end

    operations = Pair{Symbol,Any}[]
    for (key, raw) in _brm_operation_entries(brmi)
        if key in kept
            push!(operations, key => raw)
        elseif haskey(keep_prior, key)
            push!(operations, key => NamedColumn(key, keep_prior[key]))
        end
    end
    # A narrowed block with no authored `cor` statement had the default LKJ(K, 1);
    # its marginal is spelled out, the default at width `a` needs no statement.
    for (id, w) in sort!(collect(narrowed); by=first)
        (marginalize && w.a >= 2 && !(id in cor_seen)) || continue
        key = Symbol("__effect__cor__", id)
        push!(operations, key => NamedColumn(key, ExprColumn(~, ExprColumn(effect, :cor, id),
            ExprColumn(LKJCholesky, w.a, 1.0 + (w.K - w.a) / 2))))
    end
    BRMI((; operations...))
end

# `cor(:, ID) ~ LKJCholesky(K, eta)` at the `a` kept margins: the same eta, or
# the exact marginal's.
function _brm_prune_cor_prior(@nospecialize(op), w, marginalize::Bool)
    lhs, rhs = getargs(op, 2)
    spec = _as_expr_column(rhs)
    T = isnothing(spec) ? nothing : _as_distribution_type(getf(spec))
    (!isnothing(T) && T <: LKJCholesky && length(getargs(spec)) == 2) || throw(ArgumentError(
        "prune_responses: shared block `$(join(getargs(lhs), ", "))` loses margins, and its " *
        "correlation prior is not `LKJCholesky(K, eta)`, whose restriction is exact"))
    K, eta = map(_brm_numeric_constant, getargs(spec))
    (!isnothing(K) && Int(K) == w.K && !isnothing(eta)) || throw(ArgumentError(
        "prune_responses: shared block `$(join(getargs(lhs), ", "))` declares " *
        "`LKJCholesky($(getargs(spec)[1]), $(getargs(spec)[2]))`; pruning needs numeric " *
        "`K = $(w.K)` and eta"))
    ExprColumn(~, lhs, ExprColumn(getf(spec), w.a, marginalize ? eta + (w.K - w.a) / 2 : eta))
end
