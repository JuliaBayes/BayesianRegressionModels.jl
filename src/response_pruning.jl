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
# survives. Kernel cells (`@plate for`, `kernel(...) do`) first lose the
# observations of pruned responses and the cell statements only they reached,
# so a cell shared between kept and pruned responses stays. A shared `|ID|`
# block that keeps `a` of its `K` margins keeps its per-margin scale priors.
# Its correlation prior is the formula's own spelling at width `a` by default
# (LKJ(a, eta), the model one would write without the pruned responses);
# `marginalize = true` takes the exact marginal of the full block instead:
# LKJ(K, eta) restricted to `a` margins is LKJ(a, eta + (K - a)/2).
# ==============================================================================

# The prior address of a prior statement (`effect(...)`, `sd(...)`,
# `cor(...)`, `term_*`): its raw slot Symbols, head first.
function _brm_prune_prior_address(@nospecialize(op))
    lhs, _ = getargs(op, 2)
    Tuple(getargs(lhs))
end

# The kernel cell of a model operation, or `nothing`: its `~` operation and
# kernel call, the cell formals, the columns bound to them, and the body
# statements (`_sb_kernel_lambda_parts`, which shares the original statements).
function _brm_prune_cell(@nospecialize(raw))
    op = _as_expr_column(parent(raw))
    (isnothing(op) || getf(op) !== (~)) && return nothing
    args = getargs(op)
    length(args) == 2 || return nothing
    rhs = _as_expr_column(args[2])
    (isnothing(rhs) || getf(rhs) !== kernel) && return nothing
    dcols = getargs(rhs)
    isempty(dcols) && return nothing
    parts = _sb_kernel_lambda_parts(first(dcols))
    isnothing(parts) && return nothing
    params, body = parts
    (; op, rhs, params, positionals=dcols[2:end], body)
end

# The observation statements of a kernel cell, as the emitters read them
# (`_sb_cell_param_observed`): a top-level `formal ~ …` or `formal[j] ~ …`. Each
# is its body index, the response names it observes (the formal and the data
# column bound to it), and that column.
function _brm_prune_cell_observations(cell)
    out = Tuple{Int,Set{Symbol},Any}[]
    for (k, statement) in enumerate(cell.body)
        Meta.isexpr(statement, :call) && length(statement.args) >= 3 &&
            statement.args[1] === :~ || continue
        formal = _sb_plan_lhs_name(statement.args[2])
        i = isnothing(formal) ? nothing : findfirst(==(formal), cell.params)
        isnothing(i) && continue
        column = i <= length(cell.positionals) ?
                 _sb_kernel_positional_column(cell.positionals[i]) : nothing
        aliases = Set{Symbol}([formal])
        column isa NamedColumn && push!(aliases, name(column))
        push!(out, (k, aliases, column))
    end
    out
end

# The kernel operations of `brmi` with the responses each observes in its cell:
# one name set per observation statement, holding the cell-local formal and the
# data column bound to it. Bound response columns are added to `bound`.
function _brm_prune_kernel_observations(@nospecialize(brmi::BRMI), bound=Set{Symbol}())
    out = Dict{Symbol,Vector{Set{Symbol}}}()
    for (key, raw) in _brm_operation_entries(brmi)
        cell = _brm_prune_cell(raw)
        isnothing(cell) && continue
        observed = Set{Symbol}[]
        for (_, aliases, column) in _brm_prune_cell_observations(cell)
            push!(observed, aliases)
            column isa NamedColumn && !(parent(column) isa MissingColumn) &&
                push!(bound, name(column))
        end
        isempty(observed) || (out[key] = observed)
    end
    out
end

# The names a cell statement reads: an assignment's right-hand side, an
# observation's law and any index on its left-hand side, or the whole statement.
_brm_prune_statement_reads(statement) =
    _brm_cell_value_refs!(Set{Symbol}(), statement)
function _brm_prune_statement_reads(statement::Expr)
    Meta.isexpr(statement, :(=), 2) && statement.args[1] isa Symbol &&
        return _brm_cell_value_refs!(Set{Symbol}(), statement.args[2])
    if Meta.isexpr(statement, :call) && length(statement.args) >= 3 &&
       statement.args[1] === :~
        reads = Set{Symbol}()
        foreach(value -> _brm_cell_value_refs!(reads, value), statement.args[3:end])
        lhs = statement.args[2]
        lhs isa Expr && foreach(value -> _brm_cell_value_refs!(reads, value), lhs.args[2:end])
        return reads
    end
    _brm_cell_value_refs!(Set{Symbol}(), statement)
end

# A kernel cell without the observations of responses outside `wanted` and the
# statements only they reached. A cell statement stays when the cell's value,
# a kept observation, a statement that is neither an assignment nor an
# observation, or a cell value no statement reads (an output the cell reports)
# reaches it — except a reported output computed from what only the pruned
# observations reach, which goes with them. Reading a name reaches every
# statement that assigns or observes it, so a kept statement that reads a
# pruned response keeps its observation, exactly as a formula-level likelihood
# reading a response keeps it. A formal that only removed statements read
# leaves the call with its positional; the model values the cell captures
# follow from the remaining body. Returns `raw` itself when nothing is removed.
function _brm_prune_cell_body(@nospecialize(raw), key::Symbol, wanted)
    cell = _brm_prune_cell(raw)
    (isnothing(cell) || length(cell.params) != length(cell.positionals)) && return raw
    statements = cell.body
    indices = [k for (k, s) in enumerate(statements) if !(s isa LineNumberNode)]
    isempty(indices) && return raw
    final = last(indices)
    defines = Dict{Int,Symbol}()
    for k in indices
        s = statements[k]
        Meta.isexpr(s, :(=), 2) && s.args[1] isa Symbol && (defines[k] = s.args[1])
    end
    roots = Set{Int}([final])
    pruned = Set{Int}()
    for (k, aliases, _) in _brm_prune_cell_observations(cell)
        defines[k] = _sb_plan_lhs_name(statements[k].args[2])
        push!(isempty(intersect(aliases, wanted)) ? pruned : roots, k)
    end
    reads = Dict(k => _brm_prune_statement_reads(statements[k]) for k in indices)
    outputs = Int[]
    for k in indices
        haskey(defines, k) || (push!(roots, k); continue)
        Meta.isexpr(statements[k], :call) && continue # an observation is no output
        v = defines[k]
        any(j -> j != k && v in reads[j], indices) || push!(outputs, k)
    end
    function closure(from)
        found = Set{Int}()
        pending = collect(from)
        while !isempty(pending)
            k = pop!(pending)
            k in found && continue
            push!(found, k)
            for (j, v) in defines
                j != k && v in reads[k] && push!(pending, j)
            end
        end
        found
    end
    # What only pruned observations reach goes, and so does a reported output
    # computed from it; every other reported output stays.
    only_pruned = setdiff(closure(pruned), closure(roots))
    reported = filter(k -> isdisjoint(closure((k,)), only_pruned), outputs)
    reached = closure(union(roots, reported))
    length(reached) == length(indices) && return raw

    # The remaining body, without each removed statement's source location.
    body = Any[]
    for (k, s) in enumerate(statements)
        if s isa LineNumberNode
            next = findfirst(>(k), indices)
            (isnothing(next) || indices[next] in reached) && push!(body, s)
        elseif k in reached
            push!(body, s)
        end
    end
    # A cell left ending `key = expr; key` returns `expr` directly, as the
    # `@plate for` lowering of a terminal output does.
    kept = [k for k in indices if k in reached]
    if length(kept) >= 2 && statements[final] === key &&
       get(defines, kept[end - 1], nothing) === key &&
       Meta.isexpr(statements[kept[end - 1]], :(=)) &&
       kept[end - 1] != indices[end - 1]
        at = findlast(s -> s === key, body)
        assignment = findlast(s -> Meta.isexpr(s, :(=), 2) && s.args[1] === key, body)
        body[assignment] = statements[kept[end - 1]].args[2]
        deleteat!(body, at)
    end

    function used(ks)
        names = Set{Symbol}()
        for k in ks
            union!(names, reads[k])
            haskey(defines, k) && push!(names, defines[k])
        end
        names
    end
    before, after = used(indices), used(kept)
    keep_formal = [!(p in before) || p in after for p in cell.params]
    params = cell.params[keep_formal]
    positionals = cell.positionals[keep_formal]

    lam = first(getargs(cell.rhs))
    expression = _brm_inline_expr(lam)
    signature = expression.args[1] isa Symbol && length(params) == 1 ?
                only(params) : Expr(:tuple, params...)
    lambda = Expr(:->, signature, Expr(:block, body...))
    lam isa _BRMInlineBody && (lambda = _BRMInlineBody(lambda, _brm_inline_scope(lam)))
    rhs = ExprColumn(kernel, lambda, positionals...; getkwargs(cell.rhs)...)
    lhs = first(getargs(cell.op))
    NamedColumn(name(raw), ExprColumn(~, lhs, rhs; getkwargs(cell.op)...))
end

# Every kernel cell of `brmi` without what only its pruned observations needed.
Base.@nospecializeinfer function _brm_prune_cells(@nospecialize(brmi::BRMI), wanted)
    entries = _brm_operation_entries(brmi)
    rewritten = Pair{Symbol,Any}[key => _brm_prune_cell_body(raw, key, wanted)
                                 for (key, raw) in entries]
    all(i -> last(rewritten[i]) === last(entries[i]), eachindex(entries)) && return brmi
    BRMI((; rewritten...))
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
column of an observation, or, for one observed inside a `@plate for` /
`kernel(...)` cell, the cell's formal. It defaults to the responses whose
columns are bound in the data; name the responses for a prior program, which
binds none. It acts on the formula, so every backend (`SBBRMI`, `RKBRMI`, …)
builds the pruned model.

A cell that observes kept and pruned responses stays without the pruned
observations and the cell statements only they reached; the inputs and model
values only those statements read leave with them, and with them the
predictors, parameters and margins nothing kept reads. A statement stays when
the cell's value, a kept observation, or a cell value that no statement reads
(an output the cell reports) reaches it; a reported output computed from what
only the pruned observations reach goes with them. As at formula level, a kept
statement that reads a pruned response's value keeps that response's
observation.

A shared `|ID|` block that keeps `a` of its `K` margins keeps their scale
priors. Its correlation prior is, by default, the formula's own at width `a`:
`LKJCholesky(a, eta)` with the authored (or default, 1) `eta`, the model one
would write without the pruned responses. `marginalize = true` instead keeps the
exact marginal prior of the full block, `LKJCholesky(a, eta + (K - a)/2)` —
the law the omitted-response program samples.

Refused: a name that is not a response; a joint R2D2 budget that would lose
terms; a stratified or multi-membership shared block that loses margins.

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
    # Kernel cells lose their pruned observations first, so the closure below
    # sees only what their remaining statements read.
    brmi = _brm_prune_cells(brmi, wanted)
    program = _brm_prepare_program(brmi)
    byname = Dict(op.name => op for op in program.operations)
    names = Set(keys(byname))
    kernels = _brm_prune_kernel_observations(brmi)
    cells = Set(key for (key, raw) in _brm_operation_entries(brmi)
                if !isnothing(_brm_prune_cell(raw)))

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
            key in cells && union!(roots,
                _brm_prune_kernel_globals(byname[key].expression, names))
        end
        length(roots) == before && break
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
