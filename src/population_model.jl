# ==============================================================================
# population_model.jl — the exact Stan-side population program.
#
# `population_draws` zeroes a grouping factor's random-effect COORDINATES in a
# draw matrix. That is exact for every block the fitted program samples, and it
# cannot touch a block the program draws in generated quantities (an omitted or
# held-out response's block, a prior program): such a block has no coordinate.
#
# `population_model` instead re-emits the program so that each random-effect
# submodel of the selected factors returns its effects multiplied by zero. The
# submodels keep their statements and internal names, and the zeroed return
# still reads every one of them, so StanBlocks places each variable exactly as
# in the fitted program: the population program's `param_unc_names` are the
# fitted program's, and fitted draws evaluate on it unchanged. Every consumer of
# a selected block — sampled or generated — reads exactly zero.
#
# Zeroing the returned effect equals zeroing the standardized draws for every
# family in `_RANEF_FAMILIES`: each returns an effect linear in its draws
# (`C * z`, `scale * xi`, a Student-t level's `sqrt(w) * C * z`, or the centered
# effect itself). So on a block both can address, the population program and
# `population_draws` agree.
#
# An exact total-coefficient block has no zero to return: its population mean is
# `A * beta`, with `beta` the population coefficient that the program recovers
# conditionally in generated quantities. Its predictor reads `Z * (A * beta)`
# instead of `rows_dot_product(total[idx, :], Z)`, which makes the predictor a
# generated quantity, so no likelihood may read it. The program therefore holds
# out every observed response: StanBlocks drops a held-out likelihood from the
# model block, keeps its predictive and pointwise twins in generated quantities,
# and decides parameter activity on the unconditioned program, so every
# coordinate still stays where the fitted program put it.
# ==============================================================================

# The population variant of a random-effect submodel: the same statements under
# the same internal names, returning `0.0 *` the block's effects.
function _sb_population_ranef(base::StanBlocks.SlicModel, family::Symbol)
    body = _sb_plan_copy(base.model)
    at = Meta.isexpr(body, :block) ?
        findfirst(s -> Meta.isexpr(s, :return), body.args) : nothing
    isnothing(at) && error(
        "population_model: random-effect submodel `$family` has no top-level `return`")
    returned = only(body.args[at].args)
    body.args[at] = Expr(:return, :(0.0 * $returned))
    StanBlocks.SlicModel(body, deepcopy(base.data), base.mod, base.observations)
end

# The submodel a random-effect declaration calls: a spliced value (configured
# priors) or the name of one of BRM's submodels.
_sb_population_submodel(head::StanBlocks.SlicModel, _mod, _block) = head
function _sb_population_submodel(head::Symbol, mod::Module, block)
    value = isdefined(mod, head) ? getfield(mod, head) :
            isdefined(@__MODULE__, head) ? getfield(@__MODULE__, head) : nothing
    value isa StanBlocks.SlicModel || error(
        "population_model: block `$(block.binding)` is emitted by `$head`, which ",
        "does not resolve to a SLIC submodel")
    value
end
_sb_population_submodel(head, _mod, block) = error(
    "population_model: block `$(block.binding)` is emitted by an unrecognized ",
    "callee `$(repr(head))`")

# Rewrite the one read of an exact total block's totals by its predictor to the
# block's population mean.
function _sb_population_total!(body::Expr, block::TotalEffectBlock)
    zn = _sb_total_design_name(block.predictor)
    fitted = _sb_total_location(block.binding, block.group_index, zn)
    population = _sb_total_population_location(zn, _sb_total_basis_name(block.predictor),
                                                _sb_total_group_design_name(block.predictor),
                                                block.population, block.group_index)
    hits = Ref(0)
    rewrite(x) = x == fitted ? (hits[] += 1; population) :
        x isa Expr ? Expr(x.head, map(rewrite, x.args)...) : x
    for i in eachindex(body.args)
        body.args[i] = rewrite(body.args[i])
    end
    hits[] == 1 || error(
        "population_model: exact total block `$(block.binding)` is read by its ",
        "predictor $(hits[]) times in the emitted body; expected exactly once (`$fitted`)")
    nothing
end

"""
    population_model(sb::SBBRMI; groups) -> SBBRMI

The population program of `sb`: the same model, in which every random-effect
block of the grouping factors `groups` (a symbol or a collection of them)
contributes exactly zero.

Every coordinate stays where the fitted program put it, so the population
program has the fitted program's `param_unc_names` and the fitted draws evaluate
on it unchanged. Its generated quantities — predictive draws, unobserved and
held-out responses, prior-program outputs — are the population-level
predictions of the selected factors:

```julia
pop = population_model(sb; groups = :subject)
model = stan_instantiate(pop).model                  # BridgeStan model
BridgeStan.param_unc_names(model) == unc             # the fitted coordinates
BridgeStan.param_constrain(model, draw; include_tp = true, include_gq = true,
                           rng = BridgeStan.StanRNG(model, seed))
```

[`brm_descriptor`](@ref)`(pop)` mounts it like any other model, so
`brm_execute(d, :predict; draws, seed)` reads the same draws.

# Exact total-coefficient blocks

A selected factor may carry exact total-coefficient blocks
([`total_effect_blocks`](@ref), the default `total_groups = :auto` for an
independent Gaussian block). Each such predictor reads its block's population
mean `Z * (A * beta)` instead of the group totals — with group-level predictors
in the block (`TotalEffectBlock.group_columns`), each group's own mean
`Z * (A * (W[g,:] .* beta))` — where `beta` is the
population coefficient the program recovers conditionally in generated
quantities (`TotalEffectBlock.population`): one exact draw per evaluation, so
the location agrees in law with `population_draws(...; rng)`. The totals,
scales and deviations still report the fitted values.

That location is a generated quantity, so no likelihood can read it: a
population program with a selected total block holds out every observed
response (recorded in its `held_out`). Each response's likelihood leaves the
model block, while its predictive draw and pointwise log-likelihood stay in
generated quantities at the population location; the coordinates are still the
fitted program's. Its descriptor offers `:predict` and `:pointwise_loglik`, not
`:fit`.

# Compared with `population_draws`

[`population_draws`](@ref) zeroes the selected factors' coordinates in a draw
matrix. On a block the fitted program samples, the two agree: each family
returns an effect linear in its standardized draws (or, centered, the effect
itself), so zero draws and a zero return are the same evaluation; an exact total
block's location has the same law under both. They differ on a block the
program draws in GENERATED QUANTITIES (an unobserved or held-out response's
block, or every block of a prior program). That block has no coordinate for
`population_draws` to zero, so its quantities keep their re-drawn between-group
variation; here they read the population mean too. A factor whose every block
is generated, which `population_draws` refuses, is supported.

# What it reports

[`ranef_blocks`](@ref)`(pop)` reports each selected block with its fitted
family, coordinates and placement and `population = true`; `transport_draws`
and `reprocess` carry the population program like any other. Its Stan source
differs from the fitted program's (`0.0 * …` on the selected effects, the
population mean on selected totals), so its descriptor id does too.

# Refused

- A selected factor with a sum-to-zero block (`s2z_groups`).
- A factor that names no random-effect block.
"""
function population_model(sb::SBBRMI; groups)
    selected = Set{Symbol}(groups isa Symbol ? (groups,) : groups)
    isempty(selected) && throw(ArgumentError(
        "population_model: `groups` names no grouping factor"))
    s2z = filter(b -> b.group in selected, s2z_effect_blocks(sb))
    isempty(s2z) || throw(ArgumentError(
        "population_model: S2Z block(s) of $(join(sort!(collect(Set(b.group for b in s2z))), ", ")) ",
        "need recovery-aware population projection, which is not implemented yet"))
    totals = filter(b -> b.group in selected, total_effect_blocks(sb))
    # A factor with a total block need not have an ordinary block as well.
    total_factors = Set(b.group for b in totals)
    all_blocks = ranef_blocks(sb)
    blocks = vcat(_ranef_select(all_blocks, setdiff(selected, total_factors)),
                  filter(b -> any(g -> _ranef_group_matches(b, g), total_factors), all_blocks))
    unique!(b -> b.binding, blocks)
    targets = Dict(b.binding => b for b in blocks if !b.population)

    body = _sb_plan_copy(sb.model.model)
    Meta.isexpr(body, :block) || error(
        "population_model: the emitted model body is not a statement block")
    rewritten = Set{Symbol}()
    for stmt in body.args
        Meta.isexpr(stmt, :call, 3) && stmt.args[1] === :~ || continue
        target = _sb_plan_lhs_name(stmt.args[2])
        block = get(targets, target, nothing)
        isnothing(block) && continue
        target in rewritten && error(
            "population_model: block `$target` is declared twice in the emitted body")
        rhs = stmt.args[3]
        Meta.isexpr(rhs, :call) || error(
            "population_model: block `$target` is not declared by a submodel call")
        base = _sb_population_submodel(rhs.args[1], sb.model.mod, block)
        rhs.args[1] = _sb_population_ranef(base, block.family)
        push!(rewritten, target)
    end
    unmatched = setdiff(Set(keys(targets)), rewritten)
    isempty(unmatched) || error(
        "population_model: random-effect block(s) $(join(sort!(collect(unmatched)), ", ")) ",
        "have no top-level declaration in the emitted body")

    # The declaration's callee is now a value, so the block's family travels in
    # its binding record, which `ranef_blocks` and the descriptor roles read.
    bindings = copy(sb.bindings)
    for b in blocks
        old = get(bindings, b.binding, nothing)
        bindings[b.binding] = isnothing(old) ?
            (; role=:random_effect, logical=b.group, family=b.family, population=true) :
            merge(old, (; family=b.family, population=true))
    end
    rewritten_totals = filter(b -> get(sb.bindings[b.binding], :population, false) !== true, totals)
    for b in rewritten_totals
        _sb_population_total!(body, b)
        bindings[b.binding] = merge(bindings[b.binding], (; population=true))
    end
    model = StanBlocks.SlicModel(body, sb.data, sb.model.mod, sb.model.observations)
    pop = SBBRMI(sb.parent, model, sb.data, sb.preproc, copy(sb.held_out), bindings,
                 sb.source)
    isempty(rewritten_totals) && return pop
    _sb_mark_held_out(pop, _sb_held_out_inventory(pop).sources)
end
