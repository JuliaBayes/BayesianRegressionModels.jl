# ==============================================================================
# prediction.jl — the two post-fit prediction modes, expressed ONCE.
#
# Every consumer that fits a BRM model eventually wants one of two things that
# the fitted posterior does not directly give it:
#
#   POPULATION-LEVEL prediction ("nore") — evaluate the model with a grouping
#       factor's random effects at their population mean (zero), i.e. the
#       prediction for an average / unobserved member of that factor.
#
#   TRANSPORTED prediction ("recov") — re-instantiate the SAME declaration at
#       new covariates and/or new group levels and reuse the FITTED draws,
#       drawing genuinely new groups' effects from the fitted covariance.
#
# Both are draw-matrix operations on the *unconstrained* parameter vector, and
# both were being hand-rolled per consumer (three independent copies in one
# downstream repo alone). Each copy re-derived two things it had no business
# knowing, and each got a different answer:
#
#   1. WHICH unconstrained coordinates carry a grouping factor's effects, and
#   2. WHETHER "set those coordinates to zero" means "population mean" at all.
#
# (2) is the dangerous one. Zeroing the random-effect coordinates gives the
# population mean ONLY under a NON-CENTERED emission, where the sampled
# coordinate is a standardised draw `z ~ std_normal()` scaled into an effect
# (`b = diag_pre_multiply(tau, L) * z`). Under a centered parameterization the
# same coordinates carry the effects themselves and a mean-zero substitution is
# simply wrong — silently, with no error and plausible numbers. Symmetrically,
# overwriting those coordinates with fresh standard normals reproduces a draw
# from the fitted covariance only under the same emission.
#
# So both facts belong HERE, next to the submodels that emit them:
#
#   * `ranef_blocks` reports every random-effect block BRM emitted, with the
#     name of the parameter that carries the standardised draw and an explicit
#     `noncentered` flag read from a table that sits beside the `@slic`
#     submodel definitions in sbimpl.jl.
#   * `ranef_coordinates` resolves that block to unconstrained coordinates BY
#     NAME and refuses (loudly) on any coordinate it cannot account for. A
#     positional splice misaligns silently the moment a template row drops a
#     covariate level; a name match cannot.
#   * `population_draws` / `transport_draws` are the two modes, both built on
#     those two primitives and neither reaching into the layout itself.
#
# The family table (`_RANEF_FAMILIES`) is the ONE place coupled to the emitted
# parameterization. Any `ranef_*` submodel missing from it is a hard error
# rather than a guess, so adding a centered emission breaks every consumer at
# the BRM boundary — loudly, once — instead of quietly returning wrong
# population-level predictions in each of them.
#
# WHAT THIS LAYER DOES NOT DO: it does not build the new model. `recov`'s model
# half is `generative_plan(plan, new_df)` (the builder form, which rebuilds the
# same declarations for genuinely new groups) or `reprocess` for a same-groups
# new grid; this file only transports DRAWS between two already-built models.
# ==============================================================================

# ---- the emission table ------------------------------------------------------
#
# One entry per `ranef_*` submodel in sbimpl.jl. Fields:
#
#   `z`      — the submodel-internal name of the STANDARDISED draw. StanBlocks
#              flattens a submodel binding `b` and internal name `z` to the Stan
#              parameter `b_z`, so the emitted parameter is `<binding>_<z>`.
#   `layout` — how Stan indexes that parameter, which fixes the coordinate NAME
#              (never a position):
#                `:group`            `<p>.<g>`                    (n_groups,)
#                `:term_group`       `<p>.<t>.<g>`   matrix[n_terms, n_groups]
#                `:group_term`       `<p>.<g>.<t>`   array[n_groups] vector[n_terms]
#                `:flat_term_group`  `<p>.<i>`, i = t + (g-1)*n_terms
#   `noncentered` — true iff the sampled coordinate is a standard normal scaled
#              into the effect. Zeroing (population mean) and re-drawing (a
#              fresh group from the fitted covariance) are BOTH valid only here.
#   `tau`    — the submodel-internal name of the per-margin FITTED between-group
#              SD vector, so the emitted carrier is `<binding>_<tau>` (a
#              `vector[n_terms]` in `ranefcoefnames` order). `:tau` for the
#              sampled-scale families, `:r2d2_tau` for the derived-scale R2D2
#              families (a transformed parameter), and `nothing` for a block
#              with no per-margin SD vector — a scalar `(1 | g)` intercept
#              (scale `exp(log_scale)`) or a stratified `gr(g, by=b)` block (one
#              `tau` per stratum). `brm_ranef_sd_coordinates` (src/descriptor.jl)
#              is the consumer; keep it in lockstep with the `@slic` bodies.
#
# Every layout below was measured against BridgeStan's `param_unc_names`, not
# inferred from the Stan type. Keep this table in lockstep with the `@slic`
# submodel definitions at the top of sbimpl.jl.
#
# THAT LOCKSTEP IS NOT FREE, AND IT HAS BROKEN ONCE. Commit 2ffac3c collapsed
# `ranef_correlated_draws` from a plate (`b_cols_z`, `matrix[K, G]`) to a flat
# reshape (`z_flat`, `vector[K*G]`) and did not touch this file, so every entry
# here kept describing the old emission. `ranef_coordinates` errored loudly on
# the bucket path -- correct behaviour, but only reachable by RUNNING a compiled
# model, which is why it survived a landing. `probe_prediction_modes.jl` now
# additionally asserts each block's `z` against the EMITTED STAN SOURCE, which
# needs no BridgeStan and catches this drift at declaration level.
const _RANEF_FAMILIES = Dict{Symbol,NamedTuple}(
    :ranef_intercept           => (; z = :xi,     layout = :group,           noncentered = true,  tau = nothing),
    :ranef_intercept_draws     => (; z = :xi,     layout = :group,           noncentered = true,  tau = nothing),
    :ranef_slope               => (; z = :xi,     layout = :group,           noncentered = true,  tau = :tau),
    :ranef_correlated          => (; z = :z_flat, layout = :flat_term_group, noncentered = true,  tau = :tau),
    :ranef_correlated_draws    => (; z = :z_flat, layout = :flat_term_group, noncentered = true,  tau = :tau),
    :ranef_correlated_draws_generic => (; z = :z_flat, layout = :flat_term_group, noncentered = true, tau = :tau),
    :ranef_correlated_draws_centered_generic => (; z = :b_cols_bc, layout = :group_term, noncentered = false, tau = :tau),
    :ranef_intercept_r2d2      => (; z = :xi,     layout = :group,           noncentered = true,  tau = nothing),
    :ranef_correlated_r2d2     => (; z = :z_flat, layout = :flat_term_group, noncentered = true,  tau = :r2d2_tau),
    :ranef_correlated_draws_r2d2 => (; z = :z_flat, layout = :flat_term_group, noncentered = true, tau = :r2d2_tau),
    :ranef_correlated_by       => (; z = :z,      layout = :group_term,      noncentered = true,  tau = nothing),
    :ranef_correlated_by_draws => (; z = :z,      layout = :group_term,      noncentered = true,  tau = nothing),
    # Centered emissions — the opt-in `SBBRMI(...; centered_groups = [:g])` path,
    # which SHIPS. They are DESCRIBED here so `ranef_blocks` can list them, and
    # refused at the operation by `_ranef_assert_noncentered`: the coordinate is
    # the effect ITSELF, so zeroing or re-drawing it would silently return a
    # different quantity. Describing is always safe; substituting is not. Layouts
    # measured against `param_unc_names` with n_terms=3, n_groups=2 so `.g.t` and
    # `.t.g` are distinguishable, against the `ranef_correlated_by` control in
    # the same capture. `tau` stays directly readable under centering (the b's
    # are centered, the scale is still sampled), so `brm_ranef_sd_coordinates`
    # resolves it regardless of the `noncentered` flag.
    :ranef_intercept_centered        => (; z = :xi, layout = :group,      noncentered = false, tau = nothing),
    :ranef_correlated_centered       => (; z = :b,  layout = :group_term, noncentered = false, tau = :tau),
    :ranef_correlated_draws_centered => (; z = :b,  layout = :group_term, noncentered = false, tau = :tau),
)

"""
    RanefBlock

One random-effect block BRM emitted, described well enough that a consumer can
address its draws without reading the generated Stan.

- `binding` — the emitted submodel binding (`:b_p_subject` for a `(… |p| g)`
  bucket, `:r_<lhs>_<g>` for a plain `(… | g)` term).
- `family` — the emitting `ranef_*` submodel (`:ranef_correlated_draws`, …).
- `group` — the grouping-factor dataframe column, or the tuple of membership
  columns for a typed `mm(...)` block. Selecting any member names that shared
  block.
- `id` — the brms-style `|ID|` bucket symbol, or `nothing` for a plain ranef.
  A bucket is shared across sub-formulas, so it is addressed as ONE block.
- `by` — the `gr(g, by=b)` stratifying column, or `nothing`.
- `levels` — the training level labels of `group`, in the index order the
  emitter assigned (`sort(unique(raw))`, or `levels()` for a
  `CategoricalVector`). `levels[g]` is the label of column `g` of the block.
- `n_terms` / `n_groups` — the block's shape.
- `z` — the emitted Stan parameter carrying the STANDARDISED draw.
- `noncentered` — see [`population_draws`](@ref). **Not always true**, and not a
  formality: `centered_groups = [:g]` emits the three `*_centered` families with
  `noncentered = false`. Those blocks are DESCRIBED normally — listing them,
  reading `levels`, resolving coordinates all work — and it is
  `population_draws` / `transport_draws` that refuse them. So a successful
  `ranef_blocks` is NOT clearance to zero or transport; branch on this flag.
- `generated` — true iff `resample_groups` moved this block's standardised draws
  to GENERATED QUANTITIES (a `reprocess(model, new_df; resample_groups = [g])`
  re-draw target; [`transport_draws`](@ref)). Such a block is re-drawn Stan-side
  per draw, so it has NO coordinate in `param_unc_names`: it is DESCRIBED (shape,
  `levels`, group count all read off the preprocessing record) but
  [`ranef_coordinates`](@ref) refuses it and `transport_draws` skips it — its
  `L` / `tau` hyperparameters are the transportable state, and they are copied by
  name. `false` for an ordinary fit (including a plain `cv_groups` build, whose
  block is still a sampled parameter).

Obtain with [`ranef_blocks`](@ref); resolve to unconstrained coordinates with
[`ranef_coordinates`](@ref).
"""
struct RanefBlock
    binding::Symbol
    family::Symbol
    group::Union{Symbol,Tuple{Vararg{Symbol}}}
    id::Union{Nothing,Symbol}
    by::Union{Nothing,Symbol}
    levels::Vector
    n_terms::Int
    n_groups::Int
    z::Symbol
    noncentered::Bool
    generated::Bool
end

Base.show(io::IO, b::RanefBlock) = print(io,
    "RanefBlock(", b.binding, " :: ", b.family, ", group=", b.group,
    isnothing(b.id) ? "" : ", id=$(b.id)",
    isnothing(b.by) ? "" : ", by=$(b.by)",
    ", ", b.n_terms, "×", b.n_groups, ", z=", b.z,
    b.generated ? ", generated" : "", ")")

# `<gname>_idx` / `<gname>__by__<bname>_idx` — the emitter's own group-index
# naming (`_sb_ensure_group_data!`, `_sb_emit_ranef_block!`), read backwards.
# The result is CROSS-CHECKED against the block's declared `n_groups` before it
# is trusted, so a naming drift surfaces as a loud mismatch rather than a
# wrong-column answer.
function _ranef_group_of_idx(idx_key::Symbol)
    s = String(idx_key)
    endswith(s, "_idx") || return nothing
    stem = s[1:(end - length("_idx"))]
    i = findfirst("__by__", stem)
    isnothing(i) ? (Symbol(stem), nothing) :
                   (Symbol(stem[1:(first(i) - 1)]), Symbol(stem[(last(i) + 1):end]))
end

# The `|ID|` bucket symbol for a binding, or `nothing`. Bucket bindings are
# `b_<id>_<group>` / `b_<id>_<group>__by__<by>` (`_sb_id_bucket_suffix`); plain
# per-target blocks are `r_<lhs>_<group>`. Structured-latent blocks
# (`hsgp(x, by=g)`) also use the `b_` prefix but carry a field name rather than a
# bucket id, so the id is only claimed when the trailing part matches the group.
function _ranef_id_of_binding(binding::Symbol, group::Symbol, by::Union{Nothing,Symbol})
    s = String(binding)
    startswith(s, "b_") || return nothing
    tail = isnothing(by) ? "_$(group)" : "_$(group)__by__$(by)"
    endswith(s, tail) || return nothing
    id = s[3:(end - length(tail))]
    isempty(id) ? nothing : Symbol(id)
end
_ranef_id_of_binding(::Symbol, ::Any, ::Union{Nothing,Symbol}) = nothing

# A zerocorr `(t1 + t2 || g)` expansion gives each scalar margin its own
# synthetic `g__nocor__N` index. That index is an emission identity, not a new
# raw column — but a user's own data column may literally carry such a name,
# so the suffix alone never decides. The emitted `:group_index`
# preprocessing record carries the fit-time level LABELS in order, and those
# labels are the origin record: everything downstream (level vectors,
# label-aligned transport) keys off labels, not off the membership coding.
# Two coexisting columns can share one coding while carrying different
# labels, so membership equality must not arbitrate — only exact ordered
# label equality identifies the column the emitter coded.
function _ranef_levels_equal(levels, recorded)
    isequal(collect(levels), collect(recorded))
end

# Formula-level grouping evidence, consulted ONLY for genuine ties (coexisting
# columns with elementwise-identical values): the set of names plain `|`
# groupings actually write, and the set of `||` zerocorr bases. A tie name
# claimed by a plain grouping is that literal block; claimed only by a
# doublepipe base it is that synthetic block. Anything else fails closed.
# Per-block formula attribution is not retained, so this is deliberately
# global: a tie claimed on both sides stays ambiguous and errors.
function _ranef_formula_group_roles(brmi)
    plain = Set{Symbol}()
    piped = Set{Symbol}()
    for key in keys(brmi.operations)
        node = brmi.operations[key]
        node isa NamedColumn || continue
        _ranef_collect_group_roles!(plain, piped, parent(node))
    end
    plain, piped
end

function _ranef_collect_group_roles!(plain, piped, x)
    x isa ExprColumn || return nothing
    f = getf(x)
    if f === doublepipe || f === Base.:|
        args = getargs(x)
        !isempty(args) && args[end] isa NamedColumn &&
            push!(f === doublepipe ? piped : plain, name(args[end]))
    end
    for a in getargs(x)
        _ranef_collect_group_roles!(plain, piped, a)
        a isa NamedColumn && _ranef_collect_group_roles!(plain, piped, parent(a))
    end
    nothing
end

function _ranef_matches_index(col::AbstractVector, idx_val::AbstractVector)
    length(col) == length(idx_val) || return false
    levels = collect(_sb_fit_levels(col))
    code = Dict{Any,Int}(level => i for (i, level) in enumerate(levels))
    all(i -> get(code, col[i], 0) == idx_val[i], eachindex(col, idx_val))
end
_ranef_matches_index(::Any, ::Any) = false

const _RANEF_NOCOR_RE = r"^(.+)__nocor__[0-9]+$"

function _ranef_raw_group(group::Symbol, brmi, idx_val, recorded_levels)
    value = String(group)
    parsed = match(_RANEF_NOCOR_RE, value)
    isnothing(parsed) && return group
    base = Symbol(only(parsed.captures))
    rec = collect(recorded_levels)
    base_col = column_data(brmi, base)
    literal_col = column_data(brmi, group)
    base_ok = !isnothing(base_col) && base_col isa AbstractVector &&
        _ranef_levels_equal(_sb_fit_levels(base_col), rec)
    literal_ok = !isnothing(literal_col) && literal_col isa AbstractVector &&
        _ranef_levels_equal(_sb_fit_levels(literal_col), rec)
    # Initialized up front so no branch-analysis slip can leave the
    # fail-closed error below referencing an unbound reason.
    reason = "no raw data column carries its recorded level labels"
    if base_ok && !literal_ok
        return base
    elseif literal_ok && !base_ok
        return group
    elseif base_ok && literal_ok
        # A resampled Stan-expression index carries no coding to verify, so it
        # joins the identical-values tie: the formula text decides, exactly as
        # at fit time (replay preserves the formula by construction).
        concrete = !(idx_val isa StanBlocks.StanExpr)
        base_repro = concrete && _ranef_matches_index(base_col, idx_val)
        literal_repro = concrete && _ranef_matches_index(literal_col, idx_val)
        if base_repro && !literal_repro
            return base
        elseif literal_repro && !base_repro
            return group
        elseif (base_repro && literal_repro) || !concrete
            # Elementwise-identical columns (or an unverifiable resample
            # index): identical values at training time do not make distinct
            # group names interchangeable for named targeting or new-data
            # replay. The formula text is the only remaining origin evidence.
            plain, piped = _ranef_formula_group_roles(brmi)
            in_plain = group in plain
            from_pipe = base in piped
            in_plain && !from_pipe && return group
            from_pipe && !in_plain && return base
            reason = "identical values leave the origin unrecoverable from the formula"
        else
            reason = "neither candidate's coding reproduces the emitted index"
        end
    end
    error("BRM prediction: grouping index `$group` is ambiguous ($reason). ",
          "The recorded emission labels are $rec; refusing to guess between ",
          "`$base` and literal `$group` — rename one of the coexisting columns ",
          "so the grouping origin is unique.")
end

_ranef_plan(sb::SBBRMI) = generative_plan(sb)
_ranef_plan(plan::GenerativePlan) = plan

"""
    ranef_blocks(model) -> Vector{RanefBlock}

Every random-effect block `model` emitted, in declaration order. `model` is an
[`SBBRMI`](@ref) or a [`GenerativePlan`](@ref).

This is the authoritative answer to "which parameters carry grouping factor
`g`'s effects, and is the emission non-centered?" — read off the emitted
declarations, not off the `@brm` source and not off the generated Stan text.

# Coverage and the loud edges

Recognised families are the `ranef_*` submodels in sbimpl.jl. A declaration
whose family *name* begins with `ranef_` but is absent from the emission table
raises: that is a new parameterization no consumer of this API can be assumed
to handle, and the whole point of routing through here is that such a change
cannot pass silently.

Structured-latent blocks whose prior is `:iid_normal` or an element-wise
non-normal (`_sb_emit_block_draw!`) emit an ordinary `std_normal` / dist
declaration rather than a `ranef_*` submodel, and are deliberately NOT reported
— there is nothing in the emission that distinguishes them from any other
vector prior. `hsgp(x, by=g)` and any other structured field whose prior is
`:correlated_normal` goes through `ranef_correlated_draws` and IS reported,
with `id === nothing`.

# Example

```julia
sb = SBBRMI(builder(df); mod=@__MODULE__)
for b in ranef_blocks(sb)
    println(b.group, " → ", b.z, "  ", b.n_terms, "×", b.n_groups)
end
```
"""
function ranef_blocks(model)
    plan = _ranef_plan(model)
    brmi = plan.parent
    data = plan.data
    out = RanefBlock[]
    seen = Set{Symbol}()
    for d in plan.declarations
        d.role === :prior || continue
        binding = get(plan.bindings, d.target, nothing)
        fam = isnothing(binding) || isnothing(binding.family) ? d.family : binding.family
        fam isa Symbol || continue
        spec = get(_RANEF_FAMILIES, fam, nothing)
        if isnothing(spec)
            startswith(String(fam), "ranef_") && error(
                "BRM prediction: declaration `$(d.target) ~ $(fam)(…)` is a ",
                "random-effect submodel with no entry in `_RANEF_FAMILIES`. Its ",
                "parameterization is unknown here, so population-level and ",
                "transported prediction cannot be derived for it. Add it to the ",
                "table in src/prediction.jl (with its measured coordinate layout ",
                "and its `noncentered` flag) in the same change that emits it.")
            continue
        end
        d.target in seen && continue
        haskey(d.keywords, :group_idx) || error(
            "BRM prediction: `$(d.target) ~ $(fam)(…)` carries no `group_idx=` ",
            "keyword; the grouping factor cannot be identified.")
        idx_key = d.keywords.group_idx
        idx_key isa Symbol || error(
            "BRM prediction: `$(d.target) ~ $(fam)(…)` has a non-symbolic ",
            "`group_idx=$(idx_key)`; expected a data key.")
        mm_entry = get(plan.preproc, idx_key, nothing)
        if mm_entry isa PreprocEntry && mm_entry.kind === :multi_membership
            group = Tuple(mm_entry.raw_ref.groups)
            # Multi-membership members are user-written column names straight
            # from the `mm(...)` call — origin metadata, never synthesis — so
            # they are used verbatim: no `__nocor__` resolution applies, not
            # even to a literally suffix-bearing member. The binding must still
            # be explicit: without this the block below reads an undefined (or
            # stale) `raw_group` from the ordinary branch.
            raw_group = group
            by = nothing
            generated = false
            levels = collect(mm_entry.const_.levels)
            n_groups = _ranef_data_int(data, mm_entry.const_.n_groups_key,
                                       d.target, fam)
        else
            parsed = _ranef_group_of_idx(idx_key)
            isnothing(parsed) && error(
                "BRM prediction: cannot read a grouping factor out of group index ",
                "key `$(idx_key)` for `$(d.target)`.")
            group, by = parsed
            idx_val = _ranef_data_vec(data, idx_key, d.target, fam)
            # Origin recovery from the `:group_index` record is needed for
            # synthetic `__nocor__` names (any index) and for resampled
            # Stan-expression indices (any name — the coding is Stan-side).
            # Every other shape — plain, bucket, and stratified `gr(g, by=b)`
            # concrete indices, the last of which carries no such record —
            # keeps the historical direct answer.
            raw_group = if isnothing(match(_RANEF_NOCOR_RE, String(group))) &&
                    !(idx_val isa StanBlocks.StanExpr)
                group
            else
                gi_entry = get(plan.preproc, idx_key, nothing)
                (gi_entry isa PreprocEntry && gi_entry.kind === :group_index) || error(
                    "BRM prediction: `$(d.target) ~ $(fam)(…)` group index `$(idx_key)` has no ",
                    "`:group_index` preprocessing record; its grouping origin cannot be recovered.")
                _ranef_raw_group(group, brmi, idx_val, gi_entry.const_.levels)
            end
            raw = column_data(brmi, raw_group)
            isnothing(raw) && error(
                "BRM prediction: grouping factor `$raw_group` (from `$(idx_key)`) is not ",
                "a raw data column of this model.")
            levels = collect(_sb_fit_levels(raw))
            if idx_val isa StanBlocks.StanExpr
                # RESAMPLE target: `resample_groups` marked this factor's index
                # with `maybecv(...)`, so it is a Stan-side expression (not a data
                # vector) and the block's standardised draws were flipped to
                # generated quantities. Its group count is NOT recoverable by
                # iterating the expression — read it off the `:group_index`
                # preprocessing record required above, and flag the block
                # `generated` so `ranef_coordinates` / `transport_draws` treat
                # it as a re-drawn non-parameter rather than looking for
                # coordinates that no longer exist in `param_unc_names`.
                generated = true
                n_groups = _ranef_data_int(data, gi_entry.const_.n_groups_key,
                                           d.target, fam)
            else
                generated = false
                # CONTROL. `levels` is reconstructed from the training column with
                # the same fit/apply split the emitter used; the emitted `n_groups`
                # is the count the Stan program was built with. If those disagree,
                # the group name derived from `idx_key` is wrong (or the level
                # coding drifted).
                n_groups = if haskey(d.keywords, :n_groups) && d.keywords.n_groups isa Symbol
                    ng_key = d.keywords.n_groups
                    if ng_key === Symbol(d.target, :_n_g)
                        # cv-contagious sizing uses a Stan-side local, not a data key.
                        maximum(idx_val)
                    else
                        _ranef_data_int(data, ng_key, d.target, fam)
                    end
                else
                    maximum(idx_val)
                end
            end
        end
        length(levels) == n_groups || error(
            "BRM prediction: grouping factor `$raw_group` has $(length(levels)) ",
            "training levels but block `$(d.target)` was emitted with ",
            "n_groups = $n_groups. The block cannot be addressed by level.")
        n_terms = if haskey(d.keywords, :n_terms) && d.keywords.n_terms isa Symbol
            _ranef_data_int(data, d.keywords.n_terms, d.target, fam)
        else
            1
        end
        push!(seen, d.target)
        push!(out, RanefBlock(d.target, fam, raw_group,
                              _ranef_id_of_binding(d.target, raw_group, by), by,
                              levels, n_terms, n_groups,
                              Symbol(d.target, :_, spec.z), spec.noncentered,
                              generated))
    end
    out
end

_ranef_data_int(data, key, target, fam) = begin
    haskey(data, key) || error(
        "BRM prediction: `$target ~ $fam(…)` references data key `$key`, which ",
        "the model's data does not carry.")
    v = data[key]
    v isa Integer || error(
        "BRM prediction: data key `$key` (for `$target`) is $(typeof(v)), not an ",
        "Integer size.")
    Int(v)
end

_ranef_data_vec(data, key, target, fam) = begin
    haskey(data, key) || error(
        "BRM prediction: `$target ~ $fam(…)` references data key `$key`, which ",
        "the model's data does not carry.")
    data[key]
end

# The Stan coordinate name of entry (t, g) of a block. Names, never positions —
# the flat ORDER of the unconstrained vector is BridgeStan's business and is
# never assumed here.
function _ranef_coord_name(b::RanefBlock, layout::Symbol, t::Int, g::Int)
    layout === :group           && return "$(b.z).$(g)"
    layout === :term_group      && return "$(b.z).$(t).$(g)"
    layout === :group_term      && return "$(b.z).$(g).$(t)"
    layout === :flat_term_group && return "$(b.z).$(t + (g - 1) * b.n_terms)"
    error("BRM prediction: unknown coordinate layout `:$layout` for `$(b.z)`.")
end

"""
    ranef_coordinates(block::RanefBlock, unc_names) -> Matrix{Int}

Resolve `block` to positions in the unconstrained parameter vector, returned as
an `n_terms × n_groups` matrix of 1-based indices into `unc_names`.

`unc_names` is the compiled model's unconstrained parameter names — BridgeStan's
`param_unc_names(model)`, in its order. Nothing here depends on that order: each
coordinate is looked up by its Stan NAME. Any coordinate the block expects and
`unc_names` does not carry is an error listing the missing names, because that
is exactly the shape a silent misalignment takes (a template that dropped a
covariate level, or a model rebuilt with a different parameterization).

    ranef_coordinates(blocks, unc_names) -> Vector{Matrix{Int}}

maps over several blocks.
"""
function ranef_coordinates(block::RanefBlock, unc_names)
    block.generated && error(
        "BRM prediction: block `$(block.binding)` (group `$(block.group)`) was ",
        "moved to generated quantities by `resample_groups`; its draws are ",
        "re-generated Stan-side, so it has no unconstrained coordinates to ",
        "resolve against `param_unc_names`. Transport copies its `L` / `tau` ",
        "hyperparameters by name and lets the target re-draw the effects; ",
        "`transport_draws` skips such a block rather than calling this.")
    layout = _RANEF_FAMILIES[block.family].layout
    pos = _ranef_name_positions(unc_names)
    out = Matrix{Int}(undef, block.n_terms, block.n_groups)
    missing_names = String[]
    for g in 1:block.n_groups, t in 1:block.n_terms
        nm = _ranef_coord_name(block, layout, t, g)
        i = get(pos, nm, 0)
        i == 0 ? push!(missing_names, nm) : (out[t, g] = i)
    end
    isempty(missing_names) || error(
        "BRM prediction: block `$(block.binding)` (group `$(block.group)`, ",
        "$(block.n_terms)×$(block.n_groups)) expects unconstrained coordinates ",
        "that this model does not have: ",
        join(first(missing_names, 8), ", "),
        length(missing_names) > 8 ? " … ($(length(missing_names)) total)" : "",
        ". The draw matrix and the model do not describe the same emission.")
    out
end

ranef_coordinates(blocks::AbstractVector{RanefBlock}, unc_names) =
    [ranef_coordinates(b, unc_names) for b in blocks]

_ranef_name_positions(unc_names) =
    Dict{String,Int}(String(n) => i for (i, n) in enumerate(unc_names))

# Resolve a user-supplied group selector to the blocks it names, erroring on a
# selector that matches nothing (a typo'd factor must not silently do nothing).
_ranef_group_symbols(group::Symbol) = (group,)
_ranef_group_symbols(group::Tuple) = group
_ranef_group_symbols(group) = (group,)
_ranef_group_matches(block::RanefBlock, group::Symbol) =
    group in _ranef_group_symbols(block.group)

function _ranef_select(blocks, groups)
    wanted = groups isa Symbol ? [groups] : collect(groups)
    out = RanefBlock[]
    for g in wanted
        hits = filter(b -> _ranef_group_matches(b, g), blocks)
        isempty(hits) && error(
            "BRM prediction: no random-effect block on grouping factor `$g`. ",
            "This model's grouped blocks are: ",
            isempty(blocks) ? "(none)" :
                join(unique(string(b.group) for b in blocks), ", "), ".")
        append!(out, hits)
    end
    unique!(b -> b.binding, out)
end

function _ranef_assert_noncentered(b::RanefBlock, what::AbstractString)
    b.noncentered || error(
        "BRM prediction: $what requires a NON-CENTERED emission, but block ",
        "`$(b.binding)` (`$(b.family)`) is centered — its coordinates carry the ",
        "effects themselves, not a standardised draw. Substituting them would ",
        "silently produce a different quantity.")
    nothing
end

_ranef_check_draws(draws, unc_names) =
    size(draws, 2) == length(unc_names) || error(
        "BRM prediction: draw matrix has $(size(draws, 2)) columns but the model ",
        "has $(length(unc_names)) unconstrained coordinates. `draws` must be ",
        "draws × coordinates, in `unc_names` order.")

"""
    population_draws(model, draws, unc_names; groups) -> Matrix{Float64}

Population-level ("no random effects") draws: a copy of `draws` with every
random-effect coordinate of the named grouping factors set to zero, so the model
evaluates at those factors' population mean.

- `model` — the `SBBRMI` / `GenerativePlan` the draws were fitted under.
- `draws` — draws × coordinates, in `unc_names` order (an unconstrained
  posterior draw matrix).
- `unc_names` — that model's `param_unc_names`.
- `groups` — a grouping-factor symbol or a collection of them. A factor naming
  no emitted block is an error.

Every other coordinate — population coefficients, residual scales, and the
random effects' own `L` / `tau` hyperparameters — is left untouched: only the
per-group standardised draws are zeroed.

# Why zeroing is the population mean here, and only here

BRM emits random effects non-centered: the sampled coordinate is
`z ~ std_normal()` and the effect is `b = diag_pre_multiply(tau, L) * z`. Zero
`z` is therefore exactly `b = 0`, the population mean. Under a centered emission
the same coordinates would hold the effects, and this substitution would quietly
return something else — so the emission is asserted per block rather than
assumed ([`ranef_blocks`](@ref)).

A brms-style `(… |ID| g)` bucket is shared across sub-formulas and is zeroed as
one block: selecting its grouping factor zeroes that factor's effects in *every*
sub-formula that shares the bucket, which is what "population-level for `g`"
means.

# Example

```julia
unc  = BridgeStan.param_unc_names(stan_model)
pop  = population_draws(sb, draws, unc; groups = :subject)
```
"""
function population_draws(model, draws::AbstractMatrix, unc_names; groups)
    _ranef_check_draws(draws, unc_names)
    blocks = _ranef_select(ranef_blocks(model), groups)
    out = Matrix{Float64}(draws)
    for b in blocks
        _ranef_assert_noncentered(b, "population-level prediction")
        out[:, vec(ranef_coordinates(b, unc_names))] .= 0.0
    end
    out
end

"""
    transport_draws(from, to, draws, unc_from, unc_to;
                    resample = (), rng = Random.default_rng()) -> Matrix{Float64}

Transport fitted `draws` from model `from` onto model `to` — the same `@brm`
declaration re-instantiated at new covariates and/or new group levels — so the
fit can be evaluated on new data without refitting.

- `from` / `to` — `SBBRMI` / `GenerativePlan` values. `to` is typically
  `generative_plan(plan, new_df)` (the builder form, which rebuilds the same
  declarations for genuinely new groups), `reprocess(sb, new_df)` when the
  groups are unchanged, or `reprocess(sb, new_df; resample_groups = [g])` — see
  the resample-target paragraph below.
- `draws` — draws × coordinates in `unc_from` order; `unc_from` / `unc_to` are
  the two models' `param_unc_names`.
- `resample` — grouping factors whose EXISTING levels should also be re-drawn
  rather than reused (leave-all-out / out-of-sample semantics). New levels are
  always drawn fresh regardless.
- `rng` — the source for those fresh draws.

Returns a draws × `length(unc_to)` matrix aligned to `to`.

# The alignment contract

Every coordinate of `to` is accounted for exactly once, by NAME:

1. A random-effect coordinate whose group LEVEL also exists in `from` (and whose
   factor is not in `resample`) is copied from that level's coordinate — by
   level label, so a reordering, an insertion, or a dropped level cannot
   misalign it.
2. A random-effect coordinate for a level `from` does not have, or for a factor
   in `resample`, is drawn `N(0, 1)`. Under BRM's non-centered emission that is
   exactly a draw of that group's effect from the fitted covariance, because the
   `L` / `tau` hyperparameters are copied per draw by rule 3.
3. Every other coordinate must exist in `from` under the same name and is copied.

Anything that does not fit those three rules raises. In particular a coordinate
of `to` that is absent from `from` and is not a random effect, a random-effect
block `to` has and `from` does not, and a block whose term count changed, are all
errors — those are the cases where a positional splice would have produced
plausible, wrong numbers.

Coordinates that exist in `from` but not in `to` (a dropped group level) are
simply not carried over.

# Resample targets (`reprocess(sb, new_df; resample_groups = [g])`)

`resample_groups` moves factor `g`'s standardised draws to the target's
GENERATED QUANTITIES, so they are re-drawn Stan-side per draw and are NOT in
`to`'s `param_unc_names`. `ranef_blocks(to)` reports such a block with
`generated = true`; `transport_draws` SKIPS it — there is no coordinate to
transport. What IS transported is everything that remains a parameter: the
population coefficients and, critically, the resampled factor's own `L` / `tau`
covariance hyperparameters, all copied by NAME (rule 3). The net effect is exact
out-of-sample semantics — the target re-draws `g`'s per-level effects from the
FITTED covariance, because that covariance is transported. `from`'s per-level
draws for `g` are dropped (they are `from`-only coordinates), which is correct:
the new levels are not the fitted ones. This is the intended consumption path
for a resample target; a positional splice of the fitted levels' draws would be
the exact wrong answer.

# Example

```julia
new_plan = generative_plan(plan, new_df)          # new subjects, new schedule
new_sb   = ...                                     # its compiled model
moved    = transport_draws(sb, new_plan, draws,
                           unc_old, unc_new)       # new subjects drawn fresh
loo      = transport_draws(sb, new_plan, draws,
                           unc_old, unc_new; resample = :subject)
```
"""
function transport_draws(from, to, draws::AbstractMatrix, unc_from, unc_to;
                         resample = (), rng::Random.AbstractRNG = Random.default_rng())
    _ranef_check_draws(draws, unc_from)
    resample_set = Set{Symbol}(resample isa Symbol ? (resample,) : resample)
    blocks_from = ranef_blocks(from)
    blocks_to   = ranef_blocks(to)
    known_groups = Set{Symbol}()
    for b in blocks_from
        union!(known_groups, _ranef_group_symbols(b.group))
    end
    for g in resample_set
        g in known_groups || error(
            "BRM prediction: `resample = $g` names no random-effect block of the ",
            "source model. Its grouped blocks are: ",
            isempty(blocks_from) ? "(none)" :
                join(unique(string(b.group) for b in blocks_from), ", "), ".")
    end
    by_key = Dict((b.id, b.group, b.by) => b for b in blocks_from)
    pos_from = _ranef_name_positions(unc_from)

    n_draws = size(draws, 1)
    out = Matrix{Float64}(undef, n_draws, length(unc_to))
    # `plan[j]` is 0 for "draw fresh", otherwise the source coordinate to copy.
    plan_idx = zeros(Int, length(unc_to))
    claimed = falses(length(unc_to))

    for bt in blocks_to
        # A resample target's block is re-drawn in the target's generated
        # quantities (see the resample-target paragraph above): it has no
        # coordinate in `unc_to`, so there is nothing to align. Its `L` / `tau`
        # hyperparameters and every population coordinate remain parameters and
        # are copied by name in the fall-through below.
        bt.generated && continue
        key = (bt.id, bt.group, bt.by)
        bf = get(by_key, key, nothing)
        isnothing(bf) && error(
            "BRM prediction: target model has random-effect block ",
            "`$(bt.binding)` (group `$(bt.group)`", isnothing(bt.id) ? "" : ", id `$(bt.id)`",
            ") with no counterpart in the source model. The two models do not ",
            "share a declaration; fitted draws cannot be transported onto it.")
        bf.family === bt.family || error(
            "BRM prediction: block `$(bt.binding)` is emitted by `$(bt.family)` in ",
            "the target model but by `$(bf.family)` in the source. The ",
            "parameterization changed; draws cannot be transported.")
        bf.n_terms == bt.n_terms || error(
            "BRM prediction: block `$(bt.binding)` has $(bt.n_terms) terms in the ",
            "target model but $(bf.n_terms) in the source. The design changed; ",
            "draws cannot be transported.")
        _ranef_assert_noncentered(bt, "transported prediction")
        coords_to = ranef_coordinates(bt, unc_to)
        coords_from = ranef_coordinates(bf, unc_from)
        level_pos = Dict(l => g for (g, l) in enumerate(bf.levels))
        fresh = any(g -> g in resample_set, _ranef_group_symbols(bt.group))
        for g in 1:bt.n_groups
            gf = fresh ? 0 : get(level_pos, bt.levels[g], 0)
            for t in 1:bt.n_terms
                j = coords_to[t, g]
                claimed[j] = true
                plan_idx[j] = gf == 0 ? 0 : coords_from[t, gf]
            end
        end
    end

    unresolved = String[]
    for (j, nm) in enumerate(unc_to)
        claimed[j] && continue
        i = get(pos_from, String(nm), 0)
        i == 0 ? push!(unresolved, String(nm)) : (plan_idx[j] = i)
        claimed[j] = i != 0
    end
    isempty(unresolved) || error(
        "BRM prediction: the target model has coordinates the source model does ",
        "not, and they are not random effects: ",
        join(first(unresolved, 8), ", "),
        length(unresolved) > 8 ? " … ($(length(unresolved)) total)" : "",
        ". Transporting would have to invent them; refusing.")

    for j in 1:length(unc_to)
        if plan_idx[j] == 0
            Random.randn!(rng, view(out, :, j))
        else
            @views out[:, j] .= draws[:, plan_idx[j]]
        end
    end
    out
end

"""
    term_draws(d::BRMDescriptor, draws, unc_names;
               predictor::Symbol, term::Symbol, to::Symbol = :zero) -> Matrix{Float64}

Ablate ONE named linear-predictor term in fitted UNCONSTRAINED draws: a copy of
`draws` with that term's basis-weight coordinates set to zero, so a subsequent
`brm_execute(d, :predict; draws = …)` regenerates every descriptor output with
the term's contribution removed and all other structure — every other term, the
population coefficients, the residual scales, the term's own length-scale and
marginal sd — held at its fitted value.

This is the structural counterpart to [`population_draws`](@ref): that zeroes a
grouping factor's random effects (a *level* intervention); this zeroes one
formula *term*. Both are draw-matrix operations on the unconstrained parameter
vector, resolved BY NAME, and neither reaches into the constrained space or the
compiler-owned coordinate order.

- `d` — the [`BRMDescriptor`](@ref) the draws were fitted under (the same object
  passed to `brm_execute(d, :predict; …)`).
- `draws` — draws × coordinates, in `unc_names` order: the unconstrained
  posterior draw matrix `:predict` consumes.
- `unc_names` — that model's `param_unc_names` (BridgeStan), in its order. The
  carrier is looked up by Stan name, so nothing here depends on that order.
- `predictor` — the linear-predictor name (`:log_F`).
- `term` — the public term label BRM derives from the formula
  (`:hsgp_op_log_dose`), the same labels [`brm_term_coordinates`](@ref) uses.
- `to` — the target value for the term's carrier. Only `:zero` is supported.

Returns a fresh draws × `length(unc_names)` matrix; `draws` is left untouched.

# Why zeroing removes exactly the term, and only for `hsgp`

An ungrouped `hsgp(x; …)` emits `beta_raw ~ std_normal()` and contributes the
summand `PHI * (sqrt_spd .* beta_raw)` to its linear predictor (`_sb_hsgp` /
`_sb_hsgp_aniso`, sbimpl.jl). `beta_raw` is a raw standard normal — an
unconstrained-domain parameter whose unconstrained coordinate IS the value
multiplied into the predictor — so setting it to zero removes that summand
exactly, whatever the fitted length-scale (`rho`) and marginal sd (`sigma`).

No other emitted term shares this "zero the raw carrier ⇒ remove the
contribution" invariant, so `term_draws` admits ONLY the ungrouped-`hsgp`
basis-weight carrier and fails closed on everything else. It resolves the
carrier through [`brm_term_coordinates`](@ref)'s `parameter = :basis_weights`
role, which exists only for an ungrouped `hsgp`; a grouped `hsgp(…; by = …)`
(whose weights are a correlated random-effect block — zero its levels with
[`population_draws`](@ref) instead), an `mo(…)` simplex, or a population term
exposes no such role and errors rather than zeroing a coordinate whose removal
would mean something else. `to = :zero` is likewise the only value with an
established meaning; any other target errors.

# Example

```julia
unc     = BridgeStan.param_unc_names(stan_model)          # unconstrained names
full    = brm_execute(d, :predict; problem, draws, seed)  # every term present
linear  = term_draws(d, draws, unc;
                     predictor = :log_F, term = :hsgp_op_log_dose)
no_hsgp = brm_execute(d, :predict; problem, draws = linear, seed)  # term removed
```
"""
function term_draws(d::BRMDescriptor, draws::AbstractMatrix, unc_names;
                    predictor::Symbol, term::Symbol, to::Symbol = :zero)
    to === :zero || error(
        "BRM prediction: `term_draws` supports only `to = :zero` (got `$to`). ",
        "Zeroing an ungrouped `hsgp` term's basis weights removes exactly its ",
        "predictor contribution; no other target value has an established ",
        "meaning, so it is refused rather than guessed.")
    _ranef_check_draws(draws, unc_names)
    # Resolve the term's basis-weight carrier BY NAME against the UNCONSTRAINED
    # names. `brm_term_coordinates` matches emitted names and validates the
    # count, and `:basis_weights` resolves only for an ungrouped `hsgp` — so a
    # non-hsgp / grouped-hsgp term fails closed here rather than in `:predict`.
    resolved = brm_term_coordinates(d, predictor, unc_names;
                                    term, parameter = :basis_weights)
    out = Matrix{Float64}(draws)
    out[:, resolved.coordinates] .= 0.0
    out
end

function _hsgp_population_curve_term(d::BRMDescriptor, predictor::Symbol,
                                     coefficient::Symbol, term::Symbol)
    all_entries = _brm_term_coordinate_entries(d.plan.parent, predictor)
    entries = [e for e in all_entries if e.term === term]
    length(entries) == 1 || error(
        "BRM prediction: term `$term` occurs $(length(entries)) times on logical " *
        "predictor `$predictor`; expected exactly one orthogonal HSGP term. " *
        "Available term labels are " *
        "$(Tuple(sort!(unique(e.term for e in all_entries), by=string))).")
    value = only(entries).value
    getf(value) === hsgp || error(
        "BRM prediction: term `$term` on logical predictor `$predictor` is not " *
        "an `hsgp(...)` term; a total HSGP population curve is undefined.")

    args = getargs(value)
    length(args) == 1 || error(
        "BRM prediction: `hsgp_population_curve` supports exactly one HSGP " *
        "axis, but term `$term` has $(length(args)).")
    axis = _sb_named_inner(:hsgp, only(args))
    name(axis) === coefficient || error(
        "BRM prediction: term `$term` uses axis `$(name(axis))`, not requested " *
        "population coefficient `$coefficient`.")
    parent(axis) isa ExprColumn || error(
        "BRM prediction: term `$term` uses a raw-data axis. " *
        "`hsgp_population_curve` is deliberately restricted to a sampled " *
        "model-derived axis so its fitted draw-wise orthogonalization is explicit.")

    kw = getkwargs(value)
    get(kw, :orthogonal_to, nothing) === :linear || error(
        "BRM prediction: term `$term` must declare " *
        "`orthogonal_to=:linear`; otherwise adding `beta*$coefficient` to the " *
        "HSGP does not have the public total-effect meaning this API promises.")
    haskey(kw, :by) && error(
        "BRM prediction: grouped HSGP term `$term` has no single population " *
        "curve; `hsgp_population_curve` excludes subject/group effects.")
    _sb_gp_iso(kw, :hsgp) || error(
        "BRM prediction: model-derived HSGP term `$term` must be isotropic.")

    K, _ = _sb_hsgp_options(kw, 1)
    fits = _sb_hsgp_domain_fits(kw, 1; required=true)
    (; value, K=only(K), fit=only(fits))
end

function _hsgp_population_curve_grid(grid, fit, term::Symbol)
    isempty(grid) && error(
        "BRM prediction: fixed grid for term `$term` must not be empty.")
    all(x -> x isa Real && isfinite(x), grid) || error(
        "BRM prediction: fixed grid for term `$term` must contain only finite " *
        "real values.")
    values = collect(Float64, grid)
    center, L = fit
    lower, upper = center - L, center + L
    all(x -> lower <= x <= upper, values) || error(
        "BRM prediction: fixed grid for term `$term` contains values outside " *
        "its fitted HSGP domain ($lower, $upper). Extrapolating this compact " *
        "basis is refused.")
    values
end

function _hsgp_population_curve_basis(PHI_grid_raw::AbstractMatrix,
                                      grid::AbstractVector,
                                      PHI_train_raw::AbstractMatrix,
                                      x_train::AbstractVector)
    n_train = length(x_train)
    size(PHI_train_raw, 1) == n_train || error(
        "BRM prediction: internal HSGP training-basis row-count mismatch.")
    size(PHI_grid_raw, 2) == size(PHI_train_raw, 2) || error(
        "BRM prediction: internal HSGP basis-width mismatch.")

    x_mean = sum(x_train) / n_train
    x_centered = x_train .- x_mean
    x_ss = sum(abs2, x_centered)
    out = Matrix{Float64}(undef, size(PHI_grid_raw))
    for b in axes(PHI_train_raw, 2)
        phi_train = @view PHI_train_raw[:, b]
        phi_mean = sum(phi_train) / n_train
        slope = x_ss > 1e-12 ?
            dot(x_centered, phi_train .- phi_mean) / x_ss : 0.0
        @views out[:, b] .= PHI_grid_raw[:, b] .- phi_mean .-
                            (grid .- x_mean) .* slope
    end
    out
end

function _hsgp_population_curve_sqrt_spd(omega2::AbstractMatrix,
                                         sigma::Real, rho::Real)
    isfinite(rho) && rho > 0 || error(
        "BRM prediction: an HSGP length-scale draw is not finite and positive " *
        "(got $rho).")
    isfinite(sigma) && sigma >= 0 || error(
        "BRM prediction: an HSGP marginal-SD draw is not finite and " *
        "nonnegative (got $sigma).")
    scale = sigma * sqrt(rho * 2.5066282746310002)
    [scale * exp(-0.25 * rho * rho * omega2[b, 1])
     for b in axes(omega2, 1)]
end

"""
    hsgp_population_curve(d::BRMDescriptor, draws, constrained_names, grid;
                          predictor::Symbol, coefficient::Symbol,
                          term::Symbol)

Evaluate the population exposure contribution of one model-derived,
one-dimensional `hsgp(...; orthogonal_to=:linear)` term on a fixed grid. The
returned named tuple contains `grid` and three draws × grid matrices:

- `linear` — the population coefficient contribution `beta * grid`;
- `hsgp` — the residual nonlinear HSGP contribution;
- `total` — `linear + hsgp`, the quantity consumers normally want.

`draws` must contain CONSTRAINED posterior draws as rows, in
`constrained_names` order. The names must include transformed parameters: the
sampled model-derived axis is required to reconstruct the fitted projection for
each draw. With BridgeStan, obtain both with `include_tp=true,
include_gq=false`.

The public addresses come from the formula. For
`mu ~ ... + x + hsgp(x; orthogonal_to=:linear, ...)`, use
`predictor=:mu`, `coefficient=:x`, and `term=:hsgp_x`. Compiler-owned carrier
names are resolved through the descriptor and never form part of this API.

The curve is population-only: it includes no intercept, other covariates, or
group-specific random slopes. The grid must lie inside the fixed `domain`
fitted by the model. Raw-data, grouped, multidimensional, non-orthogonal, and
non-HSGP terms fail closed rather than returning a quantity with different
semantics.
"""
function hsgp_population_curve(d::BRMDescriptor, draws::AbstractMatrix,
                               constrained_names, grid::AbstractVector;
                               predictor::Symbol, coefficient::Symbol,
                               term::Symbol)
    size(draws, 2) == length(constrained_names) || error(
        "BRM prediction: draw matrix has $(size(draws, 2)) columns but the " *
        "model has $(length(constrained_names)) constrained coordinates. " *
        "`draws` must be draws × coordinates, in `constrained_names` order.")
    spec = _hsgp_population_curve_term(d, predictor, coefficient, term)
    grid_values = _hsgp_population_curve_grid(grid, spec.fit, term)

    beta_coordinates = brm_population_effect_coordinates(
        d, predictor, constrained_names; coefficient).coordinates
    length(beta_coordinates) == 1 || error(
        "BRM prediction: population coefficient `$coefficient` on `$predictor` " *
        "does not resolve to exactly one scalar coordinate.")
    x_coordinates = brm_output_coordinates(
        d, coefficient, constrained_names; role=:linear_predictor)
    isempty(x_coordinates) && error(
        "BRM prediction: model-derived HSGP axis `$coefficient` has no sampled " *
        "training coordinates.")
    rho_coordinates = brm_term_coordinates(
        d, predictor, constrained_names;
        term, parameter=:length_scale).coordinates
    sigma_coordinates = brm_term_coordinates(
        d, predictor, constrained_names;
        term, parameter=:sd).coordinates
    weight_coordinates = brm_term_coordinates(
        d, predictor, constrained_names;
        term, parameter=:basis_weights).coordinates
    length(rho_coordinates) == 1 || error(
        "BRM prediction: term `$term` must expose one length-scale coordinate.")
    length(sigma_coordinates) == 1 || error(
        "BRM prediction: term `$term` must expose one marginal-SD coordinate.")
    length(weight_coordinates) == spec.K || error(
        "BRM prediction: term `$term` exposes $(length(weight_coordinates)) " *
        "basis weights but its fitted formula requires $(spec.K).")

    PHI_grid_raw, omega = _sb_apply_hsgp(
        (spec.fit,), (grid_values,), (spec.K,))
    n_draws, n_grid = size(draws, 1), length(grid_values)
    linear = Matrix{Float64}(undef, n_draws, n_grid)
    nonlinear = Matrix{Float64}(undef, n_draws, n_grid)
    for draw in 1:n_draws
        beta = draws[draw, only(beta_coordinates)]
        x_train = collect(Float64, @view draws[draw, x_coordinates])
        all(isfinite, x_train) || error(
            "BRM prediction: model-derived HSGP axis `$coefficient` contains " *
            "a non-finite value in constrained draw $draw.")
        PHI_train_raw, _ = _sb_apply_hsgp(
            (spec.fit,), (x_train,), (spec.K,))
        PHI = _hsgp_population_curve_basis(
            PHI_grid_raw, grid_values, PHI_train_raw, x_train)
        rho = draws[draw, only(rho_coordinates)]
        sigma = draws[draw, only(sigma_coordinates)]
        weights = collect(Float64, @view draws[draw, weight_coordinates])
        isfinite(beta) && all(isfinite, weights) || error(
            "BRM prediction: population slope or HSGP basis weights contain a " *
            "non-finite value in constrained draw $draw.")
        sqrt_spd = _hsgp_population_curve_sqrt_spd(omega, sigma, rho)
        @views linear[draw, :] .= beta .* grid_values
        @views nonlinear[draw, :] .= PHI * (sqrt_spd .* weights)
    end

    (; predictor, coefficient, term,
       domain=(spec.fit[1] - spec.fit[2], spec.fit[1] + spec.fit[2]),
       grid=grid_values, linear, hsgp=nonlinear, total=linear + nonlinear)
end
