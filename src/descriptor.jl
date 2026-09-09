# ==============================================================================
# descriptor.jl — ONE authoritative executable semantic model descriptor.
#
# A `@brm` block is already the authoritative declaration of a model. Until now
# a consumer that wanted to *mount* that model — render a form, offer
# fit/predict, address the draws, label a coefficient — had to assemble the
# answer from six different places:
#
#   * a hand-kept formula string / display name,
#   * `outcomes` / `linear_predictors` / `predictors` / `grouping_factors`
#     (introspection.jl) for the model shape,
#   * `generative_plan` (sbimpl.jl) for what BRM actually emitted,
#   * `popcoefnames` for coefficient labels,
#   * `data_columns` + `sb.preproc` for which dataframe columns are needed,
#   * and StanBlocks for the executable half.
#
# `brm_descriptor` collapses all six into one value derived from the single
# declaration. Nothing here is a second source of truth: every field is read
# off the `GenerativePlan` (which is itself read off the emitted SLIC body) or
# off StanBlocks' own `stan_descriptor`. The two compose BY NAME, and every
# place the correspondence can fail is an explicit, loud failure rather than a
# guess — see "Failing closed" below.
#
# Division of labour with StanBlocks (stanblocks-use §30):
#   StanBlocks' `ModelDescriptor` reports STAN-level structure — the data
#   block, the parameter/TP/GQ outputs, and which operations the traced program
#   supports. It deliberately does not know "this is a random effect", "this
#   column was z-scaled", "this coefficient is the slope on `x`".
#   Those are BRM concepts, and they are what this layer adds.
# ==============================================================================

# ---- inputs -----------------------------------------------------------------

"""
    BRMInput

One entry of the emitted Stan data block, with its BRM provenance attached.

The first eight fields mirror StanBlocks' `ModelInput` (stanblocks-use §30) —
`name`, `type` (the Stan center type), `size` (the declared size *expressions*,
not values), `constraints`, and the four flags:

- `observed` — the model conditions on this input (it is the LHS of a `~`).
- `held_out` — this input is cv-marked, so its likelihood contribution is
  dropped and it re-draws in generated quantities.
- `derived` — this input is another input's declared *size* (`y_n` for
  `vector[y_n] y`). Re-binding `y` re-derives it; never ask a user for it.
- `inlined` — folded into the generated source, never reaches the data JSON.

The last two are BRM's addition — the **schema link** back to the dataframe:

- `column` — the dataframe column this Stan input was built from, or `nothing`
  when it has no single raw column (a design matrix, a level count, a size).
- `transform` — the BRM preprocessing kind applied on the way (`:zscale`,
  `:center`, `:standardize`, `:factor`, `:mo`, `:spline`, `:gp`, `:hsgp`,
  `:static`, `:group_index`, `:ranef_factor_dummy`, `:multi_membership`, `:kernel_subject_count`,
  `:kernel_ragged`, `:protect`),
  or `nothing` when the column is passed through untransformed.

`column`/`transform` are read from the plan's `preproc` record and the BRMI's
own data columns — they are never inferred from the input's *name*.
"""
struct BRMInput
    name::Symbol
    type::Symbol
    size::Tuple
    constraints::NamedTuple
    observed::Bool
    held_out::Bool
    derived::Bool
    inlined::Bool
    column::Union{Nothing,Symbol}
    transform::Union{Nothing,Symbol}
end

# ---- outputs ----------------------------------------------------------------

"""
    BRMOutput

One thing the compiled model produces, with its BRM meaning attached.

Fields `name` … `source` mirror StanBlocks' `ModelOutput` (stanblocks-use §30):
`kind` is `:parameter` / `:transformed_parameter` / `:generated_quantity`, and
`generative` is `:posterior` / `:draw` / `:pointwise_loglik` / `:derived` with
`source` naming the observation a `:draw` or `:pointwise_loglik` belongs to.

BRM adds:

- `role` — what this output *means in the formula*:

  | role | what it is |
  | --- | --- |
  | `:population_effect` | a `popefs` block — the population-level design and its coefficients |
  | `:random_effect` | a `ranef_*` block and its internals |
  | `:group_block` | a `kernel(...)` / `plate` result and everything declared per cell |
  | `:parameter` | an ordinary declared prior (`sigma ~ Exponential(1)`) |
  | `:linear_predictor` | a formula linear predictor (`mu ~ 1 + x + (1\\|g)`) — an assignment, so no `~` declaration binds it |
  | `:posterior_predictive` | a predictive draw of an observation |
  | `:pointwise_loglik` | an observation's per-element log-likelihood |
  | `:stan_derived` | a Stan-level output no BRM declaration owns |

- `declaration` — the `GenerativeDeclaration` this output came from, or
  `nothing` for `:linear_predictor` / `:stan_derived`. Its twelve fields
  (family, dimension, constraints, the verbatim expression, …) are documented
  on [`GenerativeDeclaration`](@ref); reach for them instead of re-parsing.
  An output whose `name` differs from `declaration.target` is one of that
  block's *internals* (`pop_mu_beta_pop` under `pop_mu`).
- `logical` — the BRM-level quantity whose value this emitted Stan output
  physically carries, or `nothing` for an internal. It is narrower than
  `declaration`: `pop_mu_beta_pop` belongs to the `pop_mu` declaration but is
  one of its internals, so it carries no logical target. For a ragged
  `kernel(...)` result, `logical == :loc` can accompany an emitted `name` such
  as `:loc__pl_mem_1`; consumers keep the logical identity while addressing
  BridgeStan through the emitted name.

  **Every named value a `kernel(...)` cell assigns carries one too** — no
  annotation. The author bound the name and the value is in every posterior
  draw, so it is addressable by that name:

  ```julia
  pk_loc ~ kernel(t, dv, qt_y, log_CL, qt_base) do ts, yy, qy, lCL, qbase
      conc   = exp(-exp(lCL) .* ts)
      qt_loc = qbase .+ slope .* conc     # a cell value, not an observation
      qy ~ normal(qt_loc, qt_sigma)
      conc                                 # the RETURN
  end

  brm_output(d, :qt_loc)     # BRMOutput(name = :pk_loc_qt_loc__pl_mem_1,
                             #           logical = :qt_loc, role = :group_block, …)
  brm_output(d, :conc)       # the same cell's other named value
  brm_output(d, :pk_loc)     # the collected return
  ```

  Cell values are ordinary `:group_block` outputs of the plate declaration and
  compose with `brm_outputs` / `brm_output_coordinates` / `segments` like any
  other; their emitted `name` stays compiler-owned and unparsed. Only a
  top-level `name = ...` counts: a slice parameter is an argument (address the
  positional it came from), and an in-cell `~` is an observation (addressable
  through its own column's twins, below).

  Two consequences are deliberate. **Scratch is addressable too** — `CL =
  exp(lCL)` gives `logical == :CL` — so `brm_outputs(d)` is wider than the set
  of quantities an author would call primary. And **two cells may name a value
  the same thing**; both are claimed, `brm_output(d, :mu)` then refuses and
  names the candidates with their owners, and `brm_outputs(d; logical=:mu)`
  returns both for selection on `declaration.target`. That is the same
  one-logical-many-carriers contract an observation's two twins already use;
  the model still builds and unambiguous names still resolve directly.

  ⚠ **A predictive twin is not a substitute for the location it was drawn
  from.** In the model above `qt_y_gen` is `normal_..._rng(qt_loc, qt_sigma)` —
  noise ADDED — so it answers a different question than `qt_loc` does.

  For a predictive or pointwise-loglik twin it is StanBlocks' `source` — the
  OBSERVED QUANTITY — which is not always a declaration target. A top-level
  `y ~ Normal(mu, sigma)` gives `y_gen` `logical == :y`, where `:y` is also the
  declaration; a plate-nested `yy ~ normal(...)` over column `dv` gives
  `dv_gen` `logical == :dv`, where the declaration is the cell-local `yy` and
  `:dv` is a data column. Both twins of one observation share it, so
  [`brm_output`](@ref) needs `role=` to pick between them.
- `labels` — per-element labels when BRM can name them (population
  coefficients, via [`popcoefnames`](@ref)), otherwise `nothing`. A UI that
  would otherwise print `beta_pop.1`, `beta_pop.2` can print `x`, `g`.
- `segments` — group boundaries for a RAGGED quantity, carried through from
  StanBlocks' `ModelOutput.segments`, otherwise `nothing`. Group `g` occupies
  `segments[g-1]+1 : segments[g]` (with `segments[0] ≡ 0`) *within this
  output's own coordinates* — so it composes directly with the vector
  [`brm_output_coordinates`](@ref) returns, and a consumer never re-derives
  per-subject boundaries from a length column.

`role` is derived from the declaration, never from the output's name — a user
variable may legitimately be called `beta_pop` or end in `_gen`.
"""
struct BRMOutput
    name::Symbol
    kind::Symbol
    type::Symbol
    size::Tuple
    constraints::NamedTuple
    generative::Symbol
    source::Union{Nothing,Symbol}
    role::Symbol
    declaration::Union{Nothing,GenerativeDeclaration}
    logical::Union{Nothing,Symbol}
    labels::Union{Nothing,Vector{Symbol}}
    segments::Union{Nothing,Vector{Int}}
end

# Preserve the earlier positional constructors for downstream code that creates
# display-only BRMOutput values. `segments` and `logical` are appended rather
# than inserted precisely so those calls keep compiling; descriptor-derived
# values always supply both explicitly below.
BRMOutput(name, kind, type, size, constraints, generative, source, role,
          declaration, logical, labels) =
    BRMOutput(name, kind, type, size, constraints, generative, source, role,
              declaration, logical, labels, nothing)
BRMOutput(name, kind, type, size, constraints, generative, source, role,
          declaration, labels) =
    BRMOutput(name, kind, type, size, constraints, generative, source, role,
              declaration, nothing, labels, nothing)

# ---- operations -------------------------------------------------------------

"""
    BRMOperation

One executable operation the declaration supports. Operations are **derived**,
never listed: each appears exactly when the model actually supports it, so a
consumer can render one button per operation without maintaining its own list
and without rendering a button that will fail.

- `name` / `title` — the identifier and a human label.
- `inputs` — the names this operation needs. **Read `origin` to know which
  namespace they live in**: `:stan` operations take Stan data keys (the
  neither-derived-nor-inlined subset), `:brm` operations take *dataframe
  columns*.
- `outputs` — the descriptor output names it produces (empty when it produces
  a new descriptor or the Stan source rather than draws).
- `origin` — `:stan` (delegated to StanBlocks' `stan_execute`), `:brm`
  (BRM-level: replay the declaration on new data), or `:override` (supplied by
  the consumer through `brm_descriptor(...; operations=…)`).
- `run` — the callable `brm_execute` invokes, `(descriptor; kwargs...) -> result`.

See [`brm_descriptor`](@ref) for the derivation table and the extension points.
"""
struct BRMOperation
    name::Symbol
    title::String
    inputs::Tuple{Vararg{Symbol}}
    outputs::Tuple{Vararg{Symbol}}
    origin::Symbol
    run::Any
end

# ---- highlighted Stan definitions ------------------------------------------

"""
    BRMHighlight

One caller-selected Stan definition to feature when presenting a
[`BRMDescriptor`](@ref).

- `name` is the stable definition name resolved by StanBlocks.
- `caption` is optional presentation text supplied by the caller.
- `definition` is StanBlocks' authoritative definition descriptor, including
  its emitted source and direct dependency links.
- `closure` is that definition plus its exact transitive included dependencies
  in StanBlocks' authoritative emission order.

BRM never copies or parses the generated Stan source to construct either.

Pass symbols and/or `name => caption` pairs through the ordered `highlights=`
keyword of [`brm_descriptor`](@ref). The full executable definition inventory
remains available on `d.stan.definitions`; `d.highlights` is only the selected,
ordered presentation layer.
"""
struct BRMHighlight
    name::Symbol
    caption::Union{Nothing,String}
    definition::StanBlocks.ModelDefinition
    closure::Tuple{Vararg{StanBlocks.ModelDefinition}}
end

# ---- the descriptor ---------------------------------------------------------

"""
    BRMDescriptor

The one authoritative, executable, reflectable value for a `@brm` declaration.
Construct with [`brm_descriptor`](@ref).

| field | what it is |
| --- | --- |
| `id` | stable content identity — StanBlocks' descriptor id, the hash of the generated Stan source, which is also the key `instantiate` caches the compiled artifact under. Stable across processes; independent of `name` |
| `name` | informational label (`name=` at construction, else the plan's own) |
| `formula` | the canonical rendering of the parsed declaration (`show` of the `BRMI`). Not the literal characters you typed — it is *derived* from the declaration, so unlike a hand-kept `formula_src` string it cannot drift from the model that runs |
| `plan` | the [`GenerativePlan`](@ref) — every emitted `~` site, in order |
| `stan` | StanBlocks' `ModelDescriptor` for the same model, if you need the Stan-level view verbatim |
| `highlights` | ordered caller-selected [`BRMHighlight`](@ref)s, resolved by stable name against `stan.definitions`; the full included-definition inventory remains on `stan` |
| `inputs` | `Tuple` of [`BRMInput`](@ref) — the data block plus its dataframe provenance |
| `outputs` | `Tuple` of [`BRMOutput`](@ref) — everything the model produces, with BRM roles |
| `operations` | `Tuple` of [`BRMOperation`](@ref) — derived, not listed |
| `columns` | the dataframe columns the declaration reads. **This is the schema** a form should collect for a replay |
| `unpredictable` | observation targets whose predictive draw the emitted Stan program does **not** produce. Empty in the ordinary case; see below |

`brm_columns(d)`, `brm_operation(d, name)`, `brm_execute(d, name; …)` are the
accessors.
"""
struct BRMDescriptor{P,S}
    id::String
    name::Symbol
    formula::String
    plan::P
    stan::S
    highlights::Tuple{Vararg{BRMHighlight}}
    inputs::Tuple{Vararg{BRMInput}}
    outputs::Tuple{Vararg{BRMOutput}}
    operations::Tuple{Vararg{BRMOperation}}
    columns::Tuple{Vararg{Symbol}}
    unpredictable::Tuple{Vararg{Symbol}}
end

# ---- role derivation --------------------------------------------------------

# The BRM meaning of a `:prior` declaration, from the family BRM emitted. These
# are BRM's own submodel names (sbimpl.jl) — a closed set we own, not a guess
# about user code.
_brm_declaration_role(d::GenerativeDeclaration) = begin
    d.role === :observation && return :observation
    # A non-empty `context` means the declaration lives INSIDE a plate cell —
    # a per-group parameter of a `kernel(...)` block, whatever its family.
    # Formula-declared BSV is emitted by the ordinary top-level `ranef_*`
    # declarations and therefore keeps its `:random_effect` role. Only values
    # declared by the plate cell itself belong to this `:group_block` context.
    isempty(d.context) || return :group_block
    f = d.family
    f isa Symbol || return :parameter
    # `_popefs_coefs` / `_popefs_normal_coefs` are the `normal_id_glm` fusion's
    # coefficient-returning siblings (sbimpl.jl) — same block, same
    # `beta_pop`, so the same role and the same `popcoefnames` labels.
    f in (:popefs, :_popefs_normal, :_popefs_coefs, :_popefs_normal_coefs) &&
        return :population_effect
    startswith(String(f), "ranef") && return :random_effect
    f === :plate && return :group_block
    :parameter
end

# ---- schema: which dataframe column does this Stan input come from? ---------

# A preproc record names its source either directly (a column NAME Symbol, for
# factor/mo/spline) or as an axis-name tuple (gp/hsgp), or as a column-node tree
# (zscale/center/standardize and
# the protect/implicit-fn fallback). Walk the tree to its single data leaf; a
# tree touching several columns has no single source column, and we say so
# rather than picking one.
_brm_raw_column(x::Symbol) = x
_brm_raw_column(x::NamedColumn) = _brm_raw_column_inner(x, parent(x))
_brm_raw_column_inner(x, ::DataColumn) = name(x)
_brm_raw_column_inner(_x, _) = nothing
_brm_raw_column(x::ExprColumn) = begin
    found = Symbol[]
    for a in getargs(x)
        c = _brm_raw_column(a)
        isnothing(c) || push!(found, c)
    end
    unique!(found)
    length(found) == 1 ? only(found) : nothing
end
_brm_raw_column(x::Tuple) = begin
    found = unique(Symbol[c for a in x for c in (_brm_raw_column(a),) if !isnothing(c)])
    length(found) == 1 ? only(found) : nothing
end
_brm_raw_column(_) = nothing

function _brm_input_schema(plan, key::Symbol, df_columns)
    e = get(plan.preproc, key, nothing)
    isnothing(e) || return (_brm_raw_column(e.raw_ref), e.kind)
    key in df_columns && return (key, nothing)
    (nothing, nothing)
end

# ---- construction -----------------------------------------------------------

_brm_plan_of(plan::GenerativePlan) = plan
_brm_plan_of(sb::SBBRMI) = generative_plan(sb)

# The formula-owned population address space. `block` is derived FORWARDS with
# the same helper sbimpl uses to emit the design, so an inert `log_Vc` predictor
# and a linked `log(Vc)` predictor never become indistinguishable through name
# parsing. Keep this as a vector: duplicate public logical predictors must stay
# observable to the fail-closed query below rather than being overwritten by a
# Dict constructor.
_brm_population_effect_entries(brmi) = [
    (; logical=l.name,
       block=Symbol(:pop_, _sb_lp_emitted_name(l.name, l.link_lhs_fn)),
       link=l.link_lhs_fn)
    for l in linear_predictors(brmi)
    if !_brm_is_joint_covariance_factor(brmi, l.name)
]

function _brm_is_joint_covariance_factor(brmi, name::Symbol)
    haskey(brmi.operations, name) || return false
    value = brmi.operations[name]
    value isa NamedColumn || return false
    op = parent(value)
    op isa ExprColumn && getf(op) === (~) || return false
    _, rhs = getargs(op, 2)
    rhs isa ExprColumn && getf(rhs) === LKJCovarianceFactor
end

_brm_joint_covariance_factors(brmi) = Symbol[
    l.name for l in linear_predictors(brmi)
    if _brm_is_joint_covariance_factor(brmi, l.name)]

_brm_highlight_spec(x::Union{Symbol,AbstractString}) =
    (Symbol(x), nothing, nothing)
_brm_highlight_spec(x::Pair{<:Union{Symbol,AbstractString}}) =
    (Symbol(first(x)), isnothing(last(x)) ? nothing : String(last(x)), nothing)
_brm_highlight_spec(x::BRMHighlight) =
    (x.name, x.caption, x.definition.signature)
_brm_highlight_spec(x) = error(
    "brm_descriptor: each `highlights` entry must be a definition name, a " *
    "`name => caption` pair, or a BRMHighlight; got $(repr(x)).")

function _brm_highlights(stan, specs)
    entries = specs isa Union{Symbol,AbstractString,Pair,BRMHighlight} ?
              (specs,) : specs
    selected = BRMHighlight[]
    seen = Set{Symbol}()
    for spec in entries
        name, caption, signature = _brm_highlight_spec(spec)
        name in seen && error(
            "brm_descriptor: definition `$name` is selected more than once in " *
            "`highlights`; each highlight must be unique.")
        push!(seen, name)
        definition = StanBlocks.stan_definition(stan, name; signature)
        closure = StanBlocks.stan_definition_closure(stan, definition)
        push!(selected, BRMHighlight(name, caption, definition, closure))
    end
    Tuple(selected)
end

# Label a population-effect output's elements, when BRM can. `popcoefnames`
# already owns this (and is the documented way not to re-parse `beta_pop.N`);
# it can legitimately return `nothing`, and it errors on shapes it cannot
# resolve (`hsgp(x, by=g)` and friends) — neither is a reason to fail descriptor
# construction, so an unlabelled output simply carries `labels = nothing`.
_brm_labels(brmi, lp::Symbol) =
    try
        v = popcoefnames(brmi, lp)
        isnothing(v) ? nothing : collect(Symbol, v)
    catch
        nothing
    end

# The Stan identifier a declaration resolves to.
#
# At top level that is the target itself. Inside a plate the emitted name is
# the context joined with the target — `kernel_z` declared in the `pred` cell
# becomes the Stan parameter `pred_kernel_z`. That is the same join BRM already
# uses to build a declaration's `draw`, so it is a rule we own, not a guess.
_brm_stan_name(d::GenerativeDeclaration) =
    isempty(d.context) ? d.target : Symbol(join((d.context..., d.target), "_"))

# Which Stan output does a declaration own?
#
# StanBlocks inlines a called submodel under its binding name: `pop_mu ~
# popefs(; X=X_mu)` yields the transformed parameter `pop_mu` (the submodel's
# return) AND the submodel's own internals prefixed with it
# (`pop_mu_beta_pop`, `pop_mu_n_covariates`). So a declaration owns its exact
# resolved name plus everything under `<resolved>_`.
#
# Resolution order, most authoritative first:
#   1. StanBlocks' own `source` link (a `_gen` / `_likelihood` twin). Never
#      parse the suffix ourselves — `source` is the authoritative answer.
#   2. exact name match against a declaration target,
#   3. the LONGEST `<target>_` prefix (nested submodels then resolve to the
#      innermost declaration that owns them, not an outer one).
function _brm_owner(o, by_name, targets)
    isnothing(o.source) || return get(by_name, o.source, nothing)
    haskey(by_name, o.name) && return by_name[o.name]
    best = nothing
    s = String(o.name)
    for t in targets
        p = String(t) * "_"
        startswith(s, p) || continue
        (isnothing(best) || length(String(t)) > length(String(best))) && (best = t)
    end
    isnothing(best) ? nothing : by_name[best]
end

# Find the emitted Stan outputs referenced by a traced logical binding. This
# walks StanBlocks' traced expression graph, not rendered identifiers: a ragged
# plate result is a compile-time `RaggedVector(mem, ends)` view, and only `mem`
# is a posterior-producing descriptor output. Cell locals do not occur in that
# binding even though `_brm_owner` correctly assigns them to the same plate
# declaration.
_brm_output_leaves!(found, x::Symbol, candidates) =
    x in candidates && push!(found, x)
_brm_output_leaves!(found, x::StanBlocks.StanExpr, candidates) =
    _brm_output_leaves!(found, StanBlocks.expr(x), candidates)
_brm_output_leaves!(found, x::StanBlocks.CanonicalExpr, candidates) =
    foreach(a -> _brm_output_leaves!(found, a, candidates), x.args)
_brm_output_leaves!(found, x::Expr, candidates) =
    foreach(a -> _brm_output_leaves!(found, a, candidates), x.args)
_brm_output_leaves!(found, x::Tuple, candidates) =
    foreach(a -> _brm_output_leaves!(found, a, candidates), x)
_brm_output_leaves!(_found, _x, _candidates) = nothing

# Every named value a `kernel(...)` cell assigns: plate target => value names.
#
# Read from the BRM DECLARATION, not recovered from the emitted program. There
# is nothing to recover there: the emitted plate call carries only StanBlocks'
# own `outer=` keyword, and the promoted carrier's name is compiler-owned —
# exactly the name a consumer is forbidden to parse, and the reason this exists.
#
# A cell value IS a named BRM-level quantity: the author bound it, and it is
# saved in every posterior draw. Nothing extra is required to say so, which is
# why this reads the body rather than an annotation (decision `1tpze5q`).
# Top-level `name = ...` only — a slice parameter is an argument, and an in-cell
# `~` is an observation already addressable through its own column's twins.
#
# Same walk shape as `_sb_collect_id_buckets` (sbimpl.jl): an operation is a
# `NamedColumn` over a `~` `ExprColumn`, whose second argument is the RHS term.
function _brm_kernel_cell_values(brmi)
    cells = Pair{Symbol,Vector{Symbol}}[]
    for (target, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc)); isnothing(op) && continue
        getf(op) === (~) || continue
        _, rhs_raw = getargs(op, 2)
        rhs = _as_expr_column(rhs_raw); isnothing(rhs) && continue
        getf(rhs) === kernel || continue
        args = getargs(rhs)
        isempty(args) && continue
        lam = first(args)
        (lam isa Expr && length(lam.args) >= 2) || continue
        body = lam.args[2]
        stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
        # `unique` because a cell may rebind one name; both assignments are the
        # same binding downstream, so it is one quantity, claimed once.
        names = unique!(Symbol[s.args[1] for s in stmts
                               if Meta.isexpr(s, :(=)) && s.args[1] isa Symbol])
        isempty(names) || push!(cells, target => names)
    end
    cells
end

function _brm_logical_outputs(stan, by_name, targets, cell_values,
                              formula_predictors)
    logical = Dict{Symbol,Symbol}()
    output_names = Set{Symbol}(o.name for o in stan.outputs)
    model_names = Set{Symbol}(keys(stan.model))

    # One assignment point, so every branch below gets the same conflict check.
    # Assigning directly would let a second claim silently overwrite the first,
    # which is precisely the ambiguity this error exists to refuse.
    claim! = (emitted, target) -> begin
        if haskey(logical, emitted) && logical[emitted] !== target
            error(
                "brm_descriptor: emitted output `$emitted` carries two logical " *
                "targets (`$(logical[emitted])` and `$target`). The traced " *
                "model is ambiguous; give the declarations distinct results.")
        end
        logical[emitted] = target
    end

    # StanBlocks' `source` link IS the logical identity of a predictive or
    # pointwise-loglik twin, and it is authoritative — the same link
    # `_brm_owner` trusts first. Resolve on it DIRECTLY rather than through a
    # declaration, because the observed quantity is not always a declaration:
    #
    #   y ~ Normal(mu, sigma)            ->  `y_gen.source === :y`, and `:y` IS
    #                                        a declaration target.
    #
    #   pred ~ kernel(dose, dv, ...) do dd, yy, ls
    #       yy ~ normal(mu, sigma)       ->  `dv_gen.source === :dv`, but the
    #   end                                  declaration is the CELL-LOCAL `yy`
    #                                        (context `(:pred,)`), and `:dv` is
    #                                        a data COLUMN, not a target.
    #
    # Keying off declarations covered only the first shape, so every
    # plate-nested observation's predictive carrier had no logical target at
    # all — which is exactly why a consumer had to filter
    # `descriptor.stan.outputs` and hardcode the emitter-owned `_gen` suffix.
    #
    # Both twins of one observation share a logical target; `role`
    # (`:posterior_predictive` vs `:pointwise_loglik`) separates them, so the
    # query takes `role` to disambiguate.
    for o in stan.outputs
        isnothing(o.source) || claim!(o.name, o.source)
    end

    # A formula-level linear predictor is not a declaration: sbimpl emits it
    # from an `=` assignment after combining the population and group blocks.
    # Its public formula name is nevertheless stable and is already the exact
    # rule used below to assign `role = :linear_predictor`. Claim that same
    # output here so role-qualified posterior lookup does not require a
    # consumer to fall back to the emitter-owned Stan name.
    for logical in formula_predictors
        logical in output_names && claim!(logical, logical)
    end

    # The outputs one declaration owns, and the emitted carriers a BRM-side name
    # resolves to within them. Two shapes, one rule: a scalar per-cell value is
    # emitted under the context-joined name directly, while a per-cell VECTOR is
    # promoted to plate memory and only the traced binding names the promoted
    # carrier. Factored out because the cell-value pass below applies exactly
    # this rule to a different name.
    owned_by = decl -> Set{Symbol}(o.name for o in stan.outputs
                                   if _brm_owner(o, by_name, targets) === decl)
    carriers = (resolved, owned) -> begin
        found = Set{Symbol}()
        if resolved in output_names
            push!(found, resolved)
        elseif resolved in model_names
            _brm_output_leaves!(found, stan.model[resolved], owned)
        end
        found
    end

    for (resolved, decl) in by_name
        decl.role === :observation && continue
        owned = owned_by(decl)
        isempty(owned) && continue
        foreach(emitted -> claim!(emitted, decl.target), carriers(resolved, owned))
    end

    # ---- named `kernel(...)` cell values ------------------------------------
    #
    # A named deterministic cell value is OWNED by the plate declaration but
    # never occurs in its RETURN binding, so the loop above cannot reach it. It
    # has a binding of its own, though — the same context join BRM already owns
    # (`_brm_stan_name`) — so it resolves through exactly the rule above, with no
    # new naming convention and no compiler-owned suffix parsed on either side.
    #
    # Claimed unconditionally (decision `1tpze5q`). The author named the value
    # and it is saved in every draw; requiring a second annotation to say so
    # bought nothing but a keyword. Two consequences are deliberate:
    #
    #   * Scratch is addressable too. `CL = exp(lCL)` becomes `logical = :CL`.
    #     Nothing resolves to the wrong carrier; `brm_outputs(d)` is simply wider
    #     than the set of quantities an author would call primary.
    #   * Two cells may give a value the SAME name (`mu` in a PK cell and a PD
    #     cell). Both are claimed, and `brm_output(d, :mu)` then refuses and
    #     names the candidates with their owners — the descriptor's existing
    #     one-logical-many-carriers contract, the same one an observation's two
    #     twins already use. It is NOT a construction failure: the model builds,
    #     `brm_outputs(d; logical=:mu)` returns both, and every unambiguous name
    #     around them still resolves directly.
    #
    # A cell value shadowing an existing model binding cannot reach here at all:
    # StanBlocks' plate tracing refuses it (`AssertionError: name ∉ keys(info)`).
    for (target, locals) in cell_values
        decl = get(by_name, target, nothing)
        isnothing(decl) && continue
        owned = owned_by(decl)
        for lc in locals
            # An empty result is not an error: the compiler may legitimately
            # decline to promote a value (folded away, loop-invariant). Under an
            # unconditional rule that is an ordinary cell, not a failed request,
            # so it stays unclaimed rather than raising on every model.
            foreach(emitted -> claim!(emitted, lc), carriers(Symbol(target, "_", lc), owned))
        end
    end
    logical
end

# Indices of the SINGLE output that physically carries a formula quantity's
# constrained draws for a declaration, among `outputs` matching `pred`.
#
# In an ordinary fit that carrier is a sampled `:parameter`; in a likelihood-free
# `regime="prior"` program (no observation `~`) StanBlocks re-draws the same
# quantity from its prior into generated quantities under the SAME constrained
# name (the `_rng` companions), so its kind is `:generated_quantity`. Both
# physically hold the quantity's constrained draws and are addressable against
# BridgeStan's constrained names.
#
# The `:parameter` carrier is PREFERRED: the GQ carrier is used only when NO
# sampled carrier matches `pred`. So a fit is byte-for-byte unchanged (the
# sampled carrier still wins), and if a future prior-draw companion ever emits a
# GQ twin BESIDE a sampled parameter, the sampled one is still selected — the
# label attach does not strip a valid fitted carrier's labels and a fitted
# coordinate query does not start failing on the twin. A genuine `:parameter`
# ambiguity (two sampled carriers) is preserved for the caller's own
# `length(...) == 1` fail-closed check, because the GQ fallback fires only on an
# EMPTY parameter match and so cannot mask it. A `:transformed_parameter` (a
# derived value such as a fused `mu`, or an R2D2 `tau`) is deliberately NOT a
# draw carrier here — those are matched by exact carrier name where addressed —
# which keeps the population/categorical/term single-carrier guarantee intact.
function _brm_carrier_indices(outputs, pred)
    idx = findall(o -> pred(o) && o.kind === :parameter, outputs)
    isempty(idx) || return idx
    findall(o -> pred(o) && o.kind === :generated_quantity, outputs)
end

# Attach coefficient labels to the population block's coefficient vector.
#
# `popefs` (sbimpl.jl) declares exactly ONE parameter, `beta_pop`, so within a
# population block the coefficient vector is the unique draw carrier (a sampled
# `:parameter` in a fit, its `:generated_quantity` re-draw in a prior program).
# Labelling that one is unambiguous; if a future `popefs` ever grew a second
# such carrier the uniqueness check fails and nothing is labelled, rather than
# the wrong output being labelled — a missing label is a plain UI, a wrong label
# is a lie about which covariate a posterior column belongs to.
function _brm_label_population!(outputs, brmi, pop_lp)
    for (block, lp) in pop_lp
        # The coefficient vector is the block INTERNAL (`pop_<lp>_beta_pop`, whose
        # name differs from the block target); the block RESULT (`pop_<lp>`, name
        # === target) is the linear-predictor value, not a coefficient carrier.
        # `name !== block` is load-bearing for a prior program: there the block
        # result assignment is ALSO emitted as a generated quantity under the
        # same target, so filtering by target alone would match two GQ carriers.
        idx = _brm_carrier_indices(outputs,
                  o -> o.role === :population_effect && !isnothing(o.declaration) &&
                       o.declaration.target === block && o.name !== block)
        length(idx) == 1 || continue
        labels = _brm_labels(brmi, lp)
        isnothing(labels) && continue
        o = outputs[idx[1]]
        outputs[idx[1]] = BRMOutput(o.name, o.kind, o.type, o.size, o.constraints,
                                    o.generative, o.source, o.role, o.declaration,
                                    o.logical, labels, o.segments)
    end
    outputs
end

# Ragged group boundaries, read from StanBlocks' `ModelOutput.segments` when the
# pinned StanBlocks emits them. Guarded by `hasproperty` rather than a version
# check: BRM is developed against a moving StanBlocks, and a hard field access
# would turn "your pin predates segments" into a MethodError at descriptor
# construction — i.e. an unrelated model would stop reflecting at all — instead
# of the honest "this consumer has no segment metadata yet".
_brm_output_segments(o) =
    hasproperty(o, :segments) ? getproperty(o, :segments) : nothing

"""
    brm_descriptor(sb::SBBRMI; name=nothing, operations=Dict(), titles=Dict(), highlights=())
    brm_descriptor(plan::GenerativePlan; …)
    brm_descriptor(builder::Function, df; mod=@__MODULE__, cv_groups=Set(), held_out=(), …)

Derive the one authoritative executable semantic descriptor for a `@brm`
declaration. See [`BRMDescriptor`](@ref) for the fields.

The `builder` form is the one to prefer — it keeps the `@brm` builder, so the
descriptor can also offer `:replay` (rebuild the declaration for genuinely new
groups):

```julia
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    y ~ Normal(mu, sigma)
end
d = brm_descriptor(builder, df; mod=@__MODULE__, name=:my_model)
```

# The derived operations

Each is offered exactly when the declaration supports it. Nothing is listed by
hand, so an operation that appears can be executed.

| operation | origin | offered when |
| --- | --- | --- |
| `:transpile` | `:stan` | always |
| `:instantiate` | `:stan` | always |
| `:fit` | `:stan` | the traced model has ≥1 parameter and ≥1 likelihood term |
| `:prior_predictive` | `:stan` | every emitted observation is held out; returns the prior-only `StanProblem` for sampling |
| `:predict` | `:stan` | the Stan program emits ≥1 posterior-predictive draw **and** ≥1 BRM observation resolves to it |
| `:pointwise_loglik` | `:stan` | the Stan program emits ≥1 pointwise log-likelihood |
| `:replay` | `:brm` | the descriptor was built from a `@brm` builder (rebuild on a new dataframe, e.g. new subjects) |
| `:reprocess` | `:brm` | the declaration has no random-effect block, or every random-effect block has frozen same-group preprocessing (plain/`|ID|`/grouped-HSGP group indices or typed `mm(...)`; stratified `gr(g, by=b)` remains unsupported) |

`:replay` / `:reprocess` take the new dataframe positionally and return a NEW
`BRMDescriptor`; their `inputs` are the dataframe columns in `d.columns`, not
Stan data keys. `:reprocess` forwards both `freeze_constants=` and the checked
new-population `resample_groups=` CV/GQ re-emission described by
[`reprocess`](@ref).

The builder form's `held_out` keyword names one response, a collection of
responses, or `:all`. A partial selection removes only those likelihoods; the
remaining observations still offer `:fit`. `held_out=:all` removes every
likelihood, preserves predictive generated quantities, and derives
`:prior_predictive` instead of mislabelling the result as a fit. Sampling uses
the returned `StanProblem` exactly as a fitted problem; pass its unconstrained
draws to `:predict` for prior-predictive observations.

# Extension points

These three keywords let a consumer extend presentation and operations rather
than fork the descriptor:

- `operations` — a `name => …` mapping applied AFTER derivation:
  * a callable `(d; kwargs...) -> result` adds a new operation (or replaces an
    existing one's `run`), recorded with `origin = :override`;
  * a full `BRMOperation` replaces the entry outright;
  * `nothing` **suppresses** a derived operation (hide a button the surface
    should not show).
- `titles` — a `name => String` mapping relabelling any operation.
- `highlights` — an ordered collection of included Stan definition names, or
  `name => caption` pairs. Names resolve through StanBlocks' authoritative
  definition inventory and fail closed when absent; order and captions are
  presentation metadata only and do not change `d.id` or the executable model.

Overriding a name the model does not offer is allowed (that is how you *add*
one); suppressing a name that was never derived is a loud error, because it
means the caller is holding a stale operation list — exactly the parallel
registry this type exists to remove.

# Failing closed

Five cases raise rather than degrade, because each is unrecoverable in a
consumer that keys on names:

1. **A declaration target that is also a Stan data input.** Every consumer
   keys on the name (form field, result target, BridgeStan lookup), so this
   cannot be auto-resolved. Rename the binding.
2. **Two declarations resolving to the same Stan output.** Ambiguous
   provenance; the descriptor refuses to pick one.
3. **`brm_operation(d, name)` for an operation the model does not offer** —
   errors and names the operations it *does* offer, so the discovery never
   moves into the consumer.
4. **Suppressing an operation that was not derived** (see above).
5. **Selecting an absent or duplicate Stan definition highlight.** BRM accepts
   only names that StanBlocks reports in the executable definition inventory.

One case deliberately does **not** raise, because it is a legitimate model:
an observation whose predictive draw the Stan program does not emit is
recorded in `d.unpredictable` and simply excluded from `:predict`'s outputs.
If that leaves no draws at all, `:predict` is not offered. The consumer never
sees a predict button that would fail — which is the point.
"""
function brm_descriptor(plan_or_sb::Union{GenerativePlan,SBBRMI};
                        name::Union{Nothing,Symbol}=nothing,
                        operations=Dict{Symbol,Any}(),
                        titles=Dict{Symbol,String}(),
                        highlights=())
    plan = _brm_plan_of(plan_or_sb)
    stan = isnothing(name) ? StanBlocks.stan_descriptor(plan.model) :
                             StanBlocks.stan_descriptor(plan.model; name)
    _brm_descriptor(plan, stan, operations, titles, highlights)
end

function brm_descriptor(builder::Function, df;
                        mod::Module=@__MODULE__, cv_groups=Set{Symbol}(),
                        held_out=(),
                        name::Union{Nothing,Symbol}=nothing,
                        operations=Dict{Symbol,Any}(),
                        titles=Dict{Symbol,String}(),
                        highlights=())
    brm_descriptor(generative_plan(builder, df; mod, cv_groups, held_out);
                   name, operations, titles, highlights)
end

function _brm_descriptor(plan, stan, operations, titles, highlight_specs)
    brmi = plan.parent
    highlights = _brm_highlights(stan, highlight_specs)

    # --- the dataframe columns this declaration reads -----------------------
    # `data_columns` covers every column the formula references as a bare
    # predictor or grouping factor, but a RESPONSE is a `~` op, not a
    # DataColumn op, so it never appears there. The plan already knows every
    # response: an observation declaration's `data_source` IS its dataframe
    # column (including a plate-local alias, `kernel_y => dv`).
    df_columns = Set{Symbol}(data_columns(brmi))
    for d in plan.declarations
        d.role === :observation && !isnothing(d.data_source) || continue
        entry = get(plan.preproc, d.data_source, nothing)
        if entry isa PreprocEntry && entry.kind === :joint_response
            union!(df_columns, Symbol.(entry.raw_ref))
        else
            push!(df_columns, d.data_source)
        end
    end

    # --- inputs: the Stan data block + its dataframe provenance -------------
    inputs = BRMInput[]
    for i in stan.inputs
        col, tf = _brm_input_schema(plan, i.name, df_columns)
        push!(inputs, BRMInput(i.name, i.type, i.size, i.constraints,
                               i.observed, i.held_out, i.derived, i.inlined,
                               col, tf))
        isnothing(col) || push!(df_columns, col)   # e.g. a zscale()'d raw column
    end
    input_names = Set{Symbol}(i.name for i in stan.inputs)

    # --- declarations, indexed by the Stan name they resolve to -------------
    # Composition is BY NAME (stanblocks-use §30), through `_brm_stan_name`.
    # An observation is data, so what it *owns* among the outputs is its
    # predictive draw and its pointwise log-likelihood, not itself — and
    # StanBlocks' `source` is the authoritative link for both.
    #
    # Which observations does the traced model actually produce a draw for?
    # Read it off StanBlocks' `source` link, NOT off the declaration's `draw`
    # field: `draw` is a stable executor-facing NAME the plan assigns, not a
    # promise the program emits it (brm-use, the `generative_plan` section).
    # For a plate-nested observation the two legitimately differ.
    draw_sources = Set{Symbol}(o.source for o in stan.outputs
                               if o.generative === :draw && !isnothing(o.source))
    by_name = Dict{Symbol,GenerativeDeclaration}()
    unpredictable = Symbol[]
    for d in plan.declarations
        sname = _brm_stan_name(d)
        haskey(by_name, sname) && error(
            "brm_descriptor: ambiguous provenance — two declarations both resolve to " *
            "`$sname`. Rename one of them.")
        if d.role === :observation
            # The observation's Stan data key is its `data_source`; `target` is
            # the emitted SLIC binding, which for a plate-nested observation is
            # a plate-local alias (`kernel_y` for the column `dv`).
            # A held-out observation inside `kernel(...)` can disappear from
            # the executable data block altogether: activity analysis retains
            # only its `<source>_n` size for the generated draw. The plan's
            # held-out set is the authoritative provenance in that case.
            (d.target in input_names ||
             (!isnothing(d.data_source) &&
              (d.data_source in input_names || d.data_source in plan.held_out))) || error(
                "brm_descriptor: observation `$(d.target)` resolves to no data input of " *
                "the emitted model — the plan and the traced model disagree; " *
                "re-derive the plan.")
            (d.target in draw_sources ||
             (!isnothing(d.data_source) && d.data_source in draw_sources)) ||
                push!(unpredictable, d.target)
        else
            d.target in input_names && error(
                "brm_descriptor: `$(d.target)` is both a declared binding and a data input " *
                "of the emitted model. Every consumer keys on the name, so this cannot be " *
                "resolved automatically — rename one of them.")
        end
        by_name[sname] = d
    end
    targets = collect(keys(by_name))
    population_effects = _brm_population_effect_entries(brmi)
    covariance_factors = _brm_joint_covariance_factors(brmi)
    logical_outputs = _brm_logical_outputs(
        stan, by_name, targets, _brm_kernel_cell_values(brmi),
        Iterators.flatten(((e.logical for e in population_effects),
                           covariance_factors)))

    # Linear-predictor names come from the FORMULA, not from the emitted body:
    # `mu = pop_mu + r_mu_g` is an `=`, so no declaration binds it, yet it is
    # the most meaningful output in the model. `pop_<lp>` is the population
    # block sbimpl emits for `lp` (sbimpl.jl `_sb_linear_predictor!`), which is
    # how a population declaration gets back to the LP whose coefficients it
    # holds — derived forwards from the formula, never parsed off the name.
    # `pop_<emitted>` is keyed by the name sbimpl EMITS the design under, which
    # is the LINKED spelling for an LHS link transformation (`log(Vc) ~ 1 + x`
    # emits `pop_log_Vc_beta_pop`), while the value stays the PUBLIC LP name
    # `popcoefnames` takes (`:Vc`). Deriving it as `Symbol(:pop_, l.name)`
    # matched no emitted block for a linked LHS, so the coefficient vector
    # silently lost its `labels` — the one thing a consumer mounts a descriptor
    # for — while the same model written with an inert `log_Vc` name kept them.
    lps = Set{Symbol}(e.logical for e in population_effects)
    pop_lp = Dict{Symbol,Symbol}(e.block => e.logical for e in population_effects)

    # --- outputs ------------------------------------------------------------
    outputs = BRMOutput[]
    for o in stan.outputs
        decl = _brm_owner(o, by_name, targets)
        role = if !isnothing(decl) && decl.role === :observation
            o.generative === :pointwise_loglik ? :pointwise_loglik : :posterior_predictive
        elseif !isnothing(decl)
            _brm_declaration_role(decl)
        elseif o.name in covariance_factors
            :parameter
        elseif o.name in lps
            :linear_predictor
        elseif o.generative === :draw
            :posterior_predictive
        elseif o.generative === :pointwise_loglik
            :pointwise_loglik
        else
            :stan_derived
        end
        push!(outputs, BRMOutput(o.name, o.kind, o.type, o.size, o.constraints,
                                 o.generative, o.source, role, decl,
                                 get(logical_outputs, o.name, nothing), nothing,
                                 _brm_output_segments(o)))
    end
    outputs = _brm_label_population!(outputs, brmi, pop_lp)

    # --- schema -------------------------------------------------------------
    columns = Tuple(sort!(collect(df_columns)))

    # --- operations ---------------------------------------------------------
    ops = _brm_derive_operations(plan, stan, outputs, columns)
    ops = _brm_apply_overrides(ops, operations, titles)

    BRMDescriptor(stan.id, stan.name, sprint(show, brmi), plan, stan, highlights,
                  Tuple(inputs), Tuple(outputs), Tuple(ops), columns,
                  Tuple(unpredictable))
end

# Every operation StanBlocks derived, delegated verbatim, plus the two BRM-level
# replay operations. Nothing is listed: `stan.operations` is itself derived from
# what the traced model supports, and the two BRM ones are gated on the
# declaration.
function _brm_reprocess_supported(plan, outputs)
    any(o -> o.role === :random_effect, outputs) || return true

    ranef_declarations = [d for d in plan.declarations
                          if _brm_declaration_role(d) === :random_effect]
    isempty(ranef_declarations) && return false

    all(ranef_declarations) do d
        haskey(d.keywords, :group_idx) || return false
        idx_key = d.keywords.group_idx
        idx_key isa Symbol || return false
        entry = get(plan.preproc, idx_key, nothing)
        entry isa PreprocEntry &&
            entry.kind in (:group_index, :multi_membership)
    end
end

function _brm_all_observations_held_out(plan)
    observations = [d for d in plan.declarations if d.role === :observation]
    isempty(observations) && return false
    all(observations) do declaration
        source = declaration.data_source
        !isnothing(source) && source in plan.held_out
    end
end

function _brm_derive_operations(plan, stan, outputs, columns)
    ops = BRMOperation[]
    predictive = Symbol[o.name for o in outputs if o.role === :posterior_predictive]
    for so in stan.operations
        # `:predict` is only real if a BRM observation actually resolves to a
        # draw the program emits. `unpredictable` observations are already
        # excluded from `predictive`.
        so.name === :predict && isempty(predictive) && continue
        push!(ops, BRMOperation(so.name, so.title, Tuple(so.inputs),
                                Tuple(so.outputs), :stan,
                                (d; kwargs...) -> StanBlocks.stan_execute(
                                    d.stan, so.name; kwargs...)))
    end

    if _brm_all_observations_held_out(plan)
        instantiate = only(op for op in stan.operations
                           if op.name === :instantiate)
        parameters = Tuple(o.name for o in outputs if o.kind === :parameter)
        push!(ops, BRMOperation(
            :prior_predictive, "Sample the prior-predictive model",
            Tuple(instantiate.inputs), parameters, :stan,
            (d; kwargs...) -> StanBlocks.stan_execute(
                d.stan, :instantiate; kwargs...)))
    end

    if !isnothing(plan.builder)
        push!(ops, BRMOperation(
            :replay, "Rebuild the declaration on a new dataframe", columns, (), :brm,
            (d, new_df; highlights=d.highlights, kwargs...) -> brm_descriptor(
                generative_plan(d.plan, new_df); highlights, kwargs...)))
    end
    if _brm_reprocess_supported(plan, outputs)
        push!(ops, BRMOperation(
            :reprocess, "Re-run preprocessing on a new dataframe", columns, (), :brm,
            (d, new_df; freeze_constants::Bool=true, highlights=d.highlights,
             resample_groups=(), kwargs...) -> brm_descriptor(
                reprocess(d.plan, new_df; freeze_constants, resample_groups);
                highlights, kwargs...)))
    end
    ops
end

function _brm_apply_overrides(ops, operations, titles)
    derived = Set{Symbol}(o.name for o in ops)
    by_name = Dict{Symbol,BRMOperation}(o.name => o for o in ops)
    order = Symbol[o.name for o in ops]
    # Sorted so a `Dict` of overrides yields the same operation order every
    # time — a consumer rendering one button per operation should not see them
    # reshuffle between processes.
    for key in sort!(Symbol[Symbol(k) for k in keys(operations)])
        v = operations[key]
        if isnothing(v)
            key in derived || error(
                "brm_descriptor: cannot suppress operation `$key` — this model does not " *
                "offer it. Offered: $(sort(collect(derived))). A stale operation list is " *
                "exactly what this descriptor removes.")
            delete!(by_name, key)
            filter!(!=(key), order)
        elseif v isa BRMOperation
            haskey(by_name, key) || push!(order, key)
            by_name[key] = v
        else
            base = get(by_name, key, nothing)
            haskey(by_name, key) || push!(order, key)
            by_name[key] = BRMOperation(
                key,
                isnothing(base) ? String(key) : base.title,
                isnothing(base) ? () : base.inputs,
                isnothing(base) ? () : base.outputs,
                :override, v)
        end
    end
    for (k, t) in pairs(titles)
        key = Symbol(k)
        haskey(by_name, key) || error(
            "brm_descriptor: cannot retitle operation `$key` — not offered. " *
            "Offered: $(sort(collect(keys(by_name)))).")
        b = by_name[key]
        by_name[key] = BRMOperation(b.name, String(t), b.inputs, b.outputs, b.origin, b.run)
    end
    BRMOperation[by_name[k] for k in order]
end

# ---- accessors --------------------------------------------------------------

"""
    brm_columns(d::BRMDescriptor) -> Tuple{Vararg{Symbol}}

The dataframe columns this declaration reads — the schema a form should
collect for a `:replay` / `:reprocess`. Equivalent to `d.columns`.
"""
brm_columns(d::BRMDescriptor) = d.columns

"""
    brm_outputs(d::BRMDescriptor; logical=nothing, role=nothing, kind=nothing)
        -> Vector{BRMOutput}

Every emitted output matching the given semantic filters, in descriptor order.
Each filter accepts a `Symbol` or any collection of them, and an omitted filter
does not constrain. This is the **discovery** query — it can legitimately return
zero, one, or many; use [`brm_output`](@ref) when exactly one is required.

```julia
brm_outputs(d; role=:posterior_predictive)          # every predictive carrier
brm_outputs(d; logical=:pk_conc)                    # every carrier of one target
brm_outputs(d; role=(:parameter, :random_effect))   # either role
```

Filtering here is on BRM meaning (`logical` / `role`) or Stan representation
(`kind`) — never on the emitted NAME, which the compiler owns.
"""
function brm_outputs(d::BRMDescriptor; logical=nothing, role=nothing, kind=nothing)
    BRMOutput[o for o in d.outputs
              if _brm_matches(o.logical, logical) &&
                 _brm_matches(o.role, role) &&
                 _brm_matches(o.kind, kind)]
end

_brm_matches(_value, ::Nothing) = true
_brm_matches(value, wanted::Symbol) = value === wanted
_brm_matches(value, wanted) = any(w -> value === w, wanted)

"""
    brm_output(d::BRMDescriptor, logical::Symbol; role=nothing) -> BRMOutput

Return the unique emitted output that physically carries the BRM quantity
`logical`. This resolves BRM meaning to Stan representation without parsing an
emitter-owned name. For example, a ragged `loc ~ kernel(...)` result can resolve
to an output named `loc__pl_mem_1` while retaining `logical == :loc`.

`logical` is normally a declaration target. It is also the name of any value a
`kernel(...)` cell assigns — see the `logical` field on [`BRMOutput`](@ref) —
which is how to address a fitted noise-free in-cell location, with no
annotation on the term.

**One logical target legitimately has several carriers, and `role` is how you
pick one.** An observation `pk_conc ~ Normal(loc, sigma)` emits a
posterior-predictive twin *and* a pointwise-log-likelihood twin, both carrying
`logical === :pk_conc`. Ask for the one you mean:

```julia
brm_output(d, :pk_conc; role=:posterior_predictive)   # the *_gen carrier
brm_output(d, :pk_conc; role=:pointwise_loglik)       # the per-element loglik
```

The emitted names (`pk_conc_gen`, …) are StanBlocks' to choose and are NOT part
of this contract; that is the whole point of asking by role.

Fails loudly when the target has no emitted carrier or still maps to several;
the caller must never guess from descriptor order in either case. The
several-carriers message lists each candidate WITH its role, so the fix is the
`role=` to add rather than a name to hardcode.
"""
function brm_output(d::BRMDescriptor, logical::Symbol; role=nothing)
    found = brm_outputs(d; logical, role)
    length(found) == 1 && return only(found)
    qualifier = isnothing(role) ? "" : " with role `$role`"
    if isempty(found)
        available = unique!(Symbol[o.role for o in d.outputs if o.logical === logical])
        hint = isempty(available) ? "" :
            " That target does emit carriers with role(s) $(Tuple(available))."
        error("brm_descriptor: logical output `$logical`$qualifier has no emitted " *
              "posterior carrier in model `$(d.name)`.$hint")
    end
    # Name the OWNING DECLARATION alongside the role. `role` separates an
    # observation's two twins, but it cannot separate two `kernel(...)` cells
    # that happen to give a value the same name — both are `:group_block`, and
    # the owner is the only thing that differs. Reporting only the role there
    # would name a discriminator that does not discriminate.
    owners = Tuple((o.name, o.role,
                    isnothing(o.declaration) ? nothing : o.declaration.target)
                   for o in found)
    distinct_roles = length(unique(o.role for o in found)) > 1
    error("brm_descriptor: logical output `$logical`$qualifier spans several emitted " *
          "carriers (name, role, owner) $owners; " *
          (distinct_roles ? "pass `role=` to select one, or use" : "these share a role, so " *
           "`role=` cannot separate them — use") *
          " `brm_outputs` to take them all and select on `declaration.target`.")
end

_brm_emitted_coordinates(output::BRMOutput, constrained_names) = begin
    stem = String(output.name)
    prefix = stem * "."
    Int[i for (i, name) in enumerate(constrained_names)
        if string(name) == stem || startswith(string(name), prefix)]
end

# Public term labels are derived forwards from the formula term with the same
# rules as `_sb_predictor_term!` (`sbimpl.jl`), while their configurable
# parameter vocabulary is the same closed set the term-prior emitter owns. The
# binding names below are used only to validate the declaration-owned output,
# never exposed as a consumer address.
_brm_term_label(::typeof(mo), t) =
    Symbol(:mo_, name(_sb_named_inner(:mo, only(getargs(t)))))
_brm_term_label(::typeof(mo1), t) =
    Symbol(:mo1_, name(_sb_named_inner(:mo1, only(getargs(t)))))
function _brm_term_label(::typeof(hsgp), t)
    axes = Tuple(name(_sb_named_inner(:hsgp, a)) for a in getargs(t))
    suffix = join(string.(axes), "_")
    kw = getkwargs(t)
    haskey(kw, :by) || return Symbol(:hsgp_, suffix)
    group = _as_named_column(kw.by)
    isnothing(group) && error(
        "brm_descriptor: `hsgp(...; by=...)` group is not a named column.")
    Symbol(:hsgp_, suffix, :_by_, name(group))
end
_brm_term_label(f, t) = _sb_term_key(t)
_brm_term_label(f, t, _target) = _brm_term_label(f, t)
_brm_term_label(::typeof(dar), t, target) =
    Symbol(:dar_, target, :_, name(_sb_named_inner(:dar, only(getargs(t)))))

_brm_term_parameter_bindings(::typeof(mo), _t) =
    (; simplex=:simplex_incr)
_brm_term_parameter_bindings(::typeof(mo1), _t) =
    (; simplex=:simplex_incr)
function _brm_term_parameter_bindings(::typeof(hsgp), t)
    rho = _sb_gp_iso(getkwargs(t), :hsgp) ? :rho_iso : :rho
    haskey(getkwargs(t), :by) && return (; length_scale=rho, sd=:sigma)
    (; length_scale=rho, sd=:sigma, basis_weights=:beta_raw)
end
_brm_term_parameter_bindings(::typeof(dar), _t) =
    (; ar=:beta, sd=:sigma, innovations=:z)
_brm_term_parameter_bindings(_f, _t) = NamedTuple()

function _brm_term_coordinate_entries(brmi, logical::Symbol)
    predictors = [lp for lp in linear_predictors(brmi) if lp.name === logical]
    length(predictors) == 1 || return NamedTuple[]
    link = only(predictors).link_lhs_fn
    target = _sb_lp_emitted_name(logical, link)
    entries = NamedTuple[]
    for terms in values(_sb_term_address_map(brmi, logical)), t in terms
        push!(entries, (; term=_brm_term_label(getf(t), t, target), value=t, link))
    end
    entries
end

_brm_resolved_dimension(plan, x::Integer) = Int(x)
_brm_resolved_dimension(plan, x::Symbol) = begin
    value = get(plan.data, x, nothing)
    value isa Integer ? Int(value) : nothing
end
_brm_resolved_dimension(_plan, _x) = nothing

function _brm_output_coordinate_count(plan, output::BRMOutput)
    isempty(output.size) && return 1
    dims = [_brm_resolved_dimension(plan, x) for x in output.size]
    any(isnothing, dims) && return nothing
    prod(dims)
end

_brm_term_coordinate_count(_f, _t, _parameter, plan, output) =
    _brm_output_coordinate_count(plan, output)
_brm_term_coordinate_count(::typeof(hsgp), t, ::Val{:length_scale}, _plan, _output) =
    _sb_gp_iso(getkwargs(t), :hsgp) ? 1 : length(getargs(t))
_brm_term_coordinate_count(::typeof(hsgp), _t, ::Val{:sd}, _plan, _output) = 1
function _brm_term_coordinate_count(::typeof(hsgp), t,
                                    ::Val{:basis_weights}, _plan, _output)
    kw = getkwargs(t)
    K, _ = _sb_hsgp_options(kw, length(getargs(t)))
    # The periodic basis carries a cosine AND a sine column per harmonic.
    _sb_gp_cov(kw, :hsgp) === :periodic ? 2 * only(K) : prod(K)
end

"""
    brm_term_coordinates(d::BRMDescriptor, logical::Symbol, constrained_names;
                         term::Symbol, parameter::Symbol)

Resolve one formula term's internal sampled parameter to its exact constrained
posterior coordinates. `logical` names the linear predictor and `term` is the
public term-output label BRM derives from the formula (for example
`:mo_op_diet` or `:hsgp_op_log_dose`). Supported parameter roles are:

| term | `parameter` |
| --- | --- |
| `mo(...)` / `mo1(...)` | `:simplex` |
| ungrouped `hsgp(...)` | `:length_scale`, `:sd`, `:basis_weights` |
| grouped `hsgp(...; by=...)` | `:length_scale`, `:sd` |
| `dar(...)` | `:ar`, `:sd`, `:innovations` |

The returned named tuple contains `logical`, `term`, `parameter`, the owning
[`BRMOutput`](@ref), `coordinates`, and the formula LHS `link` / `inverse_link`.
For example:

```julia
brm_term_coordinates(d, :log_F, constrained_names;
                     term=:mo_op_diet, parameter=:simplex)
brm_term_coordinates(d, :log_F, constrained_names;
                     term=:hsgp_op_log_dose, parameter=:basis_weights)
```

Resolution follows the formula term to its logical term output, then selects a
parameter owned by that declaration. Consumers never construct or parse the
compiler-owned carrier name. Missing or duplicate predictors/terms/owners,
unsupported parameter roles, and descriptor/artifact coordinate drift all
error rather than selecting by descriptor order.

Generated-aware: for a response-free `regime="prior"` program (no observation
`~`) StanBlocks re-draws every term carrier from its prior into generated
quantities under the same constrained name, so pass `constrained_names` built
with `include_gq=true` (`BridgeStan.param_names(prob.model; include_tp=true,
include_gq=true)`) and this resolves the GQ carrier exactly as it resolves the
sampled `:parameter` in an ordinary fit; a sampled carrier is preferred when
both exist.
"""
function brm_term_coordinates(d::BRMDescriptor, logical::Symbol,
                              constrained_names;
                              term::Symbol, parameter::Symbol)
    predictors = [lp for lp in linear_predictors(d.plan.parent)
                  if lp.name === logical]
    length(predictors) == 1 || error(
        "brm_descriptor: logical predictor `$logical` resolves to " *
        "$(length(predictors)) formula declarations; expected exactly one " *
        "public term-parameter address.")

    all_entries = _brm_term_coordinate_entries(d.plan.parent, logical)
    entries = [e for e in all_entries if e.term === term]
    length(entries) == 1 || error(
        "brm_descriptor: term `$term` occurs $(length(entries)) times on logical " *
        "predictor `$logical`; available term labels are " *
        "$(Tuple(sort!(unique(e.term for e in all_entries), by=string))).")
    entry = only(entries)

    owners = BRMOutput[
        o for o in d.outputs
        if o.logical === term && !isnothing(o.declaration)
    ]
    length(owners) == 1 || error(
        "brm_descriptor: term `$term` on logical predictor `$logical` has " *
        "$(length(owners)) logical output owners; expected exactly one.")
    owner = only(owners).declaration

    bindings = _brm_term_parameter_bindings(getf(entry.value), entry.value)
    haskey(bindings, parameter) || error(
        "brm_descriptor: term `$term` on logical predictor `$logical` exposes " *
        "no parameter role `$parameter`; available roles are $(keys(bindings)).")
    binding = getproperty(bindings, parameter)
    expected = Symbol(_brm_stan_name(owner), :_, binding)
    idxs = _brm_carrier_indices(d.outputs,
               o -> !isnothing(o.declaration) &&
                    o.declaration.target === owner.target && o.name === expected)
    length(idxs) == 1 || error(
        "brm_descriptor: term `$term` on logical predictor `$logical` resolves " *
        "parameter role `$parameter` to $(length(idxs)) declaration-owned " *
        "posterior carriers; expected exactly one. Re-reflect the model that " *
        "produced the posterior draws.")
    output = d.outputs[only(idxs)]

    coordinates = _brm_emitted_coordinates(output, constrained_names)
    expected_count = _brm_term_coordinate_count(
        getf(entry.value), entry.value, Val(parameter), d.plan, output)
    if isnothing(expected_count)
        isempty(coordinates) && error(
            "brm_descriptor: term `$term` parameter role `$parameter` resolves " *
            "to emitted `$(output.name)`, but that carrier is absent from the " *
            "supplied constrained names. Re-reflect the model that produced " *
            "the posterior draws.")
    elseif length(coordinates) != expected_count
        error("brm_descriptor: term `$term` parameter role `$parameter` owns " *
              "$expected_count constrained coordinates but resolves to " *
              "$(length(coordinates)). Re-reflect the model that produced the " *
              "posterior draws.")
    end

    (; logical, term, parameter, output, coordinates,
       link=entry.link, inverse_link=InverseFunctions.inverse(entry.link))
end

# Categorical population terms own separate `_sb_cat` parameter blocks rather
# than elements of `beta_pop`. Recover their public level semantics from the
# formula term plus the fitted preprocessing record: the former owns the
# predictor/ref address, while the latter is the frozen level order that drove
# the sampled K-1 vector. The emitted block name is used only to join two
# producer-owned descriptor records; it is never exposed as a consumer address.
function _brm_categorical_effect_entry(plan, emitted_lp::Symbol,
                                       term::NamedColumn)
    isnothing(_sb_cat_levels(term)) && return nothing
    address = name(term)
    _brm_categorical_effect_entry(
        plan, emitted_lp, address, address, identity)
end

function _brm_categorical_effect_entry(plan, emitted_lp::Symbol,
                                       term::ExprColumn)
    getf(term) === factor || return nothing
    lowered = only(_sb_terms(term))
    lowered isa NamedColumn && !isnothing(_sb_cat_levels(lowered)) || return nothing

    args = getargs(term)
    length(args) == 1 || return nothing
    predictor = _as_named_column(only(args))
    isnothing(predictor) && return nothing
    ref = _as_integer(get(getkwargs(term), :ref, 1))
    isnothing(ref) && return nothing
    decode = ref == 1 ? identity :
        level -> level == 1 ? ref : level == ref ? 1 : level
    _brm_categorical_effect_entry(
        plan, emitted_lp, name(lowered), name(predictor), decode)
end

_brm_categorical_effect_entry(_plan, _emitted_lp::Symbol, _term) = nothing

function _brm_categorical_effect_entry(plan, emitted_lp::Symbol,
                                       address::Symbol, predictor::Symbol,
                                       decode)
    key = Symbol(address, :_idx)
    preproc = get(plan.preproc, key, nothing)
    (!isnothing(preproc) && preproc.kind === :factor) || error(
        "brm_descriptor: categorical predictor `$predictor` resolves to no " *
        "fitted factor preprocessing record at `$key`. Re-reflect the model " *
        "that produced the posterior draws.")
    levels = collect(preproc.const_)
    isempty(levels) && error(
        "brm_descriptor: categorical predictor `$predictor` has an empty " *
        "fitted level set.")

    n_levels_key = Symbol(address, :_n_levels)
    n_levels = get(plan.data, n_levels_key, nothing)
    n_levels isa Integer && n_levels == length(levels) || error(
        "brm_descriptor: categorical predictor `$predictor` has " *
        "$(length(levels)) frozen levels but emitted level count " *
        "$(repr(n_levels)) at `$n_levels_key`. Re-reflect the model that " *
        "produced the posterior draws.")

    decoded = map(decode, levels)
    (; predictor, address,
       emitted=_sb_cat_block_name(emitted_lp, address),
       reference_level=first(decoded),
       nonreference_levels=decoded[2:end])
end

function _brm_categorical_effect_entries(d::BRMDescriptor, logical::Symbol,
                                         link)
    op = linear_predictor_op(d.plan.parent, logical)
    isnothing(op) && return NamedTuple[]
    _, rhs = getargs(op, 2)
    emitted_lp = _sb_lp_emitted_name(logical, link)
    entries = NamedTuple[]
    for term in _brm_additive_terms(rhs)
        entry = _brm_categorical_effect_entry(d.plan, emitted_lp, term)
        isnothing(entry) || push!(entries, entry)
    end
    entries
end

function _brm_categorical_effect_coordinates(d::BRMDescriptor,
                                             logical::Symbol,
                                             coefficient::Symbol,
                                             constrained_names,
                                             entry, block::Symbol)
    entries = [e for e in _brm_categorical_effect_entries(d, logical, entry.link)
               if e.emitted === block]
    length(entries) == 1 || error(
        "brm_descriptor: categorical address `$coefficient` on logical " *
        "predictor `$logical` resolves to $(length(entries)) fitted formula " *
        "terms; expected exactly one.")
    categorical = only(entries)

    # `_sb_cat` and `_sb_cat_normal` both bind the contrast vector as `beta`.
    # Compose that BRM-owned binding with the formula-derived block name, just
    # as `brm_term_coordinates` composes a term's declared parameter binding
    # with its owner. Selecting every declaration internal is too broad: an
    # R2D2 categorical declaration also owns its derived scale, and future
    # lowering may add further internals without changing the contrast carrier.
    # This is a forward emitter rule, not parsing of a posterior name.
    expected = Symbol(block, :_beta)
    idxs = _brm_carrier_indices(d.outputs,
               o -> !isnothing(o.declaration) && o.declaration.target === block &&
                    o.name === expected)
    length(idxs) == 1 || error(
        "brm_descriptor: categorical address `$coefficient` on logical " *
        "predictor `$logical` resolves to block `$block`, which owns " *
        "$(length(idxs)) parameter carriers; expected exactly one.")
    output = d.outputs[only(idxs)]

    coordinates = _brm_emitted_coordinates(output, constrained_names)
    expected_count = length(categorical.nonreference_levels)
    length(coordinates) == expected_count || error(
        "brm_descriptor: categorical address `$coefficient` on logical " *
        "predictor `$logical` owns $expected_count treatment contrasts but " *
        "resolves to $(length(coordinates)) constrained coordinates. " *
        "Re-reflect the model that produced the posterior draws.")
    contrasts = [
        (; nonreference_level=level,
           reference_level=categorical.reference_level,
           coordinate)
        for (level, coordinate) in
            zip(categorical.nonreference_levels, coordinates)
    ]

    (; logical, coefficient, predictor=categorical.predictor, output,
       coordinates, contrasts,
       reference_level=categorical.reference_level,
       nonreference_levels=categorical.nonreference_levels,
       link=entry.link,
       inverse_link=InverseFunctions.inverse(entry.link))
end

"""
    brm_population_effect_coordinates(d::BRMDescriptor, logical::Symbol,
                                      constrained_names;
                                      coefficient=:Intercept)

Resolve one formula-level population coefficient or categorical contrast block
to its exact constrained posterior coordinates without constructing or parsing
an emitted `pop_*` / `cat_*` name. Returns a named tuple with:

- `logical` / `coefficient` — the public formula address;
- `output` — the owning coefficient or contrast [`BRMOutput`](@ref);
- `coordinates` — the matching indices in `constrained_names`;
- `link` — the function applied on the formula LHS (`identity`, `log`, …);
- `inverse_link` — the transform from the fitted linear-predictor scale back to
  the declared quantity (`identity`, `exp`, …).

A categorical predictor is addressed by the formula column, just like its
`effect(logical, column)` prior. Its result additionally contains `predictor`,
the frozen `reference_level` and ordered `nonreference_levels`, plus
`contrasts`, which pairs every non-reference level with its reference level and
exact constrained coordinate. For example, `coefficient=:indication` resolves
the K-1 block in `log(Vc) ~ 1 + indication`; the block's emitted spelling stays
private. A `factor(g; ref=3)` term remains addressable as `coefficient=:g` and
reports the fitted reference and contrast order after recoding.

For `log(Vc) ~ 1 + weight`, ask for `logical=:Vc`. The returned intercept is on
the log scale, `link === log`, and `inverse_link === exp`; the emitted
`pop_log_Vc_beta_pop` spelling remains private.

Resolution is fail-closed. Missing or duplicate logical predictors, missing or
duplicate population carriers, unavailable/duplicate coefficient labels, and
descriptor/artifact coordinate drift all error rather than selecting by
descriptor order.

Generated-aware: for a response-free `regime="prior"` program (no observation
`~`), the numeric coefficient and categorical contrast carriers move from
`parameters` into generated quantities under the same constrained names, so
pass `constrained_names` built with `include_gq=true`
(`BridgeStan.param_names(prob.model; include_tp=true, include_gq=true)`) and
this resolves the GQ carrier exactly as it resolves the sampled `:parameter` in
an ordinary fit — including the coefficient `labels`, so a prior-only descriptor
keeps its addressable population coordinates. A sampled carrier is preferred
when both exist.
"""
function brm_population_effect_coordinates(d::BRMDescriptor, logical::Symbol,
                                           constrained_names;
                                           coefficient::Symbol=:Intercept)
    entries = [e for e in _brm_population_effect_entries(d.plan.parent)
               if e.logical === logical]
    length(entries) == 1 || error(
        "brm_descriptor: logical predictor `$logical` resolves to " *
        "$(length(entries)) formula declarations; expected exactly one public " *
        "population-effect address.")
    entry = only(entries)

    categorical = _sb_cat_address_map(d.plan.parent, logical)
    categorical_block = get(categorical, coefficient, nothing)
    numeric_labels = _brm_labels(d.plan.parent, logical)
    numeric_matches = isnothing(numeric_labels) ? Int[] :
                      findall(==(coefficient), numeric_labels)
    if !isnothing(categorical_block)
        isempty(numeric_matches) || error(
            "brm_descriptor: coefficient `$coefficient` on logical predictor " *
            "`$logical` names both a beta_pop coefficient and a categorical " *
            "contrast block; use an unambiguous formula address.")
        return _brm_categorical_effect_coordinates(
            d, logical, coefficient, constrained_names, entry,
            categorical_block)
    end

    # The coefficient vector is the block INTERNAL (name !== target); the block
    # RESULT (name === target) is the linear-predictor value. `name !== block`
    # is load-bearing for a prior program, where the block result assignment is
    # ALSO a generated quantity under the same target (see _brm_carrier_indices).
    idxs = _brm_carrier_indices(d.outputs,
               o -> o.role === :population_effect && !isnothing(o.declaration) &&
                    o.declaration.target === entry.block && o.name !== entry.block)
    length(idxs) == 1 || error(
        "brm_descriptor: logical predictor `$logical` resolves to population " *
        "block `$(entry.block)`, which owns $(length(idxs)) parameter " *
        "carriers; expected exactly one.")
    output = d.outputs[only(idxs)]

    labels = output.labels
    isnothing(labels) && error(
        "brm_descriptor: population carrier `$(output.name)` for logical " *
        "predictor `$logical` has no coefficient labels; this formula shape " *
        "does not expose a stable population-coordinate address.")
    label_indices = findall(==(coefficient), labels)
    available = unique!(vcat(copy(labels), sort!(collect(keys(categorical)))))
    length(label_indices) == 1 || error(
        "brm_descriptor: coefficient `$coefficient` occurs $(length(label_indices)) " *
        "times on logical predictor `$logical`; available labels are " *
        "$(Tuple(available)).")

    all_coordinates = _brm_emitted_coordinates(output, constrained_names)
    length(all_coordinates) == length(labels) || error(
        "brm_descriptor: population carrier `$(output.name)` has " *
        "$(length(labels)) coefficient labels but resolves to " *
        "$(length(all_coordinates)) constrained coordinates. Re-reflect the " *
        "model that produced the posterior draws.")

    (; logical, coefficient, output,
       coordinates=all_coordinates[label_indices],
       link=entry.link,
       inverse_link=InverseFunctions.inverse(entry.link))
end

"""
    brm_ranef_sd_coordinates(d::BRMDescriptor, logical::Symbol, constrained_names;
                             id::Symbol, coefficient=:Intercept)

Resolve one margin's FITTED between-group standard deviation (`tau`) in a shared
correlated random-effect block to its exact constrained posterior coordinate,
without constructing or parsing an emitter-owned `<binding>_tau` name. This is
the random-effect-scale sibling of
[`brm_population_effect_coordinates`](@ref): that resolver gives the population
LOCATION of a margin, this one gives its group SCALE, so a consumer can form a
population-predictive quantity such as `inverse_link(intercept_draw + sd_draw *
z)` with no name interpretation.

`logical` is the margin's PUBLIC linear-predictor name — the bare inner name for
an LHS-link LP (`:Vc` for `log(Vc) ~ …`); `id` is the brms-style `|ID|` bucket
symbol (`:p` for `(1 | p | subject)`); and `coefficient` is the margin's
coefficient label (`:Intercept` for an intercept-only block, a slope column for
a correlated slope). These are exactly the ordered addresses
[`ranefcoefnames`](@ref)`(brmi, id)` reports, in the same order the emitted
`tau` vector uses.

The returned named tuple carries `logical`, `id`, `coefficient`, the owning
[`BRMOutput`](@ref), `coordinates` (the one matching index into
`constrained_names`), and the margin predictor's LHS `link` / `inverse_link` —
the SAME link the population-effect resolver returns for that predictor, so the
intercept and the scale live on one scale and compose directly:

```julia
d  = brm_descriptor(builder, df; mod=@__MODULE__)
mu  = brm_population_effect_coordinates(d, :Vc, constrained_names;
                                        coefficient=:Intercept)
tau = brm_ranef_sd_coordinates(d, :Vc, constrained_names;
                               id=:p, coefficient=:Intercept)
loc   = constrained_draws[:, only(mu.coordinates)]
scale = constrained_draws[:, only(tau.coordinates)]
Vc_upper = tau.inverse_link.(loc .+ scale .* quantile(Normal(), 0.975))
```

Both the sampled-scale families (`sd(:, id) ~ Exponential(...)` and the default
half-normal) and the derived-scale R2D2 families (`sd(:, id) ~ r2d2(...)`, whose
`tau` is a transformed parameter — so pass `constrained_names` /
`constrained_draws` that INCLUDE transformed parameters) are covered; the
resolver follows the block's emitted scale carrier either way, and centering
does not change the carrier.

It fails closed — never selecting by descriptor order or parsing a
compiler-owned name — when the `|ID|` bucket, margin, or scale carrier is
absent, ambiguous, or has drifted from the supplied constrained names. A block
whose scale is not a per-margin `tau` vector — a scalar `(1 | g)` intercept
(scale `exp(log_scale)`) or a stratified `gr(g, by=b)` block (one `tau` per
stratum) — is refused with a message naming the family, rather than returning a
coordinate of a different quantity.

Generated-aware: for a response-free `regime="prior"` program (no observation
`~`) StanBlocks re-draws the shared-`|ID|` scale from `brm_ranef_sd_rng` into
generated quantities under the same `<binding>_tau` name (the resolver already
follows that name + `:random_effect` role, so no kind branch is needed). Pass
`constrained_names` with `include_gq=true` (and, as for the R2D2 derived-scale
family, `include_tp=true`) and it resolves the GQ `tau` carrier exactly as it
resolves the sampled one in a fit.
"""
function brm_ranef_sd_coordinates(d::BRMDescriptor, logical::Symbol,
                                  constrained_names;
                                  id::Symbol, coefficient::Symbol=:Intercept)
    brmi = d.plan.parent

    # 1. The shared block for this `|ID|` bucket.
    all_blocks = ranef_blocks(d.plan)
    blocks = [b for b in all_blocks if b.id === id]
    if isempty(blocks)
        available = unique!(Symbol[b.id for b in all_blocks if !isnothing(b.id)])
        error("brm_descriptor: no shared random-effect `|ID|` block `$id`. This " *
              "model's shared blocks are: " *
              (isempty(available) ? "(none)" : join(string.(available), ", ")) *
              ". A plain `(… | g)` block has no `|ID|` and is not addressed here.")
    end
    length(blocks) == 1 || error(
        "brm_descriptor: `|ID|` bucket `$id` identifies $(length(blocks)) " *
        "random-effect blocks with different grouping factors; the ID-only " *
        "address is ambiguous.")
    block = only(blocks)

    # 2. The margin's index in `ranefcoefnames` order, which is the `tau`-vector
    # order — never a position guess and never a parsed carrier name.
    margins = ranefcoefnames(brmi, id)
    isnothing(margins) && error(
        "brm_descriptor: `ranefcoefnames(brmi, :$id)` is `nothing`; `$id` names " *
        "no addressable shared random-effect block.")
    matches = findall(
        m -> m.predictor === logical && m.coefficient === coefficient, margins)
    length(matches) == 1 || error(
        "brm_descriptor: margin `(predictor=$logical, coefficient=$coefficient)` " *
        "occurs $(length(matches)) times in block `|$id|`; its ordered margins " *
        "are $(Tuple(margins)).")
    margin_index = only(matches)

    # 3. The block's emitted scale carrier. `_RANEF_FAMILIES` (src/prediction.jl)
    # is the ONE place coupled to the emitted names.
    spec = _RANEF_FAMILIES[block.family]
    tau_suffix = get(spec, :tau, nothing)
    isnothing(tau_suffix) && error(
        "brm_descriptor: random-effect block `$(block.binding)` (`|$id|`, family " *
        "`$(block.family)`) has no per-margin `tau` scale vector to resolve. A " *
        "scalar `(1 | g)` intercept parameterises its scale as `exp(log_scale)` " *
        "and a stratified `gr(g, by=b)` block emits one `tau` per stratum; " *
        "neither is a single per-margin coordinate.")
    tau_name = Symbol(block.binding, :_, tau_suffix)

    outputs = BRMOutput[
        o for o in d.outputs
        if o.role === :random_effect && o.name === tau_name &&
           !isnothing(o.declaration) && o.declaration.target === block.binding
    ]
    length(outputs) == 1 || error(
        "brm_descriptor: random-effect block `$(block.binding)` (`|$id|`) resolves " *
        "its scale carrier to `$tau_name`, which owns $(length(outputs)) descriptor " *
        "outputs; expected exactly one. Re-reflect the model that produced the " *
        "posterior draws.")
    output = only(outputs)

    coordinates = _brm_emitted_coordinates(output, constrained_names)
    length(coordinates) == block.n_terms || error(
        "brm_descriptor: scale carrier `$tau_name` for block `|$id|` owns " *
        "$(block.n_terms) marginal SDs but resolves to $(length(coordinates)) " *
        "constrained coordinates. Re-reflect the model that produced the " *
        "posterior draws.")

    predictors = [lp for lp in linear_predictors(brmi) if lp.name === logical]
    length(predictors) == 1 || error(
        "brm_descriptor: margin predictor `$logical` resolves to " *
        "$(length(predictors)) linear predictors; expected exactly one.")
    link = only(predictors).link_lhs_fn

    (; logical, id, coefficient, output,
       coordinates=[coordinates[margin_index]],
       link, inverse_link=InverseFunctions.inverse(link))
end

"""
    brm_output_coordinates(d::BRMDescriptor, logical::Symbol, constrained_names;
                           role=nothing) -> Vector{Int}

Resolve a logical BRM output to its columns in BridgeStan's constrained
`param_names` (or an equivalent posterior name vector). Resolution first uses
[`brm_output`](@ref) to obtain the descriptor's emitted name, then matches only
that exact name or its documented container coordinates (`name.1`,
`name.1.1`, …). It never parses a compiler-owned plate suffix.

`role` disambiguates a target with several carriers, exactly as on
[`brm_output`](@ref) — a posterior-predictive slice of an observation is

```julia
brm_output_coordinates(d, :pk_conc, param_names; role=:posterior_predictive)
```

The returned integers index `constrained_names` in their existing order. For a
ragged carrier they compose with that output's `segments`: group `g` is
`coordinates[segments[g-1]+1 : segments[g]]`.

A missing carrier is an error, which catches descriptor/artifact drift instead
of returning an empty posterior slice.
"""
function brm_output_coordinates(d::BRMDescriptor, logical::Symbol,
                                constrained_names; role=nothing)
    output = brm_output(d, logical; role)
    coordinates = _brm_emitted_coordinates(output, constrained_names)
    isempty(coordinates) && error(
        "brm_descriptor: logical output `$logical` resolves to emitted " *
        "`$(output.name)`, but that carrier is absent from the supplied constrained " *
        "names. Re-reflect the model that produced the posterior draws.")
    coordinates
end

"""
    brm_operation(d::BRMDescriptor, name::Symbol) -> BRMOperation

The named operation. **Fails closed**: an operation this model does not offer
errors and names the ones it does, so the discovery never moves into the
consumer.
"""
function brm_operation(d::BRMDescriptor, name::Symbol)
    for op in d.operations
        op.name === name && return op
    end
    error("brm_descriptor: model `$(d.name)` does not offer operation `$name`. " *
          "Offered: $([op.name for op in d.operations]).")
end

"""
    brm_execute(d::BRMDescriptor, name::Symbol, args...; kwargs...)

Run a derived operation.

```julia
brm_execute(d, :transpile)                       # the Stan source
prob = brm_execute(d, :fit)                      # a BridgeStan-backed StanProblem
prior_prob = brm_execute(prior_d, :prior_predictive) # prior-only StanProblem
brm_execute(d, :predict; problem=prob, draws=theta_unc, seed=1234)
brm_execute(d, :replay, new_df)                  # a NEW BRMDescriptor
```

`:stan`-origin operations forward to StanBlocks' `stan_execute` (data keywords
re-bind inputs; `:predict` requires `draws` and `seed`).
`:prior_predictive` delegates to StanBlocks' `:instantiate`: like `:fit`, it
returns the `StanProblem` a sampler consumes, without calling a model that has
no likelihood a fit. `:brm`-origin operations take the new dataframe
positionally and return a new descriptor.
Unknown names fail closed via [`brm_operation`](@ref).
"""
brm_execute(d::BRMDescriptor, name::Symbol, args...; kwargs...) =
    brm_operation(d, name).run(d, args...; kwargs...)

# ---- display ----------------------------------------------------------------

Base.show(io::IO, d::BRMDescriptor) = begin
    print(io, "BRMDescriptor `", d.name, "` (id ", d.id, ")\n")
    print(io, "  formula:    ", replace(strip(d.formula), "\n" => "\n              "), "\n")
    print(io, "  columns:    ", join(d.columns, ", "), "\n")
    print(io, "  inputs:     ", join(required_brm_inputs(d), ", "), "\n")
    print(io, "  operations: ", join((op.name for op in d.operations), ", "), "\n")
    isempty(d.highlights) ||
        print(io, "  highlights: ", join((h.name for h in d.highlights), ", "), "\n")
    isempty(d.unpredictable) ||
        print(io, "  no predictive draw for: ", join(d.unpredictable, ", "), "\n")
    for o in d.outputs
        print(io, "  ", o.name, " :: ", o.role, " (", o.kind, "/", o.generative, ")")
        isnothing(o.logical) || o.name === o.logical || print(io, " => ", o.logical)
        isnothing(o.labels) || print(io, " [", join(o.labels, ", "), "]")
        print(io, "\n")
    end
end

"""
    required_brm_inputs(d::BRMDescriptor) -> Vector{Symbol}

The Stan data keys a consumer must actually supply — the
neither-`derived`-nor-`inlined` subset of `d.inputs`. The BRM analogue of
StanBlocks' `required_inputs`; for the *dataframe* schema use
[`brm_columns`](@ref).
"""
required_brm_inputs(d::BRMDescriptor) =
    Symbol[i.name for i in d.inputs if !i.derived && !i.inlined]
