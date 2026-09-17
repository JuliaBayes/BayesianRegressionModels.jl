# The intermediate representation

Every `@brm` block parses to a typed intermediate representation before
anything is emitted for a backend. That representation is what the **IR**
pane — the second pane of every generated model comparison on this site —
shows, automatically and everywhere. This page explains how to read it.

## From formula block to `BRMI`

`@brm` lowers a formula block to a [`BRMI`](@ref): a `NamedTuple` of
operations keyed by left-hand-side name. Each entry is either a **statement**
(`loc ~ 1 + age`, `y ~ Normal(loc, err)`) or a **data** column with its
element type and length. Priors are first-class operations too, addressed by
`effect(...)` / `sd(...)` / `cor(...)` keys, so a model and its priors
compose with plain `Base.merge`.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
ir_demo_model = (@brm begin
    y ~ Normal(loc, err)
    loc ~ 1 + age + (1 | subj)
    err ~ Exponential(1)
end)((;
    age=[21.0, 38.0, 55.0, 29.0, 47.0, 61.0],
    subj=[1, 1, 2, 2, 3, 3],
    y=[0.2, 1.1, -0.4, 0.7, 1.4, 1.0],
))
""", :ir_demo_model; title="IR demo model")
```

Read the IR pane top to bottom: statements first, then the data each
statement consumes. Grouping parentheses stay where Julia's precedence
needs them (`(1 | subj)` binds looser than `~`), redundant ones are
dropped.

## The lowering chain

One `BRMI` feeds every backend through two further stages:

- [`VBRMI`](@ref) — the prepared, modification-frozen model: prior
  addresses matched, group declarations resolved, fitted
  preprocessing/replay recorded. Its summary prints the flat parameter
  dimension plus the materialized columns and blocks.
- [`SBBRMI`](@ref) / [`TuringBRMI`](@ref) — the backend lowerings shown in
  the remaining panes: the StanBlocks model and Stan source on one side,
  the native `Turing` model on the other.

Nothing in the chain re-parses the formula: backends consume the same
operations, so the IR pane and the emitted panes cannot drift apart —
they are rendered from one object in one build step.

## Introspection entry points

The [`API`](@ref api) documents the query surface over these types,
including prior lookup (`term_priors`, `effect_priors`) and the
`brm_descriptor` views each backend exposes. Start from the IR pane when
an emitted backend surprises you: if a statement or data column is wrong
there, it is wrong in every pane below it.
