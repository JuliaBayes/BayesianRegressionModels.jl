# Progress reporting through Treebars, a weak dependency. A `progress` argument
# is `nothing` (no reporting), a Treebars node (the call builds its own node
# under it) or a Treebars backend symbol such as `:state` (the call starts a new
# root). These hooks run their function without reporting for `nothing`;
# BayesianRegressionModelsTreebarsExt implements them for Treebars nodes and
# backend symbols with Treebars' own wrappers.

# `f(node)` under a new node `description` below `progress`.
_brm_with_progress(f, ::Nothing; description) = f(nothing)
_brm_with_progress(f, progress; description) = throw(ArgumentError(
    "progress: expected nothing, a Treebars node, or a Treebars backend symbol " *
    "such as :state with Treebars loaded; got $(repr(progress))"))

# `f(phases)` with one pending phase per label of `labels`, a NamedTuple of
# descriptions; `phases` has the same keys.
_brm_with_phases(f, ::Nothing, labels::NamedTuple) = f(map(_ -> nothing, labels))

# `f(phase)` while `phase` runs.
_brm_enter_phase(f, ::Nothing) = f(nothing)

_brm_throw_if_interrupted(::Nothing) = nothing
_brm_interrupt_requested(::Nothing) = false

# A phase whose result already exists: it did no work and was not bypassed.
_brm_reuse_phase!(::Nothing) = nothing

# One of BRM's own preparation steps. Nothing partial exists before it ends, so
# an interrupt requested on the phase or an ancestor stops the call here with
# Treebars' `ProgressInterrupted`.
_brm_run_phase(f, phase) = _brm_enter_phase(phase) do node
    _brm_throw_if_interrupted(node)
    f(node)
end
