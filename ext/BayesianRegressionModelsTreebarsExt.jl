module BayesianRegressionModelsTreebarsExt

# BRM's progress hooks (src/progress.jl) for Treebars nodes and backend symbols.

using BayesianRegressionModels
using Treebars: ProgressNode, with_progress, with_prepared_phases,
    with_prepared_progress, reuse_progress!, throw_if_interrupted,
    interrupt_requested
const BRM = BayesianRegressionModels

BRM._brm_with_progress(f, progress::Union{ProgressNode,Symbol}; description) =
    with_progress(f, progress; description)
BRM._brm_with_phases(f, node::ProgressNode, labels::NamedTuple) =
    with_prepared_phases(f, node, labels)
BRM._brm_enter_phase(f, phase::ProgressNode) = with_prepared_progress(f, phase)
BRM._brm_throw_if_interrupted(node::ProgressNode) = throw_if_interrupted(node)
BRM._brm_interrupt_requested(node::ProgressNode) = interrupt_requested(node)
BRM._brm_reuse_phase!(phase::ProgressNode) = (reuse_progress!(phase); nothing)

end
