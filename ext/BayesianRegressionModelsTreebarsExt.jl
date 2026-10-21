module BayesianRegressionModelsTreebarsExt

# BRM's progress hooks (src/progress.jl) for Treebars nodes and backend symbols.
# Treebars is read through its module, so the extension loads on every Treebars
# 0.1 release; reporting needs `reuse_progress!` and the interrupt flag
# (Treebars a2f9e41 or later), and an older Treebars fails only where they run.

using BayesianRegressionModels
import Treebars
using Treebars: ProgressNode
const BRM = BayesianRegressionModels

BRM._brm_with_progress(f, progress::Union{ProgressNode,Symbol}; description) =
    Treebars.with_progress(f, progress; description)
BRM._brm_with_phases(f, node::ProgressNode, labels::NamedTuple) =
    Treebars.with_prepared_phases(f, node, labels)
BRM._brm_enter_phase(f, phase::ProgressNode) = Treebars.with_prepared_progress(f, phase)
BRM._brm_throw_if_interrupted(node::ProgressNode) = Treebars.throw_if_interrupted(node)
BRM._brm_interrupt_requested(node::ProgressNode) = Treebars.interrupt_requested(node)
BRM._brm_reuse_phase!(phase::ProgressNode) = (Treebars.reuse_progress!(phase); nothing)

end
