module BayesianRegressionModelsNLMEEstimationExt

# The NLMEEstimation.jl model protocol for an RK-lowered BRM population model
# (`BRMNLMEModel`, from `brm_nlme_model`). RK evaluates the whole program at once,
# so the model declares `BatchedEvaluation`: one batched call is one lockstep
# evaluation (`brm_nlme_loglikelihoods[_and_gradients]`) whatever the batch size,
# and the estimators make one such call per iteration for all subjects still
# iterating. No per-subject method is implemented here, because each would cost a
# full pass. No Hessian method and no AD backend are declared either: RK exposes
# first-order AD only (ReactiveKernels snag `rkppl-second-ord-ba21b602`), so
# `empirical_bayes` runs its gradient-only `QuasiNewtonEBE`.

using BayesianRegressionModels
import NLMEEstimation as NE
const BRM = BayesianRegressionModels

NE.nsubjects(m::BRM.BRMNLMEModel) = length(m.view.levels)

NE.nlme_layout(m::BRM.BRMNLMEModel) = NE.NLMELayout(;
    ntheta=length(m.theta), nsigma=length(m.sigma), eta_blocks=m.eta_blocks)

NE.evaluation_style(::BRM.BRMNLMEModel) = NE.BatchedEvaluation()

# All subjects' random effects with column `k` of `H` placed at subject
# `subjects[k]` and the other subjects at zero; by conditional independence the
# other subjects' values do not affect the batch's.
function _batch_eta(m::BRM.BRMNLMEModel, subjects, H::AbstractMatrix)
    n = length(m.view.levels)
    size(H, 2) == length(subjects) || throw(DimensionMismatch(
        "η matrix has $(size(H, 2)) columns for $(length(subjects)) subjects"))
    all(i -> 1 <= i <= n, subjects) || throw(ArgumentError("subjects $subjects are outside 1:$n"))
    allunique(subjects) || throw(ArgumentError(
        "subjects $subjects are not unique; one lockstep evaluation holds one η per subject"))
    full = zeros(promote_type(Float64, eltype(H)), size(H, 1), n)
    full[:, subjects] .= H
    full
end

function NE.conditional_loglikelihoods(m::BRM.BRMNLMEModel, subjects, θ, σ, H)
    values = BRM.brm_nlme_loglikelihoods(m, θ, σ, _batch_eta(m, subjects, H))
    values[subjects]
end

function NE.conditional_loglikelihoods_and_gradients(m::BRM.BRMNLMEModel, subjects, θ, σ, H)
    values, G = BRM.brm_nlme_loglikelihoods_and_gradients(m, θ, σ, _batch_eta(m, subjects, H))
    values[subjects], G[:, subjects]
end

# Intercept margins only: their population coefficient and random effect both
# add one unit to the predictor, so the density depends on them only through
# their sum. Slope margins are left undeclared until BRM proves their
# population and random-effect columns are the same transformed column.
function NE.mu_referencing(m::BRM.BRMNLMEModel)
    refs = [r for r in m.view.mu_references if r.coefficient === :Intercept]
    isempty(refs) && return nothing
    X = zeros(Float64, sum(m.eta_blocks), length(refs))
    for (j, r) in enumerate(refs)
        X[r.subject, j] = 1.0
    end
    NE.MuReferencing([findfirst(==(r.population), m.theta) for r in refs],
        fill(X, length(m.view.levels)))
end

NE.parameter_names(m::BRM.BRMNLMEModel) = (
    theta=string.(m.view.coordinates[m.theta]),
    sigma=string.(m.view.coordinates[m.sigma]),
    eta=[string(b.predictor, ":", b.coefficient) for b in m.view.block_margins])

end
