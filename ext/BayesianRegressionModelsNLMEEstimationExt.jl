module BayesianRegressionModelsNLMEEstimationExt

# The NLMEEstimation.jl model protocol for an RK-lowered BRM population model
# (`BRMNLMEModel`, from `brm_nlme_model`). The protocol is per subject, while
# RK evaluates the whole program at once: every per-subject call here costs one
# full pass, and BRM's lockstep `brm_nlme_loglikelihoods[_and_gradients]`
# evaluate all subjects together. No Hessian method and no AD backend are
# declared: RK exposes first-order AD only (ReactiveKernels snag
# `rkppl-second-ord-ba21b602`).

using BayesianRegressionModels
import NLMEEstimation as NE
const BRM = BayesianRegressionModels

NE.nsubjects(m::BRM.BRMNLMEModel) = length(m.view.levels)

NE.nlme_layout(m::BRM.BRMNLMEModel) = NE.NLMELayout(;
    ntheta=length(m.theta), nsigma=length(m.sigma), eta_blocks=m.eta_blocks)

# All subjects' random effects with subject `i`'s set to `η` and every other
# subject's at zero; only column `i` of the result is read back.
function _subject_eta(m::BRM.BRMNLMEModel, i::Integer, η::AbstractVector)
    n = length(m.view.levels)
    1 <= i <= n || throw(ArgumentError("subject $i is outside 1:$n"))
    H = zeros(promote_type(Float64, eltype(η)), sum(m.eta_blocks), n)
    H[:, i] .= η
    H
end

NE.conditional_loglikelihood(m::BRM.BRMNLMEModel, i::Integer, θ, σ, η) =
    BRM.brm_nlme_loglikelihoods(m, θ, σ, _subject_eta(m, i, η))[i]

function NE.conditional_loglikelihood_and_gradient(m::BRM.BRMNLMEModel, i::Integer,
        θ, σ, η)
    values, G = BRM.brm_nlme_loglikelihoods_and_gradients(m, θ, σ, _subject_eta(m, i, η))
    values[i], G[:, i]
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
