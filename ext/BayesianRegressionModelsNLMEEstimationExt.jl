module BayesianRegressionModelsNLMEEstimationExt

# The NLMEEstimation.jl model protocol for an RK-lowered BRM population model
# (`BRMNLMEModel`, from `brm_nlme_model`). The model declares
# `BatchedEvaluation`: larger batches use one lockstep RK evaluation, while
# singleton batches and explicit per-subject calls use one observation cell.
# Colored RK Hessian-vector products give all subjects' η Hessian
# blocks in one pass per η coordinate, so `empirical_bayes` uses `NewtonEBE`.
# Gaussian observations use one :observations query and colored forward
# pushforwards for all η Jacobians.

using BayesianRegressionModels
using ADTypes: AutoEnzyme
import Enzyme
import NLMEEstimation as NE
const BRM = BayesianRegressionModels

NE.nsubjects(m::BRM.BRMNLMEModel) = length(m.view.levels)

NE.nlme_layout(m::BRM.BRMNLMEModel) = NE.NLMELayout(;
    ntheta=length(m.theta), nsigma=length(m.sigma), eta_blocks=m.eta_blocks)

NE.evaluation_style(::BRM.BRMNLMEModel) = NE.BatchedEvaluation()
NE.provides_eta_hessian(::BRM.BRMNLMEModel) = true
# SAEM's numerical M-step differentiates the batched conditional density in
# θ and σ. The RK pointwise query mixes active and constant overlay reads.
NE.ad_backend(::BRM.BRMNLMEModel) =
    AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
BRM._brm_nlme_forward_backend(::AutoEnzyme) = AutoEnzyme(; mode=Enzyme.Forward)

function _check_subjects(m::BRM.BRMNLMEModel, subjects)
    n = length(m.view.levels)
    all(i -> 1 <= i <= n, subjects) || throw(ArgumentError(
        "subjects $subjects are outside 1:$n"))
    allunique(subjects) || throw(ArgumentError(
        "subjects $subjects are not unique; one lockstep evaluation holds one η per subject"))
    nothing
end

# All subjects' random effects with column `k` of `H` placed at subject
# `subjects[k]` and the other subjects at zero; by conditional independence the
# other subjects' values do not affect the batch's.
function _batch_eta(m::BRM.BRMNLMEModel, subjects, H::AbstractMatrix)
    size(H, 2) == length(subjects) || throw(DimensionMismatch(
        "η matrix has $(size(H, 2)) columns for $(length(subjects)) subjects"))
    _check_subjects(m, subjects)
    full = zeros(promote_type(Float64, eltype(H)), size(H, 1), length(m.view.levels))
    full[:, subjects] .= H
    full
end

# Gaussian moments and Jacobians still evaluate all requested subjects in one
# joint RK pass.
NE.observation_model(m::BRM.BRMNLMEModel) =
    m.observations_query === nothing || m.observations_pushforward === nothing ?
        nothing : NE.GaussianObservations()

function NE.observations(m::BRM.BRMNLMEModel, subjects::AbstractVector{<:Integer})
    NE.observation_model(m) isa NE.GaussianObservations || throw(ArgumentError(
        "BRMNLMEModel has no prepared Gaussian observation model"))
    _check_subjects(m, subjects)
    [copy(m.observed[i]) for i in subjects]
end

function NE.observation_moments(m::BRM.BRMNLMEModel,
        subjects::AbstractVector{<:Integer}, θ, σ, H::AbstractMatrix)
    NE.observation_model(m) isa NE.GaussianObservations || throw(ArgumentError(
        "BRMNLMEModel has no prepared Gaussian observation model"))
    u = BRM._brm_nlme_point(m, θ, σ, _batch_eta(m, subjects, H))
    moments = BRM._brm_nlme_moments(m, BRM._rk_observation_values(m.observations_query, u))
    moments[subjects]
end

function NE.observation_moments_and_jacobians(m::BRM.BRMNLMEModel,
        subjects::AbstractVector{<:Integer}, θ, σ, H::AbstractMatrix)
    NE.observation_model(m) isa NE.GaussianObservations || throw(ArgumentError(
        "BRMNLMEModel has no prepared Gaussian observation model"))
    u = BRM._brm_nlme_point(m, θ, σ, _batch_eta(m, subjects, H))
    values, tangents = BRM._rk_observation_value_and_pushforward(
        m.observations_pushforward, u)
    moments = BRM._brm_nlme_moments(m, values, tangents)
    moments[subjects]
end

function _singleton_cell(m::BRM.BRMNLMEModel, subjects, H::AbstractMatrix)
    length(subjects) == 1 && m.cell_query !== nothing || return nothing
    size(H, 2) == 1 || throw(DimensionMismatch(
        "η matrix has $(size(H, 2)) columns for 1 subject"))
    _check_subjects(m, subjects)
    only(subjects), view(H, :, 1)
end

function NE.conditional_loglikelihoods(m::BRM.BRMNLMEModel, subjects, θ, σ, H)
    cell = _singleton_cell(m, subjects, H)
    cell === nothing || return [BRM.brm_nlme_loglikelihood(m, cell[1], θ, σ, cell[2])]
    values = BRM.brm_nlme_loglikelihoods(m, θ, σ, _batch_eta(m, subjects, H))
    values[subjects]
end

NE.conditional_loglikelihood(m::BRM.BRMNLMEModel, i::Integer, θ, σ, η) =
    BRM.brm_nlme_loglikelihood(m, i, θ, σ, η)

NE.conditional_loglikelihood_and_gradient(m::BRM.BRMNLMEModel,
        i::Integer, θ, σ, η) =
    BRM.brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, η)

function NE.conditional_loglikelihoods_and_gradients(m::BRM.BRMNLMEModel, subjects, θ, σ, H)
    cell = _singleton_cell(m, subjects, H)
    if cell !== nothing
        value, gradient = BRM.brm_nlme_loglikelihood_and_gradient(
            m, cell[1], θ, σ, cell[2])
        return [value], reshape(gradient, :, 1)
    end
    values, G = BRM.brm_nlme_loglikelihoods_and_gradients(m, θ, σ, _batch_eta(m, subjects, H))
    values[subjects], G[:, subjects]
end

function NE.conditional_loglikelihood_hessian_blocks(m::BRM.BRMNLMEModel, subjects, θ, σ, H)
    blocks = BRM._brm_nlme_hessian_blocks(m, θ, σ, _batch_eta(m, subjects, H))
    blocks[subjects]
end

# The view records only population columns whose shift can be absorbed into a
# random-effect margin on every row of each subject.
function NE.mu_referencing(m::BRM.BRMNLMEModel)
    refs = m.view.mu_references
    isempty(refs) && return nothing
    design = [zeros(Float64, sum(m.eta_blocks), length(refs))
        for _ in m.view.levels]
    for (j, r) in enumerate(refs), i in eachindex(design)
        design[i][r.subject, j] = r.design[i]
    end
    NE.MuReferencing([findfirst(==(r.population), m.theta) for r in refs],
        design)
end

NE.parameter_names(m::BRM.BRMNLMEModel) = (
    theta=string.(m.view.coordinates[m.theta]),
    sigma=string.(m.view.coordinates[m.sigma]),
    eta=[string(b.predictor, ":", b.coefficient) for b in m.view.block_margins])

end
