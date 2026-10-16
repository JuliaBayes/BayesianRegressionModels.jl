module BayesianRegressionModelsNLMEEstimationExt

# The NLMEEstimation.jl model protocol for an RK-lowered BRM population model
# (`BRMNLMEModel`, from `brm_nlme_model`). The model declares
# `BatchedEvaluation`: larger batches use one lockstep RK evaluation, while
# singleton batches and explicit per-subject calls use one observation cell.
# Online SAEM uses those cells for its population gradient, avoiding a reverse
# derivative through the entire batched RK evaluator. The retained-draw
# mixture keeps batched gradients to avoid repeating cell work for each draw.
# Colored RK Hessian-vector products give all subjects' η Hessian
# blocks in one pass per η coordinate, so `empirical_bayes` uses `NewtonEBE`.
# Gaussian observations use one :observations query and colored forward
# pushforwards for all η Jacobians.

using BayesianRegressionModels
using ADTypes: AutoEnzyme
import DifferentiationInterface as DI
using LinearAlgebra
import Enzyme
import NLMEEstimation as NE
import ReactiveKernelsPPL as RKPPL
const BRM = BayesianRegressionModels

NE.nsubjects(m::BRM.BRMNLMEModel) = length(m.view.levels)

NE.nlme_layout(m::BRM.BRMNLMEModel) = BRM._brm_nlme_ragged(m.view) ?
    NE.RaggedNLMELayout(; ntheta=length(m.theta), nsigma=length(m.sigma),
        family_sizes=m.view.family_sizes, subject_families=m.view.subject_families) :
    NE.NLMELayout(; ntheta=length(m.theta), nsigma=length(m.sigma),
        eta_blocks=m.eta_blocks)

NE.evaluation_style(::BRM.BRMNLMEModel) = NE.BatchedEvaluation()
# Only a prepared compact subject-cell sampler differentiates one subject's
# population coordinates; every other model uses the lockstep batch.
NE.saem_population_gradient_style(m::BRM.BRMNLMEModel) =
    m.cell_gradient === nothing ? NE.BatchedEvaluation() : NE.PerSubjectEvaluation()
NE.saem_population_gradient_style(m::BRM.BRMNLMEModel, ::Val{:online}) =
    NE.saem_population_gradient_style(m)
NE.saem_population_gradient_style(::BRM.BRMNLMEModel, ::Val{:mixture}) =
    NE.BatchedEvaluation()
NE.provides_eta_hessian(::BRM.BRMNLMEModel) = true
# Other numerical derivatives still use Enzyme. The RK pointwise query mixes
# active and constant overlay reads.
NE.ad_backend(::BRM.BRMNLMEModel) =
    AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
BRM._brm_nlme_forward_backend(::AutoEnzyme) = AutoEnzyme(; mode=Enzyme.Forward)

# An NLME covariance block is packed as the log diagonal and raw lower
# triangle of its Cholesky factor C. BRM instead samples marginal scales τ and
# an LKJ correlation Cholesky factor R, with C = diag(τ) R. These helpers move
# the *authored* RK prior to NLME's ω measure; the standardized subject draws
# are omitted because NLMEEstimation supplies their Gaussian prior itself.
function _replace_prior_declaration(values::NamedTuple, declaration::Symbol, value)
    hasproperty(values, declaration) && return merge(values,
        NamedTuple{(declaration,)}((value,)))
    parts = Symbol.(split(String(declaration), '.'))
    replace_path(nt, path) = length(path) == 1 ?
        merge(nt, NamedTuple{(only(path),)}((value,))) :
        merge(nt, NamedTuple{(first(path),)}((replace_path(nt[first(path)], path[2:end]),)))
    replace_path(values, parts)
end

_prior_positions(m, declaration) = findall(name ->
    BRM._brm_nlme_owned(name, declaration), m.view.coordinates)

function _prior_entry(layout, position)
    only(e for e in layout.entries if e.offset <= position < e.offset + e.size)
end

function _scale_jacobian_derivative(entry, u)
    transform = entry.transform
    transform === :identity && return 0.0
    transform in (:exp, :floored, :upper) && return 1.0
    transform in (:interval, :logistic) && return 1 - 2 / (1 + exp(-u))
    BRM._brm_nlme_error("population prior: unsupported random-effect scale " *
        "transform `$transform` for `$(entry.name)`")
end

function _prior_omega_map(m::BRM.BRMNLMEModel, θ, σ, ω)
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "θ has length $(length(θ)); expected $(length(m.theta))"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "σ has length $(length(σ)); expected $(length(m.sigma))"))
    layout = m.backend.model.layout
    length(ω) == NE.nomega(NE.nlme_layout(m)) || throw(DimensionMismatch(
        "ω has length $(length(ω)); expected $(NE.nomega(NE.nlme_layout(m)))"))
    u = copy(m.unit)
    u[m.theta] = θ
    u[m.sigma] = σ
    values = RKPPL.constrain(layout, u)
    chol = NE.omega_cholesky(NE.nlme_layout(m), ω)
    C = BRM._brm_nlme_ragged(m.view) ? chol : Matrix(chol)
    records = [r for r in BRM._brm_nlme_records(m.backend.plan) if r.kind === :ranef]
    J = zeros(Float64, length(ω), length(ω))
    omega_positions = Int[]
    omega_logjac_gradient = Float64[]
    log_omega_jacobian = 0.0
    log_coordinate_jacobian = 0.0
    log_omega_jacobian_gradient = zeros(Float64, length(ω))
    saturated = false
    margin_start = 1
    omega_start = 1
    for (family, record) in enumerate(records)
        K = length(record.margins)
        block = BRM._brm_nlme_ragged(m.view) ? C[family] :
            C[margin_start:(margin_start + K - 1), margin_start:(margin_start + K - 1)]
        τ = [norm(view(block, i, 1:i)) for i in 1:K]
        scales = isnothing(record.scales) ? (record.scale,) : record.scales
        scale_positions = Int[]
        if isnothing(record.scales)
            values = _replace_prior_declaration(values, record.scale, τ)
            append!(scale_positions, _prior_positions(m, record.scale))
        else
            length(scales) == K || BRM._brm_nlme_error(
                "population prior: scale declarations do not match the random-effect block")
            for (declaration, scale) in zip(scales, τ)
                values = _replace_prior_declaration(values, declaration, scale)
                append!(scale_positions, _prior_positions(m, declaration))
            end
        end
        length(scale_positions) == K || BRM._brm_nlme_error(
            "population prior: expected $K scale coordinates for `$(record.group)`")
        factor_positions = isnothing(record.L) ? Int[] : _prior_positions(m, record.L)
        length(factor_positions) == K * (K - 1) ÷ 2 || BRM._brm_nlme_error(
            "population prior: correlation coordinates do not match the random-effect block")
        append!(omega_positions, scale_positions, factor_positions)
        margin_start += K
        omega_start += K * (K + 1) ÷ 2
    end
    length(omega_positions) == length(ω) || BRM._brm_nlme_error(
        "population prior: BRM covariance coordinates do not match NLME ω")
    # Preserve the supplied unconstrained population point exactly after the
    # constrained round trip. Only BRM's scale coordinates are replaced here;
    # the correlation coordinates are written from C below.
    mapped = RKPPL.unconstrain(layout, values)
    mapped[m.theta] = θ
    mapped[m.sigma] = σ
    mapped[findall(==(:subject_effect), m.view.role)] .= 0.0
    # Build the small covariance-coordinate Jacobian in the same block order
    # as NLMELayout. Row i has i raw Cholesky entries, its last stored as log.
    margin_start = 1
    omega_start = 1
    urow = 1
    for (family, record) in enumerate(records)
        K = length(record.margins)
        scale_positions = omega_positions[urow:(urow + K - 1)]
        factor_positions = omega_positions[(urow + K):(urow + K * (K + 1) ÷ 2 - 1)]
        block = BRM._brm_nlme_ragged(m.view) ? C[family] :
            C[margin_start:(margin_start + K - 1), margin_start:(margin_start + K - 1)]
        τ = [norm(view(block, i, 1:i)) for i in 1:K]
        local_omega = [omega_start + i * (i - 1) ÷ 2 + j - 1 for i in 1:K, j in 1:K]
        for i in 1:K
            scale_position = scale_positions[i]
            entry = _prior_entry(layout, scale_position)
            entry.transform in (:exp, :floored, :upper, :interval, :logistic) ||
                BRM._brm_nlme_error("population prior: unsupported scale transform " *
                    "`$(entry.transform)` for `$(entry.name)`")
            entry.sampling === nothing || BRM._brm_nlme_error(
                "population prior: external random-effect scale transforms are not covered")
            entry.lo isa Real && entry.hi isa Real || BRM._brm_nlme_error(
                "population prior: random-effect scale bounds must be fixed")
            scale_logjac = RKPPL._logjac_elt(entry, mapped[scale_position])
            log_coordinate_jacobian += scale_logjac
            push!(omega_logjac_gradient,
                _scale_jacobian_derivative(entry, mapped[scale_position]))
            inverse_derivative = (entry.transform === :upper ? -1.0 : 1.0) *
                exp(-scale_logjac)
            diagonal = local_omega[i, i]
            log_omega_jacobian += 2ω[diagonal] - i * log(τ[i])
            for j in 1:i
                col = local_omega[i, j]
                dC = j == i ? block[i, i] : 1.0
                J[urow + i - 1, col] = inverse_derivative * block[i, j] * dC / τ[i]
                log_omega_jacobian_gradient[col] +=
                    (j == i ? 2.0 : 0.0) - i * block[i, j] * dC / τ[i]^2
            end
        end
        factor_index = 1
        for i in 2:K, j in 1:(i - 1)
            position = factor_positions[factor_index]
            entry = _prior_entry(layout, position)
            entry.kind === :cholesky_corr && entry.transform === :lkj ||
                BRM._brm_nlme_error("population prior: expected an LKJ correlation factor")
            # The LKJ coordinate is atanh of the partial correlation
            # C[i,j] / ‖C[i,j:i]‖, i.e. asinh(C[i,j] / ‖C[i,j+1:i]‖). That form
            # is finite for every finite ω, while the partial correlation
            # itself rounds to ±1 once ‖C[i,j+1:i]‖ ≪ |C[i,j]|.
            mapped[position] = asinh(block[i, j] / norm(view(block, i, (j + 1):i)))
            z = tanh(mapped[position])
            # Past |u| ≈ 19, tanh(u) rounds to ±1: RK's LKJ log-Jacobian
            # log(1 - tanh(u)²) is then -Inf and the ω density underflows.
            abs(z) < 1 || (saturated = true)
            push!(omega_logjac_gradient, -(i - j + 1) * z)
            S = sum(abs2, view(block, i, j:i))
            T = sum(abs2, view(block, i, (j + 1):i))
            for k in j:i
                col = local_omega[i, k]
                du_dC = k == j ? inv(sqrt(S)) :
                    -block[i, j] * block[i, k] / (sqrt(S) * T)
                J[urow + K + factor_index - 1, col] =
                    du_dC * (k == i ? block[i, i] : 1.0)
            end
            factor_index += 1
        end
        if K > 1
            log_coordinate_jacobian += RKPPL.lkj_chol_logjac(mapped[factor_positions], K)
        end
        margin_start += K
        omega_start += K * (K + 1) ÷ 2
        urow += K * (K + 1) ÷ 2
    end
    correction = log_omega_jacobian - log_coordinate_jacobian
    (; mapped, omega_positions, J, omega_logjac_gradient,
        log_omega_jacobian_gradient, correction, saturated)
end

# A finite ω whose correlation coordinate saturates has an underflowed prior
# density in BRM's coordinates: its log is -Inf, and its gradient is undefined.
# NLMEEstimation's M-step line search and FOCE(I) objective reject such a trial.
function NE.population_logprior(m::BRM.BRMNLMEModel, θ, σ, ω)
    mapping = _prior_omega_map(m, θ, σ, ω)
    mapping.saturated && return -Inf
    # BRM's standardized z prior is one N(0,1) per subject and margin at z=0.
    # NLMEEstimation owns the η prior, so remove precisely that constant term.
    z_constant = -sum(length, m.view.subject_coordinates) * log(2π) / 2
    BRM._rk_nlme_population_prior_value(m.population_prior, mapping.mapped) +
        mapping.correction - z_constant
end

function NE.population_logprior_and_gradient(m::BRM.BRMNLMEModel, θ, σ, ω)
    mapping = _prior_omega_map(m, θ, σ, ω)
    mapping.saturated && return -Inf, fill(NaN, length(θ)), fill(NaN, length(σ)),
        fill(NaN, length(ω))
    value, gradient = BRM._rk_nlme_population_prior_value_gradient(
        m.population_prior, mapping.mapped)
    z_constant = -sum(length, m.view.subject_coordinates) * log(2π) / 2
    gω = mapping.J' *
        (gradient[mapping.omega_positions] - mapping.omega_logjac_gradient) +
        mapping.log_omega_jacobian_gradient
    value + mapping.correction - z_constant,
        gradient[m.theta], gradient[m.sigma], gω
end

# brm_nlme_start: pack the natural covariance as ω, then map ω back onto BRM's
# authored scale and correlation coordinates. Both must reproduce the point.
function BRM._brm_nlme_start_coordinates(m::BRM.BRMNLMEModel, θ, σ, Ω, u)
    layout = NE.nlme_layout(m)
    ω = NE.omega_parameters(layout, Ω)
    unpacked = NE.omega_matrix(layout, ω)
    same = BRM._brm_nlme_ragged(m.view) ?
        all(isapprox(a, b; rtol=1e-10, atol=1e-12) for (a, b) in zip(unpacked, Ω)) :
        isapprox(unpacked, Ω; rtol=1e-10, atol=1e-12)
    same || BRM._brm_nlme_error(
        "brm_nlme_start: NLMEEstimation's ω does not reproduce the covariance")
    mapping = _prior_omega_map(m, θ, σ, ω)
    positions = mapping.omega_positions
    max_error = maximum(abs.(mapping.mapped[positions] .- u[positions]); init=0.0)
    max_error <= 1e-8 * (1 + maximum(abs, u[positions]; init=0.0)) ||
        BRM._brm_nlme_error("brm_nlme_start: ω maps back to BRM covariance " *
            "coordinates $max_error away from the point")
    population_logprior = NE.population_logprior(m, θ, σ, ω)
    isfinite(population_logprior) || BRM._brm_nlme_error(
        "brm_nlme_start: the population log prior is $population_logprior at the point")
    (; omega=ω, population_logprior, max_error)
end

NE.population_prior_provenance(m::BRM.BRMNLMEModel) = (
    source=:BRM_RK_authored_priors,
    theta=copy(m.view.coordinates[m.theta]),
    sigma=copy(m.view.coordinates[m.sigma]),
    omega=:NLME_log_Cholesky,
    subject_prior=:NLMEEstimation,
)

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
    BRM._brm_nlme_ragged(m.view) && throw(DimensionMismatch(
        "ragged subjects need one compact η vector per subject, not a matrix"))
    size(H, 2) == length(subjects) || throw(DimensionMismatch(
        "η matrix has $(size(H, 2)) columns for $(length(subjects)) subjects"))
    _check_subjects(m, subjects)
    full = zeros(promote_type(Float64, eltype(H)), size(H, 1), length(m.view.levels))
    full[:, subjects] .= H
    full
end

function _batch_eta(m::BRM.BRMNLMEModel, subjects,
        H::AbstractVector{<:AbstractVector})
    length(H) == length(subjects) || throw(DimensionMismatch(
        "η has $(length(H)) subject vectors for $(length(subjects)) subjects"))
    _check_subjects(m, subjects)
    full = [zeros(Float64, length(block)) for block in m.view.subject_coordinates]
    for (k, i) in enumerate(subjects)
        length(H[k]) == length(full[i]) || throw(DimensionMismatch(
            "subject $i η has length $(length(H[k])), expected $(length(full[i]))"))
        full[i] = Vector{Float64}(H[k])
    end
    full
end

# NLMEEstimation's generic batched population derivative expands a selected
# η batch inside the differentiated function. For ragged subjects, that
# allocation has no θ/σ derivative and Enzyme cannot compile its nested-vector
# mutation. Expand the constant η batch first, then differentiate only the
# population coordinates through the one lockstep RK likelihood query.
function _ragged_population_objective(x, m, subjects, full, ntheta)
    values = BRM.brm_nlme_loglikelihoods(m, view(x, 1:ntheta),
        view(x, ntheta+1:length(x)), full)
    sum(values[subjects])
end

function NE._saem_population_value_gradient(::NE.BatchedEvaluation,
        m::BRM.BRMNLMEModel, subjects, θ, σ,
        H::AbstractVector{<:AbstractVector})
    full = _batch_eta(m, subjects, H)
    DI.value_and_gradient(_ragged_population_objective, NE.ad_backend(m), vcat(θ, σ),
        DI.Constant(m), DI.Constant(subjects), DI.Constant(full),
        DI.Constant(length(θ)))
end

# Enzyme's argument lowering asserts on a large BRMNLMEModel passed as a
# constant, even when the differentiated function only sums its active input.
# Keep the differentiated batch context to the conditional-likelihood inputs.
_flat_population_context(m::BRM.BRMNLMEModel) =
    (; unit=m.unit, theta=m.theta, sigma=m.sigma,
       subject_coordinates=m.view.subject_coordinates,
       nsubjects=length(m.view.levels), pointwise=m.pointwise,
       attribution=m.attribution)

function _flat_population_objective(x, context, subjects, full, ntheta)
    u = BRM._brm_nlme_flat_point(context.unit, context.theta, context.sigma,
        context.subject_coordinates, view(x, 1:ntheta),
        view(x, ntheta+1:length(x)), full)
    fields = BRM._rk_pointwise_values(context.pointwise, u)
    values = zeros(Float64, context.nsubjects)
    BRM._brm_nlme_sum_fields!(values, context.attribution, fields, Val(1))
    sum(values[subjects])
end

function NE._saem_population_value_gradient(::NE.BatchedEvaluation,
        m::BRM.BRMNLMEModel, subjects, θ, σ, H::AbstractMatrix)
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "θ has length $(length(θ)), expected $(length(m.theta))"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "σ has length $(length(σ)), expected $(length(m.sigma))"))
    full = _batch_eta(m, subjects, H)
    DI.value_and_gradient(_flat_population_objective, NE.ad_backend(m), vcat(θ, σ),
        DI.Constant(_flat_population_context(m)), DI.Constant(subjects),
        DI.Constant(full), DI.Constant(length(θ)))
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
        subjects::AbstractVector{<:Integer}, θ, σ, H)
    NE.observation_model(m) isa NE.GaussianObservations || throw(ArgumentError(
        "BRMNLMEModel has no prepared Gaussian observation model"))
    u = BRM._brm_nlme_point(m, θ, σ, _batch_eta(m, subjects, H))
    moments = BRM._brm_nlme_moments(m, BRM._rk_observation_values(m.observations_query, u))
    moments[subjects]
end

function NE.observation_moments_and_jacobians(m::BRM.BRMNLMEModel,
        subjects::AbstractVector{<:Integer}, θ, σ, H)
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

function _singleton_cell(m::BRM.BRMNLMEModel, subjects,
        H::AbstractVector{<:AbstractVector})
    length(subjects) == 1 && m.cell_query !== nothing || return nothing
    length(H) == 1 || throw(DimensionMismatch(
        "η has $(length(H)) vectors for 1 subject"))
    _check_subjects(m, subjects)
    only(subjects), only(H)
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

# The same prepared subject-cell sampler also differentiates the population
# coordinates. SAEM can use this protocol method without compiling a reverse
# derivative of the entire batched RK evaluation.
function NE.conditional_loglikelihood_and_population_gradient(m::BRM.BRMNLMEModel,
        i::Integer, θ, σ, η)
    value, indices, selected_gradient, _ =
        BRM._brm_nlme_cell_value_and_selected_gradient(m, i, θ, σ, η)
    gθ = [selected_gradient[searchsortedfirst(indices, j)] for j in m.theta]
    gσ = [selected_gradient[searchsortedfirst(indices, j)] for j in m.sigma]
    value, gθ, gσ
end

function NE.conditional_loglikelihoods_and_gradients(m::BRM.BRMNLMEModel, subjects, θ, σ, H)
    cell = _singleton_cell(m, subjects, H)
    if cell !== nothing
        value, gradient = BRM.brm_nlme_loglikelihood_and_gradient(
            m, cell[1], θ, σ, cell[2])
        return [value], H isa AbstractMatrix ? reshape(gradient, :, 1) : [gradient]
    end
    values, G = BRM.brm_nlme_loglikelihoods_and_gradients(m, θ, σ, _batch_eta(m, subjects, H))
    values[subjects], G isa AbstractMatrix ? G[:, subjects] : G[subjects]
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

NE.parameter_names(m::BRM.BRMNLMEModel) = BRM._brm_nlme_parameter_names(m)

end
