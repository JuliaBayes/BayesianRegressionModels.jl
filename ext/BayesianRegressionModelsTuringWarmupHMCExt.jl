module BayesianRegressionModelsTuringWarmupHMCExt

using BayesianRegressionModels
using Distributions: LogNormal, Normal
using LogDensityProblems
using Turing
using WarmupHMC

const BRM = BayesianRegressionModels
const DynamicPPL = Turing.DynamicPPL

const _SIMPLE_CONTRACT =
    "native online adaptive centering currently supports exactly one " *
    "single-response identity-link Gaussian predictor with fixed observation " *
    "scale, default population priors, no term models or R2D2 geometry, and " *
    "one ordinary default-prior noncentered scalar `(1 | group)` block"

const _HSGP_CONTRACT =
    "native HSGP online adaptive centering currently supports exactly one " *
    "single-response Gaussian model with identity-linked `mu`, log-linked " *
    "`sigma`, default population priors, zero-location LogNormal HSGP priors, " *
    "no random effects, and one " *
    "ungrouped isotropic squared-exponential HSGP over the same observed axis " *
    "on each predictor"

_unsupported(reason) = error("Turing backend: $_SIMPLE_CONTRACT; $reason.")
_unsupported_hsgp(reason) = error("Turing backend: $_HSGP_CONTRACT; $reason.")

function _has_hsgp_term(backend::BRM.TuringBRMI)
    plan = backend.plan
    plan isa BRM._TuringGenericPlan || return false
    any(component -> any(term -> term.callable === BRM.hsgp, component.terms),
        plan.predictors)
end

function _hsgp_prior_scale(prior, logical, role)
    prior isa BRM.ExprColumn || _unsupported_hsgp(
        "predictor `$logical` uses a non-expression $role prior")
    BRM.getf(prior) === LogNormal || _unsupported_hsgp(
        "predictor `$logical` uses a non-LogNormal $role prior")
    isempty(BRM.getkwargs(prior)) || _unsupported_hsgp(
        "predictor `$logical` uses keyword arguments in its $role prior")
    args = BRM.getargs(prior)
    length(args) == 2 && first(args) isa Real && iszero(first(args)) ||
        _unsupported_hsgp(
            "predictor `$logical` requires a zero-location LogNormal $role prior")
    scale = last(args)
    scale isa Real && isfinite(scale) && scale > 0 || _unsupported_hsgp(
        "predictor `$logical` has a non-positive LogNormal $role prior scale")
    Float64(scale)
end

function _hsgp_component_contract(component, logical)
    component.predictor.name === logical || _unsupported_hsgp(
        "the required `$logical` predictor is absent")
    isempty(component.random_effects) || _unsupported_hsgp(
        "predictor `$logical` also contains random-effect geometry")
    all(isnothing, component.priors) || _unsupported_hsgp(
        "predictor `$logical` overrides a population prior")
    isnothing(component.r2d2) || _unsupported_hsgp(
        "predictor `$logical` uses R2D2 geometry")
    length(component.terms) == 1 || _unsupported_hsgp(
        "predictor `$logical` has $(length(component.terms)) prepared terms")
    term = only(component.terms)
    term.callable === BRM.hsgp || _unsupported_hsgp(
        "predictor `$logical` does not contain an HSGP term")
    state = term.state
    !get(state, :latent, false) || _unsupported_hsgp(
        "predictor `$logical` uses a model-derived HSGP axis")
    isnothing(state.by) || _unsupported_hsgp(
        "predictor `$logical` uses a grouped HSGP")
    state.cov === :exp_quad || _unsupported_hsgp(
        "predictor `$logical` uses covariance `$(state.cov)`")
    state.iso || _unsupported_hsgp(
        "predictor `$logical` uses anisotropic length scales")
    length(term.source) == 1 || _unsupported_hsgp(
        "predictor `$logical` uses $(length(term.source)) HSGP axes")
    _hsgp_prior_scale(state.rho_prior, logical, "length-scale")
    _hsgp_prior_scale(state.sigma_prior, logical, "marginal-SD")
    state.rho_lower isa Real && isfinite(state.rho_lower) && state.rho_lower >= 0 ||
        _unsupported_hsgp(
            "predictor `$logical` has a non-finite HSGP length-scale lower bound")
    term
end

function _two_hsgp_contract(backend::BRM.TuringBRMI)
    plan = backend.plan
    plan isa BRM._TuringGenericPlan || _unsupported_hsgp(
        "multi-response models are outside this contract")
    length(plan.predictors) == 2 || _unsupported_hsgp(
        "found $(length(plan.predictors)) predictors rather than `mu` and `sigma`")
    isempty(plan.joint_r2d2) || _unsupported_hsgp(
        "joint R2D2 geometry changes the population frame")
    isempty(plan.parameters) || _unsupported_hsgp(
        "free distribution parameters introduce additional coordinates")
    isempty(plan.assignments) || _unsupported_hsgp(
        "prepared assignments introduce additional model geometry")
    isnothing(plan.missing_response) || _unsupported_hsgp(
        "missing responses change the observation target")
    isnothing(plan.response_modifier) || _unsupported_hsgp(
        "response modifiers change the Gaussian target")
    isnothing(plan.observation_weight) || _unsupported_hsgp(
        "observation weights change the Gaussian target")
    isnothing(plan.response_fit) || _unsupported_hsgp(
        "response fitting metadata changes the observation target")

    mu_index = findfirst(c -> c.predictor.name === :mu, plan.predictors)
    sigma_index = findfirst(c -> c.predictor.name === :sigma, plan.predictors)
    isnothing(mu_index) && _unsupported_hsgp("the `mu` predictor is absent")
    isnothing(sigma_index) && _unsupported_hsgp("the `sigma` predictor is absent")
    mu = plan.predictors[mu_index]
    sigma = plan.predictors[sigma_index]
    mu.predictor.link_lhs_fn === identity || _unsupported_hsgp(
        "the `mu` predictor is not identity-linked")
    sigma.predictor.link_lhs_fn === log || _unsupported_hsgp(
        "the `sigma` predictor is not log-linked")
    mu_term = _hsgp_component_contract(mu, :mu)
    sigma_term = _hsgp_component_contract(sigma, :sigma)
    mu_term.source == sigma_term.source || _unsupported_hsgp(
        "the two HSGPs do not share the same observed source axis")

    distribution = plan.distribution
    distribution isa BRM._BRMPreparedExpr || _unsupported_hsgp(
        "the observation distribution is not a prepared `Normal` call")
    distribution.callable === Normal || _unsupported_hsgp(
        "the observation distribution is not `Normal`")
    isempty(distribution.kwargs) || _unsupported_hsgp(
        "keyword arguments in `Normal` are unsupported")
    length(distribution.args) == 2 || _unsupported_hsgp(
        "`Normal` does not receive exactly `mu` and `sigma`")
    mean_argument, scale_argument = distribution.args
    mean_argument isa BRM._BRMPreparedRef && mean_argument.name === :mu &&
        mean_argument.axis === :observation || _unsupported_hsgp(
            "the `Normal` mean is not the fitted `mu` predictor")
    scale_argument isa BRM._BRMPreparedRef && scale_argument.name === :sigma &&
        scale_argument.axis === :observation || _unsupported_hsgp(
            "the `Normal` scale is not the fitted `sigma` predictor")
    (; plan, mu_index, sigma_index, mu, sigma, mu_term, sigma_term)
end

_root_varname(name::Symbol) = DynamicPPL.VarName{name}()
_field_varname(name::Symbol, field::Symbol) =
    DynamicPPL.VarName{name}(DynamicPPL.Property{field}())

function _coordinate_range(ranges, varname, role)
    haskey(ranges, varname) || _unsupported_hsgp(
        "the DynamicPPL problem has no `$varname` $role coordinates")
    collect(ranges[varname].range)
end

struct TuringHSGPGradientComponent
    block::BRM._HSGPAdaptiveCenteringBlock
    design::Matrix{Float64}
    fixed::Vector{Float64}
    PHI::Matrix{Float64}
    beta_indices::Vector{Int}
    rho_prior_scale::Float64
    sd_prior_scale::Float64
end

function _hsgp_gradient_component(
    component, component_index, n_components, term, ranges,
)
    logical = component.predictor.name
    beta_site = n_components == 1 ? :beta_pop :
        (component_index == 1 ? :beta_pop : Symbol(:beta_pop_, logical))
    term_site = Symbol(:term_, logical, :_1)
    n_beta = size(component.design.matrix, 2)
    beta_varname = _root_varname(beta_site)
    beta_indices = if iszero(n_beta)
        haskey(ranges, beta_varname) ? collect(ranges[beta_varname].range) : Int[]
    else
        _coordinate_range(ranges, beta_varname, "population-effect")
    end
    rho_indices = _coordinate_range(
        ranges, _field_varname(term_site, :rho), "length-scale")
    sd_indices = _coordinate_range(
        ranges, _field_varname(term_site, :sigma), "marginal-SD")
    effect_indices = _coordinate_range(
        ranges, _field_varname(term_site, :beta_raw), "basis-weight")
    length(beta_indices) == n_beta ||
        _unsupported_hsgp(
            "`$beta_site` occupies $(length(beta_indices)) coordinates for " *
            "$n_beta population-design columns")
    length(rho_indices) == 1 || _unsupported_hsgp(
        "`$term_site.rho` occupies $(length(rho_indices)) coordinates")
    length(sd_indices) == 1 || _unsupported_hsgp(
        "`$term_site.sigma` occupies $(length(sd_indices)) coordinates")

    state = term.state
    PHI = Matrix{Float64}(state.PHI)
    omega2 = Matrix{Float64}(state.omega2)
    size(PHI, 2) == length(effect_indices) || _unsupported_hsgp(
        "`$term_site.beta_raw` occupies $(length(effect_indices)) coordinates " *
        "for $(size(PHI, 2)) prepared basis columns")
    size(omega2) == (length(effect_indices), 1) || _unsupported_hsgp(
        "`$term_site` has spectral-frequency shape $(size(omega2)) for " *
        "$(length(effect_indices)) basis weights")
    size(PHI, 1) == length(component.design.fixed) || _unsupported_hsgp(
        "`$term_site` has $(size(PHI, 1)) rows for a predictor with " *
        "$(length(component.design.fixed)) rows")

    term_label = Symbol(:hsgp_, join(string.(term.source), "_"))
    block = BRM._HSGPAdaptiveCenteringBlock(
        logical,
        term_label,
        0.0,
        effect_indices,
        rho_indices,
        [Float64(state.rho_lower)],
        only(sd_indices),
        0.0,
        omega2,
    )
    TuringHSGPGradientComponent(
        block,
        Matrix{Float64}(component.design.matrix),
        collect(Float64, component.design.fixed),
        PHI,
        beta_indices,
        _hsgp_prior_scale(state.rho_prior, logical, "length-scale"),
        _hsgp_prior_scale(state.sigma_prior, logical, "marginal-SD"),
    )
end

function _two_hsgp_geometry(backend, problem, contract)
    problem.model === backend.model || _unsupported_hsgp(
        "the DynamicPPL problem was not built from this backend's exact model")
    problem.transform_strategy isa DynamicPPL.LinkAll || _unsupported_hsgp(
        "the DynamicPPL problem is not linked; construct it with `DynamicPPL.LinkAll()`")
    ranges = DynamicPPL.get_all_ranges_and_transforms(problem)
    n_components = length(contract.plan.predictors)
    mu = _hsgp_gradient_component(
        contract.mu, contract.mu_index, n_components, contract.mu_term, ranges)
    sigma = _hsgp_gradient_component(
        contract.sigma, contract.sigma_index, n_components,
        contract.sigma_term, ranges)
    blocks = [mu.block, sigma.block]

    claimed = Int[]
    for component in (mu, sigma)
        append!(claimed, component.beta_indices)
        append!(claimed, component.block.effects)
        append!(claimed, component.block.length_scales)
        push!(claimed, component.block.sd)
    end
    length(unique(claimed)) == length(claimed) || _unsupported_hsgp(
        "DynamicPPL metadata assigns overlapping coordinates to the two HSGPs")
    sort!(claimed) == collect(1:LogDensityProblems.dimension(problem)) ||
        _unsupported_hsgp(
            "the DynamicPPL problem contains coordinates outside the two-HSGP contract")
    blocks, mu, sigma
end

function _simple_random_intercept_block(backend::BRM.TuringBRMI)
    plan = backend.plan
    plan isa BRM._TuringGenericPlan ||
        _unsupported("multi-response models are outside this first contract")
    length(plan.predictors) == 1 ||
        _unsupported("found $(length(plan.predictors)) predictors")
    isempty(plan.joint_r2d2) ||
        _unsupported("joint R2D2 geometry changes the group scale frame")
    isempty(plan.parameters) || _unsupported(
        "free distribution parameters introduce constrained DynamicPPL coordinates",
    )
    isempty(plan.assignments) ||
        _unsupported("prepared assignments require a separate gradient kernel")
    isnothing(plan.missing_response) ||
        _unsupported("missing responses change the observation target")
    isnothing(plan.response_modifier) ||
        _unsupported("response modifiers change the Gaussian target")
    isnothing(plan.observation_weight) ||
        _unsupported("observation weights change the Gaussian target")
    isnothing(plan.response_fit) ||
        _unsupported("response fitting metadata changes the observation target")

    component = only(plan.predictors)
    component.predictor.link_lhs_fn === identity ||
        _unsupported("the predictor link is not the identity function")
    all(isnothing, component.priors) ||
        _unsupported("population-prior overrides change the analytic gradient")
    isempty(component.terms) ||
        _unsupported("generated term models, including HSGP, are not yet included")
    isnothing(component.r2d2) ||
        _unsupported("R2D2 geometry changes the group scale frame")
    length(component.random_effects) == 1 ||
        _unsupported("found $(length(component.random_effects)) random-effect blocks")

    block = only(component.random_effects)
    block isa BRM._BRMRandomEffectPlan ||
        _unsupported("multi-membership random effects need a separate coordinate map")
    block.intercept_only ||
        _unsupported("correlated and zero-correlation slope blocks are not yet included")
    !block.centered ||
        _unsupported("the generated model is centered rather than the required c=0 frame")
    isnothing(block.by) ||
        _unsupported("stratified `by` blocks have multiple scale frames")
    isnothing(block.id) ||
        _unsupported("shared-ID group geometry is not part of the simple case")
    all(isnothing, block.sd_prior) ||
        _unsupported("a group-scale prior override replaces the `log_scale` coordinate")
    block.lkj_eta == 1.0 ||
        _unsupported("a nondefault LKJ setting is not part of the simple case")
    size(block.matrix, 2) == 1 && all(isone, block.matrix) ||
        _unsupported("the random-effect design is not a literal intercept column")
    block
end

function _fixed_gaussian_scale(plan)
    distribution = plan.distribution
    distribution isa BRM._BRMPreparedExpr ||
        _unsupported("the observation distribution is not a prepared `Normal` call")
    distribution.callable === Normal ||
        _unsupported("the observation distribution is not `Normal`")
    isempty(distribution.kwargs) ||
        _unsupported("keyword arguments in `Normal` are outside the simple case")
    length(distribution.args) == 2 ||
        _unsupported("`Normal` must have a predictor mean and fixed scale")
    mean_argument, scale_argument = distribution.args
    mean_argument isa BRM._BRMPreparedRef &&
        mean_argument.name === only(plan.predictors).predictor.name &&
        mean_argument.axis === :observation ||
        _unsupported("the `Normal` mean is not the fitted predictor")
    scale_argument isa Real ||
        _unsupported("the `Normal` scale is not a fixed real number")
    scale = Float64(scale_argument)
    isfinite(scale) && scale > 0 ||
        _unsupported("the fixed `Normal` scale must be finite and positive")
    scale
end

mutable struct TuringAdaptiveCenteringState
    log_scale_index::Int
    effect_indices::Vector{Int}
    sources::Vector{Float64}
end

struct TuringAdaptiveCenteringArgument{KIND} <: Function
    state::TuringAdaptiveCenteringState
    pair_number::Int
end

function (::TuringAdaptiveCenteringArgument{:location})(x)
    zero(eltype(x))
end

function (arg::TuringAdaptiveCenteringArgument{:log_scale})(x)
    x[arg.state.log_scale_index]
end

function _sync_sources!(state, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "Turing adaptive-centering plan has $(length(state.sources)) cells but " *
        "the WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (pair_number, (idx, value)) in enumerate(ir.pairs)
        expected = state.effect_indices[pair_number]
        idx == expected || throw(ArgumentError(
            "Turing adaptive-centering pair $pair_number addresses raw coordinate " *
            "$idx, but DynamicPPL metadata requires $expected; pair ordering changed",
        ))
        state.sources[pair_number] = Float64(value.source.c)
    end
    ir
end

struct TuringAdaptiveCenteringFrame{T}
    source::Vector{T}
    log_scale::T
    scale::T
    innovation::Vector{T}
    invariant_gradient::Vector{T}
end

function _prepare_frame(state, ir, position, gradient)
    _sync_sources!(state, ir)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    source = T.(state.sources)
    log_scale = T(position[state.log_scale_index])
    scale = exp(log_scale)
    innovation = Vector{T}(undef, length(source))
    invariant_gradient = Vector{T}(undef, length(source))
    for pair_number in eachindex(source)
        idx = state.effect_indices[pair_number]
        source_scale = scale^source[pair_number]
        innovation[pair_number] = position[idx] / source_scale
        invariant_gradient[pair_number] = source_scale * gradient[idx]
    end
    TuringAdaptiveCenteringFrame(
        source, log_scale, scale, innovation, invariant_gradient,
    )
end

function _score_candidate(frame, pair_number, _idx, _value, candidate)
    1 <= pair_number <= length(frame.source) ||
        throw(BoundsError(frame.source, pair_number))
    target_source = candidate.c
    current_source = frame.source[pair_number]
    candidate_scale = frame.scale^target_source
    (
        (target_source - current_source) * frame.log_scale,
        candidate_scale * frame.innovation[pair_number],
        frame.invariant_gradient[pair_number] / candidate_scale,
    )
end

function _coordinate_indices(problem, n_beta, n_groups)
    beta_name = DynamicPPL.@varname(beta_pop)
    log_scale_name = DynamicPPL.@varname(group_1_1.log_scale)
    effects_name = DynamicPPL.@varname(group_1_1.z)
    ranges = DynamicPPL.get_all_ranges_and_transforms(problem)
    haskey(ranges, beta_name) || _unsupported(
        "the DynamicPPL problem has no `beta_pop` coordinates",
    )
    haskey(ranges, log_scale_name) || _unsupported(
        "the DynamicPPL problem has no `group_1_1.log_scale` coordinate",
    )
    haskey(ranges, effects_name) || _unsupported(
        "the DynamicPPL problem has no `group_1_1.z` coordinates",
    )
    beta_range = ranges[beta_name].range
    log_scale_range = ranges[log_scale_name].range
    effects_range = ranges[effects_name].range
    length(beta_range) == n_beta || _unsupported(
        "`beta_pop` occupies $(length(beta_range)) coordinates for $n_beta " *
        "population-design columns",
    )
    length(log_scale_range) == 1 || _unsupported(
        "`group_1_1.log_scale` occupies $(length(log_scale_range)) coordinates",
    )
    length(effects_range) == n_groups || _unsupported(
        "`group_1_1.z` occupies $(length(effects_range)) coordinates for " *
        "$n_groups fitted groups",
    )
    beta_indices = collect(beta_range)
    log_scale_index = first(log_scale_range)
    effect_indices = collect(effects_range)
    coordinates = sort!(vcat(beta_indices, [log_scale_index], effect_indices))
    coordinates == collect(1:LogDensityProblems.dimension(problem)) ||
        _unsupported(
            "the DynamicPPL problem contains coordinates outside the simple case",
        )
    beta_indices, log_scale_index, effect_indices
end

function _adaptive_centering_reparametrizer(log_scale_index, effect_indices)
    state = TuringAdaptiveCenteringState(
        log_scale_index, effect_indices, zeros(length(effect_indices)),
    )
    pairs = [begin
        location = TuringAdaptiveCenteringArgument{:location}(state, pair_number)
        log_scale = TuringAdaptiveCenteringArgument{:log_scale}(state, pair_number)
        idx => WarmupHMC.Reparametrization(
            WarmupHMC.PartiallyCentered(0.0),
            WarmupHMC.PartiallyCentered(0.0),
            location,
            log_scale,
        )
    end for (pair_number, idx) in enumerate(effect_indices)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

struct TuringSimpleRandomInterceptProblem{P,D,F,Y,G,B,E}
    problem::P
    design::D
    fixed::F
    response::Y
    group_indices::G
    beta_indices::B
    log_scale_index::Int
    effect_indices::E
    observation_scale::Float64
end

function TuringSimpleRandomInterceptProblem(
    problem, plan, block, beta_indices, log_scale_index, effect_indices,
)
    component = only(plan.predictors)
    TuringSimpleRandomInterceptProblem(
        problem,
        component.design.matrix,
        component.design.fixed,
        plan.response,
        block.indices,
        beta_indices,
        log_scale_index,
        effect_indices,
        _fixed_gaussian_scale(plan),
    )
end

LogDensityProblems.capabilities(::Type{<:TuringSimpleRandomInterceptProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::TuringSimpleRandomInterceptProblem) =
    LogDensityProblems.dimension(problem.problem)
function LogDensityProblems.logdensity(
    problem::TuringSimpleRandomInterceptProblem, position,
)
    LogDensityProblems.logdensity(problem.problem, position)
end

function LogDensityProblems.logdensity_and_gradient(
    problem::TuringSimpleRandomInterceptProblem, position,
)
    gradient = fill!(similar(position), zero(eltype(position)))
    log_scale = position[problem.log_scale_index]
    scale = exp(log_scale)
    inverse_variance = inv(problem.observation_scale^2)

    for observation in eachindex(problem.response)
        mean = problem.fixed[observation]
        for (column, idx) in enumerate(problem.beta_indices)
            mean += problem.design[observation, column] * position[idx]
        end
        group = problem.group_indices[observation]
        effect_idx = problem.effect_indices[group]
        group_effect = scale * position[effect_idx]
        residual_score =
            (problem.response[observation] - mean - group_effect) * inverse_variance
        for (column, idx) in enumerate(problem.beta_indices)
            gradient[idx] += problem.design[observation, column] * residual_score
        end
        gradient[effect_idx] += scale * residual_score
        gradient[problem.log_scale_index] += group_effect * residual_score
    end

    for idx in problem.beta_indices
        gradient[idx] -= position[idx]
    end
    gradient[problem.log_scale_index] -= log_scale
    for idx in problem.effect_indices
        gradient[idx] -= position[idx]
    end
    LogDensityProblems.logdensity(problem.problem, position), gradient
end

struct TuringTwoHSGPProblem{P}
    problem::P
    response::Vector{Float64}
    mu::TuringHSGPGradientComponent
    log_sigma::TuringHSGPGradientComponent
end

LogDensityProblems.capabilities(::Type{<:TuringTwoHSGPProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::TuringTwoHSGPProblem) =
    LogDensityProblems.dimension(problem.problem)
function LogDensityProblems.logdensity(problem::TuringTwoHSGPProblem, position)
    LogDensityProblems.logdensity(problem.problem, position)
end

function _hsgp_predictor_frame(component, position)
    n = length(component.fixed)
    eta = Vector{eltype(position)}(undef, n)
    for observation in 1:n
        value = component.fixed[observation]
        for (column, idx) in enumerate(component.beta_indices)
            value += component.design[observation, column] * position[idx]
        end
        eta[observation] = value
    end
    scales = Vector{eltype(position)}(undef, length(component.block.effects))
    for basis in eachindex(component.block.effects)
        scale = exp(BRM._adaptive_hsgp_log_scale(
            position, component.block, basis))
        scales[basis] = scale
        coefficient = scale * position[component.block.effects[basis]]
        for observation in 1:n
            eta[observation] += component.PHI[observation, basis] * coefficient
        end
    end
    eta, scales
end

function _accumulate_hsgp_gradient!(
    gradient, component, position, predictor_score, scales,
)
    for (column, idx) in enumerate(component.beta_indices)
        value = -position[idx]
        for observation in eachindex(predictor_score)
            value += component.design[observation, column] *
                     predictor_score[observation]
        end
        gradient[idx] = value
    end

    block = component.block
    rho_idx = only(block.length_scales)
    rho_offset = exp(position[rho_idx])
    rho = only(block.length_scale_lower) + rho_offset
    sd_score = zero(eltype(position))
    rho_score = zero(eltype(position))
    for basis in eachindex(block.effects)
        projected_score = zero(eltype(position))
        for observation in eachindex(predictor_score)
            projected_score += component.PHI[observation, basis] *
                               predictor_score[observation]
        end
        effect_idx = block.effects[basis]
        standard_weight = position[effect_idx]
        scale = scales[basis]
        gradient[effect_idx] = scale * projected_score - standard_weight
        spectral_score = scale * standard_weight * projected_score
        sd_score += spectral_score
        rho_score += spectral_score * rho_offset *
            (0.5 / rho - 0.5 * rho * block.omega2[basis, 1])
    end

    # The supported native HSGP priors are LogNormal(0, s). In the linked
    # DynamicPPL frame, marginal SD is exp(v), while length scale is
    # rho_lower + exp(u). These terms include both prior and link Jacobian.
    sd_idx = block.sd
    gradient[sd_idx] = sd_score -
        position[sd_idx] / component.sd_prior_scale^2
    gradient[rho_idx] = rho_score + one(eltype(position)) -
        rho_offset *
        (log(rho) / component.rho_prior_scale^2 + one(eltype(position))) / rho
    gradient
end

function LogDensityProblems.logdensity_and_gradient(
    problem::TuringTwoHSGPProblem, position,
)
    mu, mu_scales = _hsgp_predictor_frame(problem.mu, position)
    log_sigma, sigma_scales =
        _hsgp_predictor_frame(problem.log_sigma, position)
    mu_score = Vector{eltype(position)}(undef, length(problem.response))
    log_sigma_score = similar(mu_score)
    for observation in eachindex(problem.response)
        residual = problem.response[observation] - mu[observation]
        inverse_scale = exp(-log_sigma[observation])
        standardized = residual * inverse_scale
        mu_score[observation] = standardized * inverse_scale
        log_sigma_score[observation] = standardized^2 - one(eltype(position))
    end

    gradient = fill!(similar(position), zero(eltype(position)))
    _accumulate_hsgp_gradient!(
        gradient, problem.mu, position, mu_score, mu_scales)
    _accumulate_hsgp_gradient!(
        gradient, problem.log_sigma, position, log_sigma_score, sigma_scales)
    LogDensityProblems.logdensity(problem.problem, position), gradient
end

function _warmuphmc_hsgp_extension()
    extension = Base.get_extension(
        BRM, :BayesianRegressionModelsWarmupHMCExt)
    isnothing(extension) && error(
        "Turing backend: the base WarmupHMC extension is unavailable")
    extension
end

function _two_hsgp_adaptive_problem(backend, problem, ad_backend)
    contract = _two_hsgp_contract(backend)
    blocks, mu, sigma = _two_hsgp_geometry(backend, problem, contract)
    warmup_extension = _warmuphmc_hsgp_extension()
    state, ir = warmup_extension._adaptive_hsgp_centering_reparametrizer(blocks)
    scoring_plan = WarmupHMC.CandidateScoringPlan(
        (ir_, position, gradient) ->
            warmup_extension._prepare_frame(
                state, ir_, position, gradient),
        warmup_extension._score_candidate;
        synchronize! = ir_ ->
            warmup_extension._sync_sources!(state, ir_),
    )
    gradient_problem = TuringTwoHSGPProblem(
        problem, collect(Float64, contract.plan.response), mu, sigma)
    WarmupHMC.ReparametrizedProblem(
        ir, gradient_problem, ad_backend; scoring_plan)
end

"""
    adaptive_centering_problem(
        backend::TuringBRMI,
        problem::DynamicPPL.LogDensityFunction,
        ad_backend,
    )

Wrap a supported native Turing model in WarmupHMC's online partial-centering
transform. `problem` must be a
`DynamicPPL.LogDensityFunction` built from `backend.model`. The adapter prepares
the exact DynamicPPL density while supplying the bounded model's closed-form
gradient to WarmupHMC. DynamicPPL's own range metadata identifies every native
coordinate; no compiled-Stan names or ordering are reused. `ad_backend`
differentiates only WarmupHMC's small coordinate transform.

The first contract is an identity-link `Normal(predictor, fixed_scale)` model
with default standard-Normal population, log-scale, and innovation priors.
Construct the problem with `DynamicPPL.UnlinkAll()`; a linked or constrained
coordinate system is outside this milestone.

The generated Turing model is the fixed target `c=0`. Warmup begins in that
literal frame and scores the usual `c=0:0.1:1` candidates independently for
each group innovation. At `c=1`, a sampler coordinate is the model-scale group
effect and is mapped back to the model's `z` by division by `exp(log_scale)`.

The HSGP contract is the two-predictor Gaussian shape `mu ~ hsgp(x)` plus
`log(sigma) ~ hsgp(x)`. Construct that problem with `DynamicPPL.LinkAll()`.
Each basis weight uses the same zero-location, per-basis log spectral scale,
exact transport/Jacobian, and candidate scoring implementation as compiled
StanBlocks; only coordinate lookup is DynamicPPL-specific.
"""
function BRM.adaptive_centering_problem(
    backend::BRM.TuringBRMI,
    problem::DynamicPPL.LogDensityFunction,
    ad_backend;
)
    _has_hsgp_term(backend) &&
        return _two_hsgp_adaptive_problem(backend, problem, ad_backend)
    block = _simple_random_intercept_block(backend)
    problem.model === backend.model || _unsupported(
        "the DynamicPPL problem was not built from this backend's exact model",
    )
    problem.transform_strategy isa DynamicPPL.UnlinkAll || _unsupported(
        "the DynamicPPL problem is linked; construct it with `DynamicPPL.UnlinkAll()`",
    )

    n_beta = size(only(backend.plan.predictors).design.matrix, 2)
    beta_indices, log_scale_index, effect_indices =
        _coordinate_indices(problem, n_beta, length(block.levels))
    state, ir = _adaptive_centering_reparametrizer(
        log_scale_index, effect_indices,
    )
    scoring_plan = WarmupHMC.CandidateScoringPlan(
        (ir_, position, gradient) ->
            _prepare_frame(state, ir_, position, gradient),
        _score_candidate;
        synchronize! = ir_ -> _sync_sources!(state, ir_),
    )
    gradient_problem = TuringSimpleRandomInterceptProblem(
        problem,
        backend.plan,
        block,
        beta_indices,
        log_scale_index,
        effect_indices,
    )
    WarmupHMC.ReparametrizedProblem(
        ir, gradient_problem, ad_backend; scoring_plan,
    )
end

end # module
