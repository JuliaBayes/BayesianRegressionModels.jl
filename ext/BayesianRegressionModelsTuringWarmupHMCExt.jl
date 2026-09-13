module BayesianRegressionModelsTuringWarmupHMCExt

using BayesianRegressionModels
using Distributions: Normal
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

_unsupported(reason) = error("Turing backend: $_SIMPLE_CONTRACT; $reason.")

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

"""
    adaptive_centering_problem(
        backend::TuringBRMI,
        problem::DynamicPPL.LogDensityFunction,
        ad_backend,
    )

Wrap the minimal native Turing random-intercept model in WarmupHMC's online
partial-centering transform. `problem` must be a
`DynamicPPL.LogDensityFunction` built from `backend.model`. The adapter prepares
the exact DynamicPPL density while supplying the bounded model's closed-form
gradient to WarmupHMC. DynamicPPL's own range metadata identifies `beta_pop`,
`group_1_1.log_scale`, and every scalar coordinate of `group_1_1.z`; no
compiled-Stan names or ordering are reused. `ad_backend` differentiates only
WarmupHMC's small coordinate transform.

The first contract is an identity-link `Normal(predictor, fixed_scale)` model
with default standard-Normal population, log-scale, and innovation priors.
Construct the problem with `DynamicPPL.UnlinkAll()`; a linked or constrained
coordinate system is outside this milestone.

The generated Turing model is the fixed target `c=0`. Warmup begins in that
literal frame and scores the usual `c=0:0.1:1` candidates independently for
each group innovation. At `c=1`, a sampler coordinate is the model-scale group
effect and is mapped back to the model's `z` by division by `exp(log_scale)`.
"""
function BRM.adaptive_centering_problem(
    backend::BRM.TuringBRMI,
    problem::DynamicPPL.LogDensityFunction,
    ad_backend;
)
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
