"""A semantic axis whose name is part of its type and whose keys are explicit."""
struct NativePPLAxis{Name,K}
    keys::K
end

NativePPLAxis(name::Symbol, keys::K) where {K} = NativePPLAxis{name,K}(keys)
native_axis_name(::NativePPLAxis{Name}) where {Name} = Name
Base.length(axis::NativePPLAxis) = length(axis.keys)

"""An external value required by a native PPL plan."""
struct NativePPLInput{Name,Role,A,T}
    axis::A
end

NativePPLInput(name::Symbol, role::Symbol, axis::A, ::Type{T}) where {A,T} =
    NativePPLInput{name,role,A,T}(axis)
native_input_name(::NativePPLInput{Name}) where {Name} = Name
native_input_role(::NativePPLInput{Name,Role}) where {Name,Role} = Role
Base.eltype(::NativePPLInput{Name,Role,A,T}) where {Name,Role,A,T} = T

abstract type NativePPLSupport end
struct NativePPLRealSupport <: NativePPLSupport end
struct NativePPLPositiveSupport <: NativePPLSupport end
struct NativePPLCholeskyCorrelationSupport{K} <: NativePPLSupport end

"""A typed map from an unconstrained coordinate to a parameter's support."""
abstract type NativePPLTransform{S<:NativePPLSupport} end
struct NativePPLIdentityTransform <: NativePPLTransform{NativePPLRealSupport} end
struct NativePPLExpTransform <: NativePPLTransform{NativePPLPositiveSupport} end
struct NativePPLCholeskyCorrelationTransform{K} <:
       NativePPLTransform{NativePPLCholeskyCorrelationSupport{K}} end

@inline native_transform_forward(::NativePPLIdentityTransform, value) = value
@inline native_transform_forward(::NativePPLExpTransform, value) = exp(value)
@inline native_transform_logforward(transform::NativePPLTransform, value) =
    log(native_transform_forward(transform, value))
@inline native_transform_logforward(::NativePPLExpTransform, value) = value
@inline native_transform_logabsdetjac(::NativePPLIdentityTransform, value) = zero(value)
@inline native_transform_logabsdetjac(::NativePPLExpTransform, value) = value

"""A parameter block and its coordinates in the flat unconstrained vector."""
struct NativePPLParameter{
    Name,S<:NativePPLSupport,Tr<:NativePPLTransform{S},A,R,
}
    support::S
    transform::Tr
    axis::A
    unconstrained::R
end

NativePPLParameter(name::Symbol, support::S, transform::Tr,
                   axis::A, unconstrained::R) where {S,Tr,A,R} =
    NativePPLParameter{name,S,Tr,A,R}(support, transform, axis, unconstrained)
native_parameter_name(::NativePPLParameter{Name}) where {Name} = Name

@inline function native_parameter_value(parameter::NativePPLParameter,
                                         position::AbstractVector,
                                         coordinate::Int)
    native_transform_forward(parameter.transform, position[coordinate])
end

@inline function native_parameter_logabsdetjac(parameter::NativePPLParameter,
                                               position::AbstractVector,
                                               coordinate::Int)
    native_transform_logabsdetjac(parameter.transform, position[coordinate])
end

@inline function native_parameter_logvalue(parameter::NativePPLParameter,
                                           position::AbstractVector,
                                           coordinate::Int)
    native_transform_logforward(parameter.transform, position[coordinate])
end

@inline native_parameter_value(parameter::NativePPLParameter,
                               position::AbstractVector) =
    native_parameter_value(parameter, position, first(parameter.unconstrained))
@inline native_parameter_logabsdetjac(parameter::NativePPLParameter,
                                      position::AbstractVector) =
    native_parameter_logabsdetjac(
        parameter, position, first(parameter.unconstrained))
@inline native_parameter_logvalue(parameter::NativePPLParameter,
                                  position::AbstractVector) =
    native_parameter_logvalue(parameter, position, first(parameter.unconstrained))

abstract type NativePPLNode end
abstract type NativePPLFactor end
abstract type NativePPLLocationNode{Name} <: NativePPLNode end

"""An intercept plus one coefficient for each named continuous feature."""
struct NativePPLAffineNode{Name,Inputs,A,R} <: NativePPLLocationNode{Name}
    axis::A
    intercept_index::Int
    slope_indices::R
end

"""Broadcast one scalar parameter value over an observation axis."""
struct NativePPLScalarBroadcastNode{Name,Parameter,A} <:
       NativePPLLocationNode{Name}
    axis::A
    unconstrained_index::Int
end

function NativePPLScalarBroadcastNode(
    name::Symbol, parameter::Symbol, axis::A, unconstrained_index::Int,
) where {A}
    unconstrained_index > 0 || throw(ArgumentError(
        "native PPL scalar broadcast coordinate must be positive"))
    NativePPLScalarBroadcastNode{name,parameter,A}(
        axis, unconstrained_index)
end

native_node_name(::NativePPLScalarBroadcastNode{Name}) where {Name} = Name
native_scalar_parameter(
    ::NativePPLScalarBroadcastNode{Name,Parameter},
) where {Name,Parameter} = Parameter
native_affine_inputs(::NativePPLScalarBroadcastNode) = ()

function NativePPLAffineNode(name::Symbol, inputs::Tuple, axis::A,
                             intercept_index::Int, slope_indices::R) where {A,R}
    isempty(inputs) && throw(ArgumentError(
        "native PPL affine node requires at least one input"))
    all(input -> input isa Symbol, inputs) || throw(ArgumentError(
        "native PPL affine inputs must be named Symbols; got $inputs"))
    length(unique(inputs)) == length(inputs) || throw(ArgumentError(
        "native PPL affine inputs must be unique; got $inputs"))
    length(inputs) == length(slope_indices) || throw(ArgumentError(
        "native PPL affine input and slope-coordinate counts must match"))
    NativePPLAffineNode{name,inputs,A,R}(
        axis, intercept_index, slope_indices)
end
NativePPLAffineNode(name::Symbol, input::Symbol, axis::A,
                    intercept_index::Int, slope_index::Int) where {A} =
    NativePPLAffineNode(
        name, (input,), axis, intercept_index, (slope_index,))
native_node_name(::NativePPLAffineNode{Name}) where {Name} = Name
native_affine_inputs(::NativePPLAffineNode{Name,Inputs}) where {Name,Inputs} =
    Inputs
function native_affine_input(node::NativePPLAffineNode)
    inputs = native_affine_inputs(node)
    length(inputs) == 1 || throw(ArgumentError(
        "native PPL affine node has $(length(inputs)) inputs; use " *
        "`native_affine_inputs`"))
    only(inputs)
end

"""A staged elementwise exponential link over one named deterministic node."""
struct NativePPLExpNode{Name,Input,A} <: NativePPLNode
    axis::A
end

NativePPLExpNode(name::Symbol, input::Symbol, axis::A) where {A} =
    NativePPLExpNode{name,input,A}(axis)
native_node_name(::NativePPLExpNode{Name}) where {Name} = Name
native_exp_input(::NativePPLExpNode{Name,Input}) where {Name,Input} = Input

"""A fitted mean-centering transform over one raw input column."""
struct NativePPLCenterNode{Name,Input,A,T} <: NativePPLNode
    axis::A
    mean::T
end

NativePPLCenterNode(name::Symbol, input::Symbol, axis::A, mean::T) where {A,T} =
    NativePPLCenterNode{name,input,A,T}(axis, mean)
native_node_name(::NativePPLCenterNode{Name}) where {Name} = Name
native_center_input(::NativePPLCenterNode{Name,Input}) where {Name,Input} = Input
native_fitted_transform_input(node::NativePPLCenterNode) =
    native_center_input(node)

"""A fitted corrected-sample-SD standardization over one raw input column."""
struct NativePPLZScaleNode{Name,Input,A,T} <: NativePPLNode
    axis::A
    mean::T
    scale::T
end

NativePPLZScaleNode(name::Symbol, input::Symbol, axis::A,
                    mean::T, scale::T) where {A,T} =
    NativePPLZScaleNode{name,input,A,T}(axis, mean, scale)
native_node_name(::NativePPLZScaleNode{Name}) where {Name} = Name
native_zscale_input(::NativePPLZScaleNode{Name,Input}) where {Name,Input} = Input
native_fitted_transform_input(node::NativePPLZScaleNode) =
    native_zscale_input(node)

"""Independent standard-normal prior over an unconstrained parameter range."""
struct NativePPLStandardNormalFactor{Parameter,R} <: NativePPLFactor
    unconstrained::R
end

NativePPLStandardNormalFactor(parameter::Symbol, unconstrained::R) where {R} =
    NativePPLStandardNormalFactor{parameter,R}(unconstrained)

"""Generating Normal factor for one unconditioned scalar stochastic site."""
struct NativePPLScalarNormalFactor{Site,Location,Scale,L,S} <:
       NativePPLFactor
    unconstrained_index::Int
    location::L
    scale::S
end

function NativePPLScalarNormalFactor(
    site::Symbol, location_name::Symbol, scale_name::Symbol,
    unconstrained_index::Int, location::L, scale::S,
) where {L,S}
    unconstrained_index > 0 || throw(ArgumentError(
        "native PPL scalar Normal site coordinate must be positive"))
    NativePPLScalarNormalFactor{
        site,location_name,scale_name,L,S,
    }(unconstrained_index, location, scale)
end

"""Exponential prior on a constrained scalar; its parameter owns the transform."""
struct NativePPLExponentialFactor{Parameter,T} <: NativePPLFactor
    unconstrained_index::Int
    scale::T
end

NativePPLExponentialFactor(parameter::Symbol, index::Int, scale::T) where {T} =
    NativePPLExponentialFactor{parameter,T}(index, scale)

"""Row-wise Normal observation factor."""
struct NativePPLNormalFactor{Response,Location,Scale,A} <: NativePPLFactor
    axis::A
end

NativePPLNormalFactor(response::Symbol, location::Symbol, scale::Symbol, axis::A) where {A} =
    NativePPLNormalFactor{response,location,scale,A}(axis)

"""Row-wise Bernoulli observation factor parameterized by logits."""
struct NativePPLBernoulliLogitFactor{Response,Logit,A} <: NativePPLFactor
    axis::A
end

NativePPLBernoulliLogitFactor(response::Symbol, logit::Symbol, axis::A) where {A} =
    NativePPLBernoulliLogitFactor{response,logit,A}(axis)

"""Row-wise Poisson observation factor linked to a staged positive rate."""
struct NativePPLPoissonFactor{Response,Rate,A} <: NativePPLFactor
    axis::A
end

NativePPLPoissonFactor(response::Symbol, rate::Symbol, axis::A) where {A} =
    NativePPLPoissonFactor{response,rate,A}(axis)

abstract type NativePPLQuery end
struct NativePPLLinearPredictor <: NativePPLQuery end
struct NativePPLPointwiseLogLikelihood <: NativePPLQuery end
struct NativePPLPosteriorPredictive <: NativePPLQuery end
struct NativePPLNodeOutput{Name} <: NativePPLQuery end
NativePPLNodeOutput(name::Symbol) = NativePPLNodeOutput{name}()
native_ppl_node_output_name(::NativePPLNodeOutput{Name}) where {Name} = Name

"""Rule selecting the prepared executor's numeric scalar as an output eltype."""
struct NativePPLPreparedElementType end

"""Rule selecting a fixed semantic output element type."""
struct NativePPLFixedElementType{T} end

"""Dense, one-dimensional caller-owned output layout."""
struct NativePPLDenseVectorLayout end

"""Dense draw-by-observation output owned by the caller."""
struct NativePPLDenseMatrixLayout end

"""Pre-execution semantic axis, element-type rule, and layout for one output."""
struct NativePPLOutputSignature{A,E,L}
    axis::A
    element_type::E
    layout::L
end

"""Pre-execution signature for a draw-by-observation query block."""
struct NativePPLBatchOutputSignature{D,A,E,L}
    draw_axis::D
    observation_axis::A
    element_type::E
    layout::L
end

native_output_axis(signature::NativePPLOutputSignature) = signature.axis
native_output_axis(signature::NativePPLBatchOutputSignature) =
    signature.observation_axis
native_output_axes(signature::NativePPLOutputSignature) = (signature.axis,)
native_output_axes(signature::NativePPLBatchOutputSignature) =
    (signature.draw_axis, signature.observation_axis)
native_output_draw_axis(signature::NativePPLBatchOutputSignature) =
    signature.draw_axis
native_output_eltype(signature::NativePPLOutputSignature, prepared) =
    native_output_eltype(signature.element_type, prepared)
native_output_eltype(signature::NativePPLBatchOutputSignature, prepared) =
    native_output_eltype(signature.element_type, prepared)
native_output_eltype(::NativePPLPreparedElementType, prepared) = eltype(prepared)
native_output_eltype(::NativePPLPreparedElementType, ::Type{T}) where {T<:AbstractFloat} = T
native_output_eltype(::NativePPLFixedElementType{T}, _) where {T} = T
native_output_layout(signature::NativePPLOutputSignature) = signature.layout
native_output_layout(signature::NativePPLBatchOutputSignature) = signature.layout
native_output_layout_accepts(::NativePPLDenseVectorLayout, output) =
    output isa DenseVector
native_output_layout_accepts(::NativePPLDenseMatrixLayout, output) =
    output isa DenseMatrix

"""
Output, frequency, effect, and lifetime contract for one graph query.

The query kind and execution properties are type parameters so query planning
does not depend on a post-hoc name registry.
"""
struct NativePPLQuerySpec{Kind,Stage,Effect,Lifetime,S}
    output::S
end

NativePPLQuerySpec(kind::Symbol, stage::Symbol, effect::Symbol,
                   lifetime::Symbol, output::S) where {S} =
    NativePPLQuerySpec{kind,stage,effect,lifetime,S}(output)
native_query_name(::NativePPLQuerySpec{Kind}) where {Kind} = Kind
native_query_effect(::NativePPLQuerySpec{Kind,Stage,Effect}) where {Kind,Stage,Effect} =
    Effect
native_query_output(spec::NativePPLQuerySpec) = spec.output

function _native_ppl_queries(
    observation_axis,
    predictive_element=NativePPLPreparedElementType(),
)
    numeric_output = NativePPLOutputSignature(
        observation_axis, NativePPLPreparedElementType(),
        NativePPLDenseVectorLayout())
    predictive_output = predictive_element isa NativePPLPreparedElementType ?
        numeric_output : NativePPLOutputSignature(
            observation_axis, predictive_element, NativePPLDenseVectorLayout())
    linear_predictor = NativePPLQuerySpec(
        :linear_predictor, :per_draw, :workspace, :until_next_evaluation,
        numeric_output)
    pointwise_loglikelihood = NativePPLQuerySpec(
        :pointwise_loglikelihood, :per_draw, :workspace,
        :until_next_evaluation, numeric_output)
    posterior_predictive = NativePPLQuerySpec(
        :posterior_predictive, :per_draw, :rng, :caller_owned,
        predictive_output)
    (; linear_predictor, pointwise_loglikelihood, posterior_predictive)
end

_native_ppl_queries(observation_axis, ::NativePPLNormalFactor) =
    _native_ppl_queries(observation_axis)
_native_ppl_queries(observation_axis, ::NativePPLBernoulliLogitFactor) =
    _native_ppl_queries(observation_axis, NativePPLFixedElementType{Bool}())
_native_ppl_queries(observation_axis, ::NativePPLPoissonFactor) =
    _native_ppl_queries(observation_axis, NativePPLFixedElementType{Int}())

"""
    NativePPLPlan

Typed, inspectable native-Julia execution plan. `bindings` are the initial
input values captured from the source `BRMI`; execution preparation copies
them into separately owned buffers, so the plan itself is never a workspace.

This is deliberately internal while the public native-PPL naming is unsettled.
The staged location schedule supports both affine predictor graphs and a
parameter-backed scalar broadcast produced by active component composition;
other graph shapes fail closed.
"""
struct NativePPLPlan{A,I,P,N,F,Q,B}
    axes::A
    inputs::I
    parameters::P
    nodes::N
    factors::F
    queries::Q
    bindings::B
end

"""Buffers required by one primal native-PPL evaluation."""
mutable struct NativePPLBuffers{T,V<:Vector{T}}
    location::V
    pointwise_loglikelihood::V
end

"""Inspectable allocation contract for a prepared native PPL plan."""
struct NativePPLWorkspaceSpec{A}
    observation_axis::A
    gradient_length::Int
end

"""Validated, numeric input binding for a native PPL plan."""
struct NativePPLPrepared{P,X,Y,S}
    plan::P
    predictors::X
    response::Y
    workspace_spec::S
end

"""Explicit absence of an observation binding in a prediction-only replay."""
struct NativePPLNoResponse end

function Base.eltype(prepared::NativePPLPrepared)
    isempty(prepared.predictors) ||
        return eltype(first(values(prepared.predictors)))
    native_ppl_has_response(prepared) && return eltype(prepared.response)
    float(eltype(prepared.plan.inputs.response))
end
function Base.getproperty(prepared::NativePPLPrepared, name::Symbol)
    name === :predictor || return getfield(prepared, name)
    predictors = getfield(prepared, :predictors)
    length(predictors) == 1 || throw(ArgumentError(
        "native PPL prepared model has $(length(predictors)) predictors; " *
        "use `.predictors`"))
    only(values(predictors))
end
function Base.propertynames(prepared::NativePPLPrepared, private::Bool=false)
    names = fieldnames(typeof(prepared))
    length(getfield(prepared, :predictors)) == 1 ?
        (names..., :predictor) : names
end
native_ppl_has_response(prepared::NativePPLPrepared) =
    !(prepared.response isa NativePPLNoResponse)
LogDensityProblems.dimension(prepared::NativePPLPrepared) =
    LogDensityProblems.dimension(prepared.plan)

"""
Reusable storage for density and reverse-mode gradient execution.

The returned gradient and pointwise buffers alias this workspace. Callers that
need to retain a result across another evaluation must copy it. A workspace is
single-task state; concurrent callers use separate workspaces over the same
immutable `NativePPLPrepared` value.
"""
struct NativePPLWorkspace{T,B,G,D}
    primal::B
    gradient::G
    derivative::D
end

Base.eltype(::NativePPLWorkspace{T}) where {T} = T

function NativePPLWorkspace(prepared::NativePPLPrepared, ::Type{T}=eltype(prepared)) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError(
        "native PPL workspace element type must be concrete; got $T"))
    n = length(prepared.workspace_spec.observation_axis)
    buffers() = NativePPLBuffers(zeros(T, n), zeros(T, n))
    NativePPLWorkspace{T,NativePPLBuffers{T,Vector{T}},Vector{T},Nothing}(
        buffers(), zeros(T, prepared.workspace_spec.gradient_length), nothing)
end

function _native_ppl_workspace end
_native_ppl_workspace(prepared::NativePPLPrepared, ::Type{T}) where {T<:AbstractFloat} =
    NativePPLWorkspace(prepared, T)
_native_ppl_workspace(::NativePPLPrepared, ::Type{<:AbstractFloat}, backend) =
    throw(ArgumentError(
        "native PPL derivative workspaces require loading DifferentiationInterface; " *
        "only AutoEnzyme is tested, recommended, and guaranteed"))

"""
    NativePPL.LogDensityProblem(prepared, backend)

An order-1 `LogDensityProblems` adapter over a conditioned native PPL value.
The adapter owns the derivative workspace prepared by `backend`; construct one
adapter per concurrent task or sampler chain. Gradient results are copied at
the interface boundary because `LogDensityProblems` gives their ownership to
the caller, while the lower-level `NativePPL.logdensity_and_gradient!` API
deliberately returns a reusable workspace buffer.

`DifferentiationInterface.AutoEnzyme()` is the supported backend. WarmupHMC is
not required to construct or use this adapter.
"""
struct NativePPLLogDensityProblem{P,W}
    prepared::P
    workspace::W

    function NativePPLLogDensityProblem(
        prepared::P,
        workspace::W,
        ::Val{:workspace},
    ) where {P<:NativePPLPrepared,W<:NativePPLWorkspace}
        native_ppl_has_response(prepared) || throw(ArgumentError(
            "native PPL log-density adapters require a conditioned response; " *
            "the prepared value is prediction-only"))
        workspace.derivative === nothing && throw(ArgumentError(
            "native PPL log-density adapters require a derivative-prepared " *
            "workspace; construct one with AutoEnzyme()"))
        new{P,W}(prepared, workspace)
    end
end

function NativePPLLogDensityProblem(prepared::NativePPLPrepared, backend)
    workspace = _native_ppl_workspace(prepared, eltype(prepared), backend)
    NativePPLLogDensityProblem(prepared, workspace, Val(:workspace))
end

Base.eltype(problem::NativePPLLogDensityProblem) = eltype(problem.prepared)
LogDensityProblems.capabilities(::Type{<:NativePPLLogDensityProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::NativePPLLogDensityProblem) =
    LogDensityProblems.dimension(problem.prepared)
LogDensityProblems.logdensity(problem::NativePPLLogDensityProblem,
                              position::AbstractVector) =
    _native_ppl_logdensity!(problem.workspace, problem.prepared, position)
function LogDensityProblems.logdensity_and_gradient(
    problem::NativePPLLogDensityProblem,
    position::AbstractVector,
)
    density, gradient = _native_ppl_logdensity_and_gradient!(
        problem.workspace, problem.prepared, position)
    density, copy(gradient)
end

LogDensityProblems.dimension(plan::NativePPLPlan) =
    sum(length(parameter.unconstrained) for parameter in plan.parameters)

function Base.show(io::IO, plan::NativePPLPlan)
    print(io, "NativePPLPlan(", LogDensityProblems.dimension(plan),
          " unconstrained parameters, ", length(plan.axes.observation),
          " observations)\n")
    print(io, "  inputs: ",
          join((string(name) for name in
                (keys(plan.inputs.predictors)...,
                 native_input_name(plan.inputs.response))), ", "), "\n")
    print(io, "  parameters: ",
          join((string(native_parameter_name(parameter)) for parameter in plan.parameters),
               ", "), "\n")
    print(io, "  nodes: ",
          join((string(name) for name in
                (keys(plan.nodes.transforms)...,
                 native_node_name(plan.nodes.location),
                 (hasproperty(plan.nodes, :rate) ?
                  (native_node_name(plan.nodes.rate),) : ())...)), ", "), "\n")
    print(io, "  factors: ",
          join((string(nameof(typeof(factor))) for factor in plan.factors), ", "), "\n")
    print(io, "  queries: ",
          join((string(native_query_name(query)) for query in plan.queries), ", "))
end

function Base.show(io::IO, prepared::NativePPLPrepared)
    print(io, "NativePPLPrepared(", length(prepared.plan.axes.observation),
          " observations, eltype=", eltype(prepared))
    native_ppl_has_response(prepared) || print(io, ", prediction-only")
    print(io, ")")
end

function Base.show(io::IO, workspace::NativePPLWorkspace)
    print(io, "NativePPLWorkspace(eltype=", eltype(workspace),
          ", location=", length(workspace.primal.location),
          ", pointwise_loglikelihood=",
          length(workspace.primal.pointwise_loglikelihood),
          ", gradient=", length(workspace.gradient), ")")
end

"""Fail-closed diagnostic for a formula capability not yet in the native PPL."""
struct NativePPLCapabilityError <: Exception
    capability::Symbol
    detail::String
end

function Base.showerror(io::IO, err::NativePPLCapabilityError)
    print(io, "native PPL does not support capability `", err.capability,
          "` in the walking skeleton: ", err.detail)
end

_native_ppl_ref_name(x::NamedColumn) = _native_ppl_ref_name(x, parent(x))
_native_ppl_ref_name(x::NamedColumn, ::MissingColumn) = name(x)
_native_ppl_ref_name(x::NamedColumn, operation::ExprColumn{typeof(~)}) = name(x)
_native_ppl_ref_name(_, _) = nothing
_native_ppl_ref_name(_) = nothing

function _native_ppl_sampling_rhs(brmi::BRMI, key::Symbol)
    haskey(brmi.operations, key) || throw(NativePPLCapabilityError(
        :missing_binding, "formula name `$key` has no defining operation"))
    named = brmi.operations[key]
    named isa NamedColumn || throw(NativePPLCapabilityError(
        :operation_shape, "`$key` is not a named formula operation"))
    operation = parent(named)
    operation isa ExprColumn || throw(NativePPLCapabilityError(
        :operation_shape, "`$key` is a data binding, not a sampling statement"))
    getf(operation) === (~) || throw(NativePPLCapabilityError(
        :operation_shape, "`$key` is not defined by `~`"))
    isempty(getkwargs(operation)) || throw(NativePPLCapabilityError(
        :sampling_keywords, "sampling statement `$key` has keywords"))
    getargs(operation, 2)
end

function _native_ppl_predictor_term(term, key::Symbol)
    if term isa NamedColumn && parent(term) isa DataColumn
        return (; column=term, transform=:identity)
    end
    if term isa ExprColumn &&
       (getf(term) === center || getf(term) === zscale ||
        getf(term) === standardize)
        transform = getf(term) === center ? :center : :zscale
        spelling = getf(term) === standardize ? :standardize : transform
        isempty(getkwargs(term)) || throw(NativePPLCapabilityError(
            :predictor_transform,
            "`$spelling` in predictor `$key` cannot have keywords"))
        args = getargs(term)
        length(args) == 1 || throw(NativePPLCapabilityError(
            :predictor_transform,
            "`$spelling` in predictor `$key` needs one raw data column"))
        column = only(args)
        column isa NamedColumn && parent(column) isa DataColumn ||
            throw(NativePPLCapabilityError(
                :predictor_transform,
                "`$spelling` in predictor `$key` must wrap one raw data column"))
        return (; column, transform)
    end
    throw(NativePPLCapabilityError(
        :predictor_terms,
        "each non-intercept term in `$key` must be a raw data column, " *
        "`center(x)`, or `zscale(x)` (`standardize(x)` is an alias); " *
        "offsets, interactions, groups, and other transforms are not lowered yet"))
end

function _native_ppl_offset_term(term, key::Symbol)
    term isa ExprColumn && getf(term) === offset || return nothing
    isempty(getkwargs(term)) || throw(NativePPLCapabilityError(
        :predictor_offset,
        "`offset(...)` in predictor `$key` cannot have keywords"))
    arguments = getargs(term)
    length(arguments) == 1 || throw(NativePPLCapabilityError(
        :predictor_offset,
        "`offset(...)` in predictor `$key` needs one scalar site or " *
        "row-valued data expression"))
    argument = only(arguments)
    if argument isa NamedColumn && parent(argument) isa DataColumn
        return (; kind=:data, name=name(argument), column=argument,
                transform=:identity)
    end
    if argument isa ExprColumn && getf(argument) === log
        isempty(getkwargs(argument)) || throw(NativePPLCapabilityError(
            :predictor_offset,
            "`log` in offset for predictor `$key` cannot have keywords"))
        log_arguments = getargs(argument)
        length(log_arguments) == 1 || throw(NativePPLCapabilityError(
            :predictor_offset,
            "`log` in offset for predictor `$key` needs one data column"))
        column = only(log_arguments)
        column isa NamedColumn && parent(column) isa DataColumn || throw(
            NativePPLCapabilityError(
                :predictor_offset,
                "`log` in offset for predictor `$key` must wrap one raw " *
                "data column"))
        return (; kind=:data, name=name(column), column, transform=:log)
    end
    sampled_name = _native_ppl_ref_name(argument)
    sampled_name === nothing && throw(NativePPLCapabilityError(
        :predictor_offset,
        "`offset(...)` in predictor `$key` must reference one sampled site, " *
        "raw data column, or `log(data_column)`"))
    (; kind=:sampled, name=sampled_name, column=nothing,
       transform=:identity)
end

function _native_ppl_varying_intercept(term, key::Symbol)
    term isa ExprColumn && getf(term) === (|) || return nothing
    isempty(getkwargs(term)) || throw(NativePPLCapabilityError(
        :group_term,
        "grouped term in predictor `$key` cannot have keywords"))
    arguments = getargs(term)
    length(arguments) == 3 || throw(NativePPLCapabilityError(
        :group_term,
        "grouped term in predictor `$key` must use `(1 | id | group)` " *
        "or `(0 + x | id | group)`"))
    coefficient, id, group = arguments
    predictors = if coefficient == 1
        (nothing,)
    elseif coefficient isa ExprColumn && getf(coefficient) === (+)
        isempty(getkwargs(coefficient)) || throw(NativePPLCapabilityError(
            :group_term,
            "varying-slope expression in `$key` cannot have keywords"))
        coefficient_terms = getargs(coefficient)
        intercept_count = count(==(1), coefficient_terms)
        suppression_count = count(==(0), coefficient_terms)
        (intercept_count == 1 && suppression_count == 0) ||
            (intercept_count == 0 && suppression_count == 1) || throw(
            NativePPLCapabilityError(
                :group_term,
                "grouped coefficients in `$key` must contain exactly one " *
                "intercept `1` or suppression `0`"))
        slope_terms = filter(term -> !(term isa Number && term in (0, 1)),
                             coefficient_terms)
        isempty(slope_terms) && throw(NativePPLCapabilityError(
            :group_term,
            "grouped coefficients in `$key` require at least one " *
            "slope predictor"))
        parsed = map(
            slope -> _native_ppl_predictor_term(slope, key), slope_terms)
        intercept_count == 1 ? (nothing, parsed...) : Tuple(parsed)
    else
        throw(NativePPLCapabilityError(
            :group_term,
            "grouped term in `$key` must be `1`, `0 + x + ...`, or " *
            "`1 + x + ...`"))
    end
    id isa Symbol || throw(NativePPLCapabilityError(
        :group_term, "grouped-term ID in `$key` must be a Symbol"))
    group isa NamedColumn && parent(group) isa DataColumn || throw(
        NativePPLCapabilityError(
            :group_term,
            "grouped term in `$key` must use one raw data column"))
    predictor = length(predictors) == 1 ? only(predictors) : nothing
    (; id, group, predictor, predictors)
end

function _native_ppl_affine_components(brmi::BRMI, key::Symbol;
                                       link=nothing)
    lhs, predictor = _native_ppl_sampling_rhs(brmi, key)
    if link === nothing
        _native_ppl_ref_name(lhs) === key || throw(NativePPLCapabilityError(
            :linked_predictor,
            "`$key` must have a bare, unlinked left-hand side"))
    else
        lhs isa ExprColumn && getf(lhs) === link || throw(
            NativePPLCapabilityError(
                :linked_predictor,
                "`$key` must use `$link($key)` on the left-hand side"))
        isempty(getkwargs(lhs)) || throw(NativePPLCapabilityError(
            :linked_predictor,
            "linked predictor `$key` cannot use link keywords"))
        arguments = getargs(lhs)
        length(arguments) == 1 &&
            _native_ppl_ref_name(only(arguments)) === key || throw(
            NativePPLCapabilityError(
                :linked_predictor,
                "linked predictor must be exactly `$link($key)`"))
    end
    predictor isa Number && predictor == 1 && return (
        ; predictors=(), offsets=(), data_offsets=(), groups=(),
        intercept=true)
    predictor isa ExprColumn && getf(predictor) === (+) ||
        throw(NativePPLCapabilityError(:predictor_terms,
            "`$key` must be an additive affine formula such as `1 + x + z`"))
    isempty(getkwargs(predictor)) || throw(NativePPLCapabilityError(
        :predictor_keywords, "predictor `$key` has keywords"))
    terms = getargs(predictor)
    !isempty(terms) || throw(NativePPLCapabilityError(:predictor_terms,
        "`$key` must contain an intercept"))

    intercepts = filter(term -> term isa Number && term == 1, terms)
    suppressions = filter(term -> term isa Number && term == 0, terms)
    has_intercept = if length(intercepts) == 1 && isempty(suppressions)
        true
    elseif isempty(intercepts) && length(suppressions) == 1
        false
    else
        throw(NativePPLCapabilityError(:predictor_terms,
            "`$key` must contain exactly one `1` intercept or one `0` " *
            "intercept suppression"))
    end
    nonintercept_terms = filter(
        term -> !(term isa Number && term in (0, 1)), terms)
    parsed_offsets = map(
        term -> _native_ppl_offset_term(term, key), nonintercept_terms)
    sampled_offsets = Tuple(
        parsed.name for parsed in parsed_offsets
        if parsed !== nothing && parsed.kind === :sampled)
    data_offsets = Tuple(
        parsed for parsed in parsed_offsets
        if parsed !== nothing && parsed.kind === :data)
    varying_groups = Tuple(filter(!isnothing,
        map(term -> _native_ppl_varying_intercept(term, key),
            nonintercept_terms)))
    predictor_terms = Tuple(
        term for (term, parsed_offset) in
            zip(nonintercept_terms, parsed_offsets)
        if parsed_offset === nothing &&
           _native_ppl_varying_intercept(term, key) === nothing)
    parsed = Tuple(_native_ppl_predictor_term(term, key)
                   for term in predictor_terms)
    predictor_names = map(term -> name(term.column), parsed)
    length(unique(predictor_names)) == length(predictor_names) || throw(
        NativePPLCapabilityError(
            :predictor_terms,
            "`$key` must use each raw predictor at most once; got " *
            "$(predictor_names)"))
    length(unique(sampled_offsets)) == length(sampled_offsets) || throw(
        NativePPLCapabilityError(
            :predictor_offset,
            "`$key` must use each sampled offset at most once; got " *
            "$(sampled_offsets)"))
    data_offset_names = map(offset -> offset.name, data_offsets)
    length(unique(data_offset_names)) == length(data_offset_names) || throw(
        NativePPLCapabilityError(
            :predictor_offset,
            "`$key` must use each data offset at most once; got " *
            "$(data_offset_names)"))
    (; predictors=parsed, offsets=sampled_offsets,
       data_offsets, groups=varying_groups, intercept=has_intercept)
end

function _native_ppl_affine_predictors(brmi::BRMI, key::Symbol)
    components = _native_ppl_affine_components(brmi, key)
    components.intercept || throw(NativePPLCapabilityError(
        :predictor_terms,
        "`$key` must contain an intercept in the current affine slice"))
    components.predictors
end

function _native_ppl_exponential_prior(brmi::BRMI, key::Symbol)
    lhs, prior = _native_ppl_sampling_rhs(brmi, key)
    _native_ppl_ref_name(lhs) === key || throw(NativePPLCapabilityError(
        :parameter_transform, "scale `$key` must have a bare left-hand side"))
    prior isa ExprColumn && getf(prior) === Exponential ||
        throw(NativePPLCapabilityError(:scale_prior,
            "scale `$key` must use `Exponential(scale)`"))
    isempty(getkwargs(prior)) || throw(NativePPLCapabilityError(
        :scale_prior, "`Exponential` prior for `$key` cannot have keywords"))
    args = getargs(prior)
    length(args) == 1 && only(args) isa Real || throw(NativePPLCapabilityError(
        :scale_prior, "`Exponential` prior for `$key` needs one numeric scale"))
    scale = only(args)
    isfinite(scale) && scale > 0 || throw(NativePPLCapabilityError(
        :scale_prior, "`Exponential` scale for `$key` must be finite and positive"))
    scale
end

function _native_ppl_intercept_normal_prior(brmi::BRMI, key::Symbol)
    matches = Tuple{Symbol,Any}[]
    for (operation_key, named) in pairs(brmi.operations)
        named isa NamedColumn || continue
        operation = parent(named)
        operation isa ExprColumn && getf(operation) === (~) || continue
        isempty(getkwargs(operation)) || continue
        lhs, prior = getargs(operation, 2)
        lhs isa ExprColumn && getf(lhs) === effect || continue
        isempty(getkwargs(lhs)) || continue
        target = getargs(lhs)
        target == (key, :Intercept) || continue
        push!(matches, (operation_key, prior))
    end
    isempty(matches) && return (
        location=0.0, scale=1.0, operation=nothing)
    length(matches) == 1 || throw(NativePPLCapabilityError(
        :coefficient_prior,
        "linear predictor `$key` has multiple intercept prior declarations"))
    operation_key, prior = only(matches)
    prior isa ExprColumn && getf(prior) === Normal || throw(
        NativePPLCapabilityError(
            :coefficient_prior,
            "`effect($key, Intercept)` must use `Normal(location, scale)`"))
    isempty(getkwargs(prior)) || throw(NativePPLCapabilityError(
        :coefficient_prior,
        "Normal intercept prior for `$key` cannot have keywords"))
    arguments = getargs(prior)
    length(arguments) == 2 && all(argument -> argument isa Real, arguments) ||
        throw(NativePPLCapabilityError(
            :coefficient_prior,
            "Normal intercept prior for `$key` needs numeric location and scale"))
    location, scale = arguments
    isfinite(location) || throw(NativePPLCapabilityError(
        :coefficient_prior,
        "Normal intercept prior location for `$key` must be finite"))
    isfinite(scale) && scale > zero(scale) || throw(
        NativePPLCapabilityError(
            :coefficient_prior,
            "Normal intercept prior scale for `$key` must be finite and positive"))
    (; location, scale, operation=operation_key)
end

function _native_ppl_fit_mean(values::AbstractVector, name::Symbol,
                              transform::Symbol)
    _brm_fit_mean_numeric(values, name, transform,
        message -> NativePPLCapabilityError(:predictor_transform, message))
end

_native_ppl_fit_center(values::AbstractVector, name::Symbol) =
    _native_ppl_fit_mean(values, name, :center)

function _native_ppl_fit_zscale(values::AbstractVector, name::Symbol)
    _brm_fit_zscale_numeric(values, name,
        message -> NativePPLCapabilityError(:predictor_transform, message))
end

function _native_ppl_validate_coefficient_prior(
    factor::NativePPLStandardNormalFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
) where {Parameter}
    factor.unconstrained == parameter.unconstrained || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled coefficient prior and parameter coordinates must agree"))
    nothing
end
function _native_ppl_validate_coefficient_prior(
    factor::NativePPLScalarNormalFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
) where {Parameter}
    factor.unconstrained_index == only(parameter.unconstrained) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled scalar-site factor and parameter coordinate must agree"))
    factor.location isa Real && isfinite(factor.location) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled scalar-site Normal location must be finite"))
    factor.scale isa Real && isfinite(factor.scale) && factor.scale > zero(factor.scale) ||
        throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled scalar-site Normal scale must be finite and positive"))
    nothing
end
_native_ppl_validate_coefficient_prior(::Any, ::Any) = throw(
    NativePPLCapabilityError(
        :graph_identity,
        "compiled coefficient prior must target the coefficient parameter"))

function _native_ppl_validate_scale_prior(
    factor::NativePPLExponentialFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
) where {Parameter}
    factor.unconstrained_index in parameter.unconstrained || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled scale prior coordinate must belong to the scale parameter"))
    nothing
end
_native_ppl_validate_scale_prior(::Any, ::Any) = throw(
    NativePPLCapabilityError(
        :graph_identity,
        "compiled scale prior must target the scale parameter"))

@inline _native_ppl_location_parameter(
    plan::NativePPLPlan,
) = _native_ppl_location_parameter(plan.nodes.location, plan.parameters)
@inline function _native_ppl_location_parameter(
    ::NativePPLAffineNode, parameters,
)
    hasproperty(parameters, :coefficients) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled affine graph is missing its coefficient parameter"))
    parameters.coefficients
end
@inline function _native_ppl_location_parameter(
    ::NativePPLScalarBroadcastNode{Location,Parameter}, parameters,
) where {Location,Parameter}
    hasproperty(parameters, :site) && return parameters.site
    hasproperty(parameters, Parameter) && return getproperty(
        parameters, Parameter)
    hasproperty(parameters, :coefficients) && return parameters.coefficients
    throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled scalar graph is missing parameter `$Parameter`"))
end

@inline function _native_ppl_location_factor(plan::NativePPLPlan)
    hasproperty(plan.factors, :site_prior) && return plan.factors.site_prior
    hasproperty(plan.factors, :coefficient_prior) &&
        return plan.factors.coefficient_prior
    throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled graph is missing its location-generating factor"))
end

function _native_ppl_validate_likelihood_graph(
    factor::NativePPLNormalFactor{Response,Location,Scale},
    plan::NativePPLPlan, observation_axis,
) where {Response,Location,Scale}
    factor.axis === observation_axis || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled Normal factor must carry the plan observation axis"))
    Response === native_input_name(plan.inputs.response) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled Normal factor must observe the response input"))
    Location === native_node_name(plan.nodes.location) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled Normal factor must consume the affine location"))
    hasproperty(plan.parameters, :scale) || throw(NativePPLCapabilityError(
        :graph_identity, "compiled Normal graph is missing its scale parameter"))
    scale = plan.parameters.scale
    scale isa NativePPLParameter || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled Normal scale must be a typed parameter block"))
    hasproperty(plan.axes, :scale) && scale.axis === plan.axes.scale || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled Normal scale parameter must carry the scale axis"))
    Scale === native_parameter_name(scale) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled Normal factor must consume the scale parameter"))
    hasproperty(plan.factors, :scale_prior) || throw(NativePPLCapabilityError(
        :graph_identity, "compiled Normal graph is missing its scale prior"))
    _native_ppl_validate_scale_prior(
        plan.factors.scale_prior, scale)
    hasproperty(plan.nodes, :rate) && throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled Normal graph cannot contain a Poisson rate node"))
    nothing
end

function _native_ppl_validate_likelihood_graph(
    factor::NativePPLBernoulliLogitFactor{Response,Location},
    plan::NativePPLPlan, observation_axis,
) where {Response,Location}
    factor.axis === observation_axis || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled BernoulliLogit factor must carry the plan observation axis"))
    Response === native_input_name(plan.inputs.response) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled BernoulliLogit factor must observe the response input"))
    Location === native_node_name(plan.nodes.location) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled BernoulliLogit factor must consume the affine location"))
    hasproperty(plan.nodes, :rate) && throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled BernoulliLogit graph cannot contain a Poisson rate node"))
    nothing
end

function _native_ppl_validate_likelihood_graph(
    factor::NativePPLPoissonFactor{Response,Rate},
    plan::NativePPLPlan, observation_axis,
) where {Response,Rate}
    factor.axis === observation_axis || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled Poisson factor must carry the plan observation axis"))
    Response === native_input_name(plan.inputs.response) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled Poisson factor must observe the response input"))
    hasproperty(plan.nodes, :rate) || throw(NativePPLCapabilityError(
        :graph_identity, "compiled Poisson graph is missing its rate node"))
    Rate === native_node_name(plan.nodes.rate) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled Poisson factor must consume the exponential rate"))
    nothing
end

_native_ppl_validate_likelihood_graph(
    ::Any, ::NativePPLPlan, ::Any,
) = throw(NativePPLCapabilityError(
    :graph_identity, "compiled graph has an unsupported likelihood factor"))

function _native_ppl_validate_predictor_graph(plan::NativePPLPlan)
    observation_axis = plan.axes.observation
    predictors = plan.inputs.predictors
    response = plan.inputs.response
    native_input_role(response) === :response || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled response input must preserve its response role"))
    predictor_names = Tuple(keys(predictors))
    length(unique(predictor_names)) == length(predictor_names) || throw(
        NativePPLCapabilityError(
            :graph_identity, "compiled predictor identities must be unique"))
    for (name, predictor) in pairs(predictors)
        native_input_name(predictor) === name || throw(
            NativePPLCapabilityError(
                :graph_identity,
                "compiled predictor key `$name` must match its semantic identity"))
        native_input_role(predictor) === :predictor || throw(
            NativePPLCapabilityError(
                :graph_identity,
                "compiled predictor `$name` must preserve its predictor role"))
        native_input_name(predictor) !== native_input_name(response) || throw(
            NativePPLCapabilityError(
                :graph_identity,
                "compiled predictor and response must have distinct identities"))
        predictor.axis === observation_axis || throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled predictor `$name` must carry the plan observation axis"))
    end
    response.axis === observation_axis || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled response input must carry the plan observation axis"))

    hasproperty(plan.nodes, :location) || throw(NativePPLCapabilityError(
        :graph_identity, "compiled graph is missing its location node"))
    location = plan.nodes.location
    location isa NativePPLLocationNode || throw(NativePPLCapabilityError(
        :graph_identity, "compiled location must be a typed location node"))
    location.axis === observation_axis || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled location must carry the plan observation axis"))
    coefficients = _native_ppl_location_parameter(plan)
    coefficients isa NativePPLParameter || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled location source must be a typed parameter block"))
    coefficients.axis === plan.axes.coefficient || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "compiled location parameter must carry the coefficient axis"))
    _native_ppl_validate_coefficient_prior(
        _native_ppl_location_factor(plan), coefficients)

    hasproperty(plan.nodes, :transforms) || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled graph must carry its fitted transforms NamedTuple"))
    transforms = plan.nodes.transforms
    if location isa NativePPLAffineNode
        isempty(predictors) && throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled affine graph must contain at least one predictor input"))
        location.intercept_index == first(coefficients.unconstrained) &&
            location.slope_indices == Tuple(Iterators.drop(
                coefficients.unconstrained, 1)) || throw(
                NativePPLCapabilityError(
                    :graph_identity,
                    "compiled affine coordinates must match the coefficient parameter"))
        transformed_raw_inputs = Symbol[]
        expected_location_inputs = Symbol[]
        for (name, transform) in pairs(transforms)
            transform isa Union{NativePPLCenterNode,NativePPLZScaleNode} ||
                throw(NativePPLCapabilityError(
                    :graph_identity,
                    "compiled predictor transform must be center or zscale"))
            transform.axis === observation_axis || throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled predictor transform must carry the plan observation axis"))
            raw_input = native_fitted_transform_input(transform)
            hasproperty(predictors, raw_input) ||
                throw(NativePPLCapabilityError(
                    :graph_identity,
                    "compiled predictor transform `$name` must consume a raw predictor"))
            name === native_node_name(transform) || throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled predictor transform key `$name` must match its identity"))
            raw_input in transformed_raw_inputs && throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled predictor `$raw_input` cannot feed multiple fitted transforms"))
            push!(transformed_raw_inputs, raw_input)
            name !== native_node_name(location) ||
                throw(NativePPLCapabilityError(
                    :graph_identity,
                    "compiled predictor transform and affine location must have distinct identities"))
        end
        for predictor_name in predictor_names
            matching = [name for (name, transform) in pairs(transforms)
                        if native_fitted_transform_input(transform) === predictor_name]
            push!(expected_location_inputs,
                  isempty(matching) ? predictor_name : only(matching))
        end
        location_inputs = native_affine_inputs(location)
        length(unique(location_inputs)) == length(location_inputs) || throw(
            NativePPLCapabilityError(
                :graph_identity,
                "compiled affine location inputs must be unique"))
        Set(location_inputs) == Set(expected_location_inputs) ||
            throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled affine location must consume every compiled predictor path"))
    else
        isempty(predictors) || throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled scalar-broadcast location cannot carry predictor inputs"))
        isempty(transforms) || throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled scalar-broadcast location cannot carry fitted transforms"))
        native_scalar_parameter(location) ===
            native_parameter_name(coefficients) || throw(
                NativePPLCapabilityError(
                    :graph_identity,
                    "compiled scalar location must consume its parameter block"))
        location.unconstrained_index == only(coefficients.unconstrained) ||
            throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled scalar location coordinate must match its parameter"))
    end

    if hasproperty(plan.nodes, :rate)
        rate = plan.nodes.rate
        rate isa NativePPLExpNode || throw(NativePPLCapabilityError(
            :graph_identity, "compiled rate must be a typed exponential node"))
        rate.axis === observation_axis || throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled exponential rate must carry the plan observation axis"))
        native_exp_input(rate) === native_node_name(location) ||
            throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled exponential rate must consume the affine location"))
        native_node_name(rate) !== native_node_name(location) ||
            throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled exponential rate and affine location must have distinct identities"))
    end
    hasproperty(plan.factors, :likelihood) || throw(NativePPLCapabilityError(
        :graph_identity, "compiled graph is missing its likelihood factor"))
    _native_ppl_validate_likelihood_graph(
        plan.factors.likelihood, plan, observation_axis)

    expected_queries = (
        :linear_predictor, :pointwise_loglikelihood, :posterior_predictive)
    keys(plan.queries) == expected_queries || throw(NativePPLCapabilityError(
        :graph_identity,
        "compiled graph must expose the canonical typed query set"))
    for (name, query) in pairs(plan.queries)
        query isa NativePPLQuerySpec || throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled query `$name` must be a typed query specification"))
        native_query_name(query) === name || throw(NativePPLCapabilityError(
            :graph_identity,
            "compiled query key `$name` must match its typed identity"))
        output = native_query_output(query)
        output isa NativePPLOutputSignature || throw(
            NativePPLCapabilityError(
                :graph_identity,
                "compiled query `$name` must have a vector output signature"))
        native_output_axis(output) === observation_axis ||
            throw(NativePPLCapabilityError(
                :graph_identity,
                "compiled query outputs must carry the plan observation axis"))
    end
    nothing
end

"""Compatibility wrapper over the sole public BRM-to-Plan compiler path."""
_native_ppl_plan(brmi::BRMI) = NativePPL.compile(brmi)

function _native_ppl_copy_input(::Type{T}, input, role::Symbol, name::Symbol) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(input))
    for (i, value) in enumerate(input)
        value isa Real && isfinite(value) || throw(ArgumentError(
            "native PPL $role `$name` contains a non-finite or non-real value at row $i"))
        output[i] = value
        isfinite(output[i]) || throw(ArgumentError(
            "native PPL $role `$name` cannot be represented as $T at row $i"))
    end
    output
end

function _native_ppl_prepare_response(::Type{T}, response::AbstractVector,
                                      name::Symbol, observations::Int) where {T<:AbstractFloat}
    output = _native_ppl_copy_input(T, response, :response, name)
    length(output) == observations || throw(DimensionMismatch(
        "native PPL response `$name` has $(length(output)) rows but predictor has " *
        "$observations"))
    output
end

_native_ppl_prepare_response(::Type{<:AbstractFloat},
                             response::NativePPLNoResponse,
                             ::Symbol, ::Int) = response

_native_ppl_validate_response(::NativePPLNormalFactor, ::AbstractVector, ::Symbol) =
    nothing

function _native_ppl_is_count(value)
    value isa Real && isfinite(value) && value >= zero(value) &&
        isinteger(value) || return false
    try
        Int(value)
        true
    catch
        false
    end
end

function _native_ppl_validate_response(::NativePPLBernoulliLogitFactor,
                                       response::AbstractVector, name::Symbol)
    for (i, value) in enumerate(response)
        value == 0 || value == 1 || throw(ArgumentError(
            "native PPL BernoulliLogit response `$name` must be Bool/0/1; " *
            "got $value at row $i"))
    end
    nothing
end
function _native_ppl_validate_response(::NativePPLPoissonFactor,
                                       response::AbstractVector, name::Symbol)
    for (i, value) in enumerate(response)
        _native_ppl_is_count(value) || throw(ArgumentError(
            "native PPL Poisson response `$name` must be a nonnegative " *
            "integer-valued count representable as Int; got $value at row $i"))
    end
    nothing
end
_native_ppl_validate_response(::NativePPLFactor, ::NativePPLNoResponse, ::Symbol) =
    nothing

function _native_ppl_apply_predictor!(
    node::NativePPLCenterNode{Name,Input},
    ::NativePPLInput{Input},
    predictor::Vector{T},
) where {Name,Input,T}
    mean = T(node.mean)
    isfinite(mean) || throw(ArgumentError(
        "native PPL fitted center mean for `$Input` cannot be represented as $T"))
    for i in eachindex(predictor)
        predictor[i] -= mean
        isfinite(predictor[i]) || throw(ArgumentError(
            "native PPL centered predictor `$Input` is non-finite at row $i"))
    end
    predictor
end

function _native_ppl_apply_predictor!(
    node::NativePPLZScaleNode{Name,Input},
    ::NativePPLInput{Input},
    predictor::Vector{T},
) where {Name,Input,T}
    mean = T(node.mean)
    scale = T(node.scale)
    isfinite(mean) || throw(ArgumentError(
        "native PPL fitted zscale mean for `$Input` cannot be represented as $T"))
    isfinite(scale) && scale > zero(T) || throw(ArgumentError(
        "native PPL fitted zscale sample SD for `$Input` cannot be represented " *
        "as a finite positive $T"))
    for i in eachindex(predictor)
        raw = predictor[i]
        standardized = (raw - mean) / scale
        if !isfinite(standardized)
            # The separated form can represent a wide-span rebound value when
            # `raw - mean` overflows even though the standardized result does not.
            standardized = raw / scale - mean / scale
        end
        isfinite(standardized) || throw(ArgumentError(
            "native PPL standardized predictor `$Input` is non-finite at row $i"))
        predictor[i] = standardized
    end
    predictor
end

function _native_ppl_feature_source(plan::NativePPLPlan, feature_name::Symbol)
    if hasproperty(plan.inputs.predictors, feature_name)
        return (; raw_name=feature_name, transform=nothing)
    end
    hasproperty(plan.nodes.transforms, feature_name) || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "affine feature `$feature_name` is neither a raw predictor nor a " *
            "fitted transform"))
    transform = getproperty(plan.nodes.transforms, feature_name)
    (; raw_name=native_fitted_transform_input(transform), transform)
end

function _native_ppl_prepare_predictors(plan::NativePPLPlan, bindings,
                                        ::Type{T}) where {T<:AbstractFloat}
    feature_names = native_affine_inputs(plan.nodes.location)
    prepared = map(feature_names) do feature_name
        source = _native_ppl_feature_source(plan, feature_name)
        raw_input = getproperty(plan.inputs.predictors, source.raw_name)
        raw = getproperty(bindings, source.raw_name)
        predictor = _native_ppl_copy_input(
            T, raw, :predictor, source.raw_name)
        length(predictor) == length(plan.axes.observation) ||
            throw(DimensionMismatch(
                "native PPL predictor `$(source.raw_name)` has " *
                "$(length(predictor)) rows but the observation axis has " *
                "$(length(plan.axes.observation))"))
        source.transform === nothing ||
            _native_ppl_apply_predictor!(
                source.transform, raw_input, predictor)
        predictor
    end
    NamedTuple{feature_names}(prepared)
end

_native_ppl_validate_response_conversion(
    ::NativePPLFactor, ::Any, ::Any, ::Symbol,
) = nothing
function _native_ppl_validate_response_conversion(
    ::NativePPLPoissonFactor,
    response::AbstractVector,
    prepared_response::AbstractVector,
    name::Symbol,
)
    for i in eachindex(response, prepared_response)
        prepared_response[i] == response[i] || throw(ArgumentError(
            "native PPL Poisson response `$name` count $(response[i]) at row $i " *
            "cannot be represented exactly as $(eltype(prepared_response))"))
    end
    nothing
end

function _native_ppl_prepare_bindings(plan::NativePPLPlan, predictors,
                                      response, ::Type{T}) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError(
        "native PPL prepared element type must be concrete; got $T"))
    _native_ppl_validate_predictor_graph(plan)
    response_name = native_input_name(plan.inputs.response)
    prepared_predictors = _native_ppl_prepare_predictors(plan, predictors, T)
    _native_ppl_validate_response(
        plan.factors.likelihood, response, response_name)
    prepared_response = _native_ppl_prepare_response(
        T, response, response_name, length(plan.axes.observation))
    _native_ppl_validate_response_conversion(
        plan.factors.likelihood, response, prepared_response, response_name)
    workspace_spec = NativePPLWorkspaceSpec(
        plan.axes.observation, LogDensityProblems.dimension(plan))
    NativePPLPrepared(
        plan, prepared_predictors, prepared_response, workspace_spec)
end

"""
    _native_ppl_prepare(plan; T=Float64) -> NativePPLPrepared

Validate and copy the plan's initial input binding into numeric, executor-owned
storage. The returned value is shareable; mutable evaluation state is supplied
separately by `NativePPLWorkspace`.
"""
function _native_ppl_prepare(plan::NativePPLPlan; T::Type{<:AbstractFloat}=Float64)
    isconcretetype(T) || throw(ArgumentError(
        "native PPL prepared element type must be concrete; got $T"))
    response_name = native_input_name(plan.inputs.response)
    predictor_names = Tuple(keys(plan.inputs.predictors))
    predictors = NamedTuple{predictor_names}(map(
        name -> getproperty(plan.bindings, name), predictor_names))
    response = _native_ppl_response_binding(plan.bindings, response_name)
    _native_ppl_prepare_bindings(plan, predictors, response, T)
end


function _native_ppl_require_response(prepared::NativePPLPrepared,
                                      operation::AbstractString)
    native_ppl_has_response(prepared) || throw(ArgumentError(
        "native PPL $operation requires an observed response binding; " *
        "this prepared value is prediction-only, so rebind it with both " *
        "predictor and response inputs"))
    prepared.response
end

function _native_ppl_check_location_execution(workspace::NativePPLWorkspace,
                                              prepared::NativePPLPrepared,
                                              position::AbstractVector)
    dimension = LogDensityProblems.dimension(prepared)
    observations = length(prepared.plan.axes.observation)
    length(position) == dimension || throw(DimensionMismatch(
        "native PPL position has length $(length(position)); expected $dimension"))
    for (name, predictor) in pairs(prepared.predictors)
        length(predictor) == observations || throw(DimensionMismatch(
            "native PPL prepared predictor `$name` has $(length(predictor)) " *
            "rows; expected $observations"))
    end
    length(workspace.primal.location) == observations || throw(DimensionMismatch(
        "native PPL workspace location has $(length(workspace.primal.location)) rows; expected $observations"))
    eltype(position) === eltype(workspace) || throw(ArgumentError(
        "native PPL position eltype $(eltype(position)) does not match workspace eltype $(eltype(workspace))"))
    eltype(prepared) === eltype(workspace) || throw(ArgumentError(
        "native PPL prepared eltype $(eltype(prepared)) does not match workspace eltype $(eltype(workspace))"))
    Base.mightalias(position, workspace.primal.location) && throw(ArgumentError(
        "native PPL position must not alias the workspace location buffer"))
    nothing
end

function _native_ppl_check_execution(workspace::NativePPLWorkspace,
                                     prepared::NativePPLPrepared,
                                     position::AbstractVector)
    response = _native_ppl_require_response(prepared, "log density")
    _native_ppl_check_location_execution(workspace, prepared, position)
    dimension = LogDensityProblems.dimension(prepared)
    observations = length(prepared.plan.axes.observation)
    length(workspace.gradient) == dimension || throw(DimensionMismatch(
        "native PPL workspace gradient has length $(length(workspace.gradient)); expected $dimension"))
    length(response) == observations || throw(DimensionMismatch(
        "native PPL prepared response has $(length(response)) rows; expected $observations"))
    length(workspace.primal.pointwise_loglikelihood) == observations ||
        throw(DimensionMismatch(
            "native PPL workspace pointwise likelihood has " *
            "$(length(workspace.primal.pointwise_loglikelihood)) rows; expected $observations"))
    Base.mightalias(position, workspace.gradient) && throw(ArgumentError(
        "native PPL position must not alias the workspace gradient"))
    Base.mightalias(position, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL position must not alias the workspace pointwise likelihood buffer"))
    nothing
end

const _NATIVE_PPL_LOG2PI = log(2π)
const _NATIVE_PPL_HALF_LOG2PI = _NATIVE_PPL_LOG2PI / 2

# Distributions may supply formula-boundary vocabulary, but numerical execution
# is PPL-owned. Keep these vectorized factor/transform kernels independent of
# Distributions and Bijectors so later compiler passes (including a possible
# Reactant lowering) see the complete executable semantics here.

@inline function _native_ppl_factor_logdensity(
    factor::NativePPLStandardNormalFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
    position::AbstractVector{T},
) where {Parameter,T}
    half = T(0.5)
    normalizer = T(_NATIVE_PPL_HALF_LOG2PI)
    density = zero(T)
    for coordinate in factor.unconstrained
        value = native_parameter_value(parameter, position, coordinate)
        logjac = native_parameter_logabsdetjac(parameter, position, coordinate)
        density += -half * value * value - normalizer + logjac
    end
    density
end

@inline function _native_ppl_factor_logdensity(
    factor::NativePPLScalarNormalFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
    position::AbstractVector{T},
) where {Parameter,T}
    coordinate = factor.unconstrained_index
    value = native_parameter_value(parameter, position, coordinate)
    logjac = native_parameter_logabsdetjac(parameter, position, coordinate)
    location = T(factor.location)
    scale = T(factor.scale)
    standardized = (value - location) / scale
    -T(0.5) * standardized * standardized - log(scale) -
        T(_NATIVE_PPL_HALF_LOG2PI) + logjac
end

@inline function _native_ppl_factor_logdensity(
    factor::NativePPLExponentialFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
    position::AbstractVector{T},
) where {Parameter,T}
    coordinate = factor.unconstrained_index
    value = native_parameter_value(parameter, position, coordinate)
    logjac = native_parameter_logabsdetjac(parameter, position, coordinate)
    prior_scale = T(factor.scale)
    -log(prior_scale) - value / prior_scale + logjac
end

@inline function _native_ppl_factor_logdensity!(
    ::NativePPLNormalFactor{Response,Location,Scale},
    ::NativePPLInput{Response},
    ::NativePPLLocationNode{Location},
    scale_parameter::NativePPLParameter{Scale},
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {Response,Location,Scale,T}
    scale = native_parameter_value(scale_parameter, position)
    log_scale = native_parameter_logvalue(scale_parameter, position)
    inverse_scale = inv(scale)
    half = T(0.5)
    normalizer = T(_NATIVE_PPL_HALF_LOG2PI)
    density = zero(T)
    for i in eachindex(prepared.response)
        difference = prepared.response[i] - buffers.location[i]
        residual = iszero(difference) ? zero(T) : difference * inverse_scale
        pointwise = -log_scale - normalizer - half * residual * residual
        buffers.pointwise_loglikelihood[i] = pointwise
        density += pointwise
    end
    density
end

@inline _native_ppl_softplus(value) = value > zero(value) ?
    value + log1p(exp(-value)) : log1p(exp(value))

@inline function _native_ppl_factor_logdensity!(
    ::NativePPLBernoulliLogitFactor{Response,Logit},
    ::NativePPLInput{Response},
    ::NativePPLLocationNode{Logit},
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {Response,Logit,T}
    density = zero(T)
    for i in eachindex(prepared.response)
        logit = buffers.location[i]
        pointwise = isone(prepared.response[i]) ?
            -_native_ppl_softplus(-logit) : -_native_ppl_softplus(logit)
        buffers.pointwise_loglikelihood[i] = pointwise
        density += pointwise
    end
    density
end

@inline function _native_ppl_stirling_correction(value::T) where {T}
    inverse = inv(value)
    inverse2 = inverse * inverse
    inverse * (
        T(1 / 12) + inverse2 * (
            T(-1 / 360) + inverse2 * (
                T(1 / 1260) + inverse2 * (
                    T(-1 / 1680) + inverse2 * T(1 / 1188)))))
end

@inline function _native_ppl_logfactorial(::Type{T}, count::Int) where {T}
    count < 2 && return zero(T)
    if count < 32
        value = zero(T)
        for k in 2:count
            value += log(T(k))
        end
        return value
    end

    value = T(count)
    correction = _native_ppl_stirling_correction(value)
    (value + T(0.5)) * log(value) - value +
        T(_NATIVE_PPL_HALF_LOG2PI) + correction
end

@inline _native_ppl_logfactorial(value::T) where {T<:AbstractFloat} =
    _native_ppl_logfactorial(T, Int(value))

@inline function _native_ppl_delta_minus_expm1(value::T) where {T}
    abs(value) > T(0.01) && return value - expm1(value)
    series = T(1 / 479001600)
    series = T(1 / 39916800) + value * series
    series = T(1 / 3628800) + value * series
    series = T(1 / 362880) + value * series
    series = T(1 / 40320) + value * series
    series = T(1 / 5040) + value * series
    series = T(1 / 720) + value * series
    series = T(1 / 120) + value * series
    series = T(1 / 24) + value * series
    series = T(1 / 6) + value * series
    series = T(1 / 2) + value * series
    -(value * value) * series
end

@inline function _native_ppl_poisson_logdensity(count::T, log_rate::T) where {T}
    iszero(count) && return -exp(log_rate)
    count_int = Int(count)
    count_int < 32 && return count * log_rate - exp(log_rate) -
        _native_ppl_logfactorial(T, count_int)

    delta = log_rate - log(count)
    count * _native_ppl_delta_minus_expm1(delta) -
        T(0.5) * log(count) - T(_NATIVE_PPL_HALF_LOG2PI) -
        _native_ppl_stirling_correction(count)
end

@inline function _native_ppl_factor_logdensity!(
    ::NativePPLPoissonFactor{Response,Rate},
    ::NativePPLInput{Response},
    ::NativePPLExpNode{Rate,LogRate},
    ::NativePPLLocationNode{LogRate},
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {Response,Rate,LogRate,T}
    density = zero(T)
    for i in eachindex(prepared.response)
        count = prepared.response[i]
        log_rate = buffers.location[i]
        pointwise = _native_ppl_poisson_logdensity(count, log_rate)
        buffers.pointwise_loglikelihood[i] = pointwise
        density += pointwise
    end
    density
end


@inline function _native_ppl_logistic(value)
    if value >= zero(value)
        inverse = exp(-value)
        inv(one(value) + inverse)
    else
        forward = exp(value)
        forward / (one(value) + forward)
    end
end

@inline function _native_ppl_rand_poisson(
    rng::AbstractRNG, ::Type{T}, log_rate::T,
) where {T<:AbstractFloat}
    isnan(log_rate) && throw(DomainError(
        log_rate, "native PPL Poisson log-rate must not be NaN"))
    log_rate == -T(Inf) && return 0
    rate = exp(log_rate)
    max_rate = min(T(typemax(Int)), maxintfloat(T)) / T(4)
    isfinite(rate) && rate <= max_rate || throw(DomainError(
        log_rate,
        "native PPL Poisson rate is too large for an Int predictive output"))
    iszero(rate) && return 0

    if rate < T(10)
        threshold = exp(-rate)
        count = 0
        product = rand(rng, T)
        while product > threshold
            count += 1
            product *= rand(rng, T)
        end
        return count
    end

    # PTRS transformed rejection. This branch avoids the underflow and linear
    # work of product inversion for moderate and large rates.
    root_rate = sqrt(rate)
    b = T(0.931) + T(2.53) * root_rate
    a = T(-0.059) + T(0.02483) * b
    inverse_alpha = T(1.1239) + T(1.1328) / (b - T(3.4))
    squeeze = T(0.9277) - T(3.6224) / (b - T(2))
    while true
        u = rand(rng, T) - T(0.5)
        v = rand(rng, T)
        us = T(0.5) - abs(u)
        iszero(us) && continue
        candidate_value = floor((T(2) * a / us + b) * u + rate + T(0.43))
        candidate_value < zero(T) && continue
        candidate_value <= T(typemax(Int)) || throw(DomainError(
            candidate_value,
            "native PPL Poisson draw is too large for an Int predictive output"))
        count = Int(candidate_value)
        us >= T(0.07) && v <= squeeze && return count
        (us < T(0.013) && v > us) && continue
        acceptance = log(
            v * inverse_alpha / (a / (us * us) + b))
        target = _native_ppl_poisson_logdensity(T(count), log_rate)
        acceptance <= target && return count
    end
end


@inline function _native_ppl_factor_simulate!(
    rng::AbstractRNG,
    ::NativePPLNormalFactor{Response,Location,Scale},
    ::NativePPLInput{Response},
    ::NativePPLLocationNode{Location},
    output::AbstractVector,
    scale_parameter::NativePPLParameter{Scale},
    position::AbstractVector,
    buffers::NativePPLBuffers{T},
) where {Response,Location,Scale,T}
    scale = native_parameter_value(scale_parameter, position)
    for i in eachindex(output)
        output[i] = buffers.location[i] + scale * randn(rng, T)
    end
    output
end

@inline function _native_ppl_factor_simulate!(
    rng::AbstractRNG,
    ::NativePPLPoissonFactor{Response,Rate},
    ::NativePPLInput{Response},
    ::NativePPLExpNode{Rate,LogRate},
    ::NativePPLLocationNode{LogRate},
    output::AbstractVector{Int},
    position::AbstractVector,
    buffers::NativePPLBuffers{T},
) where {Response,Rate,LogRate,T}
    for i in eachindex(output)
        output[i] = _native_ppl_rand_poisson(rng, T, buffers.location[i])
    end
    output
end

@inline function _native_ppl_factor_simulate!(
    rng::AbstractRNG,
    ::NativePPLBernoulliLogitFactor{Response,Logit},
    ::NativePPLInput{Response},
    ::NativePPLLocationNode{Logit},
    output::AbstractVector{Bool},
    position::AbstractVector,
    buffers::NativePPLBuffers{T},
) where {Response,Logit,T}
    for i in eachindex(output)
        output[i] = rand(rng, T) < _native_ppl_logistic(buffers.location[i])
    end
    output
end


@inline function _native_ppl_model_simulate!(
    rng::AbstractRNG,
    factor::NativePPLNormalFactor,
    output::AbstractVector,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    buffers::NativePPLBuffers,
)
    _native_ppl_factor_simulate!(
        rng, factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, output, prepared.plan.parameters.scale,
        position, buffers)
end

@inline function _native_ppl_model_simulate!(
    rng::AbstractRNG,
    factor::NativePPLBernoulliLogitFactor,
    output::AbstractVector{Bool},
    prepared::NativePPLPrepared,
    position::AbstractVector,
    buffers::NativePPLBuffers,
)
    _native_ppl_factor_simulate!(
        rng, factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, output, position, buffers)
end

@inline function _native_ppl_model_simulate!(
    rng::AbstractRNG,
    factor::NativePPLPoissonFactor,
    output::AbstractVector{Int},
    prepared::NativePPLPrepared,
    position::AbstractVector,
    buffers::NativePPLBuffers,
)
    _native_ppl_factor_simulate!(
        rng, factor, prepared.plan.inputs.response,
        prepared.plan.nodes.rate, prepared.plan.nodes.location,
        output, position, buffers)
end

@inline function _native_ppl_location_kernel!(position::AbstractVector{T},
                                              prepared::NativePPLPrepared,
                                              buffers::NativePPLBuffers{T}) where {T}
    node = prepared.plan.nodes.location
    _native_ppl_location_kernel!(node, position, prepared, buffers)
end

@inline function _native_ppl_location_kernel!(
    node::NativePPLAffineNode,
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {T}
    coefficient_parameter = _native_ppl_location_parameter(prepared.plan)
    intercept = native_parameter_value(
        coefficient_parameter, position, node.intercept_index)
    predictors = values(prepared.predictors)
    for i in eachindex(first(predictors))
        location = intercept
        for j in eachindex(node.slope_indices)
            slope = native_parameter_value(
                coefficient_parameter, position, node.slope_indices[j])
            location += slope * predictors[j][i]
        end
        buffers.location[i] = location
    end
    nothing
end


@inline function _native_ppl_location_kernel!(
    node::NativePPLScalarBroadcastNode{Location,Parameter},
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {Location,Parameter,T}
    parameter = _native_ppl_location_parameter(prepared.plan)
    parameter isa NativePPLParameter{Parameter} || throw(
        NativePPLCapabilityError(
            :graph_identity,
            "scalar location `$Location` must consume parameter `$Parameter`"))
    value = native_parameter_value(
        parameter, position, node.unconstrained_index)
    fill!(buffers.location, value)
    nothing
end

@inline function _native_ppl_model_logdensity!(
    likelihood_factor::NativePPLNormalFactor,
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {T}
    scale_factor = prepared.plan.factors.scale_prior
    coefficient_factor = _native_ppl_location_factor(prepared.plan)
    coefficient_parameter = _native_ppl_location_parameter(prepared.plan)
    scale_parameter = prepared.plan.parameters.scale
    density = _native_ppl_factor_logdensity(
        coefficient_factor, coefficient_parameter, position)
    density += _native_ppl_factor_logdensity(
        scale_factor, scale_parameter, position)
    density += _native_ppl_factor_logdensity!(
        likelihood_factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, scale_parameter, position, prepared,
        buffers)
    density
end


@inline function _native_ppl_model_logdensity!(
    likelihood_factor::NativePPLBernoulliLogitFactor,
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {T}
    density = _native_ppl_factor_logdensity(
        _native_ppl_location_factor(prepared.plan),
        _native_ppl_location_parameter(prepared.plan),
        position)
    density += _native_ppl_factor_logdensity!(
        likelihood_factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, position, prepared, buffers)
    density
end

@inline function _native_ppl_model_logdensity!(
    likelihood_factor::NativePPLPoissonFactor,
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {T}
    density = _native_ppl_factor_logdensity(
        _native_ppl_location_factor(prepared.plan),
        _native_ppl_location_parameter(prepared.plan),
        position)
    density += _native_ppl_factor_logdensity!(
        likelihood_factor, prepared.plan.inputs.response,
        prepared.plan.nodes.rate, prepared.plan.nodes.location,
        position, prepared, buffers)
    density
end


@inline function _native_ppl_logdensity_kernel(position::AbstractVector{T},
                                               prepared::NativePPLPrepared,
                                               buffers::NativePPLBuffers{T}) where {T}
    _native_ppl_location_kernel!(position, prepared, buffers)
    _native_ppl_model_logdensity!(
        prepared.plan.factors.likelihood, position, prepared, buffers)
end

function _native_ppl_location!(workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector)
    _native_ppl_check_location_execution(workspace, prepared, position)
    _native_ppl_location_kernel!(position, prepared, workspace.primal)
    workspace.primal.location
end

"""
    _native_ppl_logdensity!(workspace, prepared, position)

Evaluate in the caller-owned reusable workspace. Derived locations and
pointwise likelihoods are refreshed as part of the same graph execution.
"""
function _native_ppl_logdensity!(workspace::NativePPLWorkspace,
                                 prepared::NativePPLPrepared,
                                 position::AbstractVector)
    _native_ppl_check_execution(workspace, prepared, position)
    _native_ppl_logdensity_kernel(position, prepared, workspace.primal)
end

# DifferentiationInterface is optional. Its extension installs the concrete
# typed method and owns derivative preparation. AutoEnzyme is the sole backend
# this package tests, recommends, or guarantees; other DI backends remain
# outside the compatibility contract.
function _native_ppl_logdensity_and_gradient! end
function _native_ppl_logdensity_and_gradient!(workspace::NativePPLWorkspace,
                                              prepared::NativePPLPrepared,
                                              position::AbstractVector)
    _native_ppl_require_response(prepared, "gradient")
    throw(ArgumentError(
        "native PPL gradients require a DifferentiationInterface-prepared workspace; " *
        "construct one with AutoEnzyme() for the supported path"))
end
_native_ppl_logdensity_and_gradient!(args...) = throw(ArgumentError(
    "native PPL gradients require a DifferentiationInterface-prepared workspace; " *
    "construct one with AutoEnzyme() for the supported path"))

function _native_ppl_required_binding(bindings, name::Symbol, role::Symbol)
    hasproperty(bindings, name) || throw(ArgumentError(
        "native PPL rebind is missing required $role input `$name`"))
    value = getproperty(bindings, name)
    value isa AbstractVector || throw(ArgumentError(
        "native PPL $role input `$name` must be an AbstractVector; got $(typeof(value))"))
    value
end


function _native_ppl_response_binding(bindings, name::Symbol)
    hasproperty(bindings, name) || return NativePPLNoResponse()
    _native_ppl_required_binding(bindings, name, :response)
end

_native_ppl_rebind_likelihood(
    ::NativePPLNormalFactor{Response,Location,Scale}, axis,
) where {Response,Location,Scale} =
    NativePPLNormalFactor(Response, Location, Scale, axis)
_native_ppl_rebind_likelihood(
    ::NativePPLBernoulliLogitFactor{Response,Logit}, axis,
) where {Response,Logit} =
    NativePPLBernoulliLogitFactor(Response, Logit, axis)
_native_ppl_rebind_likelihood(
    ::NativePPLPoissonFactor{Response,Rate}, axis,
) where {Response,Rate} =
    NativePPLPoissonFactor(Response, Rate, axis)

function _native_ppl_rebind_transform(
    old_transform::NativePPLCenterNode,
    observation_axis,
    predictor_name::Symbol,
    predictor,
    freeze_constants::Bool,
)
    mean = freeze_constants ? old_transform.mean :
        _native_ppl_fit_center(predictor, predictor_name)
    NativePPLCenterNode(
        native_node_name(old_transform), predictor_name,
        observation_axis, mean)
end

function _native_ppl_rebind_transform(
    old_transform::NativePPLZScaleNode,
    observation_axis,
    predictor_name::Symbol,
    predictor,
    freeze_constants::Bool,
)
    fit = freeze_constants ?
        (; mean=old_transform.mean, scale=old_transform.scale) :
        _native_ppl_fit_zscale(predictor, predictor_name)
    NativePPLZScaleNode(
        native_node_name(old_transform), predictor_name,
        observation_axis, fit.mean, fit.scale)
end

function _native_ppl_rebind_nodes(plan::NativePPLPlan, observation_axis,
                                  predictors::NamedTuple,
                                  freeze_constants::Bool)
    old_location = plan.nodes.location
    location_name = native_node_name(old_location)
    transform_names = Tuple(keys(plan.nodes.transforms))
    transform_values = map(transform_names) do transform_name
        old_transform = getproperty(plan.nodes.transforms, transform_name)
        predictor_name = native_fitted_transform_input(old_transform)
        hasproperty(predictors, predictor_name) || throw(
            NativePPLCapabilityError(
                :graph_identity,
                "rebound fitted transform `$transform_name` is missing raw " *
                "predictor `$predictor_name`"))
        _native_ppl_rebind_transform(
            old_transform, observation_axis, predictor_name,
            getproperty(predictors, predictor_name), freeze_constants)
    end
    transforms = NamedTuple{transform_names}(transform_values)
    location = _native_ppl_rebind_location(old_location, observation_axis)
    nodes = (; transforms, location)
    hasproperty(plan.nodes, :rate) || return nodes

    old_rate = plan.nodes.rate
    native_exp_input(old_rate) === location_name || throw(NativePPLCapabilityError(
        :graph_identity,
        "rebound exponential node must consume the compiled affine predictor"))
    rate = NativePPLExpNode(
        native_node_name(old_rate), location_name, observation_axis)
    merge(nodes, (; rate))
end

function _native_ppl_rebind_location(
    old_location::NativePPLAffineNode, observation_axis)
    NativePPLAffineNode(
        native_node_name(old_location), native_affine_inputs(old_location),
        observation_axis, old_location.intercept_index,
        old_location.slope_indices)
end

function _native_ppl_rebind_location(
    old_location::NativePPLScalarBroadcastNode, observation_axis)
    NativePPLScalarBroadcastNode(
        native_node_name(old_location), native_scalar_parameter(old_location),
        observation_axis, old_location.unconstrained_index)
end

"""
    _native_ppl_rebind(prepared, bindings;
                       T=eltype(prepared), freeze_constants=true)

Rebind the same graph semantics to compatible predictor vectors and an optional
response vector. Omitting the response creates an explicit prediction-only prepared
value. The observation axis and every node/factor/query carrying it are rebuilt
from the new row count; parameter coordinates and semantic identities are reused.
Fitted preprocessing constants are reused by default. Pass
`freeze_constants=false` to refit them from the rebound predictor.
"""
function _native_ppl_rebind(prepared::NativePPLPrepared, bindings;
                            T::Type{<:AbstractFloat}=eltype(prepared),
                            freeze_constants::Bool=true)
    plan = prepared.plan
    _native_ppl_validate_predictor_graph(plan)
    predictor_names = Tuple(keys(plan.inputs.predictors))
    response_name = native_input_name(plan.inputs.response)
    expected_names = Set((predictor_names..., response_name))
    extra_names = setdiff(Set(keys(bindings)), expected_names)
    isempty(extra_names) || throw(ArgumentError(
        "native PPL rebind contains undeclared values: " *
        join(sort!(collect(extra_names)), ", ")))
    predictor_values = map(predictor_names) do predictor_name
        predictor = _native_ppl_required_binding(
            bindings, predictor_name, :predictor)
        eltype(predictor) <: Real && !(eltype(predictor) <: Integer) ||
            throw(ArgumentError(
                "native PPL predictor `$predictor_name` must preserve the " *
                "compiled continuous real input role; got eltype " *
                "$(eltype(predictor))"))
        predictor
    end
    predictors = NamedTuple{predictor_names}(predictor_values)
    response = _native_ppl_response_binding(bindings, response_name)
    observation_count = isempty(predictor_values) ?
        (response isa AbstractVector ? length(response) :
         length(plan.axes.observation)) : length(first(predictor_values))
    for (predictor_name, predictor) in pairs(predictors)
        length(predictor) == observation_count || throw(DimensionMismatch(
            "native PPL predictor `$predictor_name` has $(length(predictor)) " *
            "rows; expected $observation_count"))
    end
    if response isa AbstractVector
        eltype(response) <: Real || throw(ArgumentError(
            "native PPL response `$response_name` must preserve its compiled real input role; " *
            "got eltype $(eltype(response))"))
        observation_count == length(response) || throw(DimensionMismatch(
            "native PPL predictors have $observation_count rows but " *
            "response `$response_name` has $(length(response))"))
    end
    observation_count > 0 || throw(DimensionMismatch(
        "native PPL rebound observation axis cannot be empty"))

    observation_axis = NativePPLAxis(
        :observation, Base.OneTo(observation_count))
    predictor_inputs = NamedTuple{predictor_names}(map(
        (predictor_name, predictor) -> NativePPLInput(
            predictor_name, :predictor, observation_axis, eltype(predictor)),
        predictor_names, predictor_values))
    response_eltype = response isa AbstractVector ? eltype(response) :
        (isempty(predictor_values) ? T : eltype(plan.inputs.response))
    response_input = NativePPLInput(
        response_name, :response, observation_axis, response_eltype)

    nodes = _native_ppl_rebind_nodes(
        plan, observation_axis, predictors, freeze_constants)
    likelihood = _native_ppl_rebind_likelihood(
        plan.factors.likelihood, observation_axis)
    new_bindings = response isa AbstractVector ?
        merge(predictors, NamedTuple{(response_name,)}((response,))) : predictors

    rebound_plan = NativePPLPlan(
        merge(plan.axes, (; observation=observation_axis)),
        (; predictors=predictor_inputs, response=response_input),
        plan.parameters,
        nodes,
        merge(plan.factors, (; likelihood)),
        _native_ppl_queries(observation_axis, likelihood),
        new_bindings,
    )
    _native_ppl_prepare_bindings(rebound_plan, predictors, response, T)
end

_native_ppl_query_spec(plan::NativePPLPlan, ::NativePPLLinearPredictor) =
    plan.queries.linear_predictor
_native_ppl_query_spec(plan::NativePPLPlan, ::NativePPLPointwiseLogLikelihood) =
    plan.queries.pointwise_loglikelihood
_native_ppl_query_spec(plan::NativePPLPlan, ::NativePPLPosteriorPredictive) =
    plan.queries.posterior_predictive
_native_ppl_query_spec(::NativePPLPlan,
                       ::NativePPLNodeOutput{Name}) where {Name} =
    throw(NativePPLCapabilityError(
        :query,
        "named node output `$Name` requires the typed factor-DAG executor"))
_native_ppl_query_spec(prepared::NativePPLPrepared, query::NativePPLQuery) =
    _native_ppl_query_spec(prepared.plan, query)

function _native_ppl_check_batch_positions(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
)
    dimension = LogDensityProblems.dimension(prepared)
    size(positions, 2) == dimension || throw(DimensionMismatch(
        "native PPL draw matrix has $(size(positions, 2)) coordinate columns; " *
        "expected $dimension"))
    axes(positions, 1) == Base.OneTo(size(positions, 1)) ||
        throw(ArgumentError(
            "native PPL batch queries require a one-based draw axis; got " *
            "$(axes(positions, 1))"))
    axes(positions, 2) == Base.OneTo(dimension) || throw(ArgumentError(
        "native PPL batch queries require one-based coordinate columns; got " *
        "$(axes(positions, 2))"))
    nothing
end

function _native_ppl_batch_output_signature_unchecked(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLQuery,
)
    element_signature = native_query_output(
        _native_ppl_query_spec(prepared, query))
    NativePPLBatchOutputSignature(
        NativePPLAxis(:draw, Base.OneTo(size(positions, 1))),
        native_output_axis(element_signature),
        element_signature.element_type,
        NativePPLDenseMatrixLayout(),
    )
end

function _native_ppl_batch_output_signature(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLQuery,
)
    _native_ppl_check_batch_positions(prepared, positions)
    _native_ppl_batch_output_signature_unchecked(prepared, positions, query)
end

function _native_ppl_named_map(f, values::NamedTuple{Names}) where {Names}
    NamedTuple{Names}(map(f, Tuple(values)))
end

function _native_ppl_check_bundle_queries(queries::NamedTuple)
    for (name, query) in pairs(queries)
        query isa NativePPLQuery || throw(ArgumentError(
            "native PPL bundle query `$name` must be a typed graph query; " *
            "got $(typeof(query))"))
    end
    nothing
end

function _native_ppl_batch_output_signature(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    queries::NamedTuple,
)
    _native_ppl_check_batch_positions(prepared, positions)
    _native_ppl_check_bundle_queries(queries)
    _native_ppl_named_map(
        query -> _native_ppl_batch_output_signature_unchecked(
            prepared, positions, query),
        queries)
end

function _native_ppl_allocate_output(signature::NativePPLOutputSignature,
                                     prepared::NativePPLPrepared)
    _native_ppl_allocate_output(
        native_output_layout(signature), signature, prepared)
end

function _native_ppl_allocate_output(::NativePPLDenseVectorLayout,
                                     signature::NativePPLOutputSignature,
                                     prepared::NativePPLPrepared)
    axis = native_output_axis(signature)
    axis.keys == Base.OneTo(length(axis)) || throw(NativePPLCapabilityError(
        :output_layout,
        "dense-vector allocation requires a one-based semantic axis; " *
        "got $(axis.keys)"))
    T = native_output_eltype(signature, prepared)
    Vector{T}(undef, length(axis))
end

function _native_ppl_allocate_output(signature::NativePPLBatchOutputSignature,
                                     prepared::NativePPLPrepared)
    _native_ppl_allocate_output(
        native_output_layout(signature), signature, prepared)
end

function _native_ppl_allocate_output(::NativePPLDenseMatrixLayout,
                                     signature::NativePPLBatchOutputSignature,
                                     prepared::NativePPLPrepared)
    draw_axis, observation_axis = native_output_axes(signature)
    draw_axis.keys == Base.OneTo(length(draw_axis)) ||
        throw(NativePPLCapabilityError(
            :output_layout,
            "dense-matrix allocation requires a one-based draw axis; got " *
            "$(draw_axis.keys)"))
    observation_axis.keys == Base.OneTo(length(observation_axis)) ||
        throw(NativePPLCapabilityError(
            :output_layout,
            "dense-matrix allocation requires a one-based observation axis; got " *
            "$(observation_axis.keys)"))
    T = native_output_eltype(signature, prepared)
    Matrix{T}(undef, length(draw_axis), length(observation_axis))
end


_native_ppl_allocate_output(signatures::NamedTuple,
                            prepared::NativePPLPrepared) =
    _native_ppl_named_map(
        signature -> _native_ppl_allocate_output(signature, prepared),
        signatures)

function _native_ppl_check_query_output(output::AbstractVector,
                                        prepared::NativePPLPrepared,
                                        query::NativePPLQuery)
    query_spec = _native_ppl_query_spec(prepared, query)
    query_name = native_query_name(query_spec)
    signature = native_query_output(query_spec)
    expected_axis = native_output_axis(signature).keys
    axes(output, 1) == expected_axis || throw(DimensionMismatch(
        "native PPL `$query_name` output axis $(axes(output, 1)) does not match " *
        "the declared observation axis $expected_axis"))
    expected_eltype = native_output_eltype(signature, prepared)
    eltype(output) === expected_eltype || throw(ArgumentError(
        "native PPL `$query_name` output eltype $(eltype(output)) does not match " *
        "declared output eltype $expected_eltype"))
    native_output_layout_accepts(native_output_layout(signature), output) ||
        throw(ArgumentError(
            "native PPL `$query_name` output $(typeof(output)) does not satisfy " *
            "declared layout $(typeof(native_output_layout(signature)))"))
    length(output) == length(expected_axis) || throw(DimensionMismatch(
        "native PPL `$query_name` output has length $(length(output)); expected " *
        "$(length(expected_axis)) for the observation axis"))
    output
end

function _native_ppl_check_batch_execution(
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLQuery,
)
    signature = _native_ppl_batch_output_signature(prepared, positions, query)
    draw_axis, observation_axis = native_output_axes(signature)
    axes(output) == (draw_axis.keys, observation_axis.keys) ||
        throw(DimensionMismatch(
            "native PPL batch output axes $(axes(output)) do not match " *
            "declared draw-by-observation axes " *
            "$((draw_axis.keys, observation_axis.keys))"))
    expected_eltype = native_output_eltype(signature, prepared)
    eltype(output) === expected_eltype || throw(ArgumentError(
        "native PPL batch output eltype $(eltype(output)) does not match " *
        "declared output eltype $expected_eltype"))
    native_output_layout_accepts(native_output_layout(signature), output) ||
        throw(ArgumentError(
            "native PPL batch output $(typeof(output)) does not satisfy " *
            "declared layout $(typeof(native_output_layout(signature)))"))
    eltype(positions) === eltype(workspace) || throw(ArgumentError(
        "native PPL draw matrix eltype $(eltype(positions)) does not match " *
        "workspace eltype $(eltype(workspace))"))
    eltype(prepared) === eltype(workspace) || throw(ArgumentError(
        "native PPL prepared eltype $(eltype(prepared)) does not match " *
        "workspace eltype $(eltype(workspace))"))
    length(workspace.primal.location) == length(observation_axis) ||
        throw(DimensionMismatch(
            "native PPL workspace location has " *
            "$(length(workspace.primal.location)) rows; expected " *
            "$(length(observation_axis))"))
    Base.mightalias(output, positions) && throw(ArgumentError(
        "native PPL batch output must not alias the draw matrix"))
    Base.mightalias(positions, workspace.primal.location) && throw(ArgumentError(
        "native PPL draw matrix must not alias the workspace location buffer"))
    Base.mightalias(positions, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL draw matrix must not alias the workspace pointwise buffer"))
    Base.mightalias(positions, workspace.gradient) && throw(ArgumentError(
        "native PPL draw matrix must not alias the workspace gradient buffer"))
    Base.mightalias(output, workspace.primal.location) && throw(ArgumentError(
        "native PPL batch output must not alias the workspace location buffer"))
    Base.mightalias(output, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL batch output must not alias the workspace pointwise buffer"))
    Base.mightalias(output, workspace.gradient) && throw(ArgumentError(
        "native PPL batch output must not alias the workspace gradient buffer"))
    signature
end

_native_ppl_check_batch_query_state(
    ::NativePPLWorkspace, ::NativePPLPrepared, ::NativePPLLinearPredictor,
) = nothing
_native_ppl_check_batch_query_state(
    ::NativePPLWorkspace, ::NativePPLPrepared, ::NativePPLPosteriorPredictive,
) = nothing
function _native_ppl_check_batch_query_state(
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    ::NativePPLPointwiseLogLikelihood,
)
    response = _native_ppl_require_response(
        prepared, "batched pointwise log likelihood")
    observations = length(prepared.plan.axes.observation)
    length(response) == observations || throw(DimensionMismatch(
        "native PPL prepared response has $(length(response)) rows; expected " *
        "$observations"))
    length(workspace.primal.pointwise_loglikelihood) == observations ||
        throw(DimensionMismatch(
            "native PPL workspace pointwise likelihood has " *
            "$(length(workspace.primal.pointwise_loglikelihood)) rows; expected " *
            "$observations"))
    length(workspace.gradient) == LogDensityProblems.dimension(prepared) ||
        throw(DimensionMismatch(
            "native PPL workspace gradient has $(length(workspace.gradient)) " *
            "coordinates; expected $(LogDensityProblems.dimension(prepared))"))
    nothing
end

function _native_ppl_evaluate!(output::AbstractVector,
                               workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector,
                               query::NativePPLLinearPredictor)
    _native_ppl_check_query_output(output, prepared, query)
    _native_ppl_location!(workspace, prepared, position)
    copyto!(output, workspace.primal.location)
end

function _native_ppl_evaluate!(output::AbstractVector,
                               workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector,
                               query::NativePPLPointwiseLogLikelihood)
    _native_ppl_check_query_output(output, prepared, query)
    _native_ppl_require_response(prepared, "pointwise log likelihood")
    _native_ppl_logdensity!(workspace, prepared, position)
    copyto!(output, workspace.primal.pointwise_loglikelihood)
end

function _native_ppl_simulate!(rng::AbstractRNG,
                               output::AbstractVector,
                               workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector,
                               query::NativePPLPosteriorPredictive)
    _native_ppl_check_query_output(output, prepared, query)
    _native_ppl_location!(workspace, prepared, position)
    _native_ppl_model_simulate!(
        rng, prepared.plan.factors.likelihood, output,
        prepared, position, workspace.primal)
end

@inline function _native_ppl_draw_parameter!(
    rng::AbstractRNG,
    position::AbstractVector{T},
    factor::NativePPLStandardNormalFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
) where {T,Parameter}
    for coordinate in factor.unconstrained
        position[coordinate] = randn(rng, T)
    end
    nothing
end


@inline function _native_ppl_draw_parameter!(
    rng::AbstractRNG,
    position::AbstractVector{T},
    factor::NativePPLScalarNormalFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
) where {T,Parameter}
    position[factor.unconstrained_index] =
        T(factor.location) + T(factor.scale) * randn(rng, T)
    nothing
end


@inline function _native_ppl_draw_parameter!(
    rng::AbstractRNG,
    position::AbstractVector{T},
    factor::NativePPLExponentialFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
) where {T,Parameter}
    constrained = T(factor.scale) * randexp(rng, T)
    position[factor.unconstrained_index] = log(constrained)
    nothing
end


function _native_ppl_check_prior_predictive(
    output::AbstractVector,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
)
    _native_ppl_check_query_output(
        output, prepared, NativePPLPosteriorPredictive())
    _native_ppl_check_location_execution(workspace, prepared, position)
    Base.mightalias(output, position) && throw(ArgumentError(
        "native PPL prior-predictive output must not alias its position"))
    Base.mightalias(output, workspace.primal.location) && throw(ArgumentError(
        "native PPL prior-predictive output must not alias the workspace " *
        "location buffer"))
    Base.mightalias(output, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL prior-predictive output must not alias the workspace " *
            "pointwise buffer"))
    Base.mightalias(output, workspace.gradient) && throw(ArgumentError(
        "native PPL prior-predictive output must not alias the workspace " *
        "gradient buffer"))
    Base.mightalias(position, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL prior-predictive position must not alias the " *
            "workspace pointwise buffer"))
    Base.mightalias(position, workspace.gradient) && throw(ArgumentError(
        "native PPL prior-predictive position must not alias the workspace " *
        "gradient buffer"))
    for (name, predictor) in pairs(prepared.predictors)
        Base.mightalias(output, predictor) && throw(ArgumentError(
            "native PPL prior-predictive output must not alias prepared " *
            "predictor `$name`"))
        Base.mightalias(position, predictor) && throw(ArgumentError(
            "native PPL prior-predictive position must not alias prepared " *
            "predictor `$name`"))
    end
    if native_ppl_has_response(prepared)
        Base.mightalias(output, prepared.response) && throw(ArgumentError(
            "native PPL prior-predictive output must not alias the prepared " *
            "response"))
        Base.mightalias(position, prepared.response) && throw(ArgumentError(
            "native PPL prior-predictive position must not alias the prepared " *
            "response"))
    end

    location_parameter = _native_ppl_location_parameter(prepared.plan)
    location_factor = _native_ppl_location_factor(prepared.plan)
    _native_ppl_validate_coefficient_prior(
        location_factor, location_parameter)
    if hasproperty(prepared.plan.factors, :scale_prior)
        hasproperty(prepared.plan.parameters, :scale) || throw(
            NativePPLCapabilityError(
                :graph_identity,
                "compiled prior-predictive schedule is missing its scale " *
                "parameter"))
        _native_ppl_validate_scale_prior(
            prepared.plan.factors.scale_prior,
            prepared.plan.parameters.scale)
    end
    nothing
end


"""
    NativePPL.simulate_prior!(rng, position, output, workspace, prepared)

Draw every currently supported free parameter/site from its generating factor,
then simulate the response in graph order. `position` receives unconstrained
coordinates and `output` receives the prior-predictive response. Validation is
completed before the RNG is advanced.
"""
function _native_ppl_simulate_prior!(
    rng::AbstractRNG,
    position::AbstractVector,
    output::AbstractVector,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
)
    _native_ppl_check_prior_predictive(
        output, workspace, prepared, position)
    _native_ppl_draw_parameter!(
        rng, position, _native_ppl_location_factor(prepared.plan),
        _native_ppl_location_parameter(prepared.plan))
    if hasproperty(prepared.plan.factors, :scale_prior)
        _native_ppl_draw_parameter!(
            rng, position, prepared.plan.factors.scale_prior,
            prepared.plan.parameters.scale)
    end
    _native_ppl_simulate!(
        rng, output, workspace, prepared, position,
        NativePPLPosteriorPredictive())
end

function _native_ppl_evaluate_draws!(
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLLinearPredictor,
)
    _native_ppl_check_batch_execution(
        output, workspace, prepared, positions, query)
    _native_ppl_check_batch_query_state(workspace, prepared, query)
    for draw in axes(positions, 1)
        _native_ppl_location!(
            workspace, prepared, @view(positions[draw, :]))
        for observation in axes(output, 2)
            @inbounds output[draw, observation] =
                workspace.primal.location[observation]
        end
    end
    output
end

function _native_ppl_evaluate_draws!(
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLPointwiseLogLikelihood,
)
    _native_ppl_check_batch_execution(
        output, workspace, prepared, positions, query)
    _native_ppl_check_batch_query_state(workspace, prepared, query)
    for draw in axes(positions, 1)
        _native_ppl_logdensity!(
            workspace, prepared, @view(positions[draw, :]))
        for observation in axes(output, 2)
            @inbounds output[draw, observation] =
                workspace.primal.pointwise_loglikelihood[observation]
        end
    end
    output
end

function _native_ppl_simulate_draws!(
    rng::AbstractRNG,
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLPosteriorPredictive,
)
    _native_ppl_check_batch_execution(
        output, workspace, prepared, positions, query)
    _native_ppl_check_batch_query_state(workspace, prepared, query)
    for draw in axes(positions, 1)
        position = @view positions[draw, :]
        _native_ppl_location!(workspace, prepared, position)
        _native_ppl_model_simulate!(
            rng, prepared.plan.factors.likelihood,
            @view(output[draw, :]), prepared, position, workspace.primal)
    end
    output
end

function _native_ppl_check_bundle_outputs(
    outputs::NamedTuple{Names},
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    queries::NamedTuple{Names},
) where {Names}
    _native_ppl_check_batch_positions(prepared, positions)
    _native_ppl_check_bundle_queries(queries)
    output_values = Tuple(outputs)
    query_values = Tuple(queries)
    for i in eachindex(output_values)
        output_values[i] isa AbstractMatrix || throw(ArgumentError(
            "native PPL bundle output `$((Names[i]))` must be an " *
            "AbstractMatrix; got $(typeof(output_values[i]))"))
        _native_ppl_check_batch_execution(
            output_values[i], workspace, prepared, positions, query_values[i])
        _native_ppl_check_batch_query_state(
            workspace, prepared, query_values[i])
    end
    for i in eachindex(output_values), j in (i + 1):length(output_values)
        Base.mightalias(output_values[i], output_values[j]) &&
            throw(ArgumentError(
                "native PPL bundle outputs `$((Names[i]))` and `$((Names[j]))` " *
                "must not alias"))
    end
    nothing
end

function _native_ppl_check_bundle_outputs(
    outputs::NamedTuple,
    ::NativePPLWorkspace,
    ::NativePPLPrepared,
    ::AbstractMatrix,
    queries::NamedTuple,
)
    throw(ArgumentError(
        "native PPL bundle output keys $(keys(outputs)) do not match " *
        "query keys $(keys(queries))"))
end

function _native_ppl_bundle_requires_rng(prepared::NativePPLPrepared,
                                         queries::NamedTuple)
    _native_ppl_check_bundle_queries(queries)
    for query in Tuple(queries)
        native_query_effect(_native_ppl_query_spec(prepared, query)) === :rng &&
            return true
    end
    false
end

function _native_ppl_bundle_has_pointwise(queries::NamedTuple)
    for query in Tuple(queries)
        query isa NativePPLPointwiseLogLikelihood && return true
    end
    false
end

@inline function _native_ppl_write_bundle_query!(
    rng,
    output::AbstractMatrix,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    ::NativePPLLinearPredictor,
)
    for observation in axes(output, 2)
        @inbounds output[draw, observation] =
            workspace.primal.location[observation]
    end
    nothing
end

@inline function _native_ppl_write_bundle_query!(
    rng,
    output::AbstractMatrix,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    ::NativePPLPointwiseLogLikelihood,
)
    for observation in axes(output, 2)
        @inbounds output[draw, observation] =
            workspace.primal.pointwise_loglikelihood[observation]
    end
    nothing
end

@inline function _native_ppl_write_bundle_query!(
    rng::AbstractRNG,
    output::AbstractMatrix,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    ::NativePPLPosteriorPredictive,
)
    _native_ppl_model_simulate!(
        rng, prepared.plan.factors.likelihood, @view(output[draw, :]),
        prepared, position, workspace.primal)
    nothing
end

@inline _native_ppl_write_bundle_queries!(
    rng, ::Tuple{}, ::Tuple{}, draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
) = nothing

@inline function _native_ppl_write_bundle_queries!(
    rng,
    outputs::Tuple,
    queries::Tuple,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
)
    _native_ppl_write_bundle_query!(
        rng, first(outputs), draw, workspace, prepared, position,
        first(queries))
    _native_ppl_write_bundle_queries!(
        rng, Base.tail(outputs), Base.tail(queries), draw,
        workspace, prepared, position)
end

function _native_ppl_execute_draws!(
    rng,
    outputs::NamedTuple,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    queries::NamedTuple,
)
    rng === nothing && _native_ppl_bundle_requires_rng(prepared, queries) &&
        throw(ArgumentError(
            "native PPL query bundle contains an RNG-effect query; " *
            "call `execute_draws!` with an explicit RNG"))
    _native_ppl_check_bundle_outputs(
        outputs, workspace, prepared, positions, queries)
    isempty(queries) && return outputs

    pointwise = _native_ppl_bundle_has_pointwise(queries)
    output_values = Tuple(outputs)
    query_values = Tuple(queries)
    for draw in axes(positions, 1)
        position = @view positions[draw, :]
        _native_ppl_location!(workspace, prepared, position)
        pointwise && _native_ppl_model_logdensity!(
            prepared.plan.factors.likelihood, position, prepared,
            workspace.primal)
        _native_ppl_write_bundle_queries!(
            rng, output_values, query_values, draw,
            workspace, prepared, position)
    end
    outputs
end

"""
Namespaced walking-skeleton API for the native PPL.

`DifferentiationInterface.AutoEnzyme()` is the sole tested, recommended, and
guaranteed derivative backend. Other DI backends are outside this prototype's
compatibility contract even when they happen to execute.
"""
module NativePPL

const BRM = parentmodule(@__MODULE__)

const Plan = BRM.NativePPLPlan
const Prepared = BRM.NativePPLPrepared
const Workspace = BRM.NativePPLWorkspace
const LogDensityProblem = BRM.NativePPLLogDensityProblem
const CapabilityError = BRM.NativePPLCapabilityError
const OutputSignature = BRM.NativePPLOutputSignature
const BatchOutputSignature = BRM.NativePPLBatchOutputSignature
const DenseVectorLayout = BRM.NativePPLDenseVectorLayout
const DenseMatrixLayout = BRM.NativePPLDenseMatrixLayout
const LinearPredictor = BRM.NativePPLLinearPredictor
const PointwiseLogLikelihood = BRM.NativePPLPointwiseLogLikelihood
const PosteriorPredictive = BRM.NativePPLPosteriorPredictive
const NodeOutput = BRM.NativePPLNodeOutput

prepare(plan::Plan; kwargs...) = BRM._native_ppl_prepare(plan; kwargs...)
workspace(prepared::Prepared, ::Type{T}=eltype(prepared)) where {T<:AbstractFloat} =
    BRM._native_ppl_workspace(prepared, T)
workspace(prepared::Prepared, ::Type{T}, backend) where {T<:AbstractFloat} =
    BRM._native_ppl_workspace(prepared, T, backend)
rebind(prepared::Prepared, bindings; kwargs...) =
    BRM._native_ppl_rebind(prepared, bindings; kwargs...)
has_response(prepared::Prepared) = BRM.native_ppl_has_response(prepared)
output_signature(plan::Plan, query::BRM.NativePPLQuery) =
    BRM.native_query_output(BRM._native_ppl_query_spec(plan, query))
output_signature(prepared::Prepared, query::BRM.NativePPLQuery) =
    output_signature(prepared.plan, query)
batch_output_signature(prepared::Prepared, positions::AbstractMatrix,
                       query::BRM.NativePPLQuery) =
    BRM._native_ppl_batch_output_signature(prepared, positions, query)
batch_output_signature(prepared::Prepared, positions::AbstractMatrix,
                       queries::NamedTuple) =
    BRM._native_ppl_batch_output_signature(prepared, positions, queries)
output_axis(signature::OutputSignature) = BRM.native_output_axis(signature)
output_axis(signature::BatchOutputSignature) = BRM.native_output_axis(signature)
output_axes(signature::Union{OutputSignature,BatchOutputSignature}) =
    BRM.native_output_axes(signature)
output_draw_axis(signature::BatchOutputSignature) =
    BRM.native_output_draw_axis(signature)
output_eltype(signature::OutputSignature, prepared::Prepared) =
    BRM.native_output_eltype(signature, prepared)
output_eltype(signature::BatchOutputSignature, prepared::Prepared) =
    BRM.native_output_eltype(signature, prepared)
output_eltype(signature::OutputSignature, ::Type{T}) where {T<:AbstractFloat} =
    BRM.native_output_eltype(signature, T)
output_eltype(signature::BatchOutputSignature, ::Type{T}) where {T<:AbstractFloat} =
    BRM.native_output_eltype(signature, T)
output_layout(signature::OutputSignature) = BRM.native_output_layout(signature)
output_layout(signature::BatchOutputSignature) = BRM.native_output_layout(signature)
allocate_output(signature::OutputSignature, prepared::Prepared) =
    BRM._native_ppl_allocate_output(signature, prepared)
allocate_output(signature::BatchOutputSignature, prepared::Prepared) =
    BRM._native_ppl_allocate_output(signature, prepared)
allocate_output(signatures::NamedTuple, prepared::Prepared) =
    BRM._native_ppl_allocate_output(signatures, prepared)
allocate_output(prepared::Prepared, query::BRM.NativePPLQuery) =
    allocate_output(output_signature(prepared, query), prepared)
logdensity!(work::Workspace, prepared::Prepared, position::AbstractVector) =
    BRM._native_ppl_logdensity!(work, prepared, position)
logdensity_and_gradient!(work::Workspace, prepared::Prepared,
                         position::AbstractVector) =
    BRM._native_ppl_logdensity_and_gradient!(work, prepared, position)
evaluate!(output::AbstractVector, work::Workspace, prepared::Prepared,
          position::AbstractVector, query::BRM.NativePPLQuery) =
    BRM._native_ppl_evaluate!(output, work, prepared, position, query)
function evaluate(work::Workspace, prepared::Prepared,
                  position::AbstractVector, query::BRM.NativePPLQuery)
    output = allocate_output(prepared, query)
    evaluate!(output, work, prepared, position, query)
end
simulate!(rng::BRM.AbstractRNG, output::AbstractVector, work::Workspace,
          prepared::Prepared, position::AbstractVector,
          query::BRM.NativePPLQuery=PosteriorPredictive()) =
    BRM._native_ppl_simulate!(rng, output, work, prepared, position, query)
simulate_prior!(rng::BRM.AbstractRNG, position::AbstractVector,
                output::AbstractVector, work::Workspace,
                prepared::Prepared) =
    BRM._native_ppl_simulate_prior!(
        rng, position, output, work, prepared)
function simulate(rng::BRM.AbstractRNG, work::Workspace, prepared::Prepared,
                  position::AbstractVector,
                  query::BRM.NativePPLQuery=PosteriorPredictive())
    output = allocate_output(prepared, query)
    simulate!(rng, output, work, prepared, position, query)
end
function simulate_prior(rng::BRM.AbstractRNG, work::Workspace,
                        prepared::Prepared)
    position = Vector{eltype(work)}(
        undef, BRM.LogDensityProblems.dimension(prepared))
    output = allocate_output(prepared, PosteriorPredictive())
    simulate_prior!(rng, position, output, work, prepared)
    (; position, response=output)
end
evaluate_draws!(output::AbstractMatrix, work::Workspace, prepared::Prepared,
                positions::AbstractMatrix, query::BRM.NativePPLQuery) =
    BRM._native_ppl_evaluate_draws!(
        output, work, prepared, positions, query)
function evaluate_draws(work::Workspace, prepared::Prepared,
                        positions::AbstractMatrix,
                        query::BRM.NativePPLQuery)
    signature = batch_output_signature(prepared, positions, query)
    output = allocate_output(signature, prepared)
    evaluate_draws!(output, work, prepared, positions, query)
end
simulate_draws!(rng::BRM.AbstractRNG, output::AbstractMatrix,
                work::Workspace, prepared::Prepared,
                positions::AbstractMatrix,
                query::BRM.NativePPLQuery=PosteriorPredictive()) =
    BRM._native_ppl_simulate_draws!(
        rng, output, work, prepared, positions, query)
function simulate_draws(rng::BRM.AbstractRNG, work::Workspace,
                        prepared::Prepared, positions::AbstractMatrix,
                        query::BRM.NativePPLQuery=PosteriorPredictive())
    signature = batch_output_signature(prepared, positions, query)
    output = allocate_output(signature, prepared)
    simulate_draws!(rng, output, work, prepared, positions, query)
end
execute_draws!(outputs::NamedTuple, work::Workspace, prepared::Prepared,
               positions::AbstractMatrix, queries::NamedTuple) =
    BRM._native_ppl_execute_draws!(
        nothing, outputs, work, prepared, positions, queries)
execute_draws!(rng::BRM.AbstractRNG, outputs::NamedTuple, work::Workspace,
               prepared::Prepared, positions::AbstractMatrix,
               queries::NamedTuple) =
    BRM._native_ppl_execute_draws!(
        rng, outputs, work, prepared, positions, queries)
function execute_draws(work::Workspace, prepared::Prepared,
                       positions::AbstractMatrix, queries::NamedTuple)
    BRM._native_ppl_bundle_requires_rng(prepared, queries) &&
        throw(ArgumentError(
            "native PPL query bundle contains an RNG-effect query; " *
            "call `execute_draws` with an explicit RNG"))
    signatures = batch_output_signature(prepared, positions, queries)
    outputs = allocate_output(signatures, prepared)
    execute_draws!(outputs, work, prepared, positions, queries)
end
function execute_draws(rng::BRM.AbstractRNG, work::Workspace,
                       prepared::Prepared, positions::AbstractMatrix,
                       queries::NamedTuple)
    signatures = batch_output_signature(prepared, positions, queries)
    outputs = allocate_output(signatures, prepared)
    execute_draws!(rng, outputs, work, prepared, positions, queries)
end

include("native_ppl_authoring.jl")
include("native_ppl_julianic.jl")

export Plan, Prepared, Workspace, LogDensityProblem, CapabilityError
export OutputSignature, BatchOutputSignature
export DenseVectorLayout, DenseMatrixLayout
export LinearPredictor, PointwiseLogLikelihood, PosteriorPredictive, NodeOutput
export compile, prepare, workspace, rebind, has_response
export output_signature, batch_output_signature
export output_axis, output_axes, output_draw_axis, output_eltype, output_layout
export allocate_output, evaluate, simulate, simulate_prior
export evaluate_draws, simulate_draws, execute_draws
export logdensity!, logdensity_and_gradient!, evaluate!, simulate!
export simulate_prior!
export evaluate_draws!, simulate_draws!, execute_draws!

end
