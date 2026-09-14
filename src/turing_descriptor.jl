# Native semantic descriptors for TuringBRMI. This adapter deliberately does
# not manufacture a Stan descriptor: `stan === nothing`, and every output name
# is a DynamicPPL site or a returned BRM quantity.

function _brm_turing_parameter_shape(parameter)
    parameter.prior.callable === LKJCovarianceFactor &&
        return (:matrix, (:dynamic, :dynamic))
    shape = _brm_distribution_shape(parameter.prior.callable,
                                     parameter.prior.args)
    variate = isnothing(shape) ? nothing : first(shape)
    variate === Distributions.Univariate && return (:real, ())
    variate === Distributions.Matrixvariate && return (:matrix, (:dynamic, :dynamic))
    (:vector, (:dynamic,))
end

function _brm_turing_components(plan::_TuringGenericPlan)
    plan.predictors
end
function _brm_turing_components(plan::_TuringMultiResponsePlan)
    ordered = Any[]
    seen = Set{Symbol}()
    for child in plan.plans, component in child.predictors
        component.predictor.name in seen && continue
        push!(seen, component.predictor.name)
        push!(ordered, component)
    end
    Tuple(ordered)
end

function _brm_turing_parameters(plan::_TuringGenericPlan)
    plan.parameters
end
function _brm_turing_parameters(plan::_TuringMultiResponsePlan)
    ordered = Any[]
    seen = Set{Symbol}()
    for child in plan.plans, parameter in child.parameters
        parameter.name in seen && continue
        push!(seen, parameter.name)
        push!(ordered, parameter)
    end
    Tuple(ordered)
end

_brm_turing_response_plans(plan::_TuringGenericPlan) = (plan,)
_brm_turing_response_plans(plan::_TuringMultiResponsePlan) = plan.plans

function _brm_turing_outputs(plan, brmi, semantics)
    outputs = BRMOutput[]
    for parameter in _brm_turing_parameters(plan)
        native_type, native_size = _brm_turing_parameter_shape(parameter)
        push!(outputs, BRMOutput(parameter.name, :parameter,
            native_type, native_size,
            (;), :posterior, nothing, :parameter, nothing, parameter.name,
            nothing, nothing))
    end
    components = _brm_turing_components(plan)
    shared = _turing_shared_group_plans(components)
    shared_members = plan isa _TuringGenericPlan ?
        Set((member.component_index, member.block_index)
            for group in shared for member in group.members) :
        Set((member.predictor, member.block_index)
            for group in shared for member in group.members)
    joint_by_predictor = Dict{Symbol,Int}()
    for (joint_index, joint) in enumerate(plan.joint_r2d2)
        for mapping in joint.predictors
            joint_by_predictor[mapping.predictor] = joint_index
        end
        push!(outputs, BRMOutput(Symbol(:r2d2_joint_, joint_index), :parameter,
            :named_tuple, (), (;), :posterior, nothing, :parameter, nothing,
            joint.id, nothing, nothing))
    end
    for (index, component) in enumerate(components)
        logical = component.predictor.name
        get(semantics.roles, logical, nothing) === :linear_predictor || error(
            "brm_descriptor: Turing predictor `$logical` is absent from the " *
            "shared semantic program")
        beta = length(components) == 1 ? :beta_pop :
            (plan isa _TuringGenericPlan && index == 1 ? :beta_pop :
             Symbol(:beta_pop_, logical))
        labels = _brm_labels(brmi, logical)
        r2d2_derived = !isnothing(component.r2d2) || haskey(joint_by_predictor, logical)
        if !isnothing(component.r2d2)
            push!(outputs, BRMOutput(Symbol(:r2d2_, logical), :parameter,
                :named_tuple, (), (;), :posterior, nothing, :parameter, nothing,
                logical, nothing, nothing))
        end
        push!(outputs, BRMOutput(beta,
            r2d2_derived ? :transformed_parameter : :parameter, :vector,
            (size(component.design.matrix, 2),), (;),
            r2d2_derived ? :derived : :posterior, nothing,
            :population_effect, nothing, logical, labels,
            nothing))
        push!(outputs, BRMOutput(logical, :transformed_parameter, :vector,
            (size(component.design.matrix, 1),), (;), :derived, nothing,
            :linear_predictor, nothing,
            logical, nothing, nothing))
        for (group_index, block) in enumerate(component.random_effects)
            member_key = plan isa _TuringGenericPlan ? (index, group_index) :
                         (logical, group_index)
            member_key in shared_members && continue
            site = plan isa _TuringGenericPlan ?
                Symbol(:group_, index, :_, group_index) :
                Symbol(:group_, logical, :_, group_index)
            push!(outputs, BRMOutput(site, :parameter, :named_tuple, (), (;),
                :posterior, nothing, :random_effect, nothing, block.group,
                nothing, nothing))
        end
        for term_index in eachindex(component.terms)
            site = Symbol(:term_, logical, :_, term_index)
            push!(outputs, BRMOutput(site, :parameter, :named_tuple, (), (;),
                :posterior, nothing, :parameter, nothing, site, nothing, nothing))
        end
    end
    for (index, group) in enumerate(shared)
        push!(outputs, BRMOutput(Symbol(:shared_group_, index), :parameter,
            :named_tuple, (), (;), :posterior, nothing, :random_effect,
            nothing, first(group.key), nothing, nothing))
    end
    seen_assignments = Set{Symbol}()
    for child in _brm_turing_response_plans(plan), assignment in child.assignments
        assignment.name in seen_assignments && continue
        push!(seen_assignments, assignment.name)
        rowwise = _brm_has_row_ref(assignment.expression)
        push!(outputs, BRMOutput(assignment.name, :transformed_parameter,
            rowwise ? :vector : :real, rowwise ? (:dynamic,) : (), (;),
            :derived, nothing, :stan_derived, nothing, assignment.name,
            nothing, nothing))
    end
    seen_responses = Set{Symbol}()
    for child in _brm_turing_response_plans(plan)
        response = child.response_name
        response in seen_responses && continue
        push!(seen_responses, response)
        push!(outputs, BRMOutput(response, :generated_quantity, :vector,
            (length(child.response),), (;), :draw, response, :posterior_predictive,
            nothing, response, nothing, nothing))
        push!(outputs, BRMOutput(Symbol(response, :_loglik),
            :generated_quantity, :vector, (length(child.response),), (;),
            :pointwise_loglik, response, :pointwise_loglik, nothing, response,
            nothing, nothing))
    end
    outputs
end

function _brm_turing_inputs(backend::TuringBRMI)
    responses = Set(child.response_name for child in
                    _brm_turing_response_plans(backend.plan))
    data_names = Set{Symbol}()
    for child in _brm_turing_response_plans(backend.plan)
        union!(data_names, keys(child.context.data))
    end
    Tuple(BRMInput(name, :data, (:dynamic,), (;), name in responses, false,
                   false, false, name, nothing)
          for name in sort!(collect(data_names)))
end

_brm_descriptor_select(value, names) =
    (; (name => getproperty(value, name) for name in names)...)

function _brm_descriptor_select_generated(value, names)
    hasproperty(value, :responses) || return _brm_descriptor_select(value, names)
    (; (name => begin
            index = findfirst(response -> hasproperty(response, name), value.responses)
            isnothing(index) && error("generated quantity $name was not returned")
            getproperty(value.responses[index], name)
        end for name in names)...)
end

function _brm_descriptor_pointwise(backend, parameters; kwargs...)
    value = turing_pointwise_loglikelihoods(backend, parameters; kwargs...)
    (; (Symbol(name, :_loglik) => getproperty(value, name)
        for name in propertynames(value))...)
end

function _brm_turing_operations(backend::TuringBRMI, outputs, columns,
                                descriptor_operations, descriptor_titles)
    predictive = Tuple(o.name for o in outputs if o.role === :posterior_predictive)
    pointwise = Tuple(o.name for o in outputs if o.role === :pointwise_loglik)
    children = _brm_turing_response_plans(backend.plan)
    returned = Set{Symbol}()
    for child in children
        union!(returned, (p.name for p in child.parameters))
        union!(returned, (p.predictor.name for p in child.predictors))
        union!(returned, (a.name for a in child.assignments))
    end
    generated = Tuple(o.name for o in outputs if o.name in returned &&
                      o.role !== :posterior_predictive &&
                      o.role !== :pointwise_loglik)
    BRMOperation[
        BRMOperation(:pointwise_loglik, "Compute pointwise log-likelihoods", (),
            pointwise, :brm,
            (_d, parameters; kwargs...) ->
                _brm_descriptor_pointwise(backend, parameters; kwargs...)),
        BRMOperation(:generated_quantities, "Compute generated quantities", (),
            generated,
            :brm,
            (_d, parameters; kwargs...) -> _brm_descriptor_select_generated(
                turing_generated_quantities(backend, parameters; kwargs...),
                generated)),
        BRMOperation(:predict, "Draw posterior predictions", (), predictive, :brm,
            (_d, parameters; kwargs...) ->
                turing_posterior_predictive(backend, parameters; kwargs...)),
        BRMOperation(:reprocess, "Re-run preprocessing on a new dataframe",
            columns, (), :brm,
            (d, new_df; kwargs...) -> brm_descriptor(
                reprocess(backend, new_df; kwargs...); name=d.name,
                operations=descriptor_operations, titles=descriptor_titles)),
    ]
end

function brm_descriptor(backend::TuringBRMI;
                        name::Union{Nothing,Symbol}=nothing,
                        operations=Dict{Symbol,Any}(),
                        titles=Dict{Symbol,String}(), highlights=())
    isempty(highlights) || error(
        "brm_descriptor: Turing models have no Stan definitions to highlight")
    brmi = backend.parent
    semantics = _brm_descriptor_semantics(brmi)
    columns = Tuple(sort!(unique(Symbol[semantics.columns...,
        (child.response_name for child in _brm_turing_response_plans(backend.plan))...])))
    outputs = _brm_turing_outputs(backend.plan, brmi, semantics)
    ops = _brm_apply_overrides(
        _brm_turing_operations(backend, outputs, columns, operations, titles),
        operations, titles)
    descriptor_name = isnothing(name) ? :turing_model : name
    formula = _brm_descriptor_formula(brmi)
    id = "turing:" * formula
    BRMDescriptor(id, descriptor_name, formula, backend.plan, nothing,
        (), _brm_turing_inputs(backend), Tuple(outputs), Tuple(ops), columns, ())
end
