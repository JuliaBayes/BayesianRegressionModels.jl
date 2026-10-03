# Selected continuous declarations redraw inside the original dependency graph.
# The ordinary cohort path has no selection and retains its fitted completions.
const _SB_FRESH_COVARIATES_KEY = :__brm_fresh_covariates__

function _brm_covariate_members(lhs::JointResponseColumn)
    joint_response_names(lhs)
end
function _brm_covariate_members(lhs)
    source = _brm_observation_name(lhs)
    isnothing(source) ? () : (source,)
end

function _brm_covariate_inventory(brmi::BRMI)
    program = _brm_prepare_program(brmi)
    refs = Set{Symbol}()
    for operation in program.operations
        expression = operation.expression
        expression isa ExprColumn || continue
        args = getargs(expression)
        length(args) == 2 || continue
        rhs = last(args)
        # Joint member projections are the declaration's own aliases.
        rhs isa ExprColumn && getf(rhs) === brm_joint_column && continue
        _brm_operation_references!(refs, rhs)
    end
    byname = Dict(operation.name => operation for operation in program.operations)
    records = NamedTuple[]
    for key in program.order
        operation = byname[key]
        operation.role === :observation || continue
        lhs, rhs = getargs(operation.expression, 2)
        rhs isa ExprColumn || continue
        members = _brm_covariate_members(lhs)
        any(in(refs), members) || continue
        family = getf(rhs)
        supported = family in (Normal, LogNormal, MvNormalCholesky)
        raw = lhs isa JointResponseColumn ?
            map(c -> parent(parent(c)), joint_response_columns(lhs)) :
            (_sb_materialize_vec(lhs isa ExprColumn && getf(lhs)===mi ? only(getargs(lhs)) : lhs),)
        lengths = map(length, raw)
        all(==(first(lengths)), lengths) || error(
            "BRM modeled covariates: declaration $(members) has inconsistent member rows")
        dependencies = operation.dependencies
        push!(records, (; key, members, family, supported, rows=first(lengths),
            missing=map(c -> count(ismissing,c), raw), dependencies))
    end
    Tuple(records)
end

"""
    modeled_covariates(model)

Ordered modeled continuous declarations used by later model operations. Each
record exposes logical `members`, `family`, `supported`, `rows`, per-member
`missing` counts, `dependencies`, `resampled` and `value_role`. Select a scalar member with
`reprocess(...; resample_covariates=[:x])`; selecting a joint member redraws its
whole ordered block. This inventory never requires generated Stan names.
"""
function modeled_covariates(brmi::BRMI)
    map(_brm_covariate_inventory(brmi)) do record
        (; record.members, record.family, record.supported, record.rows,
           record.missing, record.dependencies, resampled=false,value_role=nothing)
    end
end
function modeled_covariates(model::Union{SBBRMI,GenerativePlan})
    selected = _sb_existing_covariates(model.preproc)
    map(modeled_covariates(model.parent)) do record
        active=any(in(selected),record.members)
        merge(record, (; resampled=active,value_role=active ? :covariate_draw : nothing))
    end
end

function _sb_covariate_selection(brmi, request)
    values = request === nothing ? () : request isa Symbol ? (request,) : request
    (values isa Tuple || values isa AbstractVector || values isa AbstractSet) &&
        all(v -> v isa Symbol,values) || throw(ArgumentError(
            "BRM resample_covariates expects a Symbol or collection of Symbols"))
    names = Set{Symbol}(values)
    isempty(names) && return names
    inventory = _brm_covariate_inventory(brmi)
    selected = Set{Symbol}()
    for name in names
        hits = filter(record -> name in record.members, inventory)
        length(hits) == 1 || throw(ArgumentError(
            "BRM resample_covariates: `$name` must name one modeled covariate member"))
        record = only(hits)
        record.supported || throw(ArgumentError(
            "BRM resample_covariates: $(record.members) requires Normal, LogNormal or MvNormalCholesky"))
        union!(selected, record.members)
    end
    program = _brm_prepare_program(brmi)
    dependencies = Dict(operation.name=>operation.dependencies for operation in program.operations)
    selected_bindings = union(selected,Set(record.key for record in inventory
        if any(in(selected),record.members)))
    depends_on_fresh(key,seen=Set{Symbol}()) = key in selected_bindings ||
        (!(key in seen) && (push!(seen,key);any(dep->depends_on_fresh(dep,seen),get(dependencies,key,()))))
    for record in inventory
        any(in(selected),record.members) && continue
        any(depends_on_fresh,record.dependencies) && throw(ArgumentError(
            "BRM resample_covariates: include dependent modeled declaration $(record.members); retaining its fitted missing cells while refreshing its conditioning covariates is unsupported"))
    end
    selected
end

# Promote a complete observed scalar to the same runtime-value classification
# as an mi() declaration. Its original observed values still fit the anchors.
_sb_covariate_value(x, selected) = x
_sb_covariate_value(x::Tuple, selected) = map(v -> _sb_covariate_value(v,selected),x)
_sb_covariate_value(x::AbstractArray, selected) = map(v -> _sb_covariate_value(v,selected),x)
function _sb_covariate_value(x::NamedColumn, selected)
    if name(x) in selected && parent(x) isa DataColumn
        raw = parent(parent(x))
        if raw isa AbstractVector{<:Real} && !(Missing <: eltype(raw))
            return NamedColumn(name(x),DataColumn(Union{Missing,eltype(raw)}[raw...]))
        end
    end
    NamedColumn(name(x),_sb_covariate_value(parent(x),selected))
end
function _sb_covariate_value(x::ExprColumn, selected)
    args = map(v -> _sb_covariate_value(v,selected),getargs(x))
    if getf(x) === (~)
        lhs, rhs = args
        if lhs isa NamedColumn && name(lhs) in selected && parent(lhs) isa DataColumn
            args = (ExprColumn(mi,lhs;_brm_population_value=Val(:fresh_covariate)),rhs)
        elseif lhs isa ExprColumn && getf(lhs)===mi && name(only(getargs(lhs))) in selected
            args = (ExprColumn(mi,only(getargs(lhs));_brm_population_value=Val(:fresh_covariate)),rhs)
        end
    end
    kwargs = getkwargs(x)
    converted = NamedTuple{keys(kwargs)}(
        map(v -> _sb_covariate_value(v,selected),values(kwargs)))
    ExprColumn(getf(x),args...;converted...)
end
function _sb_covariate_value(x::JointResponseColumn, selected)
    JointResponseColumn(map(c -> _sb_covariate_value(c,selected),joint_response_columns(x)),
        x.impute || any(in(selected),joint_response_names(x)))
end
function _sb_covariate_brmi(brmi, selected)
    operations = brmi.operations
    BRMI(NamedTuple{keys(operations)}(
        map(v -> _sb_covariate_value(v,selected),values(operations))))
end

function _sb_fresh_covariate_inputs!(data, key, members, family, rows)
    nkey, keep = Symbol(:brm_fresh_,key,:_n), Symbol(:brm_fresh_,key,:_keep)
    any(haskey(data,k) for k in (nkey,keep)) && error(
        "BRM resample_covariates: reserved prediction input for $(members) collides with data")
    data[nkey] = StanBlocks.stan.maybecv(nkey,rows)
    data[keep] = 0.0
    _sb_record_preproc!(data,nkey,PreprocEntry(:fresh_covariate,
        (; members, family, rows, keep),members,true))
    nkey,keep
end

function _sb_emit_fresh_scalar!(stmts,data,key,plan,rhs)
    translated = Any[]
    _sb_likelihood!(translated,key,rhs,data)
    call = only(translated).args[3]
    Meta.isexpr(call,:call) || error("BRM resample_covariates: expected an elementwise law")
    nkey,keep = _sb_fresh_covariate_inputs!(data,key,(plan.source,),getf(rhs),length(plan.values))
    # This zero-density observation retains source posterior carriers through
    # the unconditioned reachability pass, even for unused joint members.
    push!(stmts,Expr(:call,:~,keep,Expr(:call,:dummy,call.args[2:end]...)))
    fresh = Expr(:call,call.args[1],Expr(:parameters,Expr(:kw,:n,nkey)),call.args[2:end]...)
    push!(stmts,Expr(:call,:~,plan.source,fresh))
    nothing
end

function _sb_emit_fresh_joint!(stmts,data,key,lhs,rhs)
    members = joint_response_names(lhs);K=length(members)
    means,factor = getargs(rhs,2)
    length(means)==K || error("BRM resample_covariates: joint mean/member dimensions differ")
    rows = length(parent(parent(first(joint_response_columns(lhs)))))
    factor_name = _sb_joint_factor_reference(key,factor,K)
    mean_specs = map(eachindex(members)) do i
        _sb_joint_mean_reference(key,members[i],means[i],data,rows)
    end
    mean_exprs = map(spec -> spec.expression,mean_specs)
    nkey,keep = _sb_fresh_covariate_inputs!(data,key,members,MvNormalCholesky,rows)
    push!(stmts,Expr(:call,:~,keep,Expr(:call,:dummy,mean_exprs...,factor_name)))
    cells = [Symbol(key,:_fresh_mean_,i) for i in 1:K]
    value = Symbol(key,:_fresh_value)
    result = Symbol(key,:_fresh_rows)
    sampling = Expr(:call,:~,Expr(:(::),value,Expr(:ref,:vector,K)),
        Expr(:call,:multi_normal_cholesky,Expr(:vect,cells...),factor_name))
    body = Expr(:block,sampling,value)
    platecall = Expr(:call,:plate,Expr(:parameters,Expr(:kw,:outer,Expr(:tuple,nkey))),mean_exprs...)
    draw = Expr(:do,platecall,Expr(:->,Expr(:tuple,cells...),body))
    push!(stmts,Expr(:call,:~,result,draw))
    push!(stmts,Expr(:(=),key,Expr(:call,:to_vector,result)))
    delete!(data,_joint_response_data_key(lhs))
    nothing
end

function _sb_existing_covariates(preproc)
    selected = Set{Symbol}()
    for entry in values(preproc)
        entry.kind === :fresh_covariate && union!(selected,entry.const_.members)
    end
    selected
end

function _sb_covariate_resample_groups(preproc)
    groups = Set{Symbol}()
    for entry in values(preproc)
        entry.kind === :fresh_covariate || continue
        union!(groups,get(entry.const_,:groups,()))
    end
    groups
end

function _sb_record_covariate_groups(sb,groups)
    preproc = Dict(sb.preproc)
    for (key,entry) in preproc
        entry.kind === :fresh_covariate || continue
        preproc[key] = PreprocEntry(entry.kind,
            merge(entry.const_,(; groups=Tuple(sort!(collect(groups))))),entry.raw_ref,entry.dim_coupled)
    end
    SBBRMI(sb.parent,sb.model,sb.data,preproc,copy(sb.held_out),sb.bindings)
end

function _sb_reprocess_covariates(sb,new_df,selected,groups,freeze)
    freeze || throw(ArgumentError(
        "BRM covariate population prediction requires frozen observed training anchors"))
    isempty(total_effect_blocks(sb)) || throw(ArgumentError(
        "BRM covariate resampling currently requires total_groups=() in the fitted artifact"))
    isempty(s2z_effect_blocks(sb)) || throw(ArgumentError(
        "BRM covariate resampling does not support S2Z coordinate transport"))
    previous = _sb_existing_covariates(sb.preproc)
    old_groups = _sb_covariate_resample_groups(sb.preproc)
    baseline = SBBRMI(sb.parent;mod=sb.model.mod,total_groups=(),held_out=sb.held_out,
        cv_groups=old_groups,resample_covariates=previous,_frozen_preproc=sb.preproc)
    isempty(old_groups) || (baseline=_sb_mark_resample_groups(baseline,old_groups))
    stan_code(baseline) == stan_code(sb) || throw(ArgumentError(
        "BRM covariate resampling requires the fitted non-centered, non-CV geometry; constructor-only geometry cannot be inferred"))
    # Complete observed covariates acquire runtime transforms only in this
    # prediction mode. Fit those transforms on the original training values.
    training = SBBRMI(_sb_covariate_brmi(sb.parent,selected);
        mod=sb.model.mod,total_groups=(),held_out=sb.held_out,_frozen_preproc=sb.preproc)
    frozen = merge(Dict(sb.preproc),training.preproc)
    rebound = _sb_rebind_brmi(sb.parent,new_df)
    target = SBBRMI(rebound;mod=sb.model.mod,total_groups=(),cv_groups=groups,
        held_out=sb.held_out,resample_covariates=selected,_frozen_preproc=frozen)
    for (key,entry) in sb.preproc
        entry.kind in (:missing_response,:joint_missing_response) || continue
        sources = entry.raw_ref isa Symbol ? (entry.raw_ref,) : Tuple(entry.raw_ref)
        any(in(selected),sources) && continue
        fresh = get(target.preproc,key,nothing)
        isnothing(fresh) && error("BRM covariate replay lost fitted missing-response provenance `$key`")
        entry.const_ == fresh.const_ || error(
            "BRM covariate replay requires the same fitted missing-row positions for unselected $(sources)")
    end
    isempty(groups) || (_sb_assert_cv_reemission(target,groups);target=_sb_mark_resample_groups(target,groups))
    _sb_record_covariate_groups(target,groups)
end
