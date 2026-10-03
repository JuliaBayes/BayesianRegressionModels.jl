# Snapshots expose semantic facts without leaking the mutable compiler carriers.
_brmd_snapshot(x) = isbitstype(typeof(x)) ? x :
    (; (key => _brmd_snapshot(getfield(x,key)) for key in fieldnames(typeof(x)))...)
_brmd_snapshot(x::Union{Function,Type,Module,Symbol,Nothing,AbstractString}) = x
# GlobalRef holds an internal Core.Binding which can point back at the ref.
# Retain this static identity; never reflect through Julia's binding machinery.
_brmd_snapshot(x::GlobalRef) = x
_brmd_snapshot(x::CA.CategoricalValue) = _brmd_snapshot(CA.unwrap(x))
_brmd_snapshot(x::Union{StanBlocks.ValueUDF,StanBlocks.ValueFamily,StanBlocks.SlicModel}) = x
_brmd_snapshot(x::Tuple) = map(_brmd_snapshot,x)
_brmd_snapshot(x::NamedColumn) = BRMDescriptionReference(name(x), :observation)
_brmd_snapshot(x::ExprColumn) = (; callable=getf(x),
    arguments=map(_brmd_snapshot,getargs(x)), keywords=map(_brmd_snapshot,getkwargs(x)))
_brmd_snapshot(x::AbstractArray{T,N}) where {T,N} = N==1 ?
    Tuple(_brmd_snapshot(v) for v in x) : (;size=size(x),values=Tuple(_brmd_snapshot(v) for v in x))
_brmd_snapshot(x::NamedTuple) = map(_brmd_snapshot, x)
_brmd_snapshot(x::AbstractDict) = Tuple(
    key => _brmd_snapshot(x[key]) for key in sort!(collect(keys(x)); by=string))
_brmd_snapshot(x::Expr) = (; syntax=x.head, arguments=Tuple(
    _brmd_snapshot(a) for a in x.args if !(a isa LineNumberNode)))

function _brmd_children(x)
    nodes = BRMDescriptionComponent[]
    _brmd_children!(nodes, x)
    Tuple(nodes)
end
_brmd_children!(nodes, x::BRMDescriptionComponent) = push!(nodes, x)
_brmd_children!(_nodes, _x) = nothing
_brmd_children!(nodes, xs::Tuple) = foreach(x -> _brmd_children!(nodes, x), xs)
_brmd_children!(nodes, xs::NamedTuple) = _brmd_children!(nodes, values(xs))

function _brmd_component(env, id, kind, callable, args, kwargs=NamedTuple(); outputs=env.outputs,extra_children=())
    BRMDescriptionComponent(id, kind, callable, args, kwargs, env.axes,
        outputs, env.priors, env.constants, (_brmd_children((args, values(kwargs)))...,extra_children...),
        merge(env.provenance, (; path=id)), env.notation, env.bindings)
end

_brmd_value(x::Number, _env, _id) = x
_brmd_value(x::AbstractString, _env, _id) = String(x)
_brmd_value(x::QuoteNode, _env, _id) = _brmd_snapshot(x.value)
_brmd_value(x::LineNumberNode, _env, _id) = nothing
_brmd_value(x::BRMDescriptionReference,_env,_id)=x

# Julia parses a string immediately before a declaration as Core.@doc.
# Documentation does not change the declared value or introduce a model call.
_brmd_documented(x)=x isa Expr && x.head===:macrocall &&
    first(x.args)==GlobalRef(Core,Symbol("@doc")) ? last(x.args) : x
function _brmd_value(x::GlobalRef,env,id)
    bound=_brmd_binding(x,env.mod)
    bound===x ? _brmd_snapshot(x) : _brmd_value(bound,env,id)
end
function _brmd_value(x::Symbol, env, _id)
    parameters=filter(b->b.name===x && b.role===:parameter &&
        length(get(b,:path,(x,)))==1,env.bindings)
    length(parameters)==1 && return only(parameters).value
    if x in keys(env.references)
        axis=env.references[x]
        logical=axis===:cell ? (:cell,env.provenance.cell_owner,x) : x
        return BRMDescriptionReference(x,axis,logical)
    end
    any(b->b.name===x,env.bindings) && return BRMDescriptionReference(x,:local)
    bound=_brmd_binding(x,env.mod)
    bound isa Union{Function,Type,StanBlocks.ValueUDF,StanBlocks.ValueFamily,StanBlocks.SlicModel} && return bound
    BRMDescriptionReference(x, get(env.references, x, :local))
end
_brmd_value(x::NamedColumn, env, _id) =
    BRMDescriptionReference(name(x), get(env.references, name(x), :observation))
_brmd_value(x, _env, _id) = _brmd_snapshot(x)
function _brmd_value(x::MultiMembershipTerm,env,id)
    args=_brmd_value(x.groups,env,(id...,:memberships))
    kwargs=(;weights=_brmd_value(x.weights,env,(id...,:weights)),normalize=x.normalize)
    _brmd_component(env,id,:call,mm,args,kwargs)
end
_brmd_value(xs::Union{Tuple,AbstractArray}, env, id) =
    Tuple(_brmd_value(x, env, (id..., i)) for (i,x) in enumerate(xs))
_brmd_value(xs::NamedTuple, env, id) = NamedTuple{keys(xs)}(
    Tuple(_brmd_value(x, env, (id..., k)) for (k,x) in pairs(xs)))
_brmd_value(x::JointResponseColumn,env,id) =
    (;columns=_brmd_value(joint_response_columns(x),env,(id...,:columns)),impute=x.impute)

function _brmd_value(x::ExprColumn, env, id)
    args = Tuple(_brmd_value(x, env, (id..., :argument, i))
                 for (i,x) in enumerate(getargs(x)))
    kwargs = _brmd_value(getkwargs(x), env, (id..., :keyword))
    included=()
    if last(id)===:expression
        sites=filter(p->p.target===env.provenance.owner && isempty(p.context),env.descriptor.plan.declarations)
        if length(sites)==1
            site=only(sites)
            f=_brmd_binding(site.family,env.mod)
            if f isa StanBlocks.SlicModel && f.mod !== @__MODULE__
                included=(_brmd_included_model(env,f,site.keywords,(id...,:included),(:parameter,site.target)),)
                env=merge(env,(;bindings=only(included).bindings))
            end
        end
    end
    _brmd_component(env, id, :call, getf(x), args, kwargs;extra_children=included)
end

function _brmd_included_model(env,model,kwargs,id,path)
    bindings=NamedTuple[]
    for (name,value) in sort!(collect(pairs(merge(model.data,Dict(pairs(kwargs)))));by=p->string(first(p)))
        public=_brmd_included_input(value,env,(id...,:binding,name))
        public=_brmd_resolve_alias(public,env.bindings)
        push!(bindings,(;name,role=:alias,value=public,prior_ids=(),path=(name,)))
    end
    parameters=Dict{Tuple,Vector{Tuple}}()
    for p in env.priors
        logical=get(p.source,:binding_id,p.id)
        length(logical)>length(path) && logical[1:length(path)]==path || continue
        push!(get!(parameters,logical,Tuple[]),p.id)
    end
    for (logical,ids) in sort!(collect(parameters);by=p->string(first(p)))
        localpath=logical[length(path)+1:end]
        push!(bindings,(;name=last(logical),role=:parameter,
            value=BRMDescriptionReference(last(logical),:scalar,logical),prior_ids=Tuple(ids),path=localpath))
    end
    subenv=merge(env,(;mod=model.mod,bindings=Tuple(bindings),included_path=path))
    body=model.model
    if body isa Expr && body.head===:block
        supplied=Set(b.name for b in bindings if b.role===:alias)
        # Fixed declarations are defaults when the included call supplies
        # that name. Sampling statements still contribute their conditioned
        # density and must remain in the semantic body.
        statements=filter(body.args) do authored
            stmt=_brmd_documented(authored)
            !(stmt isa Expr && stmt.head===:(=) &&
                first(stmt.args) isa Symbol && first(stmt.args) in supplied)
        end
        body=Expr(:block,statements...)
    end
    # Preserve authored fixed/deterministic bindings without evaluating code.
    # Literal assignments expose their value; composed RHSs expose the same
    # public semantic representation used by ordinary arguments.
    if body isa Expr && body.head===:block
        for authored in body.args
            stmt=_brmd_documented(authored)
            stmt isa Expr && stmt.head===:(=) && first(stmt.args) isa Symbol || continue
            name,rhs=stmt.args
            value=_brmd_value(rhs,subenv,(id...,:binding,name))
            push!(bindings,(;name,role=rhs isa Union{Number,AbstractString,QuoteNode} ? :constant : :deterministic,
                value,prior_ids=(),path=(name,)))
            subenv=merge(subenv,(;bindings=Tuple(bindings)))
        end
    end
    publicbody=_brmd_value(body,subenv,(id...,:body))
    _brmd_component(subenv,id,:submodel,model,(publicbody,))
end

function _brmd_included_input(value,env,id)
    data=env.descriptor.plan.data
    key=value isa Symbol && haskey(data,value) ? value : nothing
    if isnothing(key) && value isa AbstractArray
        matches=sort!([k for (k,v) in pairs(data) if v===value];by=string)
        isempty(matches) || (key=first(matches))
    end
    if !isnothing(key)
        prepared=data[key]
        if prepared isa AbstractArray
            axis=get(env.references,key,eltype(prepared)<:AbstractArray ? :ragged : :observation)
            logical=key in env.descriptor.columns ? key : (:prepared_data,key)
            return BRMDescriptionReference(key,axis,logical)
        end
        return _brmd_snapshot(prepared)
    end
    _brmd_value(value,env,id)
end

_brmd_resolve_alias(x,_bindings,_seen=())=x
function _brmd_resolve_alias(x::BRMDescriptionReference,bindings,seen=())
    x.logical isa Tuple && return x
    x.name in seen && return x
    matches=filter(b->b.name===x.name && b.role in (:alias,:constant,:deterministic),bindings)
    length(matches)==1 || return x
    value=only(matches).value
    value isa BRMDescriptionComponent && return x
    _brmd_resolve_alias(value,bindings,(seen...,x.name))
end

function _brmd_value(x::ExprColumn{typeof(kernel)},env,id)
    raw=getargs(x)
    lambda=first(raw)
    outer=Tuple(_brmd_value(a,env,(id...,:argument,i)) for (i,a) in enumerate(raw[2:end]))
    params=lambda.args[1] isa Expr && lambda.args[1].head===:tuple ? lambda.args[1].args : (lambda.args[1],)
    length(params)==length(outer) || error("description: kernel argument binding mismatch")
    aliases=Tuple((; name=p,role=:alias,value=v,prior_ids=()) for (p,v) in zip(params,outer))
    names=Symbol[]
    _brmd_cell_assignments!(names,lambda.args[2])
    refs=copy(env.references)
    foreach(n->refs[n]=:cell,names)
    localenv=merge(env,(; references=refs,bindings=(env.bindings...,aliases...),
        provenance=merge(env.provenance,(;cell_owner=env.provenance.owner,cell_names=Tuple(unique(names))))))
    body=_brmd_value(lambda,localenv,(id...,:cell))
    _brmd_component(localenv,id,:call,kernel,(body,outer...),_brmd_value(getkwargs(x),localenv,(id...,:keyword)))
end
_brmd_cell_assignments!(_names,_x)=nothing
function _brmd_cell_assignments!(names,x::Expr)
    x.head===:(=) && first(x.args) isa Symbol && push!(names,first(x.args))
    foreach(a->_brmd_cell_assignments!(names,a),x.args)
end

# Resolve only static bindings, never eval an expression or invoke user code.
_brmd_binding(x::GlobalRef, _mod) = isdefined(x.mod, x.name) ? getfield(x.mod, x.name) : x
_brmd_binding(x::QuoteNode, mod) = _brmd_binding(x.value,mod)
_brmd_binding(x, _mod) = x
function _brmd_binding(x::Symbol, mod)
    # SLIC resolves its closed builtin token vocabulary before module names.
    isdefined(StanBlocks.stan.builtin,x) && return getfield(StanBlocks.stan.builtin,x)
    for scope in (mod, @__MODULE__, StanBlocks, Base)
        isdefined(scope, x) && return getfield(scope, x)
    end
    mod!==Main && isdefined(Main,x) && return getfield(Main,x)
    x
end
function _brmd_binding(x::Expr, mod)
    if x.head === :. && length(x.args) == 2
        scope = _brmd_binding(x.args[1], mod)
        key = x.args[2] isa QuoteNode ? x.args[2].value : x.args[2]
        scope isa Module && key isa Symbol && isdefined(scope, key) && return getfield(scope, key)
    end
    deepcopy(x)
end

function _brmd_value(x::Expr, env, id)
    declaration=_brmd_documented(x)
    declaration===x || return _brmd_value(declaration,env,(id...,:documented))
    if x.head===:tuple && length(x.args)==1 &&
       only(x.args) isa Expr && only(x.args).head===:parameters
        fields=only(x.args).args
        names=Tuple(a isa Symbol ? a :
            a isa Expr && a.head in (:kw,:(=)) && first(a.args) isa Symbol ?
            first(a.args) : nothing for a in fields)
        if all(n->n isa Symbol,names) && length(unique(names))==length(names)
            values_=Tuple(_brmd_value(a isa Symbol ? a : last(a.args),env,(id...,:field,n))
                for (n,a) in zip(names,fields))
            return NamedTuple{names}(values_)
        end
    end
    if x.head===Symbol("'") && length(x.args)==1
        args=(_brmd_value(only(x.args),env,(id...,:argument,1)),)
        return _brmd_component(env,id,:call,adjoint,args)
    end
    if x.head===:(=) && first(x.args) isa Symbol
        lhs=first(x.args)
        out=(;name=lhs,logical=lhs,role=:deterministic,kind=:cell_assignment,segments=nothing)
        env=merge(env,(; outputs=(out,)))
    end
    if x.head === :. && length(x.args)==2 && x.args[2] isa Expr && x.args[2].head === :tuple
        args = Tuple(_brmd_value(a,env,(id...,:argument,i))
                     for (i,a) in enumerate(x.args[2].args))
        return _brmd_component(env,id,:call,_brmd_binding(x.args[1],env.mod),args,(; broadcast=true))
    end
    if x.head === :call
        rawargs = Any[]
        rawkeys = Pair{Symbol,Any}[]
        for a in x.args[2:end]
            if a isa Expr && a.head === :parameters
                for kw in a.args
                    kw isa Expr && kw.head === :kw || error("description: unsupported keyword syntax")
                    push!(rawkeys, kw.args[1] => kw.args[2])
                end
            else
                push!(rawargs, a)
            end
        end
        args = Tuple(_brmd_value(a, env, (id..., :argument, i)) for (i,a) in enumerate(rawargs))
        kwargs = (; (k => _brmd_value(v, env, (id..., :keyword, k)) for (k,v) in rawkeys)...)
        head=first(x.args)
        if head===:~ && length(args)==2
            lhs=first(rawargs)
            aliases=filter(b->b.name===lhs && b.role===:alias,env.bindings)
            sources=isempty(aliases) ? () : _brmd_response_sources(only(aliases).value,env.bindings)
            attribution=_brmd_response_provenance(env.descriptor,lhs,sources,env.observed,env.heldout)
            role=attribution.observation_role!==:unconditioned || !isempty(aliases) ? :observation : :parameter
            localenv=merge(env,(;provenance=merge(env.provenance,(;owner=lhs,
                attribution...))))
            if role===:observation
                outputs=Tuple((;name=o.name,logical=o.logical,role=o.role,kind=o.kind,
                    segments=_brmd_snapshot(o.segments)) for o in env.descriptor.outputs
                    if o.logical in sources)
                localenv=merge(localenv,(;outputs))
                # Likelihood hooks and their observation parent share the
                # actual response outputs, including held-out predictive twins.
                args=Tuple(_brmd_value(a,localenv,(id...,:argument,i)) for (i,a) in enumerate(rawargs))
            end
            included=()
            rhs=last(rawargs)
            f=rhs isa Expr && rhs.head===:call ? _brmd_binding(first(rhs.args),env.mod) : nothing
            if f isa StanBlocks.SlicModel
                _,kws=_sb_plan_call_parts(rhs)
                path=(get(env,:included_path,(:parameter,env.provenance.owner))...,lhs)
                included=(_brmd_included_model(localenv,f,kws,(id...,:included),path),)
            end
            return _brmd_component(localenv,id,role,nothing,args,kwargs;extra_children=included)
        end
        # Julia's dotted operator syntax is an elementwise use of the same
        # actual arithmetic callable, including in authored SLIC priors.
        if head isa Symbol && startswith(string(head),".") && length(string(head))>1
            base=Symbol(string(head)[2:end])
            bound=_brmd_binding(base,env.mod)
            if bound isa Function
                return _brmd_component(env,id,:call,bound,args,merge(kwargs,(;broadcast=true)))
            end
        end
        return _brmd_component(env, id, :call, _brmd_binding(head, env.mod), args, kwargs)
    end
    args = Tuple(_brmd_value(a, env, (id..., x.head, i))
                 for (i,a) in enumerate(x.args) if !(a isa LineNumberNode))
    _brmd_component(env, id, :syntax, x.head, args)
end

function _brmd_response_sources(value,bindings,seen=())
    if value isa BRMDescriptionReference
        value.name in seen && return (value.name,)
        aliases=filter(b->b.name===value.name && b.role===:alias,bindings)
        isempty(aliases) && return (value.name,)
        return _brmd_response_sources(only(aliases).value,bindings,(seen...,value.name))
    elseif value isa BRMDescriptionComponent && value.callable===ragged
        # The second argument identifies groups, not observed responses.
        return _brmd_response_sources(first(value.arguments),bindings,seen)
    end
    ()
end

function _brmd_observation_sets(d)
    observed=Set{Symbol}(); heldout=Set{Symbol}(d.plan.held_out)
    for input in d.inputs
        sources=Symbol[input.name]
        isnothing(input.column) || push!(sources,input.column)
        entry=get(d.plan.preproc,input.name,nothing)
        if entry isa PreprocEntry
            if entry.kind in (:joint_response,:joint_missing_response)
                append!(sources,entry.raw_ref)
            elseif entry.kind===:kernel_ragged
                source=first(entry.raw_ref)
                source isa Symbol && push!(sources,source)
            elseif entry.kind===:missing_response
                push!(sources,entry.raw_ref)
            end
        end
        input.held_out && union!(heldout,sources)
        input.observed && !input.held_out && union!(observed,sources)
    end
    observed,heldout
end

function _brmd_response_provenance(d,owner,sources,observed,heldout)
    for (key,entry) in d.plan.preproc
        entry isa PreprocEntry || continue
        joint=entry.kind in (:joint_response,:joint_missing_response)
        scalar=entry.kind===:missing_response
        joint || scalar || continue
        members=joint ? Tuple(entry.raw_ref) : (entry.raw_ref,)
        matches=owner===key || scalar && owner===entry.raw_ref ||
            joint && (owner===get(entry.const_,:completion_key,nothing) ||
                      owner===_joint_response_operation_key(members))
        matches || continue
        missing=length(get(entry.const_,:missing_indices,()))
        total=joint ? length(members)*get(entry.const_,:nobs,length(d.plan.data[key])) :
            missing+length(entry.const_.observed_indices)
        fixed=total-missing
        held=key in heldout || any(s->s in heldout,members)
        bound=key in observed || any(s->s in observed,members)
        role=held ? :held_out : !bound || fixed==0 ? :unconditioned :
            missing>0 ? :partially_observed : :conditioned
        return (;observation_role=role,observation_sources=members,
            observed_entries=fixed,missing_entries=missing,joint_width=joint ? length(members) : 1)
    end
    isempty(sources) && owner isa Symbol && (sources=(owner,))
    role=any(s->s in heldout,sources) ? :held_out :
        any(s->s in observed,sources) ? :conditioned : :unconditioned
    (;observation_role=role,observation_sources=sources)
end

function _brmd_covariance_factor_binding(d,owner)
    d.plan.model.model isa Expr && d.plan.model.model.head===:block || return nothing
    for stmt in d.plan.model.model.args
        stmt isa Expr && stmt.head===:(=) && first(stmt.args)===owner || continue
        rhs=last(stmt.args)
        rhs isa Expr && rhs.head===:call && length(rhs.args)==3 || continue
        _brmd_binding(first(rhs.args),d.plan.model.mod)===StanBlocks.stan.builtin.diag_pre_multiply || continue
        scales,correlation=rhs.args[2:3]
        scales isa Symbol && correlation isa Symbol || continue
        return (;scales=(:parameter,scales),correlation=(:parameter,correlation))
    end
    nothing
end

function _brmd_constants(plan)
    Tuple((; input=key, kind=entry.kind, source=_brmd_snapshot(entry.raw_ref),
             value=_brmd_snapshot(entry.const_), dimension_coupled=entry.dim_coupled)
          for (key,entry) in sort!(collect(plan.preproc); by=p -> string(first(p))))
end

function _brmd_environment(d, owner, kind, priors, constants, notation=();groups=())
    refs = Dict{Symbol,Symbol}(key => get(d.plan.data,key,nothing) isa Number ? :scalar : :observation for key in d.columns)
    for i in d.inputs
        isnothing(i.column) && continue
        i.transform===:kernel_ragged && (refs[i.column]=:ragged)
    end
    for o in d.outputs
        isnothing(o.logical) || (refs[o.logical] = o.role === :parameter ? :scalar : :observation)
    end
    axes = Tuple((; kind=entry.kind, source=_brmd_snapshot(entry.raw_ref),
                   fitted=_brmd_snapshot(entry.const_)) for (_,entry) in
                   sort!(collect(d.plan.preproc);by=p->string(first(p)))
                  if entry.kind in (:group_index,:kernel_ragged,:multi_membership,:kernel_subject_count))
    outputs = [o for o in d.outputs if o.logical === owner]
    public_outputs=Tuple((;name=o.name,logical=o.logical,role=o.role,kind=o.kind,
                           segments=_brmd_snapshot(o.segments)) for o in outputs)
    axes = (axes..., ((; kind=:output, owner, size=o.size, segments=_brmd_snapshot(o.segments))
                       for o in outputs)...)
    observed,heldout=_brmd_observation_sets(d)
    attribution=_brmd_response_provenance(d,owner,(),observed,heldout)
    population=filter(b->b.role===:population_effect && b.logical===owner,collect(values(d.plan.bindings)))
    design_columns=length(population)==1 ? get(only(population),:design_columns,nothing) : nothing
    provenance = (; model_id=d.id, owner, declaration=kind, attribution...,
        covariance_factor=_brmd_covariance_factor_binding(d,owner),
        design_columns=isnothing(design_columns) ? nothing : map(c->_brmd_design_column(d,c),design_columns),
        random_effects=Tuple((;id=g.key,group=g.block.group,shared_id=g.block.id,margins=g.margins) for g in groups))
    bindings=Tuple((;name=last(p.id),role=:parameter,
                    value=BRMDescriptionReference(last(p.id),:scalar,p.id),prior_ids=(p.id,))
                   for p in priors if last(p.id) isa Symbol &&
                   owner in p.id && first(p.id) in (:parameter,:kernel))
    (; mod=d.plan.model.mod, references=refs, axes, priors, constants, provenance,
       notation,bindings,outputs=public_outputs,descriptor=d,
       observed,heldout)
end

function _brmd_design_column(d,c)
    p=c.preprocess
    prepared=get(d.plan.preproc,c.label,nothing)
    preprocess=if isnothing(p) && !isnothing(prepared)
        (;kind=prepared.kind,const_=_brmd_snapshot(prepared.const_),
          raw_ref=_brmd_snapshot(prepared.raw_ref),dependencies=())
    elseif isnothing(p)
        nothing
    else
        (;kind=isnothing(prepared) ? p.kind : prepared.kind,
          const_=_brmd_snapshot(isnothing(prepared) ? p.const_ : prepared.const_),
          raw_ref=_brmd_snapshot(isnothing(prepared) ? p.raw_ref : prepared.raw_ref),
          dependencies=Tuple(_brmd_design_column(d,dep) for dep in p.dependencies))
    end
    (;label=c.label,source=c.source,effect_addresses=c.effect_addresses,
      effect_block=c.effect_block,preprocess,term=_brmd_snapshot(get(c,:term,nothing)))
end

function _brmd_components(d, priors, constants, notation=();groups=())
    program = _brm_prepare_program(d.plan.parent)
    byname = Dict(op.name => op for op in program.operations)
    result = BRMDescriptionComponent[]
    for key in program.order
        op = byname[key]
        op.role === :prior_modifier && continue
        op.expression isa ExprColumn || continue
        args = getargs(op.expression)
        length(args) == 2 || continue
        lhs, rhs = args
        # Prior selectors are resolved in the effective inventory, not rendered
        # as spurious linear predictors.
        lhs isa ExprColumn && getf(lhs) === effect && continue
        owner = something(_brm_lp_logical_name(d.plan.parent, key), key)
        is_observation = any(p -> p.role === :observation && isempty(p.context) && p.target === key,
                             d.plan.declarations)
        role = is_observation ? :observation : op.role
        sites=filter(p->p.target===key && isempty(p.context),d.plan.declarations)
        if length(sites)==1
            f=_brmd_binding(only(sites).family,d.plan.model.mod)
            f isa StanBlocks.SlicModel && f.mod !== (@__MODULE__) && (role=:submodel_output)
        end
        env = _brmd_environment(d, owner, role, priors, constants, notation;groups)
        expr = _brmd_value(rhs, env, (role, owner, :expression))
        outputs = Tuple((; name=o.name, logical=o.logical, role=o.role,
                          kind=o.kind, segments=_brmd_snapshot(o.segments))
                        for o in d.outputs if o.logical === owner)
        push!(result, _brmd_component(env, (role, owner), role, nothing,
              (_brmd_value(lhs, env, (role, owner, :lhs)), expr); outputs))
    end
    Tuple(result)
end

_brm_lp_logical_name(brmi, key) = begin
    op = _named_op(brmi.operations[key])
    isnothing(op) && return nothing
    lhs = first(getargs(op))
    peeled = _peel_lp_lhs(lhs)
    isnothing(peeled) ? nothing : last(peeled)
end
