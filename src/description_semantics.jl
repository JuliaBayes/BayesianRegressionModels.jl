# Snapshots expose semantic facts without leaking the mutable compiler carriers.
_brmd_snapshot(x) = isbitstype(typeof(x)) ? x :
    (; (key => _brmd_snapshot(getfield(x,key)) for key in fieldnames(typeof(x)))...)
_brmd_snapshot(x::Union{Function,Type,Module,Symbol,Nothing,AbstractString}) = x
_brmd_snapshot(x::Union{StanBlocks.ValueUDF,StanBlocks.ValueFamily,StanBlocks.SlicModel}) = x
_brmd_snapshot(x::Tuple) = map(_brmd_snapshot,x)
_brmd_snapshot(x::NamedColumn) = BRMDescriptionReference(name(x), :observation)
_brmd_snapshot(x::ExprColumn) = (; callable=getf(x),
    arguments=map(_brmd_snapshot,getargs(x)), keywords=map(_brmd_snapshot,getkwargs(x)))
_brmd_snapshot(x::AbstractArray) = Tuple(_brmd_snapshot(v) for v in x)
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
function _brmd_value(x::Symbol, env, _id)
    parameters=filter(b->b.name===x && b.role===:parameter &&
        length(get(b,:path,(x,)))==1,env.bindings)
    length(parameters)==1 && return only(parameters).value
    x in keys(env.references) && return BRMDescriptionReference(x,env.references[x])
    any(b->b.name===x,env.bindings) && return BRMDescriptionReference(x,:local)
    bound=_brmd_binding(x,env.mod)
    bound isa Union{Function,Type,StanBlocks.ValueUDF,StanBlocks.ValueFamily,StanBlocks.SlicModel} && return bound
    BRMDescriptionReference(x, get(env.references, x, :local))
end
_brmd_value(x::NamedColumn, env, _id) =
    BRMDescriptionReference(name(x), get(env.references, name(x), :observation))
_brmd_value(x, _env, _id) = _brmd_snapshot(x)
_brmd_value(xs::Union{Tuple,AbstractArray}, env, id) =
    Tuple(_brmd_value(x, env, (id..., i)) for (i,x) in enumerate(xs))
_brmd_value(xs::NamedTuple, env, id) = NamedTuple{keys(xs)}(
    Tuple(_brmd_value(x, env, (id..., k)) for (k,x) in pairs(xs)))

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
        public=if value isa Symbol && haskey(env.descriptor.plan.data,value) &&
                  !(value in env.descriptor.columns)
            _brmd_snapshot(env.descriptor.plan.data[value])
        else
            _brmd_value(value,env,(id...,:binding,name))
        end
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
    # Preserve authored fixed/deterministic bindings without evaluating code.
    # Literal assignments expose their value; composed RHSs expose the same
    # public semantic representation used by ordinary arguments.
    if model.model isa Expr && model.model.head===:block
        for stmt in model.model.args
            stmt isa Expr && stmt.head===:(=) && first(stmt.args) isa Symbol || continue
            name,rhs=stmt.args
            value=_brmd_value(rhs,subenv,(id...,:binding,name))
            push!(bindings,(;name,role=rhs isa Union{Number,AbstractString,QuoteNode} ? :constant : :deterministic,
                value,prior_ids=(),path=(name,)))
            subenv=merge(subenv,(;bindings=Tuple(bindings)))
        end
    end
    body=_brmd_value(model.model,subenv,(id...,:body))
    _brmd_component(subenv,id,:submodel,model,(body,))
end

function _brmd_value(x::ExprColumn{typeof(kernel)},env,id)
    raw=getargs(x)
    lambda=first(raw)
    outer=Tuple(_brmd_value(a,env,(id...,:argument,i)) for (i,a) in enumerate(raw[2:end]))
    params=lambda.args[1] isa Expr && lambda.args[1].head===:tuple ? lambda.args[1].args : (lambda.args[1],)
    length(params)==length(outer) || error("description: kernel argument binding mismatch")
    aliases=Tuple((; name=p,role=:alias,value=v,prior_ids=()) for (p,v) in zip(params,outer))
    localenv=merge(env,(; bindings=(env.bindings...,aliases...)))
    body=_brmd_value(lambda,localenv,(id...,:cell))
    _brmd_component(localenv,id,:call,kernel,(body,outer...),_brmd_value(getkwargs(x),localenv,(id...,:keyword)))
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
            observed=!isempty(aliases) && only(aliases).value isa BRMDescriptionReference &&
                     only(aliases).value.name in env.observed
            heldout=!isempty(aliases) && only(aliases).value isa BRMDescriptionReference &&
                    only(aliases).value.name in env.heldout
            role=observed || heldout || !isempty(aliases) ? :observation : :parameter
            localenv=merge(env,(;provenance=merge(env.provenance,(;owner=lhs,
                observation_role=heldout ? :held_out : observed ? :conditioned : :unconditioned))))
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

function _brmd_constants(plan)
    Tuple((; input=key, kind=entry.kind, source=_brmd_snapshot(entry.raw_ref),
             value=_brmd_snapshot(entry.const_), dimension_coupled=entry.dim_coupled)
          for (key,entry) in sort!(collect(plan.preproc); by=p -> string(first(p))))
end

function _brmd_environment(d, owner, kind, priors, constants, notation=())
    refs = Dict{Symbol,Symbol}(key => :observation for key in d.columns)
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
    observed = any(i -> i.column === owner && i.observed && !i.held_out,d.inputs)
    held_out = owner in d.plan.held_out || any(i -> i.column === owner && i.held_out,d.inputs)
    observation_role = held_out ? :held_out : observed ? :conditioned : :unconditioned
    population=filter(b->b.role===:population_effect && b.logical===owner,collect(values(d.plan.bindings)))
    design_columns=length(population)==1 ? get(only(population),:design_columns,nothing) : nothing
    provenance = (; model_id=d.id, owner, declaration=kind, observation_role,
        design_columns=isnothing(design_columns) ? nothing : map(c->_brmd_design_column(d,c),design_columns))
    bindings=Tuple((;name=last(p.id),role=:parameter,
                    value=BRMDescriptionReference(last(p.id),:scalar,p.id),prior_ids=(p.id,))
                   for p in priors if last(p.id) isa Symbol &&
                   owner in p.id && first(p.id) in (:parameter,:kernel))
    (; mod=d.plan.model.mod, references=refs, axes, priors, constants, provenance,
       notation,bindings,outputs=public_outputs,descriptor=d,
       observed=Set(i.column for i in d.inputs if i.observed && !i.held_out),
       heldout=union(d.plan.held_out,Set(i.column for i in d.inputs if i.held_out)))
end

function _brmd_design_column(d,c)
    p=c.preprocess
    prepared=get(d.plan.preproc,c.label,nothing)
    preprocess=if isnothing(p)
        nothing
    else
        (;kind=isnothing(prepared) ? p.kind : prepared.kind,
          const_=_brmd_snapshot(isnothing(prepared) ? p.const_ : prepared.const_),
          raw_ref=_brmd_snapshot(isnothing(prepared) ? p.raw_ref : prepared.raw_ref),
          dependencies=Tuple(_brmd_design_column(d,dep) for dep in p.dependencies))
    end
    (;label=c.label,source=c.source,effect_addresses=c.effect_addresses,
      effect_block=c.effect_block,preprocess)
end

function _brmd_components(d, priors, constants, notation=())
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
        env = _brmd_environment(d, owner, role, priors, constants, notation)
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
