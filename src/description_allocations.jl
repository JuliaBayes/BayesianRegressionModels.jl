# Allocation facts are bound to the emitter's selected parameter identities.
# The two R2D2 geometries deliberately retain their different scale formulas.
_brmd_allocation_scale_math(id)= "s_{"*join((_brmd_escape(x) for x in (id[2],id[4:end]...)),",")*"}"

function _brmd_allocation_notation(d,notation,labels,texts)
    result=NamedTuple[notation...]
    function add(key,symbol,meaning)
        isnothing(key) && return
        id=haskey(d.plan.data,key) ? (:prepared_data,key) : (:parameter,key)
        supplied=get(labels,id,NamedTuple())
        supplied isa AbstractString && (supplied=(;meaning=String(supplied)))
        filter!(n->n.name!=id,result)
        push!(result,merge((;name=id,symbol,meaning,axis=:scalar),supplied))
    end
    for (_,b) in sort!(collect(d.plan.bindings);by=p->string(first(p)))
        scheme=get(b,:prior_scheme,(;kind=:ordinary))
        if scheme.kind===:r2d2
            sub=_brmd_escape(b.logical)
            add(scheme.names.r2_name,"R^2_{"*sub*"}",_brmd_text(texts,:r2_meaning))
            add(scheme.names.phi_name,"\\phi_{"*sub*"}",_brmd_text(texts,:simplex_shares_meaning))
            add(scheme.names.tau_name,"T_{"*sub*"}",_brmd_text(texts,:total_scale_meaning))
        end
        allocation=get(b,:allocation_scheme,nothing)
        isnothing(allocation) && continue
        block=only(n for n in result if n.name==allocation.block)
        sub=block.symbol*","*string(allocation.budget)
        add(allocation.r2_name,"R^2_{"*sub*"}",_brmd_text(texts,:budget_r2_meaning))
        add(allocation.phi_name,"\\phi_{"*sub*"}",_brmd_text(texts,:budget_simplex_meaning))
    end
    Tuple(result)
end

function _brmd_allocation_reference(d,key)
    isnothing(key) && return nothing
    BRMDescriptionReference(key,:scalar,haskey(d.plan.data,key) ? (:prepared_data,key) : (:parameter,key))
end

# Only read authored prepared assignments. Never evaluate the expression.
function _brmd_prepared_rhs(d,x,seen=())
    x isa Symbol || return x
    x in seen && return x
    body=d.plan.model.model
    body isa Expr && body.head===:block || return x
    matches=filter(s->s isa Expr && s.head===:(=) && first(s.args)===x,body.args)
    length(matches)==1 || return x
    _brmd_prepared_rhs(d,last(only(matches).args),(seen...,x))
end

function _brmd_allocation_components(d,priors,constants,notation,texts,groups)
    roots=BRMDescriptionComponent[]
    for (_,binding) in sort!(collect(d.plan.bindings);by=p->string(first(p)))
        scheme=get(binding,:prior_scheme,(;kind=:ordinary))
        if scheme.kind!==:ordinary
            env=_brmd_environment(d,binding.logical,:prior_allocation,priors,constants,notation,texts)
            kwargs=if scheme.kind===:r2d2
                (;r2=_brmd_allocation_reference(d,scheme.names.r2_name),
                    phi=_brmd_allocation_reference(d,scheme.names.phi_name),
                    total_scale=_brmd_allocation_reference(d,scheme.names.tau_name),
                    anchor=scheme.spec.tau_bsv)
            elseif scheme.kind===:r2d2m2
                (;r2=_brmd_allocation_reference(d,scheme.spec.r2_name),
                    phi=_brmd_allocation_reference(d,scheme.spec.phi_name),
                    total_scale=_brmd_value(scheme.spec.tau_name,env,(:allocation,binding.logical,:reference)))
            else
                NamedTuple()
            end
            push!(roots,_brmd_component(env,(:allocation,binding.logical),:prior_allocation,
                nothing,(_brmd_snapshot(scheme),),kwargs))
        end
        a=get(binding,:allocation_scheme,nothing)
        isnothing(a) && continue
        env=_brmd_environment(d,a.block,:prior_allocation,priors,constants,notation,texts)
        kwargs=(;r2=_brmd_allocation_reference(d,a.r2_name),phi=_brmd_allocation_reference(d,a.phi_name))
        push!(roots,_brmd_component(env,(:allocation,a.block...,a.budget),:prior_allocation,
            nothing,(_brmd_snapshot(a),),kwargs))
    end
    for group in groups
        group.block.family in (:ranef_intercept_r2d2,:ranef_correlated_r2d2,:ranef_correlated_draws_r2d2) || continue
        env=_brmd_environment(d,group.key,:prior_allocation,priors,constants,notation,texts)
        raw=get(group.declaration.keywords,:tau,get(group.declaration.keywords,:scale,nothing))
        isnothing(raw) && continue
        expression=_brmd_value(_brmd_prepared_rhs(d,raw),env,(:allocation,group.key...,:scale_expression))
        push!(roots,_brmd_component(env,(:allocation,group.key...,:derived_scales),:prior_allocation,
            nothing,((;kind=:derived_group_scales,block=group.key,n_terms=group.block.n_terms,margins=group.margins),expression)))
    end
    Tuple(roots)
end

function _brmd_builtin_kind(::Val{:prior_allocation},c)
    scheme=first(c.arguments)
    scheme.kind===:derived_group_scales && return BRMDescriptionFragment(
        prose=(_brmd_text(c,:derived_block_scales;margins=Tuple((m.predictor,m.coefficient) for m in scheme.margins)),),
        equations=("\\mathrm{SD}_{"*_brmd_block_math(c,scheme.block)*"}="*brm_description_math(c,c.arguments[2]),),covers=(c.id,))
    scheme.kind in (:r2d2,:r2d2m2) || return nothing
    equations=String[]; prose=String[]
    r2=c.keywords.r2; phi=c.keywords.phi
    for ref in (r2,phi)
        isnothing(ref) && continue
        p=brm_description_prior(c,ref.logical)
        push!(equations,brm_description_math(c,ref)*"\\sim"*_brmd_distribution_math(c,p.distribution))
    end
    if haskey(scheme,:spec)
        spec=scheme.spec
        total=brm_description_math(c,c.keywords.total_scale)
        if scheme.kind===:r2d2
            push!(prose,_brmd_text(c,:whole_predictor_r2d2))
            isnothing(c.keywords.anchor) || push!(equations,total*"="*string(c.keywords.anchor))
            if isnothing(r2)
                push!(prose,_brmd_text(c,:no_allocated_columns))
            end
        else
            push!(prose,_brmd_text(c,:joint_block_allocation))
        end
        for (label,index) in zip(spec.labels,spec.share_idx)
            index>0 || continue
            sub=join((_brmd_escape(x) for x in (c.provenance.owner,label)),",")
            variance="V_{"*sub*"}"
            scale=brm_description_math(c,BRMDescriptionReference(:allocated_sd,:scalar,
                (:allocation,c.provenance.owner,:population_scale,label)))
            push!(equations,scale*"="*total*"\\sqrt{\\frac{{"*brm_description_math(c,phi)*"}_{"*string(index)*"}"*brm_description_math(c,r2)*"}{"*variance*"}}")
            push!(equations,variance*"=\\frac{1}{n-1}\\sum_{j=1}^{n}(X_{"*sub*",j}-\\bar X_{"*sub*"})^2")
        end
        cats=get(spec,:cat_lookup,())
        for (_,cat) in cats
            # Logical contrast identities come from the effective prior inventory.
            for p in c.priors
                p.source.kind===:conditional && get(p.source,:allocation,nothing)==c.id &&
                    get(p.source,:share,0) in cat.phi_start:(cat.phi_start+cat.n_contrasts-1) || continue
                scale=brm_description_math(c,p.distribution.arguments[2])
                index=p.source.share
                logical=p.distribution.arguments[2].logical
                sub=join((_brmd_escape(x) for x in (logical[2],logical[4:end]...)),",")
                variance="V_{"*sub*"}"
                push!(equations,scale*"="*total*"\\sqrt{\\frac{{"*brm_description_math(c,phi)*"}_{"*string(index)*"}"*brm_description_math(c,r2)*"}{"*variance*"}}")
                push!(equations,variance*"=\\frac{m(n-m)}{n(n-1)}")
            end
        end
        push!(prose,_brmd_text(c,:allocation_variance))
        spec.n_shares==1 && push!(prose,_brmd_text(c,:degenerate_simplex))
    else
        push!(prose,_brmd_text(c,:r2d2m2_budget;n_phi=scheme.n_phi))
        scheme.n_phi==1 && push!(prose,_brmd_text(c,:degenerate_simplex))
    end
    BRMDescriptionFragment(prose=Tuple(prose),equations=Tuple(equations),covers=(c.id,))
end
