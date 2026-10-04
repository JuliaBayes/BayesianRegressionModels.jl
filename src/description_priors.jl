# Effective priors share the emitter's selector resolver. Never reinterpret
# selector precedence in a reporting consumer.
function _brmd_prior(d, id, expression, support, source, anchors, notation;bindings=())
    env = _brmd_environment(d, id, :prior, (), _brmd_constants(d.plan), notation)
    isempty(bindings) || (env=merge(env,(;bindings)))
    if expression isa ExprColumn
        support=merge(support,(; (k=>_brmd_snapshot(v) for (k,v) in pairs(getkwargs(expression))
            if k in (:lower,:upper))...))
    end
    value = _brmd_value(expression, env, (:prior, id...))
    if value isa BRMDescriptionComponent && _brmd_law(value.callable)===:uniform
        a=value.arguments
        support=merge((;lower=isempty(a) ? 0.0 : first(a),upper=isempty(a) ? 1.0 : a[2]),support)
    end
    anchor = get(anchors,id,nothing)
    BRMPriorDescription(id, value, support, source,
        isnothing(anchor) ? nothing : String(anchor), env.provenance)
end

_brmd_default_normal() = ExprColumn(Normal,0.0,1.0)
_brmd_support(_f) = NamedTuple()
_brmd_support(f::StanBlocks.ValueFamily) = _brmd_snapshot(f.support)
_brmd_support(::Type{<:Exponential}) = (; lower=0.0)
_brmd_support(::Type{<:LogNormal}) = (; lower=0.0)
_brmd_support(::Type{<:Gamma}) = (; lower=0.0)
_brmd_support(::Type{<:InverseGamma}) = (; lower=0.0)
_brmd_support(::Type{<:Beta}) = (; lower=0.0, upper=1.0)
_brmd_support(::Type{<:Dirichlet}) = (; domain=:simplex)
for (name,support) in ((:exponential,(;lower=0.0)),(:gamma,(;lower=0.0)),(:inv_gamma,(;lower=0.0)),
        (:lognormal,(;lower=0.0)),(:beta,(;lower=0.0,upper=1.0)),
        (:dirichlet,(;domain=:simplex)),(:lkj_corr_cholesky,(;domain=:cholesky_correlation)))
    if isdefined(StanBlocks.stan.builtin,name)
        f=getfield(StanBlocks.stan.builtin,name)
        @eval _brmd_support(::$(typeof(f))) = $support
    end
end

function _brmd_population_priors!(priors,d,anchors,notation)
    overrides = _sb_effect_prior_overrides(d.plan.parent; frozen_preproc=d.plan.preproc,provenance=true)
    for entry in _brm_population_effect_entries(d.plan.parent)
        labels = Tuple(label for output in d.outputs
                       if output.declaration !== nothing && output.declaration.target === entry.block
                       && output.labels !== nothing for label in output.labels)
        labels = Tuple(unique(labels))
        settings = _sb_pop_effect_overrides(overrides,entry.logical)
        binding=get(d.plan.bindings,entry.block,nothing)
        scheme=isnothing(binding) ? (;kind=:ordinary) : get(binding,:prior_scheme,(;kind=:ordinary))
        if scheme.kind===:horseshoe
            site=only(filter(p->p.target===entry.block && isempty(p.context),d.plan.declarations))
            model=_brmd_binding(site.family,d.plan.model.mod)
            values=Dict(k=>_brmd_substitute(v,d.plan.data) for (k,v) in pairs(site.keywords))
            _brmd_submodel_priors!(priors,d,model,values,
                (:population_internal,entry.logical),anchors,notation)
        end
        for (i,label) in enumerate(labels)
            configured = isnothing(settings) ? nothing : settings[i]
            expression = isnothing(configured) ? _brmd_default_normal() : configured.expression
            source = isnothing(configured) ? (; kind=:default) :
                     (; kind=:selector,selector=configured.spelling,specificity=configured.rank)
            if scheme.kind in (:r2d2,:r2d2m2) && scheme.spec.share_idx[i]>0
                expression=ExprColumn(Normal,0.0,BRMDescriptionReference(:allocated_sd,:scalar,
                    (:allocation,entry.logical,:population_scale,label)))
                source=(;kind=:conditional,allocation=(:allocation,entry.logical),share=scheme.spec.share_idx[i])
            end
            push!(priors,_brmd_prior(d,(:population,entry.logical,label),expression,
                _brmd_support(getf(expression)),source,anchors,notation))
        end
        cats = _sb_cat_effect_overrides(overrides,entry.logical)
        for cat in _brm_categorical_effect_entries(d,entry.logical,entry.link)
            cfg = get(cats,cat.emitted,nothing)
            for (i,level) in enumerate(cat.nonreference_levels)
                configured = cfg isa AbstractVector ? cfg[i] : cfg
                expression = isnothing(configured) ? _brmd_default_normal() : configured.expression
                source = (; kind=isnothing(configured) ? :default : :selector,
                    selector=isnothing(configured) ? nothing : configured.spelling,
                    specificity=isnothing(configured) ? nothing : configured.rank,
                    predictor=cat.predictor, level=_brmd_snapshot(level),
                    reference=_brmd_snapshot(cat.reference_level), coding=cat.coding)
                allocated=scheme.kind===:r2d2m2 ? get(scheme.spec.cat_lookup,cat.emitted,nothing) : nothing
                if !isnothing(allocated)
                    expression=ExprColumn(Normal,0.0,BRMDescriptionReference(:allocated_sd,:scalar,
                        (:allocation,entry.logical,:categorical_scale,cat.address,i)))
                    source=merge(source,(;kind=:conditional,allocation=(:allocation,entry.logical),
                        share=allocated.phi_start+i-1))
                end
                push!(priors,_brmd_prior(d,(:population,entry.logical,cat.address,:level,i),
                    expression,_brmd_support(getf(expression)),source,anchors,notation))
            end
        end
    end
end

function _brmd_ranef_metadata(d)
    buckets = _sb_collect_id_buckets(d.plan.parent)
    overrides = _sb_ranef_effect_overrides(d.plan.parent,buckets)
    result = NamedTuple[]
    for (index,block) in enumerate(ranef_blocks(d.plan))
        key = (:random_effect, isnothing(block.id) ? (:independent,index) : block.id, block.group)
        declarations = filter(x -> x.target === block.binding,d.plan.declarations)
        declaration = only(declarations)
        binding=get(d.plan.bindings,block.binding,NamedTuple())
        margins = if isnothing(block.id)
            haskey(binding,:columns) ? Tuple((;predictor=binding.predictor,coefficient=label)
                for label in binding.columns) : Tuple((; predictor=x.logical, coefficient=label)
                  for x in d.outputs if x.declaration !== nothing &&
                      x.declaration.target === block.binding && x.labels !== nothing
                  for label in x.labels)
        else
            Tuple(ranefcoefnames(d.plan.parent,block.id))
        end
        # Independent || blocks are actually emitted as separate one-column
        # blocks. Dimension and membership follow the prepared executable plan.
        cfg = nothing
        if !isnothing(block.id)
            hits = [v for (k,v) in overrides if first(k) === block.id]
            isempty(hits) || (cfg=only(hits))
        end
        push!(result,(; key,block,margins,configuration=cfg,declaration,
            correlated=block.n_terms > 1,
            shared=!isnothing(block.id)))
    end
    Tuple(result)
end

function _brmd_ranef_priors!(priors,d,groups,anchors,notation)
    for group in groups
        cfg = group.configuration
        if !isnothing(group.block.by)
            site=group.declaration
            model=_brmd_binding(site.family,d.plan.model.mod)
            values=Dict(k=>_brmd_substitute(v,d.plan.data) for (k,v) in pairs(site.keywords))
            _brmd_submodel_priors!(priors,d,model,values,group.key,anchors,notation)
            continue
        end
        # R2D2 derives these scales. Its emitted hyperpriors are inventoried
        # below; never substitute an independent half-normal prior for them.
        spec = get(_RANEF_FAMILIES,group.block.family,nothing)
        logscale = group.block.family in (:ranef_intercept,:ranef_intercept_draws,:ranef_intercept_centered)
        derived = group.block.family in (:ranef_intercept_r2d2,
            :ranef_correlated_r2d2,:ranef_correlated_draws_r2d2,
            :ranef_correlated_by,:ranef_correlated_by_draws)
        if logscale
            push!(priors,_brmd_prior(d,(group.key...,:log_scale),_brmd_default_normal(),NamedTuple(),
                (;kind=:default,transformed_quantity=:sd,transform=:exp),anchors,notation))
            push!(priors,_brmd_prior(d,(group.key...,:sd,1),ExprColumn(LogNormal,0.0,1.0),
                (;lower=0.0),(;kind=:induced,from=(group.key...,:log_scale)),anchors,notation))
        elseif !derived
            for i in 1:group.block.n_terms
                expression = isnothing(cfg) ? nothing : cfg.sd_prior[i]
                expression = something(expression,_brmd_default_normal())
                margin = i <= length(group.margins) ? group.margins[i] : (; index=i)
                push!(priors,_brmd_prior(d,(group.key...,:sd,i),expression,
                    merge(_brmd_support(getf(expression)),(; lower=0.0)),
                    (; kind=isnothing(cfg) ? :default : :selector, margin),anchors,notation))
            end
        end
        if group.block.noncentered
            push!(priors,_brmd_prior(d,(group.key...,:standardized_deviations),
                ExprColumn(Normal,0.0,1.0),NamedTuple(),
                (;kind=:generated,dimension=(group.block.n_terms,group.block.n_groups)),anchors,notation))
        else
            push!(priors,_brmd_prior(d,(group.key...,:deviations),
                ExprColumn(MvNormalCholesky,zeros(group.block.n_terms),
                    BRMDescriptionReference(:C,:covariance,(group.key...,:cholesky_scale))),NamedTuple(),
                (;kind=:conditional,dimension=(group.block.n_groups,group.block.n_terms),covariance=group.key),anchors,notation))
        end
        if group.correlated
            eta = isnothing(cfg) ? 1.0 : cfg.lkj_eta
            push!(priors,_brmd_prior(d,(group.key...,:correlation),
                ExprColumn(LKJCholesky,group.block.n_terms,eta),
                (; domain=:cholesky_correlation),
                (; kind=isnothing(cfg) ? :default : :selector),anchors,notation))
        end
    end
end

# Expand BRM-owned SLIC submodels from their authored source. This is neither
# a new model build nor a scan of emitted Stan. It makes generated latent
# priors and hyperpriors visible. Unknown scientific submodels remain opaque
# semantic calls for the public description hook.
_brmd_substitute(x, _values) = x
_brmd_substitute(x::Symbol, values) = get(values,x,x)
_brmd_substitute(x::Expr, values) = Expr(x.head,
    (_brmd_substitute(a,values) for a in x.args)...)
struct _BRMDFlatPrior end
const _brmd_flat_prior = _BRMDFlatPrior()
function _brmd_flat_priors!(priors,d,model,values,path,declarations,anchors,notation)
    sampled=Set(p.target for p in declarations)
    model.model isa Expr && model.model.head===:block || return
    for stmt in model.model.args
        stmt isa Expr && stmt.head===:(::) && first(stmt.args) isa Symbol || continue
        target=first(stmt.args)
        target in sampled && continue
        annotation=last(stmt.args)
        push!(priors,_brmd_prior(d,(path...,target),ExprColumn(_brmd_flat_prior),NamedTuple(),
            (;kind=:flat,proper=false,annotation=_brmd_snapshot(annotation)),anchors,notation))
    end
end
function _brmd_submodel_priors!(priors,d,model,values,path,anchors,notation,stack=())
    model isa StanBlocks.SlicModel || return
    model in stack && error("description: recursive SLIC submodel at $path")
    declarations = GenerativeDeclaration[]
    _sb_plan_collect!(declarations,model.model,Dict{Symbol,Symbol}(k=>k for k in keys(values)),(),Set{Symbol}(),Set{Symbol}(),Set())
    _brmd_flat_priors!(priors,d,model,values,path,declarations,anchors,notation)
    bindings=Tuple((;name=p.target,role=:parameter,path=(p.context...,p.target),
        value=BRMDescriptionReference(p.target,:scalar,(path...,p.context...,p.target)),prior_ids=())
        for p in declarations if p.role===:prior && p.family!==:plate)
    for p in declarations
        p.role === :prior || continue
        p.family===:plate && continue
        f = _brmd_binding(p.family,model.mod)
        args = map(x -> _brmd_substitute(x,values),p.arguments)
        kwargs = map(x -> _brmd_substitute(x,values),p.keywords)
        nested_values = Dict{Symbol,Any}(pairs(kwargs))
        f isa StanBlocks.SlicModel && merge!(nested_values,Dict(pairs(f.data)))
        binding_id = (path...,p.context...,p.target)
        lhs=p.expression.args[2]
        lhs isa Expr && lhs.head===:(::) && (lhs=first(lhs.args))
        indices=lhs isa Expr && lhs.head===:ref ? Tuple(_brmd_snapshot(a) for a in lhs.args[2:end]) : ()
        logical=isempty(indices) ? binding_id : (binding_id...,:index,indices...)
        if f isa StanBlocks.SlicModel
            _brmd_submodel_priors!(priors,d,f,nested_values,logical,anchors,notation,(stack...,model))
        else
            raw = ExprColumn(f,args...;kwargs...)
            support = merge(_brmd_support(f),map(x -> _brmd_substitute(x,values),p.constraints))
            push!(priors,_brmd_prior(d,logical,raw,support,
                (; kind=:generated, family=f, dimension=_brmd_snapshot(map(a->_brmd_substitute(a,values),p.dimension)),binding_id),
                anchors,notation;bindings))
        end
    end
end

function _brmd_priors(d, anchors, notation;groups=_brmd_ranef_metadata(d))
    priors = BRMPriorDescription[]
    _brmd_population_priors!(priors,d,anchors,notation)
    _brmd_ranef_priors!(priors,d,groups,anchors,notation)
    categories=Set(cat.emitted for entry in _brm_population_effect_entries(d.plan.parent)
        for cat in _brm_categorical_effect_entries(d,entry.logical,entry.link))
    for declaration in d.plan.declarations
        role = _brm_declaration_role(declaration,d.plan.bindings)
        role in (:population_effect,:random_effect) && continue
        declaration.target in categories && continue
        # A plate declares a grouped computation, not a distribution on its
        # returned value. Its cell priors are already separate declarations.
        declaration.family === :plate && continue
        f = _brmd_binding(declaration.family,d.plan.model.mod)
        logical = isempty(declaration.context) ? (:parameter,declaration.target) :
            (:kernel,declaration.context...,declaration.target)
        if f isa StanBlocks.SlicModel
            values = Dict{Symbol,Any}(k => _brmd_substitute(v,d.plan.data)
                                     for (k,v) in pairs(declaration.keywords))
            _brmd_submodel_priors!(priors,d,f,values,logical,anchors,notation)
        else
            sampled=any(o->o.role===:parameter && !isnothing(o.declaration) &&
                o.declaration.target===declaration.target &&
                o.declaration.context==declaration.context,d.outputs)
            declaration.role === :prior || sampled || continue
            raw = Expr(:call,declaration.family,Expr(:parameters,
                (Expr(:kw,k,_brmd_substitute(v,d.plan.data)) for (k,v) in pairs(declaration.keywords))...),
                (_brmd_substitute(a,d.plan.data) for a in declaration.arguments)...)
            push!(priors,_brmd_prior(d,logical,raw,merge(_brmd_support(f),declaration.constraints),
                (; kind=:declaration, family=f),anchors,notation))
        end
    end
    Tuple(priors), groups
end
