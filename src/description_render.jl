# User labels are presentation metadata, not inferred units or transforms.
_brmd_escape(s) = replace(string(s), "\\" => "\\textbackslash{}", "_" => "\\_",
    "{" => "\\{", "}" => "\\}", "%" => "\\%", "#" => "\\#", "&" => "\\&",
    "\$" => "\\\$", "^" => "\\textasciicircum{}", "~" => "\\textasciitilde{}")
_brmd_identifier(x) = "\\mathrm{" * _brmd_escape(x) * "}"

"""
    brm_description_symbol(context, name::Symbol)

Return the LaTeX notation bound to an actual logical name. Default identifiers
are escaped. An explicit notation `symbol` is trusted caller-supplied LaTeX.
"""
function brm_description_symbol(context::BRMDescriptionContext, name::Union{Symbol,Tuple})
    fallback=name isa Symbol && name in get(context.provenance,:cell_names,()) ?
        _brmd_identifier(string(context.provenance.cell_owner)*"."*string(name)) : _brmd_identifier(name)
    found = filter(n -> n.name == name, context.notation)
    isempty(found) || return get(only(found), :symbol, fallback)
    fallback
end

"""
    brm_description_math(context, argument)

Render any public argument/reference/component as escaped LaTeX using the
context's notation. Rendering an unknown call does not establish coverage.
"""
function brm_description_math(c::BRMDescriptionContext,x::BRMDescriptionReference)
    x.logical isa Tuple && any(n -> n.name==x.logical,c.notation) &&
        return brm_description_symbol(c,x.logical)
    aliases=filter(b -> x.axis!==:cell && b.name===x.name && b.role in (:alias,:constant),c.bindings)
    if !isempty(aliases)
        value=only(aliases).value
        # Drop this alias before following it. This terminates identity aliases,
        # longer alias cycles and cycles through deterministic expressions.
        remaining=Tuple(b for b in c.bindings if b.name!==x.name)
        localcontext=BRMDescriptionComponent(c.id,c.kind,c.callable,c.arguments,c.keywords,
            c.axes,c.outputs,c.priors,c.fitted_constants,c.children,c.provenance,c.notation,remaining)
        isequal(value,x) || return brm_description_math(localcontext,value)
    end
    any(n->n.name==x.name && haskey(n,:symbol),c.notation) &&
        return brm_description_symbol(c,x.name)
    x.logical isa Symbol && any(n->n.name==(:parameter,x.logical),c.notation) &&
        return brm_description_symbol(c,(:parameter,x.logical))
    x.logical isa Symbol && any(n->n.name==(:prepared_data,x.logical),c.notation) &&
        return brm_description_symbol(c,(:prepared_data,x.logical))
    x.logical isa Tuple && first(x.logical)===:allocation && length(x.logical)>=4 &&
        return _brmd_allocation_scale_math(x.logical)
    x.axis===:covariance && x.logical isa Tuple && last(x.logical)===:cholesky_scale &&
        return "C_{"*_brmd_block_math(c,x.logical[1:end-1])*"}"
    # A block SD margin, as in D=diag(SD); exact-total blocks have one margin.
    x.logical isa Tuple && first(x.logical)===:random_effect && length(x.logical)>=3 &&
        x.logical[end-1]===:sd && return "\\mathrm{SD}_{"*_brmd_block_math(c,x.logical[1:end-2])*
            (x.logical[end]==1 ? "" : ","*string(x.logical[end]))*"}"
    brm_description_symbol(c,x.name)
end

_brmd_call_math(::typeof(StanBlocks.stan.builtin.dims),args,_kwargs,c) =
    "\\operatorname{"*_brmd_text(c,:math_shape)*"}\\left("*only(args)*"\\right)"
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.dims),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:shape_vector),),covers=(c.id,))
_brmd_call_math(::typeof(StanBlocks.stan.builtin.to_vector),args,_kwargs,c) =
    "\\operatorname{vec}_{\\mathrm{"*_brmd_text(c,:math_column_major)*"}}\\left("*only(args)*"\\right)"
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.to_vector),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:vector_conversion),),covers=(c.id,))
function _brmd_call_math(::typeof(StanBlocks.stan.builtin.to_matrix),args,_kwargs,c)
    length(args)==1 && return "\\operatorname{"*_brmd_text(c,:math_matrix)*"}\\left("*only(args)*"\\right)"
    order=length(args)==3 ? "\\mathrm{"*_brmd_text(c,:math_column_major)*"}" :
        last(args)=="0" ? "\\mathrm{"*_brmd_text(c,:math_row_major)*"}" :
        "\\mathrm{"*_brmd_text(c,:math_order)*"}("*last(args)*")"
    "\\operatorname{"*_brmd_text(c,:math_reshape)*"}_{"*args[2]*"\\times "*args[3]*","*order*"}\\left("*first(args)*"\\right)"
end
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.to_matrix),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:matrix_conversion),),covers=(c.id,))
function _brmd_call_math(::typeof(StanBlocks.stan.builtin.rep_matrix),args,_kwargs,_c)
    length(args)==3 && return first(args)*"\\,\\mathbf1_{"*args[2]*"\\times "*args[3]*"}"
    "\\operatorname{rep}_{\\mathrm{matrix}}\\left("*join(args,",")*"\\right)"
end
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.rep_matrix),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:matrix_replication),),covers=(c.id,))
for f in (adjoint,transpose)
    @eval _brmd_call_math(::$(typeof(f)),args,_kwargs,_c) =
        "\\left("*only(args)*"\\right)^{\\mathsf T}"
    @eval _brmd_builtin_call(::$(typeof(f)),c)=BRMDescriptionFragment(
        prose=(_brmd_text(c,:transposition),),covers=(c.id,))
end
function _brmd_call_math(::typeof(range),args,kwargs,c)
    "\\operatorname{range}\\left("*join((args...,
        (_brmd_identifier(k)*"="*brm_description_math(c,v) for (k,v) in pairs(kwargs))...),",")*"\\right)"
end
_brmd_builtin_call(::typeof(range),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:range_grid),),
    equations=("r_k=a+(k-1)d,\\quad k=1,\\ldots,n",),covers=(c.id,))
_brmd_call_math(::typeof(constant),args,_kwargs,_c) =
    "\\operatorname{constant}\\left("*only(args)*"\\right)"
# A constant the emitted model never reads has no fitted record to report.
function _brmd_builtin_call(::typeof(constant),c)
    owner=c.provenance.owner
    fitted=[k.value.value for k in c.fitted_constants if k.kind===:constant && k.input===owner]
    prose=isempty(fitted) ? _brmd_text(c,:constant;owner,expression=only(c.arguments)) :
        _brmd_text(c,:constant_fitted;owner,expression=only(c.arguments),value=only(fitted))
    BRMDescriptionFragment(prose=(prose,),covers=(c.id,))
end
_brmd_call_math(::Colon,args,_kwargs,_c)="\\operatorname{range}\\left("*join(args,",")*"\\right)"
_brmd_builtin_call(::Colon,c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:colon_range),),covers=(c.id,))
brm_description_math(_c::BRMDescriptionContext, x::Number) = string(x)
brm_description_math(_c::BRMDescriptionContext, x::AbstractString) =
    "\\text{" * _brmd_escape(x) * "}"
brm_description_math(c::BRMDescriptionContext, xs::Tuple) =
    "\\left[" * join((brm_description_math(c,x) for x in xs), ", ") * "\\right]"
brm_description_math(_c::BRMDescriptionContext, x::Symbol) = _brmd_identifier(x)
function brm_description_math(c::BRMDescriptionContext,x::NamedTuple)
    haskey(x,:size) && haskey(x,:values) && return "\\operatorname{"*_brmd_text(c,:math_reshape)*"}\\left("*
        brm_description_math(c,x.values)*","*join(x.size,",")*"\\right)"
    all(k->haskey(x,k),(:callable,:arguments,:keywords)) && return _brmd_call_math(x.callable,
        map(a->brm_description_math(c,a),x.arguments),x.keywords,c)
    haskey(x,:columns) && return brm_description_math(c,x.columns)
    "\\left\\{"*join((_brmd_identifier(k)*"="*brm_description_math(c,v) for (k,v) in pairs(x)),";\\,")*"\\right\\}"
end
brm_description_math(_c::BRMDescriptionContext, x) = _brmd_identifier(string(x))
_brmd_callable_name(f::Symbol) = f
_brmd_callable_name(f::Union{Function,Type}) = nameof(f)
_brmd_callable_name(f::Union{StanBlocks.ValueUDF,StanBlocks.ValueFamily}) = f.label
_brmd_callable_name(f) = Symbol(nameof(typeof(f)))
brm_description_math(_c::BRMDescriptionContext,x::Union{Function,Type,StanBlocks.ValueUDF,StanBlocks.ValueFamily,StanBlocks.SlicModel}) =
    _brmd_identifier(_brmd_callable_name(x))

function brm_description_math(c::BRMDescriptionContext, x::BRMDescriptionComponent)
    x.callable===kernel && return _brmd_kernel_math(c,x)
    if x.kind===:syntax
        if x.callable===:ref
            return "{"*brm_description_math(c,first(x.arguments))*"}_{"*
                join((brm_description_math(c,a) for a in x.arguments[2:end]),",")*"}"
        elseif x.callable in (:tuple,:vect)
            return "\\left["*join((brm_description_math(c,a) for a in x.arguments),", ")*"\\right]"
        elseif x.callable===:(=) && first(x.arguments) isa BRMDescriptionReference
            return brm_description_symbol(c,first(x.arguments).name)*"="*brm_description_math(c,last(x.arguments))
        elseif x.callable===:return
            return brm_description_math(c,only(x.arguments))
        elseif x.callable===:. && length(x.arguments)==2
            base,field=x.arguments
            base isa NamedTuple && field isa Symbol && haskey(base,field) &&
                return brm_description_math(c,base[field])
            return brm_description_math(c,base)*"."*brm_description_math(c,field)
        end
    end
    if x.callable===StanBlocks.stan.builtin.maybe_index
        value=first(x.arguments)
        (value isa Number || value isa BRMDescriptionReference && value.axis===:scalar) &&
            return brm_description_math(c,value)
    end
    args = map(a -> brm_description_math(c,a), x.arguments)
    _brmd_call_math(x.callable,args,x.keywords,c)
end
_brmd_call_math(f, args, kwargs, c) = _brmd_identifier(_brmd_callable_name(f)) *
    "\\left(" * join((args..., (string(k) * "=" * brm_description_math(c,v)
         for (k,v) in pairs(kwargs))...), ", ") * "\\right)"
_brmd_call_math(::typeof(+), args, _kwargs, _c) = "\\left(" * join(args," + ") * "\\right)"
_brmd_call_math(::typeof(-), args, _kwargs, _c) = "\\left(" *
    (length(args)==1 ? "-" * only(args) : join(args," - ")) * "\\right)"
_brmd_call_math(::typeof(*), args, _kwargs, _c) = "\\left(" * join(args," \\cdot ") * "\\right)"
_brmd_call_math(::typeof(/), args, _kwargs, _c) = length(args)==2 ?
    "\\frac{" * args[1] * "}{" * args[2] * "}" : join(args,"/")
_brmd_call_math(::typeof(^), args, _kwargs, _c) = "{" * args[1] * "}^{" * args[2] * "}"
_brmd_call_math(::typeof(getindex), args, _kwargs, _c) =
    "{" * args[1] * "}_{" * join(args[2:end], ",") * "}"
_brmd_call_math(::typeof(exp), args, _kwargs, _c) = "\\exp\\left(" * only(args) * "\\right)"
_brmd_call_math(::typeof(log), args, _kwargs, _c) = "\\log\\left(" * only(args) * "\\right)"
_brmd_call_math(::typeof(sqrt), args, _kwargs, _c) = "\\sqrt{" * only(args) * "}"
_brmd_call_math(::typeof(logistic), args, _kwargs, _c) = "\\operatorname{logit}^{-1}\\left(" * only(args) * "\\right)"
_brmd_call_math(::typeof(StanBlocks.stan.builtin.inv_logit),args,_kwargs,_c) =
    "\\frac{1}{1+\\exp\\left(-"*only(args)*"\\right)}"
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.inv_logit),c)=BRMDescriptionFragment(covers=(c.id,))
for f in (StanBlocks.stan.builtin.max,StanBlocks.stan.builtin.min)
    op=f===StanBlocks.stan.builtin.max ? "max" : "min"
    @eval _brmd_call_math(::$(typeof(f)),args,_kwargs,_c) =
        "\\operatorname{"*$op*"}\\left("*join(args,",")*"\\right)"
    @eval _brmd_builtin_call(::$(typeof(f)),c)=BRMDescriptionFragment(
        prose=(_brmd_text(c,:extremum;op=$op),),covers=(c.id,))
end

# Distribution equations use explicit parameter conventions. A normal's second
# constructor argument is an SD; the conventional Gaussian equation uses variance.
_brmd_law(_f) = nothing
_brmd_law(::_BRMDFlatPrior) = :improper_flat
_brmd_law(::Type{<:Flat}) = :improper_flat
_brmd_law(::Type{<:Normal}) = :normal_sd
_brmd_law(::Type{<:Exponential}) = :exponential_scale
_brmd_law(::Type{<:LogNormal}) = :lognormal_sd
_brmd_law(::Type{<:Gamma}) = :gamma_scale
_brmd_law(::Type{<:Beta}) = :beta
_brmd_law(::Type{<:Cauchy}) = :cauchy
_brmd_law(::Type{<:TDist}) = :student_t_standard
_brmd_law(::Type{<:LocationScale}) = :affine
_brmd_law(::Type{<:InverseGamma}) = :inverse_gamma
_brmd_law(::Type{<:Bernoulli}) = :bernoulli
_brmd_law(::Type{<:BernoulliLogit}) = :bernoulli_logit
_brmd_law(::Type{<:Binomial}) = :binomial
_brmd_law(::Type{<:BinomialLogit}) = :binomial_logit
_brmd_law(::Type{<:Poisson}) = :poisson
_brmd_law(::Type{<:NegativeBinomial}) = :negative_binomial
_brmd_law(::Type{<:Dirichlet}) = :dirichlet
_brmd_law(::Type{<:LKJCholesky}) = :lkj
_brmd_law(::Type{<:Weibull}) = :weibull
_brmd_law(::Type{<:Uniform}) = :uniform
_brmd_law(::typeof(MvNormalCholesky)) = :mvnormal_cholesky
_brmd_law(::typeof(StanBlocks.stan.builtin.dummy)) = :allocation_only
_brmd_law(::typeof(brm_total)) = :exact_total
_brmd_law(::typeof(brm_icar)) = :icar
# These are exact producer-owned StanBlocks bindings, never a name heuristic
# applied to arbitrary user callables.
for (name,law) in ((:normal,:normal_sd),(:std_normal,:standard_normal),
        (:exponential,:exponential_rate),(:lognormal,:lognormal_sd),
        (:gamma,:gamma_rate),(:beta,:beta),(:cauchy,:cauchy),(:student_t,:student_t_location_scale),
        (:bernoulli,:bernoulli),(:bernoulli_logit,:bernoulli_logit),
        (:binomial,:binomial),(:binomial_logit,:binomial_logit),(:poisson,:poisson),
        (:dirichlet,:dirichlet),(:lkj_corr_cholesky,:lkj),(:weibull,:weibull),(:uniform,:uniform),(:inv_gamma,:inverse_gamma),
        (:multi_normal_cholesky,:mvnormal_cholesky))
    if isdefined(StanBlocks.stan.builtin,name)
        f=getfield(StanBlocks.stan.builtin,name)
        @eval _brmd_law(::$(typeof(f))) = $(QuoteNode(law))
    end
end
_brmd_law_math(_c,::Val{:normal_sd},a) = "\\mathcal N\\left(" *
    (isempty(a) ? "0,1" : first(a)*","*(length(a)<2 ? "1" : "{"*a[2]*"}^{2}")) * "\\right)"
_brmd_law_math(_c,::Val{:standard_normal},_a) = "\\mathcal N(0,1)"
_brmd_law_math(c,::Val{:improper_flat},_a) = "\\operatorname{Flat}_{\\mathrm{"*_brmd_text(c,:math_improper)*"}}"
_brmd_law_math(c,::Val{:allocation_only},_a) = "1\\quad\\text{"*_brmd_text(c,:math_no_density)*"}"
_brmd_law_math(c,::Val{:exponential_scale},a) = "\\operatorname{Exponential}_{\\mathrm{"*_brmd_text(c,:math_scale)*"}}(" *
    (isempty(a) ? "1" : only(a)) * ")"
_brmd_law_math(c,::Val{:exponential_rate},a) = "\\operatorname{Exponential}_{\\mathrm{"*_brmd_text(c,:math_rate)*"}}(" * only(a) * ")"
_brmd_law_math(c,::Val{:gamma_scale},a) = "\\operatorname{Gamma}_{\\mathrm{"*_brmd_text(c,:math_shape_scale)*"}}(" * join(a,",") * ")"
_brmd_law_math(c,::Val{:gamma_rate},a) = "\\operatorname{Gamma}_{\\mathrm{"*_brmd_text(c,:math_shape_rate)*"}}(" * join(a,",") * ")"
_brmd_law_math(c,::Val{:lognormal_sd},a) = "\\operatorname{LogNormal}_{\\mathrm{"*_brmd_text(c,:math_log_sd)*"}}(" * join(a,",") * ")"
_brmd_law_math(_c,::Val{:bernoulli_logit},a) = "\\operatorname{Bernoulli}(\\operatorname{logit}^{-1}(" * only(a) * "))"
_brmd_law_math(_c,::Val{:binomial_logit},a) = "\\operatorname{Binomial}(" * a[1] * ",\\operatorname{logit}^{-1}(" * a[2] * "))"
_brmd_law_math(_c,::Val{:lkj},a) = "\\operatorname{LKJCholesky}(" * join(a,",") * ")"
# `brm_icar(node1, node2, comp_start, comp_size)`: the edge lists and component
# segments are data, named rather than spelled out.
_brmd_law_math(c,::Val{:icar},_a) = "\\operatorname{ICAR}(\\text{"*_brmd_text(c,:math_icar_edges)*
    "};\\ \\text{"*_brmd_text(c,:math_icar_sum_to_zero)*"})"
_brmd_law_math(_c,::Val{:uniform},a) = "\\operatorname{Uniform}(" * (isempty(a) ? "0,1" : join(a,",")) * ")"
_brmd_law_math(_c,::Val{:student_t_standard},a) = "t_{"*only(a)*"}(0,1)"
_brmd_law_math(_c,::Val{:student_t_location_scale},a) = "t_{"*a[1]*"}("*join(a[2:3],",")*")"
_brmd_law_math(_c,::Val{:mvnormal_cholesky},a) = "\\mathcal N(" * a[1] * "," * a[2] * a[2] * "^{\\mathsf T})"
# The absorbed coefficients' prior p(beta) is integrated out exactly. With
# group-level predictors each group's mean uses its own column values w_i.
_brmd_exact_total_math(sds,group_level::Bool=false) = "\\int\\prod_{i}\\mathcal N\\left(\\mathbf t_{i}\\mid "*
    (group_level ? "A(\\mathbf w_{i}\\circ\\boldsymbol\\beta)" : "A\\boldsymbol\\beta")*",\\operatorname{diag}\\left("*
    sds*"\\right)^{2}\\right)\\,p(\\boldsymbol\\beta)\\,d\\boldsymbol\\beta"
_brmd_law_math(_c,::Val{:exact_total},a) = _brmd_exact_total_math(a[1])
_brmd_law_math(_c,::Val{L},a) where L = "\\operatorname{" * _brmd_escape(L) * "}(" * join(a,",") * ")"
_brmd_law_prose(c,::Val{:normal_sd}) = _brmd_text(c,:law_normal_sd)
_brmd_law_prose(c,::Val{:standard_normal}) = _brmd_text(c,:law_standard_normal)
_brmd_law_prose(c,::Val{:icar}) = _brmd_text(c,:law_icar)
_brmd_law_prose(c,::Val{:improper_flat}) = _brmd_text(c,:law_improper_flat)
_brmd_law_prose(c,::Val{:allocation_only}) = _brmd_text(c,:law_allocation_only)
_brmd_law_prose(c,::Val{:exponential_rate}) = _brmd_text(c,:law_exponential_rate)
_brmd_law_prose(c,::Val{:exponential_scale}) = _brmd_text(c,:law_exponential_scale)
_brmd_law_prose(c,::Val{:gamma_rate}) = _brmd_text(c,:law_gamma_rate)
_brmd_law_prose(c,::Val{:gamma_scale}) = _brmd_text(c,:law_gamma_scale)
_brmd_law_prose(c,::Val{:lognormal_sd}) = _brmd_text(c,:law_lognormal_sd)
_brmd_law_prose(c,::Val{:student_t_standard}) = _brmd_text(c,:law_student_t_standard)
_brmd_law_prose(c,::Val{:student_t_location_scale}) = _brmd_text(c,:law_student_t_location_scale)
_brmd_law_prose(c,::Val{:affine}) = _brmd_text(c,:law_affine)
_brmd_law_prose(c,::Val{:mvnormal_cholesky}) = _brmd_text(c,:law_mvnormal_cholesky)
_brmd_law_prose(c,::Val{:exact_total}) = _brmd_text(c,:law_exact_total)
_brmd_law_prose(c,::Val{L}) where L = _brmd_text(c,:law_declared;law=L)

# Children `(id..., :coordinate, i)` of a BRM-generated vector prior.
_brmd_coordinates(x::BRMDescriptionComponent) = Tuple(child for child in x.children
    if length(child.id)==length(x.id)+2 && child.id[1:length(x.id)]==x.id && child.id[end-1]===:coordinate)
function _brmd_coordinate_math(c,x,coordinates)
    support=x.callable isa StanBlocks.ValueFamily ? x.callable.support :
        _brmd_shared_bounds(coordinate.keywords for coordinate in coordinates)
    parts=map(coordinates) do coordinate
        law=_brmd_distribution_math(c,coordinate)
        lower=get(coordinate.keywords,:lower,nothing); upper=get(coordinate.keywords,:upper,nothing)
        lower==get(support,:lower,nothing) && upper==get(support,:upper,nothing) && return law
        law*"\\ \\text{"*_brmd_text(c,:math_on)*"}\\ ["*(isnothing(lower) ? "-\\infty" : brm_description_math(c,lower))*","*
            (isnothing(upper) ? "\\infty" : brm_description_math(c,upper))*"]"
    end
    length(parts)==1 ? only(parts) : "\\left["*join(parts,";\\ ")*"\\right]"
end
function _brmd_distribution_math(c,x::BRMDescriptionComponent)
    coordinates=_brmd_coordinates(x)
    isempty(coordinates) || return _brmd_coordinate_math(c,x,coordinates)
    law = _brmd_law(x.callable)
    isnothing(law) && return brm_description_math(c,x)
    law===:affine && return "\\operatorname{LocationScale}("*
        brm_description_math(c,x.arguments[1])*","*brm_description_math(c,x.arguments[2])*","*
        _brmd_distribution_math(c,x.arguments[3])*")"
    law===:exact_total && return _brmd_exact_total_math(
        brm_description_math(c,x.arguments[1]),haskey(x.keywords,:group_columns))
    _brmd_law_math(c,Val(law),map(a -> brm_description_math(c,a),x.arguments))
end
_brmd_distribution_math(c,x) = brm_description_math(c,x)

_brmd_covariate_math(c,x) = brm_description_math(c,x)
_brmd_covariate_math(c,x::NamedTuple)=haskey(x,:callable) ?
    _brmd_transformed_math(x.callable,c,x) : brm_description_math(c,x)
function _brmd_covariate_math(c,x::BRMDescriptionComponent)
    _brmd_transformed_math(x.callable,c,x)
end
_brmd_transformed_math(_f,c,x) = brm_description_math(c,x)
function _brmd_transform_constant(c,x,kind)
    source = isempty(x.arguments) ? nothing : first(x.arguments)
    source isa BRMDescriptionReference || return nothing
    entries = filter(e -> e.kind in (kind,Symbol(:missing_,kind)) && e.source isa BRMDescriptionReference &&
                           e.source.name === source.name,c.fitted_constants)
    length(entries)==1 || return nothing
    entry=only(entries)
    entry.kind===kind && return entry.value
    fit=entry.value.fit
    kind===:center ? fit.mean : (fit.mean,fit.scale)
end
function _brmd_transformed_math(::typeof(center),c,x)
    anchor=_brmd_transform_constant(c,x,:center)
    isnothing(anchor) && return brm_description_math(c,x)
    "(" * brm_description_math(c,only(x.arguments)) * "-" * string(anchor) * ")"
end
function _brmd_scale_math(c,x,kind)
    anchor=_brmd_transform_constant(c,x,kind)
    isnothing(anchor) && return brm_description_math(c,x)
    fitted = anchor isa NamedTuple ? Tuple(values(anchor)) : anchor
    fitted isa Tuple && length(fitted)==2 || return brm_description_math(c,x)
    "\\frac{" * brm_description_math(c,only(x.arguments)) * "-" * string(fitted[1]) * "}{" * string(fitted[2]) * "}"
end
_brmd_transformed_math(::typeof(zscale),c,x) = _brmd_scale_math(c,x,:zscale)
_brmd_transformed_math(::typeof(standardize),c,x) = _brmd_scale_math(c,x,:standardize)

_brmd_beta(c,label) = "\\beta_{" * _brmd_escape(c.provenance.owner) * "," * _brmd_escape(label) * "}"
function _brmd_factor_math(c,name)
    ps=filter(p -> length(p.id)>=3 && p.id[1]===:population && p.id[2]===c.provenance.owner &&
                  get(p.source,:predictor,nothing)===name,c.priors)
    isempty(ps) && return nothing
    join((_brmd_beta(c,string(name)*"="*string(p.source.level)) *
          "\\,\\mathbf 1\\{" * brm_description_symbol(c,name) * "=" *
          brm_description_math(c,string(p.source.level)) * "\\}" for p in ps)," + ")
end
_brmd_term_math(c,x::Integer) = x==1 ? _brmd_beta(c,:Intercept) : string(x)
_brmd_term_math(c,x::Number) = string(x)
_brmd_term_math(c,x::BRMDescriptionReference) = something(_brmd_factor_math(c,x.name),
    _brmd_beta(c,x.name)*"\\,"*brm_description_math(c,x))
_brmd_term_math(c,x) = brm_description_math(c,x)
_brmd_term_math(c,x::BRMDescriptionComponent) = _brmd_term_call_math(x.callable,c,x)
_brmd_term_call_math(_f,c,x) = _brmd_beta(c,_brmd_callable_name(x.callable))*"\\,"*_brmd_covariate_math(c,x)
_brmd_term_call_math(::typeof(+),c,x) = join((_brmd_term_math(c,a) for a in x.arguments)," + ")
_brmd_term_call_math(::typeof(-),c,x) = join((_brmd_term_math(c,a) for a in x.arguments)," - ")
_brmd_term_call_math(::typeof(offset),c,x) = _brmd_covariate_math(c,only(x.arguments))
_brmd_term_call_math(::typeof(&),c,x) = _brmd_beta(c,:interaction)*"\\," *
    join((_brmd_covariate_math(c,a) for a in x.arguments),"\\,")
function _brmd_term_call_math(::typeof(factor),c,x)
    a=only(x.arguments)
    a isa BRMDescriptionReference || return brm_description_math(c,x)
    something(_brmd_factor_math(c,a.name),"0")
end
for f in (s,t2,gp,hsgp,me,mi,mo,mo1,ar,dar,rw,cdar,car,interval_censored)
    @eval _brmd_term_call_math(::$(typeof(f)),c,x) = brm_description_math(c,x)
end
_brmd_term_call_math(::typeof(kernel),c,x) = _brmd_kernel_math(c,x)
function _brmd_kernel_math(c,x)
    "\\mathcal K_{"*_brmd_escape(x.provenance.owner)*",g_j}"
end
function _brmd_group_term_math(c,x)
    effects=first(x.arguments)
    group=last(x.arguments)
    membership=group isa BRMDescriptionComponent && group.callable===mm
    group isa BRMDescriptionComponent && group.callable===gr && (group=first(group.arguments))
    group_name=membership ? Tuple(a.name for a in group.arguments) : group isa BRMDescriptionReference ? group.name : :group
    shared=length(x.arguments)==3 ? x.arguments[2] : nothing
    shared_id=shared isa BRMDescriptionReference ? shared.name : shared
    blocks=filter(g->g.group==group_name && g.shared_id==shared_id &&
        any(m->m.predictor===c.provenance.owner,g.margins),c.provenance.random_effects)
    isempty(blocks) && return "\\mathbf z_{"*_brmd_escape(c.provenance.owner)*","*_brmd_escape(group_name)*",j}^{\\mathsf T}\\mathbf b_{"*
        _brmd_escape(c.provenance.owner)*","*_brmd_escape(group_name)*",g_j}"
    join(((membership ? "\\sum_m\\widetilde w_{jm}\\," : "")*
        "\\mathbf z_{"*_brmd_escape(c.provenance.owner)*","*_brmd_block_math(c,g.id)*",j}^{\\mathsf T}\\mathbf b_{"*
        _brmd_block_math(c,g.id)*(membership ? ",g_{jm}}" : ",g_j}") for g in blocks)," + ")
end
_brmd_term_call_math(::typeof(|),c,x) = _brmd_group_term_math(c,x)
_brmd_term_call_math(::typeof(doublepipe),c,x) = _brmd_group_term_math(c,x)

# Built-in semantic coverage is explicit. Unknown callable identities stay gaps.
_brmd_builtin_call(_f,c) = begin
    law=_brmd_law(c.callable)
    isnothing(law) ? nothing : isempty(_brmd_coordinates(c)) ?
        BRMDescriptionFragment(covers=(c.id,)) : _brmd_coordinatewise_fragment(c)
end
_brmd_coordinatewise_fragment(c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:coordinatewise_prior),),
    covers=(c.id,))
for f in (+,-,*,/,^,exp,log,sqrt,logistic,log1pexp,abs,sum,cumsum,maximum,minimum,
          getindex,getproperty,identity,|,doublepipe,&,gr,offset,protect,factor,ragged,
          center,zscale,standardize)
    @eval _brmd_builtin_call(::$(typeof(f)),c) = BRMDescriptionFragment(covers=(c.id,))
end
_brmd_builtin_call(::StanBlocks.SlicModel,c)=BRMDescriptionFragment(covers=(c.id,))
# Only BRM-generated vector priors decompose; other value families stay gaps.
_brmd_builtin_call(::StanBlocks.ValueFamily,c)=isempty(_brmd_coordinates(c)) ? nothing :
    _brmd_coordinatewise_fragment(c)
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.vcat),c)=isempty(_brmd_coordinates(c)) ? nothing :
    _brmd_coordinatewise_fragment(c)
function _brmd_builtin_call(::typeof(mm),c)
    normalized=c.keywords.normalize
    weights=c.keywords.weights
    n=length(c.arguments)
    definition=isnothing(weights) ? (normalized ? "\\widetilde w_{jm}=1/"*string(n) : "\\widetilde w_{jm}=1") :
        normalized ? "\\widetilde w_{jm}=w_{jm}/\\sum_h w_{jh}" : "\\widetilde w_{jm}=w_{jm}"
    prose=join((_brmd_text(c,:multi_membership;groups=join((_brmd_path_markdown(a.name) for a in c.arguments),", ")),
        _brmd_text(c,normalized ? :multi_membership_normalized : :multi_membership_unscaled),
        _brmd_text(c,:multi_membership_weights;weights=isnothing(weights) ? _brmd_text(c,:equal_weights) :
            "\$"*brm_description_math(c,weights)*"\$"))," ")
    BRMDescriptionFragment(prose=(prose,),
        equations=(definition,),covers=(c.id,))
end
function _brmd_builtin_call(::Type{<:LocationScale},c)
    location,scale,base=c.arguments
    loc=brm_description_math(c,location); s=brm_description_math(c,scale)
    prose=_brmd_text(c,:affine_density)
    base isa BRMDescriptionComponent && _brmd_law(base.callable)===:student_t_standard &&
        (prose*=" "*_brmd_text(c,:affine_student_t))
    BRMDescriptionFragment(prose=(prose,),equations=("Z\\sim{}"*_brmd_distribution_math(c,base),
        "p_Y(y)=\\frac{1}{"*s*"}f_Z\\left(\\frac{y-"*loc*"}{"*s*"}\\right)"),covers=(c.id,))
end
# A bound is a keyword or, when `position` is an index, that positional argument.
function _brmd_bound_math(c,key,position,fallback)
    positional=!isnothing(position) && length(c.arguments)>=position
    haskey(c.keywords,key) || positional || return fallback
    value=haskey(c.keywords,key) ? c.keywords[key] : c.arguments[position]
    value isa Real && isinf(value) && return value<0 ? "-\\infty" : "\\infty"
    value===nothing ? fallback : brm_description_math(c,value)
end
# The Distributions wrappers also take positional bounds; StanBlocks'
# combinators take `lower=`/`upper=` keywords only.
_brmd_bounds_math(c,positional) = (_brmd_bound_math(c,:lower,positional ? 2 : nothing,"-\\infty"),
    _brmd_bound_math(c,:upper,positional ? 3 : nothing,"\\infty"))
_brmd_builtin_call(::typeof(truncated),c) = _brmd_truncation_fragment(c,_brmd_bounds_math(c,true)...)
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.truncated),c) =
    _brmd_truncation_fragment(c,_brmd_bounds_math(c,false)...)
function _brmd_truncation_fragment(c,lower,upper)
    BRMDescriptionFragment(prose=(_brmd_text(c,:truncation),),
        equations=("p(y)=\\frac{f(y)\\,\\mathbf1\\{"*lower*"\\le y\\le "*upper*"\\}}{F("*upper*")-F("*lower*"^-)}",),covers=(c.id,))
end
_brmd_builtin_call(::typeof(censored),c) = _brmd_censoring_fragment(c,_brmd_bounds_math(c,true)...)
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.censored),c) =
    _brmd_censoring_fragment(c,_brmd_bounds_math(c,false)...)
function _brmd_censoring_fragment(c,lower,upper)
    rows=String[]
    lower!="-\\infty" && push!(rows,"F("*lower*")&y="*lower)
    condition=(lower=="-\\infty" ? "y" : lower*"<y")*(upper=="\\infty" ? "" : "<"*upper)
    lower=="-\\infty" && upper=="\\infty" && (condition="y\\in\\mathbb R")
    push!(rows,"f(y)&"*condition)
    upper!="\\infty" && push!(rows,"1-F("*upper*"^-)&y="*upper)
    BRMDescriptionFragment(prose=(_brmd_text(c,:censoring),),
        equations=("p(y)=\\begin{cases}"*join(rows,"\\\\")*"\\end{cases}",),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(interval_censored),c)
    base=first(c.arguments)
    if base isa BRMDescriptionReference
        return BRMDescriptionFragment(prose=(_brmd_text(c,:interval_censored_predictor),),covers=(c.id,))
    end
    upper=_brmd_bound_math(c,:upper,2,"U_j")
    BRMDescriptionFragment(prose=(_brmd_text(c,:interval_censored_response),),
        equations=("\\mathcal L_j=F("*upper*")-F(L_j)\\quad\\text{"*_brmd_text(c,:math_continuous)*"}",
            "\\mathcal L_j=F("*upper*")-F(L_j-1)\\quad\\text{"*_brmd_text(c,:math_integer)*"}"),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(weighted),c)
    weight=c.arguments[2]
    w=weight isa BRMDescriptionComponent && !isempty(weight.arguments) ?
        brm_description_math(c,first(weight.arguments)) : brm_description_math(c,weight)
    if weight isa BRMDescriptionComponent && weight.callable===aweights
        base=first(c.arguments)
        base isa BRMDescriptionComponent && _brmd_law(base.callable)===:normal_sd || return nothing
        a=map(x->brm_description_math(c,x),base.arguments)
        mean=isempty(a) ? "0" : first(a)
        sd=length(a)<2 ? "1" : a[2]
        output=brm_description_math(c,BRMDescriptionReference(c.provenance.owner,:observation))
        return BRMDescriptionFragment(prose=(_brmd_text(c,:analytic_weights),),
            equations=(output*"\\sim\\mathcal N("*mean*",{"*sd*"}^{2}/"*w*")",),covers=(c.id,))
    end
    _brmd_weight_fragment(c,w)
end
_brmd_weight_fragment(c,w) = BRMDescriptionFragment(prose=(_brmd_text(c,:observation_weights),),
    equations=("\\log\\mathcal L=\\sum_j "*w*"\\log p(y_j\\mid\\theta)",),covers=(c.id,))
# StanBlocks' combinators inside a cell: `weighted(family, w, args...)` and
# `interval_censored(family, lower, upper, args...)`, or the same with a family
# call in place of the token and its arguments (StanBlocks `lpxf_builtin.jl`).
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.weighted),c) =
    length(c.arguments)<2 ? nothing : _brmd_weight_fragment(c,brm_description_math(c,c.arguments[2]))
function _brmd_builtin_call(::typeof(StanBlocks.stan.builtin.interval_censored),c)
    length(c.arguments)<3 && return nothing
    lower,upper=map(a->brm_description_math(c,a),c.arguments[2:3])
    BRMDescriptionFragment(prose=(_brmd_text(c,:interval_censored_cell),),
        equations=("\\mathcal L=F("*upper*")-F("*lower*")",),covers=(c.id,))
end
for f in (aweights,fweights,weights)
    @eval _brmd_builtin_call(::$(typeof(f)),c)=BRMDescriptionFragment(covers=(c.id,))
end
function _brmd_builtin_call(::typeof(StanBlocks.stan.builtin.maybe_index),c)
    BRMDescriptionFragment(prose=(_brmd_text(c,:imputation_split),),covers=(c.id,))
end
function _brmd_call_math(::typeof(StanBlocks.stan.builtin.maybe_index),args,_kwargs,c)
    "\\operatorname{"*_brmd_text(c,:math_select)*"}_{\\mathrm{"*_brmd_text(c,:math_scalar_vector)*"}}\\left("*join(args,",")*"\\right)"
end
function _brmd_builtin_call(::typeof(kernel),c)
    aliases=filter(b->b.role===:alias,c.bindings)
    definitions=String[]; notes=NamedTuple[]; counter=Ref(0)
    binding_prose=String[]
    for (index,b) in enumerate(aliases)
        symbol="a_{"*string(c.provenance.description_number)*","*string(index)*"}"
        push!(notes,(;name=(:kernel_binding,c.id,b.name),symbol,
            meaning=_brmd_text(c,:kernel_input_meaning;name=b.name,owner=c.provenance.owner)))
        value=_brmd_compact_expression(c,b.value,definitions,notes,counter;root=true)
        localcontext=_brmd_render_context(c;notation=(c.notation...,notes...))
        push!(definitions,_brmd_assignment_equation(localcontext,symbol,value))
        push!(binding_prose,_brmd_text(c,:kernel_input;name=b.name,symbol))
    end
    body=last(first(c.arguments).arguments)
    final=body isa BRMDescriptionComponent && body.callable===:block && !isempty(body.arguments) ? last(body.arguments) : body
    readable=!(final isa BRMDescriptionComponent && final.kind===:syntax &&
        final.callable in (:if,:for,:while,:block))
    equations=readable ? (_brmd_assignment_equation(c,_brmd_kernel_math(c,c),final),) : ()
    BRMDescriptionFragment(prose=(_brmd_text(c,:kernel_mapping;owner=c.provenance.owner),binding_prose...),
        equations=(equations...,definitions...),notation=Tuple(notes),covers=(c.id,))
end
function _brmd_call_math(::typeof(StanBlocks.stan.builtin.rep_vector),args,_kwargs,_c)
    first(args)*"\\,\\mathbf1_{"*args[2]*"}"
end
_brmd_builtin_call(::typeof(StanBlocks.stan.builtin.rep_vector),c)=BRMDescriptionFragment(covers=(c.id,))
function _brmd_call_math(::typeof(addprop),args,_kwargs,_c)
    "\\sqrt{"*args[2]*"^2+("*args[1]*"\\,"*args[3]*")^2}"
end
_brmd_builtin_call(::typeof(addprop),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:additive_proportional_error),),covers=(c.id,))
function _brmd_builtin_call(::typeof(s),c)
    BRMDescriptionFragment(prose=(_brmd_text(c,:thin_plate_spline),),
        equations=("f(x_j)=X_{\\mathrm{"*_brmd_text(c,:math_null)*"},j}\\beta+Z_{\\mathrm{"*_brmd_text(c,:math_penalized)*
            "},j}u,\\quad u\\mid\\tau\\sim\\mathcal N(0,\\tau^2 I)",),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(t2),c)
    BRMDescriptionFragment(prose=(_brmd_text(c,:tensor_smooth),),
        equations=("f(x_j,z_j)=X_{0,j}\\beta+\\sum_{h\\in\\{rr,rn,nr\\}}Z_{h,j}u_h,\\quad u_h\\mid\\tau_h\\sim\\mathcal N(0,\\tau_h^2I)",),covers=(c.id,))
end
function _brmd_gp_fragment(c,approximate)
    covariance=get(c.keywords,:cov,:exp_quad)
    covname=covariance isa BRMDescriptionReference ? covariance.name : covariance
    axes=join(("\$"*brm_description_math(c,a)*"\$" for a in c.arguments),", ")
    prose=String[_brmd_text(c,approximate ? :hilbert_space_process : :gaussian_process;covariance=covname,axes)]
    equations=String[]
    if !approximate
        jitter=brm_description_math(c,get(c.keywords,:jitter,1e-9))
        push!(equations,"f\\mid\\ell,\\tau\\sim\\mathcal N(0,K)")
        if covname===:periodic
            period=brm_description_math(c,c.keywords.period)
            push!(equations,"K_{ab}=\\tau^2\\exp\\left(-\\frac{2\\sin^2(\\pi(x_a-x_b)/"*period*")}{\\ell^2}\\right)+"*jitter*"\\,\\mathbf1\\{a=b\\}")
        elseif covname===:exp_quad
            push!(equations,"K_{ab}=\\tau^2\\exp\\left(-\\frac12\\sum_r\\frac{(x_{ar}-x_{br})^2}{\\ell_r^2}\\right)+"*jitter*"\\,\\mathbf1\\{a=b\\}")
            push!(prose,_brmd_text(c,get(c.keywords,:iso,true)===true ? :isotropic_length_scale : :anisotropic_length_scale))
        else
            return nothing
        end
    elseif covname===:periodic
        period=brm_description_math(c,c.keywords.period)
        push!(equations,"f(x)=\\sum_{k=1}^{M}q_k\\left[z_{k,c}\\cos(2\\pi kx/"*period*")+z_{k,s}\\sin(2\\pi kx/"*period*")\\right]")
        push!(equations,"q_k=\\tau\\sqrt{2e^{-a}I_k(a)},\\quad a=\\ell^{-2},\\quad z_{k,c},z_{k,s}\\sim\\mathcal N(0,1)")
        push!(prose,_brmd_text(c,:periodic_harmonics))
    elseif covname===:exp_quad
        grouped=haskey(c.keywords,:by)
        z=grouped ? "z_{g(j),k}" : "z_k"
        push!(equations,"f(x_j)=\\sum_{k=1}^{M}\\phi_k(x_j)q_k"*z*",\\quad "*z*"\\sim\\mathcal N(0,1)")
        push!(equations,"q_k=\\tau\\prod_r(\\sqrt{2\\pi}\\ell_r)^{1/2}\\exp\\left(-\\frac14\\sum_r\\ell_r^2\\omega_{kr}^2\\right)")
        push!(equations,"\\phi_k(x)=\\prod_r L_r^{-1/2}\\sin[\\omega_{kr}(x_r-c_r+L_r)],\\quad\\omega_{kr}=\\frac{k_r\\pi}{2L_r}")
        sources=Tuple(a.name for a in c.arguments if a isa BRMDescriptionReference)
        fitted=filter(e->e.kind===:hsgp && e.source isa Tuple &&
            Tuple(a.name for a in e.source if a isa BRMDescriptionReference)==sources,c.fitted_constants)
        for e in fitted
            fits=get(e.value,:fits,nothing)
            isnothing(fits) || push!(prose,_brmd_text(c,:approximation_domain;fits,sizes=get(e.value,:K,())))
        end
        get(c.keywords,:iso,true)===true && push!(prose,_brmd_text(c,:isotropic_length_scale))
        grouped && push!(prose,_brmd_text(c,:grouped_basis_weights))
        if get(c.keywords,:orthogonal_to,nothing)!==nothing
            push!(equations,"\\Phi_{\\mathrm{"*_brmd_text(c,:math_used)*"}}=(I-P_{[1,x]})\\Phi_{\\mathrm{"*_brmd_text(c,:math_raw)*"}}")
            push!(prose,_brmd_text(c,:orthogonal_projection))
        end
        push!(prose,_brmd_text(c,:partial_centering))
    else
        return nothing
    end
    BRMDescriptionFragment(; prose=Tuple(prose),equations=Tuple(equations),covers=(c.id,))
end
_brmd_builtin_call(::typeof(gp),c) = _brmd_gp_fragment(c,false)
_brmd_builtin_call(::typeof(hsgp),c) = _brmd_gp_fragment(c,true)
function _brmd_builtin_call(::typeof(me),c)
    BRMDescriptionFragment(prose=(_brmd_text(c,:measurement_error),),
        equations=(brm_description_math(c,c.arguments[1])*"\\mid x^*\\sim\\mathcal N(x^*,{"*brm_description_math(c,c.arguments[2])*"}^2)",),covers=(c.id,))
end
_brmd_builtin_call(::typeof(mi),c) = BRMDescriptionFragment(
    prose=(_brmd_text(c,c.provenance.observation_role===:covariate_draw ? :fresh_completion : :missing_entries),),covers=(c.id,))
function _brmd_builtin_call(::typeof(LKJCovarianceFactor),c)
    binding=get(c.provenance,:covariance_factor,nothing)
    isnothing(binding) && return nothing
    owner=c.provenance.owner
    scales=any(n->n.name==binding.scales,c.notation) ? brm_description_symbol(c,binding.scales) :
        "s_{"*_brmd_escape(owner)*"}"
    correlation=any(n->n.name==binding.correlation,c.notation) ? brm_description_symbol(c,binding.correlation) :
        "L_{"*_brmd_escape(owner)*",\\mathrm{"*_brmd_text(c,:math_correlation)*"}}"
    factor=brm_description_symbol(c,owner)
    BRMDescriptionFragment(
        prose=(_brmd_text(c,:covariance_factor),),
        equations=(factor*"=\\operatorname{diag}("*scales*")"*correlation,
            "\\Sigma_{"*_brmd_escape(owner)*"}="*factor*factor*"^{\\mathsf T}"),
        notation=((;name=binding.scales,symbol=scales,meaning=_brmd_text(c,:covariance_scales_meaning;owner)),
            (;name=binding.correlation,symbol=correlation,meaning=_brmd_text(c,:covariance_correlation_meaning;owner))),
        covers=(c.id,))
end
function _brmd_call_math(::typeof(brm_joint_column),args,_kwargs,c)
    "\\operatorname{"*_brmd_text(c,:math_column)*"}_{"*args[2]*";"*args[3]*"\\times "*args[4]*"}\\left("*args[1]*"\\right)"
end
_brmd_builtin_call(::typeof(brm_joint_column),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:joint_column)*" "*_brmd_text(c,c.provenance.observation_role===:covariate_draw ?
        :joint_column_draw : :joint_column_missing),),covers=(c.id,))
for f in (mo,mo1)
    @eval _brmd_builtin_call(::$(typeof(f)),c) = BRMDescriptionFragment(
        prose=(_brmd_text(c,:monotonic),),
        equations=("m(k)=\\sum_{h<k}\\zeta_h,\\quad\\zeta_h\\ge0,\\quad\\sum_h\\zeta_h=1,\\quad m(1)=0",),covers=(c.id,))
end
_brmd_builtin_call(::typeof(ar),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:autoregressive),),
    equations=("a_t=\\phi a_{t-1}+\\epsilon_t,\\quad\\phi=\\tanh(\\phi_{\\mathrm{"*_brmd_text(c,:math_raw)*
        "}}),\\quad\\epsilon_t\\sim\\mathcal N(0,1)",),covers=(c.id,))
_brmd_builtin_call(::typeof(cdar),c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:damped_walk),),
    equations=("\\delta_1=\\sigma Lz_1,\\quad\\delta_t=\\rho\\delta_{t-1}+\\sigma\\sqrt{1-\\rho^2}Lz_t,\\quad z_t\\sim\\mathcal N(0,I)",),covers=(c.id,))
# Temporal terms require their recurrence, rather than just a term caption.
_brmd_builtin_call(::typeof(rw),c) = BRMDescriptionFragment(
    prose=(_brmd_text(c,:random_walk),),
    equations=("f_1=0,\\quad f_{t+1}=f_t+\\sigma z_t,\\quad z_t\\sim\\mathcal N(0,1)",),covers=(c.id,))
# The areal field's law depends on `type=`; the keyword rides on the call.
function _brmd_builtin_call(::typeof(car),c)
    type=get(c.keywords,:type,nothing)
    # a literal `type=:bym2` is snapshotted as a reference to its name
    type isa BRMDescriptionReference && (type=type.name)
    common=_brmd_text(c,:areal_effect)
    type===:bym2 ? BRMDescriptionFragment(
        prose=(common*" "*_brmd_text(c,:areal_bym2),),
        equations=("b_k=\\sigma\\left(\\sqrt{\\rho/s_{c(k)}}\\,\\phi_k+\\sqrt{1-\\rho}\\,\\theta_k\\right),\\quad\\theta_k\\sim\\mathcal N(0,1)",
            "\\log p(\\phi)=-\\tfrac12\\sum_{i\\sim j}(\\phi_i-\\phi_j)^2+\\sum_c\\log\\mathcal N\\!\\left(\\textstyle\\sum_{k\\in c}\\phi_k\\mid 0,\\,0.001\\,n_c\\right)"),
        covers=(c.id,)) :
    BRMDescriptionFragment(
        prose=(common*" "*_brmd_text(c,:areal_icar_isolated),),
        equations=("b_k=\\sigma\\,\\phi_k",
            "\\log p(\\phi)=-\\tfrac12\\sum_{i\\sim j}(\\phi_i-\\phi_j)^2+\\sum_c\\log\\mathcal N\\!\\left(\\textstyle\\sum_{k\\in c}\\phi_k\\mid 0,\\,0.001\\,n_c\\right)"),
        covers=(c.id,))
end
_brmd_builtin_call(::typeof(dar),c) = BRMDescriptionFragment(
    prose=(_brmd_text(c,:differenced_autoregressive),),
    equations=("d_t=\\rho d_{t-1}+\\sigma z_t,\\quad f_{t+1}=f_t+d_t,\\quad z_t\\sim\\mathcal N(0,1)",),covers=(c.id,))

function _brmd_builtin_fragment(c)
    _brmd_builtin_kind(Val(c.kind),c)
end
_brmd_builtin_kind(::Val{:call},c) = _brmd_builtin_call(c.callable,c)
_brmd_builtin_kind(::Val{:submodel},c) = BRMDescriptionFragment(covers=(c.id,))
_brmd_builtin_kind(::Val{:submodel_output},c)=BRMDescriptionFragment(
    prose=(_brmd_text(c,:submodel_output;owner=c.provenance.owner),),covers=(c.id,))
function _brmd_builtin_kind(::Val{:syntax},c)
    c.callable in (:block,:tuple,:vect,:ref,:(=),:return,:->,:.,:(::),:kw,:parameters) || return nothing
    c.callable===:(=) && first(c.arguments) isa BRMDescriptionReference &&
        return _brmd_assignment_fragment(c,brm_description_symbol(c,first(c.arguments).name),last(c.arguments))
    BRMDescriptionFragment(covers=(c.id,))
end
_brmd_builtin_kind(::Val{:parameter},c) = BRMDescriptionFragment(covers=(c.id,))
_brmd_builtin_kind(::Val{:assignment},c) = _brmd_assignment_fragment(c,
    brm_description_math(c,c.arguments[1]),c.arguments[2];
    prose=(_brmd_text(c,:assignment;owner=c.provenance.owner),))

function _brmd_assignment_fragment(c,lhs,rhs;prose=())
    definitions=String[]; notes=NamedTuple[]; counter=Ref(0)
    reduced=length(lhs*brm_description_math(c,rhs))<=160 ? rhs :
        _brmd_compact_expression(c,rhs,definitions,notes,counter;root=true)
    localcontext=_brmd_render_context(c;notation=(c.notation...,notes...))
    BRMDescriptionFragment(prose=prose,
        equations=(_brmd_assignment_equation(localcontext,lhs,reduced),definitions...),
        notation=Tuple(notes),covers=(c.id,))
end

# Line breaks follow the public expression structure. No labels, arguments or
# fitted values are clipped, and these relations never cover a scientific child.
function _brmd_sum_equation(lhs,terms)
    body=isempty(terms) ? "0" : join(terms," + ")
    length(lhs*body)<=160 && return lhs*"="*body
    rows=(lhs*"&="*first(terms),("&\\quad + "*t for t in terms[2:end])...)
    "\\begin{aligned}"*join(rows,"\\\\\n")*"\\end{aligned}"
end
function _brmd_addends(c,x)
    x isa BRMDescriptionComponent && x.callable===(+) ?
        Tuple(t for a in x.arguments for t in _brmd_addends(c,a)) : (brm_description_math(c,x),)
end
function _brmd_assignment_equation(c,lhs,rhs)
    ordinary=lhs*"="*brm_description_math(c,rhs)
    length(ordinary)<=160 && return ordinary
    if rhs isa NamedTuple
        args=Tuple(_brmd_identifier(k)*"="*brm_description_math(c,v) for (k,v) in pairs(rhs))
        rows=(lhs*"&=\\bigl\\{",("&\\quad "*a*(i==length(args) ? "\\bigr\\}" : ";") for (i,a) in enumerate(args))...)
        return "\\begin{aligned}"*join(rows,"\\\\\n")*"\\end{aligned}"
    end
    rhs isa BRMDescriptionComponent || return ordinary
    rhs.callable===(+) && return _brmd_sum_equation(lhs,_brmd_addends(c,rhs))
    if rhs.kind===:syntax && rhs.callable in (:tuple,:vect)
        args=map(a->brm_description_math(c,a),rhs.arguments)
        rows=(lhs*"&=\\bigl[",("&\\quad "*a*(i==length(args) ? "\\bigr]" : ",") for (i,a) in enumerate(args))...)
        return "\\begin{aligned}"*join(rows,"\\\\\n")*"\\end{aligned}"
    end
    rhs.kind===:call && !(rhs.callable in (+,-,*,/,^,exp,log,sqrt,logistic,
        StanBlocks.stan.builtin.inv_logit)) && _brmd_law(rhs.callable)===nothing || return ordinary
    _brmd_call_equation(c,lhs,rhs,"=")
end
function _brmd_call_equation(c,lhs,rhs,relation)
    ordinary=lhs*relation*brm_description_math(c,rhs)
    args=(map(a->brm_description_math(c,a),rhs.arguments)...,
        (_brmd_identifier(k)*"="*brm_description_math(c,v) for (k,v) in pairs(rhs.keywords))...)
    isempty(args) && return ordinary
    rows=String[lhs*"&"*relation*_brmd_identifier(_brmd_callable_name(rhs.callable))*"\\bigl("]
    append!(rows,("&\\quad "*a*(i==length(args) ? "\\bigr)" : ",") for (i,a) in enumerate(args)))
    "\\begin{aligned}"*join(rows,"\\\\\n")*"\\end{aligned}"
end
function _brmd_distribution_equation(c,lhs,rhs)
    # End the control word even when the following law starts with a letter.
    ordinary=lhs*"\\sim{}"*_brmd_distribution_math(c,rhs)
    length(ordinary)<=160 && return ordinary
    rhs isa BRMDescriptionComponent && rhs.kind===:call &&
        _brmd_law(rhs.callable)===nothing || return ordinary
    _brmd_call_equation(c,lhs,rhs,"\\sim")
end
_brmd_population_addends(c,x)=x isa BRMDescriptionComponent && x.callable===(+) ?
    Tuple(t for a in x.arguments for t in _brmd_population_addends(c,a)) : (_brmd_term_math(c,x),)
function _brmd_builtin_kind(::Val{:predictor},c)
    lhs,rhs=c.arguments
    columns=c.provenance.design_columns
    definitions=String[]
    notation=NamedTuple[]
    if columns===nothing
        body=_brmd_term_math(c,rhs)
        hasintercept=occursin("Intercept",body)
        terms=_brmd_population_addends(c,rhs)
    else
        expanded=Tuple(_brmd_beta(c,col.label)*(col.label===:Intercept ? "" : "\\,"*_brmd_column_math(c,col)) for col in columns)
        compact=length(join(expanded," + "))>160
        fixed=Tuple(begin
            coordinate=compact ? _brmd_prepared_coordinate(c,col,k,definitions,notation) : _brmd_column_math(c,col)
            _brmd_beta(c,col.label)*(col.label===:Intercept ? "" : "\\,"*coordinate)
        end for (k,col) in enumerate(columns))
        extras=_brmd_extra_terms(c,rhs)
        terms=(fixed...,extras...)
        hasintercept=any(col->col.label===:Intercept,columns)
    end
    equation=_brmd_sum_equation(brm_description_math(c,lhs),terms)
    prose=_brmd_text(c,:predictor;owner=c.provenance.owner)*" "*
        _brmd_text(c,hasintercept ? :population_intercept : :no_population_intercept)
    BRMDescriptionFragment(prose=(prose,),equations=(equation,definitions...),notation=Tuple(notation),covers=(c.id,))
end

function _brmd_prepared_coordinate(c,column,k,definitions,notation)
    p=column.preprocess
    isnothing(p) && return _brmd_column_math(c,column)
    p.kind in (:center,:zscale,:standardize,:missing_center,:missing_zscale,:missing_standardize) ||
        return _brmd_column_math(c,column)
    owner=c.provenance.owner
    sub=_brmd_escape(owner)*","*string(k)
    x="\\widetilde X_{"*sub*",j}"
    center="c_{"*sub*"}"
    scale="s_{"*sub*"}"
    scaled=!(p.kind in (:center,:missing_center))
    if p.kind in (:missing_center,:missing_zscale,:missing_standardize)
        mean=p.const_.fit.mean
        sd=p.const_.fit.scale
    elseif scaled
        fitted=p.const_ isa NamedTuple ? Tuple(values(p.const_)) : p.const_
        mean,sd=fitted
    else
        mean=p.const_
        sd=nothing
    end
    raw=brm_description_math(c,p.raw_ref)
    push!(definitions,x*"="*(scaled ? "\\frac{"*raw*"-"*center*"}{"*scale*"}" : raw*"-"*center))
    push!(definitions,center*"="*brm_description_math(c,mean))
    scaled && push!(definitions,scale*"="*brm_description_math(c,sd))
    meaning=_brmd_text(c,:prepared_column_meaning;kind=p.kind,column=column.label,owner)
    push!(notation,(;name=(:prepared_column,owner,column.label),symbol=x,meaning,axis=:observation))
    push!(notation,(;name=(:fitted,owner,column.label,:center),symbol=center,meaning=_brmd_text(c,:centering_constant_meaning;column=column.label,owner)))
    scaled && push!(notation,(;name=(:fitted,owner,column.label,:scale),symbol=scale,meaning=_brmd_text(c,:scaling_constant_meaning;column=column.label,owner)))
    x
end

function _brmd_column_math(c,column)
    p=column.preprocess
    isnothing(p) && return get(column,:term,nothing)!==nothing ?
        _brmd_covariate_math(c,column.term) : column.source===nothing ? "1" : brm_description_symbol(c,column.source)
    if p.kind===:interaction
        operands=map(p.raw_ref) do label
            found=filter(dep->dep.label===label,p.dependencies)
            isempty(found) ? brm_description_symbol(c,label) : _brmd_column_math(c,only(found))
        end
        return join(operands,"\\,")
    elseif p.kind in (:population_factor_dummy,:ranef_factor_dummy)
        k=p.const_
        level=k.levels[k.level]
        if k.ref isa Integer
            level=level==1 ? k.ref : level==k.ref ? 1 : level
        end
        return "\\mathbf1\\{"*brm_description_symbol(c,column.source)*"="*brm_description_math(c,level)*"\\}"
    elseif p.kind===:center
        return "("*brm_description_math(c,p.raw_ref)*"-"*string(p.const_)*")"
    elseif p.kind in (:zscale,:standardize)
        anchor=p.const_
        fitted=anchor isa NamedTuple ? Tuple(values(anchor)) : anchor
        return "\\frac{"*brm_description_math(c,p.raw_ref)*"-"*string(fitted[1])*"}{"*string(fitted[2])*"}"
    elseif p.kind===:protect
        return brm_description_math(c,p.raw_ref)
    elseif p.kind in (:missing_center,:missing_zscale,:missing_standardize)
        fit=p.const_.fit
        centered="("*brm_description_math(c,p.raw_ref)*"-"*string(fit.mean)*")"
        return p.kind===:missing_center ? centered : "\\frac{"*centered*"}{"*string(fit.scale)*"}"
    end
    "X_{"*_brmd_escape(column.label)*",j}"
end
_brmd_extra_terms(_c,_x)=()
function _brmd_extra_terms(c,x::BRMDescriptionReference)
    factor=_brmd_factor_math(c,x.name)
    isnothing(factor) ? () : (factor,)
end
function _brmd_extra_terms(c,x::BRMDescriptionComponent)
    x.callable===(+) && return Tuple(t for a in x.arguments for t in _brmd_extra_terms(c,a))
    x.callable in (|,doublepipe,factor,offset,s,t2,gp,hsgp,mo1,dar,rw,cdar,car,kernel) ?
        (_brmd_term_math(c,x),) : ()
end
function _brmd_builtin_kind(::Val{:observation},c)
    lhs,rhs=c.arguments
    role=c.provenance.observation_role
    owner=c.provenance.owner
    prose= role===:conditioned ? _brmd_text(c,:observation_conditioned;owner) :
           role===:held_out ? _brmd_text(c,:observation_held_out;owner) :
           role===:latent_parameter ? _brmd_text(c,:observation_latent_parameter;owner) :
           role===:covariate_draw ? _brmd_text(c,:observation_covariate_draw;owner) :
           role===:partially_observed ? _brmd_text(c,:observation_partially_observed;owner,
               observed=c.provenance.observed_entries,missing=c.provenance.missing_entries) :
           _brmd_text(c,:observation_unconditioned;owner)
    law=rhs isa BRMDescriptionComponent ? _brmd_law(rhs.callable) : nothing
    isnothing(law) || (prose*=" "*_brmd_law_prose(c,Val(law)))
    equation=brm_description_math(c,lhs)*"\\sim{}"*_brmd_distribution_math(c,rhs)
    if length(equation)<=160
        return BRMDescriptionFragment(prose=(prose,),equations=(equation,),covers=(c.id,))
    end
    definitions=String[]; notes=NamedTuple[]; counter=Ref(0)
    reduced=_brmd_compact_expression(c,rhs,definitions,notes,counter;root=true)
    localcontext=_brmd_render_context(c;notation=(c.notation...,notes...))
    equation=_brmd_distribution_equation(localcontext,brm_description_math(localcontext,lhs),reduced)
    BRMDescriptionFragment(prose=(prose,),equations=(equation,definitions...),notation=Tuple(notes),covers=(c.id,))
end

function _brmd_render_context(c;notation=c.notation,provenance=c.provenance)
    BRMDescriptionComponent(c.id,c.kind,c.callable,c.arguments,c.keywords,c.axes,
        c.outputs,c.priors,c.fitted_constants,c.children,provenance,notation,c.bindings)
end
function _brmd_compact_expression(c,x,definitions,notes,counter;root=false,seen=())
    length(brm_description_math(c,x))<=100 && return x
    if x isa BRMDescriptionReference && x.axis!==:cell && !(x.name in seen)
        aliases=filter(b->b.name===x.name && b.role in (:alias,:constant),c.bindings)
        if length(aliases)==1 && !isequal(only(aliases).value,x)
            return _brmd_compact_expression(c,only(aliases).value,definitions,notes,counter;
                root,seen=(seen...,x.name))
        end
    end
    if x isa BRMDescriptionComponent
        args=map(a->_brmd_compact_expression(c,a,definitions,notes,counter;seen),x.arguments)
        kwargs=map(a->_brmd_compact_expression(c,a,definitions,notes,counter;seen),x.keywords)
        reduced=BRMDescriptionComponent(x.id,x.kind,x.callable,args,kwargs,x.axes,
            x.outputs,x.priors,x.fitted_constants,x.children,x.provenance,x.notation,x.bindings)
    elseif x isa Union{Tuple,NamedTuple}
        reduced=map(a->_brmd_compact_expression(c,a,definitions,notes,counter;seen),x)
    else
        return x
    end
    root && return reduced
    counter[]+=1
    key=(:description_expression,c.id,counter[])
    scope=get(c.provenance,:description_expression_scope,string(get(c.provenance,:description_number,0)))
    symbol="\\xi_{"*scope*","*string(counter[])*"}"
    push!(notes,(;name=key,symbol,meaning=_brmd_text(c,:intermediate_expression_meaning)))
    localcontext=_brmd_render_context(c;notation=(c.notation...,notes...))
    push!(definitions,_brmd_assignment_equation(localcontext,symbol,reduced))
    BRMDescriptionReference(:intermediate_expression,:local,key)
end
function _brmd_builtin_kind(::Val{:random_effect},c)
    k=c.keywords
    id=_brmd_block_math(c,c.id)
    subscript=k.by===nothing ? id : id*",s(i)"
    D="D_{"*subscript*"}"; omega="\\Omega_{"*subscript*"}"; L="L_{"*subscript*"}"; C="C_{"*subscript*"}"
    covariance=k.correlated ? D*omega*D : D*"^2"
    prose=join((_brmd_text(c,:group_deviations;group=k.group,n_terms=k.n_terms,n_groups=k.n_groups),
        _brmd_text(c,k.shared ? :shared_covariance_block : :independent_covariance_block),
        _brmd_text(c,k.correlated ? :correlated_margins : :uncorrelated_margins))," ")
    k.by===nothing || (prose*=" "*_brmd_text(c,:stratified_covariance))
    haskey(k,:total) && (prose*=" "*_brmd_text(c,:exact_total_deviations;owner=last(k.total)))
    nu=get(k,:student_t_nu,nothing)
    law=isnothing(nu) ? "\\sim\\mathcal N_{"*string(k.n_terms)*"}(0,"*covariance*")" :
        "\\sim t_{"*brm_description_math(c,nu)*","*string(k.n_terms)*"}(0,"*covariance*")"
    mixture=isnothing(nu) ? () : ("\\mathbf b_{"*id*",i}=\\sqrt{w_{"*id*",i}}\\,\\mathbf u_{"*id*",i},\\quad "*
        "\\mathbf u_{"*id*",i}\\sim\\mathcal N_{"*string(k.n_terms)*"}(0,"*covariance*"),\\quad "*
        "w_{"*id*",i}\\sim\\operatorname{InvGamma}\\left(\\tfrac{"*brm_description_math(c,nu)*"}{2},\\tfrac{"*
        brm_description_math(c,nu)*"}{2}\\right)",)
    isnothing(nu) || (prose*=" "*_brmd_text(c,:student_t_deviations;nu=nu isa Symbol ? "`$nu`" : nu))
    margins=Tuple(m for m in k.margins if m.predictor isa Symbol)
    design=Tuple("\\mathbf z_{"*_brmd_escape(owner)*","*id*",j}=["*
        join((m.predictor!==owner ? "0" : m.coefficient===:Intercept ? "1" : brm_description_symbol(c,m.coefficient)
            for m in margins),",")*"]" for owner in unique(m.predictor for m in margins))
    definition=k.correlated ? omega*"="*L*L*"^{\\mathsf T},\\quad "*C*"="*D*L : C*"="*D
    BRMDescriptionFragment(prose=(prose,),equations=("\\mathbf b_{"*id*",i}"*law,mixture...,
        definition*",\\quad "*D*"=\\operatorname{diag}(\\mathrm{SD}_{"*subscript*"})",design...),covers=(c.id,))
end
_brmd_matrix_math(x::NamedTuple)=begin
    rows,cols=x.size
    "\\begin{bmatrix}"*join((join((string(x.values[r+(j-1)*rows]) for j in 1:cols),"&") for r in 1:rows),"\\\\")*"\\end{bmatrix}"
end
function _brmd_builtin_kind(::Val{:total_effect},c)
    k=c.keywords
    blocks=map(key->_brmd_block_math(c,key),k.blocks)
    beta="\\boldsymbol\\beta="*"\\left["*join((_brmd_beta(c,label) for label in k.population_columns),",")*"\\right]^{\\mathsf T}"
    deviations="\\mathbf b_{i}=\\left["*join(("\\mathbf b_{"*b*",i}" for b in blocks),",")*"\\right]^{\\mathsf T}"
    sds="\\left["*join(("\\mathrm{SD}_{"*b*"}" for b in blocks),",")*"\\right]"
    sd="\\operatorname{diag}\\left("*sds*"\\right)"
    precision="\\operatorname{diag}(q)"*(isempty(k.mixture) ? "" : "\\operatorname{diag}(\\lambda)")
    prose=String[_brmd_text(c,:exact_total_effect;owner=c.provenance.owner,group=k.group,
        columns=join((string(l) for l in k.population_columns),", "))]
    isempty(k.mixture) || push!(prose,_brmd_text(c,:student_t_scale_mixture))
    any(iszero,k.precision) && push!(prose,_brmd_text(c,:zero_precision))
    group_columns=get(k,:group_columns,())
    isempty(group_columns) && return BRMDescriptionFragment(prose=Tuple(prose),equations=(
        "\\mathbf t_{i}=A\\boldsymbol\\beta+\\mathbf b_{i},\\quad "*beta*",\\quad "*deviations,
        "A="*_brmd_matrix_math(k.A)*",\\quad m="*brm_description_math(c,k.location)*",\\quad q="*brm_description_math(c,k.precision),
        "p(\\mathbf t\\mid\\mathrm{SD})="*_brmd_law_math(c,Val(:exact_total),(sds,)),
        "\\boldsymbol\\beta\\mid\\mathbf t\\sim\\mathcal N\\left(Q^{-1}h,Q^{-1}\\right),\\quad Q="*precision*"+J\\,A^{\\mathsf T}"*sd*"^{-2}A,\\quad h="*precision*"m+A^{\\mathsf T}"*sd*"^{-2}\\sum_{i=1}^{J}\\mathbf t_{i}"),
        covers=(c.id,))
    push!(prose,_brmd_text(c,length(group_columns)==1 ? :group_level_predictor : :group_level_predictors;
        columns=join(("`$l`" for l in group_columns),", "),group=k.group))
    weights="\\mathbf w_{i}=\\left["*join((label in group_columns ? "{"*brm_description_symbol(c,label)*"}_{i}" : "1"
        for label in k.population_columns),",")*"\\right]^{\\mathsf T}"
    rows="\\mathbf a_{k}"
    BRMDescriptionFragment(prose=Tuple(prose),equations=(
        "\\mathbf t_{i}=A(\\mathbf w_{i}\\circ\\boldsymbol\\beta)+\\mathbf b_{i},\\quad "*beta*",\\quad "*weights*",\\quad "*deviations,
        "A="*_brmd_matrix_math(k.A)*",\\quad m="*brm_description_math(c,k.location)*",\\quad q="*brm_description_math(c,k.precision),
        "p(\\mathbf t\\mid\\mathrm{SD})="*_brmd_exact_total_math(sds,true),
        "\\boldsymbol\\beta\\mid\\mathbf t\\sim\\mathcal N\\left(Q^{-1}h,Q^{-1}\\right),\\quad Q="*precision*"+\\sum_{k}\\mathrm{SD}_{k}^{-2}\\,("*rows*rows*"^{\\mathsf T})\\circ(W^{\\mathsf T}W),\\quad h="*precision*"m+\\sum_{k}\\mathrm{SD}_{k}^{-2}\\,"*rows*"\\circ(W^{\\mathsf T}\\mathbf t_{\\cdot k})",
        _brmd_text(c,:math_total_row)*",\\quad \\mathbf t_{\\cdot k}=[t_{1k},\\ldots,t_{Jk}]^{\\mathsf T}"),
        covers=(c.id,))
end
_brmd_builtin_kind(::Val{K},_c) where K = nothing

function _brmd_notation(d,labels)
    # An exact-total scale carrier is rendered as its blocks' SDs, never by name.
    carriers=Set(t.binding.scale for t in _brmd_total_bindings(d))
    names=unique((d.columns..., (o.logical for o in d.outputs if o.logical !== nothing && !(o.logical in carriers))...,
        sort!(collect(keys(labels));by=string)...))
    result=NamedTuple[]
    for name in names
        label=get(labels,name,NamedTuple())
        label isa AbstractString && (label=(; meaning=String(label)))
        label isa NamedTuple || throw(ArgumentError("labels[$name] must be text or a notation NamedTuple"))
        push!(result,merge((; name,meaning=string(name),axis=:declared,
            meaning_supplied=haskey(label,:meaning)),_brmd_snapshot(label)))
    end
    Tuple(result)
end

_brmd_block_math(c,key)=brm_description_symbol(c,key)
function _brmd_block_notation(notation,groups,labels,texts)
    result=NamedTuple[notation...]
    for (index,group) in enumerate(sort!(collect(groups);by=g->string(g.key)))
        supplied=get(labels,group.key,NamedTuple())
        supplied isa AbstractString && (supplied=(;meaning=String(supplied)))
        margins=join((_brmd_path_markdown((m.predictor,m.coefficient)) for m in group.margins),"; ")
        note=merge((;name=group.key,symbol=string(index),axis=:covariance_block,
            meaning=_brmd_text(texts,:covariance_block_meaning;index,margins)),supplied)
        filter!(n->n.name!=group.key,result)
        push!(result,note)
    end
    Tuple(result)
end

function _brmd_hook(c,hooks)
    matches=filter(p -> first(p) === c.callable,hooks)
    length(matches)>1 && throw(ArgumentError("description: duplicate hooks for $(c.callable)"))
    fragment=isempty(matches) ? brm_describe_component(c.callable,c) : last(only(matches))(c)
    isnothing(fragment) || fragment isa BRMDescriptionFragment || throw(ArgumentError("description hook must return BRMDescriptionFragment or nothing"))
    fragment
end

"""
    brm_description(d::BRMDescriptor; hooks=(), labels=Dict(), prior_anchors=Dict(), texts=description_texts())
    brm_description(prepared::Union{SBBRMI,GenerativePlan}; ...)

Describe the selected prepared model without fitting or sampling. Per-call hooks
are actual-callable => context -> fragment pairs. `labels` binds logical symbols
and supplied meanings/units. `prior_anchors` maps logical prior tuple IDs to
report links. `texts` selects the language BRM writes in, e.g.
`description_texts(:de)`; the description keeps it for its Markdown and hooks
read it with `brm_description_texts(context)`. Unsupported semantic nodes remain
explicit coverage gaps; an incomplete result is never labeled a complete model
description.
"""
function brm_description(d::BRMDescriptor; hooks=(),labels=Dict(),prior_anchors=Dict(),
        texts::BRMDescriptionTexts=_BRMD_EN_TEXTS)
    notation=_brmd_notation(d,labels)
    groups=_brmd_ranef_metadata(d)
    notation=_brmd_block_notation(notation,groups,labels,texts)
    notation=_brmd_allocation_notation(d,notation,labels,texts)
    constants=_brmd_constants(d.plan)
    priors,groups=_brmd_priors(d,prior_anchors,notation,texts;groups)
    roots=collect(_brmd_components(d,priors,constants,notation,texts;groups))
    # Inventory allocation semantics explicitly; a Gaussian conditional prior
    # alone must never establish coverage of an unexplained hierarchical scale.
    append!(roots,_brmd_allocation_components(d,priors,constants,notation,texts,groups))
    for group in groups
        env=_brmd_environment(d,group.key,:random_effect,priors,constants,notation,texts)
        b=group.block
        kwargs=(; group=b.group,id=b.id,n_terms=b.n_terms,n_groups=b.n_groups,
            levels=_brmd_snapshot(b.levels),margins=group.margins,
            correlated=group.correlated,shared=group.shared,noncentered=b.noncentered,by=b.by,
            student_t_nu=_brmd_ranef_student_t_nu(group))
        haskey(group,:total) && (kwargs=merge(kwargs,(;total=(:total_effect,group.total.binding.logical))))
        push!(roots,_brmd_component(env,group.key,:random_effect,nothing,(),kwargs))
    end
    for t in _brmd_total_bindings(d)
        b=t.binding.total
        env=_brmd_environment(d,b.predictor,:total_effect,priors,constants,notation,texts;groups)
        carriers=(b.binding,b.scales,b.population,b.deviations)
        outputs=Tuple((;name=o.name,logical=o.logical,role=o.role,kind=o.kind,
            segments=_brmd_snapshot(o.segments)) for o in d.outputs if o.name in carriers)
        kwargs=(; predictor=b.predictor,group=b.group,
            blocks=Tuple(g.key for g in groups if haskey(g,:total) && g.block.binding===t.name),
            columns=b.columns,population_columns=b.population_columns,A=_brmd_snapshot(b.A),
            location=_brmd_snapshot(b.location),precision=_brmd_snapshot(b.precision),mixture=Tuple(b.mixture))
        isempty(b.group_columns) || (kwargs=merge(kwargs,(; group_columns=b.group_columns)))
        push!(roots,_brmd_component(env,(:total_effect,b.predictor),:total_effect,nothing,(),kwargs;outputs))
    end
    for prior in priors
        prior.distribution isa BRMDescriptionComponent && push!(roots,prior.distribution)
    end
    inventory=Tuple(node for root in roots for node in brm_description_components(root))
    ids=Set(c.id for c in inventory)
    length(ids)==length(inventory) || error("description: duplicate logical semantic component IDs")
    covered=Set{Tuple}()
    prose=String[]; equations=String[]; added_notation=NamedTuple[]
    for (index,c) in enumerate(inventory)
        fragment=_brmd_hook(c,hooks)
        if isnothing(fragment)
            rendered=_brmd_render_context(c;provenance=merge(c.provenance,(;description_number=index)))
            fragment=_brmd_builtin_fragment(rendered)
        end
        isnothing(fragment) && continue
        subtree=Set(node.id for node in brm_description_components(c))
        for id in fragment.covers
            id in ids || throw(ArgumentError("description hook claimed nonexistent component $id"))
            id in subtree || throw(ArgumentError("description hook claimed a component outside its subtree: $id"))
            push!(covered,id)
        end
        append!(prose,fragment.prose); append!(equations,fragment.equations)
        append!(added_notation,fragment.notation)
    end
    coverage=Tuple(BRMDescriptionCoverage(c.id,c.id in covered ? :covered : :unsupported,
        c.id in covered ? "" : _brmd_text(texts,:coverage_gap;callable=_brmd_callable_name(c.callable),id=c.id),
        c.provenance) for c in inventory)
    diagnostics=Tuple(x.reason for x in coverage if x.status !== :covered)
    BRMDescription(d.id,Tuple(roots),Tuple(prose),Tuple(equations),
        (notation...,added_notation...),priors,coverage,diagnostics,isempty(diagnostics),texts)
end
brm_description(prepared::Union{SBBRMI,GenerativePlan};kwargs...) =
    brm_description(brm_descriptor(prepared);kwargs...)

function _brmd_family_caption(x::BRMDescriptionComponent)
    coordinates=_brmd_coordinates(x)
    isempty(coordinates) && return "`"*string(_brmd_callable_name(x.callable))*"`"
    _brmd_text(x,:coordinatewise_family;
        families=join(unique("`"*string(_brmd_callable_name(c.callable))*"`" for c in coordinates),", "))
end

function _brmd_prior_equation(p)
    c=p.distribution
    c isa BRMDescriptionComponent || return string(p.distribution)
    support=isempty(p.support) ? "" : "\\quad " * join(
        (_brmd_identifier(k)*"="*brm_description_math(c,v) for (k,v) in pairs(p.support)),", ")
    _brmd_distribution_math(c,c)*support
end
_brmd_path_markdown(id::Tuple)=join((part isa Tuple ? "("*_brmd_path_markdown(part)*")" : _brmd_path_markdown(part) for part in id)," / ")
_brmd_path_markdown(part)="`"*replace(string(part),"|"=>"\\|")*"`"
_brmd_prior_id_markdown(id)=_brmd_path_markdown(id)
function _brmd_prior_definition(p,label)
    c=p.distribution
    c isa BRMDescriptionComponent || return (string(p.distribution),)
    definitions=String[]; notes=NamedTuple[]; counter=Ref(0)
    context=_brmd_render_context(c;provenance=merge(c.provenance,
        (;description_expression_scope=label)))
    reduced=_brmd_compact_expression(context,c,definitions,notes,counter;root=true)
    support=map(Tuple(pairs(p.support))) do (k,v)
        value=_brmd_compact_expression(context,v,definitions,notes,counter;root=true)
        localcontext=_brmd_render_context(context;notation=(context.notation...,notes...))
        _brmd_assignment_equation(localcontext,_brmd_identifier(k),value)
    end
    localcontext=_brmd_render_context(context;notation=(context.notation...,notes...))
    (_brmd_distribution_math(localcontext,reduced),support...,definitions...)
end

"""
    brm_description_markdown(description)

Render deterministic Markdown with display LaTeX, an effective prior table,
notation entries and explicit coverage diagnostics. Every prior row owns a model-scoped
anchor; component references link to these targets by default. `prior_anchors`
remain additional outbound links. Use `prefix` for repeated model instances.
The page is written in the description's language (`texts=` of `brm_description`).
"""
function brm_description_markdown(description::BRMDescription;prefix=nothing)
    io=IOBuffer()
    t=description.texts
    text(name;values...)=_brmd_text(t,name;values...)
    println(io,text(description.complete ? :complete_description : :incomplete_description))
    for paragraph in description.prose
        println(io,"\n",paragraph)
    end
    for equation in description.equations
        println(io,"\n\$\$\n",equation,"\n\$\$")
    end
    if !isempty(description.priors)
        targets=Dict{String,Tuple}()
        for p in description.priors
            target=brm_description_prior_anchor(p;prefix)
            haskey(targets,target) && throw(ArgumentError(
                "priors $(targets[target]) and $(p.id) share the link target $target"))
            targets[target]=p.id
        end
        println(io,"\n",text(:effective_priors),"\n\n| ",text(:logical_parameter_column)," | ",
            text(:distribution_column)," | ",text(:prior_listing_column)," |\n| --- | --- | --- |")
        for (index,p) in enumerate(description.priors)
            label=text(:prior_prefix)*string(index)
            family=p.distribution isa BRMDescriptionComponent ? _brmd_family_caption(p.distribution) : text(:declared_prior)
            link=isnothing(p.anchor) ? "" : "["*text(:prior_listing_link)*"]("*replace(p.anchor," "=>"%20")*")"
            println(io,"| <span id=\"",brm_description_prior_anchor(p;prefix),"\"></span>",label," | ",
                text(:prior_definition_reference;family,label)," | ",link," |")
        end
        println(io,"\n",text(:prior_definitions))
        for (index,p) in enumerate(description.priors)
            label=text(:prior_prefix)*string(index)
            println(io,"\n**",label,".** ",text(:prior_logical_parameter;id=_brmd_prior_id_markdown(p.id)))
            for equation in _brmd_prior_definition(p,label)
                println(io,"\n\$\$\n",equation,"\n\$\$")
            end
        end
        prior_keys=Dict(p.id=>text(:prior_prefix)*string(index) for (index,p) in enumerate(description.priors))
        seen=Set{Tuple}()
        links=Tuple(c=>brm_description_prior_references(c) for c in description.components
            if c.provenance.declaration!==:prior)
        any(p->!isempty(last(p)),links) && println(io,"\n",text(:component_prior_references),"\n\n| ",
            text(:component_column)," | ",text(:effective_priors_column)," |\n| --- | --- |")
        component_ids=Tuple{Int,Tuple}[]
        for (c,ids) in links
            isempty(ids) && continue
            ids in seen && continue
            push!(seen,ids)
            index=length(component_ids)+1
            push!(component_ids,(index,c.id))
            references=join(("["*prior_keys[id]*"](#"*
                brm_description_prior_anchor(description,id;prefix)*")" for id in ids),", ")
            println(io,"| ",text(:component_prefix),index," | ",references," |")
        end
        if !isempty(component_ids)
            println(io,"\n",text(:component_reference_keys))
            for (index,id) in component_ids
                println(io,"\n**",text(:component_prefix),index,".** ",_brmd_path_markdown(id),".")
            end
        end
    end
    if !isempty(description.notation)
        println(io,"\n",text(:notation))
        for (index,n) in enumerate(description.notation)
            println(io,"\n**",text(:notation_prefix),index,".** ",text(:quantity;name=_brmd_path_markdown(n.name)))
            haskey(n,:symbol) && println(io,"\n",text(:symbol;symbol=n.symbol))
            meaning=get(n,:meaning,string(n.name))
            get(n,:meaning_supplied,haskey(n,:meaning)) && println(io,"\n",text(:meaning;meaning))
            println(io,"\n",text(:axis;axis=_brmd_path_markdown(get(n,:axis,:declared))))
            haskey(n,:unit) && println(io,"\n",text(:supplied_unit;unit=n.unit))
        end
    end
    if !description.complete
        println(io,"\n",text(:coverage_gaps))
        for diagnostic in description.diagnostics
            println(io,"\n- ",diagnostic)
        end
    end
    String(take!(io))
end
