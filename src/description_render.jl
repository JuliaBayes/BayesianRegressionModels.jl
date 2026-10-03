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
    found = filter(n -> n.name == name, context.notation)
    isempty(found) || return get(only(found), :symbol, _brmd_identifier(name))
    _brmd_identifier(name)
end

"""
    brm_description_math(context, argument)

Render any public argument/reference/component as escaped LaTeX using the
context's notation. Rendering an unknown call does not establish coverage.
"""
function brm_description_math(c::BRMDescriptionContext,x::BRMDescriptionReference)
    aliases=filter(b -> b.name===x.name && b.role in (:alias,:constant,:deterministic),c.bindings)
    if !isempty(aliases)
        value=only(aliases).value
        isequal(value,x) || return brm_description_math(c,value)
    end
    x.logical isa Tuple && any(n -> n.name==x.logical,c.notation) &&
        return brm_description_symbol(c,x.logical)
    brm_description_symbol(c,x.name)
end
brm_description_math(_c::BRMDescriptionContext, x::Number) = string(x)
brm_description_math(_c::BRMDescriptionContext, x::AbstractString) =
    "\\text{" * _brmd_escape(x) * "}"
brm_description_math(c::BRMDescriptionContext, xs::Tuple) =
    "\\left[" * join((brm_description_math(c,x) for x in xs), ", ") * "\\right]"
brm_description_math(_c::BRMDescriptionContext, x::Symbol) = _brmd_identifier(x)
function brm_description_math(c::BRMDescriptionContext,x::NamedTuple)
    haskey(x,:size) && haskey(x,:values) && return "\\operatorname{reshape}\\left("*
        brm_description_math(c,x.values)*","*join(x.size,",")*"\\right)"
    haskey(x,:callable) && return _brmd_call_math(x.callable,
        map(a->brm_description_math(c,a),x.arguments),x.keywords,c)
    _brmd_identifier(string(x))
end
brm_description_math(_c::BRMDescriptionContext, x) = _brmd_identifier(string(x))
_brmd_callable_name(f::Symbol) = f
_brmd_callable_name(f::Union{Function,Type}) = nameof(f)
_brmd_callable_name(f::Union{StanBlocks.ValueUDF,StanBlocks.ValueFamily}) = f.label
_brmd_callable_name(f) = Symbol(nameof(typeof(f)))
brm_description_math(_c::BRMDescriptionContext,x::Union{Function,Type,StanBlocks.ValueUDF,StanBlocks.ValueFamily,StanBlocks.SlicModel}) =
    _brmd_identifier(_brmd_callable_name(x))

function brm_description_math(c::BRMDescriptionContext, x::BRMDescriptionComponent)
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
_brmd_call_math(::typeof(exp), args, _kwargs, _c) = "\\exp\\left(" * only(args) * "\\right)"
_brmd_call_math(::typeof(log), args, _kwargs, _c) = "\\log\\left(" * only(args) * "\\right)"
_brmd_call_math(::typeof(sqrt), args, _kwargs, _c) = "\\sqrt{" * only(args) * "}"
_brmd_call_math(::typeof(logistic), args, _kwargs, _c) = "\\operatorname{logit}^{-1}\\left(" * only(args) * "\\right)"

# Distribution equations use explicit parameter conventions. A normal's second
# constructor argument is an SD; the conventional Gaussian equation uses variance.
_brmd_law(_f) = nothing
_brmd_law(::_BRMDFlatPrior) = :improper_flat
_brmd_law(::Type{<:Normal}) = :normal_sd
_brmd_law(::Type{<:Exponential}) = :exponential_scale
_brmd_law(::Type{<:LogNormal}) = :lognormal_sd
_brmd_law(::Type{<:Gamma}) = :gamma_scale
_brmd_law(::Type{<:Beta}) = :beta
_brmd_law(::Type{<:Cauchy}) = :cauchy
_brmd_law(::Type{<:TDist}) = :student_t
_brmd_law(::Type{<:Bernoulli}) = :bernoulli
_brmd_law(::Type{<:BernoulliLogit}) = :bernoulli_logit
_brmd_law(::Type{<:Binomial}) = :binomial
_brmd_law(::Type{<:BinomialLogit}) = :binomial_logit
_brmd_law(::Type{<:Poisson}) = :poisson
_brmd_law(::Type{<:NegativeBinomial}) = :negative_binomial
_brmd_law(::Type{<:Dirichlet}) = :dirichlet
_brmd_law(::Type{<:LKJCholesky}) = :lkj
_brmd_law(::Type{<:Weibull}) = :weibull
_brmd_law(::typeof(MvNormalCholesky)) = :mvnormal_cholesky
# These are exact producer-owned StanBlocks bindings, never a name heuristic
# applied to arbitrary user callables.
for (name,law) in ((:normal,:normal_sd),(:std_normal,:standard_normal),
        (:exponential,:exponential_rate),(:lognormal,:lognormal_sd),
        (:gamma,:gamma_rate),(:beta,:beta),(:cauchy,:cauchy),(:student_t,:student_t),
        (:bernoulli,:bernoulli),(:bernoulli_logit,:bernoulli_logit),
        (:binomial,:binomial),(:binomial_logit,:binomial_logit),(:poisson,:poisson),
        (:dirichlet,:dirichlet),(:lkj_corr_cholesky,:lkj),(:weibull,:weibull))
    if isdefined(StanBlocks.stan.builtin,name)
        f=getfield(StanBlocks.stan.builtin,name)
        @eval _brmd_law(::$(typeof(f))) = $(QuoteNode(law))
    end
end
_brmd_law_math(::Val{:normal_sd},a) = "\\mathcal N\\left(" *
    (isempty(a) ? "0,1" : first(a)*","*(length(a)<2 ? "1" : "{"*a[2]*"}^{2}")) * "\\right)"
_brmd_law_math(::Val{:standard_normal},_a) = "\\mathcal N(0,1)"
_brmd_law_math(::Val{:improper_flat},_a) = "\\operatorname{Flat}_{\\mathrm{improper}}"
_brmd_law_math(::Val{:exponential_scale},a) = "\\operatorname{Exponential}_{\\mathrm{scale}}(" *
    (isempty(a) ? "1" : only(a)) * ")"
_brmd_law_math(::Val{:exponential_rate},a) = "\\operatorname{Exponential}_{\\mathrm{rate}}(" * only(a) * ")"
_brmd_law_math(::Val{:gamma_scale},a) = "\\operatorname{Gamma}_{\\mathrm{shape,scale}}(" * join(a,",") * ")"
_brmd_law_math(::Val{:gamma_rate},a) = "\\operatorname{Gamma}_{\\mathrm{shape,rate}}(" * join(a,",") * ")"
_brmd_law_math(::Val{:lognormal_sd},a) = "\\operatorname{LogNormal}_{\\mathrm{log\\,SD}}(" * join(a,",") * ")"
_brmd_law_math(::Val{:bernoulli_logit},a) = "\\operatorname{Bernoulli}(\\operatorname{logit}^{-1}(" * only(a) * "))"
_brmd_law_math(::Val{:binomial_logit},a) = "\\operatorname{Binomial}(" * a[1] * ",\\operatorname{logit}^{-1}(" * a[2] * "))"
_brmd_law_math(::Val{:lkj},a) = "\\operatorname{LKJCholesky}(" * join(a,",") * ")"
_brmd_law_math(::Val{:mvnormal_cholesky},a) = "\\mathcal N(" * a[1] * "," * a[2] * a[2] * "^{\\mathsf T})"
_brmd_law_math(::Val{L},a) where L = "\\operatorname{" * _brmd_escape(L) * "}(" * join(a,",") * ")"
_brmd_law_prose(::Val{:normal_sd}) = "Gaussian distribution; the second argument is the residual standard deviation."
_brmd_law_prose(::Val{:standard_normal}) = "Standard Gaussian distribution."
_brmd_law_prose(::Val{:improper_flat}) = "Unpenalized coefficient with an improper flat prior (constant log density)."
_brmd_law_prose(::Val{:exponential_rate}) = "Exponential distribution parameterized by rate; its mean/scale is the reciprocal of that rate."
_brmd_law_prose(::Val{:exponential_scale}) = "Exponential distribution parameterized by scale (its mean), not rate."
_brmd_law_prose(::Val{:gamma_rate}) = "Gamma distribution parameterized by shape and rate."
_brmd_law_prose(::Val{:gamma_scale}) = "Gamma distribution parameterized by shape and scale."
_brmd_law_prose(::Val{:lognormal_sd}) = "Lognormal distribution; location and standard deviation are on the log scale."
_brmd_law_prose(::Val{L}) where L = "$(L) distribution with the declared arguments."

function _brmd_distribution_math(c,x::BRMDescriptionComponent)
    law = _brmd_law(x.callable)
    isnothing(law) && return brm_description_math(c,x)
    _brmd_law_math(Val(law),map(a -> brm_description_math(c,a),x.arguments))
end
_brmd_distribution_math(c,x) = brm_description_math(c,x)

_brmd_covariate_math(c,x) = brm_description_math(c,x)
function _brmd_covariate_math(c,x::BRMDescriptionComponent)
    _brmd_transformed_math(x.callable,c,x)
end
_brmd_transformed_math(_f,c,x) = brm_description_math(c,x)
function _brmd_transform_constant(c,x,kind)
    source = isempty(x.arguments) ? nothing : first(x.arguments)
    source isa BRMDescriptionReference || return nothing
    entries = filter(e -> e.kind === kind && e.source isa BRMDescriptionReference &&
                           e.source.name === source.name,c.fitted_constants)
    length(entries)==1 ? only(entries).value : nothing
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
for f in (s,t2,gp,hsgp,me,mi,mo,mo1,ar,dar,rw,cdar,interval_censored,kernel)
    @eval _brmd_term_call_math(::$(typeof(f)),c,x) = brm_description_math(c,x)
end
function _brmd_group_term_math(c,x)
    effects=first(x.arguments)
    args=effects isa BRMDescriptionComponent && effects.callable === (+) ? effects.arguments : (effects,)
    group=last(x.arguments)
    group_name=group isa BRMDescriptionReference ? group.name : :group
    join((a==1 ? "b_{"*_brmd_escape(c.provenance.owner)*",0,"*_brmd_escape(group_name)*"}" :
          "b_{"*_brmd_escape(c.provenance.owner)*","*_brmd_escape(group_name)*"}\\,"*brm_description_math(c,a)
          for a in args if a!=0)," + ")
end
_brmd_term_call_math(::typeof(|),c,x) = _brmd_group_term_math(c,x)
_brmd_term_call_math(::typeof(doublepipe),c,x) = _brmd_group_term_math(c,x)

# Built-in semantic coverage is explicit. Unknown callable identities stay gaps.
_brmd_builtin_call(_f,c) = begin
    law=_brmd_law(c.callable)
    isnothing(law) ? nothing : BRMDescriptionFragment(covers=(c.id,))
end
for f in (+,-,*,/,^,exp,log,sqrt,logistic,log1pexp,abs,sum,cumsum,maximum,minimum,
          getindex,getproperty,identity,|,doublepipe,&,gr,offset,protect,factor,ragged,
          center,zscale,standardize)
    @eval _brmd_builtin_call(::$(typeof(f)),c) = BRMDescriptionFragment(covers=(c.id,))
end
_brmd_builtin_call(::StanBlocks.SlicModel,c)=BRMDescriptionFragment(covers=(c.id,))
function _brmd_bound_math(c,key,position,fallback)
    value=get(c.keywords,key,length(c.arguments)>=position ? c.arguments[position] : fallback)
    value===nothing ? fallback : brm_description_math(c,value)
end
function _brmd_builtin_call(::typeof(truncated),c)
    lower=_brmd_bound_math(c,:lower,2,"-\\infty")
    upper=_brmd_bound_math(c,:upper,3,"\\infty")
    BRMDescriptionFragment(prose=("Truncation conditions the declared base distribution on its bounds. F is its CDF; F(L⁻) is the left limit, which preserves inclusive discrete bounds.",),
        equations=("p(y)=\\frac{f(y)\\,\\mathbf1\\{"*lower*"\\le y\\le "*upper*"\\}}{F("*upper*")-F("*lower*"^-)}",),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(censored),c)
    lower=_brmd_bound_math(c,:lower,2,"-\\infty")
    upper=_brmd_bound_math(c,:upper,3,"\\infty")
    BRMDescriptionFragment(prose=("Censoring contributes tail probability at a reported boundary and the base density or mass for exact interior values. F is the base CDF; its left limit handles discrete upper boundaries.",),
        equations=("p(y)=\\begin{cases}F("*lower*")&y="*lower*"\\\\f(y)&"*lower*"<y<"*upper*"\\\\1-F("*upper*"^-)&y="*upper*"\\end{cases}",),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(interval_censored),c)
    base=first(c.arguments)
    if base isa BRMDescriptionReference
        return BRMDescriptionFragment(prose=("Predictor values at the declared upper detection limit are latent within their bounds; values above the limit remain exact. The latent distribution and bounds are in the effective prior inventory.",),covers=(c.id,))
    end
    upper=_brmd_bound_math(c,:upper,2,"U_j")
    BRMDescriptionFragment(prose=("The observed response is the lower interval endpoint. Continuous intervals contribute F(U)−F(L); integer intervals include both endpoints and contribute F(U)−F(L−1). F is the declared base CDF.",),
        equations=("\\mathcal L_j=F("*upper*")-F(L_j)\\quad\\text{(continuous)}", "\\mathcal L_j=F("*upper*")-F(L_j-1)\\quad\\text{(integer)}"),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(weighted),c)
    weight=c.arguments[2]
    w=weight isa BRMDescriptionComponent && !isempty(weight.arguments) ?
        brm_description_math(c,first(weight.arguments)) : brm_description_math(c,weight)
    if weight isa BRMDescriptionComponent && weight.callable===aweights
        return BRMDescriptionFragment(prose=("Analytic weights modify Gaussian precision: the residual SD for row j is σ divided by √wⱼ. This includes the Gaussian normalization for that adjusted SD.",),
            equations=("y_j\\sim\\mathcal N(\\mu_j,\\sigma^2/"*w*")",),covers=(c.id,))
    end
    BRMDescriptionFragment(prose=("Each declared observation weight multiplies its pointwise log likelihood; weights do not replace the response distribution.",),
        equations=("\\log\\mathcal L=\\sum_j "*w*"\\log p(y_j\\mid\\theta)",),covers=(c.id,))
end
for f in (aweights,fweights,weights)
    @eval _brmd_builtin_call(::$(typeof(f)),c)=BRMDescriptionFragment(covers=(c.id,))
end
function _brmd_builtin_call(::typeof(StanBlocks.stan.builtin.maybe_index),c)
    BRMDescriptionFragment(prose=("The imputation split uses scalar distribution parameters unchanged and selects the missing-row indices from vector parameters.",),covers=(c.id,))
end
function _brmd_call_math(::typeof(StanBlocks.stan.builtin.maybe_index),args,_kwargs,c)
    "\\operatorname{select}_{\\mathrm{scalar/vector}}\\left("*join(args,",")*"\\right)"
end
function _brmd_builtin_call(::typeof(kernel),c)
    BRMDescriptionFragment(prose=("The kernel runs once per declared group; its arguments preserve their own row or event axes. Its scientific calls and cell statements are covered separately.",),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(s),c)
    BRMDescriptionFragment(prose=("A thin-plate spline uses the fitted basis and penalized coefficients, with a separately declared smoothing scale.",),
        equations=("f(x_j)=X_{\\mathrm{null},j}\\beta+Z_{\\mathrm{pen},j}u,\\quad u\\mid\\tau\\sim\\mathcal N(0,\\tau^2 I)",),covers=(c.id,))
end
function _brmd_builtin_call(::typeof(t2),c)
    BRMDescriptionFragment(prose=("The tensor smooth combines the two fitted marginal bases; its rr, rn and nr penalty blocks have independent smoothing scales.",),
        equations=("f(x_j,z_j)=X_{0,j}\\beta+\\sum_{h\\in\\{rr,rn,nr\\}}Z_{h,j}u_h,\\quad u_h\\mid\\tau_h\\sim\\mathcal N(0,\\tau_h^2I)",),covers=(c.id,))
end
function _brmd_gp_fragment(c,approximate)
    covariance=get(c.keywords,:cov,:exp_quad)
    covname=covariance isa BRMDescriptionReference ? covariance.name : covariance
    prose=("The $(approximate ? "Hilbert-space approximation to a Gaussian process" : "Gaussian process") uses covariance $(covname), the selected fitted domain and declared length-scale and marginal-SD parameters. Grouping and options are retained in the component context.",)
    equation=approximate ?
        "f(x_j)=\\sum_{k=1}^{M}\\phi_k(x_j;\\mathcal D_{\\mathrm{fit}})\\sqrt{S(\\omega_k;\\ell,\\tau)}z_k,\\quad z_k\\sim\\mathcal N(0,1)" :
        "f\\mid\\ell,\\tau\\sim\\mathcal N(0,K),\\quad K_{ab}=k(x_a,x_b;\\ell,\\tau)"
    BRMDescriptionFragment(; prose,equations=(equation,),covers=(c.id,))
end
_brmd_builtin_call(::typeof(gp),c) = _brmd_gp_fragment(c,false)
_brmd_builtin_call(::typeof(hsgp),c) = _brmd_gp_fragment(c,true)
function _brmd_builtin_call(::typeof(me),c)
    BRMDescriptionFragment(prose=("Measurement error is modeled with a latent covariate and the declared measurement standard deviation; the latent prior is listed separately.",),
        equations=(brm_description_math(c,c.arguments[1])*"\\mid x^*\\sim\\mathcal N(x^*,{"*brm_description_math(c,c.arguments[2])*"}^2)",),covers=(c.id,))
end
_brmd_builtin_call(::typeof(mi),c) = BRMDescriptionFragment(
    prose=("Missing entries are inferred from the declared joint model; observed entries remain conditioned data. No deterministic substitution is implied.",),covers=(c.id,))
for f in (mo,mo1)
    @eval _brmd_builtin_call(::$(typeof(f)),c) = BRMDescriptionFragment(
        prose=("The ordered predictor uses cumulative simplex shares over the fitted level order. `mo` has a sampled population coefficient; `mo1` adds the unit-amplitude contrast directly. Simplex priors are listed separately.",),
        equations=("m(k)=\\sum_{h<k}\\zeta_h,\\quad\\zeta_h\\ge0,\\quad\\sum_h\\zeta_h=1,\\quad m(1)=0",),covers=(c.id,))
end
_brmd_builtin_call(::typeof(ar),c)=BRMDescriptionFragment(
    prose=("The AR(1) state is added through a sampled population coefficient. Its persistence is tanh of a standard Gaussian parameter; innovations are standard Gaussian.",),
    equations=("a_t=\\phi a_{t-1}+\\epsilon_t,\\quad\\phi=\\tanh(\\phi_{\\mathrm{raw}}),\\quad\\epsilon_t\\sim\\mathcal N(0,1)",),covers=(c.id,))
_brmd_builtin_call(::typeof(cdar),c)=BRMDescriptionFragment(
    prose=("The grouped damped walk uses the declared fitted correlation factor L. Its initial state has stationary covariance σ²LLᵀ, and its subsequent innovations preserve that covariance.",),
    equations=("\\delta_1=\\sigma Lz_1,\\quad\\delta_t=\\rho\\delta_{t-1}+\\sigma\\sqrt{1-\\rho^2}Lz_t,\\quad z_t\\sim\\mathcal N(0,I)",),covers=(c.id,))
# Temporal terms require their recurrence, rather than just a term caption.
_brmd_builtin_call(::typeof(rw),c) = BRMDescriptionFragment(
    prose=("A random walk adds innovations on the declared ordered time axis.",),
    equations=("f_{t+1}=f_t+\\sigma z_t,\\quad z_t\\sim\\mathcal N(0,1)",),covers=(c.id,))
_brmd_builtin_call(::typeof(dar),c) = BRMDescriptionFragment(
    prose=("The first differences follow AR(1); the formula intercept supplies the initial level, and the integrated trajectory has no additional population coefficient.",),
    equations=("d_t=\\rho d_{t-1}+\\sigma z_t,\\quad f_{t+1}=f_t+d_t,\\quad z_t\\sim\\mathcal N(0,1)",),covers=(c.id,))

function _brmd_builtin_fragment(c)
    _brmd_builtin_kind(Val(c.kind),c)
end
_brmd_builtin_kind(::Val{:call},c) = _brmd_builtin_call(c.callable,c)
_brmd_builtin_kind(::Val{:submodel},c) = BRMDescriptionFragment(covers=(c.id,))
_brmd_builtin_kind(::Val{:submodel_output},c)=BRMDescriptionFragment(
    prose=("`$(c.provenance.owner)` is the returned value of its included scientific submodel, whose internal parameters and calls are described separately.",),covers=(c.id,))
_brmd_builtin_kind(::Val{:syntax},c) = c.callable in (:block,:tuple,:vect,:ref,:(=),:return,:->,:.,:(::),:kw,:parameters) ?
    BRMDescriptionFragment(covers=(c.id,)) : nothing
_brmd_builtin_kind(::Val{:parameter},c) = BRMDescriptionFragment(covers=(c.id,))
_brmd_builtin_kind(::Val{:assignment},c) = BRMDescriptionFragment(
    prose=("`$(c.provenance.owner)` is a deterministic assignment of its declared arguments.",),
    equations=(brm_description_math(c,c.arguments[1])*"="*brm_description_math(c,c.arguments[2]),),covers=(c.id,))
function _brmd_builtin_kind(::Val{:predictor},c)
    lhs,rhs=c.arguments
    columns=c.provenance.design_columns
    if columns===nothing
        body=_brmd_term_math(c,rhs)
        hasintercept=occursin("Intercept",body)
    else
        fixed=Tuple(_brmd_beta(c,col.label)*(col.source===nothing ? "" : "\\,"*_brmd_column_math(c,col)) for col in columns)
        extras=_brmd_extra_terms(c,rhs)
        body=join((fixed...,extras...)," + ")
        isempty(body) && (body="0")
        hasintercept=any(col->col.source===nothing,columns)
    end
    equation=brm_description_math(c,lhs)*"="*body
    prose="`$(c.provenance.owner)` combines the declared population, group and structured terms. " *
        (hasintercept ? "It includes a population intercept." : "It has no population intercept.")
    BRMDescriptionFragment(prose=(prose,),equations=(equation,),covers=(c.id,))
end

function _brmd_column_math(c,column)
    p=column.preprocess
    isnothing(p) && return column.source===nothing ? "1" : brm_description_symbol(c,column.source)
    if p.kind===:interaction
        operands=map(p.raw_ref) do label
            found=filter(dep->dep.label===label,p.dependencies)
            isempty(found) ? brm_description_symbol(c,label) : _brmd_column_math(c,only(found))
        end
        return join(operands,"\\,")
    elseif p.kind===:population_factor_dummy
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
    x.callable in (|,doublepipe,factor,offset,s,t2,gp,hsgp,mo1,dar,rw,cdar,kernel) ?
        (_brmd_term_math(c,x),) : ()
end
function _brmd_builtin_kind(::Val{:observation},c)
    lhs,rhs=c.arguments
    role=c.provenance.observation_role
    prose= role===:conditioned ? "`$(c.provenance.owner)` contributes an observation likelihood." :
           role===:held_out ? "`$(c.provenance.owner)` is held out: its density does not contribute to this fit." :
           "`$(c.provenance.owner)` is unconditioned and is generated from the declared model."
    law=rhs isa BRMDescriptionComponent ? _brmd_law(rhs.callable) : nothing
    isnothing(law) || (prose*=" "*_brmd_law_prose(Val(law)))
    BRMDescriptionFragment(prose=(prose,),equations=(brm_description_math(c,lhs)*"\\sim"*_brmd_distribution_math(c,rhs),),covers=(c.id,))
end
function _brmd_builtin_kind(::Val{:random_effect},c)
    k=c.keywords
    id=_brmd_identifier(k.group)
    covariance=k.correlated ? "D\\Omega D" : "D^2"
    prose="Group deviations for `$(k.group)` have $(k.n_terms) margin(s) across $(k.n_groups) fitted levels. " *
        (k.shared ? "The declared ID shares one covariance block across its predictors." : "This block is independent of separately declared blocks.") *
        (k.correlated ? " Its margins are correlated." : " It has no estimated correlation.")
    BRMDescriptionFragment(prose=(prose,),equations=("b_{"*id*",i}\\sim\\mathcal N_{"*string(k.n_terms)*"}(0,"*covariance*")",),covers=(c.id,))
end
_brmd_builtin_kind(::Val{K},_c) where K = nothing

function _brmd_notation(d,labels)
    names=unique((d.columns..., (o.logical for o in d.outputs if o.logical !== nothing)...,
        sort!(collect(keys(labels));by=string)...))
    result=NamedTuple[]
    for name in names
        label=get(labels,name,NamedTuple())
        label isa AbstractString && (label=(; meaning=String(label)))
        label isa NamedTuple || throw(ArgumentError("labels[$name] must be text or a notation NamedTuple"))
        push!(result,merge((; name,meaning=string(name),axis=:declared),_brmd_snapshot(label)))
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
    brm_description(d::BRMDescriptor; hooks=(), labels=Dict(), prior_anchors=Dict())
    brm_description(prepared::Union{SBBRMI,GenerativePlan}; ...)

Describe the selected prepared model without fitting or sampling. Per-call hooks
are actual-callable => context -> fragment pairs. `labels` binds logical symbols
and supplied meanings/units. `prior_anchors` maps logical prior tuple IDs to
report links. Unsupported semantic nodes remain explicit coverage gaps; an
incomplete result is never labeled a complete model description.
"""
function brm_description(d::BRMDescriptor; hooks=(),labels=Dict(),prior_anchors=Dict())
    notation=_brmd_notation(d,labels)
    constants=_brmd_constants(d.plan)
    priors,groups=_brmd_priors(d,prior_anchors,notation)
    roots=collect(_brmd_components(d,priors,constants,notation))
    # Inventory allocation semantics explicitly; a Gaussian conditional prior
    # alone must never establish coverage of an unexplained hierarchical scale.
    for (target,binding) in sort!(collect(d.plan.bindings);by=p->string(first(p)))
        scheme=get(binding,:prior_scheme,(;kind=:ordinary))
        scheme.kind===:ordinary && continue
        env=_brmd_environment(d,binding.logical,:prior_allocation,priors,constants,notation)
        push!(roots,_brmd_component(env,(:allocation,binding.logical),:prior_allocation,
            nothing,(_brmd_snapshot(scheme),)))
    end
    for group in groups
        env=_brmd_environment(d,group.key,:random_effect,priors,constants,notation)
        b=group.block
        kwargs=(; group=b.group,id=b.id,n_terms=b.n_terms,n_groups=b.n_groups,
            levels=_brmd_snapshot(b.levels),margins=group.margins,
            correlated=group.correlated,shared=group.shared,noncentered=b.noncentered)
        push!(roots,_brmd_component(env,group.key,:random_effect,nothing,(),kwargs))
    end
    for prior in priors
        prior.distribution isa BRMDescriptionComponent && push!(roots,prior.distribution)
    end
    inventory=Tuple(node for root in roots for node in brm_description_components(root))
    ids=Set(c.id for c in inventory)
    length(ids)==length(inventory) || error("description: duplicate logical semantic component IDs")
    covered=Set{Tuple}()
    prose=String[]; equations=String[]; added_notation=NamedTuple[]
    for c in inventory
        fragment=_brmd_hook(c,hooks)
        isnothing(fragment) && (fragment=_brmd_builtin_fragment(c))
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
        c.id in covered ? "" : "No scientific description for $(_brmd_callable_name(c.callable)) at $(c.id)",
        c.provenance) for c in inventory)
    diagnostics=Tuple(x.reason for x in coverage if x.status !== :covered)
    BRMDescription(d.id,Tuple(roots),Tuple(prose),Tuple(equations),
        (notation...,added_notation...),priors,coverage,diagnostics,isempty(diagnostics))
end
brm_description(prepared::Union{SBBRMI,GenerativePlan};kwargs...) =
    brm_description(brm_descriptor(prepared);kwargs...)

function _brmd_prior_equation(p)
    c=p.distribution
    c isa BRMDescriptionComponent || return string(p.distribution)
    support=isempty(p.support) ? "" : "\\quad " * join(
        (_brmd_identifier(k)*"="*brm_description_math(c,v) for (k,v) in pairs(p.support)),", ")
    _brmd_identifier(join(string.(p.id)," / "))*"\\sim"*_brmd_distribution_math(c,c)*support
end

"""
    brm_description_markdown(description)

Render deterministic Markdown with display LaTeX, an effective prior table,
notation and explicit coverage diagnostics. Report links come only from the
supplied logical `prior_anchors` map. No generated-name parsing is required.
"""
function brm_description_markdown(description::BRMDescription;prefix=nothing)
    io=IOBuffer()
    println(io,description.complete ? "Complete model description." : "Incomplete model description: semantic coverage gaps remain.")
    for paragraph in description.prose
        println(io,"\n",paragraph)
    end
    for equation in description.equations
        println(io,"\n\$\$\n",equation,"\n\$\$")
    end
    if !isempty(description.priors)
        println(io,"\nEffective priors:\n\n| Logical parameter | Distribution and support | Prior listing |\n| --- | --- | --- |")
        for p in description.priors
            label=replace(join(string.(p.id)," / "),"|"=>"\\|")
            equation=replace(_brmd_prior_equation(p),"|"=>"\\|")
            link=isnothing(p.anchor) ? "" : "[prior listing]("*replace(p.anchor," "=>"%20")*")"
            println(io,"| <a id=\"",brm_description_prior_anchor(p;prefix),"\"></a>`",label,"` | \$",equation,"\$ | ",link," |")
        end
        seen=Set{Tuple}()
        links=Tuple(c=>brm_description_prior_references(c) for c in description.components
            if c.provenance.declaration!==:prior)
        any(p->!isempty(last(p)),links) && println(io,"\nComponent prior references:\n\n| Component | Effective priors |\n| --- | --- |")
        for (c,ids) in links
            isempty(ids) && continue
            ids in seen && continue
            push!(seen,ids)
            references=join(("["*replace(join(string.(id)," / "),"|"=>"\\|")*"](#"*
                brm_description_prior_anchor(description,id;prefix)*")" for id in ids),", ")
            println(io,"| `",replace(repr(c.id),"|"=>"\\|"),"` | ",references," |")
        end
    end
    if !isempty(description.notation)
        println(io,"\nNotation:\n\n| Quantity | Meaning | Axis | Supplied unit |\n| --- | --- | --- | --- |")
        for n in description.notation
            println(io,"| `",n.name,"` | ",get(n,:meaning,string(n.name))," | ",get(n,:axis,:declared)," | ",get(n,:unit,"")," |")
        end
    end
    if !description.complete
        println(io,"\nCoverage gaps:")
        for diagnostic in description.diagnostics
            println(io,"\n- ",diagnostic)
        end
    end
    String(take!(io))
end
