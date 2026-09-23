"""An improper uniform prior on the real line. Prior-only simulation is undefined."""
struct Flat <: ContinuousUnivariateDistribution end
Distributions.logpdf(::Flat, x::Real) = zero(x)
_sb_stan_dist_name(::Type{Flat}) = :flat

const _SB_TOTAL_PLANS_KEY = :__brm_total_plans__
const _brm_total_scales = StanBlocks.SlicModel(quote
    tau ~ std_normal(n=n,lower=0.)
    return tau
end, (;), @__MODULE__)

# Compile-time information only. The corresponding numerical design and prior
# constants live in model data; descriptors/replay use the same binding record.
"""Fitted design, prior and coordinate identities for one exact total-coefficient block."""
struct TotalEffectBlock
    predictor::Symbol
    group::Symbol
    binding::Symbol
    scales::Symbol
    population::Symbol
    deviations::Symbol
    group_index::Symbol
    group_count::Symbol
    columns::Tuple
    population_columns::Tuple
    A::Matrix{Float64}
    location::Vector{Float64}
    precision::Vector{Float64}
    mixture::Vector{Int}
    mixture_name::Union{Symbol,Nothing}
end

"""Return the exact total-coefficient blocks selected during model construction."""
total_effect_blocks(model) = sort!([b.total for b in values(model.bindings) if hasproperty(b, :total)];by=b->String(b.predictor))

function _total_coordinates(model, block, names)
    pos = Dict(String(n)=>i for (i,n) in enumerate(names))
    lookup(n) = get(pos,n) do
        throw(ArgumentError("total-coefficient model expects unconstrained coordinate `$n`"))
    end
    J = model.data[block.group_count]
    K = length(block.columns)
    totals = [lookup("$(block.binding).$g.$k") for g in 1:J, k in 1:K]
    scales = [lookup("$(block.scales).$k") for k in 1:K]
    mixture = isnothing(block.mixture_name) ? Int[] :
        [lookup("$(block.mixture_name).$m") for m in eachindex(block.mixture)]
    (;totals,scales,mixture)
end

function _total_conditional(block, coordinates, draw)
    totals = draw[coordinates.totals]
    tau = exp.(draw[coordinates.scales])
    precision = copy(block.precision)
    for (m,c) in pairs(block.mixture)
        precision[c] *= exp(draw[coordinates.mixture[m]])
    end
    J = size(totals,1)
    Dinv = Diagonal(inv.(tau.^2))
    Q = Symmetric(J*block.A'*Dinv*block.A + Diagonal(precision))
    factor = cholesky(Q)
    h = block.A'*Dinv*vec(sum(totals;dims=1)) + precision.*block.location
    (;mean=factor\h, factor, tau, totals)
end

"""
    recover_population_draws(model, draws, unc_names; rng=Random.default_rng())

Draw the original population coefficients conditional on each saved total-
coefficient draw. Returns one record per predictor with `population`, `totals`,
and `deviations` arrays. This exact recovery adds conditional randomness; totals
are unchanged. Input draws must be in the compiled model's coordinate frame.
"""
function recover_population_draws(model, draws::AbstractMatrix, names;
                                  rng::Random.AbstractRNG=Random.default_rng())
    size(draws,2) == length(names) || throw(DimensionMismatch("draws and names disagree"))
    out = Dict{Symbol,NamedTuple}()
    for block in total_effect_blocks(model)
        coordinates = _total_coordinates(model,block,names)
        n,J,K = size(draws,1),size(coordinates.totals)...
        population = Matrix{Float64}(undef,n,length(block.population_columns))
        totals = Array{Float64}(undef,n,J,K)
        deviations = similar(totals)
        for i in 1:n
            conditional = _total_conditional(block,coordinates,view(draws,i,:))
            beta = conditional.mean + conditional.factor.U\randn(rng,length(conditional.mean))
            population[i,:] = beta
            totals[i,:,:] = conditional.totals
            deviations[i,:,:] = conditional.totals .- transpose(block.A*beta)
        end
        out[block.predictor] = (;population,totals,deviations,
            population_columns=block.population_columns,columns=block.columns)
    end
    out
end

"""
    select_total_centeredness(model, draws, unc_names;
        criterion=:position, gradients=nothing, grid=0:0.1:1)

Select each total coefficient's centering independently from a saved pilot.
`draws` and optional `gradients` are draws × coordinates in the COMPILED model
frame. `:position` minimizes log SD minus mean log Jacobian; `:gradient`
minimizes position-gradient correlation and requires matching exact gradients.
No new density/gradient calls are made. The returned `centeredness` vector can
be passed to `adaptive_centering_problem`; `losses` preserves every grid score.
"""
function select_total_centeredness(model,draws::AbstractMatrix,names;
        criterion=:position,gradients=nothing,grid=0.:0.1:1.)
    criterion in (:position,:gradient) || throw(ArgumentError("criterion must be :position or :gradient"))
    size(draws,2) == length(names) && size(draws,1) > 2 ||
        throw(DimensionMismatch("expected at least three draws, with one column per model coordinate"))
    if criterion === :gradient
        gradients isa AbstractMatrix && size(gradients) == size(draws) ||
            throw(DimensionMismatch("gradient criterion needs matching compiled-frame gradients"))
    end
    candidates = Float64.(collect(grid))
    !isempty(candidates) && all(c->isfinite(c) && 0<=c<=1,candidates) ||
        throw(ArgumentError("centering grid must contain finite values in [0,1]"))
    centeredness,indices,losses = Float64[],Int[],Vector{Float64}[]
    blocks = total_effect_blocks(model)
    isempty(blocks) && throw(ArgumentError("model has no exact total-coefficient blocks"))
    for block in blocks
        coords = _total_coordinates(model,block,names)
        mu = block.A*block.location
        for k in axes(coords.totals,2), g in axes(coords.totals,1)
            index = coords.totals[g,k]
            ell = view(draws,:,coords.scales[k])
            physical = view(draws,:,index)
            scores = map(candidates) do c
                candidate = c*mu[k] .+ (physical .- mu[k]).*exp.((c-1).*ell)
                if criterion === :position
                    log(std(candidate)) + mean((1-c).*ell)
                else
                    candidate_gradient = view(gradients,:,index).*exp.((1-c).*ell)
                    cor(candidate,candidate_gradient)
                end
            end
            all(isfinite,scores) || throw(ArgumentError("pilot gives degenerate/nonfinite centering scores for $(names[index])"))
            push!(centeredness,candidates[argmin(scores)])
            push!(indices,index);push!(losses,scores)
        end
    end
    (;centeredness,indices,grid=candidates,losses=permutedims(hcat(losses...)),criterion)
end

function _sb_total_population_prior(prior)
    isnothing(prior) && return (; location=0., precision=1., nu=nothing)
    prior isa ExprColumn || return nothing
    isempty(getkwargs(prior)) || return nothing
    f = getf(prior)
    args = map(_sb_effect_prior_arg, getargs(prior))
    if f === Flat && isempty(args)
        return (; location=0., precision=0., nu=nothing)
    elseif f === Normal && length(args) <= 2 && all(x -> x isa Real, args)
        loc = isempty(args) ? 0. : args[1]
        scale = length(args) < 2 ? 1. : args[2]
        scale > 0 && isfinite(scale) && isfinite(loc) || return nothing
        return (; location=Float64(loc), precision=inv(Float64(scale)^2), nu=nothing)
    elseif f === LocationScale && length(getargs(prior)) == 3
        loc, scale, base = getargs(prior)
        loc, scale = _sb_effect_prior_arg(loc), _sb_effect_prior_arg(scale)
        base isa ExprColumn && getf(base) === TDist || return nothing
        isempty(getkwargs(base)) || return nothing
        nu = only(map(_sb_effect_prior_arg, getargs(base)))
        all(x -> x isa Real && isfinite(x), (loc,scale,nu)) || return nothing
        scale > 0 && nu > 0 || return nothing
        return (; location=Float64(loc), precision=inv(Float64(scale)^2), nu=Float64(nu))
    elseif f === TDist && length(args) == 1 && only(args) isa Real && only(args) > 0
        return (; location=0., precision=1., nu=Float64(only(args)))
    end
    nothing
end

# Match declared bases, never a numerical least-squares fit to the training
# rows: coincidental collinearity is not an identity on future covariates.
function _sb_total_basis_map(population, columns)
    A = zeros(length(columns), length(population))
    intercept = findfirst(c -> isnothing(c.source), columns)
    for (p, column) in pairs(population)
        same = findfirst(c -> isequal(c.source, column.source) &&
                              isequal(c.preprocess, column.preprocess), columns)
        if !isnothing(same)
            A[same,p] = 1.
            continue
        end
        pre = column.preprocess
        isnothing(pre) && return nothing
        pre.kind in (:center, :zscale, :standardize) || return nothing
        isnothing(intercept) && return nothing
        raw_ref = pre.raw_ref isa NamedColumn && parent(pre.raw_ref) isa DataColumn ? name(pre.raw_ref) : pre.raw_ref
        raw = findfirst(c -> isnothing(c.preprocess) && isequal(c.source,raw_ref), columns)
        isnothing(raw) && return nothing
        shift, scale = pre.kind === :center ? (pre.const_,1.) : pre.const_
        A[raw,p] = inv(scale)
        A[intercept,p] = -shift/scale
    end
    rank(A) == size(A,2) ? A : nothing
end

function _sb_total_plan(brmi, prepared, predictor, overrides, buckets, sd_overrides;
                        cv_groups, centered_groups, r2d2_overrides, ranef_r2d2_overrides,
                        s2z_groups=Set{Symbol}())
    target = predictor.name
    declarations = filter(d -> d.predictor === target, prepared.context.group_declarations)
    isempty(declarations) && return nothing
    # Do not reinterpret shared/correlated/stratified/multi-membership blocks.
    any(d -> d.raw_group isa MultiMembershipTerm || d.descriptor isa Tuple, declarations) && return nothing
    plans = _brm_simple_random_effect_plans(brmi,target,prepared.context)
    isnothing(plans) && return nothing
    group = first(plans).group
    all(p -> p.group === group && (p.zero_correlation || length(p.columns) == 1),plans) || return nothing
    group in cv_groups && return nothing
    # Explicit conventional centered_groups is a request for that representation.
    group in centered_groups && return nothing
    # An explicit S2Z request is likewise a request for that representation.
    group in s2z_groups && return nothing
    haskey(r2d2_overrides,target) && return nothing
    pop, ran, direct = Any[], Any[], Any[]
    foreach(t -> _sb_classify_term!(t,pop,ran,direct), _sb_terms(predictor.expression))
    isempty(direct) || return nothing
    obs_name = get(prepared.context.target_obs,target,nothing)
    design = _brm_population_design(target,Tuple(pop),prepared.context.data,obs_name)
    if isnothing(design) && (isnothing(obs_name) || !haskey(prepared.context.data,obs_name))
        # Prior spelling: the response column is omitted, so an
        # intercept-only predictor has no observation row axis. The declared
        # grouping column spans the same rows; resolve the axis from the
        # declaration so the prior program keeps the posterior's
        # representation and names instead of dropping every `:auto` plan.
        design = _brm_population_design(target,Tuple(pop),prepared.context.data,obs_name; row_source=group)
    end
    (isnothing(design) || isempty(design.columns) || !isempty(design.fixed_terms)) && return nothing
    columns = Tuple(c for p in plans for c in p.columns)
    all(c -> isnothing(c.preprocess) || c.preprocess.kind === :protect,columns) || return nothing
    length(unique(c.label for c in columns)) == length(columns) || return nothing
    maps = [_sb_total_basis_map((column,),columns) for column in design.columns]
    all_priors = _sb_pop_effect_overrides(overrides,target)
    isnothing(all_priors) && (all_priors = fill(nothing,length(design.columns)))
    infos = map(_sb_total_population_prior,all_priors)
    any(i -> !isnothing(maps[i]) && isnothing(infos[i]),eachindex(maps)) && return nothing
    absorbed = findall(i -> !isnothing(maps[i]) && !isnothing(infos[i]),eachindex(maps))
    isempty(absorbed) && return nothing
    A = hcat((maps[i] for i in absorbed)...)
    rank(A) == size(A,2) || return nothing
    remaining = setdiff(collect(eachindex(maps)),absorbed)
    frozen = _sb_frozen_preproc_entry(prepared.context.data,Symbol(:total_A_,target),:total_basis,target)
    isnothing(frozen) || (A = copy(frozen.const_))
    prior_info = infos[absorbed]
    scale_priors = Any[]
    claimed = Any[]
    for plan in plans
        if isnothing(plan.id)
            for c in plan.columns
                # Plain scalar intercepts have the original log-normal scale
                # prior; named buckets and scalar slopes use half-normal.
                push!(scale_priors,isnothing(c.source) ? ExprColumn(LogNormal,0.,1.) : nothing)
            end
        else
            keys_ = [key for (key,bucket) in buckets if first(key) === plan.id &&
                     any(entry -> first(entry) === target,bucket.per_target)]
            length(keys_) == 1 || return nothing
            key = only(keys_)
            haskey(ranef_r2d2_overrides,key) && return nothing
            length(buckets[key].per_target) == 1 || return nothing
            sd = get(sd_overrides,key,nothing)
            append!(scale_priors,isnothing(sd) ? fill(nothing,length(plan.columns)) : sd.sd_prior)
            push!(claimed,key)
        end
    end
    # Only priors whose emitted support is the ordinary positive half-line
    # can use WHMC's logarithmic scale accessor in this first implementation.
    for prior in scale_priors
        isnothing(prior) && continue
        prior isa ExprColumn || return nothing
        isempty(getkwargs(prior)) || return nothing
        getf(prior) in (Normal,LogNormal,Cauchy,Exponential,TDist,LocationScale,Gamma) || return nothing
        isempty(_sb_prior_references(_sb_effect_prior_arg(prior))) || return nothing
    end
    (; target,group,columns,design,A,prior_info,scale_priors,claimed,absorbed,remaining,
       remaining_priors=all_priors[remaining],
       indices=first(plans).indices,levels=first(plans).levels)
end

function _sb_plan_totals(brmi,prepared,overrides,buckets,sd_overrides,selection;
                         cv_groups,centered_groups,r2d2_overrides,ranef_r2d2_overrides,
                         s2z_groups=Set{Symbol}())
    selection === :auto || selection isa Symbol || selection isa Tuple || selection isa AbstractVector || selection isa AbstractSet ||
        throw(ArgumentError("total_groups must be :auto, a grouping-factor name, or a collection (empty disables totals)"))
    requested = selection === :auto ? nothing : Set(selection isa Symbol ? (selection,) : selection)
    !isnothing(requested) && isempty(requested) && return Dict{Symbol,Any}()
    out = Dict{Symbol,Any}()
    for predictor in prepared.predictors
        plan = _sb_total_plan(brmi,prepared,predictor,overrides,buckets,sd_overrides;
                             cv_groups,centered_groups,r2d2_overrides,ranef_r2d2_overrides,
                             s2z_groups)
        isnothing(plan) && continue
        isnothing(requested) || plan.group in requested || continue
        out[predictor.name] = plan
    end
    # Preserve the existing automatic WHMC route for mixed models: its current
    # exact scoring plans use one geometry family at a time.
    if isnothing(requested) && any(d -> !haskey(out,d.predictor),prepared.context.group_declarations)
        empty!(out)
    end
    if !isnothing(requested)
        missing = setdiff(requested,Set(p.group for p in values(out)))
        isempty(missing) || throw(ArgumentError("exact totals are unavailable for group(s) $(join(missing, ", ")); requires one independent grouping structure, matched population design and supported priors"))
    end
    out
end

function _sb_emit_total!(stmts,data,target,plan;mod::Module=@__MODULE__)
    suffix = plan.target
    total, tau = Symbol(:total_,suffix), Symbol(:total_scale_,suffix)
    beta, deviations = Symbol(:population_,suffix), Symbol(:deviation_,suffix)
    idx, ng = Symbol(:total_group_,suffix), Symbol(:total_ng_,suffix)
    nk, np = Symbol(:total_nk_,suffix), Symbol(:total_np_,suffix)
    an, loc, prec = Symbol(:total_A_,suffix), Symbol(:total_location_,suffix), Symbol(:total_precision_,suffix)
    j,k,p = length(plan.levels),length(plan.columns),length(plan.absorbed)
    for (key,value) in (ng=>j,nk=>k,np=>p,an=>plan.A,
                       loc=>Float64[v.location for v in plan.prior_info],
                       prec=>Float64[v.precision for v in plan.prior_info])
        data[key] = value
        key === ng || _sb_record_static!(data,key)
    end
    data[idx] = plan.indices
    _sb_record_preproc!(data,an,PreprocEntry(:total_basis,copy(plan.A),plan.target,true))
    # The group-index entry owns the count on replay (including changed
    # vocabulary size); a second static entry would overwrite that update.
    _sb_record_preproc!(data,idx,PreprocEntry(:group_index,
        (;levels=plan.levels,n_groups_key=ng),plan.group,true))
    # Vector scale prior is assembled through the existing prior translator,
    # retaining its normalization and support rules.
    prior = _sb_vector_positive_priors(_brm_total_scales,:tau,plan.scale_priors)
    push!(stmts,:($tau ~ $(prior.model)(;n=$nk)))
    mixture_indices = findall(v -> !isnothing(v.nu),plan.prior_info)
    mixture = isempty(mixture_indices) ? nothing : Symbol(:total_mixture_,suffix)
    cp = prec
    if !isempty(mixture_indices)
        nlambda = Symbol(:total_nm_,suffix)
        data[nlambda] = length(mixture_indices); _sb_record_static!(data,nlambda)
        shape = Symbol(:total_mixture_shape_,suffix)
        data[shape] = [plan.prior_info[c].nu/2 for c in mixture_indices]
        _sb_record_static!(data,shape)
        push!(stmts,:($mixture::vector[$nlambda] ~ gamma($shape,$shape;lower=0.)))
        cp = Symbol(:total_conditional_precision_,suffix)
        entries = Any[]
        for c in 1:p
            m = findfirst(==(c),mixture_indices)
            push!(entries,isnothing(m) ? :($prec[$c]) : :($prec[$c]*$mixture[$m]))
        end
        push!(stmts,:($cp = $(Expr(:vect,entries...))))
    end
    push!(stmts,:($total::matrix[$ng,$nk] ~ brm_total($tau,$an,$loc,$cp)))
    push!(stmts,:($beta = brm_total_recover_rng($total,$tau,$an,$loc,$cp)))
    push!(stmts,:($deviations = brm_total_deviations($total,$an*$beta)))
    zcols = Any[isnothing(c.source) ? :(rep_vector(1.,num_elements($idx))) :
                    _sb_shared_population_column!(data,c) for c in plan.columns]
    zn = Symbol(:total_Z_,suffix)
    push!(stmts,:($zn = $(Expr(:call,:hcat,zcols...))))
    total_lp = :(rows_dot_product($total[$idx,:],$zn))
    if !isempty(plan.remaining)
        remaining_cols = Any[isnothing(plan.design.columns[c].source) ? :(rep_vector(1.,num_elements($idx))) :
            _sb_shared_population_column!(data,plan.design.columns[c]) for c in plan.remaining]
        xn,pop = Symbol(:X_,target),Symbol(:pop_,target)
        push!(stmts,:($xn = $(Expr(:call,:hcat,remaining_cols...))))
        prior = _sb_population_prior_rhs(plan.remaining_priors;mod)
        kwargs = Expr(:parameters,Expr(:kw,:X,xn),
            (Expr(:kw,key,value) for (key,value) in pairs(prior.kwargs))...)
        push!(stmts,Expr(:call,:~,pop,Expr(:call,prior.model,kwargs)))
        data[_SB_BINDINGS_KEY][pop] = (;role=:population_effect,logical=plan.target,family=nothing,
            population_columns=Tuple(plan.design.columns[c].label for c in plan.remaining))
        total_lp = :($total_lp + $pop)
    end
    push!(stmts,:($target = $total_lp))
    block = TotalEffectBlock(plan.target,plan.group,total,Symbol(tau,:_tau),beta,deviations,idx,ng,
        Tuple(c.label for c in plan.columns),Tuple(plan.design.columns[c].label for c in plan.absorbed),
        plan.A,copy(data[loc]),copy(data[prec]),mixture_indices,mixture)
    data[_SB_BINDINGS_KEY][total] = (;role=:total_effect,logical=plan.target,family=:brm_total,total=block)
    _sb_record_binding!(data,beta,:population_effect,plan.target)
    _sb_record_binding!(data,tau,:parameter,plan.target)
    nothing
end
