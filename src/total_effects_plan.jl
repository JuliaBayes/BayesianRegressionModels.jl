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
"""
Fitted design, prior and coordinate identities for one exact total-coefficient block.

Group `g`'s total of random column `c` has mean `sum_p A[c,p] * W[g,p] * beta[p]`
over the absorbed population coefficients `beta` (`population_columns`). `W` is
1 for a coefficient whose column matches a random column; a group-level
predictor (`group_columns`: a column constant within each group, or a random
slope column times a per-group value) carries its per-group value.
"""
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
    group_columns::Tuple
    A::Matrix{Float64}
    W::Matrix{Float64}
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

# Each group's mean total `A * (W[g,:] .* beta)`, groups × random columns.
_total_location(block, beta) = block.W*Diagonal(beta)*transpose(block.A)

function _total_conditional(block, coordinates, draw)
    totals = draw[coordinates.totals]
    tau = exp.(draw[coordinates.scales])
    precision = copy(block.precision)
    for (m,c) in pairs(block.mixture)
        precision[c] *= exp(draw[coordinates.mixture[m]])
    end
    A, W = block.A, block.W
    G = transpose(W)*W
    WT = transpose(W)*totals
    Q = Symmetric(Diagonal(precision) +
        sum((A[c,:]*transpose(A[c,:])) .* G ./ tau[c]^2 for c in axes(A,1)))
    factor = cholesky(Q)
    h = precision.*block.location + sum(A[c,:] .* WT[:,c] ./ tau[c]^2 for c in axes(A,1))
    (;mean=factor\h, factor, tau, totals)
end

"""
    recover_population_draws(model, draws, unc_names; rng=Random.default_rng())

Draw the original population coefficients conditional on each saved total-
coefficient draw. Returns one record per predictor with `population`, `totals`,
and `deviations` arrays. This exact recovery adds conditional randomness; totals
are unchanged. Input draws must be in the compiled model's coordinate frame.

A block whose totals are emitted in generated quantities (no observed
likelihood reaches it, for example an omitted response) has no saved total to
condition on, so it has no record here. Stan draws its totals, population
coefficients and deviations together when generated quantities are evaluated.
A model whose every block is generated is an error.
"""
function recover_population_draws(model, draws::AbstractMatrix, names;
                                  rng::Random.AbstractRNG=Random.default_rng())
    size(draws,2) == length(names) || throw(DimensionMismatch("draws and names disagree"))
    out = Dict{Symbol,NamedTuple}()
    for block in _sampled_total_blocks(model,"recover_population_draws",
            "Stan draws their recovered population coefficients in generated " *
            "quantities, which `brm_population_effect_coordinates` resolves")
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
            deviations[i,:,:] = conditional.totals .- _total_location(block,beta)
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
Only sampled blocks have cells: a block whose totals are emitted in generated
quantities has no sampler coordinate to center, and a model whose every block
is generated is an error.
"""
function select_total_centeredness(model,draws::AbstractMatrix,names;
        criterion=:position,gradients=nothing,grid=0.:0.1:1.)
    isempty(total_effect_blocks(model)) && throw(ArgumentError("model has no exact total-coefficient blocks"))
    blocks = _sampled_total_blocks(model,"select_total_centeredness",
        "the sampler has no total coordinate to center")
    _select_scalar_centeredness(_total_centering_cells(model,names,blocks),draws,names;
        criterion,gradients,grid)
end

# The exact total-coefficient blocks with sampler coordinates. A model with
# blocks but none sampled gives the draws nothing to condition on: refuse it
# rather than return an empty result.
function _sampled_total_blocks(model,caller,remedy)
    placement = _total_blocks_by_placement(model)
    isempty(placement.sampled) && !isempty(placement.generated) && throw(ArgumentError(
        "$caller: every exact total-coefficient block " *
        "($(join((b.binding for b in placement.generated), ", "))) is emitted in " *
        "generated quantities because no observed likelihood reaches them, so the " *
        "draws carry no total coordinate; $remedy."))
    placement.sampled
end

# One scalar cell per sampled total (group, coefficient), column-major. Each
# has a constant location (its prior total, which a group-level predictor makes
# group-specific), a log-scale coordinate and the compiled frame c = 1. The
# WarmupHMC wrapper and the pilot selector share this order. A generated block
# has no sampler coordinate and contributes no cell.
function _total_centering_cells(model,names,blocks=_total_blocks_by_placement(model).sampled)
    indices,scales,locations = Int[],Int[],Float64[]
    for block in blocks
        coords = _total_coordinates(model,block,names)
        mu = _total_location(block,block.location)
        for k in axes(coords.totals,2), g in axes(coords.totals,1)
            push!(indices,coords.totals[g,k]);push!(scales,coords.scales[k]);push!(locations,mu[g,k])
        end
    end
    (;indices,scales,locations,targets=ones(length(indices)))
end

# Pilot scorer shared by every scalar-cell family (totals, S2Z contrasts).
# A cell compiled at frame `t` maps to candidate `c` by
# `c*mu + (x - t*mu)*exp((c-t)*ell)`, whose log-Jacobian is `(c-t)*ell`.
function _select_scalar_centeredness(cells,draws::AbstractMatrix,names;
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
    centeredness,losses = Float64[],Vector{Float64}[]
    for (index,scale,mu,t) in zip(cells.indices,cells.scales,cells.locations,cells.targets)
        ell = view(draws,:,scale)
        physical = view(draws,:,index)
        scores = map(candidates) do c
            candidate = c*mu .+ (physical .- t*mu).*exp.((c-t).*ell)
            if criterion === :position
                log(std(candidate)) + mean((t-c).*ell)
            else
                candidate_gradient = view(gradients,:,index).*exp.((t-c).*ell)
                cor(candidate,candidate_gradient)
            end
        end
        all(isfinite,scores) || throw(ArgumentError("pilot gives degenerate/nonfinite centering scores for $(names[index])"))
        push!(centeredness,candidates[argmin(scores)])
        push!(losses,scores)
    end
    (;centeredness,indices=copy(cells.indices),grid=candidates,
     losses=permutedims(hcat(losses...)),criterion)
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
    matched = findall(i -> !isnothing(maps[i]) && !isnothing(infos[i]),eachindex(maps))
    isempty(matched) && return nothing
    rank(hcat((maps[i] for i in matched)...)) == length(matched) || return nothing
    indices, levels = first(plans).indices, first(plans).levels
    group_level = _sb_total_group_level_columns(prepared.context.data,target,group,design,
        columns,maps,infos,matched,indices,levels)
    absorbed = sort!(vcat(matched,collect(keys(group_level))))
    A = hcat((haskey(group_level,i) ? _sb_total_unit(length(columns),group_level[i].column) :
              maps[i] for i in absorbed)...)
    W = hcat((haskey(group_level,i) ? group_level[i].values : ones(length(levels))
              for i in absorbed)...)
    remaining = setdiff(collect(eachindex(maps)),absorbed)
    frozen = _sb_frozen_preproc_entry(prepared.context.data,Symbol(:total_A_,target),:total_basis,target)
    isnothing(frozen) || (A = copy(frozen.const_))
    # What a replay needs to check a group-level column against the fitted W:
    # its design column, the random column it multiplies and its W position.
    group_records = Tuple((; position=findfirst(==(i),absorbed),
        column=_sb_design_column_record(design.columns[i]),
        slope=_sb_design_column_record(columns[group_level[i].column]))
        for i in absorbed if haskey(group_level,i))
    prior_info = infos[absorbed]
    scale_priors = Any[]
    # Per total column: the declared `|ID|` (or `nothing`) and whether an
    # authored `sd(...)` statement selected its scale prior. Descriptions read
    # these emitted facts instead of re-deriving the defaults below.
    column_ids = Any[]
    scale_configured = Bool[]
    claimed = Any[]
    for plan in plans
        if isnothing(plan.id)
            for c in plan.columns
                # Plain scalar intercepts have the original log-normal scale
                # prior; named buckets and scalar slopes use half-normal.
                push!(scale_priors,isnothing(c.source) ? ExprColumn(LogNormal,0.,1.) : nothing)
                push!(column_ids,nothing)
                push!(scale_configured,false)
            end
        else
            keys_ = [key for (key,bucket) in buckets if first(key) === plan.id &&
                     any(entry -> first(entry) === target,bucket.per_target)]
            length(keys_) == 1 || return nothing
            key = only(keys_)
            haskey(ranef_r2d2_overrides,key) && return nothing
            length(buckets[key].per_target) == 1 || return nothing
            sd = get(sd_overrides,key,nothing)
            # The induced total covariance is Gaussian; a Student-t block's
            # per-level mixing weights have no representation in it.
            !isnothing(sd) && !isnothing(sd.dist) && return nothing
            append!(scale_priors,isnothing(sd) ? fill(nothing,length(plan.columns)) : sd.sd_prior)
            for i in eachindex(plan.columns)
                push!(column_ids,plan.id)
                push!(scale_configured,!isnothing(sd) && !isnothing(sd.sd_prior[i]))
            end
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
    (; target,group,columns,design,A,W,group_records,prior_info,scale_priors,column_ids,scale_configured,
       claimed,absorbed,remaining,remaining_priors=all_priors[remaining],indices,levels)
end

_sb_total_unit(k,c) = (u = zeros(k,1); u[c] = 1.; u)

# Stacked (group, random column) × coefficient design of the totals' mean.
_sb_total_effective_design(A,W) = reduce(vcat,(W .* transpose(A[c,:]) for c in axes(A,1)))

# Per-group `w` with `x[r] == z[r] * w[group of r]` on every row, or `nothing`.
# A group whose rows all have `z == 0` fixes no value: it reads 0, its rows
# must have `x == 0`, and `determined` is false for it.
function _sb_total_group_values(x,z,indices,J)
    values, determined = zeros(J), falses(J)
    for (r,g) in pairs(indices)
        if iszero(z[r])
            iszero(x[r]) || return nothing
        elseif determined[g]
            isapprox(x[r],z[r]*values[g];rtol=1e-12) || return nothing
        else
            values[g], determined[g] = x[r]/z[r], true
        end
    end
    (; values, determined)
end

# The one random column `x` is a per-group multiple of, or `nothing` when none
# or several are.
function _sb_total_group_level(x,columns,indices,J)
    found = nothing
    for (c,z) in pairs(columns)
        fit = _sb_total_group_values(x,z.values,indices,J)
        isnothing(fit) && continue
        isnothing(found) || return nothing
        found = (; column=c, fit.values)
    end
    found
end

# Group-level predictors join the totals' mean through W: a population column
# identical on every row of each group joins the intercept total, and one equal
# to a random slope column times a per-group value joins that slope's total.
# They are found on the fitted rows, kept only while the totals' mean design
# stays of full column rank, and then frozen: a rebuild (`generative_plan(plan,
# new_df)`) absorbs exactly the fitted columns, checks that they are still
# group-level on its rows, and keeps the fitted value of every fitted group.
function _sb_total_group_level_columns(data,target,group,design,columns,maps,infos,
                                       matched,indices,levels)
    J = length(levels)
    out = Dict{Int,NamedTuple}()
    fitted = _sb_frozen_preproc(data)
    frozen = _sb_frozen_preproc_entry(data,_sb_total_group_design_name(target),
                                      :total_group_design,target)
    if isnothing(frozen)
        # A fit recorded before group-level absorption has a basis but no W
        # record; its rebuild keeps that representation.
        !isnothing(fitted) && haskey(fitted,_sb_total_basis_name(target)) && return out
        effective = _sb_total_effective_design(hcat((maps[i] for i in matched)...),ones(J,length(matched)))
        for (i,column) in pairs(design.columns)
            isnothing(maps[i]) && !isnothing(infos[i]) && !isnothing(column.source) || continue
            # A String column's level indicators stay sampled, like a
            # CategoricalVector column's contrast block.
            _brm_is_level_indicator(column) && continue
            found = _sb_total_group_level(column.values,columns,indices,J)
            isnothing(found) && continue
            candidate = hcat(effective,_sb_total_effective_design(
                _sb_total_unit(length(columns),found.column),reshape(found.values,J,1)))
            rank(candidate) == size(candidate,2) || continue
            effective = candidate
            out[i] = found
        end
        return out
    end
    record = frozen.const_
    position = Dict(level => g for (g,level) in enumerate(record.levels))
    for entry in record.columns
        label = entry.column.label
        i = findfirst(c -> c.label === label,design.columns)
        c = findfirst(z -> z.label === entry.slope.label,columns)
        (isnothing(i) || isnothing(c)) && error(
            "exact totals: the fitted model integrates group-level column `$label` of " *
            "`$target` into its `$group` totals, but the rebuilt design no longer has it")
        fit = _sb_total_group_values(design.columns[i].values,columns[c].values,indices,J)
        isnothing(fit) && error(
            "exact totals: population column `$label` of `$target` is integrated into the " *
            "`$group` totals as a group-level predictor, so it must be the same on every " *
            "row of a `$group` level$(entry.slope.label === :Intercept ? "" : " (as a multiple of `$(entry.slope.label)`)"); " *
            "the new data break that. Fit a new model for a column that varies within `$group`.")
        values = fit.values
        for (g,level) in enumerate(levels)
            k = get(position,level,0)
            k == 0 && continue
            value = record.W[k,entry.position]
            fit.determined[g] && !isapprox(values[g],value;rtol=1e-12) && error(
                "exact totals: `$group` level `$level` has group-level `$label` = $(values[g]) " *
                "in the new data, but the fit integrated it at $value; its fitted total " *
                "cannot represent a changed value")
            values[g] = value
        end
        out[i] = (; column=c, values)
    end
    out
end

Base.@nospecializeinfer function _sb_plan_totals(@nospecialize(brmi),prepared,overrides,buckets,sd_overrides,selection;
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
        isempty(missing) || throw(ArgumentError("exact totals are unavailable for group(s) $(join(missing, ", ")); requires one independent Gaussian grouping structure, matched population design and supported priors"))
    end
    out
end

# The random-effect design `Z` and basis `A` of the block on `predictor`, which
# `population_model` reads back from a TotalEffectBlock.
_sb_total_design_name(predictor) = Symbol(:total_Z_,predictor)
_sb_total_basis_name(predictor) = Symbol(:total_A_,predictor)
# Each group's value of every absorbed coefficient's column (`W`).
_sb_total_group_design_name(predictor) = Symbol(:total_W_,predictor)
# Each row's total contribution: its group's totals against the row's design.
_sb_total_location(total,idx,zn) = :(rows_dot_product($total[$idx,:],$zn))
# The population counterpart: every group at its population mean
# `A * (W[g,:] .* beta)`, which group-level predictors keep group-specific.
_sb_total_population_location(zn,an,wn,beta,idx) =
    :(rows_dot_product(brm_total_location($wn,$an,$beta)[$idx,:],$zn))

function _sb_emit_total!(stmts,data,target,plan;mod::Module=@__MODULE__)
    suffix = plan.target
    total, tau = Symbol(:total_,suffix), Symbol(:total_scale_,suffix)
    beta, deviations = Symbol(:population_,suffix), Symbol(:deviation_,suffix)
    idx, ng = Symbol(:total_group_,suffix), Symbol(:total_ng_,suffix)
    nk, np = Symbol(:total_nk_,suffix), Symbol(:total_np_,suffix)
    an, loc, prec = _sb_total_basis_name(suffix), Symbol(:total_location_,suffix), Symbol(:total_precision_,suffix)
    wn = _sb_total_group_design_name(suffix)
    j,k,p = length(plan.levels),length(plan.columns),length(plan.absorbed)
    for (key,value) in (ng=>j,nk=>k,np=>p,an=>plan.A,
                       loc=>Float64[v.location for v in plan.prior_info],
                       prec=>Float64[v.precision for v in plan.prior_info])
        data[key] = value
        key === ng || _sb_record_static!(data,key)
    end
    data[idx] = plan.indices
    _sb_record_preproc!(data,an,PreprocEntry(:total_basis,copy(plan.A),plan.target,true))
    # W is fitted per group level: a replay keeps it and checks its rows against
    # it, a rebuild for new groups keeps every fitted group's value.
    data[wn] = plan.W
    _sb_record_preproc!(data,wn,PreprocEntry(:total_group_design,
        (; levels=plan.levels,W=copy(plan.W),group=plan.group,columns=plan.group_records),
        plan.target,true))
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
    push!(stmts,:($total::matrix[$ng,$nk] ~ brm_total($tau,$an,$wn,$loc,$cp)))
    push!(stmts,:($beta = brm_total_recover_rng($total,$tau,$an,$wn,$loc,$cp)))
    push!(stmts,:($deviations = brm_total_deviations($total,brm_total_location($wn,$an,$beta))))
    zcols = Any[isnothing(c.source) ? :(rep_vector(1.,num_elements($idx))) :
                    _sb_shared_population_column!(data,c) for c in plan.columns]
    zn = _sb_total_design_name(suffix)
    push!(stmts,:($zn = $(Expr(:call,:hcat,zcols...))))
    total_lp = _sb_total_location(total,idx,zn)
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
            population_columns=Tuple(plan.design.columns[c].label for c in plan.remaining),
            design_columns=Tuple(_sb_design_column_record(plan.design.columns[c]) for c in plan.remaining))
        total_lp = :($total_lp + $pop)
    end
    push!(stmts,:($target = $total_lp))
    block = TotalEffectBlock(plan.target,plan.group,total,Symbol(tau,:_tau),beta,deviations,idx,ng,
        Tuple(c.label for c in plan.columns),Tuple(plan.design.columns[c].label for c in plan.absorbed),
        Tuple(r.column.label for r in plan.group_records),
        plan.A,copy(plan.W),copy(data[loc]),copy(data[prec]),mixture_indices,mixture)
    # The description reads the emitted scale priors, column identities and
    # design order from this record; it never reparses the generated families.
    data[_SB_BINDINGS_KEY][total] = (;role=:total_effect,logical=plan.target,family=:brm_total,total=block,
        scale=tau,scale_priors=Tuple(plan.scale_priors),scale_configured=Tuple(plan.scale_configured),
        column_ids=Tuple(plan.column_ids),design_labels=Tuple(c.label for c in plan.design.columns),
        absorbed=Tuple(plan.absorbed))
    _sb_record_binding!(data,beta,:population_effect,plan.target)
    _sb_record_binding!(data,tau,:parameter,plan.target)
    nothing
end
