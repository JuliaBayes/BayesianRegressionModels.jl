# Posterior-preserving sum-to-zero (S2Z) planner and SBBRMI emission.
#
# Opt-in per grouping factor via `SBBRMI(...; s2z_groups=[:g], s2z_rho=...)`.
# For one ordinary exchangeable Gaussian random-effect block with matching
# population design, this emits Sean's brms PR #1919 construction: J-1 free
# orthonormal contrast coordinates per coefficient with a fixed projected
# partial map (`rho` data), exact Gaussian marginalization of the omitted
# block means into the population coefficients (`theta`), and generated
# recovery quantities. See `research/s2z_design/README.md` for the staged
# design and `src/s2z_kernel.jl` for the shared pure-Julia geometry.
#
# First scope (fail-closed outside it): scalar independent blocks
# (`zero_correlation || single column`), one grouping structure per predictor,
# J >= 2 levels, fully matched square population design, Flat/Normal
# population priors (no Student-t mixture yet), fixed scalar or per-coefficient
# `rho`, no shared `|ID|` buckets, no stratified/cv/centered/totals overlap.
#
# `s2z_coordinates=:groups` samples J group coordinates per coefficient instead
# of the J-1 contrasts: `s_j ~ N(0, tau^(2 c_j))`, `w = s ./ tau.^c` and
# `delta = tau * (w - mean(w))`. This is Sean's projected per-group map plus one
# auxiliary dimension along the direction the projection removes: `mean(w)` is
# an independent N(0, 1/J) that never reaches the likelihood, so the collapsed
# posterior is unchanged. Every group is then an independent scalar cell, which
# is what per-group WarmupHMC centering needs. `s2z_rho` holds the per-group
# power-interpolation centeredness `c` here (0 = noncentered, 1 = centered).

const _SB_S2Z_PLANS_KEY = :__brm_s2z_plans__

# Compile-time information only. Numerical design, prior and weight constants
# live in model data; descriptors/replay use the same binding record.
"""Fitted design, prior and coordinate identities for one S2Z block."""
struct S2ZEffectBlock
    predictor::Symbol
    group::Symbol
    binding::Symbol
    scales::Symbol
    population::Symbol
    deviations::Symbol
    recovered::Symbol
    effects::Symbol
    group_index::Symbol
    group_count::Symbol
    columns::Tuple
    population_columns::Tuple
    B::Matrix{Float64}
    location::Vector{Float64}
    precision::Vector{Float64}
    rho::Matrix{Float64}
    design::Matrix{Float64}
    coordinates::Symbol
end

"""Return the S2Z blocks selected during model construction."""
s2z_effect_blocks(model) = sort!(
    [b.s2z for b in values(model.bindings) if hasproperty(b, :s2z)];
    by=b->String(b.predictor))

function _s2z_resolve_rho(rho, J::Int, K::Int, group::Symbol;
                          coordinates::Symbol=:contrasts)
    isnothing(rho) && coordinates === :groups && (rho = 0.0)
    isnothing(rho) && throw(ArgumentError(
        "S2Z block `$group`: pass explicit `s2z_rho` (no public default yet); " *
        "a scalar in [0,1], one weight per coefficient, or a J-by-K matrix"))
    weights = rho isa Real ? fill(Float64(rho), J, K) :
        rho isa AbstractVector ? repeat(reshape(collect(Float64, rho), 1, :), J, 1) :
        rho isa AbstractMatrix ? collect(Float64, rho) :
        throw(ArgumentError(
            "S2Z block `$group`: `s2z_rho` must be a scalar, a vector with " *
            "one weight per coefficient, or a J-by-K matrix"))
    size(weights) == (J, K) || throw(ArgumentError(
        "S2Z block `$group`: `s2z_rho` has size $(size(weights)) but the " *
        "block has J = $J groups and K = $K coefficients"))
    all(w -> isfinite(w) && 0 <= w <= 1, weights) || throw(ArgumentError(
        "S2Z block `$group`: centering weights must lie in [0,1]"))
    weights
end

function _sb_s2z_plan(brmi, prepared, predictor, overrides, rho;
                      cv_groups, centered_groups, coordinates::Symbol=:contrasts)
    target = predictor.name
    declarations = filter(d -> d.predictor === target, prepared.context.group_declarations)
    isempty(declarations) && return nothing
    group = nothing
    try
        any(d -> d.raw_group isa MultiMembershipTerm || d.descriptor isa Tuple, declarations) &&
            throw(ArgumentError("multi-membership or stratified blocks"))
        plans = _brm_simple_random_effect_plans(brmi, target, prepared.context)
        isnothing(plans) && throw(ArgumentError("unresolved random-effect plans"))
        group = first(plans).group
        all(p -> p.group === group && (p.zero_correlation || length(p.columns) == 1), plans) ||
            throw(ArgumentError("crossed, correlated or multi-group structure"))
        group in cv_groups && throw(ArgumentError("`cv_groups` overlap"))
        group in centered_groups && throw(ArgumentError("`centered_groups` overlap"))
        any(p -> !isnothing(p.id), plans) &&
            throw(ArgumentError("shared `|ID|` buckets"))
        J = length(first(plans).levels)
        J >= 2 || throw(ArgumentError("J = 1 level (no contrast coordinates)"))
        pop, ran, direct = Any[], Any[], Any[]
        foreach(t -> _sb_classify_term!(t, pop, ran, direct), _sb_terms(predictor.expression))
        isempty(direct) || throw(ArgumentError("direct terms"))
        design = _brm_population_design(target, Tuple(pop), prepared.context.data,
                                        get(prepared.context.target_obs, target, nothing))
        (isnothing(design) || isempty(design.columns) || !isempty(design.fixed_terms)) &&
            throw(ArgumentError("missing or fixed-term population design"))
        columns = Tuple(c for p in plans for c in p.columns)
        all(c -> isnothing(c.preprocess) || c.preprocess.kind === :protect, columns) ||
            throw(ArgumentError("transformed random-effect columns"))
        length(unique(c.label for c in columns)) == length(columns) ||
            throw(ArgumentError("duplicate random-effect columns"))
        K = length(columns)
        maps = [_sb_total_basis_map((column,), columns) for column in design.columns]
        all_priors = _sb_pop_effect_overrides(overrides, target)
        isnothing(all_priors) && (all_priors = fill(nothing, length(design.columns)))
        infos = map(_sb_total_population_prior, all_priors)
        any(i -> !isnothing(maps[i]) && isnothing(infos[i]), eachindex(maps)) &&
            throw(ArgumentError("unsupported population prior"))
        any(i -> !isnothing(maps[i]) && !isnothing(infos[i]) && !isnothing(infos[i].nu),
            eachindex(maps)) && throw(ArgumentError(
                "Student-t population prior (mixture support follows in a later slice)"))
        absorbed = findall(i -> !isnothing(maps[i]) && !isnothing(infos[i]), eachindex(maps))
        length(absorbed) == K || throw(ArgumentError(
            "population design does not fully match the $K random-effect columns"))
        A = hcat((maps[i] for i in absorbed)...)
        rank(A) == K || throw(ArgumentError("rank-deficient population/group design map"))
        # X * B = Z with B = inv(A): the omitted-mean shift of each theta.
        B = Matrix(inv(A))
        Z = hcat([collect(Float64, c.values) for c in columns]...)
        N = size(Z, 1)
        X = hcat([begin
            col = design.columns[i]
            isnothing(col.source) ? ones(Float64, N) : collect(Float64, col.values)
        end for i in absorbed]...)
        size(X, 1) == N || throw(ArgumentError("population/group row mismatch"))
        resid = maximum(abs.(X * B - Z)) / (1 + maximum(abs.(Z)))
        resid < 1e-8 || throw(ArgumentError(
            "population/group design identity fails numerically (residual $resid)"))
        remaining = setdiff(collect(eachindex(maps)), absorbed)
        frozen = _sb_frozen_preproc_entry(prepared.context.data, Symbol(:s2z_B_, target),
                                          :total_basis, target)
        isnothing(frozen) || (B = copy(frozen.const_))
        prior_info = infos[absorbed]
        scale_priors = Any[]
        for plan in plans, c in plan.columns
            push!(scale_priors, isnothing(c.source) ? ExprColumn(LogNormal, 0., 1.) : nothing)
        end
        for prior in scale_priors
            isnothing(prior) && continue
            prior isa ExprColumn || throw(ArgumentError("unsupported scale prior"))
            isempty(getkwargs(prior)) || throw(ArgumentError("unsupported scale prior"))
            getf(prior) in (Normal, LogNormal, Cauchy, Exponential, TDist, LocationScale, Gamma) ||
                throw(ArgumentError("unsupported scale prior"))
            isempty(_sb_prior_references(_sb_effect_prior_arg(prior))) ||
                throw(ArgumentError("symbolic scale prior"))
        end
        weights = _s2z_resolve_rho(rho, J, K, group; coordinates)
        return (; target, group, columns, design, B, prior_info, scale_priors,
                absorbed, remaining, weights, Z, coordinates,
                remaining_priors=all_priors[remaining],
                indices=first(plans).indices, levels=first(plans).levels)
    catch err
        err isa ArgumentError || rethrow()
        label = isnothing(group) ? string(target) : "`$group`"
        throw(ArgumentError("sbimpl: S2Z is unavailable for group $label: $(err.msg); " *
              "requires one independent Gaussian grouping structure with J >= 2, " *
              "fully matched population design and supported priors"))
    end
end

function _sb_plan_s2zs(brmi, prepared, overrides, selection, rho;
                       cv_groups, centered_groups, coordinates::Symbol=:contrasts)
    coordinates in (:contrasts, :groups) || throw(ArgumentError(
        "s2z_coordinates must be :contrasts or :groups, got $(repr(coordinates))"))
    selection isa Symbol || selection isa Tuple || selection isa AbstractVector ||
        selection isa AbstractSet ||
        throw(ArgumentError("s2z_groups must be a grouping-factor name or a collection (empty disables S2Z)"))
    requested = Set(selection isa Symbol ? (selection,) : selection)
    isempty(requested) && return Dict{Symbol,Any}()
    out = Dict{Symbol,Any}()
    for predictor in prepared.predictors
        # Selection probe only; the throwing planner below gives the reason.
        probe = _brm_simple_random_effect_plans(brmi, predictor.name, prepared.context)
        (isnothing(probe) || isempty(probe)) && continue
        any(p -> p.group in requested, probe) || continue
        plan = _sb_s2z_plan(brmi, prepared, predictor, overrides, rho;
                            cv_groups, centered_groups, coordinates)
        isnothing(plan) && continue
        out[predictor.name] = plan
    end
    missing = setdiff(requested, Set(p.group for p in values(out)))
    isempty(missing) || throw(ArgumentError(
        "S2Z requested for group(s) $(join(sort!(collect(missing)), ", ")) with no " *
        "eligible predictor; requires one independent Gaussian grouping structure, " *
        "matched population design and supported priors"))
    out
end

# S2Z Stan functions. The Helmert expansion is implicit (suffix sums, O(J));
# no dense contrast matrix is ever built or stored. Flat population directions
# (precision 0) contribute exactly 0.0 to the collapsed density; their omitted
# means are recovered from the prior conditional on the proper directions.
# (No docstrings here: docstring → @deffun AssertionError.)
StanBlocks.@deffun begin
    brm_s2z_deviations(z::matrix[jm1, m], tau::vector[m],
                       rho::matrix[j, m])::matrix[j, m] = begin
        out = rep_matrix(0., j, m)
        scale = rep_vector(1., j)
        u = rep_vector(0., j)
        w = rep_vector(0., j)
        suffix = 0.
        s = 1.
        k = 1
        for c in 1:m
            scale = 1.0 - rho[:, c] + rho[:, c] * tau[c]
            suffix = 0.
            for t in 1:jm1
                k = jm1 + 1 - t
                s = inv(sqrt(k * (k + 1)))
                suffix += z[k, c] * s
                u[k + 1] = suffix - (k + 1) * z[k, c] * s
            end
            u[1] = suffix
            w = u ./ scale
            out[:, c] = tau[c] * (w - rep_vector(mean(w), j))
        end
        out
    end

    # Group coordinates (`s2z_coordinates=:groups`): `w = s ./ tau.^c` is iid
    # N(0, 1) under the prior and `delta = tau * (w - mean(w))` has the exact
    # zero-sum N(0, tau^2 * P) law; `mean(w)` is the auxiliary dimension.
    brm_s2z_group_deviations(s::matrix[j, m], tau::vector[m],
                             c::matrix[j, m])::matrix[j, m] = begin
        out = rep_matrix(0., j, m)
        w = rep_vector(0., j)
        for col in 1:m
            w = s[:, col] .* exp(-c[:, col] * log(tau[col]))
            out[:, col] = tau[col] * (w - rep_vector(mean(w), j))
        end
        out
    end

    # Independent per-group N(0, tau^(2c)) cells, the J-dimensional standard
    # normal of `w` pulled back through `s = w .* tau.^c`.
    @lpxf brm_s2z_group_lpdf(s::matrix[j, m], tau::vector[m],
                            c::matrix[j, m])::real = begin
        lp = -0.5 * j * m * 1.8378770664093453
        for col in 1:m
            lp += -0.5 * dot_self(s[:, col] .* exp(-c[:, col] * log(tau[col]))) -
                sum(c[:, col]) * log(tau[col])
        end
        lp
    end

    brm_s2z_effects(r::matrix[j, m], mu::vector[m])::matrix[j, m] = begin
        out = r
        for c in 1:m
            out[:, c] = r[:, c] + rep_vector(mu[c], j)
        end
        out
    end

    # Contrast density: the pulled-back N(0, tau^2) on deviations, NOT an
    # N(0, I) pushforward. z enters only through r = deviations(z): the quad
    # -0.5*||r||^2/tau^2 plus the d-term normalizers -sum(log d) + log(mean d)
    # (brms's log_det_partial; the (J-1)*log(tau) cancels against the tau
    # normalizer). Endpoints check: rho = 0 gives N(z; 0, I), rho = 1 gives
    # N(z; 0, tau^2 I). An N(0, I) base here would silently rescale the
    # contrast prior variance to (tau/d)^2.
    @lpxf brm_s2z_contrast_lpdf(z::matrix[jm1, m], tau::vector[m],
                               rho::matrix[j, m])::real = begin
        r = brm_s2z_deviations(z, tau, rho)
        lp = -0.5 * jm1 * m * 1.8378770664093453
        scale = rep_vector(1., j)
        for c in 1:m
            scale = 1.0 - rho[:, c] + rho[:, c] * tau[c]
            lp += -0.5 * dot_self(r[:, c]) / square(tau[c]) - sum(log(scale)) +
                log(mean(scale))
        end
        lp
    end

    # Collapsed population density via Woodbury on the MxM capped system
    # H = S^-1 + B' diag(precision) B (always PD): flat directions carry
    # precision 0 and contribute exactly 0.0 without any index partition.
    @lpxf brm_s2z_theta_lpdf(theta::vector[p], tau::vector[m],
                             B::matrix[p, m], location::vector[p],
                             precision::vector[p], n_groups::int)::real = begin
        diff = theta - location
        Ddiff = precision .* diff
        H = diag_matrix(inv(square(tau) / n_groups)) + B' * diag_matrix(precision) * B
        LH = cholesky_decompose(H)
        u = B' * Ddiff
        w = mdivide_left_tri_low(LH, u)
        logdet_H = 2 * sum(log(diagonal(LH)))
        n_proper = 0
        logdet_V = 0.
        logdet_S = 0.
        quad = 0.
        for a in 1:p
            if precision[a] > 0.
                n_proper += 1
                logdet_V -= log(precision[a])
            end
        end
        if n_proper == 0
            return 0.
        end
        logdet_S = 2 * sum(log(tau)) - m * log(n_groups)
        quad = dot_product(Ddiff, diff) - dot_self(w)
        -0.5 * (n_proper * 1.8378770664093453 + logdet_H + logdet_S + logdet_V + quad)
    end

    # Omitted-mean recovery draw through the same Woodbury system: gain =
    # W - W B H^-1 B' D with W = S B' D. Flat population directions have
    # zero gain columns (they carry no information about the means).
    brm_s2z_recover_rng(theta::vector[p], tau::vector[m],
                        B::matrix[p, m], location::vector[p],
                        precision::vector[p], n_groups::int)::vector[m] = begin
        prior_var = square(tau) / n_groups
        diff = theta - location
        BD = B' * diag_matrix(precision)
        H = diag_matrix(inv(prior_var)) + BD * B
        W = diag_post_multiply(diag_pre_multiply(prior_var, B)', precision)
        HinvBD = mdivide_left_spd(H, BD)
        gain = W - W * B * HinvBD
        mhat = gain * diff
        cov = diag_matrix(prior_var) - gain * B * diag_matrix(prior_var)
        n_proper = 0
        for a in 1:p
            if precision[a] > 0.
                n_proper += 1
            end
        end
        if n_proper == 0
            return sqrt(prior_var) .* to_vector(normal_rng(rep_vector(0., m), 1.))
        end
        multi_normal_rng(mhat, cov)
    end
end

function _sb_emit_s2z!(stmts, data, target, plan; mod::Module=@__MODULE__)
    suffix = plan.target
    z = Symbol(plan.coordinates === :groups ? :s2z_level_ : :s2z_contrast_, suffix)
    tau = Symbol(:s2z_scale_, suffix)
    theta, r = Symbol(:s2z_theta_, suffix), Symbol(:s2z_deviation_, suffix)
    mu, beta = Symbol(:s2z_mean_, suffix), Symbol(:s2z_population_, suffix)
    b = Symbol(:s2z_effect_, suffix)
    idx, ng = Symbol(:s2z_group_, suffix), Symbol(:s2z_ng_, suffix)
    nk, np = Symbol(:s2z_nk_, suffix), Symbol(:s2z_np_, suffix)
    bn, loc, prec = Symbol(:s2z_B_, suffix), Symbol(:s2z_location_, suffix),
                    Symbol(:s2z_precision_, suffix)
    rhon, jm1 = Symbol(:s2z_rho_, suffix), Symbol(:s2z_jm1_, suffix)
    j, k, p = length(plan.levels), length(plan.columns), length(plan.absorbed)
    for (key, value) in (ng => j, jm1 => j - 1, nk => k, np => p, bn => plan.B,
                         loc => Float64[v.location for v in plan.prior_info],
                         prec => Float64[v.precision for v in plan.prior_info])
        data[key] = value
        key === ng || _sb_record_static!(data, key)
    end
    data[idx] = plan.indices
    _sb_record_preproc!(data, bn, PreprocEntry(:total_basis, copy(plan.B), plan.target, true))
    _sb_record_preproc!(data, idx, PreprocEntry(:group_index,
        (; levels=plan.levels, n_groups_key=ng), plan.group, true))
    data[rhon] = copy(plan.weights)
    _sb_record_preproc!(data, rhon, PreprocEntry(:s2z_weights,
        (; levels=plan.levels, rho=copy(plan.weights)), plan.group, true))
    prior = _sb_vector_positive_priors(_brm_total_scales, :tau, plan.scale_priors)
    push!(stmts, :($tau ~ $(prior.model)(; n=$nk)))
    if plan.coordinates === :groups
        push!(stmts, :($z::matrix[$ng, $nk] ~ brm_s2z_group($tau, $rhon)))
    else
        push!(stmts, :($z::matrix[$jm1, $nk] ~ brm_s2z_contrast($tau, $rhon)))
    end
    push!(stmts, :($theta::vector[$np] ~ brm_s2z_theta($tau, $bn, $loc, $prec, $ng)))
    push!(stmts, plan.coordinates === :groups ?
        :($r = brm_s2z_group_deviations($z, $tau, $rhon)) :
        :($r = brm_s2z_deviations($z, $tau, $rhon)))
    push!(stmts, :($mu = brm_s2z_recover_rng($theta, $tau, $bn, $loc, $prec, $ng)))
    push!(stmts, :($beta = $theta - $bn * $mu))
    push!(stmts, :($b = brm_s2z_effects($r, $mu)))
    zcols = Any[isnothing(c.source) ? :(rep_vector(1., num_elements($idx))) :
                _sb_shared_population_column!(data, c) for c in plan.columns]
    zn = Symbol(:s2z_Z_, suffix)
    push!(stmts, :($zn = $(Expr(:call, :hcat, zcols...))))
    xcols = Any[isnothing(plan.design.columns[c].source) ? :(rep_vector(1., num_elements($idx))) :
                _sb_shared_population_column!(data, plan.design.columns[c])
            for c in plan.absorbed]
    xn = Symbol(:s2z_X_, suffix)
    push!(stmts, :($xn = $(Expr(:call, :hcat, xcols...))))
    s2z_lp = :(rows_dot_product($r[$idx, :], $zn) + $xn * $theta)
    if !isempty(plan.remaining)
        remaining_cols = Any[isnothing(plan.design.columns[c].source) ? :(rep_vector(1., num_elements($idx))) :
                             _sb_shared_population_column!(data, plan.design.columns[c])
                         for c in plan.remaining]
        xn_rem, pop = Symbol(:X_, target), Symbol(:pop_, target)
        push!(stmts, :($xn_rem = $(Expr(:call, :hcat, remaining_cols...))))
        prior = _sb_population_prior_rhs(plan.remaining_priors; mod)
        kwargs = Expr(:parameters, Expr(:kw, :X, xn_rem),
            (Expr(:kw, key, value) for (key, value) in pairs(prior.kwargs))...)
        push!(stmts, Expr(:call, :~, pop, Expr(:call, prior.model, kwargs)))
        data[_SB_BINDINGS_KEY][pop] = (; role=:population_effect, logical=plan.target,
            family=nothing,
            population_columns=Tuple(plan.design.columns[c].label for c in plan.remaining))
        s2z_lp = :($s2z_lp + $pop)
    end
    push!(stmts, :($target = $s2z_lp))
    block = S2ZEffectBlock(plan.target, plan.group, z, Symbol(tau, :_tau), theta, r, mu, b,
        idx, ng, Tuple(c.label for c in plan.columns),
        Tuple(plan.design.columns[c].label for c in plan.absorbed),
        plan.B, copy(data[loc]), copy(data[prec]), copy(plan.weights), copy(plan.Z),
        plan.coordinates)
    data[_SB_BINDINGS_KEY][z] = (; role=:s2z_effect, logical=plan.target,
        family=:brm_s2z, s2z=block)
    data[_SB_BINDINGS_KEY][theta] = (; role=:population_effect, logical=plan.target,
        family=nothing, population_columns=block.population_columns)
    _sb_record_binding!(data, beta, :population_effect, plan.target)
    _sb_record_binding!(data, tau, :parameter, plan.target)
    nothing
end

function _s2z_coordinates(model, block, names)
    pos = Dict(String(n) => i for (i, n) in enumerate(names))
    lookup(n) = get(pos, n) do
        throw(ArgumentError("S2Z model expects unconstrained coordinate `$n`"))
    end
    J = model.data[block.group_count]
    K = length(block.columns)
    # One row per free contrast, or per group for `s2z_coordinates=:groups`.
    rows = block.coordinates === :groups ? J : J - 1
    contrasts = [lookup("$(block.binding).$r.$k") for r in 1:rows, k in 1:K]
    theta = [lookup("$(block.population).$p") for p in 1:length(block.population_columns)]
    scales = [lookup("$(block.scales).$k") for k in 1:K]
    (; contrasts, theta, scales)
end

function _s2z_conditional(block, coordinates, draw)
    J, K = size(block.rho)
    tau = exp.(draw[coordinates.scales])
    theta = draw[coordinates.theta]
    prior_var = tau .^ 2 ./ J
    proper = findall(>(0), block.precision)
    if isempty(proper)
        return (; mean=zeros(Float64, K),
                factor=cholesky(Diagonal(copy(prior_var))),
                sd=tau, theta)
    end
    B1 = block.B[proper, :]
    C = Diagonal(inv.(block.precision[proper])) + B1 * Diagonal(prior_var) * B1'
    factor = cholesky(Symmetric(C))
    gain = (Diagonal(prior_var) * B1') / factor
    mean = gain * (theta[proper] - block.location[proper])
    (; mean, factor, gain, B1, prior_var, sd=tau, theta)
end

# Zero-sum deviations of coefficient `k` from its sampled coordinates: Sean's
# partial map over Helmert contrasts, or the centered group coordinates.
function _s2z_block_deviations(block, values, tau, k)
    block.coordinates === :groups || return _s2z_partial_forward(
        _s2z_helmert_mul(values), tau, view(block.rho, :, k))
    w = values .* exp.(-view(block.rho, :, k) .* log(tau))
    tau .* (w .- sum(w) / length(w))
end

"""
    recover_s2z_draws(model, draws, unc_names; rng=Random.default_rng())

Draw the omitted block means conditional on each saved S2Z draw and recover
the original population coefficients (`beta = theta - B*m`) and group effects
(`b = r + m`). Returns one record per predictor with `population`, `means`,
`deviations` and `effects` arrays. Input draws must be in the compiled model's
coordinate frame.
"""
function recover_s2z_draws(model, draws::AbstractMatrix, names;
                           rng::Random.AbstractRNG=Random.default_rng())
    size(draws, 2) == length(names) || throw(DimensionMismatch("draws and names disagree"))
    out = Dict{Symbol,NamedTuple}()
    for block in s2z_effect_blocks(model)
        coordinates = _s2z_coordinates(model, block, names)
        n, (J, K) = size(draws, 1), size(block.rho)
        P = length(block.population_columns)
        population = Matrix{Float64}(undef, n, P)
        means = Matrix{Float64}(undef, n, K)
        deviations = Array{Float64}(undef, n, J, K)
        effects = similar(deviations)
        for i in 1:n
            row = view(draws, i, :)
            conditional = _s2z_conditional(block, coordinates, row)
            tau = conditional.sd
            for k in 1:K
                deviations[i, :, k] = _s2z_block_deviations(block,
                    Vector{Float64}(row[coordinates.contrasts[:, k]]), tau[k], k)
            end
            m = if hasproperty(conditional, :gain)
                cov = Diagonal(conditional.prior_var) -
                      conditional.gain * conditional.B1 * Diagonal(conditional.prior_var)
                conditional.mean + cholesky(Symmetric(cov)).L * randn(rng, K)
            else
                conditional.mean + conditional.factor.L * randn(rng, K)
            end
            means[i, :] = m
            population[i, :] = conditional.theta - block.B * m
            for k in 1:K
                effects[i, :, k] = deviations[i, :, k] .+ m[k]
            end
        end
        out[block.predictor] = (; population, means, deviations, effects,
            population_columns=block.population_columns, columns=block.columns)
    end
    out
end
