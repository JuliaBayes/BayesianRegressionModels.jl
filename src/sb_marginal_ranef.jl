# ---- Likelihood-inactive margins of a shared `|ID|` block ------------------
#
# A shared `(… | p | g)` bucket samples one correlated block across every
# predictor that names `p`. When a response is omitted from the data, some of
# those predictors may reach no likelihood at all (a joint PK/QT/TGI model fitted
# without QT keeps `qt_base`/`qt_slope` margins that only feed the forward-
# simulated QT response). StanBlocks places variables, not matrix elements, so
# the whole block stayed sampled: the active margins' Cholesky rows mix in the
# inactive margins' standardized draws.
#
# The block's law factorises exactly. Order the margins active-first (the LKJ
# density is exchangeable, so any principal submatrix will do):
#
# - the active `a × a` correlation submatrix of `LKJ(K, eta)` is
#   `LKJ(a, eta + (K - a) / 2)`, the active scales keep their own independent
#   priors, and the active effects are `diag(tau_A) * L_A * z_A`;
# - given the active block, the remaining Cholesky rows follow the onion
#   construction (Lewandowski, Kurowicka & Joe 2009, section 3.2): row
#   `j = a + r` is `[sqrt(y_r) * u_r', sqrt(1 - y_r)]` with
#   `y_r ~ Beta((j - 1) / 2, eta + (K - j) / 2)` and `u_r` uniform on the unit
#   sphere in `j - 1` dimensions, independently of the rows above it. That is the
#   exact conditional law of the `LKJ(K, eta)` completion of `L_A`.
#
# So BRM samples only the active margins (`ranef_correlated_draws_marginal` over
# top-level `<bucket>_active_L` / `<bucket>_active_tau`), and completes the block
# in one generated `ranef_correlated_draws_completion` statement bound to the
# bucket's ORIGINAL name. Its outputs `<bucket>`, `<bucket>_L`, `<bucket>_tau`
# and `<bucket>_z_flat` keep the full program's names, shapes and margin order;
# like any other omitted-response quantity they move to generated quantities.
# The joint law of every output equals the full model's; only the sampled
# coordinates shrink.
#
# Correctness never depends on the activity decision: the split is an exact
# reparametrisation of the same prior, so a margin misjudged inactive only puts
# the completion's draws back into `parameters`. The decision itself comes from
# StanBlocks' traced placement of each target's emitted gather
# (`r_<target>_<bucket suffix>`), read before any held-out or cv mark.

# Marker for the activity probe's own build: no warnings, no held-out marks, no
# further probing.
struct _SBMarginalProbe end

StanBlocks.@deffun begin
    # The permuted (active-first) Cholesky factor: `L_active` on top, then one
    # onion row per inactive margin from its squared norm `y[r]` and direction
    # coordinates (consecutive slices of `u`).
    brm_onion_extend(L_active::matrix[a, a], y::vector[m], u::vector[q],
                     k::int)::matrix[k, k] = begin
        Lp = rep_matrix(0.0, k, k)
        for i in 1:a
            for j in 1:i
                Lp[i, j] = L_active[i, j]
            end
        end
        pos = 0
        for r in 1:m
            row = a + r
            d = row - 1
            norm2 = 0.0
            for i in 1:d
                norm2 = norm2 + u[pos + i] * u[pos + i]
            end
            scale = sqrt(y[r] / norm2)
            for i in 1:d
                Lp[row, i] = scale * u[pos + i]
            end
            Lp[row, row] = sqrt(1.0 - y[r])
            pos = pos + d
        end
        Lp
    end
    # The Cholesky factor of the same correlation matrix in the bucket's own
    # margin order: `order[p]` is the original margin at permuted position `p`.
    brm_corr_cholesky_permute(Lp::matrix[k, k], order::int[k])::matrix[k, k] = begin
        Omega_p = Lp * Lp'
        Omega = rep_matrix(0.0, k, k)
        for i in 1:k
            for j in 1:k
                Omega[order[i], order[j]] = Omega_p[i, j]
            end
        end
        cholesky_decompose(Omega)
    end
    brm_scatter_margins(active::vector[a], inactive::vector[m], active_idx::int[a],
                        inactive_idx::int[m], k::int)::vector[k] = begin
        out = rep_vector(0.0, k)
        for i in 1:a
            out[active_idx[i]] = active[i]
        end
        for i in 1:m
            out[inactive_idx[i]] = inactive[i]
        end
        out
    end
    # Every margin's effects, `n_groups × k` in the bucket's margin order: the
    # active columns are `b_active` itself; each inactive column conditions on
    # the active standardized draws recovered from `b_active`.
    brm_complete_draws(b_active::matrix[g, a], Lp::matrix[k, k], tau_active::vector[a],
                       tau_inactive::vector[m], z_inactive::vector[q],
                       active_idx::int[a], inactive_idx::int[m])::matrix[g, k] = begin
        out = rep_matrix(0.0, g, k)
        za = rep_vector(0.0, a)
        for grp in 1:g
            for i in 1:a
                acc = b_active[grp, i] / tau_active[i]
                for p in 1:(i - 1)
                    acc = acc - Lp[i, p] * za[p]
                end
                za[i] = acc / Lp[i, i]
                out[grp, active_idx[i]] = b_active[grp, i]
            end
            for r in 1:m
                row = a + r
                cond = 0.0
                for p in 1:a
                    cond = cond + Lp[row, p] * za[p]
                end
                for p in 1:r
                    cond = cond + Lp[row, a + p] * z_inactive[(grp - 1) * m + p]
                end
                out[grp, inactive_idx[r]] = tau_inactive[r] * cond
            end
        end
        out
    end
    # The standardized draws `z` of `b = (diag(tau) * L * z)'`, flattened
    # column-major exactly as the ordinary block's `z_flat`.
    brm_standardize_draws(b::matrix[g, k], L::matrix[k, k],
                          tau::vector[k])::vector[k * g] = begin
        out = rep_vector(0.0, k * g)
        z = rep_vector(0.0, k)
        for grp in 1:g
            for i in 1:k
                acc = b[grp, i] / tau[i]
                for p in 1:(i - 1)
                    acc = acc - L[i, p] * z[p]
                end
                z[i] = acc / L[i, i]
                out[(grp - 1) * k + i] = z[i]
            end
        end
        out
    end
end

# The sampled active block. `L` and `tau` are caller-supplied top-level
# parameters (`<carrier>_L`, `<carrier>_tau`), which the completion reads too.
ranef_correlated_draws_marginal = StanBlocks.@slic begin
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = to_matrix(z_flat, n_terms, n_groups)
    return (diag_pre_multiply(tau, L) * z)'
end

# Centered sibling: the effects themselves are sampled.
ranef_correlated_draws_centered_marginal = StanBlocks.@slic begin
    b_cols ~ plate(; outer=(n_groups,)) do g
        bc::vector[n_terms] ~ multi_normal_cholesky(
            rep_vector(0., n_terms), diag_pre_multiply(tau, L))
        bc
    end
    return b_cols'
end

# The exact conditional completion of the inactive margins, bound to the
# bucket's original name. Its outputs mirror the ordinary non-centered block.
ranef_correlated_draws_completion = StanBlocks.@slic begin
    tau_inactive ~ std_normal(; n=n_inactive, lower=0.0)
    onion_y::vector[n_inactive] ~ beta(onion_alpha, onion_beta)
    onion_u ~ std_normal(; n=n_onion)
    z_inactive ~ std_normal(; n=n_inactive * n_groups)
    L_permuted = brm_onion_extend(L_active, onion_y, onion_u, n_terms)
    L = brm_corr_cholesky_permute(L_permuted, margin_order)
    tau = brm_scatter_margins(tau_active, tau_inactive, active_idx, inactive_idx, n_terms)
    z_flat = brm_standardize_draws(
        brm_complete_draws(b_active, L_permuted, tau_active, tau_inactive, z_inactive,
                           active_idx, inactive_idx), L, tau)
    z = to_matrix(z_flat, n_terms, n_groups)
    return brm_complete_draws(b_active, L_permuted, tau_active, tau_inactive, z_inactive,
                              active_idx, inactive_idx)
end

# A bucket the split supports: a plain grouping factor whose block is Gaussian
# with sampled scales (default, configured `sd`/`cor`, centered or not). Student-t
# mixing weights and R2D2-derived scales keep the ordinary full block.
_sb_marginal_eligible(desc, ranef_effect, r2d2_tau) =
    desc isa NamedColumn && isnothing(r2d2_tau) &&
    (isnothing(ranef_effect) || isnothing(ranef_effect.dist))

# Emit the split block for one bucket. `per_target_ranges` is the bucket's
# `target => column range` table; `inactive` names the targets no likelihood
# reaches. Returns the group index and, per target, the binding and columns its
# gather reads: active targets read the sampled carrier at their positions among
# the active margins, inactive targets read the completion at their original
# columns.
function _sb_emit_marginal_id_bucket!(stmts, data, bucket_name, n_terms_name,
                                      desc, per_target_ranges, inactive;
                                      cv_groups, centered_groups, ranef_effect,
                                      mod::Module)
    K = data[n_terms_name]
    active_cols = Int[c for (target, cols) in per_target_ranges
                      if !(target in inactive) for c in cols]
    inactive_cols = Int[c for (target, cols) in per_target_ranges
                        if target in inactive for c in cols]
    a, m = length(active_cols), length(inactive_cols)
    (a >= 1 && m >= 1) || error(
        "sbimpl: marginal `$bucket_name` needs both active and inactive margins")
    idx_name, n_name = _sb_ensure_group_data!(data, desc)
    gname = name(desc)
    # An explicit `ranef(:, ID) ~ Normal()` is the default law (see
    # `_sb_emit_id_bucket_sampling!`).
    configured = !isnothing(ranef_effect) && (ranef_effect.has_sd || ranef_effect.has_cor)
    eta = configured ? ranef_effect.lkj_eta : 1.0
    priors = configured ? ranef_effect.sd_prior : Any[nothing for _ in 1:K]
    resolved = map(p -> isnothing(p) ? ExprColumn(Normal) : p, priors)

    carrier = Symbol(bucket_name, :_active)
    L_name, tau_name = Symbol(carrier, :_L), Symbol(carrier, :_tau)
    n_active, n_inactive = Symbol(n_terms_name, :_active), Symbol(n_terms_name, :_inactive)
    # Onion row `r` sits at permuted position `j = a + r`: its squared norm is
    # `Beta((j - 1) / 2, eta + (K - j) / 2)` over a `j - 1`-dimensional direction.
    data_keys = (; active_idx=Symbol(bucket_name, :_active_margins),
                 inactive_idx=Symbol(bucket_name, :_inactive_margins),
                 margin_order=Symbol(bucket_name, :_margin_order),
                 onion_alpha=Symbol(bucket_name, :_onion_alpha),
                 onion_beta=Symbol(bucket_name, :_onion_beta),
                 n_onion=Symbol(bucket_name, :_n_onion))
    data_values = (; active_idx=active_cols, inactive_idx=inactive_cols,
                   margin_order=vcat(active_cols, inactive_cols),
                   onion_alpha=Float64[(a + r - 1) / 2 for r in 1:m],
                   onion_beta=Float64[eta + (K - a - r) / 2 for r in 1:m],
                   n_onion=m * a + m * (m - 1) ÷ 2)
    for (key, value) in ((n_active, a), (n_inactive, m), zip(data_keys, data_values)...)
        haskey(data, key) && error(
            "sbimpl: marginal `$bucket_name` reserves data key `$key`, but that " *
            "name is already used")
        data[key] = value
        _sb_record_static!(data, key)
    end

    # Under cv each statement sizes itself by its own `<binding>_n_g` local, the
    # spelling replay checks for (`_sb_assert_cv_reemission`).
    carrier_groups, completion_groups = n_name, n_name
    if gname in cv_groups
        carrier_groups, completion_groups = Symbol(carrier, :_n_g), Symbol(bucket_name, :_n_g)
        push!(stmts, :($carrier_groups = maximum($idx_name)))
        push!(stmts, :($completion_groups = maximum($idx_name)))
    end
    # The active correlation submatrix of `LKJ(K, eta)` is `LKJ(a, eta + m / 2)`.
    push!(stmts, Expr(:call, :~, Expr(:(::), L_name, Expr(:ref, :cholesky_factor_corr, n_active)),
                      Expr(:call, :lkj_corr_cholesky, eta + m / 2)))
    scale = _sb_vector_positive_prior_statement(tau_name, tau_name, resolved[active_cols],
        n_active; direct_native=true, mod, scope=@__MODULE__)
    push!(stmts, scale.statement)
    for hyper in (L_name, tau_name)
        _sb_record_binding!(data, hyper, :random_effect, gname; marginal_of=bucket_name)
    end

    centered = gname in centered_groups
    family = centered ? :ranef_correlated_draws_centered_marginal :
                        :ranef_correlated_draws_marginal
    _sb_record_binding!(data, carrier, :random_effect, gname; family,
                        marginal_of=bucket_name, margins=Tuple(active_cols))
    push!(stmts, Expr(:call, :~, carrier, Expr(:call,
        centered ? ranef_correlated_draws_centered_marginal : ranef_correlated_draws_marginal,
        Expr(:parameters, Expr(:kw, :group_idx, idx_name), Expr(:kw, :n_groups, carrier_groups),
             Expr(:kw, :n_terms, n_active), Expr(:kw, :L, L_name),
             Expr(:kw, :tau, tau_name)))))

    completion = _sb_vector_positive_priors(ranef_correlated_draws_completion,
        :tau_inactive, resolved[inactive_cols]; direct_native=true, mod)
    _sb_record_binding!(data, bucket_name, :random_effect, gname;
                        family=:ranef_correlated_draws_completion,
                        active=carrier, margins=Tuple(inactive_cols))
    push!(stmts, Expr(:call, :~, bucket_name, Expr(:call, completion.model,
        Expr(:parameters, Expr(:kw, :group_idx, idx_name), Expr(:kw, :n_groups, completion_groups),
             Expr(:kw, :n_terms, n_terms_name), Expr(:kw, :n_inactive, n_inactive),
             Expr(:kw, :L_active, L_name), Expr(:kw, :tau_active, tau_name),
             Expr(:kw, :b_active, carrier),
             (Expr(:kw, field, key) for (field, key) in pairs(data_keys))...,
             (Expr(:kw, d, d) for d in union(scale.dependencies,
                                              completion.dependencies))...))))

    position = Dict(c => i for (i, c) in enumerate(active_cols))
    ranges = [target => (target in inactive ? (bucket_name, cols) :
                         (carrier, position[first(cols)]:position[last(cols)]))
              for (target, cols) in per_target_ranges]
    (; idx_name, ranges)
end

# Which candidate targets reach no likelihood, read from StanBlocks' placement of
# the emitted gathers in `probe` (a build without cv or held-out marks). Each
# gather is a top-level `r = ...` whose RHS slices the bucket at known columns
# (`_sb_emit_id_ranef_block!`); a target is inactive when every gather of its
# columns is a generated quantity. Only a proper, nonempty subset of a bucket's
# targets is returned. A bucket the body reads any other way is left whole.
function _sb_marginal_id_probe(probe, candidates)
    kinds = Dict(o.name => o.kind for o in StanBlocks.stan_descriptor(probe.model).outputs)
    body = probe.model.model
    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    out = Dict{Tuple{Symbol,Any},Set{Symbol}}()
    for c in candidates
        reads = _sb_marginal_gathers(statements, c.bucket_name, probe.data)
        isnothing(reads) && continue
        read_cols = Set{Int}(col for (_, cols) in reads for col in cols)
        all(range -> issubset(last(range), read_cols), c.ranges) || continue
        active_cols = Set{Int}(col for (lhs, cols) in reads
                               if get(kinds, lhs, nothing) !== :generated_quantity
                               for col in cols)
        inactive = Set{Symbol}(target for (target, cols) in c.ranges
                               if isdisjoint(cols, active_cols))
        0 < length(inactive) < length(c.ranges) && (out[c.key] = inactive)
    end
    out
end

# `(lhs, columns)` for every statement reading `bucket`, or `nothing` when one
# reads it other than as a top-level `lhs = f(bucket[idx, cols])` gather.
function _sb_marginal_gathers(statements, bucket::Symbol, data)
    reads = Tuple{Symbol,Vector{Int}}[]
    for stmt in statements
        _sb_marginal_mentions(stmt, bucket) || continue
        stmt isa Expr && stmt.head === :(=) && stmt.args[1] isa Symbol || return nothing
        refs = Expr[]
        _sb_marginal_bucket_refs!(refs, stmt.args[2], bucket)
        length(refs) == 1 && length(only(refs).args) == 3 || return nothing
        colspec = only(refs).args[3]
        cols = colspec isa Integer ? Int[colspec] :
            colspec isa Symbol && get(data, colspec, nothing) isa AbstractVector{<:Integer} ?
                collect(Int, data[colspec]) : nothing
        isnothing(cols) && return nothing
        # Every other mention must be that one slice.
        _sb_marginal_mentions(stmt.args[2], bucket; skip=only(refs)) && return nothing
        push!(reads, (stmt.args[1], cols))
    end
    reads
end

function _sb_marginal_bucket_refs!(out, x::Expr, bucket)
    x.head === :ref && !isempty(x.args) && x.args[1] === bucket && return push!(out, x)
    foreach(a -> _sb_marginal_bucket_refs!(out, a, bucket), x.args)
    out
end
_sb_marginal_bucket_refs!(out, _, _) = out

# Whether `x` names `bucket` anywhere other than inside `skip` or as the LHS of
# its own `bucket ~ ...` definition.
_sb_marginal_mentions(x::Symbol, bucket; skip=nothing) = x === bucket
function _sb_marginal_mentions(x::Expr, bucket; skip=nothing)
    x === skip && return false
    x.head === :call && length(x.args) == 3 && x.args[1] === :~ && x.args[2] === bucket &&
        return _sb_marginal_mentions(x.args[3], bucket; skip)
    any(a -> _sb_marginal_mentions(a, bucket; skip), x.args)
end
_sb_marginal_mentions(x, bucket; skip=nothing) = false
