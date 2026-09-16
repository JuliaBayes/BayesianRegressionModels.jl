# Wren.jl PR #16 (epi-example @ 6e3fd026) — the renewal models on TOP-LEVEL VECTOR PARAMETERS.
#
# Decision `187g4va` (2026-09-16) made `x ~ MvNormal(...)` on a non-data LHS a Stan
# `vector[n]` parameter (sbimpl `_sb_emit_vector_prior!`, test/mvnormal_vector_parameters.jl).
# With it the PR's statements are formula statements: no `kernel(...)` cell, no dummy grouping
# ranef, no `.mem`. These are the same densities as single_patch.jl's `single_kernel_rw` and
# multi_patch.jl's `patch_model` (dims 59 / 115), spelled the way the PR spells them:
#
#   eps    ~ MvNormal(zeros(T - 1), 1.0)          iid innovations of the log-R random walk
#   z, eta ~ MvNormal(zeros(...), 1.0)            trend + spatial-deviation innovations
#   log_I0 ~ MvNormal(seed_mean, 0.5)             per-patch seeds (the PR's MvNormal(seed, 0.25 I))
#
# Gate: @brm build -> SBBRMI -> stan_code -> stanc_check -> stan_instantiate -> finite
# log-density + gradient; then NUTS recovery against the same fixed truth as recover.jl.
#   Run: julia --project=test research/epi_renewal/vector_params.jl [n_draws] [all|vector|exposed]
#
# VERIFIED (strato2, StanBlocks pin 9a958f97, BRM c54cd0c, 1000 draws, δ=0.9):
#   single_rw       gate dim 59 (same as single_kernel_rw); 44s, 4 divergences, min ESS 248, R̂ 1.018;
#                   init, sig, log_I0, cluster inside 90% intervals; R path 42/56 (75%). Parameter names
#                   are the formula names (`eps`, `log_R`), no plate suffix, no dummy-ranef coordinates.
#   patch_model_v   gate dim 115 (same as patch_model); 358s, 21 divergences, min ESS 134, R̂ 1.013; all
#                   six scalars in; log_I0 5/6; Yf 314/336 (93%).
#   patch_model_x   gate dim 115; 442s, 2 divergences, min ESS 200, R̂ 1.017; all six scalars in; log_I0
#                   5/6; K_mix 36/36, delta 48/48, R 312/336 (93%), I_t 323/336 (96%), Yf 314/336 (93%) —
#                   every quantity the PR returns, addressable by its own name.

include(joinpath(@__DIR__, "recover.jl"))   # models + fixtures + deffuns + gate + fit/recovery helpers; mains guarded

# A top-level `@brm` assignment is Julia-evaluated, so Stan builtins reach it through a
# `@deffun`; W = number of weeks is computed inside Stan from the week index (the kernel
# spelling passed it in from the cell).
StanBlocks.@deffun begin
    rw_path(init::real, sig::real, eps::vector[K])::vector[K + 1] =
        append_row(init, init + sig * cumulative_sum(eps))
    patch_expected_cases_w(z::vector[Tm1], eta::vector[PW], log_I0::vector[P], init::real, sig::real,
                           sig_d::real, rho::real, gamma::real, pop::vector[P], dist_flat::vector[PP],
                           week::int[T], gen_pmf::vector[G], delay_pmf::vector[D])::vector[T * P] =
        patch_expected_cases(z, eta, log_I0, init, sig, sig_d, rho, gamma, pop, dist_flat, week, gen_pmf, delay_pmf, max(week))
end

# ── single patch: the exact random walk as formula statements ────────────────
single_rw(d) = @brm d begin
    log_I0  ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    sig     ~ Normal(0.0, 0.05; lower=0.0)
    init    ~ Normal(log(1.3), 0.1)                        # Z_1
    eps     ~ MvNormal(zeros(length(time) - 1), 1.0)       # vector[T-1] innovations
    log_R   = rw_path(init, sig, eps)
    cases   ~ renewal_negbin(log_R, log_I0, cluster, gen_pmf, delay_pmf)
end

# ── six coupled patches: the PR's three vector statements, the coupled scan as an assignment ──
patch_model_v(d) = @brm d begin
    init    ~ Normal(log(1.3), 0.1)
    sig     ~ Normal(0.0, 0.05; lower=0.0)
    sig_d   ~ Normal(0.0, 0.2; lower=0.0)
    rho     ~ Normal(0.8, 0.1; lower=0.0, upper=1.0)
    gamma   ~ Normal(1.5, 0.5; lower=0.0)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    z       ~ MvNormal(zeros(length(time) - 1), 1.0)                 # trend innovations (T-1)
    eta     ~ MvNormal(zeros(length(pop) * maximum(week)), 1.0)      # deviation innovations (P·W)
    log_I0  ~ MvNormal(seed_mean, 0.5)                               # per-patch seeds (P)
    Yf      = patch_expected_cases_w(z, eta, log_I0, init, sig, sig_d, rho, gamma, pop, dist_flat, week, gen_pmf, delay_pmf)
    cases_flat ~ nb_clust(Yf, cluster)
end

patch_data_flat(p) = (; time=p.time, cases_flat=vec(p.cases), pop=p.pop, dist_flat=vec(p.dist), week=p.week,
                        wgrid=collect(1.0:p.n_weeks), seed_mean=p.seed_mean, gen_pmf=p.gen_pmf, delay_pmf=p.delay_pmf)

# ── the PR's returned quantities as NAMED top-level assignments ─────────────
# The PR's `patch_renewal_cases` returns (; I_t, Y_t, R, delta, K_mix) for its recovery plots.
# On the formula surface the coupled scan splits into five named assignments — each a
# transformed parameter, each posterior-addressable by its own name (`R`, `I_t`, `delta`,
# `K_mix`, `Yf`) — the same density as `patch_model_v` (dim 115).
StanBlocks.@deffun begin
    patch_delta(eta::vector[PW], pop::vector[P], dist_flat::vector[PP], sig_d::real, rho::real,
                wgrid::vector[W])::matrix[P, W] =
        deviations(eta, corr_chol(dist_flat, P, 30.0), sig_d, rho, W)
    patch_R(z::vector[Tm1], delta::matrix[P, W], init::real, sig::real, week::int[T])::matrix[T, P] = begin
        Z = append_row(init, init + sig * cumulative_sum(z))
        R::matrix[T, P]
        for g in 1:P
            for t in 1:T
                R[t, g] = exp(Z[t] + delta[g, week[t]])
            end
        end
        R
    end
    patch_I(R::matrix[T, P], K::matrix[P, P], log_I0::vector[P], gen_pmf::vector[G])::matrix[T, P] = begin
        I0 = exp(log_I0)
        growth::vector[P]
        for h in 1:P
            growth[h] = R_to_r(R[1, h], gen_pmf)
        end
        I::matrix[T, P]
        lambda::vector[P]
        for t in 1:T
            for h in 1:P
                lambda[h] = lagged_sum(col(I, h), t, gen_pmf, I0[h], growth[h], 1)
            end
            for g in 1:P
                pressure = 0.0
                for h in 1:P
                    pressure += K[g, h] * lambda[h]
                end
                I[t, g] = clamp2(R[t, g] * pressure, 0.0, 1e15)
            end
        end
        I
    end
    patch_Y(I::matrix[T, P], R::matrix[T, P], log_I0::vector[P], gen_pmf::vector[G], delay_pmf::vector[D])::vector[T * P] = begin
        I0 = exp(log_I0)
        Y::matrix[T, P]
        for g in 1:P
            growth = R_to_r(R[1, g], gen_pmf)
            for t in 1:T
                Y[t, g] = lagged_sum(col(I, g), t, delay_pmf, I0[g], growth, 0)
            end
        end
        to_vector(Y)
    end
end

patch_model_x(d) = @brm d begin
    init    ~ Normal(log(1.3), 0.1)
    sig     ~ Normal(0.0, 0.05; lower=0.0)
    sig_d   ~ Normal(0.0, 0.2; lower=0.0)
    rho     ~ Normal(0.8, 0.1; lower=0.0, upper=1.0)
    gamma   ~ Normal(1.5, 0.5; lower=0.0)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    z       ~ MvNormal(zeros(length(time) - 1), 1.0)
    eta     ~ MvNormal(zeros(length(pop) * length(wgrid)), 1.0)
    log_I0  ~ MvNormal(seed_mean, 0.5)
    K_mix   = gravity_K(pop, dist_flat, gamma)                       # matrix[P, P]  mixing weights
    delta   = patch_delta(eta, pop, dist_flat, sig_d, rho, wgrid)    # matrix[P, W]  weekly deviations
    R       = patch_R(z, delta, init, sig, week)                     # matrix[T, P]  reproduction numbers
    I_t     = patch_I(R, K_mix, log_I0, gen_pmf)                     # matrix[T, P]  infections
    Yf      = patch_Y(I_t, R, log_I0, gen_pmf, delay_pmf)            # vector[T*P]   expected cases
    cases_flat ~ nb_clust(Yf, cluster)
end

function main_exposed(; n_draws=1000)
    p = simulate_patches()
    println("── six patches, the PR's returned quantities as named assignments ──")
    gate("patch_model_x", patch_model_x, patch_data_flat(p))
    t = instantiate(patch_model_x, patch_data_flat(p))
    r = fit_model("patch_model_x", t.problem; n_draws, target_acceptance_rate=0.9)
    println("  stems: ", join(unique(stem.(r.names)), " "))
    for (nm, tr) in (("init", p.init), ("sig", p.sigma), ("sig_d", p.sigma_d), ("rho", p.rho), ("gamma", p.gamma), ("cluster", p.cluster))
        report_scalar(r, nm, tr)
    end
    report_path(r, "log_I0", p.log_I0)
    report_path(r, "K_mix", vec(p.K))
    report_path(r, "delta", vec(p.delta))
    report_path(r, "R", vec(p.R))
    report_path(r, "I_t", vec(p.I_t))
    report_path(r, "Yf", vec(p.Y_t))
    r
end

function main_vector(; n_draws=1000)
    s = simulate_single()
    d_flat = (; time=s.time, cases=s.cases, gen_pmf=s.gen_pmf, delay_pmf=s.delay_pmf)
    println("── single patch on a top-level vector parameter ──")
    gate("single_rw", single_rw, d_flat)
    p = simulate_patches()
    println("── six patches on top-level vector parameters ──")
    gate("patch_model_v", patch_model_v, patch_data_flat(p))

    println("── recovery ──")
    t1 = instantiate(single_rw, d_flat)
    r1 = fit_model("single_rw", t1.problem; n_draws, target_acceptance_rate=0.9)
    println("  stems: ", join(unique(stem.(r1.names)), " "))
    report_scalar(r1, "init", s.init); report_scalar(r1, "sig", s.sigma)
    report_scalar(r1, "log_I0", s.log_I0); report_scalar(r1, "cluster", s.cluster)
    report_path(r1, "log_R", s.R; transform=exp)

    t2 = instantiate(patch_model_v, patch_data_flat(p))
    r2 = fit_model("patch_model_v", t2.problem; n_draws, target_acceptance_rate=0.9)
    println("  stems: ", join(unique(stem.(r2.names)), " "))
    for (nm, tr) in (("init", p.init), ("sig", p.sigma), ("sig_d", p.sigma_d), ("rho", p.rho), ("gamma", p.gamma), ("cluster", p.cluster))
        report_scalar(r2, nm, tr)
    end
    report_path(r2, "log_I0", p.log_I0)
    report_path(r2, "Yf", vec(p.Y_t))
    (; r1, r2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    mode = length(ARGS) >= 2 ? ARGS[2] : "all"
    mode in ("all", "vector") && main_vector(; n_draws=n)
    mode in ("all", "exposed") && main_exposed(; n_draws=n)
end
