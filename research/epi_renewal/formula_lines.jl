# Wren.jl PR #16 (multi-patch renewal model) as brm FORMULA LINES (decision 1my2ezn: grow rw + cdar).
#
# The log-reproduction-number process is stated on the formula surface:
#   single patch   log_R ~ 1 + rw(time)                                      (intercept = Z_1, sd(:, rw) = σ)
#   six patches    log_R ~ 1 + rw(time) + cdar(week; by=patch, cor=C)        (shared walk + correlated damped weekly deviations)
# and the per-patch seeds keep the PR's own prior line `log_I0 ~ MvNormal(seed_mean, 0.5)` (a prior
# statement, not a regression term: `0 + factor(patch)` emits treatment contrasts, so it cannot give
# one seed per patch — measured 2026-09-16). The mechanistic map — renewal recursion + reporting-delay
# convolution + NegBin(mean, cluster²) — stays a family, as the PR itself writes it as a function.
#
# Everything else (PMFs, simulation, the Stan functions, the fit + recovery helpers) comes from wren16.jl.
#   Run: julia --project=test research/epi_renewal/formula_lines.jl [n_draws]

include(joinpath(@__DIR__, "wren16.jl"))   # main guarded

# ── the six-patch observation family on the long (day, patch) frame ───────────
# rows are column-major over (day, patch): i = t + (g - 1) * T, so `to_matrix` reshapes the row
# vectors without copying; T = N / P. `log_I0` is the per-patch vector parameter.
StanBlocks.@deffun begin
    patch_expected_flat(log_R::vector[N], log_I0::vector[P], gamma::real, pop::vector[P], dist_flat::vector[PP],
                        gen_pmf::vector[G], delay_pmf::vector[D])::vector[N] = begin
        T = N / P
        R = to_matrix(exp(log_R), T, P)
        K = gravity_K(pop, dist_flat, gamma)
        I = patch_I(R, K, log_I0, gen_pmf)
        patch_Y(I, R, log_I0, gen_pmf, delay_pmf)
    end
    @lhs @lpxf patch_nb_lpmf(cases::int[N], log_R::vector[N], log_I0::vector[P], cluster::real, gamma::real,
                             pop::vector[P], dist_flat::vector[PP], gen_pmf::vector[G], delay_pmf::vector[D])::real = begin
        Y = patch_expected_flat(log_R, log_I0, gamma, pop, dist_flat, gen_pmf, delay_pmf)
        lp = 0.0
        for i in 1:N
            lp += neg_binomial_2_lpmf(cases[i], Y[i], 1.0 / (cluster * cluster))
        end
        lp
    end
    patch_nb_lpmfs(cases::int[N], log_R::vector[N], log_I0::vector[P], cluster::real, gamma::real,
                   pop::vector[P], dist_flat::vector[PP], gen_pmf::vector[G], delay_pmf::vector[D])::vector[N] = begin
        Y = patch_expected_flat(log_R, log_I0, gamma, pop, dist_flat, gen_pmf, delay_pmf)
        lp::vector[N]
        for i in 1:N
            lp[i] = neg_binomial_2_lpmf(cases[i], Y[i], 1.0 / (cluster * cluster))
        end
        lp
    end
    patch_nb_rng(int[N], log_R::vector[N], log_I0::vector[P], cluster::real, gamma::real,
                 pop::vector[P], dist_flat::vector[PP], gen_pmf::vector[G], delay_pmf::vector[D])::int[N] = begin
        Y = patch_expected_flat(log_R, log_I0, gamma, pop, dist_flat, gen_pmf, delay_pmf)
        out::int[N]
        for i in 1:N
            out[i] = neg_binomial_2_rng(Y[i], 1.0 / (cluster * cluster))
        end
        out
    end
end

# ── the models ────────────────────────────────────────────────────────────────
single_rw_formula(d) = @brm d begin
    log_I0  ~ Normal(log(50.0), 0.5)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    log_R   ~ 1 + rw(time)                                  # the walk, as a formula line
    effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)        # Z_1
    sd(:, rw(time)) ~ Normal(0.0, 0.05)                     # σ
    Y       = expected_cases(log_R, log_I0, gen_pmf, delay_pmf)
    cases   ~ nb_clust(Y, cluster)
end

patch_formula(d) = @brm d begin
    gamma   ~ Normal(1.5, 0.5; lower=0.0)
    cluster ~ Normal(0.0, 0.1; lower=0.0)
    log_R   ~ 1 + rw(time) + cdar(week; by=patch, cor=C)   # shared walk + spatially correlated weekly deviations
    effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)        # Z_1
    sd(:, rw(time)) ~ Normal(0.0, 0.05)                     # σ
    sd(:, cdar(week)) ~ Normal(0.0, 0.2)                    # σ_δ
    ar(:, cdar(week)) ~ Normal(0.8, 0.1)                    # ρ
    log_I0  ~ MvNormal(seed_mean, 0.5)                      # per-patch seeds (the PR's line)
    Yf      = patch_expected_flat(log_R, log_I0, gamma, pop, dist_flat, gen_pmf, delay_pmf)
    cases_flat ~ patch_nb(log_R, log_I0, cluster, gamma, pop, dist_flat, gen_pmf, delay_pmf)
end

# the long (day, patch) frame, column-major: row i = t + (g - 1) * T
patch_long(p) = (; time=repeat(p.time; outer=p.n_patches), week=repeat(p.week; outer=p.n_patches),
                  patch=repeat(1:p.n_patches; inner=p.T), cases_flat=vec(p.cases),
                  seed_mean=p.seed_mean, C=exp.(-p.dist ./ p.ell), pop=p.pop, dist_flat=vec(p.dist),
                  gen_pmf=p.gen_pmf, delay_pmf=p.delay_pmf)

function main_formula(; n_draws=1000)
    s = simulate_single()
    t1 = build(single_rw_formula, single_data(s))
    println("single_rw_formula: dim ", LogDensityProblems.dimension(t1.problem))
    r1 = fit("single_rw_formula", t1.problem; n_draws)
    scalars = Any[]
    # the term parameters are the point: intercept = Z_1, sd(:, rw(time)) = σ
    for (nm, tr, label) in (("pop_log_R_beta_pop", s.init, "init (intercept)"), ("rw_log_R_time_sigma", s.sigma, "sigma (rw)"),
                            ("log_I0", s.log_I0, "log_I0"), ("cluster", s.cluster, "cluster"))
        q = qs(r1.cons[only(rows_for(r1, nm)), :])
        println(@sprintf("  %-18s median %.4f  95%% [%.4f, %.4f]  truth %.4f  %s", label, q.q50, q.q025, q.q975, tr, q.q025 <= tr <= q.q975 ? "in" : "OUT"))
        push!(scalars, merge((; model="single, formula lines", name=label, truth=tr), q))
    end
    R1 = band_rows(r1, "log_R", s.R; transform=exp, keys=j -> (; day=j))
    println(@sprintf("  single: R 95%% coverage %.0f%%", 100cov90(R1)))
    println("  stems: ", join(unique(stem.(r1.names)), " "))

    p = simulate_patches()
    t2 = build(patch_formula, patch_long(p))
    println("patch_formula: dim ", LogDensityProblems.dimension(t2.problem))
    r2 = fit("patch_formula", t2.problem; n_draws)
    println("  stems: ", join(unique(stem.(r2.names)), " "))
    for (nm, tr, label) in (("pop_log_R_beta_pop", p.init, "init (intercept)"), ("rw_log_R_time_sigma", p.sigma, "sigma (rw)"),
                            ("cdar_log_R_week_sigma", p.sigma_d, "sigma_d (cdar sd)"), ("cdar_log_R_week_rho", p.rho, "rho (cdar ar)"),
                            ("gamma", p.gamma, "gamma"), ("cluster", p.cluster, "cluster"))
        q = qs(r2.cons[only(rows_for(r2, nm)), :])
        println(@sprintf("  %-18s median %.4f  95%% [%.4f, %.4f]  truth %.4f  %s", label, q.q50, q.q025, q.q975, tr, q.q025 <= tr <= q.q975 ? "in" : "OUT"))
        push!(scalars, merge((; model="six patches, formula lines", name=label, truth=tr), q))
    end
    T, P = p.T, p.n_patches
    PR = band_rows(r2, "log_R", vec(p.R); transform=exp, keys=j -> (; day=(j - 1) % T + 1, patch=(j - 1) ÷ T + 1))
    PY = band_rows(r2, "Yf", vec(p.Y_t); keys=j -> (; day=(j - 1) % T + 1, patch=(j - 1) ÷ T + 1))
    PS = band_rows(r2, "log_I0", p.log_I0; keys=j -> (; patch=j))
    println(@sprintf("  six patches: 95%% coverage R %.0f%%, Yf %.0f%%, seeds %.0f%%", 100cov90(PR), 100cov90(PY), 100cov90(PS)))
    save_json("formula_single_R.json", R1); save_json("formula_patch_R.json", PR); save_json("formula_patch_Y.json", PY)
    save_json("formula_scalars.json", scalars)
    save_json("formula_receipt.json", Dict("fits" => [Dict(string(k) => v for (k, v) in pairs(r.diag)) for r in (r1, r2)]))
    (; r1, r2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_formula(; n_draws=length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000)
end
