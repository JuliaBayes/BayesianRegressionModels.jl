# Wren.jl PR #16 (epi-example @ 6e3fd026) — the SIX COUPLED PATCHES renewal model
# in native @brm.
#
#   log R_{g,t} = Z_t + delta_{g,w(t)}                      shared trend + weekly patch deviations
#   Z            : the single-patch random walk               (init, sigma, T-1 innovations)
#   delta_1      = sig_d L eta_1,  delta_w = rho delta_{w-1} + sig_d sqrt(1-rho^2) L eta_w
#                  L L' = C, C_gh = exp(-d_gh / ell), ell = 30 fixed;  sig_d ~ N+(0,0.2), rho ~ N(0.8,0.1) on [0,1]
#   K(gamma)     : row-normalised gravity mixing, K_gh ∝ p̃_g p̃_h / d_gh^gamma (g≠h), K_gg ∝ 1;  gamma ~ N+(1.5,0.5)
#   I_{g,t}      = R_{g,t} * sum_h K_gh * sum_s g_s I_{h,t-s}   (per-patch seeded history from I_{0,g}, growth from R_{g,1})
#   log I_{0,g}  ~ N(seed_mean_g, 0.5),  seed_mean = log 50 in the least populous patch, log 0.05 elsewhere
#   Y_{g,t}      = sum_d p_d I_{g,t-d};   cases_{g,t} ~ NegBin(mean Y, var Y + c^2 Y^2)
#
# THE COUPLING is the point: I_{g,t} reads EVERY patch's history through K, so the
# recursion is one sequential scan over t of a P-vector — it cannot be one
# independent kernel cell per patch (a cell sees only its own slice). It is one
# `@deffun` scan over the whole T×P system, called from a custom `@lpxf` family
# that is the @brm RESPONSE (the pattern single_patch.jl verified).
#
# What the formula surface cannot declare (measured in single_patch.jl's probe):
# a top-level VECTOR parameter. The T-1 trend innovations, the P·W deviation
# innovations and the P seeds are therefore declared inside ONE global
# `kernel(...)` cell — a pure vector-latent declarator that returns them
# concatenated; `Lat.mem` (the ragged result's flat backing) carries them to the
# top-level observation. The kernel's required grouping comes from an unused
# zero-mean dummy ranef (`g0 ~ 0 + (1 | series)`, lowered to GQ).
#
# Gate: @brm build -> SBBRMI -> stan_code -> stanc_check -> stan_instantiate ->
# finite log-density + gradient.   Run: julia --project=test research/epi_renewal/multi_patch.jl
#
# VERIFIED (strato2, test env from test/setup_env.jl, StanBlocks pin bec23bc3):
#   patch_model  OK  dim=115  (55 trend innovations + 48 deviation innovations (6 patches × 8 weeks)
#                + 6 seeds + init + sig + sig_d + rho + gamma + cluster), finite gradient;
#                the dummy ranef is GQ-lowered and costs no sampler coordinate.

include(joinpath(@__DIR__, "single_patch.jl"))   # fixtures + clamp2 / R_to_r / lagged_sum / gate; its main() is guarded

# ── spatial structure and the coupled recursion as Stan functions ────────────
StanBlocks.@deffun begin
    # d_gh from the column-major flattened distance matrix (vec(dist) in Julia)
    dist_at(dist_flat::vector[PP], g::int, h::int, P::int)::real = dist_flat[g + (h - 1) * P]
    # row-normalised gravity mixing matrix; dist^gamma spelled exp(gamma * log(dist)) (no `pow` builtin)
    gravity_K(pop::vector[P], dist_flat::vector[PP], gamma::real)::matrix[P, P] = begin
        mean_pop = sum(pop) / P
        K::matrix[P, P]
        for g in 1:P
            rs = 0.0
            for h in 1:P
                K[g, h] = g == h ? 1.0 :
                          (pop[g] / mean_pop) * (pop[h] / mean_pop) / exp(gamma * log(dist_at(dist_flat, g, h, P)))
                rs += K[g, h]
            end
            for h in 1:P
                K[g, h] = K[g, h] / rs
            end
        end
        K
    end
    # Cholesky factor of the fixed spatial correlation C_gh = exp(-d_gh / ell)
    corr_chol(dist_flat::vector[PP], P::int, ell::real)::matrix[P, P] = begin
        C::matrix[P, P]
        for g in 1:P
            for h in 1:P
                C[g, h] = exp(-dist_at(dist_flat, g, h, P) / ell)
            end
        end
        cholesky_decompose(C)
    end
    # damped weekly random walk of patch deviations with spatially correlated innovations: matrix[P, W]
    deviations(eta::vector[PW], L::matrix[P, P], sig_d::real, rho::real, W::int)::matrix[P, W] = begin
        E = to_matrix(eta, P, W)
        delta::matrix[P, W]
        prev = sig_d * (L * col(E, 1))
        delta[:, 1] = prev
        scale = sig_d * sqrt(1.0 - rho * rho)
        for w in 2:W
            prev = rho * prev + scale * (L * col(E, w))
            delta[:, w] = prev
        end
        delta
    end
    # the whole coupled system from the concatenated latents lat = [z (T-1); eta (P·W); log_I0 (P)]:
    # expected reported cases, flattened column-major (Y[t + (g-1)T]). `wgrid` supplies W by its length.
    patch_expected_cases(lat::vector[NL], init::real, sig::real, sig_d::real, rho::real, gamma::real,
                         pop::vector[P], dist_flat::vector[PP], week::int[T], wgrid::vector[W],
                         gen_pmf::vector[G], delay_pmf::vector[D])::vector[T * P] = begin
        z = lat[1:(T - 1)]
        eta = lat[T:(T - 1 + P * W)]
        log_I0 = lat[(T + P * W):(T - 1 + P * W + P)]
        Z = append_row(init, init + sig * cumulative_sum(z))     # vector[T]
        K = gravity_K(pop, dist_flat, gamma)
        L = corr_chol(dist_flat, P, 30.0)
        delta = deviations(eta, L, sig_d, rho, W)                # matrix[P, W]
        R::matrix[T, P]
        for g in 1:P
            for t in 1:T
                R[t, g] = exp(Z[t] + delta[g, week[t]])
            end
        end
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
        Y::matrix[T, P]
        for g in 1:P
            for t in 1:T
                Y[t, g] = lagged_sum(col(I, g), t, delay_pmf, I0[g], growth[g], 0)
            end
        end
        to_vector(Y)
    end
end

# ── the observation model as a custom family: cases_flat ~ patch_renewal_negbin(Lat.mem, ...) ──
StanBlocks.@deffun begin
    @lhs @lpxf patch_renewal_negbin_lpmf(cases::int[TP], lat::vector[NL], init::real, sig::real, sig_d::real,
            rho::real, gamma::real, cluster::real, pop::vector[P], dist_flat::vector[PP], week::int[T],
            wgrid::vector[W], gen_pmf::vector[G], delay_pmf::vector[D])::real = begin
        Y = patch_expected_cases(lat, init, sig, sig_d, rho, gamma, pop, dist_flat, week, wgrid, gen_pmf, delay_pmf)
        phi = 1.0 / (cluster * cluster)
        lp = 0.0
        for i in 1:TP
            lp += neg_binomial_2_lpmf(cases[i], Y[i], phi)
        end
        lp
    end
    patch_renewal_negbin_lpmfs(cases::int[TP], lat::vector[NL], init::real, sig::real, sig_d::real,
            rho::real, gamma::real, cluster::real, pop::vector[P], dist_flat::vector[PP], week::int[T],
            wgrid::vector[W], gen_pmf::vector[G], delay_pmf::vector[D])::vector[TP] = begin
        Y = patch_expected_cases(lat, init, sig, sig_d, rho, gamma, pop, dist_flat, week, wgrid, gen_pmf, delay_pmf)
        phi = 1.0 / (cluster * cluster)
        out::vector[TP]
        for i in 1:TP
            out[i] = neg_binomial_2_lpmf(cases[i], Y[i], phi)
        end
        out
    end
    patch_renewal_negbin_rng(int[TP], lat::vector[NL], init::real, sig::real, sig_d::real,
            rho::real, gamma::real, cluster::real, pop::vector[P], dist_flat::vector[PP], week::int[T],
            wgrid::vector[W], gen_pmf::vector[G], delay_pmf::vector[D])::int[TP] = begin
        Y = patch_expected_cases(lat, init, sig, sig_d, rho, gamma, pop, dist_flat, week, wgrid, gen_pmf, delay_pmf)
        phi = 1.0 / (cluster * cluster)
        out::int[TP]
        for i in 1:TP
            out[i] = neg_binomial_2_rng(Y[i], phi)
        end
        out
    end
end

# ── the @brm model ───────────────────────────────────────────────────────────
patch_model(d) = @brm d begin
    init    ~ Normal(log(1.3), 0.1)                      # Z_1
    sig     ~ Normal(0.0, 0.05; lower=0.0)               # trend RW step sd
    sig_d   ~ Normal(0.0, 0.2; lower=0.0)                # deviation sd
    rho     ~ Normal(0.8, 0.1; lower=0.0, upper=1.0)     # weekly damping
    gamma   ~ Normal(1.5, 0.5; lower=0.0)                # gravity distance exponent
    cluster ~ Normal(0.0, 0.1; lower=0.0)                # NegBin cluster factor
    g0      ~ 0 + (1 | series)                           # grouping dummy for the one-cell kernel (unused -> GQ)
    # one global cell: declares the vector latents and returns them concatenated
    Lat ~ kernel(time, pop_r, wgrid_r, seed_mean_r, g0) do ts, pp, wg, sm, gd
        z::vector[dims(ts)[1] - 1] ~ std_normal()                   # trend innovations (T-1)
        eta::vector[dims(pp)[1] * dims(wg)[1]] ~ std_normal()       # deviation innovations (P·W)
        lI0::vector[dims(pp)[1]] ~ normal(sm, 0.5)                  # per-patch seeds (P)
        append_row(z, append_row(eta, lI0))
    end
    cases_flat ~ patch_renewal_negbin(Lat.mem, init, sig, sig_d, rho, gamma, cluster,
                                      pop, dist_flat, week, wgrid, gen_pmf, delay_pmf)
end

function patch_data(p)
    wgrid = collect(1.0:p.n_weeks)
    (; series=["all"],
       time=[p.time], pop_r=[p.pop], wgrid_r=[wgrid], seed_mean_r=[p.seed_mean],   # ragged, in-cell (sizes + seed prior)
       cases_flat=vec(p.cases), pop=p.pop, dist_flat=vec(p.dist), week=p.week, wgrid,  # flat, top-level
       gen_pmf=p.gen_pmf, delay_pmf=p.delay_pmf)
end

function main_patches()
    p = simulate_patches()
    println("six-patch renewal: T=", p.T, " P=", p.n_patches, "  per-patch cases=", vec(sum(p.cases; dims=1)))
    println("── six-patch @brm (one global cell, coupled @deffun scan) ──")
    gate("patch_model", patch_model, patch_data(p))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_patches()
end
