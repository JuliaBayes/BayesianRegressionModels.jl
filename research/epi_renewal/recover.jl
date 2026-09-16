# Wren.jl PR #16 (epi-example @ 6e3fd026) — FIT the landed renewal models and check recovery
# of the fixed truth in fixtures.jl, the PR's own recovery workflow: NUTS via
# `WarmupHMC.adaptive_warmup_mcmc` on the BridgeStan target that the gate already builds,
# draws constrained through BridgeStan (parameters + transformed + generated quantities, so
# the named in-cell quantities are read back by name), ESS / R̂ / divergences, and for every
# scalar parameter the posterior mean and 90% interval against the truth; for the latent
# paths (R, Y) the fraction of truths inside their 90% intervals.
#
# Run: julia --project=test research/epi_renewal/recover.jl [n_draws]
# Writes a machine-readable summary next to the log (RECOVER_OUT, default: this directory's
# scratch dir given by the env var, else a temp dir).
#
# VERIFIED (strato2, StanBlocks pin 9a958f97, 1000 draws, seed 1):
#   single_kernel_rw  δ=0.8: 23s, 59 divergences, min ESS 34,  R̂ 1.057 — every scalar (init, sig, log_I0,
#                     cluster) inside its 90% interval; R path 40/56, Y 45/56 inside 90% intervals.
#                     δ=0.9:  8s, 18 divergences, min ESS 57,  R̂ 1.013 — all scalars in; R 41/56, Y 46/56.
#   patch_model       δ=0.9: 414s, 3 divergences, min ESS 204, R̂ 1.011 — all six scalars in (init, sig,
#                     sig_d, rho, gamma, cluster); seeds 5/6; Yf 317/336 (94%) inside 90% intervals.
#                     (δ=0.8: 361s, 1 divergence, min ESS 158, R̂ 1.013, same recovery.)
#   The PR raises the target acceptance to 0.9 for the coupled fit; here the SINGLE-patch kernel
#   spelling is the one with the harder geometry (the same density spelled on a top-level vector
#   parameter, vector_params.jl `single_rw`, samples with 4 divergences and min ESS 248 at δ=0.9).
#   In-cell names come back as `<LHS>_<name>__pl_mem_1` (`Z_Y__pl_mem_1`, `Y_Yf__pl_mem_1`).

include(joinpath(@__DIR__, "multi_patch.jl"))   # fixtures + single_patch (models, gate) + patch_model; mains are guarded

using WarmupHMC, BridgeStan, Random, Statistics, Printf
using MCMCDiagnosticTools

# ── build the BridgeStan target exactly as the gate does ─────────────────────
function instantiate(buildfn, d)
    brmi = buildfn(d)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    (; brmi, sb, problem)
end

# ── fit + constrain + diagnose ───────────────────────────────────────────────
function fit_model(name, problem; n_draws=1000, seed=1, target_acceptance_rate=0.8, tolerate_gq_failures=false)
    seconds = @elapsed fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), problem; n_draws, target_acceptance_rate, progress=nothing)
    q = convert(Matrix{Float64}, fit.posterior_position)                 # coords × draws, unconstrained model frame
    ess, rhat = MCMCDiagnosticTools.ess_rhat(reshape(permutedims(q), size(q, 2), 1, size(q, 1)))  # (iter, chains, dims)
    println(@sprintf("%s (target acceptance %.2f): %d draws in %.0fs, divergent=%d, dim=%d, min ESS=%.0f, max Rhat=%.3f",
                     name, target_acceptance_rate, size(q, 2), seconds, fit.n_divergent_samples, size(q, 1), minimum(ess), maximum(rhat)))
    flush(stdout)
    unc_names = BridgeStan.param_unc_names(problem.model)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    rng = BridgeStan.StanRNG(problem.model, seed)                        # generated quantities need an RNG
    cons = Matrix{Float64}(undef, length(names), size(q, 2))            # constrained (+tp +gq) × draws
    failed = 0
    for j in 1:size(q, 2)
        try
            cons[:, j] = BridgeStan.param_constrain(problem.model, q[:, j]; include_tp=true, include_gq=true, rng)
        catch e
            # a Stan RNG domain error in generated quantities (e.g. neg_binomial_2_rng beyond 2^30 on a
            # prior draw of the no-depletion renewal tail): the draw is a valid parameter draw, so keep it
            # with NaN generated quantities and COUNT it, rather than silently dropping or dying
            tolerate_gq_failures || rethrow()
            failed += 1
            cons[:, j] .= NaN
        end
    end
    failed > 0 && println("  ", failed, " of ", size(q, 2), " draws failed in generated quantities (Stan RNG domain: ",
                          "expected cases beyond neg_binomial_2_rng's 2^30 limit) — their generated quantities are NaN and excluded")
    (; name, fit, q, unc_names, names, cons, ess, rhat, seconds, gq_failed=failed)
end

# BridgeStan names are `x`, `x.1`, `x.1.2`: strip the index tail. A kernel cell's named
# quantities are emitted as `<LHS>_<name>__pl_mem_1` (measured: `Z_Y__pl_mem_1`,
# `Y_lI0__pl_mem_1`) and the cell's return as `<LHS>__pl_mem_1`; strip that plate-member
# suffix too, so `Y` inside `Z ~ kernel(...)` is addressed as `Y` and the return as `Z`.
stem(n) = replace(first(split(n, ['.', '['])), r"__pl_mem_\d+$" => "")
finite(x) = filter(isfinite, x)
q05(x) = quantile(finite(x), 0.05); q95(x) = quantile(finite(x), 0.95)

# the constrained rows for a quantity: exact stem, else `<LHS>_<name>` for an in-cell name
function rows_for(r, want)
    exact = [i for (i, n) in enumerate(r.names) if stem(n) == want]
    isempty(exact) || return exact
    [i for (i, n) in enumerate(r.names) if endswith(stem(n), "_" * want)]
end

function report_scalar(r, want, truth)
    idx = rows_for(r, want)
    length(idx) == 1 || (println(@sprintf("  %-10s no unique row (%d matches: %s)", want, length(idx), join(r.names[idx][1:min(end, 4)], ", "))); return false)
    x = r.cons[idx[1], :]
    lo, hi = q05(x), q95(x)
    println(@sprintf("  %-10s truth=%9.4f  post mean=%9.4f  90%% [%9.4f, %9.4f]  %s", want, truth, mean(finite(x)), lo, hi, lo <= truth <= hi ? "in" : "OUT"))
    lo <= truth <= hi
end

function report_path(r, want, truth::AbstractVector; transform=identity)
    idx = rows_for(r, want)
    isempty(idx) && (println("  ", want, ": no matching names; stems present: ", join(unique(stem.(r.names)), " ")); return nothing)
    length(idx) == length(truth) || (println(@sprintf("  %-10s %d rows vs %d truths", want, length(idx), length(truth))); return nothing)
    inside = count(j -> (x = transform.(r.cons[idx[j], :]); q05(x) <= truth[j] <= q95(x)), eachindex(truth))
    println(@sprintf("  %-10s %d/%d truths inside 90%% intervals (%.0f%%)", want, inside, length(truth), 100inside / length(truth)))
    inside / length(truth)
end

function main_recover(; n_draws=1000)
    println("stems will be printed per model so the in-cell names are known, not guessed")
    # ── single patch, exact random walk (one-cell kernel) ──
    s = simulate_single()
    d_cell = (; series=["all"], time=[s.time], cases=[s.cases], gen_pmf=[s.gen_pmf], delay_pmf=[s.delay_pmf])
    t1 = instantiate(single_kernel_rw, d_cell)
    r1 = nothing
    for delta in (0.8, 0.9)
        r1 = fit_model("single_kernel_rw", t1.problem; n_draws, target_acceptance_rate=delta)
        println("  stems: ", join(unique(stem.(r1.names)), " "))
        report_scalar(r1, "init", s.init)
        report_scalar(r1, "sig", s.sigma)
        report_scalar(r1, "log_I0", s.log_I0)
        report_scalar(r1, "cluster", s.cluster)
        report_path(r1, "Z", s.R; transform=exp)        # Z = log R (the cell's return)
        report_path(r1, "Y", s.Y_t)                      # named in-cell expectation
    end

    # ── six coupled patches ──
    p = simulate_patches()
    t2 = instantiate(patch_model, patch_data(p))
    r2 = fit_model("patch_model", t2.problem; n_draws, target_acceptance_rate=0.9)   # the PR's δ = 0.9 for the coupled fit
    println("  stems: ", join(unique(stem.(r2.names)), " "))
    report_scalar(r2, "init", p.init)
    report_scalar(r2, "sig", p.sigma)
    report_scalar(r2, "sig_d", p.sigma_d)
    report_scalar(r2, "rho", p.rho)
    report_scalar(r2, "gamma", p.gamma)
    report_scalar(r2, "cluster", p.cluster)
    report_path(r2, "lI0", p.log_I0)
    report_path(r2, "Yf", vec(p.Y_t))                # column-major, matches patch_expected_cases
    (; r1, r2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    main_recover(; n_draws=n)
end
