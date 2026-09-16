# Wren.jl PR #16 (epi-example @ 6e3fd026) — PRIOR PREDICTIVE checks in native @brm.
#
# The PR draws every component's return value and the case series from the prior before
# fitting. In BRM the same program with `held_out=:all` has every observation likelihood off:
# NUTS on it samples the PRIOR, the named in-cell expectation (`Y` / `Yf`) is the prior
# expected-case path, and the auto twin `cases_gen` / `cases_flat_gen` is the prior
# predictive case series. This checks that (a) `held_out=:all` covers an in-cell observation
# of a ragged integer slice, (b) the prior predictive spans the simulated truth, and (c) the
# prior on R (via `Z` = log R) is what the PR states (log R_1 ~ N(log 1.3, 0.1), walk sd ~ N+(0, 0.05)).
#   Run: julia --project=test research/epi_renewal/prior_predictive.jl [n_draws]
#
# VERIFIED (strato2, StanBlocks pin 9a958f97, 1000 prior draws): held_out=:all removes the in-cell
#   observation from the model block in both programs; prior NUTS 0 divergences (min ESS 701 / 658).
#   single: prior log R_1 5/50/95% = 0.087 / 0.260 / 0.425 (PR: N(log 1.3 = 0.262, 0.1)); prior
#   predictive cases day 1 [15, 91], day 56 [11, 122561] — the no-depletion tail; simulated truth
#   inside the 90% band on 56/56 days; 3/1000 draws overflow neg_binomial_2_rng (2^30), counted.
#   six patches: truth inside the 90% band on 297/336 cells; 1/1000 draws overflow; prior gamma
#   [0.63, 2.35], rho [0.65, 0.94], sig_d [0.01, 0.38] (5–95%) around the truths 1.5 / 0.8 / 0.15.

include(joinpath(@__DIR__, "recover.jl"))   # models, fixtures, fit/recovery helpers; mains guarded

function instantiate_prior(buildfn, d)
    brmi = buildfn(d)
    sb = SBBRMI(brmi; mod=@__MODULE__, held_out=:all)
    code = StanBlocks.stan_code(sb.model)
    stanc = StanBlocks.stanc_check(code; warn_pedantic=false)
    stanc.ok || error("stanc rejected the prior program:\n" * stanc.output)
    # no observation statement may survive in the model block
    model_block = code[findfirst("model {", code)[1]:findfirst("generated quantities", code)[1]]
    println("  held_out=:all model block mentions the observation family: ", occursin("nb_clust", model_block) || occursin("renewal_negbin", model_block))
    (; brmi, sb, problem=StanBlocks.stan_instantiate(sb.model), code)
end

function quantile_band(r, want, probs=(0.05, 0.5, 0.95))
    idx = rows_for(r, want)
    isempty(idx) && (println("  ", want, ": no matching names; stems: ", join(unique(stem.(r.names)), " ")); return nothing)
    reduce(hcat, [quantile(finite(r.cons[i, :]), collect(probs)) for i in idx])'   # rows: element, cols: probs
end

function main_prior(; n_draws=1000)
    # ── single patch ──
    s = simulate_single()
    d_cell = (; series=["all"], time=[s.time], cases=[s.cases], gen_pmf=[s.gen_pmf], delay_pmf=[s.delay_pmf])
    println("── single patch prior (held_out=:all) ──")
    t = instantiate_prior(single_kernel_rw, d_cell)
    r = fit_model("single_kernel_rw prior", t.problem; n_draws, tolerate_gq_failures=true)
    println("  stems: ", join(unique(stem.(r.names)), " "))
    R = quantile_band(r, "Z")                      # log R path
    if R !== nothing
        println(@sprintf("  prior log R_1: 5%%/50%%/95%% = %.3f / %.3f / %.3f  (PR: N(log 1.3 = %.3f, 0.1))", R[1, 1], R[1, 2], R[1, 3], log(1.3)))
        println(@sprintf("  prior log R_T: 5%%/50%%/95%% = %.3f / %.3f / %.3f", R[end, 1], R[end, 2], R[end, 3]))
    end
    C = quantile_band(r, "cases_gen")
    if C !== nothing
        inside = count(t -> C[t, 1] <= s.cases[t] <= C[t, 3], 1:s.T)
        println(@sprintf("  prior predictive cases: day 1 5%%/95%% = %.0f / %.0f; day T = %.0f / %.0f; simulated truth inside the 90%% band on %d/%d days",
                         C[1, 1], C[1, 3], C[end, 1], C[end, 3], inside, s.T))
    end
    # ── six patches ──
    p = simulate_patches()
    println("── six-patch prior (held_out=:all) ──")
    t2 = instantiate_prior(patch_model, patch_data(p))
    r2 = fit_model("patch_model prior", t2.problem; n_draws, tolerate_gq_failures=true)
    println("  stems: ", join(unique(stem.(r2.names)), " "))
    C2 = quantile_band(r2, "cases_flat_gen")
    if C2 !== nothing
        truth = vec(p.cases)
        inside = count(i -> C2[i, 1] <= truth[i] <= C2[i, 3], eachindex(truth))
        println(@sprintf("  prior predictive cases (T×P flattened): simulated truth inside the 90%% band on %d/%d cells; median of the last day per patch: %s",
                         inside, length(truth), join(string.(round.(Int, C2[end-p.n_patches+1:end, 2])), " ")))
    end
    for (nm, tr) in (("gamma", p.gamma), ("rho", p.rho), ("sig_d", p.sigma_d))
        idx = rows_for(r2, nm); isempty(idx) && continue
        x = r2.cons[idx[1], :]
        println(@sprintf("  prior %-6s 5%%/50%%/95%% = %.3f / %.3f / %.3f  (truth %.3f)", nm, q05(x), median(finite(x)), q95(x), tr))
    end
    (; r, r2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    main_prior(; n_draws=n)
end
