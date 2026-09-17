# Is the FILTER precise enough for THIS posterior? -- the importance-sampling reliability
# workflow of Timonen, Siccha, Bales, Lähdesmäki & Vehtari ("An importance sampling approach
# for reliable and efficient inference in Bayesian ordinary differential equation models",
# arXiv:2205.09059), applied to a continuous-discrete Gaussian filter instead of an ODE solver.
#
# The paper's setting: the likelihood needs a numerical approximation (there, an ODE solver
# with a tolerance), whose error biases the posterior by an unknown amount. Its workflow:
#   1. sample the posterior with a CHEAP approximation M;
#   2. evaluate the same draws under a more PRECISE approximation M*, and Pareto-smooth the
#      importance ratios  p_M*(theta | y) / p_M(theta | y);
#   3. Pareto k-hat small  -> M was good enough; the reweighted draws ARE the M* posterior,
#      at the cost of one likelihood evaluation per draw instead of a second MCMC run;
#      k-hat large -> M is not reliable here: raise its precision and go to 1.
#
# Here the approximation is the kernel's marginalizing filter, and its precision is the pair
# (nsub, gh) of ema_state_dependent.jl: Euler substeps per interval, and the order of the
# predict step (0 = first-order plug-in; 3 / 5 = Gauss-Hermite moment matching).
# Both are DATA, so every rung is the SAME compiled model and shares one parameter space --
# which is exactly what importance weighting between rungs needs.
#
# RESULT (100 subjects x 30, 600 draws per rung, reference nsub=32 / gh=5):
#   (nsub, gh)   fit      sd(log ratio)   k-hat   IS-ESS   cz            cz reweighted
#   ( 2, 0)        98 s   11.72           3.22      1.1    0.478+-0.091  --
#   ( 8, 0)       285 s    1.92           0.84     34.3    0.457+-0.105  --
#   ( 8, 3)      1715 s    2.75           0.81     21.7    0.483+-0.108  --
#   (16, 3)      3009 s    0.92           0.14    288.5    0.474+-0.115  0.476+-0.120   <- certified
# The precision axis that matters is the SUBSTEP COUNT (the analogue of solver step size), not
# the predict order: 3 -> 5 quadrature nodes changes the log-density by 7e-7, and moment
# matching at 8 substeps is no closer to the reference than the first-order predict. Posterior
# means of the 8-substep fits sit within 0.54 / 0.64 certified sds of the certified ones
# (2 substeps: 3.95). Reweighted estimates are reported only for a rung that passes: an
# importance-sampling estimate whose k-hat exceeds the threshold is not an estimate.
#
# Sampler: WarmupHMC.adaptive_warmup_mcmc on the BridgeStan problem.
#
# Run: julia --project=test research/ema_ctsem/ema_state_dependent_psis.jl [outdir]
#      (with `outdir`: each rung's constrained draws + PSIS weights as CSV, and the rung
#       summary `psis_ladder.tsv`)

using BayesianRegressionModels, StanBlocks, LogDensityProblems, BridgeStan, Random, WarmupHMC
using Distributions: Normal, Exponential
import Statistics, PSIS

include(joinpath(@__DIR__, "ema_state_dependent.jl"))   # filter family, fixture, with_filter, model

generating_values() = (; b0=0.5, bm=0.4, a12=-0.25, a21=-0.30, a22=-0.60, cintm=0.3, qd0=-0.2,
                         qd1=0.3, cz=0.7, sdm=0.6, l31=1.2, thr=-1.0, r1=0.3, r2=0.3)

function problem(panel, cfg)
    sb = SBBRMI(ema_state_dependent_model(with_filter(panel; cfg...)); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_sd_psis_$(hash(code)).stan"))
end

wmean(w, v) = sum(w .* v)
wsd(w, v)   = sqrt(sum(w .* (v .- wmean(w, v)) .^ 2))

function main(outdir=nothing;
              ladder=[(nsub=2, gh=0), (nsub=8, gh=0), (nsub=8, gh=3), (nsub=16, gh=3)],   # cheap -> precise
              reference=(nsub=32, gh=5),                                                  # M*
              khat_ok=0.7)
    panel = ema_state_dependent_fixture(n=100, nt=30)
    ref = problem(panel, reference)
    names = BridgeStan.param_names(ref.model)
    summary = ["nsub\tgh\tfit_seconds\treweight_seconds\tsd_log_ratio\tkhat\tis_ess\tdraws"]
    for cfg in ladder
        prob = problem(panel, cfg)
        t0 = time()
        fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), prob; n_draws=600, progress=nothing)
        tfit = time() - t0
        unc = [collect(Float64, c) for c in eachcol(fit.posterior_position)]
        C = reduce(hcat, [BridgeStan.param_constrain(prob.model, u) for u in unc])
        t1 = time()
        lr = [LogDensityProblems.logdensity(ref, u) - LogDensityProblems.logdensity(prob, u) for u in unc]
        tis = time() - t1
        res = PSIS.psis(lr)
        w = vec(res.weights) ./ sum(res.weights)
        khat = res.pareto_shape; ess = 1 / sum(abs2, w)
        push!(summary, join((cfg.nsub, cfg.gh, round(Int, tfit), round(Int, tis),
                             round(Statistics.std(lr); digits=3), round(khat; digits=3),
                             round(ess; digits=1), length(unc)), '\t'))
        println("\n== filter ", cfg, "  ->  reference ", reference, " ==")
        println("fit ", round(Int, tfit), " s   reweighting ", round(Int, tis), " s   draws ", length(unc),
                "   Pareto k-hat ", round(khat; digits=2), "   IS-ESS ", round(ess; digits=1))
        println(rpad("param", 7), rpad("true", 8), rpad("mean", 9), rpad("sd", 8), rpad("z", 8),
                rpad("IS mean", 9), rpad("IS sd", 8), "IS z")
        tr = generating_values()
        for k in keys(tr)
            v = C[findfirst(==(String(k)), names), :]
            m, s = Statistics.mean(v), Statistics.std(v); mw, sw = wmean(w, v), wsd(w, v)
            println(rpad(k, 7), rpad(tr[k], 8), rpad(round(m; digits=3), 9), rpad(round(s; digits=3), 8),
                    rpad(round((m - tr[k]) / s; digits=2), 8), rpad(round(mw; digits=3), 9),
                    rpad(round(sw; digits=3), 8), round((mw - tr[k]) / sw; digits=2))
        end
        if outdir !== nothing
            open(joinpath(outdir, "psis_nsub$(cfg.nsub)_gh$(cfg.gh).csv"), "w") do io
                println(io, join(vcat(names, ["log_ratio", "psis_weight"]), ","))
                for i in eachindex(unc); println(io, join(vcat(C[:, i], [lr[i], w[i]]), ",")); end
            end
            write(joinpath(outdir, "psis_ladder.tsv"), join(summary, '\n') * "\n")
        end
        flush(stdout)                                    # a rung takes minutes to an hour: show it when it lands
        if khat <= khat_ok
            println("k-hat <= ", khat_ok, ": this filter is reliable for this posterior -- stop; the IS columns are the reference posterior.")
            return
        end
        println("k-hat > ", khat_ok, ": not reliable here -- raise the filter's precision and refit.")
    end
    println("\nladder exhausted without a reliable rung")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? nothing : ARGS[1])
end
