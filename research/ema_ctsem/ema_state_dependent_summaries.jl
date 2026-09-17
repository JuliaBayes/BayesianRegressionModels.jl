# Posterior summaries of the certified fit, as small tables for the figures and the docs:
#   results/certified_posterior.tsv   parameter, generating value, mean, sd, 2.5 %, 97.5 %
#   results/state_dependence.tsv      the three state-dependent cells as posterior bands over
#                                     the latent state, next to the generating curves
# Input: the draws that ema_state_dependent_psis.jl writes for the rung that passed its
# Pareto k-hat check (default psis_nsub16_gh3.csv in `outdir`).
#
# Run: julia research/ema_ctsem/ema_state_dependent_summaries.jl <outdir> [draws.csv]

import Statistics: mean, std, quantile

softplus(x) = log1p(exp(x))

function main(outdir, file="psis_nsub16_gh3.csv")
    lines = readlines(joinpath(outdir, file))
    header = split(lines[1], ',')
    X = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    draws(name) = [x[findfirst(==(name), header)] for x in X]
    results = joinpath(@__DIR__, "results"); mkpath(results)

    truth = (; b0=0.5, bm=0.4, a12=-0.25, a21=-0.30, a22=-0.60, cintm=0.3, qd0=-0.2,
               qd1=0.3, cz=0.7, sdm=0.6, l31=1.2, thr=-1.0, r1=0.3, r2=0.3)
    open(joinpath(results, "certified_posterior.tsv"), "w") do io
        println(io, "parameter\tgenerating\tmean\tsd\tq025\tq975")
        for k in keys(truth)
            v = draws(String(k)); q = quantile(v, [0.025, 0.975])
            println(io, join((k, truth[k], round(mean(v); digits=4), round(std(v); digits=4),
                              round(q[1]; digits=4), round(q[2]; digits=4)), '\t'))
        end
    end

    b0, bm, qd0, qd1, cz = draws("b0"), draws("bm"), draws("qd0"), draws("qd1"), draws("cz")
    cells = (
        ("stress auto-effect DRIFT[1,1]", "mood",   range(-2, 3; length=41),
            x -> -softplus.(b0 .+ bm .* x),  x -> -softplus(truth.b0 + truth.bm * x)),
        ("stress diffusion sd DIFFUSION[1,1]", "mood", range(-2, 3; length=41),
            x -> exp.(qd0 .+ qd1 .* x),      x -> exp(truth.qd0 + truth.qd1 * x)),
        ("shock correlation DIFFUSION[2,1]", "stress", range(-2.5, 2.5; length=41),
            x -> tanh.(cz .* x),             x -> tanh(truth.cz * x)),
    )
    open(joinpath(results, "state_dependence.tsv"), "w") do io
        println(io, "cell\tstate\tx\tq025\tq50\tq975\tgenerating")
        for (cell, state, grid, posterior, generating) in cells, x in grid
            q = quantile(posterior(x), [0.025, 0.5, 0.975])
            println(io, join((cell, state, round(x; digits=3), round(q[1]; digits=4), round(q[2]; digits=4),
                              round(q[3]; digits=4), round(generating(x); digits=4)), '\t'))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
