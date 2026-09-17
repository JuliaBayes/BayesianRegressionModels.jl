# Figures of docs/src/state-space-models.md, drawn from the small tables in results/.
#
# Run (the repository's plotting environment: AlgebraOfVega + CairoMakie):
#   julia --project=research/adaptive_centering/plots research/ema_ctsem/figures.jl [outdir]
# `outdir` defaults to docs/src/assets/state-space-models.

using AlgebraOfVega, CairoMakie
import AlgebraOfGraphics

results(name) = joinpath(@__DIR__, "results", name)

function table(name)
    lines = readlines(results(name))
    header = Symbol.(split(lines[1], '\t'))
    cell(s) = something(tryparse(Float64, s), String(s))
    [NamedTuple{Tuple(header)}(Tuple(cell.(split(l, '\t')))) for l in lines[2:end]]
end

function save_figure(outdir, name, fig)
    save(joinpath(outdir, name * ".png"), fig; px_per_unit=1.5)
    println("FIGURE ", name)
end

function recovery(outdir)
    rows = [(; parameter=r.parameter, mean=r.mean - r.generating, lo=r.q025 - r.generating, hi=r.q975 - r.generating)
            for r in table("certified_posterior.tsv")]
    order = AlgebraOfGraphics.sorter([r.parameter for r in rows])
    ylabel = "posterior − generating value"
    spec = data(rows) * mapping(:parameter => order => "parameter", :lo => ylabel, :hi => ylabel) * visual(Rangebars) +
           data(rows) * mapping(:parameter => order => "parameter", :mean => ylabel) * visual(Scatter; markersize=11) +
           mapping([0.0]) * visual(HLines; linestyle=:dash, color=:black)
    fig = Figure(; size=(1100, 480), fontsize=16)
    Label(fig[0, 1], "Recovery on 100 subjects × 30 occasions: posterior mean and 95 % interval";
          fontsize=20, font=:bold, tellwidth=false)
    sdraw!(fig[1, 1], spec)
    save_figure(outdir, "recovery", fig)
end

function state_dependence(outdir)
    rows = table("state_dependence.tsv")
    fig = Figure(; size=(1500, 480), fontsize=16)
    Label(fig[0, 1:3], "The three state-dependent cells: posterior median, 95 % band, and the generating curve (dashed)";
          fontsize=20, font=:bold, tellwidth=false)
    for (i, cell) in enumerate(unique(r.cell for r in rows))
        sub = [r for r in rows if r.cell == cell]
        xlabel = "latent $(first(sub).state)"
        band = data(sub) * mapping(:x => xlabel, :q50 => cell) * lineribbon(bands=[:q025 => :q975])
        truth = data(sub) * mapping(:x => xlabel, :generating => cell) * visual(Lines; linestyle=:dash, color=:black)
        sdraw!(fig[1, i], band + truth)
    end
    save_figure(outdir, "state-dependence", fig)
end

function filter_precision(outdir)
    label(r) = string(Int(r.nsub), r.gh == 0 ? ", first-order" : ", moment-matched")
    rows = [(; rung=label(r), khat=r.khat) for r in table("psis_ladder.tsv")]
    order = AlgebraOfGraphics.sorter([r.rung for r in rows])
    spec = data(rows) * mapping(:rung => order => "filter precision: Euler substeps per interval, predict order", :khat => "Pareto k̂ toward the reference filter") *
               visual(BarPlot) +
           mapping([0.7]) * visual(HLines; linestyle=:dash, color=:black)
    fig = Figure(; size=(1000, 480), fontsize=16)
    Label(fig[0, 1], "Is the filter precise enough? Pareto k̂ per precision (dashed: 0.7)";
          fontsize=20, font=:bold, tellwidth=false)
    sdraw!(fig[1, 1], spec)
    save_figure(outdir, "filter-precision", fig)
end

function replication(outdir)
    generators = Dict("xo8" => "Xoshiro stream, 8-step generator", "xo64" => "Xoshiro stream, 64-step generator",
                      "lcg8" => "LCG stream, 8-step generator")
    rows = [(; panel=string(r.generator, " #", Int(r.seed)), generator=generators[r.generator],
               mean=r.mean, lo=r.mean - r.sd, hi=r.mean + r.sd)
            for r in table("replication.tsv") if r.parameter == "cz"]
    order = AlgebraOfGraphics.sorter([r.panel for r in rows])
    ylabel = "cz: posterior mean ± 1 sd"
    spec = data(rows) * mapping(:panel => order => "simulated panel", :lo => ylabel, :hi => ylabel;
                                color=:generator => "data generator") * visual(Rangebars) +
           data(rows) * mapping(:panel => order => "simulated panel", :mean => ylabel;
                                color=:generator => "data generator") * visual(Scatter; markersize=11) +
           mapping([0.7]) * visual(HLines; linestyle=:dash, color=:black)
    fig = Figure(; size=(1300, 520), fontsize=16)
    Label(fig[0, 1:2], "Sampling variability: the state-dependent shock correlation over 21 independent panels (dashed: generating value)";
          fontsize=20, font=:bold, tellwidth=false)
    grid = sdraw!(fig[1, 1], spec * config(axis=(; xticklabelrotation=pi / 3)))
    AlgebraOfGraphics.legend!(fig[1, 2], grid)
    save_figure(outdir, "replication", fig)
end

function main(outdir=normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "state-space-models")))
    mkpath(outdir)
    recovery(outdir); state_dependence(outdir); filter_precision(outdir); replication(outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
