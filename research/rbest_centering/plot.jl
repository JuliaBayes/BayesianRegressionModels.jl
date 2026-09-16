# Figures for one case: OUT CASE   (reads efficiency_plot.tsv and the pairs tables written by assemble.py / pairs.jl)
using AlgebraOfVega, CairoMakie, CSV, JSON
import AlgebraOfGraphics
out = ARGS[1]; case = ARGS[2]
titles = Dict("AS" => "RBesT AS: ankylosing spondylitis, 8 binomial trials", "crohn" => "RBesT crohn: CDAI change, 6 Gaussian trials")
rows = CSV.File(joinpath(out, "efficiency_plot.tsv"); delim='\t')
order = unique(String.(rows.model))
efficiency = data(rows) * mapping(:value => "", :model => sorter(order) => ""; color=:sampler => "Sampler", dodge_y=:sampler, col=:metric => "") *
    visual(Scatter; markersize=10) *
    config(width=240, height=560, facet=(; linkxaxes=:none),
        scales=scales(X=(; scale=log10), DodgeY=(; width=0.6), Color=(; categories=["Native Stan", "WHMC"], palette=["#D55E00", "#0072B2"])))
open(io -> JSON.print(io, to_vegalite(efficiency; interactive=false)), joinpath(out, "efficiency.aov.json"), "w")
figure = Figure(size=(1450, 720))
Label(figure[0, :], titles[case]; tellwidth=false, fontsize=24)
grid = sdraw!(figure[1, 1], efficiency)
AlgebraOfGraphics.legend!(figure[1, 2], grid)
Label(figure[2, :], "Relative to RBesT 1.11 NCP under its own control · population mean, between-trial SD and trial totals · one chain per arm, 10,000 draws"; tellwidth=false, fontsize=16)
save(joinpath(out, "efficiency.png"), figure; px_per_unit=1.5)
for kind in ("total", "ordinary")
    path = joinpath(out, kind * "_pairs.tsv"); isfile(path) || continue
    pairs = CSV.File(path; delim='\t')
    layer(rows) = data(rows) * mapping(:log_group_sd => "Log between-trial SD", :coordinate => "Coordinate"; row=:panel => "Selected trial", col=:column => "Visualization") *
        visual(Scatter; markersize=3, opacity=0.15) * config(width=270, height=270, facet=(; linkxaxes=:none, linkyaxes=:none))
    spec = layer(pairs)
    # The interactive envelope carries every fifth draw (2,000 per panel) so the KB brief stays small;
    # the PNG and every calculation use all 10,000 draws.
    preview = filter(r -> r.draw % 5 == 0, collect(pairs))
    open(io -> JSON.print(io, to_vegalite(layer(preview); interactive=false)), joinpath(out, kind * "_pairs.aov.json"), "w")
    pairfigure = Figure(size=(1200, 1100)); sdraw!(pairfigure[1, 1], spec)
    save(joinpath(out, kind * "_pairs.png"), pairfigure; px_per_unit=1.5)
end
println("RBEST_PLOTS_COMPLETE ", case)
