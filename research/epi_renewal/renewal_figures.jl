# Figures of the documentation page "Epidemic renewal models" (docs/src/renewal.md), drawn from the
# summaries that renewal.jl writes to results/renewal/*.json (quantile rows q025 q10 q25 q50 q75 q90 q975
# next to the simulation truth).
#
# Every figure is an AlgebraOfVega spec (AlgebraOfGraphics' algebra, pre-aggregated bands), lowered to
# Vega-Lite and rendered to docs/src/assets/renewal/<name>.png with the `vl-convert` command-line tool.
#
#   Run:  julia --project=<env with AlgebraOfVega> research/epi_renewal/renewal_figures.jl [name ...]
#         (VL_CONVERT=/path/to/vl-convert overrides the renderer)

using AlgebraOfVega, AlgebraOfGraphics, JSON

const RESULTS = joinpath(@__DIR__, "results", "renewal")
const ASSETS = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "renewal"))
const VL_CONVERT = get(ENV, "VL_CONVERT", "vl-convert")
const QUANTILES = (:q025, :q10, :q25, :q50, :q75, :q90, :q975)
const BANDS = [:q025 => :q975, :q10 => :q90, :q25 => :q75]        # 95 / 80 / 50 % central intervals
const ORANGE, PURPLE, INK = "#d95f02", "#7570b3", "#222222"

load(name) = [(; (Symbol(k) => v for (k, v) in row)...) for row in JSON.parsefile(joinpath(RESULTS, name))]
columns(rows) = (; (k => [r[k] for r in rows] for k in keys(first(rows)))...)
# counts go on a logarithmic axis; where zeros occur the axis shows count + 1
shifted(rows, fields, by) = [merge(r, (; (f => r[f] + by for f in fields)...)) for r in rows]

function render(name, spec)
    vl = to_vegalite(spec; interactive=false)
    vl["background"] = "white"                                   # the page also has a dark theme
    mkpath(ASSETS)
    mktempdir() do dir
        path = joinpath(dir, name * ".vl.json")
        open(io -> JSON.print(io, vl), path, "w")
        run(`$VL_CONVERT vl2png --input $path --output $(joinpath(ASSETS, name * ".png")) --scale 2`)
    end
    println("wrote ", joinpath(ASSETS, name * ".png"))
end

# one reference series (a line or points) that is named in the colour legend
series(rows, field, label; x) = (; x=[Float64(r[x]) for r in rows], value=[Float64(r[field]) for r in rows], series=fill(label, length(rows)))

const FIGURES = Dict{String,Function}()

FIGURES["delay"] = function ()
    rows = load("delay_pmf.json")
    xl, yl = "delay from event to report (days)", "probability"
    truth, naive = "true delay distribution", "delays as they appear in the linelist"
    (data(columns(rows)) * mapping(:lag => xl, :q50 => yl) * lineribbon(bands=BANDS)
     + data(series(rows, :truth, truth; x=:lag)) * mapping(:x => xl, :value => yl; color=:series => "") * visual(Lines; linestyle=:dash)
     + data(series(rows, :naive, naive; x=:lag)) * mapping(:x => xl, :value => yl; color=:series => "") * visual(Scatter; markersize=8)) *
        config(width=620, height=260, scales=scales(Color=(; palette=[ORANGE, INK], categories=[naive, truth])))
end

# posterior bands over days, the true path, optionally the counts (coloured by phase when the rows carry one)
function path_figure(file; ylabel, truth_label, counts=false, log_y=false, plus_one=false, rule=nothing, height=280)
    rows = load(file)
    plus_one && (rows = shifted(rows, counts ? (QUANTILES..., :truth, :observed) : (QUANTILES..., :truth), 1))
    layers = data(columns(rows)) * mapping(:day => "day", :q50 => ylabel) * lineribbon(bands=BANDS) +
             data(series(rows, :truth, truth_label; x=:day)) * mapping(:x => "day", :value => ylabel; color=:series => "") * visual(Lines; linestyle=:dash)
    categories = [truth_label]; palette = [INK]
    if counts
        groups = haskey(first(rows), :phase) ? [("reported cases, fitted", ORANGE, r -> r.phase == "fitted"), ("reported cases, held out", PURPLE, r -> r.phase == "held out")] :
                                               [("reported cases", ORANGE, r -> true)]
        for (label, colour, keep) in groups
            layers += data(series(filter(keep, rows), :observed, label; x=:day)) * mapping(:x => "day", :value => ylabel; color=:series => "") * visual(Scatter; markersize=7)
            push!(categories, label); push!(palette, colour)
        end
    end
    rule === nothing || (layers += data((; x=[rule])) * mapping(:x => "day") * visual(VLines; color=:gray, linestyle=:dot))
    colour = (; palette, categories)
    layers * config(width=620, height=height, scales=log_y ? scales(Color=colour, Y=(; scale=log10)) : scales(Color=colour))
end

FIGURES["single_R"] = () -> path_figure("single_R.json"; ylabel="reproduction number R(t)", truth_label="true R(t)")
FIGURES["single_cases"] = () -> path_figure("single_cases.json"; ylabel="daily cases (log axis)", truth_label="true expected cases", counts=true, log_y=true)
FIGURES["prior_cases"] = () -> path_figure("prior_cases.json"; ylabel="daily cases + 1 (log axis)", truth_label="true expected cases", counts=true, log_y=true, plus_one=true)
FIGURES["forecast_R"] = () -> path_figure("forecast_R.json"; ylabel="reproduction number R(t)", truth_label="true R(t)", rule=42.5, height=240)
FIGURES["forecast_cases"] = () -> path_figure("forecast_cases.json"; ylabel="daily cases (log axis)", truth_label="true expected cases", counts=true, log_y=true, rule=42.5)

# what the counts add: prior and posterior bands of R(t) on one axis
FIGURES["prior_R"] = function ()
    both = vcat([merge(r, (; fit="prior")) for r in load("prior_R.json")], [merge(r, (; fit="posterior")) for r in load("single_R.json")])
    yl = "reproduction number R(t)"
    (data(columns(both)) * mapping(:day => "day", :q50 => yl; color=:fit => "") * lineribbon(bands=BANDS)
     + data(series(load("single_R.json"), :truth, "true R(t)"; x=:day)) * mapping(:x => "day", :value => yl) * visual(Lines; color=:black, linestyle=:dash)) *
        config(width=620, height=280, scales=scales(Color=(; palette=["#1b9e77", "#999999"], categories=["posterior", "prior"])))
end

FIGURES["patches_map"] = function ()
    rows = load("patches.json")
    D = (; x=[r.x for r in rows], y=[r.y for r in rows], population=[r.population / 1000 for r in rows],
           patch=["patch $(r.patch)" * (r.origin ? " (outbreak origin)" : "") for r in rows])
    data(D) * mapping(:x => "x (km)", :y => "y (km)"; color=:patch => "", markersize=:population => "population (thousands)") * visual(Scatter) *
        config(width=340, height=300)
end

# one panel per patch, three to a row
function patch_panels(file; ylabel, counts=false)
    rows = [merge(r, (; panel="patch $(r.patch)")) for r in load(file)]
    counts && (rows = shifted(rows, (QUANTILES..., :truth, :observed), 1))
    D = columns(rows)
    layers = data(D) * mapping(:day => "day", :q50 => ylabel; layout=:panel => "") * lineribbon(bands=BANDS) +
             data(D) * mapping(:day => "day", :truth => ylabel; layout=:panel => "") * visual(Lines; color=:black, linestyle=:dash)
    counts && (layers += data((; day=D.day, observed=Float64.(D.observed), panel=D.panel)) *
                         mapping(:day => "day", :observed => ylabel; layout=:panel => "") * visual(Scatter; color=ORANGE, markersize=4))
    counts ? layers * config(width=200, height=160, columns=3, scales=scales(Y=(; scale=log10))) : layers * config(width=200, height=160, columns=3)
end
FIGURES["patch_R"] = () -> patch_panels("patch_R.json"; ylabel="R(g, t)")
FIGURES["patch_cases"] = () -> patch_panels("patch_cases.json"; ylabel="daily cases + 1 (log axis)", counts=true)

# open circle and bars: posterior median with 50 % and 95 % intervals; grey dot: prior mean; black dot: truth
FIGURES["patch_seeds"] = function ()
    rows = [merge(r, (; label="patch $(r.patch)")) for r in load("patch_seeds.json")]
    D = columns(rows)
    xl = "log I0 (log initial infections)"
    (data(D) * mapping(:q50 => xl, y=:label => "") * pointinterval(bands=[:q025 => :q975, :q25 => :q75])
     + data(D) * mapping(:prior_mean => xl, y=:label => "") * visual(Scatter; color="#999999", markersize=10)
     + data(D) * mapping(:truth => xl, y=:label => "") * visual(Scatter; color=:black, markersize=10)) *
        config(width=560, height=220)
end

main(names) = foreach(name -> render(name, FIGURES[name]()), isempty(names) ? sort!(collect(keys(FIGURES))) : names)

main(ARGS)
