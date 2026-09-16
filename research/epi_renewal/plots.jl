# Wren.jl PR #16 in native @brm — the figures, from the saved summaries that wren16.jl writes.
#
# Inputs are research/epi_renewal/results/*.json only (quantile rows: q025 q10 q25 q50 q75 q90 q975
# + truth + keys). Every figure is an AlgebraOfVega spec built with AlgebraOfGraphics' algebra and
# pre-aggregated bands (`lineribbon(bands=...)`, `pointinterval(bands=...)`), lowered with
# `to_vegalite(spec; interactive=false)` and written as results/<name>.aov.json for the brief's
# `kb-aov/v1` fences (no transforms, no params — the envelope forbids them). A static PNG is also
# written when CairoMakie is loadable, for inspection only.
#
# Run in an environment that resolves AlgebraOfVega (it is unregistered), e.g. BRM's web-macro env:
#   julia --project=/home/n/github/nsiccha/BayesianRegressionModels.jl/web-macro research/epi_renewal/plots.jl

using AlgebraOfVega, AlgebraOfGraphics, JSON, Statistics
import AlgebraOfGraphics as AoG

const RESULTS = joinpath(@__DIR__, "results")
load(name) = [(; (Symbol(k) => v for (k, v) in row)...) for row in JSON.parsefile(joinpath(RESULTS, name))]
columns(rows) = (; (k => [r[k] for r in rows] for k in keys(first(rows)))...)
const BANDS = [:q025 => :q975, :q10 => :q90, :q25 => :q75]        # 95 / 80 / 50 % central intervals

const HAS_CAIRO = try
    @eval using CairoMakie
    true
catch
    false
end

function emit(name, spec; width=640, height=300)
    vl = to_vegalite(spec; interactive=false)
    open(io -> JSON.print(io, vl), joinpath(RESULTS, name * ".aov.json"), "w")
    if HAS_CAIRO
        try
            sdraw_file(spec, joinpath(RESULTS, name * ".png"); px_per_unit=1.5)
        catch e
            println("  (static render of ", name, " skipped: ", first(split(sprint(showerror, e), '\n')), ")")
        end
    end
    println("wrote ", name, ".aov.json (", length(JSON.json(vl)), " bytes)")
    vl
end

function main()
    # 1. single patch: R_t posterior bands vs truth
    R = columns(load("single_R.json"))
    spec_R = (data(R) * mapping(:day => "day", :q50 => "reproduction number R_t") * lineribbon(bands=BANDS)
              + data(R) * mapping(:day => "day", :truth => "reproduction number R_t") * visual(Lines; color=:black, linestyle=[6, 4])) *
             config(width=640, height=280, title="Single patch: R_t recovered from 56 days of cases (50/80/95 % bands; dashed = truth)")
    emit("single_R", spec_R)

    # 2. forecast: fit through day 42, posterior predictive cases through day 56, observed points
    FC = columns(load("forecast_cases.json")); FY = columns(load("forecast_Y.json"))
    obs = (; day=FY.day, observed=Float64.(FY.observed), phase=FY.phase)
    spec_F = (data(FC) * mapping(:day => "day", :q50 => "daily cases") * lineribbon(bands=BANDS)
              + data(FY) * mapping(:day => "day", :truth => "daily cases") * visual(Lines; color=:black, linestyle=[6, 4])
              + data(obs) * mapping(:day => "day", :observed => "daily cases"; color=:phase => "observation") * visual(Scatter; markersize=7)
              + data((; x=[42.5])) * mapping(:x => "day") * visual(VLines; color=:gray, linestyle=[2, 3])) *
             config(width=640, height=300, title="Forecast: fitted on days 1–42, the random walk's prior tail carries days 43–56 (posterior predictive bands; dashed = true expected cases)")
    emit("forecast", spec_F)

    # (`opacity=` is the AoV mark spelling — `alpha` passes through untranslated; Makie linestyle symbols lower to
    #  `strokeDash: "dash"`, not a VL dash array — AlgebraOfVega snag filed 2026-09-16 — so dash arrays are explicit.)
    # 3./4. six patches — NO facets: AoV's facet lowering splits every colour group into filtered sub-layers
    #      (`transform: [{filter: ...}]`), which the kb-aov/v1 envelope rejects (AlgebraOfVega todo 1fmmgdb).
    #      One panel, one colour per patch, explicit per-patch layers with their own data (no colour MAPPING,
    #      hence no filters): 95 % band + median + dashed truth for R; median, dashed truth, points for cases.
    PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    PRr = load("patch_R.json")
    layers_R = nothing
    for g in sort(unique(r.patch for r in PRr))
        rows = columns(filter(r -> r.patch == g, PRr)); c = PALETTE[g]
        band = data(rows) * mapping(:day => "day", :q025 => "R_{g,t}", :q975 => "R_{g,t}") * visual(Band; color=c, opacity=0.18)
        med  = data(rows) * mapping(:day => "day", :q50 => "R_{g,t}") * visual(Lines; color=c, linewidth=2)
        tru  = data(rows) * mapping(:day => "day", :truth => "R_{g,t}") * visual(Lines; color=c, linestyle=[6, 4])
        layers_R = layers_R === nothing ? band + med + tru : layers_R + band + med + tru
    end
    spec_PR = layers_R * config(width=640, height=320,
        title="Six coupled patches: R_{g,t} per patch (colour = patch 1–6): posterior median with 95 % band, dashed = truth")
    emit("patch_R", spec_PR)

    PYr = load("patch_Y.json")
    layers_Y = nothing
    for g in sort(unique(r.patch for r in PYr))
        rows = columns(filter(r -> r.patch == g, PYr)); c = PALETTE[g]
        obs  = (; day=rows.day, observed=Float64.(rows.observed))
        med  = data(rows) * mapping(:day => "day", :q50 => "daily cases") * visual(Lines; color=c, linewidth=2)
        tru  = data(rows) * mapping(:day => "day", :truth => "daily cases") * visual(Lines; color=c, linestyle=[6, 4])
        pts  = data(obs) * mapping(:day => "day", :observed => "daily cases") * visual(Scatter; color=c, markersize=5)
        layers_Y = layers_Y === nothing ? med + tru + pts : layers_Y + med + tru + pts
    end
    spec_PY = layers_Y * config(width=640, height=320, scales=scales(Y=(; scale=symlog)),
        title="Six coupled patches: expected daily cases per patch (posterior median, solid; truth, dashed) and the simulated counts (points); symlog axis")
    emit("patch_cases", spec_PY)

    # 5. six patches: the PR's returned quantities recovered — K_mix, delta, log_I0 (median + 95 % interval vs truth)
    K = load("patch_K.json"); D = load("patch_delta.json"); Sd = load("patch_seeds.json")
    recov = vcat([(; kind="K_mix (36 mixing weights)", truth=r.truth, q50=r.q50, q025=r.q025, q975=r.q975, id=r.element) for r in K],
                 [(; kind="delta (48 weekly deviations)", truth=r.truth, q50=r.q50, q025=r.q025, q975=r.q975, id=r.element) for r in D],
                 [(; kind="log I0 (6 seeds)", truth=r.truth, q50=r.q50, q025=r.q025, q975=r.q975, id=r.element) for r in Sd])
    seg = vcat([(; kind=r.kind, id=r.id, truth=r.truth, y=r.q025) for r in recov], [(; kind=r.kind, id=r.id, truth=r.truth, y=r.q975) for r in recov])
    ident = vcat([(; kind=k, x=lo, y=lo) for (k, lo, hi) in ((k, minimum(r.truth for r in recov if r.kind == k), maximum(r.truth for r in recov if r.kind == k)) for k in unique(r.kind for r in recov))],
                 [(; kind=k, x=hi, y=hi) for (k, lo, hi) in ((k, minimum(r.truth for r in recov if r.kind == k), maximum(r.truth for r in recov if r.kind == k)) for k in unique(r.kind for r in recov))])
    # (non-faceted, one common axis: posterior minus truth, so K weights, deviations and seeds share a scale)
    err = [(; kind=r.kind, idx=i, err=r.q50 - r.truth, lo=r.q025 - r.truth, hi=r.q975 - r.truth) for (i, r) in enumerate(recov)]
    segs = vcat([(; kind=r.kind, idx=r.idx, y=r.lo) for r in err], [(; kind=r.kind, idx=r.idx, y=r.hi) for r in err])
    spec_K = (data(columns(segs)) * mapping(:idx => "element (36 K weights, 48 weekly deviations, 6 seeds)", :y => "posterior − truth"; group=:idx, color=:kind => "quantity") * visual(Lines; linewidth=1.5)
              + data(columns(err)) * mapping(:idx => "element (36 K weights, 48 weekly deviations, 6 seeds)", :err => "posterior − truth"; color=:kind => "quantity") * visual(Scatter; markersize=6)
              + data((; x=[0, length(err) + 1], y=[0.0, 0.0])) * mapping(:x => "element (36 K weights, 48 weekly deviations, 6 seeds)", :y => "posterior − truth") * visual(Lines; color=:black, linestyle=[6, 4])) *
             config(width=640, height=300, title="Six coupled patches: the quantities the PR returns, recovered — posterior median and 95 % interval minus the truth (dashed = 0)")
    emit("patch_recovery", spec_K)

    # 6. delay model: the daily reporting-delay PMF — truth, posterior bands over draws, at the posterior means
    Dp = columns(load("delay_pmf.json"))
    spec_D = (data(Dp) * mapping(:lag => "reporting delay (days)", :q50 => "daily mass") * lineribbon(bands=BANDS)
              + data(Dp) * mapping(:lag => "reporting delay (days)", :truth => "daily mass") * visual(Lines; color=:black, linestyle=[6, 4])
              + data(Dp) * mapping(:lag => "reporting delay (days)", :at_posterior_means => "daily mass") * visual(Scatter; color=:black, markersize=6)) *
             config(width=640, height=260, title="Reporting delay fitted from a right-truncated linelist (250 events, 80 strata): PMF bands over posterior draws, dashed = truth, points = PMF at the posterior means")
    emit("delay_pmf", spec_D)

    # 7. prior predictive (single patch): cases from the prior, symlog y, observed points
    PP = columns(load("single_prior_cases.json"))
    pp_obs = (; day=PP.day, observed=Float64.(PP.truth))
    spec_P = (data(PP) * mapping(:day => "day", :q50 => "daily cases") * lineribbon(bands=BANDS)
              + data(pp_obs) * mapping(:day => "day", :observed => "daily cases") * visual(Scatter; color=:black, markersize=5)) *
             config(width=640, height=280, scales=scales(Y=(; scale=symlog)),
                    title="Prior predictive cases (held_out=:all): 50/80/95 % bands over prior draws, points = the simulated series; symlog axis")
    emit("prior_predictive", spec_P)

    # 8. scalar parameters: posterior median + 50/95 % intervals vs truth, per parameter and model
    Sc = load("scalars.json")
    sc = columns([(; label=string(r.model, ": ", r.name), model=r.model, name=r.name, truth=r.truth, q50=r.q50, q025=r.q025, q25=r.q25, q75=r.q75, q975=r.q975) for r in Sc])
    spec_S = (data(sc) * mapping(:q50 => "value", y=:label => "parameter"; color=:model => "fit") * pointinterval(bands=[:q025 => :q975, :q25 => :q75])
              + data(sc) * mapping(:truth => "value", y=:label => "parameter") * visual(Scatter; marker=:cross, color=:black, markersize=9)) *
             config(width=640, height=420, title="Scalar parameters: posterior median with 50/95 % intervals; × = truth")
    emit("scalars", spec_S)
    println("done")
end

main()
