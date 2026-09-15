using BayesianRegressionModels, AlgebraOfVega, CairoMakie, CSV, Tables, JSON, Statistics
import AlgebraOfGraphics
include("../adaptive_centering/plots/scatter_display.jl")
const CASE, INPUT, OUTPUT = ARGS
mkpath(OUTPUT)
rows(name)=collect(Tables.namedtupleiterator(CSV.File(joinpath(INPUT,name);delim='\t',stringtype=String)))
slug(x)=replace(lowercase(x),r"[^a-z0-9]+"=>"-")

function save_panel(name,spec;title,size=(1200,800),legend=false,zoom=false)
    fig=Figure(;size,fontsize=16,figure_padding=(10,55,10,10))
    Label(fig[0,1:(legend ? 2 : 1)],title;fontsize=22,font=:bold,tellwidth=false)
    grid=sdraw!(fig[1,1],spec)
    legend && AlgebraOfGraphics.legend!(fig[1,2],grid)
    if zoom
        receipt=zoom_scatter_axes!(grid;mass=.975)
        CSV.write(joinpath(OUTPUT,name*"-display.tsv"),receipt;delim='\t')
    end
    save(joinpath(OUTPUT,name*".png"),fig;px_per_unit=1.5)
    open(io->JSON.print(io,to_vegalite(spec;interactive=false)),joinpath(OUTPUT,name*".aov.json"),"w")
    println("REFRESH_FIGURE ",name)
end

all_pairs=rows("pairs.tsv"); all_gradients=rows("gradients.tsv")
for group in unique(r.group for r in all_pairs)
    for (family,configs) in (
        ("pilot",["CP reference","NCP reference"]),
        ("posthoc",["CP reference","Post-hoc position","Post-hoc gradient"]),
        ("online",["CP reference","Online position","Online gradient"]))
        chosen=[(;r...,panel=string(findfirst(==(r.configuration),configs)," ",r.configuration),
            cell=CASE=="hsgp" ? r.cell*(r.hyper=="Length scale" ? " · ρ" : " · σ") : r.cell)
            for r in all_pairs if r.group==group && r.configuration in configs]
        height=CASE=="radon" ? 850 : 1500
        spec=data(chosen)*mapping(:hyperparameter=>(CASE=="hsgp" ? "GP hyperparameter" : "Group SD"),
            :coordinate=>"Effect coordinate";col=:panel,row=:cell)*
            visual(Scatter;color="#19679a",opacity=.12,markersize=3)*
            config(facet=(;linkxaxes=:none,linkyaxes=:none),scales=scales(X=(;scale=log10)))
        save_panel("pairs-$family-$(slug(group))",spec;title="$group: $family geometry",size=(1200,height),zoom=CASE=="hsgp")
        family=="pilot" && continue
        g=[(;r...,panel=string(findfirst(==(r.configuration),configs)," ",r.configuration))
            for r in all_gradients if r.group==group && r.configuration in configs]
        spec=data(g)*mapping(:coordinate=>"Effect coordinate",:gradient=>"Log-density gradient";col=:panel,row=:cell)*
            visual(Scatter;color="#19679a",opacity=.25,markersize=4)*config(facet=(;linkxaxes=:none,linkyaxes=:none))
        save_panel("gradients-$family-$(slug(group))",spec;title="$group: displayed position and gradient",size=(1200,CASE=="eight" ? 1500 : 850),zoom=CASE=="hsgp")
    end
end

controls=rows("controls.tsv")
spec=data(controls)*mapping(:id=>(CASE=="hsgp" ? "Basis frequency" : CASE=="radon" ? "County" : "School"),
    :centeredness=>"Centeredness";col=:group,color=:arm=>"Selection")*visual(Scatter;markersize=7)*
    config(axis=(;limits=(nothing,(0,1))))
save_panel("centeredness",spec;title="Both losses, post-hoc and online",size=(1350,500),legend=true)

function intervals(table;xlabel="Original data row",ylabel="Observed response",facets=(;))
    table=[(;r...,lo90=r.q50-r.q05,hi90=r.q95-r.q50,lo50=r.q50-r.q25,hi50=r.q75-r.q50) for r in table]
    base=data(table)
    base*mapping(:index=>xlabel,:q50=>ylabel,:lo90,:hi90;facets...)*visual(Errorbars;color="#aac9df",linewidth=1,whiskerwidth=0)+
    base*mapping(:index=>xlabel,:q50=>ylabel,:lo50,:hi50;facets...)*visual(Errorbars;color="#5789af",linewidth=3,whiskerwidth=0)
end

if CASE=="hsgp"
    curves=rows("curves.tsv"); observed=rows("observations.tsv")
    for arm in ("NCP","Post-hoc position","Post-hoc gradient","Online position","Online gradient")
        any(r->r.arm==arm,curves) || continue
        fig=Figure(size=(1400,480),fontsize=17)
        Label(fig[0,1:2],arm*": posterior functions";fontsize=22,font=:bold)
        for (i,predictor) in enumerate(("mu","log_sigma"))
            r=sort([r for r in curves if r.arm==arm && r.predictor==predictor];by=r->r.time)
            spec=brm_posteriorplot(r;xlabel="Time after impact (ms)",
                ylabel=predictor=="mu" ? "Acceleration (scaled)" : "Conditional SD (scaled)",
                observations=predictor=="mu" ? observed : nothing,observed_y=:acceleration_scaled,logscale=predictor!="mu")
            sdraw!(fig[1,i],spec)
        end
        save(joinpath(OUTPUT,"posterior-"*slug(arm)*".png"),fig;px_per_unit=1.5)
    end
else
    ppc=rows("ppc.tsv")
    getproperty.(ppc,:index)==collect(1:length(ppc)) || error("Observation order changed")
    if CASE=="radon"
        counts=Dict(g=>count(r->r.group==g,ppc) for g in unique(r.group for r in ppc))
        eligible=[g for g in keys(counts) if count(r->r.group==g && r.category=="Floor code 0",ppc)>=5 && count(r->r.group==g && r.category=="Floor code 1",ppc)>=5]
        sort!(eligible;by=g->(counts[g],parse(Int,last(split(g)))))
        selected=unique([first(eligible),eligible[cld(length(eligible),2)],last(eligible)])
        ppc=[r for r in ppc if r.group in selected]
        spec=intervals(ppc;ylabel="Log radon",facets=(;row=:group))+data(ppc)*
            mapping(:index=>"Original data row",:observation=>"Log radon";row=:group,color=:category=>"Floor code")*visual(Scatter;markersize=6)
        save_panel("ppc",spec*config(facet=(;linkxaxes=:none,linkyaxes=:all));title="Predictive intervals and observations by county",size=(1300,1000),legend=true)
    else
        spec=intervals(ppc;xlabel="School",ylabel="Reported estimate")+data(ppc)*mapping(:index=>"School",:observation=>"Reported estimate")*visual(Scatter;color="#bd3d2a",markersize=10)
        save_panel("ppc",spec;title="Posterior predictive intervals and observed estimates",size=(1050,500))
        effects=NamedTuple[]
        for arm in ("ncp","cp")
            r=rows(arm*"_coordinates.tsv")
            for j in 1:8
                q=quantile([x.theta_effect for x in r if x.school==j],[.05,.25,.5,.75,.95])
                push!(effects,(;index=j,arm=uppercase(arm),q05=q[1],q25=q[2],q50=q[3],q75=q[4],q95=q[5]))
            end
        end
        spec=intervals(effects;xlabel="School",ylabel="Treatment effect",facets=(;col=:arm))+
            data(effects)*mapping(:index=>"School",:q50=>"Treatment effect";col=:arm)*visual(Scatter;color="#19679a",markersize=8)
        save_panel("posterior-effects",spec;title="School treatment effects: NCP and CP fits",size=(1100,500))
    end
end
efficiency=collect(Tables.namedtupleiterator(CSV.File(joinpath(@__DIR__,"results",CASE,"efficiency.tsv");delim='\t')))
order=["NCP","CP","Post-hoc position","Post-hoc gradient","Online position","Online gradient"]
metriclabels=Dict("Total gradients"=>"1. Total gradients ↓",
    "Relative sampling efficiency"=>"2. Sampling efficiency ↑", "Relative total efficiency"=>"3. Total efficiency ↑")
efficiency=[(;r...,metric=metriclabels[r.metric]) for r in efficiency]
spec=data(efficiency)*mapping(:value=>"",:method=>"";col=:metric)*visual(Scatter;markersize=13,color="#19679a")*
    config(facet=(;linkxaxes=:none),scales=scales(X=(;scale=log10),Y=(;categories=reverse(order))))
save_panel("efficiency",spec;title="Gradient cost and efficiency relative to NCP",size=(1350,450))
println("REFRESH_PLOTS_COMPLETE ",CASE)
