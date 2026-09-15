using AlgebraOfVega, CairoMakie, CSV, JSON
import AlgebraOfGraphics
out=ARGS[1]
case=ARGS[2]
subtitle=case=="cluster-independent" ? "Independent regional intercepts and slopes · 17 scientific quantities" : "Regional intercepts · 10 scientific quantities"
rows=CSV.File(joinpath(out,"efficiency_plot.tsv");delim='\t')
order=unique(String.(rows.model))
efficiency=data(rows)*mapping(:value=>"",:model=>sorter(order)=>"";
    color=:sampler=>"Sampler",dodge_y=:sampler,col=:metric=>"")*
    visual(Scatter;markersize=10)*
    config(width=240,height=520,facet=(;linkxaxes=:none),
        scales=scales(X=(;scale=log10),DodgeY=(;width=.6),
            Color=(;categories=["Native Stan","WHMC"],palette=["#D55E00","#0072B2"])))
open(io->JSON.print(io,to_vegalite(efficiency;interactive=false)),joinpath(out,"efficiency.aov.json"),"w")
figure=Figure(size=(1450,680))
Label(figure[0,:],"Air pollution: " * (case=="cluster-independent" ? "regional intercepts and slopes" : "regional intercepts");tellwidth=false,fontsize=24)
grid=sdraw!(figure[1,1],efficiency)
AlgebraOfGraphics.legend!(figure[1,2],grid)
Label(figure[2,:],"Relative to brms NCP + native Stan · " * subtitle * " · one chain per arm";tellwidth=false,fontsize=16)
save(joinpath(out,"efficiency.png"),figure;px_per_unit=1.5)
for kind in ("total","s2z")
    path=joinpath(out,kind*"_pairs.tsv")
    isfile(path) || continue
    pairs=CSV.File(path;delim='\t')
    spec=data(pairs)*mapping(:log_group_sd=>"Log group SD",:coordinate=>"Coordinate";
        row=:panel=>"Selected coordinate",col=:column=>"Visualization")*
        visual(Scatter;markersize=3,opacity=.25)*
        config(width=270,height=270,facet=(;linkxaxes=:none,linkyaxes=:none))
    open(io->JSON.print(io,to_vegalite(spec;interactive=false)),joinpath(out,kind*"_pairs.aov.json"),"w")
    pairfigure=Figure(size=(1200,1100))
    sdraw!(pairfigure[1,1],spec)
    save(joinpath(out,kind*"_pairs.png"),pairfigure;px_per_unit=1.5)
end
println("AIR_PLOTS_COMPLETE")
