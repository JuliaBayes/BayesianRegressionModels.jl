# Uses research/adaptive_centering/plots; inputs are saved-fit summaries only.
using AlgebraOfVega, CairoMakie, CSV, JSON
import AlgebraOfGraphics

function save_plot(spec,output,name)
    vl=to_vegalite(spec;interactive=false)
    open(io->JSON.print(io,vl),joinpath(output,name*".aov.json"),"w")
    if name=="efficiency"
        figure=Figure(size=(1150,480))
        grid=sdraw!(figure[1,1],spec)
        AlgebraOfGraphics.legend!(figure[1,2],grid)
        save(joinpath(output,name*".png"),figure;px_per_unit=1.5)
    else
        sdraw_file(spec,joinpath(output,name*".png");px_per_unit=1.5)
    end
end

function plot_comparison(output)
    rows=CSV.File(joinpath(output,"efficiency_plot.tsv");delim='\t')
    spec=data(rows)*mapping(:value=>"Log scale",:model=>"";
        color=:sampler=>"Sampler",dodge_y=:sampler,col=:metric=>"")*
        visual(Scatter;markersize=10)*
        config(width=225,height=340,facet=(;linkxaxes=:none),
               scales=scales(X=(;scale=log10),DodgeY=(;width=0.6),
                   Color=(;categories=["Native Stan","WHMC"],palette=["#D55E00","#0072B2"])))
    save_plot(spec,output,"efficiency")
    pairs=CSV.File(joinpath(output,"s2z_pairs.tsv");delim='\t')
    pair_spec=data(pairs)*mapping(:log_group_sd=>"Log group SD",:coordinate=>"Subject contrast coordinate";
        row=:panel=>"Selected coordinate",col=:column=>"Visualization")*
        visual(Scatter;markersize=3,opacity=0.25)*
        config(width=205,height=155,facet=(;linkxaxes=:none,linkyaxes=:none))
    save_plot(pair_spec,output,"s2z_pairs")
    println("COMPARISON_PLOTS_COMPLETE\t",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    plot_comparison(only(ARGS))
end
