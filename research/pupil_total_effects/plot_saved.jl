# Uses the existing research/adaptive_centering/plots environment.
using AlgebraOfVega, CairoMakie, CSV, JSON

function plot_saved(output)
    rows = CSV.File(joinpath(output,"pairs.tsv");delim='\t')
    spec = data(rows) * mapping(:log_group_sd=>"Log group SD",:coordinate=>"Coordinate";
        row=:panel=>"Selected coordinate",col=:column=>"Visualization") *
        visual(Scatter;markersize=3,opacity=0.25) *
        config(width=205,height=155,facet=(;linkxaxes=:none,linkyaxes=:none))
    vl = to_vegalite(spec;interactive=false)
    open(io->JSON.print(io,vl),joinpath(output,"pairs.aov.json"),"w")
    sdraw_file(spec,joinpath(output,"pairs.png");px_per_unit=1.5)
    println("PLOT_COMPLETE\t",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    plot_saved(only(ARGS))
end
