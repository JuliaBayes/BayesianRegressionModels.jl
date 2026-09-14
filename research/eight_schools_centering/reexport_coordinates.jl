include("/home/n/.local/state/kb-agents-worktrees/BayesianRegressionModels-docs-eight-schools/research/eight_schools_centering/reproduce.jl")
using Serialization

"""Re-export physical-effect coordinate tables from saved fits. No sampling."""
function reexport_coordinates(fit_dir)
    stan = stan_density("reexport", fit_dir)
    read(joinpath(fit_dir, "eight-schools-reexport.stan")) ==
        read(joinpath(fit_dir, "eight-schools-model.stan")) ||
        error("re-export target differs from the saved full-fit producer")
    sel_lines = split.(readlines(joinpath(fit_dir, "offline_centeredness.tsv"))[2:end], '\t')
    selected = [parse(Float64, row[2]) for row in sel_lines]
    length(selected) == 8 || error("expected eight offline selections")
    pilot = deserialize(joinpath(fit_dir, "noncentered.jls"))
    partial_target = deserialize(joinpath(fit_dir, "partial_target.jls"))
    online = deserialize(joinpath(fit_dir, "online.jls"))
    export_coordinates("noncentered", stan, pilot, fit_dir)
    export_coordinates("partial", stan, partial_target, fit_dir; controls=selected)
    export_coordinates("online", stan, online, fit_dir)
    println("eight_schools_coordinates_reexported\t", fit_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: reexport_coordinates.jl FULL_FIT_DIRECTORY")
    reexport_coordinates(only(ARGS))
end
