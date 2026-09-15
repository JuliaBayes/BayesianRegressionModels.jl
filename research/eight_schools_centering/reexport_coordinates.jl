include(joinpath(@__DIR__, "reproduce.jl"))
using Serialization

"""Re-export physical-effect coordinate tables from saved fits. No sampling."""
function reexport_coordinates(fit_dir)
    stan = stan_density("reexport", fit_dir)
    read(joinpath(fit_dir, "eight-schools-reexport.stan")) ==
        read(joinpath(fit_dir, "eight-schools-model.stan")) ||
        error("re-export target differs from the saved full-fit producer")
    for label in ("noncentered", "centered", "partial", "online")
        fit = deserialize(joinpath(fit_dir, "$label.jls"))
        export_coordinates(label, stan, fit, fit_dir)
    end
    println("eight_schools_coordinates_reexported\t", fit_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: reexport_coordinates.jl FULL_FIT_DIRECTORY")
    reexport_coordinates(only(ARGS))
end
