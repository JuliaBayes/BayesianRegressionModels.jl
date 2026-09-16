# Validation tables for one case: CASE MATRIX_DIR SUMMARY_DIR OUT
#  recovery_seed_sensitivity.tsv : totals arms re-recovered with ten seeds (only population_mean changes)
#  per_quantity_mcse.tsv         : each arm's posterior means against rbest_ncp_native in combined-MCSE units
isdefined(@__MODULE__, :RBesTCentering) || include(joinpath(@__DIR__, "model.jl"))
using .RBesTCentering, Serialization, Statistics, Random, DelimitedFiles, MCMCDiagnosticTools
const RC = RBesTCentering
include(joinpath(@__DIR__, "run.jl"))

function read_qois(path)
    table, header = readdlm(path, '\t'; header=true)
    cols = Dict(String(h) => table[:, i] for (i, h) in enumerate(vec(header)))
    (; parameter=String.(cols["parameter"]), mean=Float64.(cols["mean"]), mcse=Float64.(cols["mcse"]), bulk_ess=Float64.(cols["bulk_ess"]))
end

function recovery_sensitivity(c, t, matrix, out)
    rows = NamedTuple[]
    for arm in ("total_ncp", "total_cp", "total_posthoc_position", "total_posthoc_gradient", "total_online_position", "total_online_gradient")
        fit = deserialize(joinpath(matrix, arm * ".jls")); draws = permutedims(fit.positions)
        for seed in 1:10
            recovered = RC.BRM.recover_population_draws(t.sb, draws, t.names; rng=Xoshiro(1000 + seed))[c.lp]
            qois = hcat(recovered.population[:, 1], exp.(draws[:, only(t.coords.scales)]), draws[:, vec(t.coords.totals)])
            ess = vec(MCMCDiagnosticTools.ess(reshape(qois, N_DRAWS, 1, size(qois, 2)); kind=:bulk))
            push!(rows, (; arm, seed, min_bulk_ess=minimum(ess), limiting_qoi=qoi_names(c)[argmin(ess)], population_mean_ess=ess[1],
                population_mean=mean(qois[:, 1])))
        end
    end
    write_tsv(joinpath(out, "recovery_seed_sensitivity.tsv"), rows)
end

function mcse_checks(c, matrix, summary, out)
    files = Dict{String,String}()
    for dir in (matrix, summary), p in readdir(dir; join=true)
        endswith(p, "-qois.tsv") && (files[replace(basename(p), "-qois.tsv" => "")] = p)
    end
    base = read_qois(files["rbest_ncp_native"]); rows = NamedTuple[]; worst = NamedTuple[]
    for (arm, path) in sort(collect(files))
        q = read_qois(path); @assert q.parameter == base.parameter
        z = abs.(q.mean .- base.mean) ./ sqrt.(q.mcse .^ 2 .+ base.mcse .^ 2)
        for k in eachindex(z)
            push!(rows, (; arm, parameter=q.parameter[k], mean=q.mean[k], baseline_mean=base.mean[k], mcse=q.mcse[k], baseline_mcse=base.mcse[k], z=z[k]))
        end
        push!(worst, (; arm, max_z=maximum(z), worst_parameter=q.parameter[argmax(z)], n_over_3=count(>(3), z)))
    end
    write_tsv(joinpath(out, "per_quantity_mcse.tsv"), rows); write_tsv(joinpath(out, "mcse_summary.tsv"), worst)
end

if abspath(PROGRAM_FILE) == @__FILE__
    case, matrix, summary, out = ARGS
    c = RC.load_case(case); mkpath(out)
    t = RC.total_model(c, joinpath(matrix, "model"))
    recovery_sensitivity(c, t, matrix, out); mcse_checks(c, matrix, summary, out)
    println("RBEST_COMPLETE ", case)
end
