# Scientific-QOI diagnostics for the native RBesT arms: CASE NATIVE_DIR OUT
# Reads each arm's sampling.tsv (post-warmup rows) and gradient_counts.tsv and writes the same
# <arm>-qois.tsv / <arm>-summary.tsv records run.jl writes for the WarmupHMC arms.
isdefined(@__MODULE__, :RBesTCentering) || include(joinpath(@__DIR__, "model.jl"))
using .RBesTCentering, DelimitedFiles, Statistics, JSON, MCMCDiagnosticTools
const RC = RBesTCentering
include(joinpath(@__DIR__, "run.jl"))   # write_tsv, qoi_names, diagnostics, N_DRAWS (no matrix runs: PROGRAM_FILE differs)

function read_tsv(path)
    table, header = readdlm(path, '\t'; header=true)
    Dict(String(h) => table[:, i] for (i, h) in enumerate(vec(header)))
end

function native_arm(c, dir, label, out)
    cols = read_tsv(joinpath(dir, "sampling.tsv")); counts = read_tsv(joinpath(dir, "gradient_counts.tsv"))
    prov = JSON.parsefile(joinpath(dir, "provenance.json"))
    @assert prov["case"] == c.name && prov["draws"] == N_DRAWS && length(cols["lp__"]) == N_DRAWS
    theta = hcat([Float64.(cols["theta.$h"]) for h in 1:c.H]...)
    qois = hcat(Float64.(cols["beta.1"]), Float64.(cols["tau.1"]), theta)
    fit = (; sampling_gradients=Int(only(counts["sampling_gradients"])), all_gradient_calls=Int(only(counts["all_gradient_calls"])),
        pilot_gradient_calls=0, total_gradient_calls=Int(only(counts["workflow_gradient_calls"])),
        divergences=Int(only(counts["divergences"])), fit_seconds=NaN, max_depth_hits=Int(only(counts["max_depth_hits"])))
    diagnostics(c, label, qois, fit, out)
    write_tsv(joinpath(out, label * "-provenance.tsv"), [(; arm=label, variant=prov["variant"], parametrization=prov["parametrization"],
        rbest_version=prov["rbest_version"], rbest_sha=prov["rbest_sha"], adapt_delta=prov["adapt_delta"], step_size=string(prov["step_size"]),
        max_treedepth=prov["max_treedepth"], warmup=prov["warmup"], cmdstan=prov["cmdstan"], max_depth_hits=fit.max_depth_hits)])
end

if abspath(PROGRAM_FILE) == @__FILE__
    case, native, out = ARGS
    c = RC.load_case(case); mkpath(out)
    for arm in ("rbest_ncp", "rbest_cp", "s2z_ncp", "s2z_cp", "stan_ncp", "stan_cp")
        dir = joinpath(native, arm)
        isfile(joinpath(dir, "provenance.json")) || (println("MISSING_NATIVE ", arm); continue)
        native_arm(c, dir, arm * "_native", out)
    end
    println("RBEST_NATIVE_ANALYSIS_COMPLETE ", case)
end
