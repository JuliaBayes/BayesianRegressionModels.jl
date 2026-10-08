# Check that the BRM model in `reproduce.jl` is the source program
# `source/unconstrained_monster.stan` with `source/parallel_incremental_data.json`:
# at random unconstrained points the two normalized log densities (with
# Jacobians) must differ by one constant, and their gradients must agree under
# the affine map between the two parameterizations.
#
#     julia --project=<env with BRM, StanBlocks, BridgeStan> research/monster/compare_to_source.jl
include(joinpath(@__DIR__, "reproduce.jl"))
import BridgeStan as BS
import JSON
using Printf, Random

const SOURCE = joinpath(@__DIR__, "source")

function monster_source_comparison(; npoints=25, seed=20261008,
                                   build=mktempdir())
    source_stan = joinpath(build, "unconstrained_monster.stan")
    cp(joinpath(SOURCE, "unconstrained_monster.stan"), source_stan; force=true)
    source = BS.StanModel(BS.compile_model(source_stan),
                          joinpath(SOURCE, "parallel_incremental_data.json"))
    brm = StanBlocks.stan_instantiate(monster_sbbrmi().model;
        path=joinpath(build, "monster_brm.stan")).model

    on, bn = BS.param_unc_names(source), BS.param_unc_names(brm)
    quantities = ("VPR", "Fwp", "Fpp", "Ff", "Fl", "Vwp", "Vpp", "Vl",
                  "Pba", "Pwp", "Ppp", "Pf", "Pl", "VMI", "KMI")
    data = JSON.parsefile(joinpath(SOURCE, "parallel_incremental_data.json");
                          allownan=true)
    eM_eM, eM_eS, eS_mu = data["population_eM_eM"], data["population_eM_eS"],
                          data["population_eS_mu"]
    function index(names, name)
        k = findfirst(==(name), names)
        isnothing(k) && error("no unconstrained coordinate $name")
        k
    end
    # BridgeStan lists an array-of-vectors block first-index-fastest while its
    # VALUES are array-major: person i, quantity j sits at (i - 1) * 15 + j.
    person0 = findfirst(startswith("unit_log_person_params."), on) - 1
    # brm[b] = shift + scale * source[s]
    rows = NamedTuple{(:b, :s, :scale, :shift),Tuple{Int,Int,Float64,Float64}}[]
    for (j, q) in enumerate(quantities)
        push!(rows, (; b=index(bn, "pop_log_$(q)_beta_pop.1"),
                     s=index(on, "unit_log_population_eM.$j"),
                     scale=log(eM_eS[j]), shift=log(eM_eM[j])))
        push!(rows, (; b=index(bn, "b_$(q)_subject_tau.1"),
                     s=index(on, "unit_log_population_eS.$j"),
                     scale=1.0, shift=log(log(eS_mu[j]))))
        for i in 1:Int(data["no_persons"])
            push!(rows, (; b=index(bn, "b_$(q)_subject_z_flat.$i"),
                         s=person0 + (i - 1) * length(quantities) + j,
                         scale=1.0, shift=0.0))
        end
    end
    push!(rows, (; b=index(bn, "log_sigma_venous"), s=index(on, "noise.1"),
                 scale=1.0, shift=0.0))
    push!(rows, (; b=index(bn, "log_sigma_exhaled"), s=index(on, "noise.2"),
                 scale=1.0, shift=0.0))
    sort(getfield.(rows, :b)) == eachindex(bn) || error("incomplete BRM map")
    sort(getfield.(rows, :s)) == eachindex(on) || error("incomplete source map")

    rng = Xoshiro(seed)
    offsets, gradient_errors = Float64[], Float64[]
    for _ in 1:npoints
        theta = 0.7 .* randn(rng, length(on))
        phi = zeros(length(bn))
        foreach(r -> phi[r.b] = r.shift + r.scale * theta[r.s], rows)
        lp_source, g_source = BS.log_density_gradient(source, theta;
            propto=false, jacobian=true)
        lp_brm, g_brm = BS.log_density_gradient(brm, phi;
            propto=false, jacobian=true)
        g_mapped = zeros(length(on))
        foreach(r -> g_mapped[r.s] = r.scale * g_brm[r.b], rows)
        push!(offsets, lp_source - lp_brm)
        push!(gradient_errors,
              maximum(abs.(g_source .- g_mapped) ./ (1 .+ abs.(g_source))))
    end
    # The constant is the Jacobian of the affine reparameterization: BRM's
    # Normal(log(eM_eM), log(eM_eS)) density of each population mean carries
    # the factor 1 / log(eM_eS) that the source's unit-scale density lacks, and
    # the source's density of tau omits the factor 2 of tau^2 -> tau.
    expected = sum(log.(log.(eM_eS))) - length(quantities) * log(2)

    timing(model, x) = (BS.log_density_gradient(model, x);
                        minimum(@elapsed(BS.log_density_gradient(model, x))
                                for _ in 1:50))
    theta = zeros(length(on)); phi = zeros(length(bn))
    foreach(r -> phi[r.b] = r.shift + r.scale * theta[r.s], rows)
    (; offset=first(offsets), expected_offset=expected,
       offset_spread=maximum(offsets) - minimum(offsets),
       max_relative_gradient_error=maximum(gradient_errors),
       source_gradient_seconds=timing(source, theta),
       brm_gradient_seconds=timing(brm, phi),
       source_dimension=length(on), brm_dimension=length(bn))
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = monster_source_comparison()
    foreach(k -> @printf("%-28s %s\n", k, getfield(result, k)), keys(result))
    result.offset_spread < 1e-8 || error("log densities differ by more than a constant")
    abs(result.offset - result.expected_offset) < 1e-8 || error("unexpected offset")
    result.max_relative_gradient_error < 1e-8 || error("gradients disagree")
    println("OK: the BRM model reproduces source/unconstrained_monster.stan")
end
