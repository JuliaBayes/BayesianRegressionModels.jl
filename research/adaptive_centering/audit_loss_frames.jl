include(joinpath(@__DIR__, "reproduce.jl"))

function audit_loss_frames(offline, online, output)
    BLAS.set_num_threads(1)
    mkpath(output)
    cmrows = split.(readlines(joinpath(offline, "centeredness.tsv"))[2:end], '\t')
    cm, cs = parse.(Float64, getindex.(cmrows, 2)), parse.(Float64, getindex.(cmrows, 3))
    ocrows = split.(readlines(joinpath(online, "online_centeredness.tsv"))[2:end], '\t')
    oc = Dict(p => [parse(Float64, r[3]) for r in ocrows if r[1] == p] for p in ("mu", "log_sigma"))
    fit = deserialize(joinpath(offline, "noncentered.jls"))
    @assert fit.complete && size(fit.posterior_position) == (44, 10000)
    base = stan_density(build_brmi(prepared_data()), "same-draw-ncp", output)
    read(joinpath(output, "motorcycle-same-draw-ncp.stan")) ==
        read(joinpath(offline, "motorcycle-noncentered.stan")) || error(
            "Generated Stan source differs from the saved pilot producer")
    descriptor = brm_descriptor(base.sb)
    base_unc = BS.param_unc_names(base.density.model)
    names, constrained = constrained_draws(base, fit)
    positions = fit.posterior_position
    base_gradients = reduce(hcat, [last(LogDensityProblems.logdensity_and_gradient(base.density, collect(q))) for q in eachcol(positions)])
    baseline = Dict{Tuple{Symbol,Int,Float64},Float64}()
    comparison = NamedTuple[]
    for (label, means, scales) in (("NCP", zeros(20), zeros(20)), ("Post-hoc", cm, cs), ("Online", oc["mu"], oc["log_sigma"]))
        target = label == "NCP" ? base : stan_density(build_brmi(prepared_data(; c_mu=means, c_sigma=scales); partial=true), "same-draw-$label", output)
        td = brm_descriptor(target.sb)
        target_unc = BS.param_unc_names(target.density.model)
        q = fill(NaN, size(positions))
        expected_basis_gradients = Dict{Symbol,Matrix{Float64}}()
        for (predictor, centering) in ((:mu, means), (:sigma, scales))
            gp = hsgp_coordinate_draws(descriptor, permutedims(constrained), names; predictor, term=:hsgp_x, centeredness=centering)
            for parameter in (:basis_weights, :length_scale, :sd)
                src = brm_term_coordinates(descriptor, predictor, base_unc; term=:hsgp_x, parameter).coordinates
                dst = brm_term_coordinates(td, predictor, target_unc; term=:hsgp_x, parameter).coordinates
                q[dst, :] = parameter == :basis_weights ? permutedims(gp.coordinates) : positions[src, :]
                if parameter == :basis_weights
                    expected_basis_gradients[predictor] = permutedims(hsgp_transform_draws(
                        permutedims(positions[src, :]), gp.log_scales; to=centering,
                        gradients=permutedims(base_gradients[src, :])).gradients)
                end
            end
        end
        @assert all(isfinite, q)
        gradients = reduce(hcat, [last(LogDensityProblems.logdensity_and_gradient(target.density, collect(x))) for x in eachcol(q)])
        @assert all(isfinite, gradients)
        rp = adaptive_centering_problem(target.sb, target.density, ENZYME_BACKEND)
        scored = candidate_scoring_losses(rp, q, gradients)
        for predictor in (:mu, :sigma)
            binding = brm_term_coordinates(td, predictor, target_unc; term=:hsgp_x, parameter=:basis_weights)
            wanted = expected_basis_gradients[predictor]
            err = maximum(abs.(gradients[binding.coordinates, :] .- wanted) ./ max.(1.0, abs.(wanted)))
            println("same_physical_draws\t", label, "\t", predictor, "\tgradient_error=", err)
            @assert err < 1e-7
            for (basis, index) in enumerate(binding.coordinates), s in scored
                s.index == index || continue
                key = (predictor, basis, Float64(s.candidate))
                label == "NCP" && (baseline[key] = s.loss)
                difference = abs(s.loss - baseline[key])
                push!(comparison, (; frame=label, predictor, basis, candidate=s.candidate, loss=s.loss, baseline=baseline[key], absolute_difference=difference))
            end
        end
        flush(stdout)
    end
    write_tsv(joinpath(output, "same_draw_loss_frame_comparison.tsv"), comparison)
    maxerr = maximum(r.absolute_difference for r in comparison)
    @assert maxerr < 1e-8
    println("same_draw_loss_frame_invariance\tcomparisons=", length(comparison), "\tmax_abs_error=", maxerr)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: audit_loss_frames.jl OFFLINE_DIR ONLINE_DIR OUTPUT_DIR")
    audit_loss_frames(ARGS...)
end

