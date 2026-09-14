include(joinpath(@__DIR__, "reproduce.jl"))
using Serialization
using Test

"""Compare the immutable source target at actual retained full-fit draws; never sample."""
function audit_saved_draws(output_dir)
    isfile(joinpath(output_dir, "noncentered.jls")) ||
        error("full pilot output is required: $output_dir/noncentered.jls")
    fit = deserialize(joinpath(output_dir, "noncentered.jls"))
    fit.complete && size(fit.posterior_position, 2) >= SOURCE_DRAWS ||
        error("saved pilot is not a completed full fit")
    source = source_model(output_dir)
    stan = stan_density("saved-audit", output_dir)
    source_names = BS.param_unc_names(source)
    brm_names = BS.param_unc_names(stan.density.model)
    layout = brm_layout(brm_names)
    receipts = NamedTuple[]
    @testset "Immutable source at retained posterior positions" begin
        @test size(fit.posterior_position) == (10, 10_000)
        @test all(isfinite, fit.posterior_position)
        for draw in round.(Int, range(1, size(fit.posterior_position, 2); length=16))
            q_brm = collect(fit.posterior_position[:, draw])
            logtau = q_brm[layout.scale]
            tau = exp(logtau)
            mu = q_brm[layout.population]
            q_source = zeros(length(source_names))
            for (i, name) in enumerate(source_names)
                q_source[i] = name == "mu" ? mu :
                    name == "tau" ? logtau :
                    mu + tau * q_brm[layout.effects[parse(Int, last(split(name, '.')))]]
            end
            source_value, source_gradient = BS.log_density_gradient(
                source, q_source; propto=false, jacobian=true)
            brm_value, brm_gradient = BS.log_density_gradient(
                stan.density.model, q_brm; propto=false, jacobian=true)
            transformed_brm_value = brm_value - length(layout.effects) * logtau
            physical_gradient = brm_physical_gradient(
                brm_gradient, brm_names, q_brm, source, q_source, source_names)
            @test isfinite(source_value) && all(isfinite, source_gradient)
            @test isapprox(transformed_brm_value, source_value; atol=1e-9, rtol=1e-11)
            @test all(isapprox.(physical_gradient, source_gradient; atol=1e-9, rtol=1e-10))
            push!(receipts, (; draw, source_value, brm_value, transformed_brm_value,
                density_absolute_error=abs(transformed_brm_value - source_value),
                max_gradient_absolute_error=maximum(abs.(physical_gradient .- source_gradient))))
        end
    end
    write_tsv(joinpath(output_dir, "saved_source_density_gradient_audit.tsv"), receipts)
    println("saved_draw_source_audit_complete\t", output_dir)
    (; receipts, source, stan, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: audit_saved_draws.jl FULL_FIT_DIRECTORY")
    audit_saved_draws(only(ARGS))
end
