include(joinpath(@__DIR__, "reproduce.jl"))
using Test

"""Compare BRM's full normalized target and all gradients with immutable PosteriorDB Stan."""
function audit_source(; output_dir=get(ENV, "BRM_RADON_AUDIT_OUTPUT", mktempdir()))
    mkpath(output_dir)
    source_path = joinpath(RESEARCH_DIR, "reference", "$MODEL_NAME.stan")
    bytes2hex(sha256(read(source_path))) == SOURCE_STAN_SHA256 ||
        error("PosteriorDB reference Stan hash mismatch")
    source_data = JSON.json((;
        N=RADON_DATA.N, J=RADON_DATA.J, county_idx=RADON_DATA.county_idx,
        floor_measure=RADON_DATA.floor_measure, log_radon=RADON_DATA.log_radon))
    source = BS.StanModel(source_path, source_data)
    stan = stan_density("audit", output_dir)
    source_names = String.(BS.param_unc_names(source))
    brm_names = String.(BS.param_unc_names(stan.density.model))
    mapping = Dict{String,String}(
        "sigma_y" => "sigma_y",
        "pop_mu_beta_pop.1" => "mu_alpha",
        "pop_mu_beta_pop.2" => "mu_beta",
        "b_county_intercept_county_idx_tau.1" => "sigma_alpha",
        "b_county_slope_county_idx_tau.1" => "sigma_beta",
    )
    for j in 1:RADON_DATA.J
        mapping["b_county_intercept_county_idx_z_flat.$j"] = "alpha_raw.$j"
        mapping["b_county_slope_county_idx_z_flat.$j"] = "beta_raw.$j"
    end
    permutation = [only(findall(isequal(mapping[name]), source_names))
                   for name in brm_names]
    receipts = NamedTuple[]
    @testset "Immutable PosteriorDB target, priors and all 777 gradients" begin
        @test length(source_names) == length(brm_names) == 2RADON_DATA.J + 5
        @test sort(permutation) == collect(1:length(source_names))

        for (point, log_scale) in enumerate((-4.0, -2.5, -1.0, -0.3, 0.0, 0.4))
            q_source = zeros(Float64, length(source_names))
            for (i, name) in enumerate(source_names)
                q_source[i] = if startswith(name, "alpha_raw.") || startswith(name, "beta_raw.")
                    0.075sin(i + 2point)
                elseif name == "sigma_alpha"
                    log_scale
                elseif name == "sigma_beta"
                    log_scale - 0.2
                elseif name == "sigma_y"
                    -0.25 + 0.1point
                elseif name == "mu_alpha"
                    0.4 + 0.07point
                elseif name == "mu_beta"
                    -0.3 - 0.06point
                else
                    error("unexpected reference coordinate $name")
                end
            end
            q_brm = q_source[permutation]
            original_value, original_gradient = BS.log_density_gradient(
                source, q_source; propto=false, jacobian=true)
            brm_value, brm_gradient = BS.log_density_gradient(
                stan.density.model, q_brm; propto=false, jacobian=true)
            @test isfinite(original_value) && all(isfinite, original_gradient)
            @test isapprox(brm_value, original_value; atol=1e-7, rtol=1e-10)
            @test isapprox(brm_gradient, original_gradient[permutation];
                           atol=1e-7, rtol=1e-10)
            push!(receipts, (; point, log_scale, original_value, brm_value,
                density_absolute_error=abs(brm_value - original_value),
                max_gradient_absolute_error=maximum(
                    abs.(brm_gradient - original_gradient[permutation]))))
        end
    end
    write_tsv(joinpath(output_dir, "source_density_gradient_audit.tsv"), receipts)
    write_tsv(joinpath(output_dir, "source_coordinate_map.tsv"), [
        (; brm=brm_names[i], original=source_names[permutation[i]])
        for i in eachindex(brm_names)])
    println("radon_source_audit_complete\t", output_dir)
    (; receipts, stan, source, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    audit_source()
end
