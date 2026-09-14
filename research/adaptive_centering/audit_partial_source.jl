# Check the second source model at actual saved posterior positions; never sample.
include(joinpath(@__DIR__, "reproduce.jl"))
using Test
import Downloads

const PARTIAL_SOURCE_SHA256 = "221c7119161b143de57e0d88eb14373ee7ee361fc367910c247770d965fb7144"

function audit_partial_source(fit_dir, output_dir)
    mkpath(output_dir)
    source_path = joinpath(output_dir, "original-adaptive-mcycle.stan")
    Downloads.download("https://raw.githubusercontent.com/generable/public-materials/$SOURCE_REVISION/blog/hsgp-reparam/stan/adaptive_mcyclex.stan", source_path)
    bytes2hex(sha256(read(source_path))) == PARTIAL_SOURCE_SHA256 || error("partial source hash mismatch")
    rows = split.(readlines(joinpath(fit_dir, "centeredness.tsv"))[2:end], '\t')
    c_mu = parse.(Float64, getindex.(rows, 2))
    c_sigma = parse.(Float64, getindex.(rows, 3))
    data = prepared_data(; c_mu, c_sigma)
    array_json(x) = "[" * join(x, ",") * "]"
    source_data = "{" * join((
        "\"x_n\":133", "\"x\":" * array_json(data.times),
        "\"y_n\":133", "\"y\":" * array_json(data.y),
        "\"n_functions\":20", "\"x_scale_scale\":4", "\"y_scale_scale\":4",
        "\"gp_loc_centerednesses_n\":20", "\"gp_loc_centerednesses\":" * array_json(c_mu),
        "\"gp_log_scale_centerednesses_n\":20", "\"gp_log_scale_centerednesses\":" * array_json(c_sigma)), ",") * "}"
    source = BS.StanModel(source_path, source_data)
    stan = stan_density(build_brmi(data; partial=true), "partial-audit", output_dir)
    # Exact generated-source equality binds saved rows to this model's names.
    # Refuse a different producer model instead of guessing its parameter order.
    read(joinpath(fit_dir, "motorcycle-partial.stan")) ==
        read(joinpath(output_dir, "motorcycle-partial-audit.stan")) ||
        error("Producer model source differs: an explicit coordinate mapping is required.")
    source_names = BS.param_unc_names(source)
    brm_names = BS.param_unc_names(stan.density.model)
    fit = deserialize(joinpath(fit_dir, "partial.jls"))
    names = Dict{String,String}()
    for (brm, original) in (("hsgp_x", "gp_loc"), ("hsgp_log_sigma_x", "gp_log_scale"))
        names["$(brm)_rho_iso"] = "$(original)_log_x_scale"
        names["$(brm)_sigma"] = "$(original)_log_y_scale"
        for j in 1:DEFAULT_K
            names["$(brm)_beta_partial.$j"] = "$(original)_adaptive_weight.$j"
        end
    end
    permutation = [only(findall(==(names[name]), source_names)) for name in brm_names]
    receipts = NamedTuple[]
    @testset "Original partial model at saved full-fit posterior positions" begin
        @test length(source_names) == length(brm_names) == 44
        @test sort(permutation) == collect(1:44)
        @test fit.complete && size(fit.posterior_position, 2) >= SOURCE_DRAWS
        for draw in round.(Int, range(1, size(fit.posterior_position, 2); length=16))
            q_brm = fit.posterior_position[:, draw]
            q_source = similar(q_brm)
            q_source[permutation] = q_brm
            original_value, original_gradient = BS.log_density_gradient(source, q_source; propto=false, jacobian=true)
            brm_value, brm_gradient = BS.log_density_gradient(stan.density.model, q_brm; propto=false, jacobian=true)
            @test isfinite(original_value) && all(isfinite, original_gradient)
            @test isapprox(brm_value, original_value; atol=1e-8, rtol=1e-10)
            @test all(isapprox.(brm_gradient, original_gradient[permutation]; atol=1e-8, rtol=1e-10))
            push!(receipts, (; draw, original_value, brm_value,
                density_absolute_error=abs(brm_value - original_value),
                max_gradient_absolute_error=maximum(abs.(brm_gradient - original_gradient[permutation]))))
        end
    end
    write_tsv(joinpath(output_dir, "partial_source_density_gradient_audit.tsv"), receipts)
    write_tsv(joinpath(output_dir, "partial_source_coordinate_map.tsv"), [
        (; brm=brm_names[i], original=source_names[permutation[i]]) for i in eachindex(brm_names)])
    println("partial_source_audit_complete\t", output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: audit_partial_source.jl FULL_FIT_DIRECTORY AUDIT_OUTPUT_DIRECTORY")
    audit_partial_source(ARGS...)
end
