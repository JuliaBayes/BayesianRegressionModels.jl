include(joinpath(@__DIR__, "reproduce.jl"))
using Test
import Downloads

const SOURCE_STAN_SHA256 = "dd9b735050533dc39eb74571e435044330293ce4875b57e53dacfe7adcfad99b"

"""Compare the actual normalized target and all gradients with the immutable blog Stan model."""
function audit_source(; output_dir=get(ENV, "BRM_ADAPTIVE_OUTPUT", mktempdir()))
    mkpath(output_dir)
    source_path = joinpath(output_dir, "original-mcycle.stan")
    if haskey(ENV, "BRM_ADAPTIVE_SOURCE")
        cp(ENV["BRM_ADAPTIVE_SOURCE"], source_path; force=false)
    else
        Downloads.download("https://raw.githubusercontent.com/generable/public-materials/$SOURCE_REVISION/blog/hsgp-reparam/stan/mcyclex.stan", source_path)
    end
    @assert bytes2hex(sha256(read(source_path))) == SOURCE_STAN_SHA256
    data = prepared_data()
    array_json(x) = "[" * join(x, ",") * "]"
    source_data = "{" * join((
        "\"x_n\":133", "\"x\":" * array_json(data.times),
        "\"y_n\":133", "\"y\":" * array_json(data.y),
        "\"n_functions\":20", "\"x_scale_scale\":4", "\"y_scale_scale\":4"), ",") * "}"
    source = BS.StanModel(source_path, source_data)
    stan = stan_density(build_brmi(data), "audit", output_dir)
    source_names = BS.param_unc_names(source)
    brm_names = BS.param_unc_names(stan.density.model)
    names = Dict{String,String}()
    for (brm, original) in (("hsgp_x", "gp_loc"), ("hsgp_log_sigma_x", "gp_log_scale"))
        names["$(brm)_rho_iso"] = "$(original)_log_x_scale"
        names["$(brm)_sigma"] = "$(original)_log_y_scale"
        for j in 1:DEFAULT_K
            names["$(brm)_beta_raw.$j"] = "$(original)_unit_weight.$j"
        end
    end
    permutation = [only(findall(==(names[name]), source_names)) for name in brm_names]
    receipts = NamedTuple[]
    @testset "Immutable blog target, hyperpriors and all 44 gradients" begin
        @test length(source_names) == length(brm_names) == 44
        @test sort(permutation) == collect(1:44)
        # Algebraic positive-scale prior + unconstraining Jacobian control.
        for q in (-5.0, -2.0, -0.5, 0.0, 0.5, 2.0)
            @test logpdf(LogNormal(0, 4), exp(q)) + q ≈ logpdf(Normal(0, 4), q)
        end
        for (point, log_rho) in enumerate((-4.0, -2.0, -1.0, -0.5, 0.0, 0.5))
            q = zeros(44)
            for (i, name) in enumerate(source_names)
                q[i] = if endswith(name, "log_x_scale")
                    log_rho + (startswith(name, "gp_log_scale") ? -0.15 : 0.0)
                elseif endswith(name, "log_y_scale")
                    -0.2 + point * 0.1
                else
                    0.08sin(i + point)
                end
            end
            original_value, original_gradient = BS.log_density_gradient(source, q; propto=false, jacobian=true)
            brm_value, brm_gradient = BS.log_density_gradient(stan.density.model, q[permutation]; propto=false, jacobian=true)
            @test isfinite(original_value) && all(isfinite, original_gradient)
            @test isapprox(brm_value, original_value; atol=1e-8, rtol=1e-10)
            @test isapprox(brm_gradient, original_gradient[permutation]; atol=1e-8, rtol=1e-10)
            push!(receipts, (; point, log_rho, original_value, brm_value,
                density_absolute_error=abs(brm_value - original_value),
                max_gradient_absolute_error=maximum(abs.(brm_gradient - original_gradient[permutation]))))
        end
    end
    write_tsv(joinpath(output_dir, "source_density_gradient_audit.tsv"), receipts)
    write_tsv(joinpath(output_dir, "source_coordinate_map.tsv"), [
        (; brm=brm_names[i], original=source_names[permutation[i]]) for i in eachindex(brm_names)])
    println("source_audit_complete\t", output_dir)
    (; receipts, stan, source, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    audit_source()
end
