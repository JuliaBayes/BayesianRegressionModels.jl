include(joinpath(@__DIR__, "reproduce.jl"))
using Test

function source_model(output_dir)
    mkpath(output_dir)
    source_path = checked_file("eight_schools.stan", SOURCE_MODEL_SHA256)
    data = read_eight_schools()
    array_json(x) = "[" * join(x, ",") * "]"
    source_data = "{" * join((
        "\"J\":8", "\"y\":" * array_json(data.y),
        "\"sigma\":" * array_json(data.sigma)), ",") * "}"
    BS.StanModel(source_path, source_data)
end

function audit_source(; output_dir=get(ENV, "BRM_EIGHT_SCHOOLS_OUTPUT", mktempdir()))
    mkpath(output_dir)
    source = source_model(output_dir)
    stan = stan_density("audit", output_dir)
    source_names = BS.param_unc_names(source)
    brm_names = BS.param_unc_names(stan.density.model)
    receipts = NamedTuple[]
    @testset "Immutable Stan eight-schools target" begin
        @test length(source_names) == length(brm_names) == 10
        @test count(contains("theta"), source_names) == 8
        for point in 1:8
            q_source = [0.4sin(i + point) for i in 1:9]
            push!(q_source, 0.25 + 0.15point)
            q_brm = brm_position(source, q_source, brm_names)
            source_value, source_gradient = BS.log_density_gradient(
                source, q_source; propto=false, jacobian=true)
            brm_value, brm_gradient = BS.log_density_gradient(
                stan.density.model, q_brm; propto=false, jacobian=true)
            scale_index = only(findall(
                name -> occursin(r"_tau(?:\.\d+)?$", name), brm_names))
            transformed_brm_value = brm_value - 8 * q_brm[scale_index]
            physical_gradient = brm_physical_gradient(
                brm_gradient, brm_names, q_brm, source, q_source, source_names)
            @test isfinite(source_value) && all(isfinite, source_gradient)
            @test isapprox(transformed_brm_value, source_value; atol=1e-9, rtol=1e-11)
            @test all(isapprox.(physical_gradient, source_gradient; atol=1e-9, rtol=1e-10))
            push!(receipts, (; point, source_value, brm_value, transformed_brm_value,
                density_absolute_error=abs(transformed_brm_value - source_value),
                max_gradient_absolute_error=maximum(abs.(physical_gradient .- source_gradient))))
        end
        # The two source improper priors are constants on their declared supports;
        # BridgeStan's positive unconstraining Jacobian is included above.
        @test Distributions.logpdf(EightSchoolsFlat(), 0.3) == 0.0
        @test Distributions.logpdf(EightSchoolsFlatPositive(), 0.3) == 0.0
        @test Distributions.logpdf(EightSchoolsFlatPositive(), -0.1) == -Inf
    end
    write_tsv(joinpath(output_dir, "source_density_gradient_audit.tsv"), receipts)
    layout = brm_layout(brm_names)
    write_tsv(joinpath(output_dir, "source_coordinate_map.tsv"), [
        (; brm=brm_names[i], source_physical=(i == layout.population ? "mu" :
            i == layout.scale ? "tau (positive; unconstrained log)" :
            "theta.$(only(findall(==(i), layout.effects)))"))
        for i in eachindex(brm_names)])
    println("source_audit_complete\t", output_dir)
    (; receipts, source, stan, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    audit_source()
end
