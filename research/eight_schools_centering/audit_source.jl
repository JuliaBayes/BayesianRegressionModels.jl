include(joinpath(@__DIR__, "reproduce.jl"))
using Test

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

    end
    centered = fixed_partial_problem(stan, ones(8))
    layout = brm_layout(brm_names)
    @testset "Manual fully centered selection" begin
        @test all(last(p).c == 1 for p in WarmupHMC.reparam_sources(centered))
        for point in 1:8
            q = [0.15sin(i + point) for i in eachindex(brm_names)]
            q[layout.scale] = -2.0 + 0.5point
            tau = exp(q[layout.scale])
            ncp = copy(q)
            ncp[layout.effects] ./= tau
            vn, gn = LogDensityProblems.logdensity_and_gradient(stan.density, ncp)
            vc, gc = LogDensityProblems.logdensity_and_gradient(centered, q)
            @test vc ≈ vn - 8log(tau) atol=1e-9
            expected = copy(gn)
            expected[layout.effects] ./= tau
            expected[layout.scale] -= dot(gn[layout.effects], ncp[layout.effects]) + 8
            @test gc ≈ expected atol=1e-8
            restored = reshape(copy(q), :, 1)
            WarmupHMC.reparametrize!(centered, restored)
            @test vec(restored) ≈ ncp atol=1e-12
        end
    end
    centered_sb = SBBRMI(build_brmi(); mod=@__MODULE__, centered_groups=[:school])
    centered_density = Base.invokelatest(StanBlocks.stan_instantiate, centered_sb.model;
        path=joinpath(output_dir, "eight-schools-native-centered.stan"))
    centered_names = BS.param_unc_names(centered_density.model)
    centered_block = only(adaptive_centering_blocks(centered_sb, centered_names))
    centered_mu = only(findall(n -> occursin("pop_theta_beta_pop", n), centered_names))
    @testset "Native centered_groups and manual c=1 agree" begin
        @test StanBlocks.stanc_check(BRM.stan_code(centered_sb)).ok
        for point in 1:8
            q = [0.2sin(i+point) for i in eachindex(brm_names)]
            q[layout.scale] = -2 + 0.5point
            qc = zeros(10)
            qc[centered_mu] = q[layout.population]
            qc[only(centered_block.log_scales)] = q[layout.scale]
            qc[vec(centered_block.effects)] = q[layout.effects]
            vw, gw = LogDensityProblems.logdensity_and_gradient(centered, q)
            vn, gn = LogDensityProblems.logdensity_and_gradient(centered_density, qc)
            @test vw ≈ vn atol=1e-9
            @test gw[layout.effects] ≈ gn[vec(centered_block.effects)] atol=1e-8
            @test gw[layout.population] ≈ gn[centered_mu] atol=1e-8
            @test gw[layout.scale] ≈ gn[only(centered_block.log_scales)] atol=1e-8
        end
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
