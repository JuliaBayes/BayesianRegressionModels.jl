using Test
using StanBlocks
using BridgeStan
using LogDensityProblems
include(joinpath(@__DIR__, "spline_parity_models.jl"))

function spline_sb_point(label, model)
    values = Dict("sigma" => -0.25, "pop_mu_beta_pop.1" => -0.3)
    if label == "s"
        merge!(values, Dict("s_x_b_fixed.$i" => v for
            (i, v) in enumerate([-0.2, -0.15])))
        merge!(values, Dict("s_x_b_pen_raw.$i" => v for
            (i, v) in enumerate([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25])))
        values["s_x_sd_pen.1"] = 0.3
    else
        for (name, vs) in (("b_fixed", [-0.2, -0.15, -0.1]),
                ("b_rr_raw", [-0.05]), ("b_rn_raw", [0.0, 0.05]),
                ("b_nr_raw", [0.1, 0.15]), ("sd_pen", [0.2, 0.25, 0.3]))
            merge!(values, Dict("t2_mu_x_z_$(name).$i" => v for
                (i, v) in enumerate(vs)))
        end
    end
    names = BridgeStan.param_unc_names(model)
    @test Set(names) == Set(keys(values))
    [values[name] for name in names]
end

@testset "canonical spline BridgeStan anchors" begin
    for (label, fixture, dimension, anchor) in (
            ("s", spline_s_parity_case, 13, SPLINE_S_SB_ANCHOR),
            ("t2", spline_t2_parity_case, 13, SPLINE_T2_SB_ANCHOR),
            ("accel", spline_accel_parity_case, 24, SPLINE_ACCEL_SB_ANCHOR))
        @testset "$label" begin
            sb = SBBRMI(fixture().brmi; mod=@__MODULE__)
            problem = StanBlocks.stan_instantiate(sb.model)
            @test LogDensityProblems.dimension(problem) == dimension
            u = label == "accel" ? fill(0.3, dimension) :
                spline_sb_point(label, problem.model)
            density(v) = BridgeStan.log_density(problem.model, v;
                propto=false, jacobian=true)
            @test density(u) ≈ anchor atol=1e-9 rtol=0
            grad = zeros(dimension)
            lp, _ = BridgeStan.log_density_gradient!(problem.model, u, grad;
                propto=false, jacobian=true)
            @test lp ≈ anchor atol=1e-9 rtol=0
            @test all(isfinite, grad)
            fd = BayesianRegressionModels._sb_central_diff(density, u, 1e-6)
            @test maximum(abs.(grad .- fd)) < 1e-4
            if label == "accel"
                @test density(zeros(dimension)) ≈ SPLINE_ACCEL_SB_ZERO_ANCHOR atol=1e-9 rtol=0
            end
        end
    end
end
