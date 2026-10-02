# Run: julia --project=test test/spline_basis_signs.jl
using Test
using LinearAlgebra
using Random

# Exercise the shared preparation directly without loading either backend.
include(joinpath(@__DIR__, "..", "src", "preparation_basis.jl"))

function has_canonical_signs(M)
    all(eachcol(M)) do col
        peak = maximum(abs, col)
        peak == 0 && return true
        first(v for v in col if abs(v) > 1e-8 * peak) > 0
    end
end

@testset "canonical spline projection signs" begin
    @testset "relative threshold, sign invariance, and input ownership" begin
        # Tiny leading values do not determine a sign; the threshold is strict
        # and relative to each column, including differently scaled columns.
        M = [0.0 -1e-10 -1e-8 1e-14 0.0;
             -2.0 1.0 1.0 -1e-6 0.0;
             1.0 -0.5 -0.5 0.5e-6 0.0]
        original = copy(M)
        expected = M .* reshape([-1, 1, 1, -1, 1], 1, :)
        canonical = _brm_canonical_column_signs(M)
        @test canonical == expected
        @test canonical !== M
        @test M == original
        @test has_canonical_signs(canonical)
        @test _brm_canonical_column_signs(canonical) == canonical
        for mask in 0:31
            signs = [iszero(mask & (1 << (j - 1))) ? 1 : -1 for j in 1:5]
            flipped = M .* reshape(signs, 1, :)
            @test _brm_canonical_column_signs(flipped) == canonical
        end
        @test _brm_canonical_column_signs(view(M, :, 1:4)) == expected[:, 1:4]
        @test M == original
    end

    @testset "TPS and tensor margins retain canonical frozen projections" begin
        rng = Xoshiro(7207)
        x = 5 .* randn(rng, 80)
        z = 3 .* randn(rng, 80) .+ 1
        original_x, original_z = copy(x), copy(z)
        for k in (3, 5, 10)
            fit = _brm_fit_spline(x; k)
            projection = copy(fit.range_projection)
            @test has_canonical_signs(projection)
            Xnull, Zpen = _brm_apply_spline(fit, x)
            @test size(Xnull) == (80, 2)
            @test size(Zpen) == (80, k - 2)
            @test transpose(Xnull) * projection ≈ zeros(2, k - 2) atol=1e-9
            _brm_apply_spline(fit, [-7.0, 0.0, 8.0])
            @test fit.range_projection == projection
        end
        for k in ((3, 3), (4, 6), (5, 5))
            fit = _brm_fit_t2(x, z; k)
            projections = map(margin -> copy(margin.range_projection), fit.margins)
            @test all(has_canonical_signs, projections)
            for margin in fit.margins
                _, penalty = _brm_cr_second_derivative_map(margin.knots)
                @test transpose(margin.range_projection) * penalty * margin.range_projection ≈
                      Matrix{Float64}(I, margin.k - 2, margin.k - 2) atol=1e-10
            end
            _brm_apply_t2(fit, [-7.0, 0.0, 8.0], [-3.0, 1.0, 9.0])
            @test all(i -> fit.margins[i].range_projection == projections[i], 1:2)
        end
        @test x == original_x
        @test z == original_z
    end
end
