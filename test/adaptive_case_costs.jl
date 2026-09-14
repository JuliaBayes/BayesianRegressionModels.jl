using Test
include(joinpath(@__DIR__, "..", "research", "adaptive_centering", "report_costs.jl"))

@testset "Case-study cost telemetry preserves historical missingness" begin
    @test ismissing(recorded_sampling_cost((;), (;), 1000))
    @test ismissing(recorded_sampling_cost(
        (; sampling_gradient_evaluations=missing), (;), 1000))
    # A newly added checkpoint field alone cannot establish pre-resume coverage.
    @test ismissing(recorded_sampling_cost(
        (;), (; sampling_evaluation_counter=300), 1000))
    @test recorded_sampling_cost((; sampling_gradient_evaluations=300),
        (; sampling_evaluation_counter=300), 1000) == 300
    for count in (0, -1, 1001, 3.5, Inf)
        @test_throws ArgumentError recorded_sampling_cost(
            (; sampling_gradient_evaluations=count),
            (; sampling_evaluation_counter=count), 1000)
    end
    @test_throws ArgumentError recorded_sampling_cost(
        (; sampling_gradient_evaluations=300), (;), 1000)
    @test_throws ArgumentError recorded_sampling_cost(
        (; sampling_gradient_evaluations=300), (; sampling_evaluation_counter=301), 1000)

    mktempdir() do dir
        for label in ("noncentered", "partial", "online")
            q = randn(Xoshiro(1), 44, 128) # Synthetic reporting fixture, not an HMC fit.
            fit = (; posterior_position=q, complete=true, n_divergent_samples=0,
                total_gradient_evaluations=1000, sampling_gradient_evaluations=300,
                fit_seconds=2.0)
            payload = (; posterior_position=q, n_divergent_samples=0,
                total_evaluation_counter=1000, sampling_evaluation_counter=300)
            serialize(joinpath(dir, "$label.jls"), fit)
            cp = joinpath(dir, "checkpoints-$label")
            mkpath(cp)
            serialize(joinpath(cp, "cp_latest.jls"), payload)
        end
        rows = report_costs(dir, dir, joinpath(dir, "reported"))
        @test length(rows) == 3
        for row in rows
            @test row.fit_seconds == 2.0
            @test row.sampling_gradient_evaluations == 300
            @test row.min_bulk_ess_per_sampling_gradient == row.min_bulk_ess / 300
            @test row.min_tail_ess_per_sampling_gradient == row.min_tail_ess / 300
            @test row.min_bulk_ess_per_total_gradient == row.min_bulk_ess / 1000
        end
    end
end
