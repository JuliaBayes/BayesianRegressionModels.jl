# test/plate_for.jl — `@brm` annotated kernel-cell surface.
#
# Run: julia --startup-file=no --project=test test/plate_for.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const PLATE_PANEL = (;
    t=[[0.2, 0.7, 1.1], [0.1, 0.5]],
    theta=[0.8, 1.2],
    catalog=[[0.05, 0.1], [0.02, 0.08]],
    index=[[1, 2, 1], [2, 1]],
    y=[[0.18, 0.55, 0.91], [0.14, 0.58]],
)

plate_panel_legacy(data) = @brm data begin
    sigma ~ Exponential(1)
    loc ~ kernel(t, theta, catalog, index, y) do t, theta, catalog, index, y
        loc = theta .* t .+ catalog[index]
        y ~ normal(loc, sigma)
        loc
    end
end

plate_panel_annotated(data) = @brm data begin
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        loc[i] = theta[i] .* t[i] .+ catalog[i][index[i]]
        y[i] ~ normal(loc[i], sigma)
    end
end

const PLATE_RAGGED = (;
    subject=["b", "a", "c"],
    event_subject=["a", "b", "a", "c", "c", "b"],
    event_time=[0.2, 0.1, 0.7, 0.3, 1.0, 0.8],
    y=[0.21, 0.11, 0.69, 0.31, 1.03, 0.82],
)

plate_ragged_legacy(data) = @brm data begin
    sigma ~ Exponential(1)
    theta ~ 1 + (1 | p | subject)
    effect(theta, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    loc ~ kernel(theta,
                 ragged(event_time, event_subject),
                 ragged(y, event_subject)) do theta, event_time, y
        loc = theta .* event_time
        y ~ normal(loc, sigma)
        loc
    end
end

plate_ragged_annotated(data) = @brm data begin
    sigma ~ Exponential(1)
    theta ~ 1 + (1 | p | subject)
    effect(theta, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(theta)
        loc[i] = theta[i] .* ragged(event_time, event_subject)[i]
        ragged(y, event_subject)[i] ~ normal(loc[i], sigma)
    end
end

plate_named_outputs_legacy(data) = @brm data begin
    sigma ~ Exponential(1)
    secondary ~ kernel(t, theta, y) do t, theta, y
        primary = theta .* t .+ sigma
        secondary = primary .+ 0.25
        y ~ normal(secondary, sigma)
        secondary
    end
end

plate_named_outputs_annotated(data) = @brm data begin
    sigma ~ Exponential(1)
    @plate for i in axes(t, 1)
        primary[i] = theta[i] .* t[i] .+ sigma
        secondary[i] = primary[i] .+ 0.25
        y[i] ~ normal(secondary[i], sigma)
    end
end

@testset "@brm @plate for — exact legacy lowering" begin
    for (label, old, new) in (
        ("no-random-effects panel", plate_panel_legacy(PLATE_PANEL),
         plate_panel_annotated(PLATE_PANEL)),
        ("formula LP + ragged secondary axes", plate_ragged_legacy(PLATE_RAGGED),
         plate_ragged_annotated(PLATE_RAGGED)),
        ("multiple named outputs", plate_named_outputs_legacy(PLATE_PANEL),
         plate_named_outputs_annotated(PLATE_PANEL)),
    )
        old_code = BRM.stan_code(SBBRMI(old; mod=@__MODULE__, total_groups=()))
        new_code = BRM.stan_code(SBBRMI(new; mod=@__MODULE__, total_groups=()))
        @testset "$label" begin
            @test new_code == old_code
            @test StanBlocks.stanc_check(new_code; warn_pedantic=false).ok
        end
    end
end

@testset "@brm @plate for — indexed outputs stay logically addressable" begin
    descriptor = brm_descriptor(SBBRMI(
        plate_named_outputs_annotated(PLATE_PANEL);
        mod=@__MODULE__, total_groups=()))
    primary = brm_output(descriptor, :primary)
    secondary = brm_output(descriptor, :secondary)
    @test primary.logical === :primary
    @test primary.role === :group_block
    @test secondary.logical === :secondary
    @test secondary.role === :group_block
end

@testset "@brm @plate for — RK log-density parity" begin
    old_backend, old_problem = consumer_problem(plate_panel_legacy(PLATE_PANEL))
    new_backend, new_problem = consumer_problem(plate_panel_annotated(PLATE_PANEL))
    @test coordinate_names(old_backend.model.layout) ==
          coordinate_names(new_backend.model.layout)
    for u in ([0.0], [0.3], [-0.4])
        old_value, old_gradient = LogDensityProblems.logdensity_and_gradient(old_problem, u)
        new_value, new_gradient = LogDensityProblems.logdensity_and_gradient(new_problem, u)
        @test new_value == old_value
        @test new_gradient == old_gradient
    end
end

@testset "@brm @plate for — syntax fails at the formula boundary" begin
    no_output = :(@brm PLATE_PANEL begin
        @plate for i in eachindex(t)
            y[i] ~ normal(theta[i] .* t[i], 1)
        end
    end)
    bad_range = :(@brm PLATE_PANEL begin
        @plate for i in 1:2
            loc[i] = theta[i] .* t[i]
        end
    end)
    bad_slice = :(@brm PLATE_PANEL begin
        @plate for i in eachindex(t)
            loc[i] = t[i, 1]
        end
    end)
    shadow = :(@brm PLATE_PANEL begin
        @plate for i in eachindex(t)
            t = 1.0
            loc[i] = theta[i] .* t[i]
        end
    end)
    @test_throws "needs an indexed deterministic output" macroexpand(@__MODULE__, no_output)
    @test_throws "range must be" macroexpand(@__MODULE__, bad_range)
    @test_throws "one cell slice" macroexpand(@__MODULE__, bad_slice)
    @test_throws "shadows a sliced input" macroexpand(@__MODULE__, shadow)
end
