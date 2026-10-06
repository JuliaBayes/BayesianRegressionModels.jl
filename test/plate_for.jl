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

# A terminal output assignment is the cell's return value. These pairs pin it
# against closures ending in an anonymous expression (top-level and in-cell
# observations) and against a closure returning a named local.
const PLATE_EVENTS = (;
    subject=["b", "a", "c"],
    weight=[60.0, 75.0, 90.0],
    obs_subject=["a", "b", "a", "c", "c", "b"],
    obs_time=[0.2, 0.1, 0.7, 0.3, 1.0, 0.8],
    obs_y=[0.81, 0.92, 0.49, 0.71, 0.33, 0.52],
)

plate_terminal_top_level_legacy(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    loc ~ kernel(log_CL, ragged(obs_time, obs_subject)) do log_CL, obs_time
        exp.(-exp(log_CL) .* obs_time)
    end
    ragged(obs_y, obs_subject) ~ Normal(loc, sigma)
end

plate_terminal_top_level_annotated(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp.(-exp(log_CL[i]) .* ragged(obs_time, obs_subject)[i])
    end
    ragged(obs_y, obs_subject) ~ Normal(loc, sigma)
end

plate_terminal_anonymous_legacy(data) = @brm data begin
    sigma ~ Exponential(1)
    loc ~ kernel(t, y, theta) do t, y, theta
        y ~ normal(theta .* t, sigma)
        theta .* t
    end
end

plate_terminal_anonymous_annotated(data) = @brm data begin
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        y[i] ~ normal(theta[i] .* t[i], sigma)
        loc[i] = theta[i] .* t[i]
    end
end

plate_terminal_named_legacy(data) = @brm data begin
    sigma ~ Exponential(1)
    loc ~ kernel(t, theta, y) do t, theta, y
        mu = theta .* t
        y ~ normal(mu, sigma)
        mu
    end
end

plate_terminal_named_annotated(data) = @brm data begin
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        mu = theta[i] .* t[i]
        y[i] ~ normal(mu, sigma)
        loc[i] = mu
    end
end

plate_terminal_named_ragged_legacy(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    loc ~ kernel(log_CL,
                 ragged(obs_time, obs_subject),
                 ragged(obs_y, obs_subject)) do log_CL, obs_time, obs_y
        mu = exp.(-exp(log_CL) .* obs_time)
        obs_y ~ normal(mu, sigma)
        mu
    end
end

plate_terminal_named_ragged_annotated(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    @plate for i in eachindex(log_CL)
        mu = exp.(-exp(log_CL[i]) .* ragged(obs_time, obs_subject)[i])
        ragged(obs_y, obs_subject)[i] ~ normal(mu, sigma)
        loc[i] = mu
    end
end

@testset "@brm kernel(...) do — deprecated compatibility spelling" begin
    @test_deprecated r"kernel.*do.*deprecated.*@plate for" plate_panel_legacy(PLATE_PANEL)
    @test_logs plate_panel_annotated(PLATE_PANEL)
end

@testset "@brm @plate for — exact legacy lowering" begin
    for (label, old, new) in (
        ("no-random-effects panel", plate_panel_legacy(PLATE_PANEL),
         plate_panel_annotated(PLATE_PANEL)),
        ("formula LP + ragged secondary axes", plate_ragged_legacy(PLATE_RAGGED),
         plate_ragged_annotated(PLATE_RAGGED)),
        ("multiple named outputs", plate_named_outputs_legacy(PLATE_PANEL),
         plate_named_outputs_annotated(PLATE_PANEL)),
        ("terminal output, top-level ragged observation",
         plate_terminal_top_level_legacy(PLATE_EVENTS),
         plate_terminal_top_level_annotated(PLATE_EVENTS)),
        ("terminal output, in-cell observation",
         plate_terminal_anonymous_legacy(PLATE_PANEL),
         plate_terminal_anonymous_annotated(PLATE_PANEL)),
        ("terminal output of a named cell local",
         plate_terminal_named_legacy(PLATE_PANEL),
         plate_terminal_named_annotated(PLATE_PANEL)),
        ("terminal named cell local, formula LP + ragged axes",
         plate_terminal_named_ragged_legacy(PLATE_EVENTS),
         plate_terminal_named_ragged_annotated(PLATE_EVENTS)),
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

    terminal = brm_descriptor(SBBRMI(
        plate_terminal_top_level_annotated(PLATE_EVENTS);
        mod=@__MODULE__, total_groups=()))
    @test brm_output(terminal, :loc).logical === :loc
    @test brm_output(terminal, :loc).role === :group_block
    named = brm_descriptor(SBBRMI(
        plate_terminal_named_ragged_annotated(PLATE_EVENTS);
        mod=@__MODULE__, total_groups=()))
    for name in (:loc, :mu)
        @test brm_output(named, name).logical === name
        @test brm_output(named, name).role === :group_block
    end
end

@testset "@brm @plate for — RK log-density parity" begin
    for (label, legacy, annotated) in (
        ("indexed output read in-cell", plate_panel_legacy, plate_panel_annotated),
        ("terminal output", plate_terminal_anonymous_legacy,
         plate_terminal_anonymous_annotated),
        ("terminal output of a named cell local", plate_terminal_named_legacy,
         plate_terminal_named_annotated),
    )
        @testset "$label" begin
            old_backend, old_problem = consumer_problem(legacy(PLATE_PANEL))
            new_backend, new_problem = consumer_problem(annotated(PLATE_PANEL))
            @test coordinate_names(old_backend.model.layout) ==
                  coordinate_names(new_backend.model.layout)
            for u in ([0.0], [0.3], [-0.4])
                old_value, old_gradient =
                    LogDensityProblems.logdensity_and_gradient(old_problem, u)
                new_value, new_gradient =
                    LogDensityProblems.logdensity_and_gradient(new_problem, u)
                @test new_value == old_value
                @test new_gradient == old_gradient
            end
        end
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
