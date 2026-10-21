# test/plate_grouping.jl — a plate over subjects reading other vectors per
# subject through model-scope `groups` bindings.
#
# Run: julia --startup-file=no --project=test test/plate_grouping.jl
#
# `events = groups(event_subject => subject)` binds, for each subject, the
# indices of `event_subject` that hold it; the cell reads `x[events[i]]`, as
# the same Julia loop does (todo `1nfukuv`, decision `0ivno7u`). The read
# lowers to the kernel input `ragged(x, event_subject, subject)`, so the Stan
# program, the RK log density and the descriptor must be identical to the
# legacy `kernel(...) do` spelling with those inputs. The retired
# `ragged(...)` statements and in-cell reads are refused.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Subject vectors (unsorted labels), an interleaved event axis, an
# observation axis and a finer lesion grouping on the event axis.
const GROUPED = (;
    subject=["b", "a", "c"],
    weight=[60.0, 75.0, 90.0],
    indication=[1.0, 0.0, 1.0],
    event_subject=["a", "b", "a", "c", "c", "b", "a"],
    event_lesion=["a1", "b1", "a2", "c1", "c1", "b1", "a1"],
    event_amount=[100.0, 50.0, 100.0, 200.0, 200.0, 50.0, 100.0],
    event_x=[0.1, -0.3, 0.4, 0.2, -0.1, 0.5, -0.2],
    obs_subject=["a", "c", "b", "a", "b", "c"],
    obs_time=[0.5, 0.25, 1.5, 1.0, 3.0, 2.0],
    obs_limit=fill(0.05, 6),
    y=[0.81, 0.76, 0.52, 0.49, 0.21, 0.33],
)

legacy_events(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    loc ~ kernel(log_CL, ragged(event_amount, event_subject, subject),
                 ragged(log_F, event_subject, subject),
                 ragged(obs_time, obs_subject, subject),
                 ragged(y, obs_subject, subject)) do log_CL, event_amount, log_F, obs_time, y
        dose = sum(event_amount .* exp.(log_F))
        loc = 1e-3 * dose .* exp.(-exp(log_CL) .* obs_time)
        y ~ normal(loc, sigma)
        loc
    end
end

bound_events(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    events = groups(event_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        dose = sum(event_amount[events[i]] .* exp.(log_F[events[i]]))
        loc[i] = 1e-3 * dose .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# The bindings may follow the cell that reads them.
bound_after(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    @plate for i in eachindex(log_CL)
        dose = sum(event_amount[events[i]] .* exp.(log_F[events[i]]))
        loc[i] = 1e-3 * dose .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
    obs = groups(obs_subject => subject)
    events = groups(event_subject => subject)
end

# An event-axis LP with a random effect finer than the subject, plus a
# censored in-cell observation whose bound is read through a keyword.
legacy_lesion(data) = @brm data begin
    sigma ~ Exponential(1)
    log_kg ~ 0 + (1 | tgl | event_lesion)
    log_CL ~ 1 + weight + (1 | p | subject)
    location ~ kernel(log_CL, ragged(log_kg, event_subject, subject),
                      ragged(obs_time, obs_subject, subject),
                      ragged(y, obs_subject, subject),
                      ragged(obs_limit, obs_subject, subject)) do log_CL, log_kg, obs_time, y, obs_limit
        growth = sum(exp.(log_kg))
        location = growth .* exp.(-exp(log_CL) .* obs_time)
        y ~ censored(Normal(location, sigma); lower=obs_limit)
        location
    end
end

bound_lesion(data) = @brm data begin
    sigma ~ Exponential(1)
    log_kg ~ 0 + (1 | tgl | event_lesion)
    events = groups(event_subject => subject)
    log_CL ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        growth = sum(exp.(log_kg[events[i]]))
        location[i] = growth .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        y[obs[i]] ~ censored(Normal(location[i], sigma); lower=obs_limit[obs[i]])
    end
end

# A per-subject LP without a random effect rides on the subject vectors the
# ranef LP defines and is read `slope[i]`. The legacy spelling grouped it by
# its own subject vector into one-element vectors and took `[1]`.
wrapped_slope(data) = @brm data begin
    sigma ~ Exponential(1)
    slope ~ 1 + indication
    log_CL ~ 1 + weight + (1 | p | subject)
    loc ~ kernel(log_CL, ragged(slope, subject), ragged(obs_time, obs_subject, subject),
                 ragged(y, obs_subject, subject)) do log_CL, slope, obs_time, y
        loc = exp(log_CL) .+ slope[1] .* obs_time
        y ~ normal(loc, sigma)
        loc
    end
end

direct_slope(data) = @brm data begin
    sigma ~ Exponential(1)
    slope ~ 1 + indication
    log_CL ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ slope[i] .* obs_time[obs[i]]
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# refused: an LP without a random effect is read on the subject vectors, so an
# event-axis LP read as `x[i]` would silently misalign; it is read through a
# binding (snag plate-body-reads-c69f8a88).
event_frame_slope(data) = @brm data begin
    sigma ~ Exponential(1)
    slope ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ sum(slope[i]) .* obs_time[obs[i]]
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

stan_of(brmi) = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
drop_column(data, column) = NamedTuple(k => v for (k, v) in pairs(data) if k !== column)

@testset "groups(g => labels): the indices of each label" begin
    @test groups(GROUPED.event_subject => GROUPED.subject) == [[2, 6], [1, 3, 7], [4, 5]]
    @test groups(["a", "b"] => ["b", "c", "a"]) == [[2], Int[], [1]]
    @test groups(GROUPED.obs_subject => GROUPED.subject; order=GROUPED.obs_time) ==
        [[3, 5], [1, 4], [2, 6]]
    @test_throws "not among the labels" groups(["a", "d"] => ["a", "b"])
    @test_throws "listed twice" groups(["a"] => ["a", "a"])
end

@testset "x[rows[i]] lowers like the legacy kernel inputs" begin
    for (label, old, new) in (
        ("data vectors, event-axis LP, in-cell response", legacy_events, bound_events),
        ("bindings after the cell", legacy_events, bound_after),
        ("finer random effect, censored keyword bound", legacy_lesion, bound_lesion),
    )
        @testset "$label" begin
            old_code = stan_of(old(GROUPED))
            new_code = stan_of(new(GROUPED))
            @test new_code == old_code
            @test StanBlocks.stanc_check(new_code; warn_pedantic=false).ok
        end
    end
end

output_fields(o) = (o.name, o.kind, o.type, o.size, o.constraints, o.generative,
                     o.source, o.role, o.logical, o.labels, o.segments)

@testset "groups bindings keep the descriptor outputs" begin
    for (old, new, lp, cell) in (
        (legacy_events, bound_events, :log_F, :loc),
        (legacy_lesion, bound_lesion, :log_kg, :location),
    )
        old_d = brm_descriptor(SBBRMI(old(GROUPED); mod=@__MODULE__, total_groups=()))
        new_d = brm_descriptor(SBBRMI(new(GROUPED); mod=@__MODULE__, total_groups=()))
        @test new_d.id == old_d.id
        # StanBlocks size expressions have no structural `==`; compare their
        # printed form.
        @test map(repr ∘ output_fields, brm_outputs(new_d)) ==
              map(repr ∘ output_fields, brm_outputs(old_d))
        @test brm_output(new_d, lp; role=:linear_predictor).logical === lp
        @test brm_output(new_d, cell).logical === cell
    end
end

@testset "groups bindings with an omitted in-cell response" begin
    omitted = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        bound_lesion(drop_column(GROUPED, :y)); mod=@__MODULE__, total_groups=())
    # The legacy spelling warns its deprecation at BRMI construction; check
    # only the omission warning here.
    reference_brmi = legacy_lesion(drop_column(GROUPED, :y))
    reference = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        reference_brmi; mod=@__MODULE__, total_groups=())
    @test !haskey(omitted.data, :y)
    @test omitted.model.observations == reference.model.observations
    @test BRM.stan_code(omitted) == BRM.stan_code(reference)
end

@testset "groups bindings — RK log-density parity" begin
    old_backend, old_problem = consumer_problem(legacy_events(GROUPED))
    new_backend, new_problem = consumer_problem(bound_events(GROUPED))
    @test coordinate_names(new_backend.model.layout) ==
          coordinate_names(old_backend.model.layout)
    n = LogDensityProblems.dimension(new_problem)
    for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
        old_value, old_gradient = LogDensityProblems.logdensity_and_gradient(old_problem, u)
        new_value, new_gradient = LogDensityProblems.logdensity_and_gradient(new_problem, u)
        @test isfinite(new_value)
        @test new_value == old_value
        @test new_gradient == old_gradient
    end
end

@testset "the bindings print with the cell" begin
    shown = sprint(show, bound_events(GROUPED))
    @test occursin("events = groups(event_subject => subject)", shown)
    @test occursin("obs = groups(obs_subject => subject)", shown)
    @test occursin("y[obs[i]] ~ normal(loc[i], sigma)", shown)
end

@testset "grouping fails at the formula boundary" begin
    # refused: a binding no cell reads leaves a vector ungrouped wherever the
    # author expected it to apply (snag plate-body-reads-c69f8a88).
    unread = :(@brm GROUPED begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        events = groups(event_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        end
    end)
    twice = :(@brm GROUPED begin
        obs = groups(obs_subject => subject)
        obs = groups(event_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = obs_time[obs[i]]
        end
    end)
    unlabelled = :(@brm GROUPED begin
        obs = groups(obs_subject)
        @plate for i in eachindex(log_CL)
            loc[i] = obs_time[obs[i]]
        end
    end)
    self_grouped = :(@brm GROUPED begin
        own = groups(subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = weight[own[i]]
        end
    end)
    # refused: `obs[i]` is the cell's indices, not a value; read a vector at
    # them.
    bare = :(@brm GROUPED begin
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = obs[i]
        end
    end)
    # refused: the retired spellings (todo 1nfukuv, guiding principle 1mwjios)
    # name the binding that replaces them.
    statement = :(@brm GROUPED begin
        ragged(obs_time, obs_subject)
        @plate for i in eachindex(log_CL)
            loc[i] = obs_time[i]
        end
    end)
    inbody = :(@brm GROUPED begin
        @plate for i in eachindex(log_CL)
            loc[i] = ragged(obs_time, obs_subject)[i]
        end
    end)
    @test_throws "no cell reads `events`" macroexpand(@__MODULE__, unread)
    @test_throws "bound twice" macroexpand(@__MODULE__, twice)
    @test_throws "rows = groups(g => labels)" macroexpand(@__MODULE__, unlabelled)
    @test_throws "matches `subject` against itself" macroexpand(@__MODULE__, self_grouped)
    @test_throws "read a vector at them, `x[obs[i]]`" macroexpand(@__MODULE__, bare)
    for retired in (statement, inbody)
        @test_throws "are retired" macroexpand(@__MODULE__, retired)
        @test_throws "rows = groups(group => labels)" macroexpand(@__MODULE__, retired)
    end
end

@testset "a per-subject LP without a random effect reads as x[i]" begin
    direct = SBBRMI(direct_slope(GROUPED); mod=@__MODULE__, total_groups=())
    @test !haskey(direct.data, :slope)
    @test StanBlocks.stanc_check(BRM.stan_code(direct); warn_pedantic=false).ok
    old_stan = consumer_stan(wrapped_slope(GROUPED), "plate_grouping_wrapped_slope")
    new_stan = consumer_stan(direct_slope(GROUPED), "plate_grouping_direct_slope")
    @test BridgeStan.param_unc_names(new_stan.model) ==
          BridgeStan.param_unc_names(old_stan.model)
    n = BridgeStan.param_unc_num(new_stan.model)
    for u in (zeros(n), collect(range(-0.4, 0.4; length=n)))
        old_value, old_gradient = BridgeStan.log_density_gradient(
            old_stan.model, u; propto=false, jacobian=true)
        new_value, new_gradient = BridgeStan.log_density_gradient(
            new_stan.model, u; propto=false, jacobian=true)
        @test isfinite(new_value)
        @test new_value ≈ old_value atol=1e-12 rtol=1e-12
        @test new_gradient ≈ old_gradient atol=1e-12 rtol=1e-12
    end

    old_backend, old_problem = consumer_problem(wrapped_slope(GROUPED))
    new_backend, new_problem = consumer_problem(direct_slope(GROUPED))
    @test coordinate_names(new_backend.model.layout) ==
          coordinate_names(old_backend.model.layout)
    m = LogDensityProblems.dimension(new_problem)
    for u in (zeros(m), collect(range(-0.4, 0.4; length=m)))
        old_value, old_gradient = LogDensityProblems.logdensity_and_gradient(old_problem, u)
        new_value, new_gradient = LogDensityProblems.logdensity_and_gradient(new_problem, u)
        @test new_value ≈ old_value atol=1e-12 rtol=1e-12
        @test new_gradient ≈ old_gradient atol=1e-12 rtol=1e-12
    end

    @test_throws "has 7 rows for 3 subjects" SBBRMI(
        event_frame_slope(GROUPED); mod=@__MODULE__, total_groups=())
    @test_throws "has 7 rows for 3 subjects" consumer_problem(event_frame_slope(GROUPED))
end
