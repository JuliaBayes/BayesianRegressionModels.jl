# test/plate_grouping.jl — a flat secondary axis declared once at model scope.
#
# Run: julia --startup-file=no --depwarn=yes --project=test test/plate_grouping.jl
# (without `--depwarn=yes`, `@test_deprecated` only evaluates its expression).
#
# A model-scope `ragged(x, group)` statement declares how `@plate for` cells
# read the flat data column or formula name `x`; the cell then reads plain
# `x[i]`. The rewrite produces the same kernel IR as the deprecated in-body
# `ragged(x, group)[i]` read, so the Stan program, the RK log density and the
# descriptor coordinates must be identical between the two spellings.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Subject frame (unsorted labels), a flat interleaved event frame, a flat
# observation frame and a finer lesion grouping on the event frame.
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

inbody_events(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    @plate for i in eachindex(log_CL)
        dose = sum(ragged(event_amount, event_subject)[i] .*
                   exp.(ragged(log_F, event_subject)[i]))
        loc[i] = 1e-3 * dose .* exp.(-exp(log_CL[i]) .* ragged(obs_time, obs_subject)[i])
        ragged(y, obs_subject)[i] ~ normal(loc[i], sigma)
    end
end

declared_events(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    ragged(event_amount, event_subject)
    ragged(log_F, event_subject)
    ragged(obs_time, obs_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        dose = sum(event_amount[i] .* exp.(log_F[i]))
        loc[i] = 1e-3 * dose .* exp.(-exp(log_CL[i]) .* obs_time[i])
        y[i] ~ normal(loc[i], sigma)
    end
end

# The declarations may follow the cell that reads them.
declared_after(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    @plate for i in eachindex(log_CL)
        dose = sum(event_amount[i] .* exp.(log_F[i]))
        loc[i] = 1e-3 * dose .* exp.(-exp(log_CL[i]) .* obs_time[i])
        y[i] ~ normal(loc[i], sigma)
    end
    ragged(y, obs_subject)
    ragged(obs_time, obs_subject)
    ragged(log_F, event_subject)
    ragged(event_amount, event_subject)
end

# An event-axis LP with a random effect finer than the subject, plus a
# censored in-cell observation whose bound is read through a keyword.
inbody_lesion(data) = @brm data begin
    sigma ~ Exponential(1)
    log_kg ~ 0 + (1 | tgl | event_lesion)
    log_CL ~ 1 + weight + (1 | p | subject)
    @plate for i in eachindex(log_CL)
        growth = sum(exp.(ragged(log_kg, event_subject)[i]))
        location[i] = growth .* exp.(-exp(log_CL[i]) .* ragged(obs_time, obs_subject)[i])
        ragged(y, obs_subject)[i] ~ censored(
            Normal(location[i], sigma); lower=ragged(obs_limit, obs_subject)[i])
    end
end

declared_lesion(data) = @brm data begin
    sigma ~ Exponential(1)
    log_kg ~ 0 + (1 | tgl | event_lesion)
    ragged(log_kg, event_subject)
    log_CL ~ 1 + weight + (1 | p | subject)
    ragged(obs_time, obs_subject)
    ragged(obs_limit, obs_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        growth = sum(exp.(log_kg[i]))
        location[i] = growth .* exp.(-exp(log_CL[i]) .* obs_time[i])
        y[i] ~ censored(Normal(location[i], sigma); lower=obs_limit[i])
    end
end

# A per-subject LP without a random effect rides on the subject frame the
# ranef LP defines. The old spelling grouped it by its own subject column into
# one-element vectors and took `[1]`.
wrapped_slope(data) = @brm data begin
    sigma ~ Exponential(1)
    slope ~ 1 + indication
    log_CL ~ 1 + weight + (1 | p | subject)
    ragged(obs_time, obs_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ ragged(slope, subject)[i][1] .* obs_time[i]
        y[i] ~ normal(loc[i], sigma)
    end
end

direct_slope(data) = @brm data begin
    sigma ~ Exponential(1)
    slope ~ 1 + indication
    log_CL ~ 1 + weight + (1 | p | subject)
    ragged(obs_time, obs_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ slope[i] .* obs_time[i]
        y[i] ~ normal(loc[i], sigma)
    end
end

# refused: an LP without a random effect is read on the subject frame, so an
# event-frame LP read as `x[i]` would silently misalign; it needs a grouping
# (snag plate-body-reads-c69f8a88).
event_frame_slope(data) = @brm data begin
    sigma ~ Exponential(1)
    slope ~ 1 + event_x
    log_CL ~ 1 + weight + (1 | p | subject)
    ragged(obs_time, obs_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) .+ sum(slope[i]) .* obs_time[i]
        y[i] ~ normal(loc[i], sigma)
    end
end

stan_of(brmi) = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
drop_column(data, column) = NamedTuple(k => v for (k, v) in pairs(data) if k !== column)

@testset "model-scope ragged(x, group) lowers like the in-body read" begin
    for (label, old, new) in (
        ("data columns, event-axis LP, in-cell response",
         inbody_events, declared_events),
        ("declarations after the cell", inbody_events, declared_after),
        ("finer random effect, censored keyword bound", inbody_lesion, declared_lesion),
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

@testset "model-scope grouping keeps the descriptor outputs" begin
    for (old, new, lp, cell) in (
        (inbody_events, declared_events, :log_F, :loc),
        (inbody_lesion, declared_lesion, :log_kg, :location),
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

@testset "model-scope grouping with an omitted in-cell response" begin
    omitted = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        declared_lesion(drop_column(GROUPED, :y)); mod=@__MODULE__, total_groups=())
    # The deprecated in-body reference warns at BRMI construction; check only
    # the omission warning here.
    reference_brmi = inbody_lesion(drop_column(GROUPED, :y))
    reference = @test_logs (:warn, r"`y` bind\(s\) no data column") SBBRMI(
        reference_brmi; mod=@__MODULE__, total_groups=())
    @test !haskey(omitted.data, :y)
    @test omitted.model.observations == reference.model.observations
    @test BRM.stan_code(omitted) == BRM.stan_code(reference)
end

@testset "model-scope grouping — RK log-density parity" begin
    old_backend, old_problem = consumer_problem(inbody_events(GROUPED))
    new_backend, new_problem = consumer_problem(declared_events(GROUPED))
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

@testset "in-body ragged(x, group)[i] is deprecated, the declaration is not" begin
    @test_deprecated r"ragged\(x, group\)\[i\].*deprecated.*model scope" inbody_events(GROUPED)
    @test_logs declared_events(GROUPED)
end

@testset "model-scope grouping fails at the formula boundary" begin
    # refused: a declaration no cell reads leaves `x` ungrouped wherever the
    # author expected it to apply (snag plate-body-reads-c69f8a88).
    unread = :(@brm GROUPED begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + (1 | p | subject)
        ragged(obs_time, obs_subject)
        ragged(event_amount, event_subject)
        @plate for i in eachindex(log_CL)
            loc[i] = exp.(-exp(log_CL[i]) .* obs_time[i])
        end
    end)
    twice = :(@brm GROUPED begin
        ragged(obs_time, obs_subject)
        ragged(obs_time, event_subject)
        @plate for i in eachindex(log_CL)
            loc[i] = obs_time[i]
        end
    end)
    malformed = :(@brm GROUPED begin
        ragged(obs_time, obs_subject, event_subject)
        @plate for i in eachindex(log_CL)
            loc[i] = obs_time[i]
        end
    end)
    self_grouped = :(@brm GROUPED begin
        ragged(obs_subject, obs_subject)
        @plate for i in eachindex(log_CL)
            loc[i] = obs_subject[i]
        end
    end)
    # refused: one name, one grouping spelling — the declaration already says
    # how the cell reads it (snag plate-body-reads-c69f8a88).
    mixed = :(@brm GROUPED begin
        ragged(obs_time, obs_subject)
        @plate for i in eachindex(log_CL)
            loc[i] = ragged(obs_time, obs_subject)[i]
        end
    end)
    @test_throws "no cell reads `event_amount[i]`" macroexpand(@__MODULE__, unread)
    @test_throws "declared twice" macroexpand(@__MODULE__, twice)
    @test_throws "model-scope plate grouping is `ragged(x, group)`" macroexpand(@__MODULE__, malformed)
    @test_throws "groups `obs_subject` by itself" macroexpand(@__MODULE__, self_grouped)
    @test_throws "read it in the cell as `obs_time[i]`" macroexpand(@__MODULE__, mixed)
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
