# test/dose_axis_operation_gather.jl — a predictor read only at the dose
# operations of an operation stream, declared on the dose axis.
#
# Run: julia --startup-file=no --project=test test/dose_axis_operation_gather.jl
#
# A kernel that walks a compact operation stream (dose and read operations
# interleaved on one flat axis) reads bioavailability only at the doses. Declared
# over the dose axis, `log_F` never sees the read operations: no fill value and
# no `missing`, and every data-derived quantity (here the HSGP interval) comes
# from the doses. Each operation names its dose by its index in the dose vectors
# (`0` on a read). The cell rebases those to positions among the subject's
# doses, `something.(indexin(op_dose[events[i]], doses[i]), 0)`, and reads
# `vcat(0.0, log_F[doses[i]])` one past them, so a read, and every operation of
# a subject without doses, picks the prepended 0 (snag predictor-level-3cf5667e,
# decision 1cw48pq).
#
# The reference is the same model on the operation axis, its read rows filled
# with the mean dose log: interior, so the HSGP interval is unchanged, and
# multiplied by a zero amount, so the value never matters. The two must give the
# same Stan and RK log density and gradient, and replay alike.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Subject "d" is a placebo subject: read operations only.
function operation_stream(; subject, weight, op_subject, op_is_dose, op_amount,
                          op_bottle, obs_subject, obs_time, y, fill_value=nothing)
    doses = log.(op_amount[op_is_dose] ./ 100)
    fill_value = something(fill_value, sum(doses) / length(doses))
    log_dose = similar(op_amount)
    log_dose[op_is_dose] .= doses
    log_dose[.!op_is_dose] .= fill_value
    base = (; subject, weight, op_subject, op_amount, obs_subject, obs_time, y)
    filled = merge(base, (; log_dose, bottle=op_bottle))
    dose_axis = merge(base, (;
        dose_subject=op_subject[op_is_dose], log_dose=doses,
        bottle=op_bottle[op_is_dose],
        op_dose=cumsum(op_is_dose) .* op_is_dose))
    (; filled, dose_axis, fill_value)
end

const TRAINING = operation_stream(;
    subject=["a", "b", "c", "d"], weight=[60.0, 75.0, 90.0, 80.0],
    op_subject=["a", "a", "b", "a", "c", "c", "b", "a", "d", "d"],
    op_is_dose=[true, false, true, true, false, true, false, false, false, false],
    op_amount=[100.0, 0.0, 50.0, 200.0, 0.0, 400.0, 0.0, 0.0, 0.0, 0.0],
    op_bottle=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    obs_subject=["a", "c", "b", "a", "b", "c", "d", "d"],
    obs_time=[0.5, 0.25, 1.5, 1.0, 3.0, 2.0, 1.0, 2.0],
    y=[0.81, 0.76, 0.52, 0.49, 0.21, 0.33, 0.01, 0.02])

filled_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 0 + bottle + log_dose + hsgp(log_dose; k=3)
    log_CL ~ 1 + weight + (1 | p | subject)
    events = groups(op_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        dose = sum(op_amount[events[i]] .* exp.(log_F[events[i]]))
        loc[i] = 1e-3 * dose .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]]) .+ 1e-3
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

dose_axis_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_F ~ 0 + bottle + log_dose + hsgp(log_dose; k=3)
    log_CL ~ 1 + weight + (1 | p | subject)
    events = groups(op_subject => subject)
    doses = groups(dose_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        F = vcat(0.0, log_F[doses[i]])
        slot = something.(indexin(op_dose[events[i]], doses[i]), 0) .+ 1
        dose = sum(op_amount[events[i]] .* exp.(F[slot]))
        loc[i] = 1e-3 * dose .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]]) .+ 1e-3
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

sb_of(brmi) = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
rk_density(backend, u) = LogDensityProblems.logdensity_and_gradient(
    rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
probe_points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

@testset "each operation's dose, rebased to its subject's doses" begin
    @test TRAINING.dose_axis.op_dose == [1, 0, 2, 3, 0, 4, 0, 0, 0, 0]
    @test TRAINING.dose_axis.dose_subject == ["a", "b", "a", "c"]
    # Subjects a, b, c, d: positions among their own doses, 0 off the doses.
    @test sb_of(dose_axis_model(TRAINING.dose_axis)).data[:kernel_loc_op_dose_ragged] ==
          [[1, 0, 2, 0], [1, 0], [0, 1], [0, 0]]
end

@testset "Stan: the dose axis equals the filled operation axis" begin
    dose_sb = sb_of(dose_axis_model(TRAINING.dose_axis))
    @test StanBlocks.stanc_check(BRM.stan_code(dose_sb); warn_pedantic=false).ok
    # The basis is evaluated at the doses only.
    @test size(dose_sb.data[:PHI_hsgp_log_dose], 1) == 4
    filled = consumer_stan(filled_model(TRAINING.filled), "dose_axis_filled")
    dose_axis = consumer_stan(dose_axis_model(TRAINING.dose_axis), "dose_axis_gather")
    names = BridgeStan.param_unc_names(filled.model)
    @test BridgeStan.param_unc_names(dose_axis.model) == names
    for u in probe_points(length(names))
        a_grad, b_grad = similar(u), similar(u)
        a, _ = BridgeStan.log_density_gradient!(filled.model, u, a_grad; propto=false)
        b, _ = BridgeStan.log_density_gradient!(dose_axis.model, u, b_grad; propto=false)
        @test isfinite(a)
        @test b ≈ a atol=1e-12 rtol=1e-12
        @test b_grad ≈ a_grad atol=1e-12 rtol=1e-12
    end
end

@testset "RK: the dose axis equals the filled operation axis" begin
    filled, filled_problem = consumer_problem(filled_model(TRAINING.filled))
    dose_axis, dose_problem = consumer_problem(dose_axis_model(TRAINING.dose_axis))
    @test coordinate_names(dose_axis.model.layout) == coordinate_names(filled.model.layout)
    for u in probe_points(LogDensityProblems.dimension(filled_problem))
        a, a_grad = LogDensityProblems.logdensity_and_gradient(filled_problem, u)
        b, b_grad = LogDensityProblems.logdensity_and_gradient(dose_problem, u)
        @test isfinite(a)
        @test b ≈ a atol=1e-12 rtol=1e-12
        @test b_grad ≈ a_grad atol=1e-12 rtol=1e-12
    end
end

# The fitted subjects on a new schedule: one dose for "a", none for "d". The
# filled replay fills its read rows with the training fill, inside the fitted
# basis interval.
const FROZEN = operation_stream(;
    subject=["a", "b", "c", "d"], weight=[60.0, 75.0, 90.0, 80.0],
    op_subject=["a", "a", "b", "b", "c", "d"],
    op_is_dose=[true, false, true, false, true, false],
    op_amount=[200.0, 0.0, 100.0, 0.0, 300.0, 0.0],
    op_bottle=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    obs_subject=["a", "b", "c", "d"], obs_time=[1.0, 2.0, 0.5, 1.0],
    y=[0.4, 0.3, 0.6, 0.01], fill_value=TRAINING.fill_value)

# New subjects, one of them without doses.
const NEW_SUBJECTS = operation_stream(;
    subject=["e", "f"], weight=[70.0, 65.0],
    op_subject=["e", "e", "e", "f"], op_is_dose=[true, false, true, false],
    op_amount=[100.0, 0.0, 300.0, 0.0], op_bottle=[0.0, 0.0, 1.0, 0.0],
    obs_subject=["e", "f"], obs_time=[1.0, 1.0], y=[0.1, 0.0],
    fill_value=TRAINING.fill_value)

@testset "replay: frozen onto a new schedule, and onto new subjects" begin
    filled_sb = sb_of(filled_model(TRAINING.filled))
    dose_sb = sb_of(dose_axis_model(TRAINING.dose_axis))
    filled_replay = reprocess(filled_sb, FROZEN.filled)
    dose_replay = reprocess(dose_sb, FROZEN.dose_axis)
    # The frozen basis at the replay's doses is the filled replay's at its dose rows.
    @test dose_replay.data[:PHI_hsgp_log_dose] ≈
          filled_replay.data[:PHI_hsgp_log_dose][[1, 3, 5], :] atol=1e-14
    @test BRM.stan_code(dose_replay) == BRM.stan_code(dose_sb)
    resampled = reprocess(dose_sb, NEW_SUBJECTS.dose_axis; resample_groups=[:subject])
    @test size(resampled.data[:PHI_hsgp_log_dose], 1) == 2

    filled_rk = RKBRMI(filled_model(TRAINING.filled))
    dose_rk = RKBRMI(dose_axis_model(TRAINING.dose_axis))
    for u in probe_points(dose_rk.model.layout.total)
        a, a_grad = rk_density(reprocess(filled_rk, FROZEN.filled), u)
        b, b_grad = rk_density(reprocess(dose_rk, FROZEN.dose_axis), u)
        @test isfinite(a)
        @test b ≈ a atol=1e-12 rtol=1e-12
        @test b_grad ≈ a_grad atol=1e-12 rtol=1e-12
    end
    @test reprocess(dose_rk, NEW_SUBJECTS.dose_axis; resample_groups=[:subject]) isa RKBRMI
end
