# test/rk_fitted_state.jl — a fitted RKBRMI's frozen preprocessing exported
# without its training rows, and designs bound from it alone (snag
# `no-data-free-exp-fec940cb`; the user picked "keep the fits in the RKPPL
# program and freeze them by value" on design decision `0f3hrtg`; RKPPL's
# `frozen_values` / `bind_data(...; frozen)` from ReactiveKernels snag
# `rkppl-bind-froze-da223711`).
#
# `brm_fitted_state(rk)` holds the values the emitted program fits from its
# fit-role inputs, the planner's Julia-side fits and the `constant(...)`
# values. Oracle: it holds no array of training-row length and round-trips
# through `Serialization`; planning a design from it alone emits the fit's
# program; and `RKBRMI(build(design), state)` binds the inputs, gives the graph
# values and the density that `reprocess(rk, design)` gives on the training
# fit, without binding a training column. Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/rk_fitted_state.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, Serialization, Statistics

module FittedStateModels
using BayesianRegressionModels, Distributions, Statistics
zscaled(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, sigma)
end
centered(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + center(x)
    y ~ Normal(mu, sigma)
end
standardized(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + standardize(x)
    y ~ Normal(mu, sigma)
end
zscaled_log(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(log(x))
    y ~ Normal(mu, sigma)
end
factors(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + g + gs + mo(r)
    y ~ Normal(mu, sigma)
end
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    xbar = constant(mean(x))
    mu ~ 1 + zscale(x) + (1 + z | subject)
    effect(mu, Intercept) ~ Normal(xbar, 1)
    y ~ Normal(mu, sigma)
end
smooth(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5)
    y ~ Normal(mu, sigma)
end
smooth_rows(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5, rows=keep)
    y ~ Normal(mu, sigma)
end
smooth_2d(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x, z; k=(3, 3))
    y ~ Normal(mu, sigma)
end
smooth_orthogonal(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + hsgp(x; k=5, orthogonal_to=:linear)
    y ~ Normal(mu, sigma)
end
smooth_clamped(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5, extrapolate=:clamp)
    y ~ Normal(mu, sigma)
end
thin_plate(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + s(x)
    y ~ Normal(mu, sigma)
end
tensor(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + t2(x, z)
    y ~ Normal(mu, sigma)
end
shrunk(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + gs + (1 | subject)
    effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
    y ~ Normal(mu, sigma)
end
completed(d) = @brm d begin
    sigma ~ Exponential(1)
    mi(xm) ~ LogNormal(0.5, 0.6)
    mu ~ 1 + standardize(xm)
    y ~ Normal(mu, sigma)
end
end
const FM = FittedStateModels

const N = 24
const TRAIN = let rng = Random.Xoshiro(7)
    x = 1 .+ 4 .* rand(rng, N)
    subject = repeat(1:6, inner=4)
    g = categorical(repeat(["a", "b", "c"], 8))
    gs = repeat(["lo", "mid", "hi"], outer=8)
    r = repeat(1:4, outer=6)
    z = randn(rng, N)
    keep = repeat([1.0, 1.0, 0.0], 8)
    xm = Union{Missing,Float64}[i % 5 == 0 ? missing : exp(0.5 + 0.3 * v)
        for (i, v) in enumerate(randn(rng, N))]
    (; y=randn(rng, N), x, z, subject, g, gs, r, keep, xm)
end
const SUPPORTED = (FM.zscaled, FM.centered, FM.standardized, FM.zscaled_log,
    FM.factors, FM.grouped, FM.smooth, FM.smooth_rows, FM.smooth_2d,
    FM.smooth_orthogonal, FM.smooth_clamped, FM.tensor, FM.shrunk)
rows(data, index) = map(column -> column[index], data)
without(data, name) = NamedTuple{Tuple(k for k in keys(data) if k !== name)}(data)

# The design inputs a replay binds, without the fit-role inputs: the training
# columns on `reprocess`, which a data-free replay replaces with `state.fitted`.
function design_inputs(plan, emitted, state)
    columns = BRM._rk_source_data_columns(plan, emitted)
    Dict(name => value for (name, value) in columns if !(name in state.fit_inputs))
end

function check_planning(build, state, rk, design; resample_groups=())
    plan, emitted = BRM._rk_fitted_replay_plan(build(design), design, state; resample_groups)
    replay = reprocess(rk, design; resample_groups)
    @test BRM._rk_program_statements(emitted) == state.program
    @test isequal(design_inputs(plan, emitted, state),
        design_inputs(replay.plan, BRM._rk_retained_program(replay), state))
end

@stestset "the fitted values replace the training columns" begin
    for build in SUPPORTED
        rk = RKBRMI(build(TRAIN))
        state = brm_fitted_state(rk)
        # Something is fitted: in the program (an anchor, a level set read
        # through a fit-role input) or by the planner (a factor's levels).
        @test !isempty(state.fitted) || !isempty(state.fits)
        @test isempty(state.fitted) == isempty(state.fit_inputs)
        # No training rows travel: every fitted value is a summary, never a
        # column with one entry per training row.
        for value in Iterators.flatten((values(state.fitted.values), values(state.fits)))
            @test !(value isa AbstractArray) || length(value) != N
            @test !any(column -> isequal(value, column), values(TRAIN))
        end
        @test state.coordinates == BRM._rk_layout_coordinate_names(rk)
        io = IOBuffer()
        serialize(io, state)
        back = deserialize(seekstart(io))
        @test isequal(back.fitted, state.fitted)
        @test isequal(back.fits, state.fits)
        @test back.program == state.program
    end
end

@stestset "a fitted value is the training data's" begin
    state = brm_fitted_state(RKBRMI(FM.zscaled(TRAIN)))
    @test state.fitted.values.x_fit_mean ≈ mean(TRAIN.x)
    @test state.fitted.values.x_fit_sd ≈ std(TRAIN.x)
    state = brm_fitted_state(RKBRMI(FM.grouped(TRAIN)))
    @test state.fitted.values.subject_fit_levels == 1:6
    @test state.constants[:xbar] ≈ mean(TRAIN.x)
    state = brm_fitted_state(RKBRMI(FM.factors(TRAIN)))
    @test collect(state.fits[(:levels, :r)]) == 1:4
    # An automatic HSGP interval: the axis mean and the boundary-scaled
    # largest deviation from it (`c = 1.5` by default).
    state = brm_fitted_state(RKBRMI(FM.smooth(TRAIN)))
    center = mean(TRAIN.x)
    @test state.fitted.values.hsgp_x_fit[1] ≈ center
    @test state.fitted.values.hsgp_x_fit[2] ≈ 1.5 * maximum(abs.(TRAIN.x .- center))
end

@stestset "a design plans from the fitted state alone" begin
    for build in SUPPORTED
        rk = RKBRMI(build(TRAIN))
        state = brm_fitted_state(rk)
        check_planning(build, state, rk, rows(TRAIN, 3:9))
        check_planning(build, state, rk, rows(TRAIN, [5]))
        check_planning(build, state, rk, without(rows(TRAIN, 2:6), :y))
    end
    rk = RKBRMI(FM.grouped(TRAIN))
    state = brm_fitted_state(rk)
    newcomers = merge(rows(TRAIN, 1:8), (; subject=repeat([101, 102], inner=4)))
    check_planning(FM.grouped, state, rk, newcomers; resample_groups=[:subject])
end

@stestset "a fitted state keeps reprocess's refusals" begin
    rk = RKBRMI(FM.factors(TRAIN))
    state = brm_fitted_state(rk)
    # refused: an unseen level has no fitted effect (brm-use reactivekernels.md,
    # "Training levels only").
    unseen = merge(rows(TRAIN, 1:3), (; gs=["lo", "mid", "new"]))
    @test_throws "new" BRM._rk_fitted_replay_plan(FM.factors(unseen), unseen, state)
    # refused: a state of another model holds other fits, so the draws would
    # not mean the same thing.
    other = brm_fitted_state(RKBRMI(FM.zscaled(TRAIN)))
    @test_throws "exported from another model" BRM._rk_fitted_replay_plan(
        FM.centered(TRAIN), TRAIN, other)
end

# An authored value of `rk`'s graph on its own bound data.
function graph_value(rk, want, u)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(want,), bound)
    value = Base.invokelatest(query, u)
    value isa Tuple ? only(value) : value
end
point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
density(rk, u) = LogDensityProblems.logdensity(
    rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
absent(v) = occursin("Absent", string(nameof(typeof(v))))

# The names a data-free binding of `rk`'s program holds absent: the fit-role
# inputs it drops, and the definitions fitted from those alone that no frozen
# value stands in for (`mu_gs_lvl_2_indicator`, read only by its frozen
# variance `mu_gs_lvl_2_indicator_var`).
function absent_names(rk, state, dropped, data)
    names = Set{Symbol}(dropped)
    for statement in BRM._rk_retained_program(rk).main.args
        Meta.isexpr(statement, :(=), 2) && statement.args[1] isa Symbol || continue
        reads = intersect(BRM._rk_source_symbols!(Set{Symbol}(), statement.args[2]), data)
        !isempty(reads) && issubset(reads, names) && push!(names, statement.args[1])
    end
    setdiff(names, keys(state.fitted.values))
end

# Data-free inputs `free` against a training replay's `replay` of the fit `rk`:
# the same values, except that a fit-role input and what is fitted from it
# alone are never bound (or bound absent), and a value RK preparation folds is
# bound frozen in place of the replay's graph statement. A resampled grouping's
# fit-role input holds the design's own values on both.
function check_inputs(free, replay, state, rk; resample_groups=())
    dropped = Set(input for input in state.fit_inputs
        if !(state.fit_twins[input] in resample_groups))
    unbound = absent_names(rk, state, dropped, Set{Symbol}(keys(replay)))
    for name in keys(free)
        if name in unbound
            @test absent(free[name])
        elseif haskey(replay, name)
            @test isequal(free[name], replay[name])
        else
            @test haskey(state.fitted.values, name)
        end
    end
    for name in keys(replay)
        name in dropped || @test haskey(free, name)
    end
end

function check_data_free(build, state, rk, design; resample_groups=(), want=:mu)
    free = RKBRMI(build(design), state; resample_groups)
    replay = reprocess(rk, design; resample_groups)
    @test free.fitted.state === state
    check_inputs(rk_bound_data(free), rk_bound_data(replay), state, rk; resample_groups)
    if isempty(resample_groups)
        @test BRM._rk_layout_coordinate_names(free) == state.coordinates
        u = point(replay)
        @test isequal(graph_value(free, want, u), graph_value(replay, want, u))
        hasproperty(design, :y) && @test density(free, u) == density(replay, u)
    end
    free
end

@stestset "a design binds from the fitted state alone" begin
    for build in SUPPORTED
        rk = RKBRMI(build(TRAIN))
        state = brm_fitted_state(rk)
        check_data_free(build, state, rk, rows(TRAIN, 3:9))
        check_data_free(build, state, rk, rows(TRAIN, [5]))
        free = check_data_free(build, state, rk, without(rows(TRAIN, 2:6), :y))
        # A data-free backend replays further designs from its state.
        again = reprocess(free, rows(TRAIN, 10:14))
        check_inputs(rk_bound_data(again), rk_bound_data(reprocess(rk, rows(TRAIN, 10:14))),
            state, rk)
        # The state serialized and read back binds the same design.
        io = IOBuffer()
        serialize(io, state)
        back = deserialize(seekstart(io))
        @test isequal(rk_bound_data(RKBRMI(build(rows(TRAIN, 3:9)), back)),
            rk_bound_data(RKBRMI(build(rows(TRAIN, 3:9)), state)))
    end
end

@stestset "new subjects bind from the fitted state alone" begin
    # A resampled grouping refits its levels from the design; every other
    # fitted value stays frozen.
    rk = RKBRMI(FM.grouped(TRAIN))
    state = brm_fitted_state(rk)
    newcomers = merge(rows(TRAIN, 1:8), (; subject=repeat([101, 102], inner=4)))
    free = check_data_free(FM.grouped, state, rk, newcomers; resample_groups=[:subject])
    @test rk_bound_data(free).subject_fit_levels == [101, 102]
    @test rk_bound_data(free).x_fit_mean == state.fitted.values.x_fit_mean
end

@stestset "a value RK preparation folds is frozen too" begin
    # A completed covariate's anchors are read only by the parameter-dependent
    # completion, so RKPPL folds them at preparation, not at binding. The
    # design keeps every missing entry, as on `reprocess`.
    rk = RKBRMI(FM.completed(TRAIN))
    state = brm_fitted_state(rk)
    @test !isempty(state.fitted)
    check_data_free(FM.completed, state, rk, rows(TRAIN, [1, 5, 10, 15, 20, 21]))
end

@stestset "a data-free binder binds what the training binder binds" begin
    rk = RKBRMI(FM.grouped(TRAIN))
    state = brm_fitted_state(rk)
    free = RKBRMI(FM.grouped(rows(TRAIN, 1:4)), state)
    for groups in ((), (:subject,))
        binder, trained = rk_design_binder(free; resample_groups=groups),
            rk_design_binder(rk; resample_groups=groups)
        @test !any(name -> name in state.fit_inputs, keys(binder.fitted))
        # A resampled grouping's levels are the design's, never frozen.
        @test (:subject in groups) != haskey(binder.fitted, :subject_fit_levels)
        # The second and third designs are of the first one's kind: bound into
        # its lowering without planning.
        for design in (rows(TRAIN, 3:9), rows(TRAIN, 11:17), rows(TRAIN, 13:16))
            check_inputs(binder(design), trained(design), state, rk; resample_groups=groups)
        end
    end
end

@stestset "a thin-plate basis cannot leave its training values" begin
    # refused: `s(x)` keeps every training value of `x` as a knot, so no
    # data-free state can carry it (snag `no-data-free-exp-fec940cb`).
    @test_throws "thin-plate" brm_fitted_state(RKBRMI(FM.thin_plate(TRAIN)))
end
