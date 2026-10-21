# test/fitted_constants.jl — `name = constant(expr)`: a scalar calibrated on
# the data when the model is built and frozen at that value on replay (snag
# `bruno-fit-time-d-dba90310`, user design decision `00kjor3`).
#
# A constant is evaluated once on whole columns and is then an ordinary scalar
# data field: Stan data on SBBRMI, a number in the RKBRMI graph. Replay keeps
# the fitted value, so `reprocess`, `reprocess(...; resample_groups)` and
# `generative_plan(plan, new_df)` need no inputs for it, while
# `reprocess(...; freeze_constants=false)` evaluates it again on the new data.
#
# Run: julia --startup-file=no --project=test test/fitted_constants.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics: mean, std, var
using LogExpFunctions: logistic
const stan_code = BRM.stan_code
const stan_data = BRM.stan_data

const PANEL = (;
    conc=[0.4, 1.1, 2.3, 0.7, 1.9, 3.1, 0.5, 1.4, 2.8, 0.9, 1.6, 2.2],
    x=[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, -0.9, -0.3, 0.1, 0.5, 0.8, 1.2],
    subject=repeat(1:4, inner=3),
    y=[0.3, 0.9, 1.6, 0.4, 1.2, 2.0, 0.1, 0.8, 1.7, 0.6, 1.1, 1.5],
)
# A Julia value a constant reads from the builder's scope.
const OFFSET = 0.5

const LOC = mean(log.(PANEL.conc))
const SCALE = std(log.(PANEL.conc))
const V = var(PANEL.y)
const CAP = maximum(log.(PANEL.conc)) + OFFSET

const CALIBRATED = @brm begin
    ec50_loc = constant(mean(log.(conc)))
    ec50_scale = constant(std(log.(conc)))
    V = constant(var(y))
    log_ec50 ~ Normal(ec50_loc, ec50_scale)
    mu ~ 1 + x + (1 | p | subject)
    r2 ~ Beta(1.0, 1.0)
    sigma = sqrt((1 - r2) * V)
    m = mu + log_ec50
    y ~ Normal(m, sigma)
end
# The same model with the three scalars supplied as data fields.
const SUPPLIED = @brm begin
    log_ec50 ~ Normal(ec50_loc, ec50_scale)
    mu ~ 1 + x + (1 | p | subject)
    r2 ~ Beta(1.0, 1.0)
    sigma = sqrt((1 - r2) * V)
    m = mu + log_ec50
    y ~ Normal(m, sigma)
end
supplied(data) = merge(data, (; ec50_loc=LOC, ec50_scale=SCALE, V))

# Replay frames: the inputs `conc` absent, and a one-row grid on which the
# constants are undefined (`std` of one value is NaN).
const NO_CONC = (; x=PANEL.x[1:6], subject=PANEL.subject[1:6], y=PANEL.y[1:6])
const ONE_ROW = (; conc=[2.0], x=[0.5], subject=[11], y=[0.2])
const ONE_ROW_NO_CONC = (; x=[0.5], subject=[11], y=[0.2])

constants(sb) = Dict(k => stan_data(sb)[k] for k in (:ec50_loc, :ec50_scale, :V))
const FITTED = Dict(:ec50_loc => LOC, :ec50_scale => SCALE, :V => V)

@stestset "a constant is the scalar data field it computes" begin
    saved = deepcopy(PANEL)
    brmi = CALIBRATED(PANEL)
    @test occursin("ec50_loc = constant(mean(log.(conc)))", sprint(show, brmi))
    sb = SBBRMI(brmi; mod=@__MODULE__)
    reference = SBBRMI(SUPPLIED(supplied(PANEL)); mod=@__MODULE__)
    @test stan_code(sb) == stan_code(reference)
    @test constants(sb) == FITTED
    @test Dict(k => e.kind for (k, e) in sb.preproc if e.kind === :constant) ==
        Dict(:ec50_loc => :constant, :ec50_scale => :constant, :V => :constant)
    @test StanBlocks.stan.compiles(sb.model)
    @test isequal(PANEL, saved)
end

@stestset "RKBRMI holds constants in prior arguments and bounds, matching Stan" begin
    builder = @brm begin
        loc = constant(mean(log.(conc)))
        scale = constant(std(log.(conc)))
        V = constant(var(y))
        cap = constant(maximum(log.(conc)) + OFFSET)
        log_ec50 ~ Normal(loc, scale; upper=cap)
        r2 ~ Beta(1.0, 1.0)
        sigma = sqrt((1 - r2) * V)
        y ~ Normal(log_ec50, sigma)
    end
    brmi = builder(PANEL)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    @test length(names) == 2
    ie, ir = coordinate(names, :log_ec50), coordinate(names, :r2)
    oracle(u) = begin
        log_ec50, r2 = CAP - exp(u[ie]), logistic(u[ir])
        sigma = sqrt((1 - r2) * V)
        logpdf(Normal(LOC, SCALE), log_ec50) + u[ie] +
            logpdf(Beta(1.0, 1.0), r2) + log(r2) + log1p(-r2) +
            sum(logpdf.(Normal(log_ec50, sigma), PANEL.y))
    end
    stan = consumer_stan(brmi, "fitted-constants-bounds")
    mapping = [names[ie] => "log_ec50", names[ir] => "r2"]
    for u in (zeros(2), [-0.4, 0.7], [0.3, -1.2])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    # Supplying the same scalars as data fields gives the same graph values.
    supplied_backend, supplied_problem = consumer_problem((@brm begin
        log_ec50 ~ Normal(loc, scale; upper=cap)
        r2 ~ Beta(1.0, 1.0)
        sigma = sqrt((1 - r2) * V)
        y ~ Normal(log_ec50, sigma)
    end)(merge(PANEL, (; loc=LOC, scale=SCALE, V, cap=CAP))))
    @test coordinate_names(supplied_backend.model.layout) == names
    for u in (zeros(2), [-0.4, 0.7])
        @test LogDensityProblems.logdensity(supplied_problem, u) ≈
            LogDensityProblems.logdensity(problem, u) atol=1e-12
    end
end

@stestset "a constant scales a population-effect prior on both backends" begin
    calibrated = @brm begin
        slope_scale = constant(2 * std(y))
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, slope_scale)
        y ~ Normal(mu, sigma)
    end
    supplied_scale = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, slope_scale)
        y ~ Normal(mu, sigma)
    end
    with_scale = merge(PANEL, (; slope_scale=2 * std(PANEL.y)))
    sb = SBBRMI(calibrated(PANEL); mod=@__MODULE__, total_groups=())
    @test stan_code(sb) == stan_code(SBBRMI(supplied_scale(with_scale); mod=@__MODULE__, total_groups=()))
    @test stan_data(sb)[:slope_scale] == 2 * std(PANEL.y)
    @test StanBlocks.stan.compiles(sb.model)
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = rk_logdensity_problem(RKBRMI(calibrated(PANEL)); ad_backend)
    rk_supplied = rk_logdensity_problem(RKBRMI(supplied_scale(with_scale)); ad_backend)
    literal_scale = eval(:(@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, $(2 * std(PANEL.y)))
        y ~ Normal(mu, sigma)
    end))
    rk_literal = rk_logdensity_problem(RKBRMI(literal_scale(PANEL)); ad_backend)
    for u in (zeros(LogDensityProblems.dimension(rk)), fill(0.3, LogDensityProblems.dimension(rk)))
        @test LogDensityProblems.logdensity(rk, u) == LogDensityProblems.logdensity(rk_supplied, u)
        @test LogDensityProblems.logdensity(rk, u) ≈ LogDensityProblems.logdensity(rk_literal, u) atol=1e-12
    end
    # A sampled scale is still refused.
    @test_throws "must be a numeric constant or a scalar data field" RKBRMI(@brm PANEL begin
        slope_scale ~ Exponential(1)
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, slope_scale)
        y ~ Normal(mu, sigma)
    end)
end

@stestset "replay keeps fitted constants without their inputs" begin
    sb = SBBRMI(CALIBRATED(PANEL); mod=@__MODULE__, total_groups=())
    frozen = reprocess(sb, NO_CONC)
    @test constants(frozen) == FITTED
    @test stan_code(frozen) == stan_code(sb)
    # A one-row grid would make `std` NaN; the fitted value is kept instead.
    for frame in (ONE_ROW, ONE_ROW_NO_CONC)
        resampled = reprocess(sb, frame; resample_groups=[:subject])
        @test constants(resampled) == FITTED
        @test count(e -> e.kind === :constant, values(resampled.preproc)) == 3
    end
end

@stestset "fresh-constant replay evaluates the constant on the new data" begin
    sb = SBBRMI(CALIBRATED(PANEL); mod=@__MODULE__, total_groups=())
    conc = PANEL.conc[1:6] .* 10
    fresh = reprocess(sb, merge(NO_CONC, (; conc)); freeze_constants=false)
    @test constants(fresh) == Dict(:ec50_loc => mean(log.(conc)),
        :ec50_scale => std(log.(conc)), :V => var(NO_CONC.y))
    @test stan_code(fresh) == stan_code(sb)
    err = try
        reprocess(sb, NO_CONC; freeze_constants=false); nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("reads `conc`", err.msg)
    err = try
        reprocess(sb, ONE_ROW; resample_groups=[:subject], freeze_constants=false); nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("one finite real number", err.msg)
    # A scalar the data supplies directly is taken from the new data as well.
    plain = SBBRMI(SUPPLIED(supplied(PANEL)); mod=@__MODULE__, total_groups=())
    replay = merge(NO_CONC, (; ec50_loc=9.0, ec50_scale=1.0, V=2.0))
    @test constants(reprocess(plain, replay; freeze_constants=false)) ==
        Dict(:ec50_loc => 9.0, :ec50_scale => 1.0, :V => 2.0)
    @test constants(reprocess(plain, replay)) == FITTED
end

@stestset "a plan rebuilt for new groups keeps fitted constants" begin
    for total_groups in (:auto, ())
        plan = generative_plan(CALIBRATED, PANEL; mod=@__MODULE__, total_groups)
        @test (total_groups === :auto) == !isempty(total_effect_blocks(plan))
        rebuilt = generative_plan(plan, ONE_ROW_NO_CONC)
        @test Dict(k => rebuilt.data[k] for k in keys(FITTED)) == FITTED
    end
end

@stestset "the descriptor reports a constant as a :constant input" begin
    descriptor = brm_descriptor(CALIBRATED, PANEL; mod=@__MODULE__)
    for name in keys(FITTED)
        input = only(i for i in descriptor.inputs if i.name === name)
        @test input.transform === :constant
        @test isnothing(input.column)
    end
    # Replay keeps the fitted value, so `conc` is not part of the replay schema.
    @test :conc ∉ brm_columns(descriptor)
    @test :y in brm_columns(descriptor)
    # Each constant is a covered component naming its expression and value.
    description = brm_description(descriptor)
    calls = filter(c -> c.callable === constant, collect(brm_description_components(description)))
    @test Set(c.provenance.owner for c in calls) == Set(keys(FITTED))
    for c in calls
        @test only(x for x in description.coverage if x.id == c.id).status === :covered
    end
    markdown = brm_description_markdown(description)
    @test occursin("operatorname{constant}", markdown)
    @test occursin("as `mean(log.(conc))`. Its fitted value is $LOC", markdown)
    supplied_description = brm_description(brm_descriptor(SUPPLIED, supplied(PANEL); mod=@__MODULE__))
    @test description.complete == supplied_description.complete
end

@stestset "a constant reads a local of the builder's scope and serves a plate" begin
    panel = (; subject_id=1:4, cobs=[PANEL.conc[3k-2:3k] for k in 1:4],
        yobs=[PANEL.y[3k-2:3k] for k in 1:4], wt=[60.0, 70.0, 80.0, 75.0])
    builder(shift) = @brm begin
        ref = constant(maximum(reduce(vcat, cobs)) + shift)
        sigma ~ Exponential(1)
        slope ~ 1 + wt + (1 | p | subject_id)
        @plate for i in eachindex(slope)
            pred = slope[i] .* cobs[i] ./ ref
            yobs[i] ~ normal(pred, sigma)
        end
    end
    sb = SBBRMI(builder(OFFSET)(panel); mod=@__MODULE__)
    @test stan_data(sb)[:ref] == maximum(PANEL.conc) + OFFSET
    @test stan_data(SBBRMI(builder(2.0)(panel); mod=@__MODULE__))[:ref] ==
        maximum(PANEL.conc) + 2.0
    @test StanBlocks.stan.compiles(sb.model)
    @test RKBRMI(builder(OFFSET)(panel)) isa RKBRMI
end

# A body built from a runtime string and `Core.eval`'d inside a function: the
# eval defines each constant's closure in a world newer than the function's,
# and the backends built in that same function evaluate it there (snag
# `declared-a-data-8e5f1f07`).
calibrate(conc) = (mean(log.(conc)), std(log.(conc)))
const RUNTIME_BODY = """
    ec50_loc = constant(calibrate(conc)[1])
    ec50_scale = constant(calibrate(conc)[2])
    V = constant(var(y))
    log_ec50 ~ Normal(ec50_loc, ec50_scale)
    mu ~ 1 + x + (1 | p | subject)
    r2 ~ Beta(1.0, 1.0)
    sigma = sqrt((1 - r2) * V)
    m = mu + log_ec50
    y ~ Normal(m, sigma)
"""
function build_from_runtime_body(body, data, replay)
    brmi = Core.eval(@__MODULE__, _brm(body; df=data))
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    builder = Core.eval(@__MODULE__, _brm(body))
    built = SBBRMI(Base.invokelatest(builder, data); mod=@__MODULE__, total_groups=())
    fresh = reprocess(sb, replay; freeze_constants=false)
    (; sb, built, fresh, rk=RKBRMI(brmi))
end

@stestset "constants of a body eval'd inside a function evaluate on both backends" begin
    conc = PANEL.conc[1:6] .* 10
    result = build_from_runtime_body(RUNTIME_BODY, PANEL, merge(NO_CONC, (; conc)))
    reference = SBBRMI(SUPPLIED(supplied(PANEL)); mod=@__MODULE__, total_groups=())
    for sb in (result.sb, result.built)
        @test constants(sb) == FITTED
        @test stan_code(sb) == stan_code(reference)
    end
    @test constants(result.fresh) == Dict(:ec50_loc => mean(log.(conc)),
        :ec50_scale => std(log.(conc)), :V => var(NO_CONC.y))
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = rk_logdensity_problem(result.rk; ad_backend)
    rk_supplied = rk_logdensity_problem(RKBRMI(SUPPLIED(supplied(PANEL))); ad_backend)
    for u in (zeros(LogDensityProblems.dimension(rk)), fill(0.3, LogDensityProblems.dimension(rk)))
        @test LogDensityProblems.logdensity(rk, u) == LogDensityProblems.logdensity(rk_supplied, u)
    end
end

refusal(f) = try
    f(); nothing
catch err
    sprint(showerror, err)
end

@stestset "constants refuse what is not one finite number of the data" begin
    # refused: a constant is computed before sampling, so it cannot read a
    # model value (decision `00kjor3`: data-calibrated scalars).
    message = refusal(() -> (@brm begin
        mu ~ 1 + x
        a = constant(mean(mu))
        y ~ Normal(mu, a)
    end)(PANEL))
    @test occursin("reads `mu`, a model value", message)
    message = refusal(() -> SBBRMI((@brm begin
        a = constant(log(-1.0 * first(conc)))
        y ~ Normal(0, a)
    end)(PANEL); mod=@__MODULE__))
    @test occursin("failed", message) && occursin("DomainError", message)
    message = refusal(() -> SBBRMI((@brm begin
        a = constant(conc)
        y ~ Normal(0, a)
    end)(PANEL); mod=@__MODULE__))
    @test occursin("one finite real number", message)
    message = refusal(() -> SBBRMI((@brm begin
        a = constant(mean(nonexistent_column))
        y ~ Normal(a, 1.0)
    end)(PANEL); mod=@__MODULE__))
    @test occursin("reads `nonexistent_column`", message)
    # refused: an unmarked whole-column reduction would broadcast over the
    # formula column; the declaration names it a constant (decision `00kjor3`).
    message = refusal(() -> (@brm begin
        a = mean(log.(conc))
        y ~ Normal(a, 1.0)
    end)(PANEL))
    @test occursin("constant(mean(log.(x)))", message)
    @test occursin("takes one expression", refusal(() -> eval(:(@brm begin
        a = constant(mean(conc), 2)
        y ~ Normal(a, 1.0)
    end))))
end
