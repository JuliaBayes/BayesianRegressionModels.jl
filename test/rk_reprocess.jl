# test/rk_reprocess.jl — frozen replay of a fitted RKBRMI on new data (snag
# `rk-frozen-prepro-78732600`; the user picked `reprocess(rk, new_data)` on
# design decision `0v6p1cf`).
#
# `reprocess(rk, new_data)` binds new rows to the fitted executable graph with
# the TRAINING fit's preprocessing. Oracle: a design made of some of the
# training rows predicts, at those rows, exactly what the training binding
# computes there, a one-row design included. A design may omit its responses
# (todo `16bb838`, on RK snag `rkppl-bind-a-des-e6cfca75`): it predicts the
# same values and has no density. Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/rk_reprocess.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module ReprocessModels
using BayesianRegressionModels, Distributions
string_factor(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + gs
    y ~ Normal(mu, sigma)
end
categorical_factor(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + g
    y ~ Normal(mu, sigma)
end
monotonic(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + mo(r)
    y ~ Normal(mu, sigma)
end
fitted_constant(d) = @brm d begin
    sigma ~ Exponential(1)
    xbar = constant(mean(x))
    b ~ Normal(0, 1)
    mu = b * (x - xbar)
    y ~ Normal(mu, sigma)
end
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
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) + (1 | g)
    y ~ Normal(mu, sigma)
end
exact_gp(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + gp(x)
    y ~ Normal(mu, sigma)
end
membership(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | mm(gs, other))
    y ~ Normal(mu, sigma)
end
# A per-subject linear predictor with a standardized subject covariate, read
# by an authored cell over each subject's read times.
pk(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dose[i] .* exp.(-exp(log_CL[i]) .* t[i])
        y[i] ~ normal(conc[i], sigma)
    end
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
completed(d) = @brm d begin
    sigma ~ Exponential(1)
    mi(xm) ~ LogNormal(0.5, 0.6)
    mu ~ 1 + standardize(xm)
    y ~ Normal(mu, sigma)
end
shrunk(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + gs + (1 | g)
    effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
    y ~ Normal(mu, sigma)
end
zscaled_interaction(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) & z
    y ~ Normal(mu, sigma)
end
censored_below(d) = @brm d begin
    mu ~ 1 + zscale(x)
    y ~ censored(Normal(mu, 1); lower=lo)
end
pk_censored(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dose[i] .* exp.(-exp(log_CL[i]) .* t[i])
        y[i] ~ censored(normal(conc[i], sigma); lower=lo[i])
    end
end
# Long-format reads joined to their subject, observed in a `@plate` cell or
# as the joined rows, censored below a per-read limit.
reads_in_cell(d) = @brm d begin
    sigma ~ Exponential(1.0)
    log_k ~ 1 + (1 | p | subject)
    effect(log_k, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(1.0)
    ragged(obs_time, obs_subject)
    ragged(obs_limit, obs_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_k)
        loc[i] = exp.(-exp(log_k[i]) .* obs_time[i])
        y[i] ~ censored(Normal(loc[i], sigma); lower=obs_limit[i])
    end
end
reads_joined(d) = @brm d begin
    sigma ~ Exponential(1.0)
    log_k ~ 1 + (1 | p | subject)
    effect(log_k, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(1.0)
    ragged(obs_time, obs_subject)
    @plate for i in eachindex(log_k)
        loc[i] = exp.(-exp(log_k[i]) .* obs_time[i])
    end
    ragged(y, obs_subject) ~ censored(Normal(loc, sigma); lower=obs_limit)
end

# A cell whose curve a callable-source provider claims as an RK graph (a scan
# over the doses and a plate over the read times), counting its emissions.
import BayesianRegressionModels: _rk_callable_source!
using ReactiveKernels
function dosed_curve(t, dose, log_CL)
    k = exp(log_CL)
    [dose * exp(-k * ti) for ti in t]
end
const CURVE_EMISSIONS = Ref(0)
function _rk_callable_source!(definitions, bindings, entry, ::typeof(dosed_curve))
    CURVE_EMISSIONS[] += 1
    push!(definitions, :(ReactiveKernels.@kernel $entry(t, dose, log_CL) = begin
        k = exp(log_CL)
        decay = ReactiveKernels.plate(eachindex(t)) do j
            exp(-k * t[j])
        end
        reads = ReactiveKernels.scan(decay; init=0.0) do carry, e
            (carry, dose * e)
        end
        return reads
    end))
    :done
end
pk_provided(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dosed_curve(t[i], dose[i], log_CL[i])
        y[i] ~ normal(conc[i], sigma)
    end
end
# A provider whose leaf binding is a new object on every emission: the same
# program text, other callables.
fresh_leaf(t, dose, log_CL) = [dose * exp(-exp(log_CL) * ti) for ti in t]
function _rk_callable_source!(definitions, bindings, entry, ::typeof(fresh_leaf))
    scale = Ref(1.0)
    leaf = Symbol(entry, :_leaf)
    push!(bindings, leaf => (t, dose, log_CL) -> scale[] .* dose .* exp.(-exp(log_CL) .* t))
    push!(definitions, :(function $entry(t, dose, log_CL)
        return $leaf(t, dose, log_CL)
    end))
    :done
end
pk_fresh_leaf(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = fresh_leaf(t[i], dose[i], log_CL[i])
        y[i] ~ normal(conc[i], sigma)
    end
end
end
const RM = ReprocessModels

const N = 12
const TRAIN = (; x=collect(range(0.3, 4.1; length=N)) .^ 1.3,
    g=categorical(["a", "b", "c"][mod1.(1:N, 3)]),
    gs=["a", "b", "c"][mod1.(1:N, 3)],
    r=[1, 2, 3, 4][mod1.(1:N, 4)],
    z=cos.(1:N),
    other=["b", "c", "a"][mod1.(1:N, 3)],
    keep=[isodd(i) || i > 8 for i in 1:N],
    xm=Union{Missing,Float64}[i % 4 == 0 ? missing : 0.4 + 0.3i for i in 1:N],
    lo=fill(-0.5, N),
    y=sin.(1:N) .+ 0.1 .* (1:N))
rows_of(data, rows) = map(v -> v isa AbstractVector ? v[rows] : v, data)
without(data, name) = NamedTuple{filter(!=(name), keys(data))}(data)

const PK = (; subject=["s1", "s2", "s3", "s4", "s5"],
    weight=[60.0, 72.0, 81.0, 55.0, 90.0], dose=[100.0, 120.0, 80.0, 100.0, 150.0],
    t=[[0.5, 1.0, 2.0], [0.25, 1.5], [0.5, 1.0, 2.0, 4.0], [1.0], [0.5, 2.0]],
    lo=[fill(0.15, 3), fill(0.15, 2), fill(0.15, 4), fill(0.15, 1), fill(0.15, 2)],
    y=[[1.0, 0.8, 0.5], [1.2, 0.7], [0.9, 0.7, 0.4, 0.2], [0.6], [1.3, 0.5]])
# Subject j's reads in the flattened `conc` of the training data.
const PK_ENDS = cumsum([0; length.(PK.t)])
pk_reads(conc, subjects) =
    reduce(vcat, [conc[(PK_ENDS[j] + 1):PK_ENDS[j + 1]] for j in subjects])

const READS = (; subject=["b", "a", "c"], obs_subject=["a", "c", "b", "a", "b", "c"],
    obs_time=[0.5, 0.25, 1.5, 1.0, 3.0, 2.0], obs_limit=[0.05, 0.05, 0.1, 0.4, 0.05, 0.05],
    y=[0.81, 0.76, 0.52, 0.4, 0.21, 0.33])
# The subjects `keep` with their reads.
function reads_of(keep)
    rows = findall(in(keep), READS.obs_subject)
    (; subject=keep, obs_subject=READS.obs_subject[rows], obs_time=READS.obs_time[rows],
        obs_limit=READS.obs_limit[rows], y=READS.y[rows])
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

# A backend whose `y` was omitted has no density: RK refuses every query that
# reads `y`, naming it.
function check_no_density(replay, u)
    err = try
        density(replay, u)
        nothing
    catch e
        e
    end
    @test err isa ReactiveKernelsPPL.ContractValidationError
    @test occursin("`y`", sprint(showerror, err))
end

# The replay predicts the training values at the training rows it holds, and
# its density prepares on the fitted graph (one program).
function check_frozen(build, rows; want=:mu)
    rk = RKBRMI(build(TRAIN))
    u = point(rk)
    full = graph_value(rk, want, u)
    saved = deepcopy(TRAIN)
    replay = reprocess(rk, rows_of(TRAIN, rows))
    @test replay.model === rk.model
    @test graph_value(replay, want, u) ≈ full[rows] atol=1e-12 rtol=1e-12
    @test isfinite(density(replay, u))
    @test isequal(TRAIN, saved)
    replay
end

@stestset "level sets: a factor or mo design keeps the training levels" begin
    # Rows 3 and 6 hold only level `c`: the design's own levels would make it
    # the reference.
    check_frozen(RM.string_factor, [3, 6])
    check_frozen(RM.categorical_factor, [3, 6])
    check_frozen(RM.monotonic, [1, 4])
    check_frozen(RM.monotonic, [7])
end

@stestset "constant(expr) keeps its fitted value" begin
    check_frozen(RM.fitted_constant, 2:5)
    check_frozen(RM.fitted_constant, [7])
end

@stestset "zscale/center/standardize keep the training anchors" begin
    for build in (RM.zscaled, RM.centered, RM.standardized, RM.zscaled_log,
            RM.zscaled_interaction)
        check_frozen(build, 2:5)
        # One row has no SD of its own; the training anchors apply.
        check_frozen(build, [7])
    end
    # The fit reads its anchors from a fit-role copy of the column.
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(RM.zscaled(TRAIN)))
    source = sprint(Base.show_unquoted, emitted.main)
    @test occursin("brm_covariate_mean(x_fit)", source)
    @test occursin("brm_covariate_sd(x_fit)", source)
end

@stestset "hsgp keeps the training basis interval" begin
    for build in (RM.smooth, RM.smooth_rows, RM.smooth_2d, RM.smooth_orthogonal,
            RM.smooth_clamped)
        check_frozen(build, 2:5)
        # One row is a degenerate axis of its own; the training interval applies.
        check_frozen(build, [7])
    end
    # Past the training data range, `extrapolate=:clamp` holds the curve at
    # the fitted edge; otherwise a value past the basis interval is refused.
    top = argmax(TRAIN.x)
    rk = RKBRMI(RM.smooth_clamped(TRAIN))
    u = point(rk)
    edge = graph_value(rk, :mu, u)[top]
    beyond = merge(rows_of(TRAIN, [top]), (; x=[2 * TRAIN.x[top]]))
    @test only(graph_value(reprocess(rk, beyond), :mu, u)) ≈ edge atol=1e-12 rtol=1e-12
    @test_throws "lies outside the fitted basis interval" reprocess(
        RKBRMI(RM.smooth(TRAIN)), merge(rows_of(TRAIN, [top]), (; x=[100 * TRAIN.x[top]])))
end

@stestset "s/t2 keep the training basis" begin
    for build in (RM.thin_plate, RM.tensor)
        # The design has fewer unique values than a basis needs of its own.
        check_frozen(build, 2:5)
        check_frozen(build, [7])
    end
end

@stestset "a completed covariate keeps its training anchors" begin
    # The design holds every missing row, so the completions are the fit's;
    # its observed rows differ, so its own anchors would differ.
    check_frozen(RM.completed, [2, 4, 7, 8, 12])
    # Dropping a missing row drops a completion coordinate: refused.
    @test_throws "changes the fitted coordinates" reprocess(
        RKBRMI(RM.completed(TRAIN)), rows_of(TRAIN, 1:6))
end

@stestset "an R2D2 budget keeps the training column variances" begin
    rk = RKBRMI(RM.shrunk(TRAIN))
    u = point(rk)
    # Rows 2:5 have other variances; the prior must be the fitted one.
    replay = check_frozen(RM.shrunk, 2:5)
    prior(rk) = Base.invokelatest(ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(rk_bound_data(rk))...), want=(:prior,),
        bound=rk_bound_data(rk)), u)
    @test only(prior(replay)) == only(prior(rk))
end

@stestset "a same-group replay keeps the fitted group effects" begin
    # Rows 3 and 6 hold only group `c`, a single subject or two in another order.
    check_frozen(RM.grouped, [3, 6])
    rk = RKBRMI(RM.pk(PK))
    u = point(rk)
    conc = graph_value(rk, :conc, u)
    for subjects in ([2], [4, 1])
        replay = reprocess(rk, rows_of(PK, subjects))
        @test graph_value(replay, :conc, u) ≈ pk_reads(conc, subjects) atol=1e-12 rtol=1e-12
        @test isfinite(density(replay, u))
    end
    @test_throws "not a training level" reprocess(rk,
        merge(rows_of(PK, [2]), (; subject=["zzz"])))
    # A new read grid and dose for in-sample subjects, without responses: the
    # cell runs on the new times with each subject's fitted clearance.
    log_CL = graph_value(rk, :log_CL, u)
    grid = [[0.1, 0.2, 3.0, 6.0, 9.0], [0.5]]
    design = merge(without(rows_of(PK, [3, 1]), :y), (; t=grid, dose=[50.0, 75.0]))
    expected = reduce(vcat, [dose .* exp.(-exp(log_CL[j]) .* t)
        for (j, dose, t) in zip([3, 1], design.dose, grid)])
    @test graph_value(reprocess(rk, design), :conc, u) ≈ expected atol=1e-12 rtol=1e-12
end

@stestset "resample_groups: new subjects take caller-supplied effects" begin
    rk = RKBRMI(RM.pk(PK))
    u = point(rk)
    conc = graph_value(rk, :conc, u)
    b = graph_value(rk, :b_p_subject, u)       # one row per training subject
    # Subject s2's covariates and reads under a new label, with s2's effect.
    replay = reprocess(rk, merge(rows_of(PK, [2]), (; subject=["new"]));
        resample_groups=[:subject])
    bound = rk_bound_data(replay)
    query = ReactiveKernels.prepare(replay.model.spec;
        have=(:unconstrained, :b_p_subject, keys(bound)...), want=(:conc,), bound)
    s2 = findfirst(==("s2"), sort(PK.subject))
    value = Base.invokelatest(query, u, b[s2:s2, :])
    value = value isa Tuple ? only(value) : value
    @test value ≈ pk_reads(conc, [2]) atol=1e-12 rtol=1e-12
    # The new subjects' effects are not the fit's coordinates: no density.
    @test_throws "different program" density(replay, u)
end

@stestset "a grouping whose levels are not frozen is refused" begin
    rk = RKBRMI(RM.membership(TRAIN))
    @test_throws "do not freeze their group levels" reprocess(rk, rows_of(TRAIN, 2:5))
end

@stestset "latent values of the training rows are refused" begin
    # Same row count, other rows: the coordinates match, the latents do not.
    @test_throws "latent values of the training rows" reprocess(
        RKBRMI(RM.exact_gp(TRAIN)), rows_of(TRAIN, reverse(1:N)))
end

@stestset "replaying the training data is the fit" begin
    for build in (RM.string_factor, RM.monotonic, RM.fitted_constant, RM.zscaled,
            RM.zscaled_log, RM.grouped, RM.smooth, RM.smooth_rows, RM.smooth_orthogonal,
            RM.thin_plate, RM.tensor)
        rk = RKBRMI(build(TRAIN))
        u = point(rk)
        @test density(reprocess(rk, TRAIN), u) == density(rk, u)
    end
    rk = RKBRMI(RM.pk(PK))
    u = point(rk)
    @test density(reprocess(rk, PK), u) == density(rk, u)
end

@stestset "an unseen level is refused" begin
    rk = RKBRMI(RM.string_factor(TRAIN))
    @test_throws "not a training level" reprocess(rk,
        merge(rows_of(TRAIN, [1, 2]), (; gs=["a", "zzz"])))
end

@stestset "a design without responses predicts the same values" begin
    # The replay binds `y` without values, so each authored value equals the
    # with-`y` replay's; one program, so the location query prepares too.
    for (build, rows) in ((RM.string_factor, [3, 6]), (RM.monotonic, [7]),
            (RM.fitted_constant, 2:5), (RM.zscaled, 2:5), (RM.zscaled, [7]),
            (RM.grouped, [3, 6]), (RM.smooth, [7]), (RM.thin_plate, 2:5),
            (RM.shrunk, 2:5), (RM.censored_below, 2:5))
        rk = RKBRMI(build(TRAIN))
        u = point(rk)
        design = rows_of(TRAIN, rows)
        replay = reprocess(rk, without(design, :y))
        @test replay.model === rk.model
        mu = graph_value(reprocess(rk, design), :mu, u)
        @test graph_value(replay, :mu, u) == mu
        observations = BRM._rk_observation_values(BRM._rk_observation_query(replay), u)
        @test observations.y.location == mu
        check_no_density(replay, u)
    end
    # One array of reads per subject: `conc` follows each subject's `t`.
    for build in (RM.pk_censored, RM.pk)
        rk = RKBRMI(build(PK))
        u = point(rk)
        for subjects in ([2], [4, 1])
            design = rows_of(PK, subjects)
            replay = reprocess(rk, without(design, :y))
            @test graph_value(replay, :conc, u) == graph_value(reprocess(rk, design), :conc, u)
            check_no_density(replay, u)
        end
    end
    # Long-format reads: the rows follow each subject's `obs_subject` reads.
    for build in (RM.reads_in_cell, RM.reads_joined), keep in (["c"], ["c", "a"])
        rk = RKBRMI(build(READS))
        u = point(rk)
        design = reads_of(keep)
        replay = reprocess(rk, without(design, :y))
        @test graph_value(replay, :loc, u) == graph_value(reprocess(rk, design), :loc, u)
        check_no_density(replay, u)
    end
    # New subjects without responses take caller-supplied effects.
    rk = RKBRMI(RM.pk(PK))
    u = point(rk)
    b = graph_value(rk, :b_p_subject, u)
    replay = reprocess(rk, without(merge(rows_of(PK, [2]), (; subject=["new"])), :y);
        resample_groups=[:subject])
    bound = rk_bound_data(replay)
    query = ReactiveKernels.prepare(replay.model.spec;
        have=(:unconstrained, :b_p_subject, keys(bound)...), want=(:conc,), bound)
    s2 = findfirst(==("s2"), sort(PK.subject))
    value = Base.invokelatest(query, u, b[s2:s2, :])
    value = value isa Tuple ? only(value) : value
    @test value ≈ pk_reads(graph_value(rk, :conc, u), [2]) atol=1e-12 rtol=1e-12
    # Other data stay required.
    @test_throws "has no column `x`" reprocess(RKBRMI(RM.zscaled(TRAIN)),
        without(without(rows_of(TRAIN, 2:5), :y), :x))
end

@stestset "freeze_constants=false rebuilds on the new data" begin
    rk = RKBRMI(RM.fitted_constant(TRAIN))
    design = rows_of(TRAIN, 2:7)
    fresh = reprocess(rk, design; freeze_constants=false)
    own = RKBRMI(RM.fitted_constant(design))
    u = point(own)
    @test fresh.model !== rk.model
    @test density(fresh, u) == density(own, u)
end

@stestset "a replay binds the fit's translated program" begin
    # Translation (emission, definition module, lowering) is per program and
    # binding per data (snag `rk-bound-data-re-ddd101cb`): a replay shares the
    # fit's program and binds only its own data, once, for every query.
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    fresh(rk) = ext._rk_translated_plan(rk.plan).columns
    for build in (RM.pk, RM.pk_provided)
        rk = RKBRMI(build(PK))
        u = point(rk)
        conc = graph_value(rk, :conc, u)
        for (design, groups) in ((rows_of(PK, [4, 1]), ()),
                (without(merge(rows_of(PK, [2]), (; subject=["new"])), :y), [:subject]))
            replay = reprocess(rk, design; resample_groups=groups)
            @test replay.program === rk.program
            @test ext._rk_program(replay).mod === ext._rk_program(rk).mod
            bound = rk_bound_data(replay)
            @test isequal(Dict(pairs(bound)), fresh(replay))
            @test rk_bound_data(replay) === bound
        end
        replay = reprocess(rk, rows_of(PK, [4, 1]))
        @test graph_value(replay, :conc, u) ≈ pk_reads(conc, [4, 1]) atol=1e-12 rtol=1e-12
        @test density(reprocess(rk, PK), u) == density(rk, u)
    end
    # The provider runs once per emission: the fit's, and each replay's own
    # (its program check). Binding a replay's data emits nothing.
    rk = RKBRMI(RM.pk_provided(PK))
    u = point(rk)
    before = RM.CURVE_EMISSIONS[]
    replay = reprocess(rk, rows_of(PK, [3]))
    @test RM.CURVE_EMISSIONS[] == before + 1
    graph_value(replay, :conc, u)
    @test isfinite(density(replay, u))
    @test RM.CURVE_EMISSIONS[] == before + 1
    # Another callable object under the same name: the replay translates its
    # own program, with its own callables.
    rk = RKBRMI(RM.pk_fresh_leaf(PK))
    u = point(rk)
    replay = reprocess(rk, rows_of(PK, [4, 1]))
    @test replay.program !== rk.program
    @test isequal(Dict(pairs(rk_bound_data(replay))), fresh(replay))
    @test graph_value(replay, :conc, u) ≈
        pk_reads(graph_value(rk, :conc, u), [4, 1]) atol=1e-12 rtol=1e-12
end
