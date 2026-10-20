# test/rk_reprocess.jl — frozen replay of a fitted RKBRMI on new data (snag
# `rk-frozen-prepro-78732600`; the user picked `reprocess(rk, new_data)` on
# design decision `0v6p1cf`).
#
# `reprocess(rk, new_data)` binds new rows to the fitted executable graph with
# the TRAINING fit's preprocessing. Oracle: a design made of some of the
# training rows predicts, at those rows, exactly what the training binding
# computes there, a one-row design included. Public synthetic models only.
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
zscaled_interaction(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) & z
    y ~ Normal(mu, sigma)
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
    y=sin.(1:N) .+ 0.1 .* (1:N))
rows_of(data, rows) = map(v -> v isa AbstractVector ? v[rows] : v, data)

const PK = (; subject=["s1", "s2", "s3", "s4", "s5"],
    weight=[60.0, 72.0, 81.0, 55.0, 90.0], dose=[100.0, 120.0, 80.0, 100.0, 150.0],
    t=[[0.5, 1.0, 2.0], [0.25, 1.5], [0.5, 1.0, 2.0, 4.0], [1.0], [0.5, 2.0]],
    y=[[1.0, 0.8, 0.5], [1.2, 0.7], [0.9, 0.7, 0.4, 0.2], [0.6], [1.3, 0.5]])
# Subject j's reads in the flattened `conc` of the training data.
const PK_ENDS = cumsum([0; length.(PK.t)])
pk_reads(conc, subjects) =
    reduce(vcat, [conc[(PK_ENDS[j] + 1):PK_ENDS[j + 1]] for j in subjects])

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
end

@stestset "a grouping whose levels are not frozen is refused" begin
    rk = RKBRMI(RM.membership(TRAIN))
    @test_throws "do not freeze their group levels" reprocess(rk, rows_of(TRAIN, 2:5))
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

@stestset "freeze_constants=false rebuilds on the new data" begin
    rk = RKBRMI(RM.fitted_constant(TRAIN))
    design = rows_of(TRAIN, 2:7)
    fresh = reprocess(rk, design; freeze_constants=false)
    own = RKBRMI(RM.fitted_constant(design))
    u = point(own)
    @test fresh.model !== rk.model
    @test density(fresh, u) == density(own, u)
end
