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
    y=sin.(1:N) .+ 0.1 .* (1:N))
rows_of(data, rows) = map(v -> v isa AbstractVector ? v[rows] : v, data)

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

@stestset "replaying the training data is the fit" begin
    for build in (RM.string_factor, RM.monotonic, RM.fitted_constant, RM.zscaled,
            RM.zscaled_log)
        rk = RKBRMI(build(TRAIN))
        u = point(rk)
        @test density(reprocess(rk, TRAIN), u) == density(rk, u)
    end
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
