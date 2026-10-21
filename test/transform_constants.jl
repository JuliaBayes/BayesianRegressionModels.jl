# test/transform_constants.jl — the public reader of fitted `center` / `zscale`
# / `standardize` constants (snag `standardize-cons-d85fd047`; the user took
# the recommended options on design decision `1snqq9i`).
#
# `brm_transform_constants(x[, name])` reports `(; name, kind, column, source,
# center, scale, observed_only)` from what the backend applies. Oracles: the
# SBBRMI's own transformed Stan data, reproduced bit for bit from the record;
# the RK graph's own transformed value; `mean`/`std` of the training values to
# floating-point tolerance. Public synthetic models only.
#
# Run: julia --project=test test/transform_constants.jl
using Test
using BayesianRegressionModels, Distributions, Statistics, Random
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

Random.seed!(20261021)
const N = 40
const TRAIN = (; age = 30 .+ 40 .* rand(N), w = 50 .+ 30 .* rand(N), p = rand(N), y = randn(N))
const NEW = (; age = 20 .+ 60 .* rand(7), w = 40 .+ 50 .* rand(7), p = rand(7), y = randn(7))

plain = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + standardize(age) + center(w) + zscale(log(w)) + p & zscale(age)
    y ~ Normal(mu, sigma)
end
completed = @brm begin
    sigma ~ Exponential(1)
    ma ~ Normal(50, 10)
    sa ~ LogNormal(2, 0.5)
    mi(age) ~ Normal(ma, sa)
    mu ~ 1 + standardize(age) + center(w)
    y ~ Normal(mu, sigma)
end
const AGE_MISSING = let a = Vector{Union{Missing,Float64}}(TRAIN.age)
    a[3] = missing; a[7] = missing; a
end

record_by_source(records, source) = only(filter(r -> r.source == source, collect(records)))

# An authored value of `rk`'s graph on its own bound data (test/rk_reprocess.jl).
function graph_value(rk, want)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(want,), bound)
    value = Base.invokelatest(query, zeros(rk.model.layout.total))
    value isa Tuple ? only(value) : value
end

@testset "SBBRMI records reproduce the transformed Stan data" begin
    sb = SBBRMI(plain(TRAIN); mod=@__MODULE__)
    records = brm_transform_constants(sb)
    logw = record_by_source(records, :(log(w)))
    @test Set(r.name for r in records) ==
          Set([:standardize_age, :center_w, :zscale_age, logw.name])

    s = brm_transform_constants(sb, :standardize_age)
    @test s == only(filter(r -> r.name === :standardize_age, collect(records)))
    @test (s.kind, s.column, s.source, s.observed_only) == (:standardize, :age, :age, false)
    @test sb.data[:standardize_age] == (TRAIN.age .- s.center) ./ s.scale
    @test s.center ≈ mean(TRAIN.age) && s.scale ≈ std(TRAIN.age)

    c = brm_transform_constants(sb, :center_w)
    @test (c.kind, c.column, c.scale, c.observed_only) == (:center, :w, 1.0, false)
    @test sb.data[:center_w] == TRAIN.w .- c.center

    @test (logw.kind, logw.column) == (:zscale, nothing)
    @test sb.data[logw.name] == (log.(TRAIN.w) .- logw.center) ./ logw.scale

    # An `&` operand is recorded too; the same column and estimator as
    # `standardize(age)` give the same constants.
    z = brm_transform_constants(sb, :zscale_age)
    @test (z.kind, z.center, z.scale) == (:zscale, s.center, s.scale)

    # Every reader of the same fit agrees.
    @test brm_transform_constants(generative_plan(sb)) == records
    @test brm_transform_constants(brm_descriptor(sb)) == records

    # refused: not a fitted center/zscale/standardize name — the error lists
    # the fitted ones instead of returning a guess.
    err = try brm_transform_constants(sb, :standardize_weight); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("Fitted transforms", err.msg) && occursin("standardize_age", err.msg)
end

@testset "replay reports the constants it applies" begin
    sb = SBBRMI(plain(TRAIN); mod=@__MODULE__)
    frozen = reprocess(sb, NEW)
    @test brm_transform_constants(frozen) == brm_transform_constants(sb)
    s = brm_transform_constants(frozen, :standardize_age)
    @test frozen.data[:standardize_age] == (NEW.age .- s.center) ./ s.scale

    refit = reprocess(sb, NEW; freeze_constants=false)
    r = brm_transform_constants(refit, :standardize_age)
    @test r.center ≈ mean(NEW.age) && r.scale ≈ std(NEW.age)
    @test refit.data[:standardize_age] == (NEW.age .- r.center) ./ r.scale
end

@testset "mi() columns: observed-only fit under the coefficient's name" begin
    sb = SBBRMI(completed((; TRAIN..., age = AGE_MISSING)); mod=@__MODULE__)
    @test !haskey(sb.preproc, :standardize_age)
    s = brm_transform_constants(sb, :standardize_age)
    fit = sb.preproc[:standardize_age_mean].const_.fit
    @test (s.kind, s.column, s.observed_only) == (:standardize, :age, true)
    @test (s.center, s.scale) == (fit.mean, fit.scale)
    @test s.center ≈ mean(skipmissing(AGE_MISSING)) && s.scale ≈ std(skipmissing(AGE_MISSING))
    c = brm_transform_constants(sb, :center_w)
    @test (c.kind, c.observed_only) == (:center, false)
end

@testset "RKBRMI reads the anchors its graph evaluates" begin
    sb = SBBRMI(plain(TRAIN); mod=@__MODULE__)
    rk = RKBRMI(plain(TRAIN))
    records = brm_transform_constants(rk)
    sb_records = brm_transform_constants(sb)
    # The `&` operand is staged under an internal RK name.
    @test Set(r.name for r in records) == setdiff(Set(r.name for r in sb_records), [:zscale_age])
    err = try brm_transform_constants(rk, :zscale_age); nothing catch e; e end
    @test err isa ErrorException && occursin("`&` interaction", err.msg)

    s = brm_transform_constants(rk, :standardize_age)
    sb_s = brm_transform_constants(sb, :standardize_age)
    # Both backends fit `standardize` with the same streaming estimator.
    @test s == sb_s
    logw = record_by_source(records, :(log(w)))
    @test logw == record_by_source(sb_records, :(log(w)))
    # `center` means: SB `sum/length`, RK the streaming mean.
    c = brm_transform_constants(rk, :center_w)
    @test (c.kind, c.scale) == (:center, 1.0)
    @test c.center ≈ brm_transform_constants(sb, :center_w).center

    # The graph's own transformed values are the record applied to the rows.
    @test graph_value(rk, :standardize_age) ≈ (TRAIN.age .- s.center) ./ s.scale
    @test graph_value(rk, :center_w) ≈ TRAIN.w .- c.center

    # A frozen replay keeps the training anchors.
    replay = reprocess(rk, NEW)
    @test brm_transform_constants(replay) == records
    @test graph_value(replay, :standardize_age) ≈ (NEW.age .- s.center) ./ s.scale
end

@testset "RKBRMI mi() columns" begin
    data = (; TRAIN..., age = AGE_MISSING)
    rk = RKBRMI(completed(data))
    s = brm_transform_constants(rk, :standardize_age)
    @test (s.kind, s.column, s.observed_only) == (:standardize, :age, true)
    sb = SBBRMI(completed(data); mod=@__MODULE__)
    @test s == brm_transform_constants(sb, :standardize_age)
    @test brm_transform_constants(rk, :center_w).observed_only == false
end
