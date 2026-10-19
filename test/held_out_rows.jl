# Per-row response hold-out: `held_out=(; y = mask)`.
#
#   julia --startup-file=no --project=test test/held_out_rows.jl

using Test
using BayesianRegressionModels, StanBlocks, Distributions
using LogDensityProblems
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
import BridgeStan

const BRM = BayesianRegressionModels

rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
pointwise(rk, u) = BRM._rk_pointwise_values(BRM._rk_pointwise_query(rk), u)

const flat = (; x=[1.0, 2.0, 0.5, 1.5, -0.3], y=[0.1, 0.2, 0.3, 0.5, -0.4],
    z=[0.0, 0.4, 0.2, 0.9, 0.1])
const late = [false, false, false, true, true]

const single = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
const joint = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    nu ~ 1 + x
    z ~ Normal(nu, 1.0)
    y ~ Normal(mu, sigma)
end

# The data with only the rows `keep` selects, for the fit a row mask must equal.
keep_rows(data, keep) = map(column -> column[keep], data)

const bounded = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ LogNormal(mu, sigma)
end

@testset "held_out row masks are validated against the response" begin
    brmi = single(flat)
    for build in ((b; kw...) -> RKBRMI(b; kw...),
            (b; kw...) -> SBBRMI(b; mod=@__MODULE__, kw...))
        @test_throws r"length 3, but the response has 5" build(brmi; held_out=(; y=late[1:3]))
        @test_throws r"must hold `Bool` values" build(brmi; held_out=(; y=[0, 0, 0, 1, 1]))
        @test_throws r"unknown response" build(brmi; held_out=(; w=late))
        @test_throws r"both as a whole response and with a row mask" build(joint(flat);
            held_out=[:y, :y => late])
        # refused: every row of the only response is not a fit (held-out-and-prior-draws)
        @test_throws r"covers every observation" build(brmi; held_out=(; y=trues(5)))
    end
end

@testset "RKBRMI: held-out rows leave the fitted density, coordinates stay" begin
    for brmi in (single(flat), joint(flat))
        full = RKBRMI(brmi)
        masked = RKBRMI(brmi; held_out=(; y=late))
        @test occursin("withheld rows: y 2 of 5", sprint(show, masked))
        @test coordinate_names(masked.model.layout) == coordinate_names(full.model.layout)
        n = length(coordinate_names(full.model.layout))
        for u in (fill(0.1, n), collect(range(-0.4, 0.3; length=n)))
            rows = pointwise(full, u)
            held = rows.y[late]
            @test LogDensityProblems.logdensity(rk_problem(masked), u) ≈
                  LogDensityProblems.logdensity(rk_problem(full), u) - sum(held) atol=1e-10
            @test pointwise(masked, u).y[.!late] ≈ rows.y[.!late] atol=1e-12
            @test all(iszero, pointwise(masked, u).y[late])
        end
    end
    # A held row is never evaluated: outside LogNormal's support it leaves the
    # target the kept rows' density.
    masked = RKBRMI(bounded(flat); held_out=(; y=late))
    kept = RKBRMI(bounded(keep_rows(flat, .!late)))
    for u in (fill(0.1, 3), [-0.4, 0.05, 0.3])
        @test LogDensityProblems.logdensity(rk_problem(masked), u) ≈
              LogDensityProblems.logdensity(rk_problem(kept), u) atol=1e-10
    end
    # An all-false mask holds nothing out; an all-true one holds the response out.
    @test sprint(show, RKBRMI(joint(flat); held_out=(; y=falses(5)))) ==
          sprint(show, RKBRMI(joint(flat)))
    @test sprint(show, RKBRMI(joint(flat); held_out=(; y=trues(5)))) ==
          sprint(show, RKBRMI(joint(flat); held_out=:y))
end

# One array per subject, observed inside `@plate for`.
const cell = (; x=[[1.0, 2.0], [0.5, 1.5, -0.4]], y=[[0.1, 0.2], [0.3, 0.5, -0.1]])
const cell_mask = [[false, true], [false, false, true]]
const cell_kept = (; x=[[1.0], [0.5, 1.5]], y=[[0.1], [0.3, 0.5]])
const incell = @brm begin
    s ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(x)
        m = s .* x[i]
        y[i] ~ Normal(m, sigma)
    end
end

# A per-subject time course observed inside a `kernel(...)` cell.
const course = @brm begin
    sigma ~ Exponential(1)
    log_k ~ 1 + (1 | p | subject)
    pred ~ kernel(t, conc, log_k) do tt, obs, lk
        m = exp(-exp(lk) .* tt)
        obs ~ normal(m, sigma)
        m
    end
end
# Subjects out of label order, so the mask must follow the data's own order.
const course_data = (; subject=[:b, :a], t=[[0.5, 1.0], [0.5, 1.0, 2.0]],
    conc=[[0.6, 0.4], [0.7, 0.3, 0.1]])
const course_mask = [[false, true], [false, false, true]]
const course_kept = (; subject=[:b, :a], t=[[0.5], [0.5, 1.0]], conc=[[0.6], [0.7, 0.3]])

# A flat response joined to a kernel's subjects, its rows out of subject order
# so the mask is gathered like the response.
const joined = @brm begin
    sigma_y ~ Exponential(1)
    sigma_qt ~ Exponential(1)
    log_scale ~ 1 + (1 | p | subject)
    pred ~ kernel(ragged(obs_idx, obs_subject), qt_y, log_scale) do idxs, qt_obs, ls
        qt_obs ~ normal(exp(ls), sigma_qt)
        idxs .* exp(ls)
    end
    ragged(obs_y, obs_subject) ~ Normal(pred, sigma_y)
end
const joined_rows = (; obs_subject=[:c, :a, :b, :a, :c], obs_idx=[1.0, 1.0, 1.0, 2.0, 2.0],
    obs_y=[1.2, 1.0, 0.9, 2.1, 2.3])
const joined_data = (; subject=[:a, :b, :c], qt_y=[0.8, 1.2, 1.7], joined_rows...)
const joined_mask = [true, false, false, true, false]
const joined_kept = (; subject=[:a, :b, :c], qt_y=[0.8, 1.2, 1.7],
    keep_rows(joined_rows, .!joined_mask)...)

# A Stan file name must be one shell word.
stan_problem(sb, label) = BRM.stan_instantiate(sb;
    path=joinpath(mktempdir(), replace(label, r"[^A-Za-z0-9]+" => "_") * ".stan"))
unc_names(problem) = BridgeStan.param_unc_names(problem.model)
stan_points(problem) = (n = length(unc_names(problem));
    (fill(0.1, n), collect(range(-0.4, 0.3; length=n)), [0.2 * sin(i) for i in 1:n]))
sb_execute(sb, operation, problem, u) =
    brm_execute(brm_descriptor(sb), operation; problem, draws=u, seed=20261019)

@testset "SBBRMI: held-out rows leave the fitted density, coordinates stay" begin
    # flat responses: the held rows' pointwise scores are exactly what leaves
    for (label, brmi) in (("single", single(flat)), ("joint", joint(flat)))
        full_sb = SBBRMI(brmi; mod=@__MODULE__)
        masked_sb = SBBRMI(brmi; mod=@__MODULE__, held_out=(; y=late))
        @test masked_sb.held_out == Set{Symbol}()
        @test masked_sb.held_out_rows == Dict(:y => late)
        full, masked = stan_problem(full_sb, "$label-full"), stan_problem(masked_sb, "$label-masked")
        @test unc_names(masked) == unc_names(full)
        offsets = map(stan_points(full)) do u
            full_rows = sb_execute(full_sb, :pointwise_loglik, full, u).y_likelihood
            masked_rows = sb_execute(masked_sb, :pointwise_loglik, masked, u).y_likelihood
            # y_likelihood keeps every row: a held row's entry is its score
            @test masked_rows ≈ full_rows atol=1e-10
            @test length(sb_execute(masked_sb, :predict, masked, u).y_gen) == 5
            LogDensityProblems.logdensity(masked, u) -
                (LogDensityProblems.logdensity(full, u) - sum(full_rows[late]))
        end
        @test all(o -> isapprox(o, first(offsets); atol=1e-8), offsets)
    end

    # the masked fit is the fit on the kept rows, gradient included
    for (label, builder, data, mask, kept) in (
            ("single", single, flat, (; y=late), keep_rows(flat, .!late)),
            ("@plate cell", incell, cell, (; y=cell_mask), cell_kept),
            ("kernel cell", course, course_data, (; conc=course_mask), course_kept),
            ("ragged join", joined, joined_data, (; obs_y=joined_mask), joined_kept))
        @testset "$label" begin
            masked = stan_problem(SBBRMI(builder(data); mod=@__MODULE__, held_out=mask),
                "$label-masked")
            reference = stan_problem(SBBRMI(builder(kept); mod=@__MODULE__), "$label-kept")
            @test unc_names(masked) == unc_names(reference)
            offsets = map(stan_points(masked)) do u
                value, gradient = LogDensityProblems.logdensity_and_gradient(masked, u)
                kept_value, kept_gradient =
                    LogDensityProblems.logdensity_and_gradient(reference, u)
                @test isfinite(value)
                @test gradient ≈ kept_gradient atol=1e-8 rtol=1e-8
                value - kept_value
            end
            @test all(o -> isapprox(o, first(offsets); atol=1e-8), offsets)
        end
    end

    # not built yet: a held row outside LogNormal's support still errors the
    # target, because StanBlocks evaluates held elements' densities before
    # skipping them (StanBlocks snag heldout-lpdf-eva-a8569726)
    masked = stan_problem(SBBRMI(bounded(flat); mod=@__MODULE__, held_out=(; y=late)),
        "bounded-masked")
    @test_broken isfinite(LogDensityProblems.logdensity(masked, fill(0.1, 3)))

    # a response bound per subject scores per subject, held rows included
    for (label, builder, data, mask, response) in (
            ("@plate cell", incell, cell, (; y=cell_mask), :y_likelihood),
            ("kernel cell", course, course_data, (; conc=course_mask), :conc_likelihood),
            ("ragged join", joined, joined_data, (; obs_y=joined_mask), :obs_y_likelihood))
        full_sb = SBBRMI(builder(data); mod=@__MODULE__)
        masked_sb = SBBRMI(builder(data); mod=@__MODULE__, held_out=mask)
        full, masked = stan_problem(full_sb, "$label-full"), stan_problem(masked_sb, "$label-masked")
        @test unc_names(masked) == unc_names(full)
        for u in stan_points(full)
            @test getproperty(sb_execute(masked_sb, :pointwise_loglik, masked, u), response) ≈
                  getproperty(sb_execute(full_sb, :pointwise_loglik, full, u), response) atol=1e-10
        end
    end
end

@testset "SBBRMI: row masks beside whole hold-outs, and through replay" begin
    brmi = joint(flat)
    full = SBBRMI(brmi; mod=@__MODULE__)
    # an all-false mask holds nothing out; an all-true one holds the response out
    @test BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, held_out=(; y=falses(5)))) ==
          BRM.stan_code(full)
    @test SBBRMI(brmi; mod=@__MODULE__, held_out=(; y=falses(5))).held_out_rows == Dict()
    whole = SBBRMI(brmi; mod=@__MODULE__, held_out=(; y=trues(5)))
    @test whole.held_out == Set([:y]) && isempty(whole.held_out_rows)
    @test BRM.stan_code(whole) == BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, held_out=:y))

    both = SBBRMI(brmi; mod=@__MODULE__, held_out=[:z, :y => late])
    @test both.held_out == Set([:z]) && both.held_out_rows == Dict(:y => late)
    @test StanBlocks.stanc_check(BRM.stan_code(both); warn_pedantic=false).ok

    masked = SBBRMI(brmi; mod=@__MODULE__, held_out=(; y=late))
    # reprocess keeps the mask position by position
    replayed = reprocess(masked, merge(flat, (; x=flat.x .+ 0.1)))
    @test replayed.held_out_rows == Dict(:y => late)
    @test BRM.stan_code(replayed) == BRM.stan_code(masked)
    # refused: a mask names rows of the data it was given (held_out docs)
    longer = map(column -> vcat(column, column[1:1]), flat)
    @test_throws r"reprocess: the row mask for `held_out` response `y` has length 5" reprocess(
        masked, longer)

    plan = generative_plan(joint, flat; mod=@__MODULE__, held_out=(; y=late))
    @test plan.held_out_rows == Dict(:y => late)
    @test BRM.stan_code(plan) == BRM.stan_code(masked)
    @test generative_plan(plan, flat).held_out_rows == Dict(:y => late)
    @test reprocess(plan, flat).held_out_rows == Dict(:y => late)
    d = brm_descriptor(joint, flat; mod=@__MODULE__, held_out=(; y=late))
    @test d.plan.held_out_rows == Dict(:y => late)
    @test :fit in Set(op.name for op in d.operations)
end

@testset "per-row hold-outs not built yet" begin
    # not built yet: RKPPL does not skip `missing` entries inside per-subject arrays
    @test_broken (RKBRMI(incell(cell); held_out=(; y=cell_mask)); true)
end
