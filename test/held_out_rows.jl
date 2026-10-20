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
            # a held row outside LogNormal's support is never evaluated
            ("bounded", bounded, flat, (; y=late), keep_rows(flat, .!late)),
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

    # y_likelihood scores a held row outside LogNormal's support as -Inf, and
    # the draws still complete
    masked_sb = SBBRMI(bounded(flat); mod=@__MODULE__, held_out=(; y=late))
    masked = stan_problem(masked_sb, "bounded-masked")
    for u in stan_points(masked)
        rows = sb_execute(masked_sb, :pointwise_loglik, masked, u).y_likelihood
        @test all(isfinite, rows[1:4]) && rows[5] == -Inf
        @test length(sb_execute(masked_sb, :predict, masked, u).y_gen) == 5
    end

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
    # refused: the fit's mask names the fit's rows (held_out docs); the error
    # names the replay's own `held_out=`
    longer = map(column -> vcat(column, column[1:1]), flat)
    @test_throws r"reprocess: the row mask for `held_out` response `y` has length 5.*`held_out=\(; y = mask\)`, or `held_out=\(\)`" reprocess(
        masked, longer)

    # a replay onto data of another shape takes that data's own masks
    longer_late = vcat(late, [true])
    remasked = reprocess(masked, longer; held_out=(; y=longer_late))
    @test remasked.held_out_rows == Dict(:y => longer_late) && isempty(remasked.held_out)
    @test BRM.stan_code(remasked) == BRM.stan_code(masked)
    # the replay's masks replace the fit's: the fit's own mask leaves no trace
    @test BRM.stan_data(remasked) ==
          BRM.stan_data(reprocess(full, longer; held_out=(; y=longer_late)))
    # with fresh constants, the replay is the model built on that data and mask
    fresh = SBBRMI(joint(longer); mod=@__MODULE__, held_out=(; y=longer_late))
    @test BRM.stan_data(reprocess(full, longer; freeze_constants=false)) ==
          BRM.stan_data(SBBRMI(joint(longer); mod=@__MODULE__))
    @test BRM.stan_data(reprocess(masked, longer; freeze_constants=false,
        held_out=(; y=longer_late))) == BRM.stan_data(fresh)
    # `held_out=()` replays with nothing held out
    plain = reprocess(masked, longer; held_out=())
    @test isempty(plain.held_out) && isempty(plain.held_out_rows)
    @test BRM.stan_code(plain) == BRM.stan_code(full)
    @test BRM.stan_data(plain) == BRM.stan_data(reprocess(full, longer))
    @test restan_data(masked, longer; held_out=()) == BRM.stan_data(plain)
    # whole-response hold-outs ride the same keyword
    both_longer = reprocess(both, longer; held_out=[:z, :y => longer_late])
    @test both_longer.held_out == Set([:z]) && both_longer.held_out_rows == Dict(:y => longer_late)
    @test BRM.stan_code(both_longer) == BRM.stan_code(both)

    plan = generative_plan(joint, flat; mod=@__MODULE__, held_out=(; y=late))
    @test plan.held_out_rows == Dict(:y => late)
    @test BRM.stan_code(plan) == BRM.stan_code(masked)
    @test generative_plan(plan, flat).held_out_rows == Dict(:y => late)
    @test reprocess(plan, flat).held_out_rows == Dict(:y => late)
    @test reprocess(plan, longer; held_out=(; y=longer_late)).held_out_rows ==
          Dict(:y => longer_late)
    @test isempty(generative_plan(plan, longer; held_out=()).held_out_rows)
    d = brm_descriptor(joint, flat; mod=@__MODULE__, held_out=(; y=late))
    @test d.plan.held_out_rows == Dict(:y => late)
    @test :fit in Set(op.name for op in d.operations)
    @test_throws r"`held_out=\(\)`" brm_execute(d, :reprocess, longer)
    @test brm_execute(d, :reprocess, longer; held_out=(; y=longer_late)).plan.held_out_rows ==
          Dict(:y => longer_late)
    @test isempty(brm_execute(d, :replay, longer; held_out=()).plan.held_out_rows)
end

# A conditioning window over per-subject time courses observed in a `@plate`
# cell: fitted on the rows before a cutoff, then replayed onto a denser time
# grid (more rows per subject) and onto a new subject, each replay with the
# cutoff mask of its own rows. `zscale(wt)` is frozen preprocessing.
const decay = @brm begin
    sigma ~ Exponential(1)
    log_k ~ 1 + zscale(wt) + (1 | subject)
    @plate for i in eachindex(log_k)
        m = exp.(-exp(log_k[i]) .* t[i])
        y[i] ~ Normal(m, sigma)
    end
end
const decay_data = (; subject=[:a, :b, :c], wt=[60.0, 75.0, 90.0],
    t=[[0.5, 1.0, 2.0], [0.5, 1.0], [0.5, 1.0, 2.0, 4.0]],
    y=[[0.6, 0.4, 0.2], [0.7, 0.5], [0.8, 0.6, 0.3, 0.1]])
after_cutoff(t) = [ti .> 1.5 for ti in t]
const decay_grid = (; subject=decay_data.subject, wt=decay_data.wt,
    t=[collect(0.25:0.25:4.0) for _ in 1:3], y=[0.9 .^ (1:16) for _ in 1:3])
const decay_kept = (; subject=decay_grid.subject, wt=decay_grid.wt,
    t=[ti[.!m] for (ti, m) in zip(decay_grid.t, after_cutoff(decay_grid.t))],
    y=[yi[.!m] for (yi, m) in zip(decay_grid.y, after_cutoff(decay_grid.t))])
const decay_future = (; subject=[:a, :d], wt=[60.0, 120.0],
    t=[collect(0.5:0.5:3.0), collect(0.5:0.5:4.0)], y=[zeros(6), zeros(8)])

@testset "SBBRMI: a per-subject window replays onto data of another shape" begin
    mask = after_cutoff(decay_data.t)
    full = SBBRMI(decay(decay_data); mod=@__MODULE__, total_groups=())
    fit = SBBRMI(decay(decay_data); mod=@__MODULE__, total_groups=(), held_out=(; y=mask))
    @test fit.held_out_rows == Dict(:y => mask)
    # refused: the fit's mask names the fit's rows (held_out docs)
    @test_throws r"`held_out=\(; y = mask\)`, or `held_out=\(\)`" reprocess(fit, decay_grid)

    # the denser grid with its own cutoff is the window fit on the grid's kept rows
    grid_mask = after_cutoff(decay_grid.t)
    dense = reprocess(fit, decay_grid; held_out=(; y=grid_mask))
    @test dense.held_out_rows == Dict(:y => grid_mask)
    @test BRM.stan_code(dense) == BRM.stan_code(fit)
    @test BRM.stan_code(reprocess(fit, decay_grid; held_out=())) ==
          BRM.stan_code(reprocess(full, decay_grid))
    @test BRM.stan_data(reprocess(fit, decay_grid; held_out=())) ==
          BRM.stan_data(reprocess(full, decay_grid))
    masked = stan_problem(dense, "decay-dense-masked")
    reference = stan_problem(reprocess(full, decay_kept), "decay-dense-kept")
    @test unc_names(masked) == unc_names(reference)
    offsets = map(stan_points(masked)) do u
        value, gradient = LogDensityProblems.logdensity_and_gradient(masked, u)
        kept_value, kept_gradient = LogDensityProblems.logdensity_and_gradient(reference, u)
        @test isfinite(value)
        @test gradient ≈ kept_gradient atol=1e-8 rtol=1e-8
        value - kept_value
    end
    @test all(o -> isapprox(o, first(offsets); atol=1e-8), offsets)

    # a new subject re-drawn in generated quantities: `held_out=()` replays the
    # window fit exactly as the unmasked fit, and `zscale(wt)` keeps the fitted
    # constants, which a fresh build would not
    @test_throws r"`held_out=\(\)`" reprocess(fit, decay_future; resample_groups=[:subject])
    unmasked_future = reprocess(full, decay_future; resample_groups=[:subject])
    plain_future = reprocess(fit, decay_future; resample_groups=[:subject], held_out=())
    @test isempty(plain_future.held_out) && isempty(plain_future.held_out_rows)
    @test BRM.stan_code(plain_future) == BRM.stan_code(unmasked_future)
    @test BRM.stan_data(plain_future) == BRM.stan_data(unmasked_future)
    @test StanBlocks.stanc_check(BRM.stan_code(plain_future); warn_pedantic=false).ok
    scaled = only(k for (k, e) in fit.preproc if e.kind === :zscale)
    @test BRM.stan_data(plain_future)[scaled] !=
          BRM.stan_data(SBBRMI(decay(decay_future); mod=@__MODULE__, total_groups=()))[scaled]
    # the re-drawn subjects reach `y`, so cv contagion moves its whole
    # observation to generated quantities, which subsumes a row mask: the
    # replay is the unmasked fit's replay (StanBlocks snag element-hold-out-0f7241b8)
    future_mask = after_cutoff(decay_future.t)
    future = reprocess(fit, decay_future; resample_groups=[:subject],
        held_out=(; y=future_mask))
    @test future.held_out_rows == Dict(:y => future_mask)
    @test BRM.stan_code(future) == BRM.stan_code(unmasked_future)
    @test BRM.stan_data(future) == BRM.stan_data(reprocess(full, decay_future;
        resample_groups=[:subject], held_out=(; y=future_mask)))
    @test StanBlocks.stanc_check(BRM.stan_code(future); warn_pedantic=false).ok
    # so does the default replay of the window fit onto its own rows
    same = reprocess(fit, decay_data; resample_groups=[:subject])
    @test same.held_out_rows == Dict(:y => mask)
    @test BRM.stan_code(same) ==
          BRM.stan_code(reprocess(full, decay_data; resample_groups=[:subject]))
end

@testset "per-row hold-outs not built yet" begin
    # not built yet: RKPPL does not skip `missing` entries inside per-subject arrays
    @test_broken (RKBRMI(incell(cell); held_out=(; y=cell_mask)); true)
end
