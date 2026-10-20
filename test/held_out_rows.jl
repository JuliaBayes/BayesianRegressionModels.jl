# Held-out rows: a fit receives only the rows it conditions on (user decision
# 1bwqnxs), so `held_out` takes no per-row masks. A window is fitted on its
# kept rows and its draws are replayed onto the rest with `reprocess`; a
# replay's own `held_out=` replaces the fit's whole-response hold-outs.
#
#   julia --startup-file=no --project=test test/held_out_rows.jl

using Test
using BayesianRegressionModels, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels

const flat = (; x=[1.0, 2.0, 0.5, 1.5, -0.3], y=[0.1, 0.2, 0.3, 0.5, -0.4],
    z=[0.0, 0.4, 0.2, 0.9, 0.1])
const late = [false, false, false, true, true]

const joint = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    nu ~ 1 + x
    z ~ Normal(nu, 1.0)
    y ~ Normal(mu, sigma)
end

@testset "per-row held_out masks are refused" begin
    brmi = joint(flat)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    # refused: a fit receives only the rows it conditions on; drop the other
    # rows from the data and replay the fit's draws onto them (user decision
    # 1bwqnxs, guiding principle 0q8hcwq)
    rows = r"`held_out` takes response Symbols, not per-row masks.*`reprocess`"
    for held_out in ((; y=late), :y => late, [:z, :y => late])
        @test_throws rows SBBRMI(brmi; mod=@__MODULE__, held_out)
        @test_throws rows RKBRMI(brmi; held_out)
        @test_throws rows generative_plan(joint, flat; mod=@__MODULE__, held_out)
        @test_throws rows reprocess(sb, flat; held_out)
    end
end

@testset "a replay's held_out replaces the fit's whole-response hold-outs" begin
    brmi = joint(flat)
    full = SBBRMI(brmi; mod=@__MODULE__)
    held = SBBRMI(brmi; mod=@__MODULE__, held_out=:z)
    shifted = merge(flat, (; x=flat.x .+ 0.1))
    kept = reprocess(held, shifted)
    @test kept.held_out == Set([:z])
    @test BRM.stan_code(kept) == BRM.stan_code(held)
    plain = reprocess(held, shifted; held_out=())
    @test isempty(plain.held_out)
    @test BRM.stan_code(plain) == BRM.stan_code(full)
    @test BRM.stan_data(plain) == BRM.stan_data(reprocess(full, shifted))
    @test restan_data(held, shifted; held_out=()) == BRM.stan_data(plain)
    swapped = reprocess(held, shifted; held_out=:y)
    @test swapped.held_out == Set([:y])
    @test BRM.stan_code(swapped) == BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, held_out=:y))

    plan = generative_plan(joint, flat; mod=@__MODULE__, held_out=:z)
    @test generative_plan(plan, shifted).held_out == Set([:z])
    @test isempty(generative_plan(plan, shifted; held_out=()).held_out)
    @test reprocess(plan, shifted; held_out=:y).held_out == Set([:y])
    d = brm_descriptor(joint, flat; mod=@__MODULE__, held_out=:z)
    @test brm_execute(d, :reprocess, shifted).plan.held_out == Set([:z])
    @test isempty(brm_execute(d, :replay, shifted; held_out=()).plan.held_out)
end

# A conditioning window over per-subject time courses observed in a `@plate`
# cell: the fit gets only the rows before a cutoff, and its draws replay onto a
# denser time grid and onto a new subject. `zscale(wt)` is frozen preprocessing.
const decay = @brm begin
    sigma ~ Exponential(1)
    log_k ~ 1 + zscale(wt) + (1 | subject)
    @plate for i in eachindex(log_k)
        m = exp.(-exp(log_k[i]) .* t[i])
        y[i] ~ Normal(m, sigma)
    end
end
before_cutoff(t) = [ti .<= 1.5 for ti in t]
const decay_data = (; subject=[:a, :b, :c], wt=[60.0, 75.0, 90.0],
    t=[[0.5, 1.0, 2.0], [0.5, 1.0], [0.5, 1.0, 2.0, 4.0]],
    y=[[0.6, 0.4, 0.2], [0.7, 0.5], [0.8, 0.6, 0.3, 0.1]])
const decay_window = (; subject=decay_data.subject, wt=decay_data.wt,
    t=[ti[k] for (ti, k) in zip(decay_data.t, before_cutoff(decay_data.t))],
    y=[yi[k] for (yi, k) in zip(decay_data.y, before_cutoff(decay_data.t))])
const decay_grid = (; subject=decay_data.subject, wt=decay_data.wt,
    t=[collect(0.25:0.25:4.0) for _ in 1:3], y=[0.9 .^ (1:16) for _ in 1:3])
const decay_future = (; subject=[:a, :d], wt=[60.0, 120.0],
    t=[collect(0.5:0.5:3.0), collect(0.5:0.5:4.0)], y=[zeros(6), zeros(8)])

@testset "a window fits its kept rows and replays onto the rest" begin
    fit = SBBRMI(decay(decay_window); mod=@__MODULE__, total_groups=())
    scaled = only(k for (k, e) in fit.preproc if e.kind === :zscale)
    for replay in (decay_data, decay_grid)
        replayed = reprocess(fit, replay)
        @test BRM.stan_code(replayed) == BRM.stan_code(fit)
        @test BRM.stan_data(replayed)[scaled] == BRM.stan_data(fit)[scaled]
        @test BRM.stan_data(replayed)[:y] ==
              BRM.stan_data(SBBRMI(decay(replay); mod=@__MODULE__, total_groups=()))[:y]
    end
    # a new subject re-drawn in generated quantities keeps the fitted
    # `zscale(wt)` constants, which a fresh build would not
    future = reprocess(fit, decay_future; resample_groups=[:subject])
    @test StanBlocks.stanc_check(BRM.stan_code(future); warn_pedantic=false).ok
    @test BRM.stan_data(future)[scaled] !=
          BRM.stan_data(SBBRMI(decay(decay_future); mod=@__MODULE__, total_groups=()))[scaled]
end
