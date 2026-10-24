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

# A replay onto rows without a response the fit observed predicts that response
# given the fit, a forecast (decision `0soyu1b`, prong `forecast`): the response
# is bound to StanBlocks' values-free `maybecv(:y)`, so every parameter stays a
# parameter and the fit's draws feed it, the draw is shaped by the cell's other
# arguments, and there is no `y` input and no `y_likelihood`. The prior program
# stays a fresh build without the response.
using LogDensityProblems, BridgeStan

StanBlocks.@deffun begin
    washout_series(R::vector[T], rho::real)::vector[T] = begin
        W::vector[T]
        C = 0.0
        for t in 1:T
            C = C * exp(-rho) + R[t]
            W[t] = C
        end
        W
    end
end

# One response per group, pre-grouped (no random effect counts the groups).
const washout = @brm begin
    b ~ Normal(b_mu, b_sd)
    logA ~ Normal(0, 1)
    logsigma ~ Normal(-1, 1)
    A = exp(logA)
    sigma = exp(logsigma)
    @plate for i in eachindex(y)
        W = washout_series(R[i], rho)
        wash[i] = A .* W[cell[i]]
        mu[i] = b .+ wash[i]
        s_obs = sqrt.(sigma .^ 2 .+ (omega .* wash[i]) .^ 2)
        y[i] ~ Normal(mu[i], s_obs)
    end
end
const washout_constants = (; b_mu=0.1, b_sd=0.05, rho=0.5, omega=0.3)
const washout_fit = (; washout_constants..., y=[[0.1, 0.2, 0.15, 0.3], [0.2, 0.1]],
    R=[[0.0, 1.0, 2.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0]], cell=[[2, 3, 4, 6], [1, 3]])
const washout_rows = (; washout_constants...,
    R=[[0.0, 1.0, 2.0, 0.0, 0.5, 0.0, 3.0], [1.0, 0.0, 0.0, 2.0]], cell=[[5, 6, 7], [2, 3, 4]])

stan_block(code, name) = (k = findfirst("\n" * name * " {", code);
    code[k[1]+1:findnext("\n}", code, k[1] + 1)[end]])
operation_names(d) = [o.name for o in d.operations]
function generated_draws(sb, prefix; seed=1)
    problem = BRM.stan_instantiate(sb)
    model = problem.model
    n = LogDensityProblems.dimension(problem)
    names = BridgeStan.param_names(model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(model, fill(0.1, n); include_tp=true,
        include_gq=true, rng=BridgeStan.StanRNG(model, seed))
    [v for (name, v) in zip(names, values) if startswith(name, prefix * ".")]
end

@testset "a replay without the response predicts it given the fit" begin
    fit = SBBRMI(washout(washout_fit); mod=@__MODULE__)
    forecast = reprocess(fit, washout_rows)
    code = BRM.stan_code(forecast)
    @test forecast.held_out == Set([:y])
    @test !occursin(r"\by\b", stan_block(code, "data"))
    @test stan_block(code, "parameters") == stan_block(BRM.stan_code(fit), "parameters")
    @test stan_block(code, "model") == """
        model {
            b ~ normal(b_mu, b_sd);
            logA ~ normal(0, 1);
            logsigma ~ normal(-1, 1);
        }"""
    @test !occursin("y_likelihood", code)
    data = BRM.stan_data(forecast)
    @test !any(k -> k in (:y, :y_mem_n, :y_ends_n), keys(data))
    # the group count comes from the other per-group columns
    @test data[:kernel_nsub_mu] == 2
    d = brm_descriptor(forecast)
    @test :predict in operation_names(d)
    @test :fit ∉ operation_names(d)
    @test :pointwise_loglik ∉ operation_names(d)
    @test :y ∉ required_brm_inputs(d)
    draw = brm_output(d, :y)
    @test draw.name === :y_gen               # the fit program's name
    @test draw.segments == [3, 6]            # shaped by `cell`
    # the fit's unconstrained draws feed it
    fitted = BRM.stan_instantiate(fit)
    @test BridgeStan.param_unc_names(BRM.stan_instantiate(forecast).model) ==
          BridgeStan.param_unc_names(fitted.model)
    y_gen = generated_draws(forecast, "y_gen")
    @test length(y_gen) == 6 && all(isfinite, y_gen)
    # replayed onto rows that carry the response, the forecast is the fit's
    # hold-out replay
    rows_with_y = merge(washout_rows, (; y=[[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]]))
    @test BRM.stan_code(reprocess(forecast, rows_with_y)) ==
          BRM.stan_code(reprocess(fit, rows_with_y; held_out=:y))
    @test BRM.stan_code(reprocess(forecast, washout_rows)) == code
end

@testset "a forecast keeps every bound response conditioned" begin
    fit = SBBRMI(joint(flat); mod=@__MODULE__)
    partial = reprocess(fit, (; x=[3.0, 4.0], y=[0.1, 0.2]))
    @test partial.held_out == Set([:z])
    code = BRM.stan_code(partial)
    @test stan_block(code, "parameters") == stan_block(BRM.stan_code(fit), "parameters")
    @test occursin("y_likelihood", code) && !occursin("z_likelihood", code)
    @test length(generated_draws(partial, "z_gen")) == 2
    d = brm_descriptor(partial)
    @test :fit in operation_names(d)          # `y` is still observed
    @test required_brm_inputs(d) == [:x, :y]
    neither = reprocess(fit, (; x=[3.0, 4.0, 5.0]))
    @test neither.held_out == Set([:y, :z])
    @test :fit ∉ operation_names(brm_descriptor(neither))
    @test length(generated_draws(neither, "y_gen")) == 3
end

@testset "a forecast of a plate counted by its random effect" begin
    fit = SBBRMI(decay(decay_window); mod=@__MODULE__, total_groups=())
    scaled = only(k for (k, e) in fit.preproc if e.kind === :zscale)
    grid = (; subject=decay_data.subject, wt=decay_data.wt,
        t=[collect(0.25:0.25:1.0) for _ in 1:3])
    forecast = reprocess(fit, grid)
    # same program as scoring placeholder responses, minus the placeholders
    placeholders = reprocess(fit, merge(grid, (; y=[zeros(4) for _ in 1:3])); held_out=:y)
    for block in ("parameters", "model")
        @test stan_block(BRM.stan_code(forecast), block) ==
              stan_block(BRM.stan_code(placeholders), block)
    end
    @test !occursin("y_likelihood", BRM.stan_code(forecast))
    @test BRM.stan_data(forecast)[scaled] == BRM.stan_data(fit)[scaled]
    @test length(generated_draws(forecast, "y_gen")) == 12
    # A forecast that also re-draws new subjects is not built yet: both
    # re-emissions rebuild the program from the formula, which without the
    # response is the prior program (todo `1fnt1zu`).
    @test_broken !isnothing(reprocess(fit, grid; resample_groups=[:subject]))
end

const grouped_response = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        pred[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
        obs_y[obs[i]] ~ Normal(pred[i], sigma)
    end
end

@testset "a forecast of a response observed through a grouping" begin
    observed = (; subject=["s2", "s1", "s3"], obs_subject=["s1", "s2", "s3", "s2", "s3", "s3"],
        obs_idx=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0], obs_y=[0.11, 0.21, 0.31, 0.22, 0.32, 0.33])
    fit = SBBRMI(grouped_response(observed); mod=@__MODULE__)
    rows = (; subject=observed.subject, obs_subject=["s1", "s3", "s3"], obs_idx=[4.0, 4.0, 5.0])
    forecast = reprocess(fit, rows)
    code = BRM.stan_code(forecast)
    @test stan_block(code, "parameters") == stan_block(BRM.stan_code(fit), "parameters")
    @test !occursin("obs_y", stan_block(code, "data"))
    @test !occursin("_likelihood", code)
    @test :fit ∉ operation_names(brm_descriptor(forecast))
    @test length(generated_draws(forecast, "kernel_pred_obs_y_ragged_gen")) == 3
    # the next replay with the response gathers it again
    @test BRM.stan_code(reprocess(forecast, merge(rows, (; obs_y=[0.1, 0.2, 0.3])))) ==
          BRM.stan_code(reprocess(fit, merge(rows, (; obs_y=[0.1, 0.2, 0.3])); held_out=:obs_y))
end

const joint_response = @brm begin
    L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
    mu1 ~ 1 + x
    mu2 ~ 1 + x
    [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
end

@testset "a forecast of a joint response is not built yet" begin
    observed = (; x=[0.1, 0.5, 0.9, 1.3], y1=[0.2, 0.4, 0.1, 0.3], y2=[1.0, 0.8, 1.2, 0.9])
    fit = SBBRMI(joint_response(observed); mod=@__MODULE__)
    # The joint emission reads its rows through its own density, which the
    # values-free mark cannot stand in for (todo `1fnt1zu`); the replay names
    # the columns it lacks instead of failing on a missing field.
    @test_broken !isnothing(BRM.stan_code(reprocess(fit, (; x=[2.0, 3.0]))))
    @test BRM.stan_code(reprocess(fit, merge(observed, (; x=observed.x .+ 0.1)))) ==
          BRM.stan_code(fit)
end
