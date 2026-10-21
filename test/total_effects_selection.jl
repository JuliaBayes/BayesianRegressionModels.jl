using Test, Random, LinearAlgebra, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, WarmupHMC, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import Enzyme
const BRM = BayesianRegressionModels

# Exact totals for some blocks of a group (snag total-groups-per-df179b1b,
# design decision 1rar7wo). Several per-unit linear predictors share one
# grouping and one `@plate` cell: `base` is the strongly identified margin,
# `c1` has no population coefficient to integrate. `total_groups=[:unit =>
# [:base]]` integrates `base` alone; the other blocks stay conventional.
const SELECTION_DATA = let rng = Xoshiro(20261021)
    J, n = 6, 8
    (; unit=["u$j" for j in 1:J],
       y=[0.5 .+ 0.3 .* randn(rng) .+ 0.1 .* randn(rng, n) for _ in 1:J],
       h=[cos.(2pi .* (1:n) ./ 24) for _ in 1:J],
       w=[rand(rng, n) for _ in 1:J])
end

selection_builder = @brm begin
    base ~ 1 + (1 | lv | unit)
    log(amp) ~ 1 + (1 | am | unit)
    log(noise) ~ 1 + (1 | ns | unit)
    c1 ~ 0 + (1 | cw | unit)
    effect(base, Intercept) ~ Normal(0.5, 1)
    sd(:, lv) ~ Exponential(1)
    effect(amp, Intercept) ~ Normal(0, 1)
    sd(:, am) ~ truncated(Normal(0.0, 1.0); lower = 0.0)
    effect(noise, Intercept) ~ Normal(-1, 0.5)
    sd(:, ns) ~ truncated(Normal(0.0, 0.5); lower = 0.0)
    sd(:, cw) ~ truncated(Normal(0.0, 0.2); lower = 0.0)
    @plate for i in eachindex(base)
        mu[i] = base[i] .+ c1[i] .* h[i] .+ amp[i] .* w[i]
        y[i] ~ Normal(mu[i], noise[i])
    end
end

predictors(sb) = sort!([b.predictor for b in total_effect_blocks(sb)]; by=string)
instantiate(sb, name) = StanBlocks.stan_instantiate(sb.model; path=joinpath(mktempdir(), "$name.stan"))

# log p_totals(x) = log p_conventional(beta, z) - J sum(log tau) - log p(beta | totals)
# for every beta: the totals integrate beta, and z = (total - location(beta)) / tau.
# Every coordinate outside `predictor`'s block has the same name in both programs.
function conventional_marginal(sb, original, x, names, predictor, id)
    block = only(b for b in total_effect_blocks(sb) if b.predictor === predictor)
    coords = BRM._total_coordinates(sb, block, names)
    conditional = BRM._total_conditional(block, coords, x)
    beta = conditional.mean .+ 0.3
    conventional, problem = original
    original_names = BridgeStan.param_unc_names(problem.model)
    ranef = only(b for b in adaptive_centering_blocks(conventional, original_names)
                 if occursin("_$(id)_", string(b.ranef.binding)))
    q = fill(NaN, length(original_names))
    for (i, name) in pairs(names)
        j = findfirst(==(name), original_names)
        isnothing(j) || (q[j] = x[i])
    end
    q[ranef.log_scales] = x[coords.scales]
    deviations = conditional.totals .- BRM._total_location(block, beta)
    q[ranef.effects[1, :]] = deviations[:, 1] ./ conditional.tau[1]
    # The block's population coefficient is the one conventional `pop_*`
    # coordinate the totals program lacks.
    population = findall(n -> startswith(n, "pop_") && !(n in names), original_names)
    @test length(population) == length(beta)
    q[population] = beta
    @test !any(isnan, q)
    J = size(conditional.totals, 1)
    BridgeStan.log_density(problem.model, q; propto=false) - J * sum(log, conditional.tau) -
        logpdf(MvNormal(conditional.mean, inv(conditional.factor)), beta)
end

@testset "total_groups selects exact totals for some blocks of a group" begin
    brmi = selection_builder(SELECTION_DATA)
    # `:auto` keeps the whole model conventional when a block must stay
    # conventional (decision 1rar7wo).
    @test isempty(total_effect_blocks(SBBRMI(brmi; mod=@__MODULE__)))

    # A whole-group request still covers every block, and its refusal names
    # each skipped predictor, why, and the subset that is available.
    err = try
        SBBRMI(brmi; mod=@__MODULE__, total_groups=[:unit]); nothing
    catch e
        sprint(showerror, e)
    end
    @test occursin("`c1`: it has no population coefficient to integrate", err)
    @test occursin("request `:unit => [:amp, :base, :noise]`", err)
    @test occursin("Gaussian grouping structure", err)

    one = SBBRMI(brmi; mod=@__MODULE__, total_groups=[:unit => [:base]])
    @test predictors(one) == [:base]
    @test BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=:unit => :base)) == BRM.stan_code(one)
    @test predictors(SBBRMI(brmi; mod=@__MODULE__, total_groups=(:unit => [:amp, :base],))) == [:amp, :base]
    code = BRM.stan_code(one)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("rows_dot_product(total_base", code)

    # Refused requests, each naming what is wrong.
    @test_throws "`c1`: it has no population coefficient" SBBRMI(brmi; mod=@__MODULE__,
        total_groups=[:unit => [:base, :c1]])
    @test_throws "`nope` has no random-effect block on `unit`" SBBRMI(brmi; mod=@__MODULE__,
        total_groups=[:unit => :nope])
    @test_throws "no random-effect block" SBBRMI(brmi; mod=@__MODULE__, total_groups=[:other => :base])
    @test_throws "both as a whole group" SBBRMI(brmi; mod=@__MODULE__,
        total_groups=[:unit, :unit => :base])
    @test_throws "`group => predictor(s)`" SBBRMI(brmi; mod=@__MODULE__, total_groups=[1])
    @test_throws "with Symbol names" SBBRMI(brmi; mod=@__MODULE__, total_groups=[:unit => Symbol[]])
    @test_throws "with Symbol names" SBBRMI(brmi; mod=@__MODULE__, total_groups=["unit" => :base])
end

@testset "the selected totals preserve the conventional posterior" begin
    brmi = selection_builder(SELECTION_DATA)
    conventional = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    original = (conventional, instantiate(conventional, "conventional"))
    # `amp` has a `truncated(Normal(0, 1); lower = 0)` scale prior: the same
    # half-line density as the conventional block's, up to its constant.
    for (predictor, id) in ((:base, :lv), (:amp, :am))
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=[:unit => predictor])
        @test predictors(sb) == [predictor]
        problem = instantiate(sb, "totals_$predictor")
        names = BridgeStan.param_unc_names(problem.model)
        for trial in 1:3
            x = 0.4 .* randn(Xoshiro(trial), length(names))
            @test BridgeStan.log_density(problem.model, x; propto=false) ≈
                conventional_marginal(sb, original, x, names, predictor, id) atol=1e-9
        end
    end
end

@testset "truncated scale priors: only a lower bound of zero" begin
    T(prior) = BRM._sb_total_scale_prior(prior)
    N = BRM.ExprColumn(Normal, 0.0, 1.0)
    @test T(BRM.ExprColumn(truncated, N; lower=0.0))
    @test T(BRM.ExprColumn(truncated, BRM.ExprColumn(Cauchy, 0.0, 2.0); lower=0))
    @test !T(BRM.ExprColumn(truncated, N; lower=0.5))
    @test !T(BRM.ExprColumn(truncated, N; lower=0.0, upper=3.0))
    @test !T(BRM.ExprColumn(truncated, N; upper=3.0))
    @test !T(BRM.ExprColumn(truncated, BRM.ExprColumn(Uniform, 0.0, 1.0); lower=0.0))
    @test T(BRM.ExprColumn(truncated, N; lower=0.0, upper=Inf))
    @test T(BRM.ExprColumn(truncated, N, 0.0, Inf))
    @test !T(BRM.ExprColumn(truncated, N, 0.0, 3.0))
    @test !T(BRM.ExprColumn(truncated, N, 0.5, Inf))
    # The positional spelling, read in a cell: `:auto` integrates the block,
    # and the density is the conventional marginal.
    positional = @brm SELECTION_DATA begin
        base ~ 1 + (1 | lv | unit)
        effect(base, Intercept) ~ Normal(0.5, 1)
        sd(:, lv) ~ truncated(Normal(0.0, 1.0), 0.0, Inf)
        @plate for i in eachindex(base)
            y[i] ~ Normal(base[i] .+ 0.2 .* w[i], 0.5)
        end
    end
    sb = SBBRMI(positional; mod=@__MODULE__)
    @test predictors(sb) == [:base]
    conventional = SBBRMI(positional; mod=@__MODULE__, total_groups=())
    original = (conventional, instantiate(conventional, "positional_conventional"))
    problem = instantiate(sb, "positional_totals")
    names = BridgeStan.param_unc_names(problem.model)
    for trial in 1:3
        x = 0.4 .* randn(Xoshiro(trial), length(names))
        @test BridgeStan.log_density(problem.model, x; propto=false) ≈
            conventional_marginal(sb, original, x, names, :base, :lv) atol=1e-9
    end
    shifted = @brm SELECTION_DATA begin
        base ~ 1 + (1 | lv | unit)
        sd(:, lv) ~ truncated(Normal(0.0, 1.0); lower = 0.5)
        @plate for i in eachindex(base)
            y[i] ~ Normal(base[i] .* w[i], 1.0)
        end
    end
    @test_throws "`sd(:, lv)` prior" SBBRMI(shifted; mod=@__MODULE__, total_groups=[:unit])
end

@testset "a rebuild for new groups keeps the fitted selection" begin
    plan = generative_plan(selection_builder, SELECTION_DATA; mod=@__MODULE__,
                           total_groups=[:unit => [:base]])
    @test predictors(plan) == [:base]
    future = let rng = Xoshiro(7)
        (; unit=["u2", "u9"], y=[randn(rng, 5) for _ in 1:2],
           h=[cos.(1:5) for _ in 1:2], w=[rand(rng, 5) for _ in 1:2])
    end
    rebuilt = generative_plan(plan, future)
    @test predictors(rebuilt) == [:base]
    @test StanBlocks.stanc_check(BRM.stan_code(rebuilt); warn_pedantic=false).ok
end

@testset "the joint WarmupHMC wrapper mixes totals with the group's conventional blocks" begin
    sb = SBBRMI(selection_builder(SELECTION_DATA); mod=@__MODULE__,
                total_groups=[:unit => [:base]])
    problem = instantiate(sb, "wrapper")
    names = BridgeStan.param_unc_names(problem.model)
    blocks = adaptive_centering_blocks(sb, names)
    # The three conventional blocks of `unit` stay ordinary adaptive cells.
    @test length(blocks) == 3
    rp = adaptive_centering_problem(sb, problem, AutoEnzyme())
    ir = WarmupHMC.reparametrizer(rp)
    @test length(ir.pairs) == 3 * length(SELECTION_DATA.unit) + length(SELECTION_DATA.unit)
    physical = 0.3 .* randn(Xoshiro(8), length(names))
    _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, physical)
    jac, mapped = ir(source)
    @test mapped ≈ physical atol=1e-10
    @test LogDensityProblems.logdensity(rp, source) ≈
        LogDensityProblems.logdensity(problem, physical) + jac atol=1e-8
end
