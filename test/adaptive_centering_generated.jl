# test/adaptive_centering_generated.jl — adaptive centering on models whose
# omitted response puts a block or term in GENERATED QUANTITIES.
#
# Each joint model has one observed and one omitted response. No likelihood
# reaches the omitted response, so the traced program draws everything only it
# reads (its random-effect block, HSGP term or cdar walk, with their scales) in
# generated quantities: none of it has an unconstrained coordinate. Every
# adaptive builder skips such a block or term, as `transport_draws` and
# `population_draws` do, instead of failing to resolve it
# (todo `BayesianRegressionModels/todos/2026-10-16T10-26-57-051-1ktli6m`,
# snag `adaptive-centeri-747f09a0`).
#
# The oracle is a twin that never declares the omitted half: it has the same
# unconstrained coordinates and density, so its wrapper must equal the joint
# model's cell for cell, in transform, density and gradient.
#
# Run: julia --project=test test/adaptive_centering_generated.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal, Poisson

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)

rng = MersenneTwister(3)
n = 24
covariates = (; x = randn(rng, n), t = randn(rng, n), subject = repeat(11:16; inner = 4))
observed = merge(covariates, (; y = randn(rng, n)))   # `z` omitted: unobserved

# The sampled half of every joint model below.
twin_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
end
const TWIN = SBBRMI(twin_builder(observed); mod = @__MODULE__, total_groups = ())
const TWIN_PROBLEM = instantiate(TWIN)

# The joint model's wrapper equals the twin's, which never declared what the
# joint model emits in generated quantities. `rename` maps a twin coordinate
# name to the joint model's spelling; coordinates are matched by name, never by
# position.
function same_as_twin(joint, twin, twin_problem; rename = identity, centeredness = 0.3)
    problem = instantiate(joint)
    unc = BS.param_unc_names(problem.model)
    twin_unc = rename.(BS.param_unc_names(twin_problem.model))
    @test sort(unc) == sort(twin_unc)
    perm = [findfirst(==(name), twin_unc) for name in unc]
    rp = adaptive_centering_problem(joint, problem, AutoEnzyme(); centeredness)
    rt = adaptive_centering_problem(twin, twin_problem, AutoEnzyme(); centeredness)
    ir, it = WarmupHMC.reparametrizer(rp), WarmupHMC.reparametrizer(rt)
    @test !isempty(ir.pairs)
    @test unc[first.(ir.pairs)] == twin_unc[first.(it.pairs)]
    for seed in 1:3
        x = 0.5 .* randn(Xoshiro(seed), length(unc))
        xt = similar(x)
        xt[perm] = x
        @test LogDensityProblems.logdensity(problem, x) ≈
              LogDensityProblems.logdensity(twin_problem, xt) rtol = 1e-12
        (jac, mapped), (twin_jac, twin_mapped) = ir(x), it(xt)
        @test jac ≈ twin_jac rtol = 1e-12
        @test mapped ≈ twin_mapped[perm] rtol = 1e-12
        density, gradient = LogDensityProblems.logdensity_and_gradient(rp, x)
        twin_density, twin_gradient = LogDensityProblems.logdensity_and_gradient(rt, xt)
        @test density ≈ twin_density rtol = 1e-12
        @test gradient ≈ twin_gradient[perm] rtol = 1e-10
    end
    rp
end

@testset "ordinary block: the generated one is skipped, the wrapper equals the twin's" begin
    joint = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        nu ~ 1 + (1 | q | subject)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end)(observed); mod = @__MODULE__, total_groups = ())
    blocks = ranef_blocks(joint)
    @test [(b.binding, b.generated) for b in blocks] ==
          [(:b_p_subject, false), (:b_q_subject, true)]
    unc = BS.param_unc_names(instantiate(joint).model)
    @test !any(startswith("b_q_subject"), unc)
    adapted = adaptive_centering_blocks(joint, unc)
    @test [b.ranef.binding for b in adapted] == [:b_p_subject]
    @test only(adapted).effects == ranef_coordinates(first(blocks), unc)
    same_as_twin(joint, TWIN, TWIN_PROBLEM)
end

@testset "R2D2 block: a generated one is skipped before its scale is read" begin
    # Its reference scale `sigma2` is generated with it.
    joint = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        nu ~ 1 + (1 + t | q | subject)
        sd(:, q) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=sigma2)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end)(observed); mod = @__MODULE__, total_groups = ())
    @test [(b.family, b.generated) for b in ranef_blocks(joint)] ==
          [(:ranef_correlated_draws, false), (:ranef_correlated_draws_r2d2, true)]
    same_as_twin(joint, TWIN, TWIN_PROBLEM)
end

@testset "HSGP terms: generated ungrouped, grouped and periodic terms are skipped" begin
    hsgp_twin = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject) + hsgp(t; k=4)
        y ~ Normal(mu, sigma)
    end)(observed); mod = @__MODULE__, total_groups = ())
    joint = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject) + hsgp(t; k=4)
        nu ~ 1 + hsgp(x; k=4)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end)(observed); mod = @__MODULE__, total_groups = ())
    unc = BS.param_unc_names(instantiate(joint).model)
    @test [(b.logical, b.term) for b in BRM._adaptive_hsgp_centering_blocks(joint, unc)] ==
          [(:mu, :hsgp_t)]
    same_as_twin(joint, hsgp_twin, instantiate(hsgp_twin))

    grouped = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        nu ~ 1 + hsgp(x; k=4, by=subject)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end)(observed); mod = @__MODULE__, total_groups = ())
    same_as_twin(grouped, TWIN, TWIN_PROBLEM)

    periodic = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        nu ~ 1 + hsgp(t; k=2, cov=:periodic, period=24.0)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end)(observed); mod = @__MODULE__, total_groups = ())
    same_as_twin(periodic, TWIN, TWIN_PROBLEM)
end

@testset "cdar walk: a generated walk is skipped" begin
    walks = (; week = repeat(1:3; inner = 4), patch = repeat(["a", "b", "c", "d"]; outer = 3),
               y = randn(MersenneTwister(11), 12),
               C = [1.0 0.3 0.1 0.0; 0.3 1.0 0.2 0.1; 0.1 0.2 1.0 0.3; 0.0 0.1 0.3 1.0])
    cdar_twin = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, sigma)
    end)(walks); mod = @__MODULE__, total_groups = ())
    joint = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + cdar(week; by=patch, cor=C)
        nu ~ 1 + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end)(walks); mod = @__MODULE__, total_groups = ())
    unc = BS.param_unc_names(instantiate(joint).model)
    @test length(BRM._adaptive_cdar_centering_blocks(joint, unc)) == 1
    same_as_twin(joint, cdar_twin, instantiate(cdar_twin))
end

# A one-margin block read only by an omitted @plate response, beside a shared
# `|ID|` block whose omitted-response margin is integrated out: the shape of the
# snag's report.
plate_data = (; ags = repeat(1:6; inner = 4), fac = collect(1:24), w = randn(MersenneTwister(5), 24),
                n_m = [[1.0, 2.0, 3.0], [2.0, 1.0, 1.0], [3.0, 3.0, 2.0], [1.0, 1.0, 2.0]],
                micro_kreis = [1, 2, 3, 4], micro_fac = [1, 5, 9, 13],
                E = rand(MersenneTwister(7), 0:5, 24))
plate_joint = SBBRMI((@brm begin
    alpha_s ~ 1 + w + (1 | ks | ags) + (1 | fb | fac)
    alpha_c ~ 1 + (1 | kc | ags)
    lp_c ~ 0 + (1 | fb | fac)
    E ~ Poisson(exp(alpha_s))
    @plate for i in eachindex(n_m)
        lc = alpha_c[micro_kreis[i]] + lp_c[micro_fac[i]]
        y_m[i] ~ Poisson(n_m[i] .* exp.(lc))
    end
end)(plate_data); mod = @__MODULE__, total_groups = ())

@testset "@plate omitted response: the generated block is skipped, the active margin adapts" begin
    @test [(b.binding, b.generated) for b in ranef_blocks(plate_joint)] ==
          [(:b_ks_ags, false), (:b_fb_fac_active, false), (:b_fb_fac, true), (:b_kc_ags, true)]
    plate_twin = SBBRMI((@brm begin
        alpha_s ~ 1 + w + (1 | ks | ags) + (1 | fb | fac)
        E ~ Poisson(exp(alpha_s))
    end)(plate_data); mod = @__MODULE__, total_groups = ())
    # Without its second margin, the twin's `fb` block is an ordinary one-margin
    # block under its own name; the joint model samples it as the active margin.
    rp = same_as_twin(plate_joint, plate_twin, instantiate(plate_twin);
                      rename = name -> replace(name, "b_fb_fac_" => "b_fb_fac_active_"))
    unc = BS.param_unc_names(instantiate(plate_joint).model)
    sampled = filter(b -> !b.generated, ranef_blocks(plate_joint))
    @test first.(WarmupHMC.reparametrizer(rp).pairs) ==
          reduce(vcat, vec.(ranef_coordinates(sampled, unc)))

    online = adaptive_centering_problem(plate_joint, instantiate(plate_joint), AutoEnzyme())
    WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20261020), online;
        n_draws=20, n_evaluations=120, stepsize_adaptation_limit=20,
        max_tree_depth=7, progress=nothing, monitor_ess=false,
    )
    adapted = [value.c for (_, value) in WarmupHMC.reparam_sources(online)]
    @test length(adapted) == 30
    @test all(c -> c in 0.0:0.1:1.0, adapted)
end

@testset "a model whose every block is generated refuses, naming what it skipped" begin
    # refused: a generated block or term has no sampler coordinate, so there is
    # nothing to adapt (todo 1ktli6m acceptance: an actionable message, not the
    # `ranef_coordinates` error).
    nothing_sampled(builder; kw...) =
        (sb = SBBRMI(builder(observed); mod = @__MODULE__, kw...);
         adaptive_centering_problem(sb, instantiate(sb), AutoEnzyme()))
    ordinary = @brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x
        nu ~ 1 + (1 | q | subject) + hsgp(t; k=3)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end
    @test_throws "Emitted in generated quantities, because no observed likelihood reaches them (for example, their response is omitted): `b_q_subject`, `hsgp_t`." nothing_sampled(
        ordinary; total_groups = ())
    totals = @brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + x
        nu ~ 1 + (1 | subject)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end
    @test_throws "`total_nu`" nothing_sampled(totals)
end
