# S2Z contrast centering through WarmupHMC: cell metadata, exact coordinate
# transport against the compiled endpoint target, scoring-frame invariance,
# and online plus post-hoc fits. Run with
# `julia --project=test test/s2z_warmuphmc.jl`.
using Test, Random, LinearAlgebra, Statistics, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels

builder = @brm begin
    mu ~ 1 + x + (1 + x || g)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 2)
    y ~ Normal(mu, 0.5)
end
# Unbalanced groups: data-poor levels favour noncentered contrasts and
# data-rich levels centered ones.
sizes = [1, 2, 3, 4, 6, 8, 10, 14]
data = let rng = Xoshiro(11)
    g = vcat((fill(j, n) for (j, n) in enumerate(sizes))...)
    x = randn(rng, length(g))
    b0, b1 = 0.8 .* randn(rng, length(sizes)), 0.3 .* randn(rng, length(sizes))
    y = 1.0 .+ 0.5 .* x .+ b0[g] .+ b1[g] .* x .+ 0.5 .* randn(rng, length(g))
    (; x, g, y)
end
J, K = length(sizes), 2

compile(sb, tag) = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(mktempdir(), "s2z_whmc_$tag.stan"))
cells_of(sb, names) = BRM._s2z_centering_cells(sb, names)

# A physical (compiled-frame) point with distinct scales per coefficient.
function physical_point(sb, names; seed=3)
    coords = BRM._s2z_coordinates(sb, only(s2z_effect_blocks(sb)), names)
    x = 0.4 .* randn(Xoshiro(seed), length(names))
    x[coords.scales] .= log.([1.7, 0.4])
    x[coords.theta] .= [1.0, 0.5]
    x, coords
end

sb0 = SBBRMI(builder(data); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.0,
    total_groups=())
sbm = SBBRMI(builder(data); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=[0.0, 1.0],
    total_groups=())
# Group coordinates accept any compiled per-group centeredness.
cgroups = reshape(collect(range(0.0, 1.0; length=J * K)), J, K)
sbg = SBBRMI(builder(data); mod=@__MODULE__, s2z_groups=[:g], s2z_coordinates=:groups,
    s2z_rho=cgroups, total_groups=())
problem0 = compile(sb0, :ncp)
problemm = compile(sbm, :mixed)
problemg = compile(sbg, :groups)
names0 = BridgeStan.param_unc_names(problem0.model)
namesm = BridgeStan.param_unc_names(problemm.model)
namesg = BridgeStan.param_unc_names(problemg.model)
println("S2Z_WHMC_COMPILED ", names0); flush(stdout)

@testset "S2Z contrast cells" begin
    block = only(s2z_effect_blocks(sb0))
    coords = BRM._s2z_coordinates(sb0, block, names0)
    cells = cells_of(sb0, names0)
    @test cells.indices == vec(coords.contrasts)
    @test cells.scales == repeat(coords.scales; inner=J - 1)
    @test cells.locations == zeros(K * (J - 1))
    @test cells.targets == zeros(K * (J - 1))
    mixed = cells_of(sbm, namesm)
    @test mixed.targets == vcat(zeros(J - 1), ones(J - 1))
    groups = cells_of(sbg, namesg)
    gcoords = BRM._s2z_coordinates(sbg, only(s2z_effect_blocks(sbg)), namesg)
    @test groups.indices == vec(gcoords.contrasts)
    @test length(groups.indices) == J * K
    @test groups.targets == vec(cgroups)
    # Interior (Sean's partial map) weights have no scalar source frame.
    interior = SBBRMI(builder(data); mod=@__MODULE__, s2z_groups=[:g],
        s2z_rho=0.4, total_groups=())
    @test_throws ArgumentError cells_of(interior, names0)
    @test_throws ArgumentError adaptive_centering_problem(interior, problem0,
        AutoEnzyme(); unc_names=names0)
    # Models without S2Z blocks contribute no cells.
    plain = SBBRMI(builder(data); mod=@__MODULE__, total_groups=())
    @test isempty(cells_of(plain, names0).indices)
    @test_throws ArgumentError select_s2z_centeredness(plain, zeros(3, 1), ["a"])
end

@testset "Exact S2Z transport ($tag)" for (tag, sb, problem, names) in (
        (:ncp, sb0, problem0, names0), (:mixed, sbm, problemm, namesm),
        (:groups, sbg, problemg, namesg))
    physical, coords = physical_point(sb, names)
    cells = cells_of(sb, names)
    ell = physical[cells.scales]
    n = length(cells.indices)
    for c in (0.0, 0.37, 1.0, collect(range(0, 1; length=n)))
        rp = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=c)
        ir = WarmupHMC.reparametrizer(rp)
        @test length(ir.pairs) == n
        _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, physical)
        cs = c isa Real ? fill(c, n) : c
        # Source coordinates are power-interpolated contrasts; nothing else moves.
        @test source[cells.indices] ≈ physical[cells.indices] .* exp.((cs .- cells.targets) .* ell)
        rest = setdiff(eachindex(physical), cells.indices)
        @test source[rest] == physical[rest]
        jac, mapped = ir(source)
        @test mapped ≈ physical atol = 1e-12
        @test jac ≈ sum((cells.targets .- cs) .* ell) atol = 1e-12
        lp, g = LogDensityProblems.logdensity_and_gradient(rp, source)
        @test lp ≈ LogDensityProblems.logdensity(problem, physical) + jac atol = 1e-10
        fd = map(eachindex(source)) do i
            h = zeros(length(source)); h[i] = 1e-5
            (LogDensityProblems.logdensity(rp, source + h) -
             LogDensityProblems.logdensity(rp, source - h)) / 2e-5
        end
        @test g ≈ fd atol = 1e-6 rtol = 1e-6
        # Candidate coordinates and gradients do not depend on the source frame
        # that supplied the physical observation.
        plan = WarmupHMC.candidate_scoring_plan(rp)
        frame = plan.prepare(ir, source, g)
        gphys = last(LogDensityProblems.logdensity_and_gradient(problem, physical))
        for (p, (index, value)) in enumerate(ir.pairs), candidate in (0.0, 0.5, 1.0)
            _, position, gradient = plan.score(frame, p, index, value,
                WarmupHMC.PartiallyCentered(candidate))
            t, e = cells.targets[p], ell[p]
            @test position ≈ physical[index] * exp((candidate - t) * e) atol = 1e-12
            @test gradient ≈ gphys[index] * exp((t - candidate) * e) atol = 1e-9
        end
    end
end

@testset "Totals and S2Z contrasts share one scalar plan" begin
    mixed_builder = @brm begin
        mu ~ 1 + (1 | g)
        log(sigma) ~ 1 + (1 | h)
        y ~ Normal(mu, sigma)
    end
    df = (; g=data.g, h=mod1.(eachindex(data.g), 3), y=data.y)
    sb = SBBRMI(mixed_builder(df); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.0,
        total_groups=[:h])
    @test length(total_effect_blocks(sb)) == 1
    @test length(s2z_effect_blocks(sb)) == 1
    problem = compile(sb, :totals)
    names = BridgeStan.param_unc_names(problem.model)
    totals = BRM._total_centering_cells(sb, names)
    s2z = cells_of(sb, names)
    @test length(totals.indices) == 3 && length(s2z.indices) == J - 1
    c = collect(range(0.1, 0.9; length=length(totals.indices) + J - 1))
    rp = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=c)
    ir = WarmupHMC.reparametrizer(rp)
    @test first.(ir.pairs) == vcat(totals.indices, s2z.indices)
    physical = 0.3 .* randn(Xoshiro(8), length(names))
    _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, physical)
    jac, mapped = ir(source)
    @test mapped ≈ physical atol = 1e-12
    lp, g = LogDensityProblems.logdensity_and_gradient(rp, source)
    @test lp ≈ LogDensityProblems.logdensity(problem, physical) + jac atol = 1e-10
    fd = map(eachindex(source)) do i
        h = zeros(length(source)); h[i] = 1e-5
        (LogDensityProblems.logdensity(rp, source + h) -
         LogDensityProblems.logdensity(rp, source - h)) / 2e-5
    end
    @test g ≈ fd atol = 1e-6 rtol = 1e-6
end

# End-to-end fits use balanced groups with a well-identified slope. On the
# unbalanced data above the slope scale forms a funnel: fixed CP diverges for
# both ordinary and S2Z coordinates, and per-contrast online controls still
# leave a few divergent transitions where per-group ordinary controls do not.
balanced = let rng = Xoshiro(12)
    g = repeat(1:J; inner=6)
    x = randn(rng, length(g))
    b0, b1 = 0.8 .* randn(rng, J), 0.6 .* randn(rng, J)
    (; x, g, y=1.0 .+ 0.5 .* x .+ b0[g] .+ b1[g] .* x .+ 0.5 .* randn(rng, length(g)))
end

@testset "Online and post-hoc S2Z contrast fits" begin
    sb = SBBRMI(builder(balanced); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.0,
        total_groups=())
    problem = compile(sb, :balanced)
    names = BridgeStan.param_unc_names(problem.model)
    physical, coords = physical_point(sb, names)
    physical[vec(coords.contrasts)] .= 0.0
    rp = adaptive_centering_problem(sb, problem, AutoEnzyme())
    _, initial = WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(rp), physical)
    fit = adaptive_warmup_mcmc(Xoshiro(27), rp; init=initial, n_draws=2000,
        nonlinear_adapt=true, monitor_ess=false,
        callback=(state, stage) -> begin
            println("S2Z_WHMC_BOUNDARY ", stage, " window=", state.outer_counter)
            flush(stdout); false
        end)
    @test size(fit.posterior_position, 2) >= 2000
    @test all(isfinite, fit.posterior_position)
    @test fit.n_divergent_samples == 0
    sources = [last(p).c for p in WarmupHMC.reparam_sources(rp)]
    @test all(c -> 0 <= c <= 1, sources)
    # Adaptation moved away from the compiled noncentered frame.
    @test any(>(0), sources)
    println("S2Z_ONLINE_COMPLETE sampling_gradients=", fit.sampling_evaluation_counter,
        " centeredness=", round.(sources; digits=1)); flush(stdout)
    pilot = permutedims(fit.posterior_position)
    # Returned draws are compiled-frame, so the usual recovery applies.
    recovered = recover_s2z_draws(sb, pilot, names; rng=Xoshiro(5))[:mu]
    @test size(recovered.population) == (size(pilot, 1), 2)
    @test all(isfinite, recovered.population)
    @test abs(mean(recovered.population[:, 1]) - 1.0) < 0.6
    @test abs(mean(recovered.population[:, 2]) - 0.5) < 0.6
    gradients = permutedims(hcat([last(LogDensityProblems.logdensity_and_gradient(
        problem, collect(row))) for row in eachrow(pilot)]...))
    for criterion in (:position, :gradient)
        selected = select_s2z_centeredness(sb, pilot, names; criterion, gradients)
        @test selected.indices == cells_of(sb, names).indices
        @test size(selected.losses) == (K * (J - 1), 11)
        @test all(c -> 0 <= c <= 1, selected.centeredness)
        fixed = adaptive_centering_problem(sb, problem, AutoEnzyme();
            centeredness=selected.centeredness)
        _, start = WarmupHMC._inverse_with_logabsdet_jacobian(
            WarmupHMC.reparametrizer(fixed), physical)
        refit = adaptive_warmup_mcmc(Xoshiro(29), fixed; init=start, n_draws=2000,
            nonlinear_adapt=false, monitor_ess=false)
        @test all(isfinite, refit.posterior_position)
        @test refit.n_divergent_samples == 0
        println("S2Z_POSTHOC_COMPLETE criterion=", criterion,
            " sampling_gradients=", refit.sampling_evaluation_counter,
            " centeredness=", selected.centeredness); flush(stdout)
    end
end

# Per-group controls on the unbalanced data: contrast controls mix data-rich
# and data-poor groups and leave divergent transitions (15 and 12 of 2000 on
# two seeds when measured); group coordinates adapt each level on its own
# (1, 1 and 0 on three seeds).
@testset "Per-group S2Z centering on unbalanced groups" begin
    sbn = SBBRMI(builder(data); mod=@__MODULE__, s2z_groups=[:g],
        s2z_coordinates=:groups, total_groups=())
    problem = compile(sbn, :groups_ncp)
    names = BridgeStan.param_unc_names(problem.model)
    fits = map(((:groups, sbn, problem), (:contrasts, sb0, problem0))) do (tag, sb, p)
        rp = adaptive_centering_problem(sb, p, AutoEnzyme())
        fit = adaptive_warmup_mcmc(Xoshiro(27), rp; n_draws=2000,
            nonlinear_adapt=true, monitor_ess=false)
        sources = [last(q).c for q in WarmupHMC.reparam_sources(rp)]
        println("S2Z_UNBALANCED ", tag, " divergent=", fit.n_divergent_samples,
            " sampling_gradients=", fit.sampling_evaluation_counter,
            " centeredness=", round.(sources; digits=1)); flush(stdout)
        (; fit, sources)
    end
    groups, contrasts = fits
    @test all(isfinite, groups.fit.posterior_position)
    @test groups.fit.n_divergent_samples < contrasts.fit.n_divergent_samples
    @test groups.fit.n_divergent_samples <= 5
    # The best-informed intercept level ends up more centered than the least.
    @test groups.sources[J] > groups.sources[1]
    recovered = recover_s2z_draws(sbn, permutedims(groups.fit.posterior_position), names;
        rng=Xoshiro(6))[:mu]
    @test all(isfinite, recovered.effects)
    @test maximum(abs, sum(recovered.deviations; dims=2)) < 1e-9
end

# ---- Sean's rule per window (s2z_rule=:fisher) ----

@testset "Sean-map reparametrization is exact" begin
    # An interior compiled frame: the map goes source -> physical -> target.
    ctarget = 0.3
    sbr = SBBRMI(builder(data); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=ctarget,
        total_groups=())
    problem = compile(sbr, :interior)
    names = BridgeStan.param_unc_names(problem.model)
    csource = reshape(collect(range(0.05, 0.95; length=J * K)), J, K)
    rp = adaptive_centering_problem(sbr, problem, AutoEnzyme(); s2z_rule=:fisher,
        centeredness=[csource])
    ir = WarmupHMC.reparametrizer(rp)
    @test WarmupHMC.reparam_sources(rp) == [csource]
    x, coords = physical_point(sbr, names; seed=9)
    ljac, y = ir(x)
    _, back = WarmupHMC._inverse_with_logabsdet_jacobian(ir, y)
    @test back ≈ x atol = 1e-10
    tau = exp.(x[coords.scales])
    expected = 0.0
    for k in 1:K
        # Same physical deviations from both frames.
        from = BRM._s2z_partial_forward(BRM._s2z_helmert_mul(x[coords.contrasts[:, k]]),
            tau[k], csource[:, k])
        to = BRM._s2z_partial_forward(BRM._s2z_helmert_mul(y[coords.contrasts[:, k]]),
            tau[k], fill(ctarget, J))
        @test from ≈ to atol = 1e-10
        expected += BRM._s2z_partial_logjac(tau[k], csource[:, k]) -
                    BRM._s2z_partial_logjac(tau[k], fill(ctarget, J))
    end
    @test ljac ≈ expected atol = 1e-10
    rest = setdiff(eachindex(x), vec(coords.contrasts))
    @test y[rest] == x[rest]
    lp, g = LogDensityProblems.logdensity_and_gradient(rp, x)
    @test lp ≈ LogDensityProblems.logdensity(problem, y) + ljac atol = 1e-9
    fd = map(eachindex(x)) do i
        h = zeros(length(x)); h[i] = 1e-5
        (LogDensityProblems.logdensity(rp, x + h) - LogDensityProblems.logdensity(rp, x - h)) / 2e-5
    end
    @test g ≈ fd atol = 1e-6 rtol = 1e-6
    @test_throws ArgumentError adaptive_centering_problem(sbg, problemg, AutoEnzyme();
        s2z_rule=:fisher)
end

@testset "Sean's per-draw candidate and likelihood plans" begin
    plan = only(BRM._s2z_information_plans(sb0))
    @test plan.family === :gaussian
    info = BRM._s2z_row_information(plan, name -> error("no lookup needed"))
    @test info ≈ fill(4.0, length(data.y))           # Normal(mu, 0.5)
    tau = [1.3, 0.4]
    infos = [zeros(K, K) for _ in 1:J]
    for n in eachindex(data.g)
        z = [1.0, data.x[n]]
        infos[data.g[n]] .+= 4.0 .* (z * z')
    end
    @test BRM._s2z_fisher_draw(plan, info, tau) ≈ BRM._s2z_fisher_candidate(infos, tau)
    df = (; x=data.x, g=data.g, b=Int.(data.y .> 1), c=round.(Int, exp.(data.y ./ 2)),
        sigma_data=fill(0.7, length(data.y)), y=data.y)
    bern = @brm begin
        eta ~ 1 + x + (1 + x || g)
        b ~ BernoulliLogit(eta)
    end
    pois = @brm begin
        lam ~ 1 + x + (1 + x || g)
        c ~ Poisson(exp(lam))
    end
    datascale = @brm begin
        mu ~ 1 + x + (1 + x || g)
        y ~ Normal(mu, sigma_data)
    end
    shifted = @brm begin
        mu ~ 1 + x + (1 + x || g)
        y ~ Normal(mu + 1.0, 0.5)
    end
    s2z(b) = SBBRMI(b(df); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.0, total_groups=())
    pb = only(BRM._s2z_information_plans(s2z(bern)))
    @test pb.family === :bernoulli_logit
    eta = collect(range(-2, 2; length=length(df.b)))
    @test BRM._s2z_row_information(pb, n -> n === :eta ? eta : error()) ≈
        BRM.logistic.(eta) .* (1 .- BRM.logistic.(eta))
    pp = only(BRM._s2z_information_plans(s2z(pois)))
    @test pp.family === :poisson_log
    @test BRM._s2z_row_information(pp, n -> n === :lam ? eta : error()) ≈ exp.(eta)
    pd = only(BRM._s2z_information_plans(s2z(datascale)))
    @test BRM._s2z_row_information(pd, n -> n === :sigma_data ? df.sigma_data : error()) ≈
        fill(1 / 0.49, length(df.y))
    # A predictor inside another expression is not a supported pattern.
    @test_throws ArgumentError BRM._s2z_information_plans(s2z(shifted))
end

@testset "Per-window Sean rule on unbalanced groups ($evidence)" for evidence in
        (:linear_pool, :nuts_weighted)
    rp = adaptive_centering_problem(sb0, problem0, AutoEnzyme(); s2z_rule=:fisher)
    fit = adaptive_warmup_mcmc(Xoshiro(27), rp; n_draws=2000, nonlinear_adapt=true,
        monitor_ess=false, nonlinear_evidence=evidence)
    weights = only(WarmupHMC.reparam_sources(rp))
    println("S2Z_FISHER ", evidence, " divergent=", fit.n_divergent_samples,
        " sampling_gradients=", fit.sampling_evaluation_counter,
        " intercept_rho=", round.(weights[:, 1]; digits=2),
        " slope_rho=", round.(weights[:, 2]; digits=2)); flush(stdout)
    @test all(isfinite, fit.posterior_position)
    @test all(w -> 0 <= w <= 1, weights)
    @test weights != zeros(J, K)                          # the rule selected
    # Better-informed intercept levels get more centered weights.
    @test weights[J, 1] > weights[1, 1]
    recovered = recover_s2z_draws(sb0, permutedims(fit.posterior_position), names0;
        rng=Xoshiro(4))[:mu]
    @test all(isfinite, recovered.effects)
    @test abs(mean(recovered.population[:, 1]) - 1.0) < 0.6
end
