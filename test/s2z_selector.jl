using Test
using BayesianRegressionModels, StanBlocks
using Statistics

const BRM = BayesianRegressionModels

s2z_stanc_ok(sb) = begin
    m = sb.model
    StanBlocks.transpiles(m) &&
        StanBlocks.stanc_check(StanBlocks.stan_code(m); warn_pedantic=false).ok
end

normal_builder_intercept() = eval(:(@brm begin
    mu ~ 1 + (1 | g)
    effect(mu, Intercept) ~ Normal(0, 5)
    y ~ Normal(mu, 1)
end))

normal_builder() = eval(:(@brm begin
    mu ~ 1 + x + (1 + x || g)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 2)
    y ~ Normal(mu, 1)
end))

const SEL_DATA = (; x=[0.0, 1.0, 0.0, 1.0, 2.0, 3.0], g=[1, 1, 2, 2, 3, 3],
    y=[0.0, 1.0, 2.0, 3.0, 1.0, 2.0])

@testset "S2Z selector: closed form on intercept-only pilot" begin
    pilot = SBBRMI(normal_builder_intercept()(SEL_DATA); total_groups=())
    sb = SBBRMI(normal_builder_intercept()(SEL_DATA);
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
    block = only(BRM.s2z_effect_blocks(sb))
    # The retained design is the verified Z matrix: a constant column here.
    @test block.design == ones(6, 1)
    # Carrier names resolve against the emitted Stan source, not convention.
    code = BRM.stan_code(pilot)
    @test occursin("r_mu_g_log_scale", code)
    names = ["r_mu_g_log_scale"]
    # Balanced groups, constant precision: raw = K / (1 + K), K = tau^2 * info.
    prec = fill(2.0, 3, 6)
    taus = [0.5, 1.0, 2.0]
    draws = reshape(log.(taus), 3, 1)
    sel = BRM.select_s2z_rho(sb, pilot, draws, names; obs_prec=prec, group=:g)
    @test sel.n_draws == 3
    @test sel.group == :g
    @test sel.columns == (:Intercept,)
    @test sel.carriers == ["r_mu_g_log_scale"]
    @test sel.sd ≈ [1.0] # median tau
    info = 2 * 2.0 # two rows per group at precision 2
    # Median of three ordered raws is the middle draw's raw, rescaled once.
    mid_raw = (1.0^2 * info) / (1 + 1.0^2 * info)
    @test sel.raw ≈ fill(mid_raw, 3, 1) atol = 1e-12
    @test sel.rho ≈ fill(mid_raw / (mid_raw + (1 - mid_raw) * 1.0), 3, 1) atol = 1e-12
    # Mean aggregation follows the same raw-then-rescale contract.
    sel_mean = BRM.select_s2z_rho(sb, pilot, draws, names;
        obs_prec=prec, group=:g, aggregate=:mean)
    raws = [(t^2 * info) / (1 + t^2 * info) for t in taus]
    mean_raw, mean_sd = sum(raws) / 3, sum(taus) / 3
    @test sel_mean.raw ≈ fill(mean_raw, 3, 1) atol = 1e-12
    @test sel_mean.sd ≈ [mean_sd]
    @test sel_mean.rho ≈
        fill(mean_raw / (mean_raw + (1 - mean_raw) * mean_sd), 3, 1) atol = 1e-12
end

@testset "S2Z selector: nocor plumbing and median-of-three" begin
    df = (; x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], g=SEL_DATA.g, y=SEL_DATA.y)
    pilot = SBBRMI(normal_builder()(df); total_groups=())
    sb = SBBRMI(normal_builder()(df);
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
    block = only(BRM.s2z_effect_blocks(sb))
    @test block.design ≈ hcat(ones(6), df.x)
    code = BRM.stan_code(pilot)
    @test occursin("r_mu_g__nocor__1_log_scale", code)
    @test occursin("r_mu_g__nocor__2_tau", code)
    # Name-based (not positional) carrier lookup: decoy column first.
    names = ["decoy_param", "r_mu_g__nocor__1_log_scale", "r_mu_g__nocor__2_tau.1"]
    taus1 = [0.5, 1.0, 2.0]
    taus2 = [0.25, 2.0, 4.0]
    draws = hcat(zeros(3), log.(taus1), log.(taus2))
    prec = fill(1.5, 3, 6)
    sel = BRM.select_s2z_rho(sb, pilot, draws, names; obs_prec=prec, group=:g)
    @test sel.carriers ==
        ["r_mu_g__nocor__1_log_scale", "r_mu_g__nocor__2_tau.1"]
    @test sel.sd ≈ [1.0, 2.0]
    # Median-of-three with ordered taus: output equals the single-draw
    # kernel candidate at the median draw. Kernel is the oracle here; the
    # selector's design/carrier/aggregation plumbing is under test.
    idx = [1, 1, 2, 2, 3, 3]
    Z = block.design
    for (k, tau) in ((1, 1.0), (2, 2.0))
        infos = [fill(sum(prec[2, n] * Z[n, k]^2 for n in findall(==(j), idx)),
            1, 1) for j in 1:3]
        @test sel.rho[:, k] ≈
            vec(BRM._s2z_fisher_candidate(infos, [tau])) atol = 1e-12
    end
    # Per-cell structure survives: unbalanced x column gives distinct rows.
    @test length(unique(sel.rho[:, 2])) == 3
end

@testset "S2Z selector: guards" begin
    pilot = SBBRMI(normal_builder()(SEL_DATA); total_groups=())
    sb = SBBRMI(normal_builder()(SEL_DATA);
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
    names = ["r_mu_g__nocor__1_log_scale", "r_mu_g__nocor__2_tau.1"]
    draws = repeat([0.0 0.0], 2, 1)
    prec = fill(1.0, 2, 6)
    @test_throws ArgumentError BRM.select_s2z_rho(sb, pilot, draws, names;
        obs_prec=prec, group=:g, aggregate=:mode)
    @test_throws DimensionMismatch BRM.select_s2z_rho(sb, pilot, draws, names[1:1];
        obs_prec=prec, group=:g)
    @test_throws DimensionMismatch BRM.select_s2z_rho(sb, pilot, draws, names;
        obs_prec=fill(1.0, 2, 5), group=:g)
    bad = copy(prec)
    bad[1, 1] = -1.0
    @test_throws ArgumentError BRM.select_s2z_rho(sb, pilot, draws, names;
        obs_prec=bad, group=:g)
    @test_throws ArgumentError BRM.select_s2z_rho(sb, pilot, draws, names;
        obs_prec=prec, group=:nonesuch)
    @test_throws ArgumentError BRM.select_s2z_rho(sb, pilot, draws, names;
        obs_prec=prec, group=:g, predictor=:nonesuch)
    @test_throws ArgumentError BRM.select_s2z_rho(sb, pilot, draws, ["a", "b"];
        obs_prec=prec, group=:g)
    # A pilot on row-reordered data fails the row-order check, not silently.
    perm = [6, 5, 4, 3, 2, 1]
    reordered = (; x=SEL_DATA.x[perm], g=SEL_DATA.g[perm], y=SEL_DATA.y[perm])
    pilot_reordered = SBBRMI(normal_builder()(reordered); total_groups=())
    @test_throws ArgumentError BRM.select_s2z_rho(sb, pilot_reordered, draws, names;
        obs_prec=prec, group=:g)
    # A pilot on different levels fails the levels check.
    other = (; x=SEL_DATA.x, g=[1, 1, 2, 2, 4, 4], y=SEL_DATA.y)
    pilot_other = SBBRMI(normal_builder()(other); total_groups=())
    @test_throws ArgumentError BRM.select_s2z_rho(sb, pilot_other, draws, names;
        obs_prec=prec, group=:g)
end

@testset "S2Z selector: per-cell refit wiring" begin
    pilot = SBBRMI(normal_builder()(SEL_DATA); total_groups=())
    sb = SBBRMI(normal_builder()(SEL_DATA);
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
    names = ["r_mu_g__nocor__1_log_scale", "r_mu_g__nocor__2_tau.1"]
    draws = [log(0.5) log(1.5); log(1.0) log(2.0); log(2.0) log(2.5)]
    prec = fill(1.5, 3, 6)
    sel = BRM.select_s2z_rho(sb, pilot, draws, names; obs_prec=prec, group=:g)
    @test size(sel.rho) == (3, 2)
    @test all(0 .<= sel.rho .<= 1)
    refit = SBBRMI(normal_builder()(SEL_DATA);
        s2z_groups=[:g], s2z_rho=sel.rho, total_groups=())
    @test only(BRM.s2z_effect_blocks(refit)).rho == sel.rho
    @test s2z_stanc_ok(refit)
    # The matrix gate validates shape, range and type loudly.
    @test_throws ArgumentError SBBRMI(normal_builder()(SEL_DATA);
        s2z_groups=[:g], s2z_rho=fill(0.5, 2, 2), total_groups=())
    bad = fill(0.5, 3, 2)
    bad[2, 1] = 1.5
    @test_throws ArgumentError SBBRMI(normal_builder()(SEL_DATA);
        s2z_groups=[:g], s2z_rho=bad, total_groups=())
    @test_throws ArgumentError SBBRMI(normal_builder()(SEL_DATA);
        s2z_groups=[:g], s2z_rho="half", total_groups=())
end
