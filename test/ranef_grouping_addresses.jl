# test/ranef_grouping_addresses.jl — `sd`/`cor`/`ranef` priors and known values
# for random-effect blocks written without an `|ID|`, addressed by their
# grouping column: plain `(… | g)` and uncorrelated `(… || g)` blocks (snag
# `sd-prior-uncorre-1002dc72`, user decision `06ntfvg`: the grouping-column
# address, and `sd(lp, ID)` setting every margin `lp` contributes).
#
# Run: julia --project=test test/ranef_grouping_addresses.jl [filter...]
# Set BRM_GROUPING_RUNTIME=0 to skip the BridgeStan and RK density checks.
#
# An addressed block lowers as a one-predictor `|ID|` block under a generated
# ID, so the checks pin it against the program a user gets by writing those
# IDs by hand, and the density of an addressed model whose priors restate the
# defaults against the unaddressed model's.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra
using Distributions: Exponential, LKJCholesky, LogNormal, Normal, TDist

const GROUPING_RUNTIME = get(ENV, "BRM_GROUPING_RUNTIME", "1") != "0"
const GROUPING_CACHE = joinpath(tempdir(), "brm-ranef-grouping-addresses")

Random.seed!(20261021)
const PROBES = repeat(1:6; inner=8)
const T = repeat(range(0, 1; length=8), 6)
const DF = (; probe=PROBES, site=repeat(1:3; inner=16),
            c1=round.(cos.(2pi .* T); digits=4), s1=round.(sin.(2pi .* T); digits=4),
            y=round.(randn(48); digits=3))

code(sb) = BRM.stan_code(sb)
build(b; kwargs...) = SBBRMI(b; mod=@__MODULE__, total_groups=(), kwargs...)
stanc_ok(sb) = StanBlocks.stanc_check(code(sb); warn_pedantic=false).ok
lines_with(c, needle) = [strip(l) for l in split(c, '\n') if occursin(needle, l)]

function instantiate(sb)
    mkpath(GROUPING_CACHE)
    BRM.stan_instantiate(sb; path=joinpath(GROUPING_CACHE, string(hash(code(sb))) * ".stan"))
end

# The reporter's shape: `||` for exact total coefficients, and a distributional
# scale with its own plain block on the same grouping column.
nocor = @brm DF begin
    mu ~ 1 + c1 + s1 + (1 + c1 + s1 || probe)
    log(sigma) ~ 1 + (1 | probe)
    y ~ Normal(mu, sigma)
end
addressed = @brm DF begin
    mu ~ 1 + c1 + s1 + (1 + c1 + s1 || probe)
    log(sigma) ~ 1 + (1 | probe)
    sd(:, probe) ~ Exponential(50.0)
    y ~ Normal(mu, sigma)
end
# The same model written with the IDs the addressed blocks lower under.
twin = @brm DF begin
    mu ~ 1 + c1 + s1 + (1 | mu__probe__nocor__1 | probe) +
         (c1 | mu__probe__nocor__2 | probe) + (s1 | mu__probe__nocor__3 | probe)
    log(sigma) ~ 1 + (1 | sigma__probe | probe)
    sd(:, mu__probe__nocor__1) ~ Exponential(50.0)
    sd(:, mu__probe__nocor__2) ~ Exponential(50.0)
    sd(:, mu__probe__nocor__3) ~ Exponential(50.0)
    sd(:, sigma__probe) ~ Exponential(50.0)
    y ~ Normal(mu, sigma)
end

@stestset "address resolution and the rewritten parent" begin
    # A model without grouping-column addresses is the model as written.
    @test BRM._brm_resolve_prior_addresses(nocor) === nocor

    sb = build(addressed)
    parent_brmi = parent(sb)
    @test [(s.class, s.id, s.predictor, s.coefficient) for s in ranef_effect_priors(parent_brmi)] ==
          [(:sd, :mu__probe__nocor__1, nothing, nothing),
           (:sd, :mu__probe__nocor__2, nothing, nothing),
           (:sd, :mu__probe__nocor__3, nothing, nothing),
           (:sd, :sigma__probe, nothing, nothing)]
    @test ranefcoefnames(parent_brmi, :mu__probe__nocor__2) ==
          [(; predictor=:mu, coefficient=:c1)]
    @test ranefcoefnames(parent_brmi, :sigma__probe) ==
          [(; predictor=:sigma, coefficient=:Intercept)]
    # Resolving the parent again changes nothing, so replay is stable.
    @test BRM._brm_resolve_prior_addresses(parent_brmi) === parent_brmi
    @test code(reprocess(sb, DF)) == code(sb)

    # The program is the hand-written split-ID program, conventional and with
    # exact totals.
    @test code(sb) == code(build(twin))
    @test code(SBBRMI(addressed; mod=@__MODULE__)) == code(SBBRMI(twin; mod=@__MODULE__))
    @test stanc_ok(sb)
end

@stestset "most-specific-wins across the blocks one column groups" begin
    layered = @brm DF begin
        mu ~ 1 + c1 + s1 + (1 + c1 + s1 || probe)
        log(sigma) ~ 1 + (1 | probe)
        sd(:, probe) ~ Exponential(50.0)
        sd(sigma, probe) ~ Exponential(1.0)
        sd(mu, probe, s1) = 0.5
        y ~ Normal(mu, sigma)
    end
    c = code(build(layered))
    @test "b_mu__probe__nocor__1_probe_tau ~ exponential((1.0 ./ 50.0));" in lines_with(c, "~")
    @test "b_mu__probe__nocor__2_probe_tau ~ exponential((1.0 ./ 50.0));" in lines_with(c, "~")
    @test "b_sigma__probe_probe_tau ~ exponential((1.0 ./ 1.0));" in lines_with(c, "~")
    @test isempty(lines_with(c, "b_mu__probe__nocor__3_probe_tau ~"))
    @test BRM.stan_data(build(layered))[:b_mu__probe__nocor__3_probe_tau] == [0.5]
    @test stanc_ok(build(layered))

    # A statement reaches only its own blocks; the rest stay as written.
    only_sigma = @brm DF begin
        mu ~ 1 + c1 + s1 + (1 + c1 + s1 || probe)
        log(sigma) ~ 1 + (1 | probe)
        sd(sigma, probe) ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end
    c = code(build(only_sigma))
    @test "r_mu_probe__nocor__1_log_scale ~ std_normal();" in lines_with(c, "~")
    @test "b_sigma__probe_probe_tau ~ exponential((1.0 ./ 1.0));" in lines_with(c, "~")

    # A reached lone intercept keeps its plain `LogNormal(0, 1)` default.
    one_slope = @brm DF begin
        mu ~ 1 + c1 + s1 + (1 + c1 + s1 || probe)
        sd(mu, probe, c1) ~ Exponential(2.0)
        y ~ Normal(mu, 1.0)
    end
    c = code(build(one_slope))
    @test "b_mu__probe__nocor__1_probe_tau ~ lognormal(0.0, 1.0);" in lines_with(c, "~")
    @test "b_mu__probe__nocor__2_probe_tau ~ exponential((1.0 ./ 2.0));" in lines_with(c, "~")
    @test "b_mu__probe__nocor__3_probe_tau ~ std_normal();" in lines_with(c, "~")

    # refused: two statements of equal specificity set one margin (`0g7cchs`).
    tie = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 | probe)
        sd(:, probe, c1) ~ Exponential(1.0)
        sd(mu, probe) ~ Exponential(2.0)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "equally specific" build(tie)
    no_margin = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 || probe)
        sd(mu, probe, s1) ~ Exponential(1.0)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "`sd(mu, probe, s1)` matches no random-effect margin" build(no_margin)
    bad_value = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 || probe)
        sd(mu, probe, c1) = -1.0
        y ~ Normal(mu, 1.0)
    end
    @test_throws "`sd(mu, probe, c1) = -1.0` must be a finite, strictly positive" build(bad_value)
    unknown = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 || probe)
        sd(:, nope) ~ Exponential(1.0)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "matches no shared" build(unknown)
end

@stestset "correlation and block laws" begin
    correlated = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 | probe)
        cor(:, probe) ~ LKJCholesky(2, 2.0)
        y ~ Normal(mu, 1.0)
    end
    c = code(build(correlated))
    @test "b_mu__probe_probe_L ~ lkj_corr_cholesky(2.0);" in lines_with(c, "~")
    # The block's unaddressed scales keep the half-normal default.
    @test only(lines_with(c, "b_mu__probe_probe_tau ~")) == "b_mu__probe_probe_tau ~ normal(0.0, 1.0);"

    two = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 | probe)
        log(sigma) ~ 1 + s1 + (1 + s1 | probe)
        cor(sigma, probe) = [1 0.3; 0.3 1]
        y ~ Normal(mu, sigma)
    end
    sb = build(two)
    @test BRM.stan_data(sb)[:b_sigma__probe_probe_L_known] ≈ Matrix(cholesky([1 0.3; 0.3 1]).L)
    @test "r_mu_probe_L ~ lkj_corr_cholesky(1.0);" in lines_with(code(sb), "~")
    @test stanc_ok(sb)
    # refused: `cor(:, g)` names the ONE correlated block grouped by `g`
    # (`06ntfvg`); with one per predictor the predictor must be named.
    ambiguous = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 | probe)
        log(sigma) ~ 1 + s1 + (1 + s1 | probe)
        cor(:, probe) ~ LKJCholesky(2, 2.0)
        y ~ Normal(mu, sigma)
    end
    @test_throws "`cor(:, probe)` is ambiguous" build(ambiguous)
    # refused: a `||` block has one margin and no correlation.
    uncorrelated = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 || probe)
        cor(:, probe) ~ LKJCholesky(2, 2.0)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "`||` blocks are uncorrelated" build(uncorrelated)
    wide = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 | probe)
        cor(mu, probe) ~ LKJCholesky(3, 2.0)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "does not match the addressed random-effect block width 2" build(wide)

    student = @brm DF begin
        mu ~ 1 + c1 + (1 | probe)
        log(sigma) ~ 1 + (1 | probe)
        ranef(:, probe) ~ TDist(4.0)
        ranef(sigma, probe) ~ Normal()
        y ~ Normal(mu, sigma)
    end
    c = code(build(student))
    @test "b_mu__probe_probe_w ~ inv_gamma((0.5 * 4.0), (0.5 * 4.0));" in lines_with(c, "~")
    @test "b_mu__probe_probe_tau ~ lognormal(0.0, 1.0);" in lines_with(c, "~")
    @test isempty(lines_with(c, "b_sigma__probe_probe_w"))
    @test stanc_ok(build(student))
end

@stestset "address boundaries" begin
    # refused: a name that is both an `|ID|` and an ID-less grouping column
    # has two meanings (`06ntfvg`).
    both = @brm DF begin
        mu ~ 1 + c1 + (1 | probe)
        log(sigma) ~ 1 + (1 | probe | site)
        sd(:, probe) ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end
    @test_throws "is both a shared `|probe|` random-effect ID and the grouping column" build(both)
    # An `|ID|` block grouped by the column is not reached by the column.
    by_id = @brm DF begin
        mu ~ 1 + c1 + (1 | probe) + (0 + c1 | p | probe)
        sd(:, probe) ~ Exponential(3.0)
        y ~ Normal(mu, 1.0)
    end
    c = code(build(by_id))
    @test "b_mu__probe_probe_tau ~ exponential((1.0 ./ 3.0));" in lines_with(c, "~")
    @test "b_p_probe_tau ~ std_normal();" in lines_with(c, "~")

    optional_model = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 || probe)
        sigma ~ Exponential(1.0)
        optional(sd(mu, probe)) ~ Exponential(2.0)
        optional(sd(tau, probe)) ~ Exponential(2.0)
        y ~ Normal(mu, sigma)
    end
    @test [s.id for s in ranef_effect_priors(parent(build(optional_model)))] ==
          [:mu__probe__nocor__1, :mu__probe__nocor__2]

    stratified = @brm DF begin
        mu ~ 1 + c1 + (1 | gr(probe, by=site))
        sd(:, probe) ~ Exponential(1.0)
        y ~ Normal(mu, 1.0)
    end
    # Covariance priors for stratified blocks are not built, with or without
    # an `|ID|`.
    @test_broken (build(stratified); true)
    joint = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 || probe)
        sd(:, probe) ~ r2d2(mean_R2=0.5, prec_R2=2.0, reference_scale=1.0)
        y ~ Normal(mu, 1.0)
    end
    # One R2D2 allocation across several independent blocks is not built.
    @test_broken (build(joint); true)
    single = @brm DF begin
        mu ~ 1 + c1 + (1 | probe)
        sd(:, probe) ~ r2d2(mean_R2=0.5, prec_R2=2.0, reference_scale=1.0)
        y ~ Normal(mu, 1.0)
    end
    @test stanc_ok(build(single))
end

@stestset "exact totals and other block emissions" begin
    auto = SBBRMI(addressed; mod=@__MODULE__)
    @test [(t.predictor, t.group, t.columns) for t in total_effect_blocks(auto)] ==
          [(:mu, :probe, (:Intercept, :c1, :s1)), (:sigma, :probe, (:Intercept,))]
    c = code(auto)
    @test "total_scale_mu_tau ~ exponential((1.0 ./ 50.0));" in lines_with(c, "~")
    @test "total_scale_sigma_tau ~ exponential((1.0 ./ 50.0));" in lines_with(c, "~")
    @test stanc_ok(auto)

    centered = build(addressed; centered_groups=(:probe,))
    @test all(b -> b.family === :ranef_correlated_draws_centered_generic,
              ranef_blocks(centered))
    @test stanc_ok(centered)
    @test stanc_ok(build(addressed; cv_groups=(:probe,)))
    s2z = SBBRMI(addressed; mod=@__MODULE__, total_groups=(), s2z_groups=(:probe,),
                 s2z_rho=0.5)
    @test occursin("s2z", code(s2z)) && stanc_ok(s2z)
end

GROUPING_RUNTIME && @stestset "BridgeStan: restated defaults are the unaddressed law" begin
    # Every scale prior restates the default of its block: the intercept's
    # `LogNormal(0, 1)` (rewritten automatically) and half-normal slopes.
    restated = @brm DF begin
        mu ~ 1 + c1 + s1 + (1 + c1 + s1 || probe)
        log(sigma) ~ 1 + (1 | probe)
        sd(mu, probe, c1) ~ Normal(0, 1)
        sd(sigma, probe) ~ LogNormal(0, 1)
        y ~ Normal(mu, sigma)
    end
    plain = instantiate(build(nocor)).model
    lowered = instantiate(build(restated)).model
    # Plain name -> lowered name of the same constrained value. A lone
    # intercept's `exp(log_scale)` is the lowered `tau`; the lowered blocks
    # also carry a 1x1 correlation factor, which is identically 1.
    rename = Dict{String,String}()
    for (k, block) in enumerate(("mu__probe__nocor__1", "mu__probe__nocor__2",
                                 "mu__probe__nocor__3"))
        plain_block = "r_mu_probe__nocor__$k"
        rename["$(plain_block)_$(k == 1 ? "log_scale" : "tau.1")"] = "b_$(block)_probe_tau.1"
        for j in 1:6
            rename["$(plain_block)_xi.$j"] = "b_$(block)_probe_z_flat.$j"
        end
    end
    # The plain block is named by the emitted predictor `log_sigma`, the
    # generated ID by the public one, `sigma`.
    rename["r_log_sigma_probe_log_scale"] = "b_sigma__probe_probe_tau.1"
    for j in 1:6
        rename["r_log_sigma_probe_xi.$j"] = "b_sigma__probe_probe_z_flat.$j"
    end
    plain_names = BridgeStan.param_names(plain)
    lowered_names = BridgeStan.param_names(lowered)
    rng = MersenneTwister(3)
    for _ in 1:3
        theta = 0.5 .* randn(rng, length(plain_names))
        values = Dict(zip(plain_names, theta))
        for (name, value) in values
            if endswith(name, "log_scale")
                values[name] = value   # a log scale: any real
            elseif occursin("_tau.", name)
                values[name] = exp(value)
            end
        end
        mapped = Dict{String,Float64}()
        for (name, value) in values
            target = get(rename, name, name)
            mapped[target] = endswith(name, "log_scale") ? exp(value) : value
        end
        for name in lowered_names
            endswith(name, "_L.1.1") && (mapped[name] = 1.0)
        end
        u_plain = BridgeStan.param_unconstrain(plain, [values[n] for n in plain_names])
        u_lowered = BridgeStan.param_unconstrain(lowered, [mapped[n] for n in lowered_names])
        @test BridgeStan.log_density(plain, u_plain; propto=false, jacobian=true) ≈
              BridgeStan.log_density(lowered, u_lowered; propto=false, jacobian=true) atol=1e-10 rtol=1e-12
    end
end

GROUPING_RUNTIME && @stestset "RKBRMI: the same law as SBBRMI" begin
    student = @brm DF begin
        mu ~ 1 + c1 + (1 + c1 | probe)
        log(sigma) ~ 1 + (1 | probe)
        sd(mu, probe) ~ Exponential(2.0)
        cor(mu, probe) ~ LKJCholesky(2, 2.0)
        ranef(sigma, probe) ~ TDist(5.0)
        y ~ Normal(mu, sigma)
    end
    # An addressed `||` block lowers as `|ID|` blocks, which RK samples.
    for (name, brmi) in (("addressed", addressed), ("student", student))
        rk, problem = consumer_problem(brmi)
        sb = build(brmi)
        stan = instantiate(sb)
        stan_names = BridgeStan.param_unc_names(stan.model)
        transport = brm_coordinate_transport(rk, sb, stan_names)
        @test length(transport) == length(coordinate_names(rk.model.layout)) == length(stan_names)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))))
            stan_u = brm_rk_point_to_stan(transport, u)
            checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
            @test checked.max_error <= 1e-12
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

GROUPING_RUNTIME && @stestset "start values and fitted scales take the generated ID" begin
    sb = build(addressed)
    model = instantiate(sb).model
    u = brm_start(sb, model, (; mu__probe__nocor__2=(; sd=0.7)))
    names = BridgeStan.param_names(model)
    values = BridgeStan.param_constrain(model, u)
    @test values[only(findall(==("b_mu__probe__nocor__2_probe_tau.1"), names))] ≈ 0.7
    d = brm_descriptor(sb)
    coordinate = brm_ranef_sd_coordinates(d, :mu, names; id=:mu__probe__nocor__2, coefficient=:c1)
    @test names[only(coordinate.coordinates)] == "b_mu__probe__nocor__2_probe_tau.1"
    @test brm_description(sb) isa Any
end
