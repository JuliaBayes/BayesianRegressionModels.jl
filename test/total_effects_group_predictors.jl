# test/total_effects_group_predictors.jl — group-level predictors in exact totals.
#
# A population column constant within each level of an exact-total block's
# grouping factor (a group-level predictor), or equal to a random slope column
# times a per-level value, is integrated out with the matched coefficients:
# group g's total has mean A * (W[g,:] .* beta). Covers selection, exactness
# against the conventional program's log density, generated quantities,
# recovery, the descriptor query, frozen replay and rebuild, transport,
# population draws, starts, the population program, descriptions and
# centering locations (snag exact-totals-gro-79490cbb).
#
# Run: julia --project=test test/total_effects_group_predictors.jl
using Test, Random, LinearAlgebra, Statistics, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
const BRM = BayesianRegressionModels

const J = 8
const GRP = repeat(1:J, inner=4)
const WG = [0., 1., 0., 1., 1., 0., 1., 0.]
const ZG = [0.4, -1.2, 0.9, 0.1, -0.6, 1.5, -0.3, 0.7]
const CG = ["a", "b", "c", "a", "b", "c", "a", "b"]
const SG = [0.5, -0.8, 1.1, 0.3, -1.4, 0.9, 0.2, -0.4]

function group_data(rng=Xoshiro(41))
    x = randn(rng, length(GRP))
    almost = ZG[GRP]
    almost[5] += 0.3               # not constant within group 2
    xw = x .* SG[GRP]              # random slope column times a per-level value
    y = 0.5 .+ 0.8 .* WG[GRP] .+ 0.3 .* ZG[GRP] .+ 0.5 .* x .+ randn(rng, J)[GRP] .+
        0.3 .* randn(rng, length(GRP))
    (; y, w=WG[GRP], z=ZG[GRP], c=CG[GRP], x, almost, xw, g=GRP)
end

const intercept_builder = @brm begin
    mu ~ 1 + w + z + x + almost + (1 | a | g)
    effect(mu, :) ~ Normal(0, 2)
    sd(:, a) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

const slope_builder = @brm begin
    mu ~ 1 + x + xw + (1 | a | g) + (0 + x | b | g)
    effect(mu, :) ~ Normal(0, 2)
    sd(:, a) ~ Normal(0, 1)
    sd(:, b) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

instantiate(sb, name) = StanBlocks.stan_instantiate(sb.model; path=joinpath(mktempdir(), name))

# The conventional program's log density at the conventional point the totals
# point and an arbitrary beta map to, minus beta's exact conditional density
# and the non-centred Jacobian: the totals program's marginal density.
function conventional_marginal(sb, block, conventional, conventional_model, names, x; shift=0.15)
    cnames = BridgeStan.param_unc_names(conventional_model)
    coords = BRM._total_coordinates(sb, block, names)
    conditional = BRM._total_conditional(block, coords, x)
    beta = conditional.mean .+ shift .* (-1) .^ eachindex(conditional.mean)
    labels = only(o for o in brm_descriptor(conventional).outputs if o.name === :pop_mu_beta_pop).labels
    sampled = [o for o in brm_descriptor(sb).outputs if o.name === :pop_mu_beta_pop]
    remaining = isempty(sampled) ? Symbol[] : only(sampled).labels
    position(label) = only(findall(==("pop_mu_beta_pop.$(only(findall(==(label), labels)))"), cnames))
    q = zeros(length(cnames))
    for (p, label) in enumerate(block.population_columns)
        q[position(label)] = beta[p]
    end
    for (r, label) in enumerate(remaining)
        q[position(label)] = x[only(findall(==("pop_mu_beta_pop.$r"), names))]
    end
    blocks = adaptive_centering_blocks(conventional, cnames)
    mean = BRM._total_location(block, beta)
    for (k, id) in enumerate(length(block.columns) == 1 ? (:a,) : (:a, :b))
        cb = only(b for b in blocks if b.ranef.id === id)
        q[cb.log_scales] = x[coords.scales[k:k]]
        q[vec(cb.effects)] = (conditional.totals[:, k] .- mean[:, k]) ./ conditional.tau[k]
    end
    BridgeStan.log_density(conventional_model, q; propto=false) -
        size(conditional.totals, 1) * sum(log, conditional.tau) -
        logpdf(MvNormal(conditional.mean, inv(conditional.factor)), beta)
end

@testset "group-level columns join the intercept total" begin
    df = group_data()
    brmi = intercept_builder(df)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    block = only(total_effect_blocks(sb))
    labels = block.population_columns
    @test first(labels) === :Intercept
    @test :w in labels && :z in labels
    @test length(labels) == 3
    @test block.group_columns == labels[2:end]
    @test !(:x in labels) && !(:almost in labels)
    @test block.A == ones(1, 3)
    @test block.W[:, 1] == ones(J)
    @test block.W[:, findfirst(==(:w), labels)] == WG
    @test block.W[:, findfirst(==(:z), labels)] == ZG
    d = brm_descriptor(sb)
    @test only(o for o in d.outputs if o.name === :pop_mu_beta_pop).labels == [:x, :almost]
    code = BRM.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("brm_total(total_scale_mu, total_A_mu, total_W_mu,", code)

    problem = instantiate(sb, "group_totals.stan")
    names = BridgeStan.param_unc_names(problem.model)
    @test length(names) == J + 1 + 2
    conventional = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test isempty(total_effect_blocks(conventional))
    conventional_model = instantiate(conventional, "group_conventional.stan").model
    coords = BRM._total_coordinates(sb, block, names)
    q = zeros(length(names)); q[vec(coords.totals)] = 0.4 .* WG .+ 0.2 .* ZG
    for trial in 1:4
        x = q .+ 0.3randn(Xoshiro(70 + trial), length(q))
        lp = BridgeStan.log_density(problem.model, x; propto=false)
        @test lp ≈ conventional_marginal(sb, block, conventional, conventional_model, names, x) atol=1e-9
        @test lp ≈ conventional_marginal(sb, block, conventional, conventional_model, names, x; shift=-0.6) atol=1e-9
    end

    # Generated quantities: the deviations are the totals around each group's
    # own population mean, from the recovered coefficients of the same draw.
    cnames = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    gq = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=true,
                                    rng=BridgeStan.StanRNG(problem.model, 5))
    pick(stem) = gq[[i for (i, n) in enumerate(cnames) if startswith(n, stem * ".")]]
    beta = pick("population_mu")
    @test pick("deviation_mu") ≈ pick("total_mu") .- vec(BRM._total_location(block, beta)) atol=1e-12
    w = brm_population_effect_coordinates(d, :mu, cnames; coefficient=:w)
    @test w.recovered
    @test cnames[only(w.coordinates)] == "population_mu.$(findfirst(==(:w), labels))"
    @test !brm_population_effect_coordinates(d, :mu, cnames; coefficient=:x).recovered

    # Julia-side recovery has the same conditional law and deviation identity.
    draws = repeat(q', 4000, 1)
    recovered = recover_population_draws(sb, draws, names; rng=Xoshiro(913))[:mu]
    conditional = BRM._total_conditional(block, coords, q)
    @test vec(mean(recovered.population; dims=1)) ≈ conditional.mean atol=0.08
    @test cov(recovered.population) ≈ inv(conditional.factor) atol=0.08
    @test maximum(i -> maximum(abs, recovered.totals[i, :, :] .- recovered.deviations[i, :, :] .-
        BRM._total_location(block, recovered.population[i, :])), 1:4000) < 1e-12

    # Population draws put every group at its own mean A * (W[g,:] .* beta).
    pop = population_draws(sb, repeat(q', 3, 1), names; groups=:g, rng=Xoshiro(13))
    for row in eachrow(pop)
        totals = row[vec(coords.totals)]
        @test norm(block.W * (block.W \ totals) - totals) < 1e-10
    end

    # Starts place each total at its group's population value.
    u = brm_start(sb, problem.model, (; mu=(; Intercept=1.0, w=2.0, z=0.5)))
    @test u[vec(coords.totals)] ≈ 1.0 .+ 2.0 .* WG .+ 0.5 .* ZG atol=1e-12

    # Adaptive-centering cells are centred at each group's prior location.
    cells = BRM._total_centering_cells(sb, names)
    @test cells.locations == vec(BRM._total_location(block, block.location))

    # The population program reads each row's group-specific mean.
    pm = population_model(sb; groups=:g)
    pcode = BRM.stan_code(pm)
    @test occursin("brm_total_location(total_W_mu, total_A_mu, population_mu)[total_group_mu, :]", pcode)
    @test StanBlocks.stanc_check(pcode; warn_pedantic=false).ok

    # Descriptions name the group-level predictors and the group-specific mean.
    r = brm_description(sb)
    @test r.complete
    md = brm_description_markdown(r)
    @test occursin("are group-level predictors", md)
    @test occursin("A(\\mathbf w_{i}\\circ\\boldsymbol\\beta)", md)
end

@testset "frozen replay and rebuild keep the fitted group-level values" begin
    df = group_data()
    plan = generative_plan(intercept_builder, df; mod=@__MODULE__)
    sb = SBBRMI(intercept_builder(df); mod=@__MODULE__)
    block = only(total_effect_blocks(sb))
    replay = reprocess(sb, (; df..., x=df.x .+ 0.7, almost=df.almost .- 0.2))
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
    @test replay.data[:total_W_mu] == sb.data[:total_W_mu]
    changed = (; df..., w=[i == 6 ? 1 - v : v for (i, v) in enumerate(df.w)])
    @test_throws "group-level predictor" reprocess(sb, changed)
    shifted = (; df..., z=df.z .+ (df.g .== 3))
    @test_throws "group-level predictor" reprocess(sb, shifted)
    @test_throws ArgumentError reprocess(sb, df; freeze_constants=false)

    # Rebuild on two fitted groups and one new group.
    keep = findall(in((7, 8)), df.g)
    new = (; y=zeros(length(keep) + 2), w=vcat(df.w[keep], 1., 1.), z=vcat(df.z[keep], 0.3, 0.3),
           c=vcat(df.c[keep], "c", "c"), x=vcat(df.x[keep], 0.1, -0.2),
           almost=vcat(df.almost[keep], 0.3, 0.3), xw=zeros(length(keep) + 2), g=vcat(df.g[keep], 9, 9))
    newplan = generative_plan(plan, new)
    newblock = only(total_effect_blocks(newplan))
    @test newblock.population_columns == block.population_columns
    @test newblock.W[1:2, :] == block.W[7:8, :]
    @test newblock.W[3, :] == [1.0, 1.0, 0.3]
    bad = (; new..., w=vcat(1 .- new.w[1:end-2], 1., 1.))
    @test_throws "integrated it at" generative_plan(plan, bad)
    ragged = (; new..., z=vcat(new.z[1:end-2], 0.3, 0.4))
    @test_throws "must be the same on every row" generative_plan(plan, ragged)

    # Transport: fitted groups keep their totals; the new group is drawn
    # around its own population mean.
    problem = instantiate(sb, "group_transport.stan")
    names = BridgeStan.param_unc_names(problem.model)
    coords = BRM._total_coordinates(sb, block, names)
    q = zeros(length(names)); q[vec(coords.totals)] = 0.4 .* WG .+ 0.2 .* ZG
    q[only(coords.scales)] = log(0.5)
    newnames = vcat(filter(n -> !startswith(n, "total_mu."), names), ["total_mu.$g.1" for g in 1:3])
    moved = transport_draws(plan, newplan, repeat(q', 6000, 1), names, newnames; rng=Xoshiro(915))
    newcoords = BRM._total_coordinates(newplan, newblock, newnames)
    @test all(moved[:, newcoords.totals[1, 1]] .== q[coords.totals[7, 1]])
    conditional = BRM._total_conditional(block, coords, q)
    expected = only(BRM._total_location(newblock, conditional.mean)[3, :])
    @test mean(moved[:, newcoords.totals[3, 1]]) ≈ expected atol=0.08

    # A fit recorded before group-level absorption keeps its representation.
    legacy_preproc = Dict(k => v for (k, v) in plan.preproc if k !== :total_W_mu)
    legacy_preproc[:total_A_mu] = BRM.PreprocEntry(:total_basis, ones(1, 1), :mu, true)
    legacy = SBBRMI(intercept_builder(df); mod=@__MODULE__, total_groups=[:g],
                    _frozen_preproc=legacy_preproc)
    legacy_block = only(total_effect_blocks(legacy))
    @test legacy_block.population_columns == (:Intercept,)
    @test isempty(legacy_block.group_columns)
end

@testset "a random slope times a per-level value joins the slope total" begin
    df = group_data()
    brmi = slope_builder(df)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    block = only(total_effect_blocks(sb))
    @test block.columns == (:Intercept, :x)
    @test block.population_columns == (:Intercept, :x, :xw)
    @test block.group_columns == (:xw,)
    @test block.A == [1. 0. 0.; 0. 1. 1.]
    @test block.W[:, 3] ≈ SG atol=1e-14
    problem = instantiate(sb, "slope_totals.stan")
    names = BridgeStan.param_unc_names(problem.model)
    conventional = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    conventional_model = instantiate(conventional, "slope_conventional.stan").model
    coords = BRM._total_coordinates(sb, block, names)
    q = zeros(length(names)); q[vec(coords.totals)] = 0.3 .* randn(Xoshiro(3), 2J)
    for trial in 1:4
        x = q .+ 0.3randn(Xoshiro(80 + trial), length(q))
        @test BridgeStan.log_density(problem.model, x; propto=false) ≈
            conventional_marginal(sb, block, conventional, conventional_model, names, x) atol=1e-9
    end
    # A column proportional to the slope only up to one row stays sampled.
    broken = (; df..., xw=[i == 9 ? v + 0.5 : v for (i, v) in enumerate(df.xw)])
    @test only(total_effect_blocks(SBBRMI(slope_builder(broken); mod=@__MODULE__))).group_columns == ()
end
