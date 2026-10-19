# S2Z on one grouping of a predictor that carries further groupings (crossed
# or nested, `|ID|` buckets included) and direct terms (`hsgp`, categorical
# blocks), flat and read by index in `@plate for` cells (snag
# `s2z-on-one-group-ec52a3b0`). Only the selected block is reparameterised:
# every other term keeps its ordinary emission and coordinates, so the
# BridgeStan density equals the ordinary model's joint density times the
# omitted-mean conditional, exactly as in `s2z_emission.jl`. The S2Z (and
# total) cells then adapt in one WarmupHMC wrapper with the ordinary and HSGP
# cells.
#
# Run: julia --project=test test/s2z_mixed_terms.jl
using Test, Random, LinearAlgebra, Statistics, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)

s2z_stanc_ok(sb) = StanBlocks.transpiles(sb.model) &&
    StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok

instantiate(sb, tag) = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(mktempdir(), "$tag.stan")).model

# Population coefficient labels of predictor `lp` in coordinate order.
pop_labels(model, lp) = let b = get(model.bindings, Symbol(:pop_, lp), nothing)
    isnothing(b) ? Symbol[] :
        hasproperty(b, :design_columns) ? [c.label for c in b.design_columns] :
        collect(b.population_columns)
end

# The ordinary model's block(s) for S2Z block `block` of predictor `lp`, one
# per coefficient: its own plain block, the `|ID|` bucket named in `ids`, or
# the `g__nocor__k` margins of a `(… || g)` split.
function ordinary_blocks(block, oblocks, lp, ids)
    base(g) = let m = match(BRM._RANEF_NOCOR_RE, String(g))
        isnothing(m) ? String(g) : only(m.captures)
    end
    id = get(ids, lp, nothing)
    # One `|ID|` per coefficient when the block's columns come from separate
    # one-margin buckets.
    id isa AbstractVector && return [only(b for b in oblocks if b.ranef.id === i) for i in id]
    own = filter(oblocks) do b
        base(b.ranef.group) == String(block.group) || return false
        isnothing(id) ? occursin("_$(lp)_", "_$(b.ranef.binding)_") : b.ranef.id === id
    end
    K = length(block.columns)
    K == 1 && return [only(own)]
    [only(b for b in own if endswith(String(b.ranef.binding), "__nocor__$k")) for k in 1:K]
end

# The ordinary joint density of S2Z point `x` with each block's omitted mean
# placed `offset` away from its conditional mean, minus that conditional: the
# S2Z density whatever the offset.
function ordinary_equivalent(sb, model, names, ordinary_sb, ordinary, onames, x;
                             offset=0.3, ids=Dict{Symbol,Any}())
    opos = Dict(n => i for (i, n) in enumerate(onames))
    xo = fill(NaN, length(onames))
    claimed = Set{Int}()
    adjust = 0.0
    oblocks = adaptive_centering_blocks(ordinary_sb, onames)
    for block in s2z_effect_blocks(sb)
        lp = block.predictor
        coords = BRM._s2z_coordinates(sb, block, names)
        union!(claimed, coords.contrasts, coords.theta, coords.scales)
        J, K = size(block.rho)
        conditional = BRM._s2z_conditional(block, coords, x)
        tau = conditional.sd
        delta = reduce(hcat, [BRM._s2z_partial_forward(
            BRM._s2z_helmert_mul(Vector{Float64}(x[coords.contrasts[:, k]])),
            tau[k], block.rho[:, k]) for k in 1:K])
        m = conditional.mean .+ offset
        beta = conditional.theta - block.B * m
        labels = pop_labels(ordinary_sb, lp)
        for (p, label) in enumerate(block.population_columns)
            xo[opos["pop_$(lp)_beta_pop.$(findfirst(==(label), labels))"]] = beta[p]
        end
        for (j, label) in enumerate(pop_labels(sb, lp))
            i = findfirst(==("pop_$(lp)_beta_pop.$j"), names)
            push!(claimed, i)
            xo[opos["pop_$(lp)_beta_pop.$(findfirst(==(label), labels))"]] = x[i]
        end
        for (k, ob) in enumerate(ordinary_blocks(block, oblocks, lp, ids))
            ob.ranef.n_terms == 1 || error("this audit maps scalar ordinary blocks")
            xo[vec(ob.effects)] = (delta[:, k] .+ m[k]) ./ tau[k]
            xo[ob.log_scales] = x[coords.scales[k:k]]
        end
        cov = isempty(findall(>(0), block.precision)) ? Diagonal(tau .^ 2 ./ J) :
            Diagonal(conditional.prior_var) -
            conditional.gain * conditional.B1 * Diagonal(conditional.prior_var)
        adjust += -J * sum(log, tau) +
            sum(BRM._s2z_partial_logjac(tau[k], block.rho[:, k]) for k in 1:K) +
            0.5 * K * log(J) - logpdf(MvNormal(conditional.mean, Symmetric(cov)), m)
    end
    for (i, n) in enumerate(names)
        i in claimed && continue
        xo[opos[n]] = x[i]
    end
    any(isnan, xo) && error("unmapped ordinary coordinates: $(onames[isnan.(xo)])")
    BridgeStan.log_density(ordinary, xo; propto=false) + adjust
end

function audit(sb, ordinary_sb, tag; trials=3, ids=Dict{Symbol,Any}())
    model, ordinary = instantiate(sb, "s2z_$tag"), instantiate(ordinary_sb, "ordinary_$tag")
    names, onames = BridgeStan.param_unc_names(model), BridgeStan.param_unc_names(ordinary)
    # S2Z drops one coordinate per coefficient (J-1 contrasts + theta + tau
    # versus J effects + beta + log tau).
    @test length(names) ==
        length(onames) - sum(b -> length(b.columns), s2z_effect_blocks(sb))
    for trial in 1:trials
        x = 0.3 .* randn(Xoshiro(40 + trial), length(names))
        @test BridgeStan.log_density(model, x; propto=false) ≈
            ordinary_equivalent(sb, model, names, ordinary_sb, ordinary, onames, x; ids) atol = 1e-8
        # The identity holds at any omitted mean.
        @test ordinary_equivalent(sb, model, names, ordinary_sb, ordinary, onames, x;
                                  offset=-0.7, ids) ≈
            ordinary_equivalent(sb, model, names, ordinary_sb, ordinary, onames, x; ids) atol = 1e-8
    end
    q = 0.2 .* randn(Xoshiro(7), length(names))
    _, grad = BridgeStan.log_density_gradient(model, q; propto=false)
    fd = map(eachindex(q)) do i
        qp, qm = copy(q), copy(q)
        qp[i] += 1e-6
        qm[i] -= 1e-6
        (BridgeStan.log_density(model, qp; propto=false) -
         BridgeStan.log_density(model, qm; propto=false)) / 2e-6
    end
    @test fd ≈ grad atol = 1e-4
    gq = BridgeStan.param_constrain(model, q; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(model, 11))
    @test all(isfinite, gq)
    (; model, names)
end

const NR, NDIS, NOBS = 4, 12, 3
const FLAT = let rng = Xoshiro(3)
    district_g = repeat(1:NDIS, inner=NOBS)
    region_g = (district_g .- 1) .% NR .+ 1
    xc, yc = randn(rng, NDIS), randn(rng, NDIS)
    (; region_g, district_g, x=randn(rng, NDIS * NOBS),
       c3=repeat([1, 2, 3], NDIS), xc=xc[district_g], yc=yc[district_g],
       y=rand(rng, 0:4, NDIS * NOBS), yr=rand(rng, 0:4, NDIS * NOBS),
       yi=rand(rng, 0:4, NDIS * NOBS))
end

two_groupings = @brm begin
    mu ~ 1 + x + (1 | region_g) + (1 | district_g)
    y ~ Poisson(exp(mu))
end

# Intercept and slope as two one-margin `|ID|` buckets on the S2Z grouping.
id_two_groupings = @brm begin
    mu ~ 1 + x + (1 | a | region_g) + (0 + x | b | region_g) + (1 | district_g)
    sd(:, b) ~ Exponential(2)
    y ~ Poisson(exp(mu))
end

# Uncorrelated intercept and slope (`||` splits them into `region_g__nocor__k`
# margins) beside a second grouping.
nocor_two_groupings = @brm begin
    mu ~ 1 + x + (1 + x || region_g) + (1 | district_g)
    y ~ Poisson(exp(mu))
end

# The consumer's shape on flat rows: a categorical block, an HSGP and a nested
# `|ID|` bucket shared across two predictors beside the S2Z grouping.
margins = @brm begin
    alpha_r ~ 1 + x + factor(c3) + hsgp(xc, yc; k=3) + (1 | region_g) +
        (1 | kreis | district_g)
    alpha_i ~ 1 + x + hsgp(xc, yc; k=3) + (1 | region_g) + (1 | kreis | district_g)
    yr ~ Poisson(exp(alpha_r))
    yi ~ Poisson(exp(alpha_i))
end

@testset "S2Z beside a second grouping" begin
    brmi = two_groupings(FLAT)
    sb = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=0.5)
    block = only(s2z_effect_blocks(sb))
    @test block.group === :region_g
    @test block.population_columns == (:Intercept,)
    @test pop_labels(sb, :mu) == [:x]
    @test s2z_stanc_ok(sb)
    @test any(r -> r.group === :district_g, ranef_blocks(sb))
    @test !any(r -> r.group === :region_g, ranef_blocks(sb))
    @test brm_descriptor(sb) isa BRMDescriptor
    ordinary = SBBRMI(brmi; mod=@__MODULE__)
    for rho in (0.0, 1.0)
        audit(SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=rho),
              ordinary, "two_groupings_$rho")
    end
    # Either grouping may be the S2Z one.
    other = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:district_g], s2z_rho=0.0)
    @test only(s2z_effect_blocks(other)).group === :district_g
    audit(other, ordinary, "two_groupings_district"; trials=1)
    names = audit(sb, ordinary, "two_groupings").names
    nocor = nocor_two_groupings(FLAT)
    sbn = SBBRMI(nocor; mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=0.0)
    @test only(s2z_effect_blocks(sbn)).columns == (:Intercept, :x)
    @test isempty(pop_labels(sbn, :mu))
    @test !any(r -> startswith(String(r.group), "region_g"), ranef_blocks(sbn))
    audit(sbn, SBBRMI(nocor; mod=@__MODULE__), "nocor_two_groupings")
    idb = id_two_groupings(FLAT)
    sbi = SBBRMI(idb; mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=1.0)
    @test only(s2z_effect_blocks(sbi)).columns == (:Intercept, :x)
    @test !any(r -> r.id in (:a, :b), ranef_blocks(sbi))
    audit(sbi, SBBRMI(idb; mod=@__MODULE__), "id_two_groupings"; ids=Dict(:mu => [:a, :b]))
    recovered = recover_s2z_draws(sb, 0.1 .* randn(Xoshiro(2), 5, length(names)), names;
        rng=Xoshiro(5))[:mu]
    @test size(recovered.effects) == (5, NR, 1)
    @test maximum(abs, sum(recovered.deviations; dims=2)) < 1e-10
end

@testset "S2Z beside an HSGP, a categorical block and a nested |ID| bucket" begin
    brmi = margins(FLAT)
    sb = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=1.0)
    @test [b.predictor for b in s2z_effect_blocks(sb)] == [:alpha_i, :alpha_r]
    @test s2z_stanc_ok(sb)
    audit(sb, SBBRMI(brmi; mod=@__MODULE__), "margins")
end

# The consumer's plate shape: one row per district, read as `alpha_r[d]`.
const PLATE = let rng = Xoshiro(4)
    (; region_g=[1 + (d % NR) for d in 1:NDIS],
       district_g=["D$(lpad(d, 2, '0'))" for d in 1:NDIS],
       x=randn(rng, NDIS), xc=randn(rng, NDIS), yc=randn(rng, NDIS),
       log_exposure=[fill(0.1d / NDIS, NOBS) for d in 1:NDIS],
       count_r=[rand(rng, 0:4, NOBS) for _ in 1:NDIS],
       count_i=[rand(rng, 0:4, NOBS) for _ in 1:NDIS])
end

# The region blocks carry their own `|ID|` so their SD priors are addressable.
plate_margins(data) = @brm data begin
    alpha_r ~ 1 + x + hsgp(xc, yc; k=3) + (1 | region_r | region_g) +
        (1 | kreis | district_g)
    alpha_i ~ 1 + x + hsgp(xc, yc; k=3) + (1 | region_i | region_g) +
        (1 | kreis | district_g)
    sd(:, region_r) ~ Exponential(0.5)
    sd(:, region_i) ~ Exponential(1)
    @plate for d in eachindex(alpha_r)
        count_r[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_r[d]))
    end
    @plate for d in eachindex(alpha_i)
        count_i[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_i[d]))
    end
end

@testset "S2Z on the outer grouping of a plate-read predictor" begin
    brmi = plate_margins(PLATE)
    sb = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=1.0)
    @test length(s2z_effect_blocks(sb)) == 2
    @test s2z_stanc_ok(sb)
    # The claimed buckets' SD priors scale the S2Z blocks.
    code = BRM.stan_code(sb)
    @test occursin("exponential", code)
    @test !any(r -> r.id in (:region_r, :region_i), ranef_blocks(sb))
    @test any(r -> r.id === :kreis, ranef_blocks(sb))
    @test brm_descriptor(sb) isa BRMDescriptor
    audit(sb, SBBRMI(brmi; mod=@__MODULE__), "plate_margins";
          ids=Dict(:alpha_r => :region_r, :alpha_i => :region_i))
end

@testset "S2Z scope that stays refused" begin
    brmi = two_groupings(FLAT)
    # refused: both blocks' means are confounded with the one intercept they
    # would be marginalised into; a joint collapse is not built.
    err = try
        SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:region_g, :district_g], s2z_rho=0.0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("several sum-to-zero groupings", err.msg)
    # refused: the selected block itself must be scalar and independent.
    correlated = @brm begin
        mu ~ 1 + x + (1 + x | region_g) + (1 | district_g)
        y ~ Poisson(exp(mu))
    end
    @test_throws ArgumentError SBBRMI(correlated(FLAT); mod=@__MODULE__,
        s2z_groups=[:region_g], s2z_rho=0.0)
    # refused: one `|ID|` bucket shared by two predictors is one correlated
    # block across them, not two independent scalar blocks.
    shared = @brm begin
        alpha_r ~ 1 + x + (1 | region | region_g)
        alpha_i ~ 1 + x + (1 | region | region_g)
        yr ~ Poisson(exp(alpha_r))
        yi ~ Poisson(exp(alpha_i))
    end
    err = try
        SBBRMI(shared(FLAT); mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=0.0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("shared `|region|` bucket", err.msg)
end

# Exact transport through the wrapper: source -> physical -> source, the
# wrapped density and its Enzyme gradient against finite differences, and
# scalar-cell candidates that do not depend on the supplying source frame.
function wrapper_audit(sb, problem, names; seed=8)
    blocks = adaptive_centering_blocks(sb, names)
    hsgp = BRM._adaptive_hsgp_centering_blocks(sb, names)
    cells = AC_EXT._scalar_centering_cells(sb, names)
    @test !isempty(cells.indices)
    # Ordinary and HSGP cells keep their joint order; scalar cells follow.
    _, joint = AC_EXT._adaptive_joint_centering_reparametrizer(
        BRM.AdaptiveCenteringBlock[b for b in blocks if b isa BRM.AdaptiveCenteringBlock],
        hsgp)
    n_other = length(joint.pairs)
    c = collect(range(0.1, 0.9; length=n_other + length(cells.indices)))
    rp = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=c)
    ir = WarmupHMC.reparametrizer(rp)
    @test first.(ir.pairs) == vcat(first.(joint.pairs), cells.indices)
    physical = 0.3 .* randn(Xoshiro(seed), length(names))
    _, source = WarmupHMC._inverse_with_logabsdet_jacobian(ir, physical)
    jac, mapped = ir(source)
    @test mapped ≈ physical atol = 1e-10
    lp, g = LogDensityProblems.logdensity_and_gradient(rp, source)
    @test lp ≈ LogDensityProblems.logdensity(problem, physical) + jac atol = 1e-8
    fd = map(eachindex(source)) do i
        h = zeros(length(source)); h[i] = 1e-5
        (LogDensityProblems.logdensity(rp, source + h) -
         LogDensityProblems.logdensity(rp, source - h)) / 2e-5
    end
    @test g ≈ fd atol = 1e-5 rtol = 1e-5
    plan = WarmupHMC.candidate_scoring_plan(rp)
    frame = plan.prepare(ir, source, g)
    gphys = last(LogDensityProblems.logdensity_and_gradient(problem, physical))
    for (j, index) in enumerate(cells.indices), candidate in (0.0, 0.5, 1.0)
        p = n_other + j
        @test first(ir.pairs[p]) == index
        _, position, gradient = plan.score(frame, p, index, last(ir.pairs[p]),
            WarmupHMC.PartiallyCentered(candidate))
        t, e, m = cells.targets[j], physical[cells.scales[j]], cells.locations[j]
        @test position ≈ candidate * m + (physical[index] - t * m) * exp((candidate - t) * e) atol = 1e-10
        @test gradient ≈ gphys[index] * exp((t - candidate) * e) atol = 1e-8
    end
    rp
end

@testset "S2Z and total cells share the joint adaptive wrapper" begin
    mixed = @brm begin
        mu ~ 1 + x + hsgp(xc, yc; k=3) + (1 | region_g) + (1 | district_g)
        y ~ Poisson(exp(mu))
    end
    sb = SBBRMI(mixed(FLAT); mod=@__MODULE__, s2z_groups=[:region_g], s2z_rho=1.0)
    problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(mktempdir(), "s2z_joint_wrapper.stan"))
    names = BridgeStan.param_unc_names(problem.model)
    @test !isempty(BRM._adaptive_hsgp_centering_blocks(sb, names))
    rp = wrapper_audit(sb, problem, names)
    # An online warm-up runs end to end on the joint wrapper.
    fit = WarmupHMC.adaptive_warmup_mcmc(Xoshiro(21), rp; n_draws=200)
    @test all(isfinite, fit.posterior_position)
    recovered = recover_s2z_draws(sb, permutedims(fit.posterior_position), names;
        rng=Xoshiro(3))[:mu]
    @test maximum(abs, sum(recovered.deviations; dims=2)) < 1e-8
    # Exact totals on one predictor beside an ordinary block on another.
    totals = @brm begin
        mu ~ 1 + x + (1 | district_g)
        log(sigma) ~ 1 + (1 | region_g)
        y ~ Normal(mu, sigma)
    end
    df = (; FLAT..., y=randn(Xoshiro(9), length(FLAT.x)))
    sbt = SBBRMI(totals(df); mod=@__MODULE__, total_groups=[:region_g])
    @test length(total_effect_blocks(sbt)) == 1
    pt = StanBlocks.stan_instantiate(sbt.model; path=joinpath(mktempdir(), "totals_joint.stan"))
    wrapper_audit(sbt, pt, BridgeStan.param_unc_names(pt.model))
    # Sean's per-window rule still runs only on an S2Z-only model.
    @test_throws ArgumentError adaptive_centering_problem(sb, problem, AutoEnzyme();
        s2z_rule=:fisher)
end
