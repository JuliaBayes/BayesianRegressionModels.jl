# Address-keyed initial values -> complete RK and Stan starting points.
# RUN: julia --project=test test/start_values.jl
#
# `brm_start` resolves the same prior addresses on both backends: the RK point
# and the Stan point must be one physical point (the semantic transport is the
# independent oracle), the addressed constrained values must be exact, and
# every unaddressed coordinate keeps its base value.
using BridgeStan, CategoricalArrays, LinearAlgebra, Random
import StanBlocks
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

const START_CACHE = mktempdir()
compiled(sb, name) = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(START_CACHE, "$name.stan")).model

# Two correlated margins of `|pk|` in `ranefcoefnames` order.
const PK_SIGMA = [0.09 0.018; 0.018 0.16]
const PK_INITS = (;
    log_CL=(; Intercept=log(2.7), weight=0.75),
    log_V=(; Intercept=log(31.5)),
    pk=PK_SIGMA,
    sigma=0.1,
)

# RK and Stan points of the same BRMI agree physically and by coordinate.
function same_point(brmi, inits, name; rk_base=nothing, kwargs...)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), kwargs...)
    model = compiled(sb, name)
    t = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(model))
    u = isnothing(rk_base) ? brm_start(rk, inits) : brm_start(rk, inits; base=rk_base)
    s = isnothing(rk_base) ? brm_start(sb, model, inits) :
        brm_start(sb, model, inits; base=brm_rk_point_to_stan(t, rk_base))
    (; rk, sb, model, t, u, s)
end

constrained(model, s) = (BridgeStan.param_constrain(model, s),
    String.(BridgeStan.param_names(model)))

@stestset "address-keyed values start the population-PK model on both backends" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    @test [(m.predictor, m.coefficient) for m in ranefcoefnames(brmi, :pk)] ==
        [(:log_CL, :Intercept), (:log_V, :Intercept)]
    saved = deepcopy(PK_INITS)
    (; rk, sb, model, t, u, s) = same_point(brmi, PK_INITS, "plate_pk")
    @test PK_INITS == saved
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    @test brm_check_coordinate_transport(t, rk, model, u).factors == 1

    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    population(lp, c) = θ[only(brm_population_effect_coordinates(d, lp, names;
        coefficient=c).coordinates)]
    @test population(:log_CL, :Intercept) ≈ log(2.7) rtol=1e-12
    @test population(:log_CL, :weight) ≈ 0.75 rtol=1e-12
    @test population(:log_V, :Intercept) ≈ log(31.5) rtol=1e-12
    tau = [θ[only(brm_ranef_sd_coordinates(d, lp, names; id=:pk).coordinates)]
        for lp in (:log_CL, :log_V)]
    @test tau ≈ sqrt.(diag(PK_SIGMA)) rtol=1e-12
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.1 rtol=1e-12
    # Cov = diag(tau) L L' diag(tau) is the given covariance.
    view = brm_nlme_view(rk)
    natural = BRM._brm_nlme_natural(rk, view, BRM._brm_nlme_families(rk, view), u)
    order = [findfirst(b -> b.predictor === lp, view.block_margins)
        for lp in (:log_CL, :log_V)]
    @test natural.omega[order, order] ≈ PK_SIGMA rtol=1e-10
    # The standardized subject draws are untouched: the natural effects are zero.
    @test all(iszero, natural.eta)
    stan_pos = Dict(n => i for (i, n) in enumerate(t.stan_names))
    @test all(iszero, s[[stan_pos[p.stan] for p in t.pairs if p.address.kind === :ranef_z]])

    # `(; sd, cor)` addresses the same block parts.
    C = PK_SIGMA ./ (sqrt.(diag(PK_SIGMA)) * sqrt.(diag(PK_SIGMA))')
    C[diagind(C)] .= 1
    parts = merge(PK_INITS, (; pk=(; sd=sqrt.(diag(PK_SIGMA)), cor=C)))
    @test brm_start(rk, parts) ≈ u rtol=1e-10 atol=1e-12
    @test brm_start(sb, model, parts) ≈ s rtol=1e-10 atol=1e-12
    # One part alone leaves the other at the base: sd only keeps L = I.
    sd_only = brm_start(rk, (; pk=(; sd=0.5)))
    only_natural = BRM._brm_nlme_natural(rk, view,
        BRM._brm_nlme_families(rk, view), sd_only)
    @test only_natural.omega ≈ 0.25I rtol=1e-12

    # A random base: addressed values are set, every other coordinate keeps it.
    base = 0.3 .* randn(Xoshiro(7), length(u))
    (; u, s) = same_point(brmi, PK_INITS, "plate_pk"; rk_base=base)
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    addressed = [i for (i, p) in enumerate(t.pairs) if p.address.kind in
        (:population, :scalar, :ranef_scale, :ranef_correlation)]
    rk_pos = Dict(n => i for (i, n) in enumerate(t.rk_names))
    untouched = [rk_pos[p.rk] for (i, p) in enumerate(t.pairs) if !(i in addressed)]
    @test !isempty(untouched)
    @test u[untouched] == base[untouched]
end

@stestset "a linked predictor is addressed by its public name" begin
    brmi = nlme_pk(NLME_DATA)
    inits = (; CL=(; Intercept=log(3.0), wt=0.5), V=(; Intercept=log(20.0)),
        pk=(; sd=[0.3, 0.2]), sigma=1.5)
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "linked_pk")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    @test θ[only(brm_population_effect_coordinates(d, :CL, names;
        coefficient=:wt).coordinates)] ≈ 0.5 rtol=1e-12
    # refused: the emitted binding is not the public address.
    @test_throws ErrorException brm_start(rk, (; log_CL=(; Intercept=1.0)))
    @test_throws ErrorException brm_start(sb, model, (; log_CL=(; Intercept=1.0)))
end

@stestset "exact total blocks start at the population value" begin
    data = (; g=repeat(["a", "b", "c", "d"], 3), x=collect(range(-1, 1; length=12)),
        y=sin.(1:12))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__)
    blocks = total_effect_blocks(sb)
    @test length(blocks) == 1
    model = compiled(sb, "totals")
    inits = (; mu=(; Intercept=1.5, x=0.2), sigma=0.4)
    s = brm_start(sb, model, inits)
    θ, names = constrained(model, s)
    block = only(blocks)
    totals = BRM._total_coordinates(sb, block, names).totals
    β = [Dict(:Intercept => 1.5, :x => 0.2)[c] for c in block.population_columns]
    @test θ[totals] ≈ repeat(transpose(block.A * β), size(totals, 1)) rtol=1e-12
    @test :Intercept in block.population_columns
    d = brm_descriptor(sb)
    if !(:x in block.population_columns)  # a retained, sampled slope
        @test θ[only(brm_population_effect_coordinates(d, :mu, names;
            coefficient=:x).coordinates)] ≈ 0.2 rtol=1e-12
    end
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.4 rtol=1e-12

    # The conventional emission of the same model samples the coefficients.
    (; u, s) = same_point(brmi, inits, "totals_conventional")
    @test length(s) == length(u)
end

# One lone `|ID|` margin per predictor: independent effects, which the default
# `total_groups=:auto` turns into exact total blocks.
function lone_eta_pk(data)
    @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | eta_cl | subject)
        log_V ~ 1 + (1 | eta_v | subject)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            Vc = exp(log_V[i])
            pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
            dv[i] ~ normal(pred[i], sigma)
        end
    end
end

@stestset "an |ID| absorbed by an exact total block keeps its scale address" begin
    brmi = lone_eta_pk(NLME_PLATE_DATA)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    blocks = Dict(b.predictor => b for b in total_effect_blocks(sb))
    @test sort!(collect(keys(blocks))) == [:log_CL, :log_V]
    model = compiled(sb, "lone_eta_totals")
    inits = (; log_CL=(; Intercept=log(2.7), weight=0.75),
        log_V=(; Intercept=log(31.5)), eta_cl=[0.09;;], eta_v=(; sd=0.2), sigma=0.7)
    saved = deepcopy(inits)
    s = brm_start(sb, model, inits)
    @test inits == saved
    @test isfinite(BridgeStan.log_density(model, s))
    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    # The `|ID|` scale is the block's total scale: the SD of each group's
    # total around `A β`, read back through the same address.
    scale(lp, id) = θ[only(brm_ranef_sd_coordinates(d, lp, names; id).coordinates)]
    @test scale(:log_CL, :eta_cl) ≈ 0.3 rtol=1e-12
    @test scale(:log_V, :eta_v) ≈ 0.2 rtol=1e-12
    for (lp, sd) in (:log_CL => 0.3, :log_V => 0.2)
        @test θ[BRM._total_coordinates(sb, blocks[lp], names).scales] ≈ [sd] rtol=1e-12
    end
    # Absorbed intercepts place every total at the population value; the
    # retained `weight` slope is sampled.
    for (lp, β) in (:log_CL => log(2.7), :log_V => log(31.5))
        @test blocks[lp].population_columns == (:Intercept,)
        totals = BRM._total_coordinates(sb, blocks[lp], names).totals
        @test θ[totals] ≈ fill(β, size(totals)) rtol=1e-12
    end
    @test θ[only(brm_population_effect_coordinates(d, :log_CL, names;
        coefficient=:weight).coordinates)] ≈ 0.75 rtol=1e-12
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.7 rtol=1e-12
    # Unaddressed coordinates keep a random base: here everything but one scale.
    unc = String.(BridgeStan.param_unc_names(model))
    k = only(BRM._total_coordinates(sb, blocks[:log_V], unc).scales)
    base = 0.3 .* randn(Xoshiro(11), length(s))
    u = brm_start(sb, model, (; eta_v=(; sd=0.2)); base)
    @test exp(u[k]) ≈ 0.2 rtol=1e-12
    rest = setdiff(eachindex(u), k)
    @test u[rest] ≈ base[rest] rtol=1e-10 atol=1e-12
    # refused: a one-margin block takes a 1×1 covariance and a positive scale.
    @test_throws DimensionMismatch brm_start(sb, model, (; eta_cl=fill(0.1, 2, 2)))
    @test_throws ErrorException brm_start(sb, model, (; eta_cl=(; sd=-0.1)))

    # Omitting the response leaves no likelihood reaching the totals: the block
    # and its scale are drawn in generated quantities, under the same address.
    prior_sb = SBBRMI(lone_eta_pk(Base.structdiff(NLME_PLATE_DATA, (; dv=nothing)));
        mod=@__MODULE__)
    prior = compiled(prior_sb, "lone_eta_prior")
    pd = brm_descriptor(prior_sb)
    gq_names = String.(BridgeStan.param_names(prior; include_tp=true, include_gq=true))
    q = brm_ranef_sd_coordinates(pd, :log_V, gq_names; id=:eta_v)
    @test q.output.kind === :generated_quantity
    @test gq_names[only(q.coordinates)] ==
        "$(only(b for b in total_effect_blocks(prior_sb) if b.predictor === :log_V).scales).1"
    # refused: the generated scale is not among the sampled names, and a start
    # point has no coordinate for it.
    @test_throws ErrorException brm_ranef_sd_coordinates(pd, :log_V,
        String.(BridgeStan.param_names(prior)); id=:eta_v)
    @test_throws ErrorException brm_start(prior_sb, prior, (; eta_v=(; sd=0.2)))
end

@stestset "a categorical predictor takes level-keyed values" begin
    data = (; subject=repeat(["a", "b", "c", "d"], 3),
        arm=categorical(repeat(["p", "q", "r"], 4)), y=sin.(1:12))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + arm + (1 | e | subject)
        y ~ Normal(mu, sigma)
    end
    inits = (; mu=(; Intercept=-0.5, arm=["q" => 0.25, "r" => -0.4]),
        e=fill(0.04, 1, 1), sigma=0.2)
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "categorical")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    q = brm_population_effect_coordinates(d, :mu, names; coefficient=:arm)
    levels = Dict(BRM._brm_transport_level(c.nonreference_level) => θ[c.coordinate]
        for c in q.contrasts)
    @test levels["q"] ≈ 0.25 rtol=1e-12
    @test levels["r"] ≈ -0.4 rtol=1e-12
    @test θ[only(brm_population_effect_coordinates(d, :mu, names;
        coefficient=:Intercept).coordinates)] ≈ -0.5 rtol=1e-12
    # refused: the reference level has no coefficient.
    @test_throws ErrorException brm_start(rk, (; mu=(; arm=["p" => 0.1])))
    @test_throws ErrorException brm_start(sb, model, (; mu=(; arm=["p" => 0.1])))
    # refused: a categorical column takes level pairs, not one number.
    @test_throws ErrorException brm_start(sb, model, (; mu=(; arm=0.1)))
end

@stestset "a vector parameter is set on its constrained scale" begin
    data = (; x=collect(range(-1, 1; length=8)), y=cos.(1:8))
    brmi = @brm data begin
        s ~ Dirichlet([1.0, 2.0, 1.5])
        mu ~ 1 + x
        loc = mu + 0.8 * s[1] - s[3]
        y ~ Normal(loc, 1)
    end
    inits = (; s=[0.5, 0.3, 0.2], mu=(; x=-0.3))
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "simplex")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    θ, names = constrained(model, s)
    @test θ[brm_output_coordinates(brm_descriptor(sb), :s, names)] ≈
        [0.5, 0.3, 0.2] rtol=1e-10
    # refused: a simplex of another length, or off the simplex.
    @test_throws DimensionMismatch brm_start(rk, (; s=[0.5, 0.5]))
    @test_throws DimensionMismatch brm_start(sb, model, (; s=[0.5, 0.5]))
    @test_throws Exception brm_start(rk, (; s=[0.5, 0.6, 0.2]))
    @test_throws Exception brm_start(sb, model, (; s=[0.5, 0.6, 0.2]))
end

@stestset "invalid addresses and values are refused" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    model = compiled(sb, "plate_pk")
    for backend in ((rk,), (sb, model))
        # refused: no such parameter, coefficient or block part.
        @test_throws ErrorException brm_start(backend..., (; tau=0.1))
        @test_throws ErrorException brm_start(backend..., (; log_V=(; weight=0.1)))
        @test_throws ErrorException brm_start(backend..., (; pk=(; scale=0.1)))
        # refused: a scale or covariance outside its support.
        @test_throws ErrorException brm_start(backend..., (; pk=(; sd=-0.1)))
        @test_throws ErrorException brm_start(backend..., (; pk=[1.0 2.0; 2.0 1.0]))
        @test_throws ErrorException brm_start(backend...,
            (; pk=(; cor=[1.0 0.2; 0.3 1.0])))
        # refused: the block has two margins.
        @test_throws DimensionMismatch brm_start(backend..., (; pk=fill(0.1, 3, 3)))
        # refused: a linear predictor's coefficients are a NamedTuple.
        @test_throws ErrorException brm_start(backend..., (; log_V=1.0))
        @test_throws ErrorException brm_start(backend..., (; sigma=NaN))
    end
    @test_throws DimensionMismatch brm_start(rk, PK_INITS; base=zeros(2))
end
