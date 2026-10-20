# S2Z under an omitted response (user decision `0vrfeep`): `s2z_groups`
# reparameterises the blocks an observed likelihood reaches. A predictor that
# no observed likelihood reaches keeps its block of the requested grouping
# conventional, and StanBlocks re-draws that block in generated quantities, so
# one `SBBRMI` call serves both the full data and the data with a response
# omitted. Before, an intercept-only such predictor was refused with a wrong
# reason ("missing or fixed-term population design"), and one with a covariate
# built an S2Z block that StanBlocks could not re-draw (`stan_code` failed).
#
# Run: julia --project=test test/s2z_omitted_response.jl
using Test, Random, Logging
using BayesianRegressionModels, StanBlocks, BridgeStan
const BRM = BayesianRegressionModels

code(sb) = StanBlocks.stan_code(sb.model)
stanc_ok(sb) = StanBlocks.stanc_check(code(sb); warn_pedantic=false).ok
s2z_on(sb) = sort!([b.predictor for b in s2z_effect_blocks(sb)])
placement(sb, name) = BRM._traced_output_kinds(sb)[name]
quiet(f) = with_logger(f, NullLogger())
# The one warning an omitted response gives; the placement probe adds none.
const OMITTED_Z = (:warn, r"`z` bind\(s\) no data column")

const N = 60
const DATA = let rng = Xoshiro(1)
    (; subject=rand(rng, 1:6, N), site=rand(rng, 1:5, N), x=randn(rng, N),
       w=randn(rng, N), y=randn(rng, N), z=randn(rng, N))
end
const NO_Z = Base.structdiff(DATA, NamedTuple{(:z,)})
const S2Z = (; total_groups=(), s2z_groups=[:site], s2z_rho=0.5)

intercept_only = @brm begin
    sigma ~ Exponential(1); sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 + x | subject)
    nu ~ 1 + (1 | site)
    y ~ Normal(mu, sigma); z ~ Normal(nu, sigma2)
end
shared = @brm begin
    sigma ~ Exponential(1); sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 | site)
    nu ~ 1 + w + (1 | site)
    y ~ Normal(mu, sigma); z ~ Normal(nu, sigma2)
end
# `nu` reads a copy of the grouping column, which S2Z on `site` cannot select.
shared_twin = @brm begin
    sigma ~ Exponential(1); sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 | site)
    nu ~ 1 + w + (1 | site_nu)
    y ~ Normal(mu, sigma); z ~ Normal(nu, sigma2)
end
renamed(sb) = (replace(code(sb), "site_nu" => "site"),
    Dict(Symbol(replace(String(k), "site_nu" => "site")) => v for (k, v) in sb.data))

@testset "intercept-only predictor of an omitted response" begin
    @test s2z_on(SBBRMI(intercept_only(DATA); S2Z...)) == [:nu]
    sb = @test_logs OMITTED_Z SBBRMI(intercept_only(NO_Z); S2Z...)
    @test isempty(s2z_effect_blocks(sb))
    @test only(r for r in ranef_blocks(sb) if r.group === :site).generated
    @test placement(sb, :nu) === :generated_quantity
    @test code(sb) == code(quiet(() -> SBBRMI(intercept_only(NO_Z); total_groups=())))
    @test stanc_ok(sb)
end

@testset "a shared grouping stays S2Z on the reached predictor" begin
    @test s2z_on(SBBRMI(shared(DATA); S2Z...)) == [:mu, :nu]
    sb = @test_logs OMITTED_Z SBBRMI(shared(NO_Z); S2Z...)
    @test s2z_on(sb) == [:mu]
    @test placement(sb, :s2z_contrast_mu) === :parameter
    @test placement(sb, :nu) === :generated_quantity
    @test only(r for r in ranef_blocks(sb) if r.group === :site).generated
    twin = quiet(() -> SBBRMI(shared_twin((; NO_Z..., site_nu=NO_Z.site)); S2Z...))
    @test renamed(twin) == (code(sb), sb.data)
    @test stanc_ok(sb)
    # A held-out response keeps every parameter it reads, so its predictor
    # stays S2Z: only omission moves the block to generated quantities.
    held = SBBRMI(shared(DATA); S2Z..., held_out=:z)
    @test s2z_on(held) == [:mu, :nu]
    @test placement(held, :s2z_contrast_nu) === :parameter
    @test stanc_ok(held)
    # Replay keeps the fitted program, as for the conventional model.
    replay = reprocess(sb, DATA)
    @test code(replay) == code(sb)
    @test s2z_on(replay) == [:mu]
end

@testset "an unreached block need not be S2Z-eligible" begin
    correlated = @brm begin
        sigma ~ Exponential(1); sigma2 ~ Exponential(1)
        mu ~ 1 + x + (1 | site)
        nu ~ 1 + w + (1 + w | site)
        y ~ Normal(mu, sigma); z ~ Normal(nu, sigma2)
    end
    # refused: the selected block itself must be scalar and independent
    # (`s2z_plan.jl` first scope); `z` makes `nu`'s block a sampled one.
    @test_throws ArgumentError SBBRMI(correlated(DATA); S2Z...)
    sb = quiet(() -> SBBRMI(correlated(NO_Z); S2Z...))
    @test s2z_on(sb) == [:mu]
    @test stanc_ok(sb)
end

@testset "a fully observed flat model builds no placement probe" begin
    for model in (intercept_only, shared)
        brmi = model(DATA)
        prepared = BRM._brm_prepare_model(brmi; program=BRM._brm_prepare_program(brmi))
        @test isempty(BRM._sb_s2z_generated_predictors(
            () -> error("probe built"), prepared, Set([:site])))
    end
end

# Predictors read as `alpha[d]` in `@plate for` cells: no top-level likelihood
# reads them, so StanBlocks' placement in the probe decides.
const NDIS, NREG, NOBS = 12, 4, 3
const PLATE = let rng = Xoshiro(4)
    (; region_g=[1 + (d % NREG) for d in 1:NDIS],
       district_g=["D$(lpad(d, 2, '0'))" for d in 1:NDIS], x=randn(rng, NDIS),
       log_exposure=[fill(0.1d / NDIS, NOBS) for d in 1:NDIS],
       count_r=[rand(rng, 0:4, NOBS) for _ in 1:NDIS],
       count_i=[rand(rng, 0:4, NOBS) for _ in 1:NDIS])
end
plate = @brm begin
    alpha_r ~ 1 + x + (1 | region_g) + (1 | district_g)
    alpha_i ~ 1 + x + (1 | region_g) + (1 | district_g)
    @plate for d in eachindex(alpha_r)
        count_r[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_r[d]))
    end
    @plate for d in eachindex(alpha_i)
        count_i[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_i[d]))
    end
end

@testset "an omitted @plate response" begin
    kw = (; total_groups=(), s2z_groups=[:region_g], s2z_rho=1.0)
    @test s2z_on(SBBRMI(plate(PLATE); kw...)) == [:alpha_i, :alpha_r]
    sb = quiet(() -> SBBRMI(plate(Base.structdiff(PLATE, NamedTuple{(:count_i,)})); kw...))
    @test s2z_on(sb) == [:alpha_r]
    @test placement(sb, :alpha_i) === :generated_quantity
    @test placement(sb, :s2z_contrast_alpha_r) === :parameter
    @test stanc_ok(sb)
end

# The omitted program's sampled density is the model without the omitted half:
# the S2Z block it keeps is unchanged, and the generated block adds nothing.
@testset "sampled density equals the pruned model's" begin
    sb = quiet(() -> SBBRMI(shared(NO_Z); S2Z...))
    pruned = SBBRMI(prune_responses(shared(NO_Z)); S2Z...)
    @test s2z_on(pruned) == [:mu]
    dir = mktempdir()
    models = [StanBlocks.stan_instantiate(m.model; path=joinpath(dir, "$tag.stan")).model
              for (m, tag) in ((sb, "omitted"), (pruned, "pruned"))]
    names = BridgeStan.param_unc_names.(models)
    @test names[1] == names[2]
    rng = Xoshiro(7)
    for _ in 1:5
        x = 0.5 .* randn(rng, length(names[1]))
        a, b = (BridgeStan.log_density(m, x; propto=false, jacobian=true) for m in models)
        @test a ≈ b rtol=1e-12
    end
end
