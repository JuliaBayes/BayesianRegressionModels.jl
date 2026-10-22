# test/rk_marginal_ranef.jl — the RK half of likelihood-inactive |ID| margins.
#
# Run: julia --project=test test/rk_marginal_ranef.jl
#
# RK plans a shared `|ID|` bucket over the reachable predictors only, so an
# omitted response already narrows it. That narrowed block must carry the
# exact marginal law — LKJ(a, eta + (K - a)/2) and the kept margins' own scale
# priors — exactly as SBBRMI samples it (test/marginal_ranef.jl).
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module RKMarginalRanef
using BayesianRegressionModels, Distributions

const DF = (;
    subject = [1, 1, 2, 2, 3, 3, 4, 4],
    x = [-1.0, 0.5, -0.5, 1.0, 0.0, 1.5, -1.5, 0.25],
    ya = [0.3, 0.9, -0.2, 0.4, 0.1, 1.2, -0.8, 0.0],
    yb = [1.1, 0.7, 0.2, 0.5, -0.3, 0.6, 0.9, 1.4],
    yc = [-0.4, -0.1, 0.8, 0.3, 0.5, 0.2, -0.6, 0.1],
)
const OMIT_B = (; DF.subject, DF.x, DF.ya, DF.yc)

plain(data) = @brm data begin
    a_loc ~ 1 + x + (1 + x | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    ya ~ Normal(a_loc, 0.5)
    yb ~ Normal(b_loc, 0.5)
    yc ~ Normal(c_loc, 0.5)
end

configured(data) = @brm data begin
    a_loc ~ 1 + x + (1 + x | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    sd(:, p) ~ Exponential(0.5)
    sd(b_loc, p) ~ Exponential(0.2)
    cor(:, p) ~ LKJCholesky(4, 2.0)
    ya ~ Normal(a_loc, 0.5)
    yb ~ Normal(b_loc, 0.5)
    yc ~ Normal(c_loc, 0.5)
end

# One row per subject; a `@plate` cell observes `ya` (reading `a_loc` and
# `b_loc`) and `yc` (reading only `c_loc`). Without `yc`, `c_loc` is read only
# by an omitted in-cell observation (todo `1784wd0`).
const CELL = (; subject=[1, 2, 3, 4], x=[-1.0, 0.5, 0.0, 1.5],
    ya=[[0.3, 0.9], [-0.2], [0.1, 1.2], [-0.8, 0.0]],
    yc=[[-0.4], [0.8, 0.3], [0.5], [-0.6, 0.1]])
const CELL_OMIT_C = (; CELL.subject, CELL.x, CELL.ya)
cell(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    @plate for i in eachindex(a_loc)
        ya[i] ~ Normal(a_loc[i] .+ b_loc[i], 0.5)
        yc[i] ~ Normal(c_loc[i], 0.5)
    end
end
end

const RKM = RKMarginalRanef

rk_buckets(plan) = plan.ranef_buckets
rk_buckets(plan::BRM._RKValuePlan) = rk_buckets(plan.regression)
rk_bucket(brmi) = only(b for b in rk_buckets(BRM._brm_rk_plan(brmi)) if b.id === :p)
rk_margins(bucket) = [(m.predictor, m.coefficient) for m in bucket.margins]

@stestset "RK plans the exact marginal of an omitted response's shared block" begin
    full = rk_bucket(RKM.plain(RKM.DF))
    @test rk_margins(full) == [(:a_loc, :Intercept), (:a_loc, :x),
                               (:b_loc, :Intercept), (:c_loc, :Intercept)]
    @test full.lkj_eta == 1.0
    omitted = rk_bucket(RKM.plain(RKM.OMIT_B))
    @test rk_margins(omitted) == [(:a_loc, :Intercept), (:a_loc, :x), (:c_loc, :Intercept)]
    @test omitted.lkj_eta == 1.5

    # `cor(:, p) ~ LKJCholesky(4, ...)` sizes the shared block, not the
    # narrowed one; `sd(b_loc, p)` addresses the omitted margin.
    full = rk_bucket(RKM.configured(RKM.DF))
    @test full.lkj_eta == 2.0
    @test length(full.sd_priors) == 4
    omitted = rk_bucket(RKM.configured(RKM.OMIT_B))
    @test omitted.lkj_eta == 2.5
    @test omitted.sd_priors == full.sd_priors[[1, 2, 4]]
    @test omitted.sd_priors != full.sd_priors[1:3]
end

@stestset "RK and Stan sample the same reduced block" begin
    for (name, builder) in (("plain", RKM.plain), ("configured", RKM.configured)),
            centered_groups in ((), [:subject])
        brmi = builder(RKM.OMIT_B)
        rk, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=RKM, total_groups=(), centered_groups)
        path = joinpath(tempdir(), "brm-rk-marginal-ranef",
            name * (isempty(centered_groups) ? "" : "_centered") * ".stan")
        mkpath(dirname(path))
        stan = BRM.stan_instantiate(sb; path)
        names = BridgeStan.param_unc_names(stan.model)
        @test any(startswith(isempty(centered_groups) ? "b_p_subject_active_z_flat." :
            "b_p_subject_active_b_cols_bc."), names)
        transport = brm_coordinate_transport(rk, sb, names)
        @test length(transport) == length(names) ==
              length(coordinate_names(rk.model.layout))
        @test length(transport.centered) == length(centered_groups)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))),
                  [0.37sin(i) for i in 1:length(transport)])
            stan_u = brm_rk_point_to_stan(transport, u)
            @test brm_check_coordinate_transport(transport, rk, stan.model, u).max_error <=
                1e-12
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

@stestset "an omitted in-cell response narrows the shared block" begin
    @test rk_margins(rk_bucket(RKM.cell(RKM.CELL))) ==
        [(:a_loc, :Intercept), (:b_loc, :Intercept), (:c_loc, :Intercept)]
    omitted = rk_bucket(RKM.cell(RKM.CELL_OMIT_C))
    @test rk_margins(omitted) == [(:a_loc, :Intercept), (:b_loc, :Intercept)]
    @test omitted.lkj_eta == 1.5
    brmi = RKM.cell(RKM.CELL_OMIT_C)
    rk, problem = consumer_problem(brmi)
    @test !any(n -> startswith(string(n), "c_loc"), coordinate_names(rk.model.layout))
    sb = SBBRMI(brmi; mod=RKM, total_groups=())
    path = joinpath(tempdir(), "brm-rk-marginal-ranef", "cell_omit_c.stan")
    mkpath(dirname(path))
    stan = BRM.stan_instantiate(sb; path)
    names = BridgeStan.param_unc_names(stan.model)
    transport = brm_coordinate_transport(rk, sb, names)
    @test length(transport) == length(names) == length(coordinate_names(rk.model.layout))
    for u in (zeros(length(transport)),
              collect(range(-0.43, 0.51; length=length(transport))))
        stan_u = brm_rk_point_to_stan(transport, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-11 rtol=2e-11
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
    end
end
