# test/conditional_effects.jl — conditional predictions / comparisons / slopes.
#
# Covers the bambi `interpret`-style layer over a fitted SBBRMI:
#   1. grids            — focal expansion, typicals, fills, fail-closed cases
#   2. means            — exact response-mean maps per family (link-vs-response
#                         agreement to 1e-12, no RNG involved)
#   3. slopes           — central differences equal constrained coefficients
#                         (Gaussian exact to 1e-8; Bernoulli/NegBinomial to 1e-6
#                         via the chain rule)
#   4. contrasts        — layout exactness plus treatment-coefficient agreement
#   5. predictive       — shape, seed determinism, Monte-Carlo agreement with
#                         response means (adaptive 5-SE tolerance)
#   6. summaries        — HDI/ETI values and error paths
#   7. error paths      — every fail-closed branch, all compile-free
#
# Run: julia --project=. test/conditional_effects.jl
# (Everything here resolves in the root env: BRM + BridgeStan +
# Distributions. No sampler, no Turing, no test-env bootstrap.)
#
# Exactly ONE Stan program is compiled (eleven outcomes sharing one source);
# every error path is exercised without instantiating.

using Test
using BayesianRegressionModels
using BridgeStan
using Distributions: Gamma, Weibull, quantile
using LogExpFunctions: logistic
using Random: MersenneTwister, randn
using Statistics: mean, std

cond_df = (;
    x=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    g=[1, 2, 3, 1, 2, 3, 1, 2, 3],
    n=[10, 10, 10, 12, 12, 12, 10, 10, 12],
    p=[0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.25, 0.45, 0.65],
    yN=[1.0, 1.5, 2.0, 2.4, 3.1, 3.4, 4.0, 4.2, 4.9],
    yB=[0, 1, 0, 1, 1, 0, 1, 0, 1],
    yNB=[0, 1, 2, 1, 3, 2, 4, 3, 5],
    yG=[0.5, 1.0, 1.5, 1.2, 2.0, 1.8, 2.5, 2.2, 3.0],
    yBi=[3, 7, 2, 8, 5, 9, 4, 6, 10],
    yBi2=[3, 7, 2, 8, 5, 9, 4, 6, 9],
    yP=[0, 1, 1, 2, 2, 3, 3, 4, 4],
    yE=[0.4, 0.9, 1.4, 1.1, 1.9, 1.7, 2.4, 2.1, 2.9],
    yBe=[0, 1, 0, 1, 1, 0, 1, 0, 1],
    yLN=[1.0, 1.5, 2.0, 2.4, 3.1, 3.4, 4.0, 4.2, 4.9],
    yNB2=[0, 1, 2, 1, 3, 2, 4, 3, 5],
)

cond_builder = @brm begin
    sigma ~ Exponential(1)
    phi ~ Exponential(1)
    theta ~ Exponential(1)
    muN ~ 1 + x + factor(g)
    muB ~ 1 + x
    muNB ~ 1 + x
    aA ~ 1 + x
    muBi ~ 1 + x
    muBi2 ~ 1 + x
    muP ~ 1 + x
    muE ~ 1 + x
    log(muLN) ~ 1 + x
    log(muNB2) ~ 1 + x
    yN ~ Normal(muN, sigma)
    yB ~ BernoulliLogit(muB)
    yNB ~ NegativeBinomial2(exp(muNB), phi)
    yG ~ Gamma(exp(aA), theta)
    yBi ~ BinomialLogit(n, muBi)
    yBi2 ~ BinomialLogit(10, muBi2)
    yP ~ Poisson(exp(muP))
    yE ~ Exponential(exp(muE))
    yBe ~ Bernoulli(p)
    yLN ~ Normal(muLN, sigma)
    yNB2 ~ NegativeBinomial2(muNB2, phi)
end

d = brm_descriptor(cond_builder, cond_df; mod=@__MODULE__, name=:cond_fixture)

@testset "grids — focal expansion and typicals" begin
    grid = brm_prediction_grid(d; focal=:x, n=5)
    @test keys(grid) == d.columns
    @test grid.x == collect(range(0.0, 4.0; length=5))
    @test grid.g == fill(1, 5) # first-seen mode on a 3/3/3 tie
    @test grid.n == fill(10, 5) # integer typical is the mode, not the mean
    @test grid.yN == fill(1.0, 5) # response fill: first training value
    @test grid.yB == fill(0, 5)

    grid_g = brm_prediction_grid(d; focal=:g)
    @test grid_g.g == [1, 2, 3] # fitted level order
    @test grid_g.x == fill(mean(cond_df.x), 3)

    grid_both = brm_prediction_grid(d; focal=[:x, :g], n=4)
    @test length(grid_both.x) == 12
    @test grid_both.x[1:4] == collect(range(0.0, 4.0; length=4)) # first fastest
    @test grid_both.g[1:4] == fill(1, 4)
    @test grid_both.g[5:8] == fill(2, 4)

    grid_dict = brm_prediction_grid(d; focal=Dict(:x => [1.0, 2.0], :g => 2))
    @test grid_dict.x == [1.0, 2.0]
    @test grid_dict.g == [2, 2]

    grid_fixed = brm_prediction_grid(d; focal=:x, n=3, fixed=(; g=3))
    @test grid_fixed.g == fill(3, 3)

    @test_throws ErrorException brm_prediction_grid(d; focal=:nope)
    @test_throws ErrorException brm_prediction_grid(d; focal=:yN)
    @test_throws ErrorException brm_prediction_grid(d; focal=:x, n=1)
    @test_throws ErrorException brm_prediction_grid(d; focal=[:x, :x])
    @test_throws ErrorException brm_prediction_grid(
        d; focal=:g, fixed=(; g="a"))
    @test_throws ErrorException brm_prediction_grid(
        d; focal=:x, fixed=(; nope=1))
    @test_throws ErrorException brm_prediction_grid(
        d; focal=Dict(:g => ["zzz"]))
    # Integer columns skip grid-time membership (an integer can be a
    # count, not just a level); an unseen *factor* level fails loudly at
    # reprocess instead.
    unseen = brm_prediction_grid(d; focal=Dict(:g => [999]))
    @test_throws ErrorException brm_execute(d, :reprocess, unseen)
    @test_throws ErrorException brm_prediction_grid(
        d; focal=Dict(:g => [1.5]))
    @test_throws ErrorException brm_prediction_grid(
        d; focal=:x, fixed=(; g=missing))
end

@testset "grids — integer focals and missing-tolerant columns" begin
    int_df = (; k=[1, 2, 3, 4, 5, 6, 7, 8], y=[1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5])
    int_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + zscale(k)
        y ~ Normal(mu, sigma)
    end
    di = brm_descriptor(int_builder, int_df; mod=@__MODULE__, name=:cond_int)
    grid = brm_prediction_grid(di; focal=:k, n=25)
    @test eltype(grid.k) <: Integer
    @test grid.k == collect(1:8) # rounded auto grid collapses to fitted set
    @test_throws ErrorException brm_prediction_grid(
        di; focal=Dict(:k => [1.5]))

    # An `mi` response is not a gridded column and not a selectable
    # outcome: the grid builds over the predictors, and the engine
    # refuses `mi` models outright.
    mi_df = (; x=[-1.0, 0.5, 2.0, 0.25],
               y=Union{Missing,Float64}[missing, 0.2, missing, -0.4])
    mi_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end
    dm = brm_descriptor(mi_builder, mi_df; mod=@__MODULE__, name=:cond_mi)
    grid_mi = brm_prediction_grid(dm; focal=:x, n=3)
    @test !haskey(grid_mi, :y)
    @test length(grid_mi.x) == 3
    unc_mi = zeros(2, 1)
    @test_throws ErrorException brm_conditional_draws(
        dm, unc_mi, grid_mi; response=:y)
end

@testset "summaries — HDI/ETI values and error paths" begin
    draws = repeat(reshape(collect(1.0:100.0), 100, 1), 1, 2)
    rows = brm_summarize_draws(draws; probs=[0.9, 0.5], how=:eti)
    @test length(rows) == 2
    @test rows[1].element == 1
    @test rows[1].mean ≈ 50.5
    @test rows[1].q50 ≈ 50.5
    @test rows[1].lower_1 ≈ quantile(1.0:100.0, 0.05)
    @test rows[1].upper_1 ≈ quantile(1.0:100.0, 0.95)
    @test rows[1].lower_2 ≈ quantile(1.0:100.0, 0.25)
    @test rows[2].element == 2
    @test (rows[2].mean, rows[2].sd, rows[2].q50, rows[2].lower_1) ==
        (rows[1].mean, rows[1].sd, rows[1].q50, rows[1].lower_1)

    hdi = brm_summarize_draws(draws; probs=[0.9], how=:hdi)
    # Order-statistic window over 1..100: 91 consecutive integers, width 90.
    # (Equal-tailed quantiles interpolate, so ETI can be a hair narrower
    # on lattice data; the skewed case below shows the strict HDI win.)
    @test hdi[1].upper_1 - hdi[1].lower_1 ≈ 90.0
    @test count(v -> hdi[1].lower_1 <= v <= hdi[1].upper_1, 1.0:100.0) == 91
    skewed = reshape(exp.(collect(range(-2.0, 2.0; length=200))), 200, 1)
    hs = brm_summarize_draws(skewed; probs=[0.8], how=:hdi)
    es = brm_summarize_draws(skewed; probs=[0.8], how=:eti)
    @test hs[1].upper_1 - hs[1].lower_1 < es[1].upper_1 - es[1].lower_1
    covered = count(v -> hs[1].lower_1 <= v <= hs[1].upper_1, vec(skewed))
    @test covered / 200 >= 0.79 # window covers ~the mass

    @test_throws ArgumentError brm_summarize_draws(draws; how=:nope)
    @test_throws ErrorException brm_summarize_draws(draws; probs=Float64[])
    @test_throws ErrorException brm_summarize_draws(draws; probs=[0.5, 0.9])
    @test_throws ErrorException brm_summarize_draws(draws; probs=[1.0])
    @test_throws ErrorException brm_summarize_draws(ones(1, 3))
    bad = copy(draws)
    bad[1, 2] = NaN
    @test_throws ErrorException brm_summarize_draws(bad)
end

@testset "contrasts — layout math on synthetic draws" begin
    grid = (; g=["a", "a", "b", "b", "c", "c"], x=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    draws = [10.0 11.0 20.0 22.0 30.0 33.0;
             12.0 13.0 24.0 26.0 36.0 39.0]
    cond = (; grid, draws, target=:mean, logical=:mu, scale=:response)
    contrast = brm_contrast_draws(cond; by=:g)
    @test contrast.pairs == [("a", "b"), ("a", "c"), ("b", "c")]
    @test contrast.labels == ["a vs b", "a vs c", "b vs c"]
    @test contrast.n_sub == 2
    @test contrast.draws == [-10.0 -11.0 -20.0 -22.0 -10.0 -11.0;
                             -12.0 -13.0 -24.0 -26.0 -12.0 -13.0]
    @test contrast.subgrid.g == ["a", "a"]
    @test contrast.subgrid.x == [1.0, 2.0]
    @test contrast.target === :mean

    ref = brm_contrast_draws(cond; by=:g, pairs=:reference)
    @test ref.pairs == [("b", "a"), ("c", "a")]
    seq = brm_contrast_draws(cond; by=:g, pairs=:sequential)
    @test seq.pairs == [("b", "a"), ("c", "b")]
    one = brm_contrast_draws(cond; by=:g, pairs="c" => "a")
    @test one.pairs == [("c", "a")]
    @test one.draws == draws[:, 5:6] .- draws[:, 1:2]
    ratio = brm_contrast_draws(cond; by=:g, pairs=[("b", "a")], how=:ratio)
    @test ratio.draws ≈ draws[:, 3:4] ./ draws[:, 1:2]

    @test_throws ErrorException brm_contrast_draws((; grid); by=:g)
    @test_throws ErrorException brm_contrast_draws(cond; by=:nope)
    @test_throws ArgumentError brm_contrast_draws(cond; by=:g, how=:lift)
    @test_throws ArgumentError brm_contrast_draws(
        cond; by=:g, pairs=:nope)
    @test_throws ErrorException brm_contrast_draws(
        cond; by=:g, pairs=[("a", "zzz")])
    single = (; grid=(; g=fill("a", 2), x=[1.0, 2.0]), draws=draws[:, 1:2])
    @test_throws ErrorException brm_contrast_draws(single; by=:g)
    uneven = (; grid=(; g=["a", "b", "b"], x=[1.0, 1.0, 2.0]),
                draws=draws[:, 1:3])
    @test_throws ErrorException brm_contrast_draws(uneven; by=:g)
    skewed_grid = (; grid=(; g=["a", "a", "b", "b"], x=[1.0, 2.0, 1.0, 9.0]),
                     draws=draws[:, 1:4])
    @test_throws ErrorException brm_contrast_draws(skewed_grid; by=:g)
    @test_throws DimensionMismatch brm_contrast_draws(
        (; grid, draws=draws[:, 1:3]); by=:g)
end

@testset "engine error paths — all compile-free" begin
    grid = brm_prediction_grid(d; focal=:x, n=4)
    unc = zeros(2, 1) # shape never reached: every branch below fails first
    @test_throws ArgumentError brm_conditional_draws(
        d, unc, grid; target=:nope)
    @test_throws ArgumentError brm_conditional_draws(
        d, unc, grid; scale=:nope)
    @test_throws ErrorException brm_conditional_draws(
        d, zeros(0, 1), grid)
    @test_throws ErrorException brm_conditional_draws(
        d, unc, (x=[1.0],); response=:yN)
    @test_throws DimensionMismatch brm_conditional_draws(
        d, unc, (; x=[1.0, 2.0], g=["a"]); response=:yN)
    # Multi-outcome fixture: :auto is ambiguous everywhere.
    @test_throws ErrorException brm_conditional_draws(d, unc, grid)
    @test_throws ErrorException brm_conditional_draws(
        d, unc, grid; response=:nope)
    @test_throws ErrorException brm_conditional_draws(
        d, unc, grid; response=:yN, logical=:muB)
    @test_throws ErrorException brm_conditional_draws(
        d, unc, grid; target=:predictive)
    # A link-scale request still needs an unambiguous linear predictor.
    @test_throws ErrorException brm_conditional_draws(
        d, unc, grid; response=:yN, scale=:link)

    # Slope validation precedes any Stan execution.
    @test_throws ArgumentError brm_slope_draws(d, unc, grid; wrt=:x, eps=0.0)
    @test_throws ArgumentError brm_slope_draws(d, unc, grid; wrt=:x, eps=Inf)
    @test_throws ErrorException brm_slope_draws(d, unc, grid; wrt=:nope)
    str_grid = (; s=["a", "b"], x=[1.0, 2.0])
    @test_throws ErrorException brm_slope_draws(d, unc, str_grid; wrt=:s)
    int_grid = brm_prediction_grid(d; focal=Dict(:n => [10, 12]))
    @test_throws ErrorException brm_slope_draws(d, unc, int_grid; wrt=:n)

    # (Observation-LHS links such as `log(y) ~ Normal(mu, sigma)` have no
    # lowering — the model itself fails to build — so the engine's LHS
    # guard is defense-in-depth only, with no test.)

    # An unmapped family names the :link / :predictive alternative.
    w_df = (; x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 1.5])
    w_builder = @brm begin
        mu ~ 1 + x
        y ~ Weibull(2.0, mu)
    end
    dw = brm_descriptor(w_builder, w_df; mod=@__MODULE__, name=:cond_weibull)
    grid_w = brm_prediction_grid(dw; focal=:x, n=3)
    @test_throws ErrorException brm_conditional_draws(dw, unc, grid_w)

    # Entangled mean arguments have no response-mean reading: a bare
    # sum is not one linear predictor, and an unsupported wrapper would
    # silently change the map.
    t_df = (; x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 1.5])
    t_sum = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu + x, sigma)
    end
    dt = brm_descriptor(t_sum, t_df; mod=@__MODULE__, name=:cond_tangled)
    grid_t = brm_prediction_grid(dt; focal=:x, n=3)
    @test_throws ErrorException brm_conditional_draws(dt, unc, grid_t)
    t_log = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(log(mu), sigma)
    end
    dl2 = brm_descriptor(t_log, t_df; mod=@__MODULE__, name=:cond_loglink)
    grid_l2 = brm_prediction_grid(dl2; focal=:x, n=3)
    @test_throws ErrorException brm_conditional_draws(dl2, unc, grid_l2)

    # A chained outcome (likelihood on another response) is refused: its
    # grid values would be the arbitrary response fill.
    ch_df = (; x=[0.0, 1.0, 2.0], y1=[1.0, 2.0, 1.5], y2=[0.5, 1.5, 1.0])
    ch_builder = @brm begin
        s ~ Exponential(1)
        mu ~ 1 + x
        y1 ~ Normal(mu, s)
        y2 ~ Normal(y1, s)
    end
    dch = brm_descriptor(ch_builder, ch_df; mod=@__MODULE__, name=:cond_chain)
    grid_ch = brm_prediction_grid(dch; focal=:x, n=3)
    @test_throws ErrorException brm_conditional_draws(
        dch, unc, grid_ch; response=:y2)

    # Joint responses have no v1 grid at all.
    joint_builder = @brm begin
        L_res ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
        mu1 ~ 1 + x
        mu2 ~ 1 + x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)
    end
    joint_df = (; x=[-1.0, 0.0, 1.0], y1=[0.1, 0.2, -0.1],
                  y2=[1.1, 0.9, 1.2])
    dj = brm_descriptor(joint_builder, joint_df; mod=@__MODULE__, name=:cond_joint)
    @test_throws ErrorException brm_prediction_grid(dj; focal=:x, n=3)
end

# ---- compiled engine --------------------------------------------------------
# One Stan program, eight outcomes. Unconstrained draws are fixed-seed
# synthetic points — the checks below verify the evaluation machinery
# (reprocess → constrain → family map), not a sampler.

grid_x = brm_prediction_grid(d; focal=:x, n=6)
d2 = brm_execute(d, :reprocess, grid_x)
prob = brm_execute(d2, :instantiate)
unc_names = BridgeStan.param_unc_names(prob.model)
unc = randn(MersenneTwister(20260918), 12, length(unc_names)) .* 0.5

function cond_constrained(prob, draws)
    tp_names = BridgeStan.param_names(prob.model; include_tp=true,
                                      include_gq=false)
    constrained = Matrix{Float64}(undef, size(draws, 1), length(tp_names))
    for (i, row) in enumerate(eachrow(draws))
        constrained[i, :] .= BridgeStan.param_constrain(
            prob.model, collect(Float64, row); include_tp=true,
            include_gq=false)
    end
    constrained, tp_names
end

@testset "means — exact family maps, no RNG" begin
    cond = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                 response=:yN, focal=:x)
    @test cond.target === :mean
    @test cond.scale === :response
    @test cond.logical === :muN
    @test cond.response === :yN
    @test cond.focal === :x
    @test size(cond.draws) == (12, 6)
    @test all(isfinite, cond.draws)
    again = brm_conditional_draws(d, unc, grid_x; problem=cond.problem,
                                  response=:yN)
    @test again.draws == cond.draws # deterministic: no RNG under :mean

    link = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                 response=:yN, scale=:link, logical=:muN)
    @test link.draws == cond.draws # Normal/identity: response == link

    blogit = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                   response=:yB)
    linkb = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                  response=:yB, scale=:link, logical=:muB)
    @test blogit.draws ≈ logistic.(linkb.draws) atol=1e-12

    nb = brm_conditional_draws(d, unc, grid_x; problem=prob, response=:yNB)
    linknb = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                   response=:yNB, scale=:link, logical=:muNB)
    @test nb.draws ≈ exp.(linknb.draws) atol=1e-12

    gam = brm_conditional_draws(d, unc, grid_x; problem=prob, response=:yG)
    linka = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                  response=:yG, scale=:link, logical=:aA)
    theta_c = let (cc, nn) = cond_constrained(prob, unc)
        cc[:, brm_output_coordinates(d2, :theta, nn)]
    end
    @test gam.draws ≈ exp.(linka.draws) .* theta_c atol=1e-12

    binom = brm_conditional_draws(d, unc, grid_x; problem=prob, response=:yBi)
    linkbi = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                   response=:yBi, scale=:link, logical=:muBi)
    trials = repeat(reshape(Float64.(grid_x.n), 1, 6), 12, 1)
    @test binom.draws ≈ trials .* logistic.(linkbi.draws) atol=1e-12

    binom2 = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                   response=:yBi2)
    linkbi2 = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                    response=:yBi2, scale=:link,
                                    logical=:muBi2)
    @test binom2.draws ≈ 10.0 .* logistic.(linkbi2.draws) atol=1e-12

    pois = brm_conditional_draws(d, unc, grid_x; problem=prob, response=:yP)
    linkp = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                  response=:yP, scale=:link, logical=:muP)
    @test pois.draws ≈ exp.(linkp.draws) atol=1e-12

    expo = brm_conditional_draws(d, unc, grid_x; problem=prob, response=:yE)
    linke = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                  response=:yE, scale=:link, logical=:muE)
    @test expo.draws ≈ exp.(linke.draws) atol=1e-12

    # A data-valued mean argument evaluates to the grid column, constant
    # across draws.
    grid_p = brm_prediction_grid(d; focal=Dict(:p => [0.1, 0.5, 0.9]))
    bernp = brm_conditional_draws(d, unc, grid_p; response=:yBe)
    @test bernp.draws == repeat(reshape([0.1, 0.5, 0.9], 1, 3), 12, 1)

    # LHS-linked declarations resolve to the public (response-mapped)
    # predictor, so response and link agree exactly here — and the
    # Monte-Carlo check below proves the carrier is the true mean.
    decl = brm_conditional_draws(d, unc, grid_x; problem=prob, response=:yLN)
    linkln = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                   response=:yLN, scale=:link, logical=:muLN)
    @test decl.draws == linkln.draws
    @test decl.logical === :muLN
    declnb = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                   response=:yNB2)
    linknb2 = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                    response=:yNB2, scale=:link,
                                    logical=:muNB2)
    @test declnb.draws == linknb2.draws
end

@testset "slopes — central differences equal constrained coefficients" begin
    slope = brm_slope_draws(d, unc, grid_x; wrt=:x, response=:yN, focal=:x)
    @test slope.target === :mean
    @test slope.wrt === :x
    @test slope.eps == 1e-4
    @test slope.response === :yN
    @test size(slope.draws) == (12, 6)

    # The Gaussian LP is linear in x: its slope IS the x coefficient.
    constrained, tp_names = cond_constrained(prob, unc)
    qx = brm_population_effect_coordinates(d2, :muN, tp_names; coefficient=:x)
    beta_x = constrained[:, only(qx.coordinates)]
    @test slope.draws ≈ repeat(beta_x, 1, 6) atol=1e-8

    # Chain rule through the family links.
    slope_b = brm_slope_draws(d, unc, grid_x; wrt=:x, response=:yB)
    qb = brm_population_effect_coordinates(d2, :muB, tp_names; coefficient=:x)
    beta_b = constrained[:, only(qb.coordinates)]
    pb = brm_conditional_draws(d, unc, grid_x; problem=prob,
                               response=:yB).draws
    @test slope_b.draws ≈ pb .* (1 .- pb) .* repeat(beta_b, 1, 6) atol=1e-6

    slope_nb = brm_slope_draws(d, unc, grid_x; wrt=:x, response=:yNB)
    qnb = brm_population_effect_coordinates(d2, :muNB, tp_names; coefficient=:x)
    beta_nb = constrained[:, only(qnb.coordinates)]
    munb = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                 response=:yNB).draws
    @test slope_nb.draws ≈ munb .* repeat(beta_nb, 1, 6) atol=1e-6

    slope_g = brm_slope_draws(d, unc, grid_x; wrt=:x, response=:yG)
    qa = brm_population_effect_coordinates(d2, :aA, tp_names; coefficient=:x)
    beta_a = constrained[:, only(qa.coordinates)]
    resp_g = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                   response=:yG).draws
    @test slope_g.draws ≈ resp_g .* repeat(beta_a, 1, 6) atol=1e-6

    # Chain rule through a declaration link: d muLN/dx = muLN * beta.
    slope_ln = brm_slope_draws(d, unc, grid_x; wrt=:x, response=:yLN)
    qln = brm_population_effect_coordinates(d2, :muLN, tp_names; coefficient=:x)
    beta_ln = constrained[:, only(qln.coordinates)]
    muln = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                 response=:yLN).draws
    @test slope_ln.draws ≈ muln .* repeat(beta_ln, 1, 6) atol=1e-6

    # An integer-compatible eps perturbs within the integers; the Gaussian
    # mean does not read the trial column, so its slope there is zero.
    int_grid = brm_prediction_grid(d; focal=Dict(:n => [10, 12]))
    slope_n = brm_slope_draws(d, unc, int_grid; wrt=:n, eps=2, response=:yN)
    @test slope_n.draws ≈ zeros(12, 2) atol=1e-12
end

@testset "contrasts — treatment coefficients across levels" begin
    grid_g = brm_prediction_grid(d; focal=:g)
    d2g = brm_execute(d, :reprocess, grid_g)
    probg = brm_execute(d2g, :instantiate)
    cond = brm_conditional_draws(d, unc, grid_g; problem=probg, response=:yN)
    @test size(cond.draws) == (12, 3)
    contrast = brm_contrast_draws(cond; by=:g, pairs=[(2, 1)])
    manual = cond.draws[:, 2:2] .- cond.draws[:, 1:1]
    @test contrast.draws ≈ manual
    @test contrast.labels == ["2 vs 1"]

    # The 2-vs-1 difference at fixed x IS the treatment coefficient.
    constrained, tp_names = cond_constrained(probg, unc)
    qg = brm_population_effect_coordinates(d2g, :muN, tp_names; coefficient=:g)
    entry = only(c for c in qg.contrasts if c.nonreference_level == 2)
    @test entry.reference_level == 1
    @test vec(contrast.draws) ≈ constrained[:, entry.coordinate] atol=1e-8

    ratio = brm_contrast_draws(cond; by=:g, pairs=:reference, how=:ratio)
    @test ratio.pairs == [(2, 1), (3, 1)]
    @test ratio.draws ≈ hcat(cond.draws[:, 2] ./ cond.draws[:, 1],
                             cond.draws[:, 3] ./ cond.draws[:, 1])
end

@testset "predictive — determinism and Monte-Carlo agreement" begin
    pred = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                 target=:predictive, response=:yN, seed=7)
    @test pred.target === :predictive
    @test isnothing(pred.logical)
    @test size(pred.draws) == (12, 6)
    pred2 = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                  target=:predictive, response=:yN, seed=7)
    @test pred2.draws == pred.draws
    pred3 = brm_conditional_draws(d, unc, grid_x; problem=prob,
                                  target=:predictive, response=:yN, seed=8)
    @test pred3.draws != pred.draws

    # Monte-Carlo means over RNG seeds agree with response means. The
    # tolerance is adaptive (5 posterior-RNG standard errors): it catches
    # a wrong family map (an O(1) scale error), not sampling noise.
    means = Dict(r => brm_conditional_draws(d, unc, grid_x; problem=prob,
                                            response=r).draws
                 for r in (:yN, :yB, :yNB, :yBi, :yBi2, :yP, :yLN, :yNB2,
                            :yG, :yE, :yBe))
    acc = Dict(r => zeros(12, 6) for r in keys(means))
    n_seeds = 64
    for seed in 1:n_seeds
        drawn = brm_predictive_draws(d2, unc; problem=prob, seed=1000 + seed)
        for r in keys(acc)
            acc[r] .+= getproperty(drawn, r)
        end
    end
    for r in keys(means)
        mc = acc[r] ./ n_seeds
        pooled = std(vec(getproperty(
            brm_predictive_draws(d2, unc; problem=prob, seed=1), r)))
        @test vec(mc) ≈ vec(means[r]) atol=5 * pooled / sqrt(n_seeds) + 1e-6
    end
end
