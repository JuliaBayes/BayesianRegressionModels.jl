# test/family_surfaces.jl — executable contracts for custom likelihood surfaces.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/family_surfaces.jl

using Test
using BayesianRegressionModels
using Distributions
using LogDensityProblems
using LogExpFunctions: logistic, logit
using Random
using Statistics
using StanBlocks

const FAMILY_SURFACE_CACHE = joinpath(tempdir(), "brm-family-surfaces")

df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    prog=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    math=[41.0, 48.0, 52.0, 57.0, 63.0, 69.0],
    y_zip=[0, 1, 0, 2, 3, 0],
    y_nb=[0, 1, 2, 4, 3, 6],
    y_ord=[1, 2, 3, 1, 2, 3],
    y_t=[-0.8, -0.2, 0.1, 0.7, 1.0, 1.4],
    y_truncated=[-0.7, -0.2, 0.1, 0.7, 1.0, 1.2],
    y_censored=[-0.5, -0.2, 0.1, 0.7, 1.0, 1.0],
    y_lognormal_censored=[0.25, 0.35, 0.60, 1.10, 1.80, 1.80],
    y_exponential_censored=[0.10, 0.30, 0.70, 1.10, 1.50, 1.50],
    y_weibull_censored=[0.20, 0.45, 0.75, 1.20, 1.60, 1.60],
    count_truncated=[1, 1, 2, 4, 3, 4],
    count_censored=[0, 1, 2, 4, 4, 4],
    # Genuine interval evidence: the response is the lower endpoint, while the
    # wrapper carries the row-wise upper endpoint.
    y_interval=[0.10, 0.25, 0.40, 0.70, 0.95, 1.20],
    y_interval_upper=[0.30, 0.50, 0.75, 1.05, 1.35, 1.70],
    y_interval_bad_upper=[0.05, 0.20, 0.35, 0.60, 0.90, 1.10],
    count_interval=[0, 0, 1, 2, 3, 4],
    count_interval_upper=[1, 2, 2, 4, 5, 6],
    y_bernoulli=[0, 1, 0, 1, 1, 0],
    trials=[8, 8, 10, 10, 12, 12],
    y_bb_shapes=[1, 2, 3, 4, 5, 7],
    y_bb_mean_precision=[0, 1, 4, 6, 8, 11],
    y_cat=[20, 10, 30, 20, 30, 10],
)

family_builder = @brm begin
    exp_prior ~ Exponential(2.5)
    gamma_prior ~ Gamma(3.0, 2.5)

    log(lambda) ~ 1 + x
    y_zip ~ ZeroInflatedPoisson(lambda, 0.25)

    # Exact catalogue shape for bambi:negative_binomial_interaction and
    # bambi:plot_pred_nb: the interaction includes a transformed operand.
    log(mu) ~ 0 + prog + zscale(math) + prog & zscale(math)
    log(phi) ~ 1
    y_nb ~ NegativeBinomial2(mu, phi)

    eta_ord ~ 1 + x
    y_ord ~ OrderedLogistic(eta_ord)

    loc ~ 1 + x
    log(scale) ~ 1
    y_t ~ LocationScale(loc, scale, TDist(4.0))

    y_truncated ~ truncated(Normal(loc, scale); lower=-0.75, upper=1.25)
    y_censored ~ censored(Normal(loc, scale); lower=-0.5, upper=1.0)
    y_lognormal_censored ~ censored(LogNormal(loc, scale); lower=0.25, upper=1.8)
    y_exponential_censored ~ censored(Exponential(lambda); lower=nothing, upper=1.5)
    y_weibull_censored ~ censored(Weibull(1.4, 1.1); upper=1.6)
    count_truncated ~ truncated(Poisson(lambda), 1, 4)
    count_censored ~ censored(Poisson(lambda); lower=0, upper=4)
    y_interval ~ interval_censored(Normal(loc, scale); upper=y_interval_upper)
    count_interval ~ interval_censored(Poisson(lambda); upper=count_interval_upper)

    y_bb_shapes ~ BetaBinomial(trials, 2.0, 4.0)

    logit(bb_mean) ~ 1 + x
    log(bb_precision) ~ 1
    y_bb_mean_precision ~ BetaBinomial2(trials, bb_mean, bb_precision)

    cat_eta2 ~ 1 + x
    cat_eta3 ~ 1 + prog
    y_cat ~ CategoricalLogit(cat_eta2, cat_eta3)
end

const BRM = BayesianRegressionModels

@testset "public numeric likelihood constructors are executable distributions" begin
    ordered = OrderedLogistic(0.2, [-1.0, 0.5])
    ordered_probs = [
        logistic(-1.0 - 0.2),
        logistic(0.5 - 0.2) - logistic(-1.0 - 0.2),
        logistic(0.2 - 0.5),
    ]
    @test ordered isa DiscreteUnivariateDistribution
    @test params(ordered) == (0.2, [-1.0, 0.5])
    @test OrderedLogistic(params(ordered)...) isa OrderedLogistic
    @test exp.(logpdf.(Ref(ordered), 1:3)) ≈ ordered_probs
    @test logpdf(ordered, 1.5) == -Inf

    categorical = CategoricalLogit(-0.3, 1.2)
    categorical_weights = exp.([0.0, -0.3, 1.2])
    categorical_probs = categorical_weights ./ sum(categorical_weights)
    @test categorical isa DiscreteUnivariateDistribution
    @test params(categorical) == (-0.3, 1.2)
    @test CategoricalLogit(params(categorical)...) isa CategoricalLogit
    @test exp.(logpdf.(Ref(categorical), 1:3)) ≈ categorical_probs
    @test logpdf(categorical, 0) == -Inf

    zip = ZeroInflatedPoisson(2.3, 0.25)
    poisson = Poisson(2.3)
    @test zip isa DiscreteUnivariateDistribution
    @test params(zip) == (2.3, 0.25)
    @test ZeroInflatedPoisson(params(zip)...) isa ZeroInflatedPoisson
    @test pdf(zip, 0) ≈ 0.25 + 0.75 * pdf(poisson, 0)
    @test pdf(zip, 4) ≈ 0.75 * pdf(poisson, 4)

    nb2 = NegativeBinomial2(3.2, 1.7)
    nb2_base = NegativeBinomial(1.7, 1.7 / (1.7 + 3.2))
    @test nb2 isa DiscreteUnivariateDistribution
    @test params(nb2) == (3.2, 1.7)
    @test NegativeBinomial2(params(nb2)...) isa NegativeBinomial2
    @test logpdf.(Ref(nb2), 0:10) ≈ logpdf.(Ref(nb2_base), 0:10)
    @test rand(MersenneTwister(42), nb2, 64) ==
          rand(MersenneTwister(42), nb2_base, 64)

    bb2 = BetaBinomial2(8, 0.4, 5.0)
    bb2_base = BetaBinomial(8, 2.0, 3.0)
    @test bb2 isa DiscreteUnivariateDistribution
    @test params(bb2) == (8, 0.4, 5.0)
    @test BetaBinomial2(params(bb2)...) isa BetaBinomial2
    @test logpdf.(Ref(bb2), 0:8) ≈ logpdf.(Ref(bb2_base), 0:8)
    @test rand(MersenneTwister(42), bb2, 64) ==
          rand(MersenneTwister(42), bb2_base, 64)

    binomial_logit = BinomialLogit(5, -0.3)
    binomial_base = Binomial(5, logistic(-0.3))
    @test binomial_logit isa DiscreteUnivariateDistribution
    @test params(binomial_logit) == (5, -0.3)
    @test BinomialLogit(params(binomial_logit)...) isa BinomialLogit
    @test logpdf.(Ref(binomial_logit), 0:5) ≈ logpdf.(Ref(binomial_base), 0:5)
    @test rand(MersenneTwister(42), binomial_logit, 64) ==
          rand(MersenneTwister(42), binomial_base, 64)

    skew_double_exponential = SkewDoubleExponential(0.3, 1.4, 0.25)
    skew_sepd = SkewedExponentialPower(
        0.3, 1.4 / (4 * 0.25 * (1 - 0.25)), 1.0, 0.25)
    @test skew_double_exponential isa ContinuousUnivariateDistribution
    @test params(skew_double_exponential) == (0.3, 1.4, 0.25)
    @test SkewDoubleExponential(params(skew_double_exponential)...) isa
          SkewDoubleExponential
    @test cdf(skew_double_exponential, 0.3) ≈ 0.25
    @test logpdf.(Ref(skew_double_exponential), -2.0:0.5:2.0) ≈
          logpdf.(Ref(skew_sepd), -2.0:0.5:2.0)
    @test rand(MersenneTwister(42), skew_double_exponential, 64) ==
          rand(MersenneTwister(42), skew_sepd, 64)

    censored_normal = TruncatedNormal(0.1, 1.2, -1.0, 2.0)
    censored_base = censored(Normal(0.1, 1.2), -1.0, 2.0)
    @test censored_normal isa UnivariateDistribution
    @test params(censored_normal) == (0.1, 1.2, -1.0, 2.0)
    @test TruncatedNormal(params(censored_normal)...) isa Distributions.Censored
    @test logpdf.(Ref(censored_normal), [-1.0, 0.2, 2.0]) ≈
          logpdf.(Ref(censored_base), [-1.0, 0.2, 2.0])
    @test rand(MersenneTwister(42), censored_normal, 64) ==
          rand(MersenneTwister(42), censored_base, 64)

    for d in (ordered, categorical, zip, nb2, bb2, binomial_logit,
              skew_double_exponential, censored_normal)
        draws = rand(MersenneTwister(20260729), d, 32)
        @test length(draws) == 32
        @test all(insupport.(Ref(d), draws))
    end

    @test_throws DomainError OrderedLogistic(0.0, [1.0, -1.0])
    @test_throws DomainError CategoricalLogit(Inf)
    @test_throws DomainError ZeroInflatedPoisson(-1.0, 0.2)
    @test_throws DomainError NegativeBinomial2(0.0, 1.0)
    @test_throws DomainError BetaBinomial2(5, 1.0, 2.0)
    @test_throws DomainError BinomialLogit(-1, 0.0)
    @test_throws DomainError SkewDoubleExponential(0.0, 0.0, 0.5)
    @test_throws DomainError SkewDoubleExponential(0.0, 1.0, 1.0)
    @test_throws DomainError TruncatedNormal(0.0, 1.0, 2.0, -2.0)
end

@testset "real likelihood distributions share the VBRMI contract" begin
    df_v = (;
        y_zip=[0, 1, 3],
        y_nb2=[0, 2, 5],
        y_bb2=[1, 3, 7],
        y_cat=[1, 2, 3],
        y_blogit=[0, 2, 5],
        y_skew=[-1.2, 0.1, 2.0],
    )
    builder_v = @brm begin
        y_zip ~ ZeroInflatedPoisson(2.3, 0.25)
        y_nb2 ~ NegativeBinomial2(3.2, 1.7)
        y_bb2 ~ BetaBinomial2(8, 0.4, 5.0)
        y_cat ~ CategoricalLogit(-0.3, 1.2)
        y_blogit ~ BinomialLogit(5, -0.3)
        y_skew ~ SkewDoubleExponential(0.3, 1.4, 0.25)
    end
    vbrmi = VBRMI(builder_v(df_v))
    expected =
        sum(logpdf.(ZeroInflatedPoisson(2.3, 0.25), df_v.y_zip)) +
        sum(logpdf.(NegativeBinomial2(3.2, 1.7), df_v.y_nb2)) +
        sum(logpdf.(BetaBinomial2(8, 0.4, 5.0), df_v.y_bb2)) +
        sum(logpdf.(CategoricalLogit(-0.3, 1.2), df_v.y_cat)) +
        sum(logpdf.(BinomialLogit(5, -0.3), df_v.y_blogit)) +
        sum(logpdf.(SkewDoubleExponential(0.3, 1.4, 0.25), df_v.y_skew))
    @test LogDensityProblems.dimension(vbrmi) == 0
    @test logpdf(vbrmi, Float64[]) ≈ expected
end

@testset "Distributions.jl constructor semantics map to native Stan" begin
    expected_names = (
        Normal=:normal,
        Cauchy=:cauchy,
        TDist=:student_t,
        Exponential=:exponential,
        Gamma=:gamma,
        Beta=:beta,
        Uniform=:uniform,
        LogNormal=:lognormal,
        Laplace=:double_exponential,
        Logistic=:logistic,
        Gumbel=:gumbel,
        Chisq=:chi_square,
        Frechet=:frechet,
        Rayleigh=:rayleigh,
        SkewNormal=:skew_normal,
        Pareto=:pareto,
        Erlang=:gamma,
        Arcsine=:beta,
        NormalCanon=:normal,
        SkewDoubleExponential=:skew_double_exponential,
        SkewedExponentialPower=:skew_double_exponential,
        Weibull=:weibull,
        InverseGamma=:inv_gamma,
        Bernoulli=:bernoulli,
        BernoulliLogit=:bernoulli_logit,
        Binomial=:binomial,
        BinomialLogit=:binomial_logit,
        BetaBinomial=:beta_binomial,
        Poisson=:poisson,
        NegativeBinomial=:neg_binomial,
    )
    for (julia_name, stan_name) in pairs(expected_names)
        @test BRM._sb_stan_dist_name(getfield(@__MODULE__, julia_name)) === stan_name
    end

    # Full-arity constructors: only the three scale/probability mismatches and
    # standard T need argument translation.  Everything else reaches the
    # faithful native Stan analogue unchanged.
    @test BRM._sb_stan_dist_args(Normal, (:mu, :sigma)) == (:mu, :sigma)
    @test BRM._sb_stan_dist_args(Cauchy, (:mu, :sigma)) == (:mu, :sigma)
    @test BRM._sb_stan_dist_args(TDist, (:nu,)) == (:nu, 0, 1)
    @test BRM._sb_stan_dist_args(Exponential, (:theta,)) == (:(1.0 ./ theta),)
    @test BRM._sb_stan_dist_args(Gamma, (:alpha, :theta)) ==
          (:alpha, :(1.0 ./ theta))
    @test BRM._sb_stan_dist_args(Beta, (:alpha, :beta)) == (:alpha, :beta)
    @test BRM._sb_stan_dist_args(Uniform, (:a, :b)) == (:a, :b)
    @test BRM._sb_stan_dist_args(LogNormal, (:mu, :sigma)) == (:mu, :sigma)
    @test BRM._sb_stan_dist_args(Laplace, (:mu, :theta)) == (:mu, :theta)
    @test BRM._sb_stan_dist_args(Logistic, (:mu, :theta)) == (:mu, :theta)
    @test BRM._sb_stan_dist_args(Gumbel, (:mu, :theta)) == (:mu, :theta)
    @test BRM._sb_stan_dist_args(Chisq, (:nu,)) == (:nu,)
    @test BRM._sb_stan_dist_args(Frechet, (:alpha, :theta)) == (:alpha, :theta)
    @test BRM._sb_stan_dist_args(Rayleigh, (:sigma,)) == (:sigma,)
    @test BRM._sb_stan_dist_args(SkewNormal, (:xi, :omega, :alpha)) ==
          (:xi, :omega, :alpha)
    @test BRM._sb_stan_dist_args(Pareto, (:alpha, :theta)) == (:theta, :alpha)
    @test BRM._sb_stan_dist_args(Erlang, (:alpha, :theta)) ==
          (:alpha, :(1.0 ./ theta))
    @test BRM._sb_stan_dist_args(Arcsine, ()) == (0.5, 0.5)
    @test_throws ArgumentError BRM._sb_stan_dist_args(Arcsine, (:a, :b))
    @test BRM._sb_stan_dist_args(NormalCanon, (:eta, :lambda)) ==
          (:(eta ./ lambda), :(inv(sqrt(lambda))))
    @test BRM._sb_stan_dist_args(
        SkewDoubleExponential, (:mu, :sigma, :tau)) == (:mu, :sigma, :tau)
    @test BRM._sb_stan_dist_args(
        SkewedExponentialPower, (:mu, :sigma, 1.0, :alpha)) ==
        (:mu, :(((4.0 .* sigma) .* alpha) .* (1.0 - alpha)), :alpha)
    @test_throws ArgumentError BRM._sb_stan_dist_args(
        SkewedExponentialPower, (:mu, :sigma, 2.0, :alpha))
    @test BRM._sb_stan_dist_args(Weibull, (:alpha, :theta)) == (:alpha, :theta)
    @test BRM._sb_stan_dist_args(InverseGamma, (:alpha, :theta)) == (:alpha, :theta)
    @test BRM._sb_stan_dist_args(Bernoulli, (:p,)) == (:p,)
    @test BRM._sb_stan_dist_args(BernoulliLogit, (:logitp,)) == (:logitp,)
    @test BRM._sb_stan_dist_args(Binomial, (:n, :p)) == (:n, :p)
    @test BRM._sb_stan_dist_args(BinomialLogit, (:n, :logitp)) == (:n, :logitp)
    @test BRM._sb_stan_dist_args(Poisson, (:lambda,)) == (:lambda,)
    @test BRM._sb_stan_dist_args(NegativeBinomial, (:r, :p)) ==
          (:r, :(p ./ (1.0 - p)))

    # Preserve every shorter constructor that Distributions.jl 0.25 exposes.
    # The assertions use `params` as the executable Julia-side receipt.
    default_cases = (
        (Normal, (), params(Normal())),
        (Normal, (:mu,), (:mu, 1.0)),
        (Cauchy, (), params(Cauchy())),
        (Cauchy, (:mu,), (:mu, 1.0)),
        (Exponential, (), (1.0,)),
        (Gamma, (), params(Gamma())),
        (Gamma, (:alpha,), (:alpha, 1.0)),
        (Beta, (), params(Beta())),
        (Beta, (:alpha,), (:alpha, :alpha)),
        (Uniform, (), params(Uniform())),
        (LogNormal, (), params(LogNormal())),
        (LogNormal, (:mu,), (:mu, 1.0)),
        (Laplace, (), params(Laplace())),
        (Laplace, (:mu,), (:mu, 1.0)),
        (Logistic, (), params(Logistic())),
        (Logistic, (:mu,), (:mu, 1.0)),
        (Gumbel, (), params(Gumbel())),
        (Gumbel, (:mu,), (:mu, 1.0)),
        (Frechet, (), params(Frechet())),
        (Frechet, (:alpha,), (:alpha, 1.0)),
        (Rayleigh, (), params(Rayleigh())),
        (SkewNormal, (), params(SkewNormal())),
        (SkewNormal, (:alpha,), (0.0, 1.0, :alpha)),
        (Pareto, (), reverse(params(Pareto()))),
        (Pareto, (:alpha,), (1.0, :alpha)),
        (Erlang, (), (1.0, 1.0)),
        (Erlang, (:alpha,), (:alpha, 1.0)),
        (Arcsine, (), (0.5, 0.5)),
        (NormalCanon, (), params(Normal())),
        (Weibull, (), params(Weibull())),
        (Weibull, (:alpha,), (:alpha, 1.0)),
        (InverseGamma, (), params(InverseGamma())),
        (InverseGamma, (:alpha,), (:alpha, 1.0)),
        (Bernoulli, (), params(Bernoulli())),
        (BernoulliLogit, (), params(BernoulliLogit())),
        (Binomial, (), params(Binomial())),
        (Binomial, (:n,), (:n, 0.5)),
        (Poisson, (), params(Poisson())),
        # p=0.5 in both shorter NegativeBinomial constructors, hence beta=1.
        (NegativeBinomial, (), (1.0, 1.0)),
        (NegativeBinomial, (:r,), (:r, 1.0)),
    )
    for (D, args, expected) in default_cases
        @test BRM._sb_stan_dist_args(D, args) == expected
    end
end

special_numeric_df = (;
    y_zip=[0, 1, 3],
    y_nb2=[0, 2, 5],
    trials=fill(8, 3),
    y_bb2=[1, 3, 7],
)

special_numeric_builder = @brm begin
    y_zip ~ ZeroInflatedPoisson(2.3, 0.25)
    y_nb2 ~ NegativeBinomial2(3.2, 1.7)
    y_bb2 ~ BetaBinomial2(trials, 0.4, 5.0)
end

@testset "numeric special distributions agree with native Stan paths" begin
    descriptor = brm_descriptor(
        special_numeric_builder, special_numeric_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    cache = joinpath(tempdir(), "brm-distribution-semantics")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    pointwise = brm_execute(
        descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260729)

    expected = (;
        y_zip=ZeroInflatedPoisson(2.3, 0.25),
        y_nb2=NegativeBinomial2(3.2, 1.7),
        y_bb2=BetaBinomial2(8, 0.4, 5.0),
    )
    for (target, dist) in pairs(expected)
        actual = getproperty(pointwise, Symbol(target, :_likelihood))
        observed = getproperty(special_numeric_df, target)
        @test actual ≈ logpdf.(dist, observed) atol=1e-10
    end

    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=zeros(0, 32), seed=20260729)
    for (target, dist) in pairs(expected)
        draws = getproperty(predictions, Symbol(target, :_gen))
        @test length(draws) == length(getproperty(special_numeric_df, target)) * 32
        @test all(insupport.(Ref(dist), draws))
    end
end

generic_df = (;
    y_normal=[-0.8, 0.1, 1.4],
    y_cauchy=[-1.2, 0.3, 2.1],
    y_t=[-0.9, 0.2, 1.7],
    y_exp=[0.2, 1.3, 3.1],
    y_gamma=[0.4, 2.0, 5.5],
    y_beta=[0.1, 0.5, 0.9],
    y_uniform=[-0.5, 0.25, 1.5],
    y_lognormal=[0.3, 1.2, 4.0],
    y_laplace=[-1.1, 0.4, 2.2],
    y_logistic=[-1.3, 0.2, 2.1],
    y_gumbel=[-0.8, 0.4, 2.3],
    y_chisq=[0.2, 1.3, 4.2],
    y_frechet=[0.4, 1.5, 4.0],
    y_rayleigh=[0.2, 1.1, 3.2],
    y_skew_normal=[-1.0, 0.3, 2.2],
    y_pareto=[1.0, 1.8, 4.5],
    y_erlang=[0.3, 1.4, 4.2],
    y_arcsine=[0.1, 0.5, 0.9],
    y_normal_canon=[-0.8, 0.2, 1.4],
    y_weibull=[0.2, 1.5, 3.8],
    y_invgamma=[0.3, 1.4, 5.0],
    y_bernoulli=[0, 1, 1],
    y_bernoulli_logit=[1, 0, 1],
    y_binomial=[1, 3, 4],
    y_binomial_logit=[1, 3, 4],
    y_poisson=[0, 2, 6],
    y_nb=[0, 3, 7],
    y_skew=[-1.2, 0.1, 2.0],
    y_sepd=[-0.8, 0.4, 1.7],
)

generic_builder = @brm begin
    y_normal ~ Normal(0.3, 1.2)
    y_cauchy ~ Cauchy(-0.2, 1.4)
    y_t ~ TDist(4.5)
    y_exp ~ Exponential(2.5)
    y_gamma ~ Gamma(3.0, 2.5)
    y_beta ~ Beta(2.0, 4.0)
    y_uniform ~ Uniform(-1.0, 2.0)
    y_lognormal ~ LogNormal(0.2, 0.8)
    y_laplace ~ Laplace(-0.1, 1.3)
    y_logistic ~ Logistic(0.2, 1.1)
    y_gumbel ~ Gumbel(0.2, 1.1)
    y_chisq ~ Chisq(3.5)
    y_frechet ~ Frechet(2.2, 1.4)
    y_rayleigh ~ Rayleigh(1.5)
    y_skew_normal ~ SkewNormal(0.2, 1.1, 2.0)
    y_pareto ~ Pareto(2.2, 1.0)
    y_erlang ~ Erlang(3, 1.5)
    y_arcsine ~ Arcsine()
    y_normal_canon ~ NormalCanon(0.2, 1.1)
    y_weibull ~ Weibull(1.7, 2.2)
    y_invgamma ~ InverseGamma(3.2, 1.8)
    y_bernoulli ~ Bernoulli(0.65)
    y_bernoulli_logit ~ BernoulliLogit(-0.4)
    y_binomial ~ Binomial(5, 0.35)
    y_binomial_logit ~ BinomialLogit(5, -0.3)
    y_poisson ~ Poisson(2.3)
    y_nb ~ NegativeBinomial(2.5, 0.4)
    y_skew ~ SkewDoubleExponential(0.3, 1.4, 0.25)
    y_sepd ~ SkewedExponentialPower(-0.1, 0.8, 1.0, 0.7)
end

@testset "generic native-Stan mappings with complete GQ hooks are exact" begin
    descriptor = brm_descriptor(generic_builder, generic_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    cache = joinpath(tempdir(), "brm-distribution-semantics")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    pointwise = brm_execute(
        descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260729)

    expected = (;
        y_normal=Normal(0.3, 1.2),
        y_cauchy=Cauchy(-0.2, 1.4),
        y_t=TDist(4.5),
        y_exp=Exponential(2.5),
        y_gamma=Gamma(3.0, 2.5),
        y_beta=Beta(2.0, 4.0),
        y_uniform=Uniform(-1.0, 2.0),
        y_lognormal=LogNormal(0.2, 0.8),
        y_laplace=Laplace(-0.1, 1.3),
        y_logistic=Logistic(0.2, 1.1),
        y_gumbel=Gumbel(0.2, 1.1),
        y_chisq=Chisq(3.5),
        y_frechet=Frechet(2.2, 1.4),
        y_rayleigh=Rayleigh(1.5),
        y_skew_normal=SkewNormal(0.2, 1.1, 2.0),
        y_pareto=Pareto(2.2, 1.0),
        y_erlang=Erlang(3, 1.5),
        y_arcsine=Arcsine(),
        y_normal_canon=NormalCanon(0.2, 1.1),
        y_weibull=Weibull(1.7, 2.2),
        y_invgamma=InverseGamma(3.2, 1.8),
        y_bernoulli=Bernoulli(0.65),
        y_bernoulli_logit=BernoulliLogit(-0.4),
        y_binomial=Binomial(5, 0.35),
        y_binomial_logit=BinomialLogit(5, -0.3),
        y_poisson=Poisson(2.3),
        y_nb=NegativeBinomial(2.5, 0.4),
        y_skew=SkewDoubleExponential(0.3, 1.4, 0.25),
        y_sepd=SkewedExponentialPower(-0.1, 0.8, 1.0, 0.7),
    )
    for (target, dist) in pairs(expected)
        actual = getproperty(pointwise, Symbol(target, :_likelihood))
        observed = getproperty(generic_df, target)
        @test actual ≈ logpdf.(dist, observed) atol=1e-10
    end

    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=zeros(0, 32), seed=20260729)
    for (target, dist) in pairs(expected)
        draws = getproperty(predictions, Symbol(target, :_gen))
        @test length(draws) == length(getproperty(generic_df, target)) * 32
        @test all(insupport.(Ref(dist), draws))
    end
end

@testset "Laplace is the q=0.5 asymmetric-Laplace special case" begin
    mu = 0.3
    quantile_scale = 0.7
    q = 0.5

    for y in (-1.1, 0.0, mu, 0.9, 2.2)
        u = (y - mu) / quantile_scale
        rho = u * (q - (u < 0))
        check_loss_logpdf = log(q * (1 - q)) - log(quantile_scale) - rho

        # brms/check-loss scale s maps to Laplace scale theta = 2s.
        @test logpdf(Laplace(mu, 2quantile_scale), y) ≈ check_loss_logpdf

        # Bambi/PyMC uses kappa=1 at q=0.5 and b as inverse Laplace scale.
        b = inv(2quantile_scale)
        bambi_logpdf = log(b / 2) - b * abs(y - mu)
        @test logpdf(Laplace(mu, inv(b)), y) ≈ bambi_logpdf
    end
end

semantic_df = (;
    y_exp=[0.2, 1.3, 3.1],
    y_gamma=[0.4, 2.0, 5.5],
    y_nb=[0, 3, 7],
)

semantic_builder = @brm begin
    y_exp ~ Exponential(2.5)
    y_gamma ~ Gamma(3.0, 2.5)
    y_nb ~ NegativeBinomial(2.5, 0.4)
end

@testset "translated scale and probability semantics reach every Stan path" begin
    descriptor = brm_descriptor(semantic_builder, semantic_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)

    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("y_exp ~ exponential((1.0 ./ 2.5));", code)
    @test occursin(
        "y_exp_likelihood = exponential_lpdfs(y_exp, (1.0 ./ 2.5));", code)
    @test occursin(
        "y_exp_gen = exponential_vector_rng(y_exp_n, (1.0 ./ 2.5));", code)
    @test occursin("y_gamma ~ gamma(3.0, (1.0 ./ 2.5));", code)
    @test occursin(
        "y_gamma_likelihood = gamma_lpdfs(y_gamma, 3.0, (1.0 ./ 2.5));", code)
    @test occursin(
        "y_gamma_gen = gamma_vector_rng(y_gamma_n, 3.0, (1.0 ./ 2.5));", code)
    @test occursin(
        "y_nb ~ neg_binomial(2.5, (0.4 ./ (1.0 - 0.4)));", code)
    @test occursin(
        "y_nb_likelihood = neg_binomial_lpmfs(y_nb, 2.5, (0.4 ./ (1.0 - 0.4)));",
        code)
    @test occursin(
        "y_nb_gen = neg_binomial_int_rng(y_nb_n, 2.5, (0.4 ./ (1.0 - 0.4)));",
        code)

    cache = joinpath(tempdir(), "brm-distribution-semantics")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    theta = Float64[]

    expected_pointwise = (;
        y_exp=logpdf.(Exponential(2.5), semantic_df.y_exp),
        y_gamma=logpdf.(Gamma(3.0, 2.5), semantic_df.y_gamma),
        y_nb=logpdf.(NegativeBinomial(2.5, 0.4), semantic_df.y_nb),
    )
    @test LogDensityProblems.dimension(problem) == 0
    # Stan sampling statements intentionally drop parameter-independent
    # constants, so the zero-parameter target is not the normalized Julia
    # density.  Exact normalized equality is checked through the pointwise
    # lpdf/lpmf outputs below; the parameter-dependent model target gets its
    # own gradient-equivalence test afterwards.
    @test isfinite(LogDensityProblems.logdensity(problem, theta))

    pointwise = brm_execute(
        descriptor, :pointwise_loglik; problem, draws=theta, seed=20260729)
    @test pointwise.y_exp_likelihood ≈ expected_pointwise.y_exp atol=1e-10
    @test pointwise.y_gamma_likelihood ≈ expected_pointwise.y_gamma atol=1e-10
    @test pointwise.y_nb_likelihood ≈ expected_pointwise.y_nb atol=1e-10

    # Exercise native Stan RNGs on the exact same translated arguments.  A
    # deterministic large generated-quantities batch checks the Julia mean and
    # variance contracts without requiring the two libraries' RNG streams to
    # be byte-identical.
    predictions = brm_execute(
        descriptor, :predict; problem, draws=zeros(0, 4096), seed=20260729)
    for (draws, dist, mean_atol, var_atol) in (
        (predictions.y_exp_gen, Exponential(2.5), 0.12, 0.5),
        (predictions.y_gamma_gen, Gamma(3.0, 2.5), 0.25, 1.5),
        (predictions.y_nb_gen, NegativeBinomial(2.5, 0.4), 0.2, 0.9),
    )
        @test mean(draws) ≈ mean(dist) atol=mean_atol
        @test var(vec(draws)) ≈ var(dist) atol=var_atol
    end
end

nb_density_builder = @brm begin
    log(r) ~ 1
    y_nb ~ NegativeBinomial(r, p)
end

@testset "NegativeBinomial model-density gradient matches Julia semantics" begin
    df_a = (; y_nb=[0, 3, 7], p=fill(0.4, 3))
    df_b = (; y_nb=[1, 2, 4], p=fill(0.4, 3))
    desc_a = brm_descriptor(nb_density_builder, df_a; mod=@__MODULE__)
    desc_b = brm_descriptor(nb_density_builder, df_b; mod=@__MODULE__)
    @test desc_a.id == desc_b.id
    @test occursin("p ./ (1.0 - p)", brm_execute(desc_a, :transpile))

    cache = joinpath(tempdir(), "brm-distribution-semantics")
    isdir(cache) || mkpath(cache)
    path = joinpath(cache, string(desc_a.id) * ".stan")
    problem_a = brm_execute(desc_a, :fit; path)
    problem_b = brm_execute(desc_b, :fit; path)
    @test LogDensityProblems.dimension(problem_a) == 1
    q = [0.3]
    _, grad_a = LogDensityProblems.logdensity_and_gradient(problem_a, q)
    _, grad_b = LogDensityProblems.logdensity_and_gradient(problem_b, q)

    loglik_delta(x) = begin
        d = NegativeBinomial(exp(only(x)), 0.4)
        sum(logpdf.(d, df_a.y_nb)) - sum(logpdf.(d, df_b.y_nb))
    end
    h = 1e-6
    julia_gradient =
        (loglik_delta(q .+ h) - loglik_delta(q .- h)) / (2h)
    # Priors and transforms are identical across the two data bindings, so
    # subtracting BridgeStan gradients isolates the model likelihood while
    # cancelling every unrelated target contribution and Jacobian.
    @test only(grad_a .- grad_b) ≈ julia_gradient atol=2e-6
end

@testset "SBBRMI lowers density, pointwise log-lik and RNG paths" begin
    plan = generative_plan(family_builder, df; mod=@__MODULE__)
    # StanBlocks 0eaebfa renamed the public HOF tokens without aliases. This
    # guard therefore also pins the exact compatibility boundary: the model
    # must resolve all three BRM wrappers against that public vocabulary.
    @test StanBlocks.stan.transpiles(plan.model)
    code = BayesianRegressionModels.stan_code(plan)

    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # These unused prior-only variables are correctly moved to generated
    # quantities by SLIC; their native RNG calls still prove the shared prior
    # path consumed the same scale -> rate normalization.
    @test occursin("exp_prior = exponential_rng((1.0 ./ 2.5));", code)
    @test occursin("gamma_prior = gamma_rng(3.0, (1.0 ./ 2.5));", code)
    @test occursin("zero_inflated_poisson(", code)
    @test occursin("neg_binomial_2(", code)
    @test occursin("ordered_logistic(", code)
    @test occursin("student_t(", code)
    @test occursin("beta_binomial(", code)
    @test occursin("bb_mean .* bb_precision", code)
    @test occursin("(1 - bb_mean) .* bb_precision", code)
    @test occursin("y_cat ~ categorical_logit(y_cat_categorical_logits);", code)
    @test occursin("categorical_logit_lpmf(", code)
    @test occursin("y_cat_categorical_logits", code)
    @test occursin(r"matrix\[\(2 \+ 1\), [^]]+\] y_cat_categorical_logits", code)
    @test !occursin("brm_categorical_logit", code)
    @test occursin("int_prog_x_zscale_math", code)

    sb = SBBRMI(family_builder(df); mod=@__MODULE__)
    math_mu = sum(df.math) / length(df.math)
    math_sd = sqrt(sum((x - math_mu)^2 for x in df.math) /
                   (length(df.math) - 1))
    expected_interaction = df.prog .* ((df.math .- math_mu) ./ math_sd)
    @test sb.data[:int_prog_x_zscale_math] ≈ expected_interaction
    @test sb.data[:y_cat] == [2, 1, 3, 2, 3, 1]
    @test sb.preproc[:y_cat].const_.levels == [10, 20, 30]

    new_df = merge(df, (;
        prog=[2.0, 1.0, 0.0, 0.0, 1.0, 2.0],
        math=[45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
    ))
    replayed = reprocess(sb, new_df)
    expected_replayed = new_df.prog .* ((new_df.math .- math_mu) ./ math_sd)
    @test replayed.data[:int_prog_x_zscale_math] ≈ expected_replayed
    @test replayed.data[:y_cat] == sb.data[:y_cat]
    unseen_cat = merge(new_df, (; y_cat=[10, 20, 30, 40, 10, 20]))
    @test_throws "not a training level" reprocess(sb, unseen_cat)

    for target in (:y_zip, :y_nb, :y_ord, :y_t, :y_bb_shapes,
                   :y_bb_mean_precision, :y_cat, :y_truncated, :y_censored,
                   :y_lognormal_censored, :y_exponential_censored,
                   :y_weibull_censored, :count_truncated, :count_censored,
                   :y_interval, :count_interval)
        declaration = only(d for d in plan.declarations if d.target === target)
        @test declaration.role === :observation
        @test !isnothing(declaration.draw)
        @test occursin(string(declaration.draw), code)
        @test occursin(string(target, "_likelihood"), code)
    end

    families = Dict(d.target => d.family for d in plan.declarations
                    if d.role === :observation)
    @test families[:y_zip] === :zero_inflated_poisson
    @test families[:y_nb] === :neg_binomial_2
    @test families[:y_ord] === :ordered_logistic
    @test families[:y_t] === :student_t
    @test families[:y_bb_shapes] === :beta_binomial
    @test families[:y_bb_mean_precision] === :beta_binomial
    @test families[:y_cat] === :categorical_logit
    @test families[:y_truncated] === :truncated
    @test families[:y_censored] === :censored
    @test families[:y_lognormal_censored] === :censored
    @test families[:y_exponential_censored] === :censored
    @test families[:y_weibull_censored] === :censored
    @test families[:count_truncated] === :truncated
    @test families[:count_censored] === :censored
    @test families[:y_interval] === :interval_censored
    @test families[:count_interval] === :interval_censored
end

@testset "mi() missing-response transpiles for scalar and vector family args" begin
    # Regression for snag `sbimpl-mi-impute`: `mi(y) ~ Normal(mu, sigma)` with a
    # SCALAR `sigma` parameter used to emit `sigma[Jmis]` inside `_sb_mi_normal`,
    # which the StanBlocks tracer rejects (`tracetype not defined for anything`)
    # because a `real` scalar cannot be int-vector indexed. The submodel now
    # slices each family arg with `maybe_index`, so a scalar broadcasts and a
    # vector is indexed. This is the canonical `docs/src/feature-atlas.md`
    # `missing_gaussian` example.
    scalar_scale = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end)((;
        x=[-1.0, 0.5, 2.0, 0.25],
        y=Union{Missing,Float64}[0.2, missing, -0.4, missing],
    ))
    sb = SBBRMI(scalar_scale; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # The scalar scale broadcasts — it must never be int-vector indexed.
    @test !occursin("sigma[", code)
    # The full-length location IS indexed at the observed and missing rows.
    @test occursin("mu[Jobs_y]", code)
    @test occursin("mu[Jmis_y]", code)
    # Observed rows contribute the likelihood; imputed values are the
    # optimised-away posterior-predictive draws reassembled by `mi_merge`.
    @test occursin("y_obs ~ normal(mu[Jobs_y], sigma);", code)
    @test occursin("mi_merge(", code)

    # A distributional (full-length vector) scale must still be indexed — the
    # `maybe_index` slice adapts per arg rank, so this path is unregressed.
    vector_scale = (@brm begin
        mu ~ 1 + x
        log(s) ~ 1 + x
        mi(y) ~ Normal(mu, s)
    end)((;
        x=[-1.0, 0.5, 2.0, 0.25],
        y=Union{Missing,Float64}[0.2, missing, -0.4, missing],
    ))
    vcode = BayesianRegressionModels.stan_code(SBBRMI(vector_scale; mod=@__MODULE__))
    @test StanBlocks.stanc_check(vcode; warn_pedantic=false).ok
    @test occursin("s[Jobs_y]", vcode)
    @test occursin("s[Jmis_y]", vcode)
end

@testset "Julia-native wrapper surface and operation resolution" begin
    brmi = family_builder(df)
    rhs_for(target) = begin
        op = parent(getproperty(brmi.operations, target))
        only(a for a in getargs(op) if a isa ExprColumn)
    end

    truncated_rhs = rhs_for(:y_truncated)
    @test getf(truncated_rhs) === truncated
    @test keys(getkwargs(truncated_rhs)) == (:lower, :upper)
    @test getf(only(getargs(truncated_rhs))) === Normal

    censored_rhs = rhs_for(:y_censored)
    @test getf(censored_rhs) === censored
    @test getf(only(getargs(censored_rhs))) === Normal

    exponential_rhs = rhs_for(:y_exponential_censored)
    @test BayesianRegressionModels._sb_normalize_bound(
        getkwargs(exponential_rhs).lower) === nothing

    interval_rhs = rhs_for(:y_interval)
    @test getf(interval_rhs) === interval_censored
    @test keys(getkwargs(interval_rhs)) == (:upper,)

    unsupported_builder = @brm begin
        eta ~ 1 + x
        y_bernoulli ~ censored(BernoulliLogit(eta); lower=0, upper=1)
    end
    # Construction retains the requested composition. A missing native CDF is
    # diagnosed when the producer resolves that operation, without a BRM family
    # allowlist deciding which otherwise valid calls may reach it.
    @test_throws Exception BRM.stan_code(SBBRMI(
        unsupported_builder(df); mod=@__MODULE__))
end

@testset "wrapper bounds fail before Stan" begin
    bad_interval = @brm begin
        loc ~ 1 + x
        log(scale) ~ 1
        y_interval ~ interval_censored(Normal(loc, scale); upper=y_interval_bad_upper)
    end
    @test_throws "lower endpoints must be strictly below upper endpoints" SBBRMI(
        bad_interval(df); mod=@__MODULE__)

    bad_keyword = @brm begin
        loc ~ 1 + x
        log(scale) ~ 1
        y_truncated ~ truncated(Normal(loc, scale); lower=-1.0, typo=1.0)
    end
    @test_throws "accepts only `lower` and `upper` keywords" SBBRMI(
        bad_keyword(df); mod=@__MODULE__)

    bad_discrete_bounds = @brm begin
        log(lambda) ~ 1 + x
        count_truncated ~ truncated(Poisson(lambda); lower=1.5, upper=4.0)
    end
    @test_throws "discrete lower bounds must be integers" SBBRMI(
        bad_discrete_bounds(df); mod=@__MODULE__)

    missing_bounds = @brm begin
        loc ~ 1 + x
        log(scale) ~ 1
        y_truncated ~ truncated(Normal(loc, scale))
    end
    @test_throws "needs at least one non-`nothing` bound" SBBRMI(
        missing_bounds(df); mod=@__MODULE__)
end

@testset "reserved-keyword data columns are rejected before Stan" begin
    # BRM emits each data column verbatim as a Stan `data` identifier, so a
    # column named after a Stan reserved keyword (`lower`/`upper`/`real`/...)
    # emitted `vector[lower_n] lower;` — invalid Stan that `transpiles()`
    # accepted but `stanc` rejected. The SBBRMI constructor now rejects the
    # collision with an actionable BRM error (snag `reserved-keyword`).
    reserved_builder = @brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        lower ~ interval_censored(Normal(mu, sigma); upper=upper)
    end
    reserved_df = (;
        x=[-1.0, 0.0, 1.0],
        lower=[-0.4, 0.1, 0.8],
        upper=[-0.1, 0.4, 1.2],
    )
    @test_throws "reserved keyword" SBBRMI(reserved_builder(reserved_df); mod=@__MODULE__)

    # The documented rename (`lower`/`upper` -> `y_lower`/`y_upper`) builds and
    # its emitted program is stanc-clean.
    renamed_builder = @brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y_lower ~ interval_censored(Normal(mu, sigma); upper=y_upper)
    end
    renamed_df = (;
        x=[-1.0, 0.0, 1.0],
        y_lower=[-0.4, 0.1, 0.8],
        y_upper=[-0.1, 0.4, 1.2],
    )
    plan = generative_plan(renamed_builder, renamed_df; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(plan)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

composed_numeric_df = (;
    y_truncated=[-0.6, 0.1, 1.1],
    y_lower_truncated=[-0.25, 0.2, 1.8],
    y_censored=[-0.5, 0.2, 1.0],
    count_truncated=[1, 2, 4],
    count_censored=[0, 2, 4],
    interval_lower=[-0.4, 0.0, 0.5],
    interval_upper=[-0.1, 0.4, 1.0],
    count_interval_lower=[0, 1, 3],
    count_interval_upper=[1, 3, 5],
)

composed_numeric_builder = @brm begin
    y_truncated ~ truncated(Normal(0.2, 1.1); lower=-0.75, upper=1.25)
    y_lower_truncated ~ truncated(Normal(0.2, 1.1); lower=-0.25)
    y_censored ~ censored(Normal(0.2, 1.1); lower=-0.5, upper=1.0)
    count_truncated ~ truncated(Poisson(2.2); lower=1, upper=4)
    count_censored ~ censored(Poisson(2.2); lower=0, upper=4)
    interval_lower ~ interval_censored(Normal(0.2, 1.1); upper=interval_upper)
    count_interval_lower ~ interval_censored(
        Poisson(2.2); upper=count_interval_upper)
end

@testset "wrapper pointwise and predictive semantics match Distributions" begin
    descriptor = brm_descriptor(
        composed_numeric_builder, composed_numeric_df; mod=@__MODULE__)
    code = brm_execute(descriptor, :transpile)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    mkpath(FAMILY_SURFACE_CACHE)
    problem = brm_execute(
        descriptor, :instantiate;
        path=joinpath(FAMILY_SURFACE_CACHE, string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 0
    pointwise = brm_execute(
        descriptor, :pointwise_loglik;
        problem, draws=Float64[], seed=20260729)

    normal = Normal(0.2, 1.1)
    poisson = Poisson(2.2)
    expected = (;
        y_truncated=logpdf.(
            Ref(truncated(normal; lower=-0.75, upper=1.25)),
            composed_numeric_df.y_truncated),
        y_lower_truncated=logpdf.(
            Ref(truncated(normal; lower=-0.25, upper=nothing)),
            composed_numeric_df.y_lower_truncated),
        y_censored=logpdf.(
            Ref(censored(normal; lower=-0.5, upper=1.0)),
            composed_numeric_df.y_censored),
        count_truncated=logpdf.(
            Ref(truncated(poisson; lower=1, upper=4)),
            composed_numeric_df.count_truncated),
        count_censored=logpdf.(
            Ref(censored(poisson; lower=0, upper=4)),
            composed_numeric_df.count_censored),
        interval_lower=log.(
            cdf.(Ref(normal), composed_numeric_df.interval_upper) .-
            cdf.(Ref(normal), composed_numeric_df.interval_lower)),
        count_interval_lower=log.(
            cdf.(Ref(poisson), composed_numeric_df.count_interval_upper) .-
            cdf.(Ref(poisson), composed_numeric_df.count_interval_lower)),
    )
    for (target, values) in pairs(expected)
        actual = getproperty(pointwise, Symbol(target, :_likelihood))
        @test actual ≈ values atol=1e-10
    end

    predictions = brm_execute(
        descriptor, :predict;
        problem, draws=zeros(0, 128), seed=20260729)
    @test all(-0.75 .<= predictions.y_truncated_gen .<= 1.25)
    @test all(predictions.y_lower_truncated_gen .>= -0.25)
    @test all(-0.5 .<= predictions.y_censored_gen .<= 1.0)
    @test all(1 .<= predictions.count_truncated_gen .<= 4)
    @test all(0 .<= predictions.count_censored_gen .<= 4)
    # Interval evidence changes the likelihood only: predictive draws remain
    # on the uncoarsened base-family scale and therefore need not lie inside
    # any observed interval.
    @test any(predictions.interval_lower_gen .< minimum(composed_numeric_df.interval_lower))
    @test all(predictions.count_interval_lower_gen .>= 0)
end

@testset "wrapper families execute through BridgeStan" begin
    sb = SBBRMI(family_builder(df); mod=@__MODULE__)
    mkpath(FAMILY_SURFACE_CACHE)
    path = joinpath(FAMILY_SURFACE_CACHE,
                    string(hash(BayesianRegressionModels.stan_code(sb))) * ".stan")
    problem = StanBlocks.stan_instantiate(sb.model; path)
    n = LogDensityProblems.dimension(problem)
    q = [0.05 * ((i % 5) - 2) for i in 1:n]
    lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    @test isfinite(lp)
    @test length(gradient) == n
    @test all(isfinite, gradient)
end
