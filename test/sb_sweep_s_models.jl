# test/sb_sweep_s_models.jl — @slic port builders for the v1 SB-parity sweep S driver.
#
# Included (not standalone) by test/sb_sweep_s.jl AFTER the
# ReactiveKernelsPPLExamples using block (builders consume the M.* data
# bindings). Split out so transpile probes can include the builders without
# running the record loop.

using SHA, Statistics  # batch 8: sha-pinned mcycle.csv data prep

# rate_2: two independent Beta-Binomials. (@brm refused: vector-of-sampled-
# params likelihood arg cannot lift to Stan.)
function rate_2_s()
    M = Rate2Example
    return @slic (; k1=M.RATE2_K1, n1=M.RATE2_N1, k2=M.RATE2_K2, n2=M.RATE2_N2) begin
        theta1 ~ beta(1, 1)
        theta2 ~ beta(1, 1)
        k1 ~ binomial(n1, theta1)
        k2 ~ binomial(n2, theta2)
    end
end

# rate_4: rate + prior-only thetaprior. (@brm refused: SBBRMI demotes the
# prior-only param to generated quantities.) StanBlocks demotes
# observation-unreached params in @slic too, so thetaprior rides `0 *` into
# the likelihood — density- and gradient-exact, keeping both sampled.
function rate_4_s()
    M = Rate4Example
    return @slic (; k=M.RATE4_K, n=M.RATE4_N) begin
        theta ~ beta(1, 1)
        thetaprior ~ beta(1, 1)
        theta_eff = theta + 0 * thetaprior
        k ~ binomial(n, theta_eff)
    end
end

# dugongs (BUGS form, inventory): length = α − β·λ^age with (α, β, u_λ,
# log_τ); λ ∈ (0.5,1) lub-mapped (Jacobian log(0.5)+log(s)+log(1−s), exactly
# Stan's), τ = exp(log_τ) (Jacobian log_τ); α/β ~ N(0,1000),
# λ ~ Uniform(0.5,1), τ ~ Gamma(1e-4,1e-4). (The partner's DUG *probe* is the
# von-Bertalanffy form — a different model, ported in sb_sweep_probes.jl.)
function dugongs_s()
    M = DugongsGrowthExample
    return @slic (; ages=M.DUGONGS_AGE, lengths=M.DUGONGS_LENGTH) begin
        alpha ~ normal(0, 1000)
        beta ~ normal(0, 1000)
        lambda ~ uniform(0.5, 1.0; lower=0.5, upper=1.0)
        tau ~ gamma(1e-4, 1e-4)
        sigma = 1 / sqrt(tau)
        mu = alpha - beta * exp(log(lambda) * ages)
        lengths ~ normal(mu, sigma)
    end
end
# pilots: crossed REs, mu = a[group] + b[scenario]; a ~ N(10*mua, sa),
# b ~ N(10*mub, sb); mua/mub ~ N(0,1); sa/sb/sy implicit-U[0,100] spelled
# explicit (-log(100) each; same [0,100] declaration both sides).
# (@brm refused: two hierarchical means, one population intercept.)
function pilots_s()
    M = PilotsExample
    g, s, y = M.PILOTS_GROUP_ID, M.PILOTS_SCENARIO_ID, M.PILOTS_Y
    @assert maximum(g) == 5 && maximum(s) == 8 && length(y) == 40
    return @slic (; group=g, scenario=s, y=y, ng=5, ns=8) begin
        mua ~ normal(0, 1)
        mub ~ normal(0, 1)
        sa ~ uniform(0, 100; lower=0, upper=100)
        sb ~ uniform(0, 100; lower=0, upper=100)
        sy ~ uniform(0, 100; lower=0, upper=100)
        a::vector[ng] ~ normal(10 * mua, sa)
        b::vector[ns] ~ normal(10 * mub, sb)
        mu = a[group] + b[scenario]
        y ~ normal(mu, sy)
    end
end

# glmm_poisson: cubic trend + per-obs RE, Poisson_log. Box uniforms with
# Stan lub transforms matching RK term-by-term; beta2's declaration
# [-10,20] is WIDER than its prior U(-10,10) (the .stan idiom) — spelled
# via split decl/prior bounds. AUDITED 2026-09-28: emitted decl reads
# [lower=-10, upper=20], prior uniform(-10, 10); Jacobian matches.
# (@brm refused: cannot express the decl/prior-bound split.)
function glmm_poisson_s()
    M = GLMMPoissonExample
    yr, c = M.GLMM_POISSON_YEAR, M.GLMM_POISSON_C
    n = length(c)
    return @slic (; year=yr, year2=yr .* yr, year3=yr .* yr .* yr, counts=c, n=n) begin
        alpha ~ uniform(-20, 20; lower=-20, upper=20)
        beta1 ~ uniform(-10, 10; lower=-10, upper=10)
        beta2 ~ uniform(-10, 10; lower=-10, upper=20)
        beta3 ~ uniform(-10, 10; lower=-10, upper=10)
        sigma ~ uniform(0, 5; lower=0, upper=5)
        eps::vector[n] ~ normal(0, sigma)
        mu = alpha + beta1 * year + beta2 * year2 + beta3 * year3 + eps
        counts ~ poisson(exp(mu))
    end
end

# sum_to_zero helpers: verbatim S2Z pivot (loop form, same IEEE op sequence
# as the source's vectorized form) + the effects family (K N(0,tau) terms +
# the +log(tau) subspace normalization, which has no native spelling).
@deffun begin
    s2z_pivot_vec(free::vector[7])::vector[8] = begin
        w::vector[7]
        for i in 1:7
            w[i] = free[i] / sqrt(i * (i + 1.0))
        end
        S = 0.0
        for i in 1:7
            S += w[i]
        end
        out::vector[8]
        out[1] = S
        prefix = 0.0
        for i in 1:7
            prefix += w[i]
            out[i + 1] = S - prefix + w[i] - (i + 1) * w[i]
        end
        out
    end
    @lpxf s2z_free_lpdf(free::vector[7], tau::real)::real = begin
        eff = s2z_pivot_vec(free)
        rv = log(tau)
        for i in 1:8
            rv += normal_lpdf(eff[i], 0.0, tau)
        end
        rv
    end
    s2z_free_lpdfs(free::vector[7], tau::real)::vector[7] = begin
        eff = s2z_pivot_vec(free)
        tot = log(tau)
        for i in 1:8
            tot += normal_lpdf(eff[i], 0.0, tau)
        end
        out::vector[7]
        for i in 1:7
            out[i] = tot / 7
        end
        out
    end
    s2z_free_rng(vector[7], tau::real)::vector[7] = begin
        out::vector[7]
        for i in 1:7
            out[i] = normal_rng(0.0, tau)
        end
        out
    end
end

# sum_to_zero: 8-schools S2Z model. alpha ~ N(0, sqrt(25 + tau^2/8)),
# tau ~ truncated-Cauchy(0,5) normalized, effects via the pivot with the
# +log(tau) subspace term (custom family), y ~ N(alpha + effect, se).
# (@brm refused: S2Z-exactness doubt; verbatim port is cheaper + certain.)
function sum_to_zero_s()
    M = EightSchoolsExample
    y, se = M.EIGHT_SCHOOLS_Y, M.EIGHT_SCHOOLS_SIGMA
    @assert length(y) == 8 && length(se) == 8
    return @slic (; y=y, se=se) begin
        tau ~ truncated(cauchy, 0.0, 5.0; lower=0.0)
        ascale = sqrt(25.0 + tau * tau / 8.0)
        alpha ~ normal(0, ascale)
        free::vector[7] ~ s2z_free(tau)
        effects = s2z_pivot_vec(free)
        y ~ normal(alpha + effects, se)
    end
end

# mvnormal: GLS with KNOWN AR(1) covariance, 4 data-form variants × beta ~
# N(0,10). The precision-chol variant forms Omega = Lc*Lc' host-side; all
# four forms agree BITWISE at the zeros point (-16.705053677430385), so no
# tolerance question arises — the record pins the exact program.
# (@brm refused: no known-covariance GLS surface.)
#
# The precision forms route through a thin custom family: StanBlocks ships
# no typed `multi_normal_prec_lpdfs` companion (snag
# missing-multi-no-54b27d39 on StanBlocks), so the native
# `y ~ multi_normal_prec(mu, Omega)` observation dies in the GQ twin with
# "tracetype not defined for y_likelihood::anything". The wrapper below is
# value-identical (direct builtin delegation, same arg order); the record
# pins the exact emitted program including the wrapper.
@deffun begin
    @lpxf sb_mvn_prec_lpdf(y::vector[n], mu::vector[n], Omega::matrix[n,n])::real =
        multi_normal_prec_lpdf(y, mu, Omega)
    sb_mvn_prec_lpdfs(y::vector[n], mu::vector[n], Omega::matrix[n,n])::real =
        multi_normal_prec_lpdf(y, mu, Omega)
    sb_mvn_prec_rng(vector[n], mu::vector[n], Omega::matrix[n,n])::vector[n] =
        multi_normal_rng(mu, inverse_spd(Omega))
end
function _mvnormal_s(Sigma, L, Omega, Omegah, form::Symbol)
    M = MVNormalRegressionExample
    X, y = M.MVREG_X, M.MVREG_Y
    k = size(X, 2)
    @assert length(y) == size(X, 1)
    base = (; X=X, y=y, k=k)
    if form === :cov
        return @slic (; X=X, y=y, k=k, Sigma=Sigma) begin
            beta::vector[k] ~ normal(0, 10)
            mu = X * beta
            y ~ multi_normal(mu, Sigma)
        end
    elseif form === :chol
        return @slic (; X=X, y=y, k=k, L=L) begin
            beta::vector[k] ~ normal(0, 10)
            mu = X * beta
            y ~ multi_normal_cholesky(mu, L)
        end
    elseif form === :prec
        return @slic (; X=X, y=y, k=k, Omega=Omega) begin
            beta::vector[k] ~ normal(0, 10)
            mu = X * beta
            y ~ sb_mvn_prec(mu, Omega)
        end
    else
        return @slic (; X=X, y=y, k=k, Omega=Omegah) begin
            beta::vector[k] ~ normal(0, 10)
            mu = X * beta
            y ~ sb_mvn_prec(mu, Omega)
        end
    end
end
mvnormal_cov_s() = _mvnormal_s(MVNormalRegressionExample.MVREG_COVARIANCE,
    zeros(0, 0), zeros(0, 0), zeros(0, 0), :cov)
mvnormal_chol_s() = _mvnormal_s(zeros(0, 0),
    MVNormalRegressionExample.MVREG_CHOL, zeros(0, 0), zeros(0, 0), :chol)
mvnormal_prec_s() = _mvnormal_s(zeros(0, 0), zeros(0, 0),
    MVNormalRegressionExample.MVREG_PRECISION, zeros(0, 0), :prec)
function mvnormal_prec_chol_s()
    Lc = MVNormalRegressionExample.MVREG_PRECISION_CHOL
    return _mvnormal_s(zeros(0, 0), zeros(0, 0), zeros(0, 0), Lc * Lc', :prechol)
end

# Batch 3: marginalized mixtures + BUGS GLMMs + survey.
#
# Mixture families (scalar-observation, @plate-driven; the per-observation
# label marginal is a stable log_mix / K-way log-sum-exp, exactly Stan's).
# (@brm refused: no mixture-density spelling.)
@deffun begin
    @lpxf mix2u_lpdf(y::real, mu::vector[2], theta::real)::real =
        log_mix(theta, normal_lpdf(y, mu[1], 1.0), normal_lpdf(y, mu[2], 1.0))
    mix2u_lpdfs(y::real, mu::vector[2], theta::real)::real =
        log_mix(theta, normal_lpdf(y, mu[1], 1.0), normal_lpdf(y, mu[2], 1.0))
    mix2u_rng(mu::vector[2], theta::real)::real = begin
        z = bernoulli_rng(theta)
        if (z == 1)
            normal_rng(mu[1], 1.0)
        else
            normal_rng(mu[2], 1.0)
        end
    end
    @lpxf mix2s_lpdf(y::real, mu::vector[2], sigma::vector[2], theta::real)::real =
        log_mix(theta, normal_lpdf(y, mu[1], sigma[1]), normal_lpdf(y, mu[2], sigma[2]))
    mix2s_lpdfs(y::real, mu::vector[2], sigma::vector[2], theta::real)::real =
        log_mix(theta, normal_lpdf(y, mu[1], sigma[1]), normal_lpdf(y, mu[2], sigma[2]))
    mix2s_rng(mu::vector[2], sigma::vector[2], theta::real)::real = begin
        z = bernoulli_rng(theta)
        if (z == 1)
            normal_rng(mu[1], sigma[1])
        else
            normal_rng(mu[2], sigma[2])
        end
    end
    @lpxf mixk_lpdf(y::real, mu::vector[K], sigma::vector[K], theta::vector[K])::real = begin
        rv = log(theta[1]) + normal_lpdf(y, mu[1], sigma[1])
        for k in 2:K
            rv = log_sum_exp(rv, log(theta[k]) + normal_lpdf(y, mu[k], sigma[k]))
        end
        rv
    end
    mixk_lpdfs(y::real, mu::vector[K], sigma::vector[K], theta::vector[K])::real =
        mixk_lpdf(y, mu, sigma, theta)
    mixk_rng(mu::vector[K], sigma::vector[K], theta::vector[K])::real = begin
        z = categorical_rng(theta)
        normal_rng(mu[z], sigma[z])
    end
    # Survey: the discrete survey-count n marginalizes to ONE scalar density
    # over the returns vector k (log_sum_exp over n = 1..nmax of
    # log(1/nmax) + binomial_lpmf(k|n,theta)). lchoose is spelled via lgamma
    # (StanBlocks has no lchoose builtin); the k[i] > n cells carry -Inf via
    # negative_infinity(), contributing exp(-Inf) = 0 exactly as the source.
    @lpxf survey_mix_lpmf(k::int[m], theta::real, sk::real, m::int, nmax::int, log_1_nmax::real)::real = begin
        lt = log(theta)
        l1t = log1m(theta)
        lp_parts::vector[nmax]
        for n in 1:nmax
            lc_n = 0.0
            bad = 0
            for i in 1:m
                if (k[i] > n)
                    bad = 1
                else
                    lc_n += lgamma(n + 1) - lgamma(k[i] + 1) - lgamma(n - k[i] + 1)
                end
            end
            if (bad == 1)
                lp_parts[n] = negative_infinity()
            else
                lp_parts[n] = log_1_nmax + lc_n + sk * lt + (n * m - sk) * l1t
            end
        end
        log_sum_exp(lp_parts)
    end
    survey_mix_lpmfs(k::int[m], theta::real, sk::real, m::int, nmax::int, log_1_nmax::real)::real =
        survey_mix_lpmf(k, theta, sk, m, nmax, log_1_nmax)
    survey_mix_rng(int[m], theta::real, sk::real, m::int, nmax::int, log_1_nmax::real)::int[m] = begin
        probs = rep_vector(1.0 / nmax, nmax)
        n = categorical_rng(probs)
        out::int[m]
        for i in 1:m
            out[i] = binomial_rng(n, theta)
        end
        out
    end
end

# normal_mixture: 2-comp N mixture, KNOWN unit variance, theta flat (the
# uniform(0,1) density is exactly 0), mu ~ N(0,10).
function normal_mixture_s()
    M = NormalMixtureExample
    y = M.NORMAL_MIXTURE_Y
    return @slic (; y=y, n=length(y)) begin
        theta ~ uniform(0.0, 1.0; lower=0.0, upper=1.0)
        mu::vector[2] ~ normal(0, 10)
        @plate for i in 1:n
            y[i] ~ mix2u(mu, theta)
        end
    end
end

# normal_mixture_k: K-comp mixture, simplex theta + bounded sigma both
# IMPLICIT-uniform (Jacobian-only: flat() with/without decl bounds), mu ~ N(0,10).
function normal_mixture_k_s()
    M = NormalMixtureKExample
    y = M.NORMAL_MIXTURE_K_Y
    K = M.NORMAL_MIXTURE_K_K
    return @slic (; y=y, n=length(y), K=K) begin
        theta::simplex[K] ~ flat()
        mu::vector[K] ~ normal(0, 10)
        sigma::vector[K] ~ flat(; lower=0.0, upper=10.0)
        @plate for i in 1:n
            y[i] ~ mixk(mu, sigma, theta)
        end
    end
end

# low_dim_gauss_mix: ordered[2] mu (label-switching break), sigma ~ N(0,2)
# PLAIN (half-normal, no log2 — kwargs bounds are unnormalized), theta ~ Beta(5,5).
function low_dim_gauss_mix_s()
    M = LowDimGaussMixExample
    y = M.LOW_DIM_GAUSS_MIX_Y
    return @slic (; y=y, n=length(y)) begin
        mu::ordered[2] ~ normal(0, 2)
        sigma::vector[2] ~ normal(0, 2; lower=0.0)
        theta ~ beta(5, 5; lower=0.0, upper=1.0)
        @plate for i in 1:n
            y[i] ~ mix2s(mu, sigma, theta)
        end
    end
end

# low_dim_gauss_mix_collapse: same, free mu.
function low_dim_gauss_mix_collapse_s()
    M = LowDimGaussMixCollapseExample
    y = M.LOW_DIM_GAUSS_MIX_COLLAPSE_Y
    return @slic (; y=y, n=length(y)) begin
        mu::vector[2] ~ normal(0, 2)
        sigma::vector[2] ~ normal(0, 2; lower=0.0)
        theta ~ beta(5, 5; lower=0.0, upper=1.0)
        @plate for i in 1:n
            y[i] ~ mix2s(mu, sigma, theta)
        end
    end
end

# seeds: BUGS binomial GLMM; tau ~ Gamma(1e-3,1e-3) on the PRECISION
# (@brm refused: priors are sd-scale), sigma = 1/sqrt(tau), per-plate RE.
function seeds_s()
    M = SeedsExample
    counts, totals, x1, x2 = M.SEEDS_COUNTS, M.SEEDS_TOTALS, M.SEEDS_X1, M.SEEDS_X2
    I = length(counts)
    return @slic (; counts=counts, totals=totals, x1=x1, x2=x2, I=I) begin
        alpha0 ~ normal(0, 1000)
        alpha1 ~ normal(0, 1000)
        alpha12 ~ normal(0, 1000)
        alpha2 ~ normal(0, 1000)
        tau ~ gamma(0.001, 0.001; lower=0.0)
        sigma = 1.0 / sqrt(tau)
        b::vector[I] ~ normal(0, sigma)
        eta = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * (x1 .* x2) + b
        counts ~ binomial_logit(totals, eta)
    end
end

# seeds_centered: N(0,1) fixed effects, sigma ~ Cauchy(0,1) PLAIN half-Cauchy
# (no log2), mean-centered RE b = c - mean(c); likelihood spelled
# binomial(totals, inv_logit(eta)) exactly as the source authors it.
# (@brm refused: exact-structure + variance-scale-adjacent prior.)
function seeds_centered_s()
    M = SeedsCenteredExample
    counts = M.SEEDS_CENTERED_COUNTS
    totals = M.SEEDS_CENTERED_TOTALS
    x1 = M.SEEDS_CENTERED_X1
    x2 = M.SEEDS_CENTERED_X2
    I = length(counts)
    return @slic (; counts=counts, totals=totals, x1=x1, x2=x2, I=I) begin
        alpha0 ~ normal(0, 1)
        alpha1 ~ normal(0, 1)
        alpha12 ~ normal(0, 1)
        alpha2 ~ normal(0, 1)
        sigma ~ cauchy(0, 1; lower=0.0)
        c::vector[I] ~ normal(0, sigma)
        eta = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * (x1 .* x2) + (c - sum(c) / I)
        counts ~ binomial(totals, inv_logit(eta))
    end
end

# surgical: hierarchical binomial-logit; sigmasq ~ InvGamma(1e-3,1e-3) on the
# VARIANCE (@brm refused: sd-scale priors), b ~ N(mu, sigma) centered on mu.
function surgical_s()
    M = SurgicalExample
    successes, totals = M.SURGICAL_SUCCESSES, M.SURGICAL_TOTALS
    N = length(successes)
    return @slic (; successes=successes, totals=totals, N=N) begin
        mu ~ normal(0, 1000)
        sigmasq ~ inv_gamma(0.001, 0.001; lower=0.0)
        sigma = sqrt(sigmasq)
        b::vector[N] ~ normal(mu, sigma)
        successes ~ binomial_logit(totals, b)
    end
end

# survey: returns vector k observes the n-marginalized family; theta
# implicit-uniform (flat, Jacobian-only). dim = 1.
function survey_s()
    M = SurveyModelExample
    return @slic (; k=M.SURVEY_K, sk=M.SURVEY_SK, m=M.SURVEY_M, nmax=M.SURVEY_NMAX,
        log_1_nmax=M.SURVEY_LOG1NMAX) begin
        theta ~ flat(; lower=0.0, upper=1.0)
        k ~ survey_mix(theta, sk, m, nmax, log_1_nmax)
    end
end

# Batch 4: capture-recapture augmentation marginals + binary IRT.
#
# M-family augmentation families: per-individual detection log-likelihood
# plus the inclusion-indicator marginal (s>0: log(omega) + bern; s==0:
# log_sum_exp(log(omega) + bern, log(1-omega))). All box priors are
# implicit-uniform (flat, Jacobian-only). (@brm refused: augmentation
# marginals have no GLM spelling.)
@deffun begin
    @lpxf aug0_lpmf(si::int, lc::real, omega::real, p::real, T::int)::real = begin
        obs = log(omega) + lc + si * log(p) + (T - si) * log1m(p)
        if (si > 0)
            obs
        else
            log_sum_exp(obs, log1m(omega))
        end
    end
    aug0_lpmfs(si::int, lc::real, omega::real, p::real, T::int)::real =
        aug0_lpmf(si, lc, omega, p, T)
    aug0_rng(lc::real, omega::real, p::real, T::int)::int = bernoulli_rng(omega)
    @lpxf augb_counts_lpmf(si::int, ai::real, bi::real, ei::real, fi::real, omega::real, p::real, c::real)::real = begin
        bern = ai * log(p) + bi * log1m(p) + ei * log(c) + fi * log1m(c)
        obs = log(omega) + bern
        if (si > 0)
            obs
        else
            log_sum_exp(obs, log1m(omega))
        end
    end
    augb_counts_lpmfs(si::int, ai::real, bi::real, ei::real, fi::real, omega::real, p::real, c::real)::real =
        augb_counts_lpmf(si, ai, bi, ei, fi, omega, p, c)
    augb_counts_rng(ai::real, bi::real, ei::real, fi::real, omega::real, p::real, c::real)::int =
        bernoulli_rng(omega)
    @lpxf augh_lpmf(yi::int, lc::real, ep::real, omega::real, T::int)::real = begin
        obs = log(omega) + binomial_logit_lpmf(yi, T, ep)
        if (yi > 0)
            obs
        else
            log_sum_exp(obs, log1m(omega))
        end
    end
    augh_lpmfs(yi::int, lc::real, ep::real, omega::real, T::int)::real =
        augh_lpmf(yi, lc, ep, omega, T)
    augh_rng(lc::real, ep::real, omega::real, T::int)::int = bernoulli_rng(omega)
    # Shared marginal over a precomputed per-individual bern (Mt/Mth/Mtbh).
    @lpxf augb_lpmf(si::int, bi::real, omega::real)::real = begin
        obs = log(omega) + bi
        if (si > 0)
            obs
        else
            log_sum_exp(obs, log1m(omega))
        end
    end
    augb_lpmfs(si::int, bi::real, omega::real)::real = augb_lpmf(si, bi, omega)
    augb_rng(bi::real, omega::real)::int = bernoulli_rng(omega)
    # Mt bern[i] = (Y . logit_p)[i] + bern0, bern0 = SUM log1m(p). Loop form
    # (same IEEE op sequence as the source's matvec + reduction up to BLAS
    # blocking — ~1ulp, inside any parity tolerance).
    mt_bern(Y::matrix[M,T], p::vector[T])::vector[M] = begin
        bern0 = 0.0
        for j in 1:T
            bern0 += log1m(p[j])
        end
        out::vector[M]
        for i in 1:M
            acc = 0.0
            for j in 1:T
                acc += Y[i, j] * logit(p[j])
            end
            out[i] = acc + bern0
        end
        out
    end
    # Mth bern[i] = SUM_j (Y*lp - log1p_exp(lp)), lp = logit(mean_p) + eps.
    mth_bern(Y::matrix[M,T], mean_p::vector[T], eps::vector[M])::vector[M] = begin
        out::vector[M]
        for i in 1:M
            acc = 0.0
            for j in 1:T
                lp = logit(mean_p[j]) + eps[i]
                acc += Y[i, j] * lp - log1p_exp(lp)
            end
            out[i] = acc
        end
        out
    end
    # Mtbh: same + the behavioural gamma*Yprev coefficient.
    mtbh_bern(Y::matrix[M,T], Yprev::matrix[M,T], mean_p::vector[T], eps::vector[M], gamma::real)::vector[M] = begin
        out::vector[M]
        for i in 1:M
            acc = 0.0
            for j in 1:T
                lp = logit(mean_p[j]) + eps[i] + gamma * Yprev[i, j]
                acc += Y[i, j] * lp - log1p_exp(lp)
            end
            out[i] = acc
        end
        out
    end
end

# m0: single detection probability + augmentation marginal. dim = 2.
function m0_s()
    M = M0Example
    return @slic (; s=M.M0_S, lchoose=M.M0_LCHOOSE, T=M.M0_T, MM=M.M0_M) begin
        omega ~ flat(; lower=0.0, upper=1.0)
        p ~ flat(; lower=0.0, upper=1.0)
        @plate for i in 1:MM
            s[i] ~ aug0(lchoose[i], omega, p, T)
        end
    end
end

# mb: behavioural (trap) response via the 4-count collapse. dim = 3.
function mb_s()
    M = MbExample
    return @slic (; s=M.MB_S, a=M.MB_A, b=M.MB_B, e=M.MB_E, f=M.MB_F, T=M.MB_T, MM=M.MB_M) begin
        omega ~ flat(; lower=0.0, upper=1.0)
        p ~ flat(; lower=0.0, upper=1.0)
        c ~ flat(; lower=0.0, upper=1.0)
        @plate for i in 1:MM
            s[i] ~ augb_counts(a[i], b[i], e[i], f[i], omega, p, c)
        end
    end
end

# mh: individual-heterogeneity RE (noncentered, sigma in [0,5]) +
# binomial-logit + augmentation marginal.
function mh_s()
    M = MhExample
    return @slic (; y=M.MH_Y, lchoose=M.MH_LCHOOSE, T=M.MH_T, MM=M.MH_M) begin
        omega ~ flat(; lower=0.0, upper=1.0)
        mean_p ~ flat(; lower=0.0, upper=1.0)
        sigma ~ flat(; lower=0.0, upper=5.0)
        eps_raw::vector[MM] ~ normal(0, 1)
        ep = logit(mean_p) + sigma * eps_raw
        @plate for i in 1:MM
            y[i] ~ augh(lchoose[i], ep[i], omega, T)
        end
    end
end

# mt: per-occasion p[j] + augmentation marginal.
function mt_s()
    M = MtExample
    return @slic (; Y=M.MT_Y, s=M.MT_S, T=M.MT_T, MM=M.MT_M) begin
        omega ~ flat(; lower=0.0, upper=1.0)
        p::vector[T] ~ flat(; lower=0.0, upper=1.0)
        bern = mt_bern(Y, p)
        @plate for i in 1:MM
            s[i] ~ augb(bern[i], omega)
        end
    end
end

# mth: occasion + RE outer-sum logit + augmentation marginal.
function mth_s()
    M = MthModelExample
    return @slic (; Y=M.MTH_Y, s=M.MTH_S, T=M.MTH_T, MM=M.MTH_M) begin
        omega ~ flat(; lower=0.0, upper=1.0)
        mean_p::vector[T] ~ flat(; lower=0.0, upper=1.0)
        sigma ~ flat(; lower=0.0, upper=5.0)
        eps_raw::vector[MM] ~ normal(0, 1)
        eps = sigma * eps_raw
        bern = mth_bern(Y, mean_p, eps)
        @plate for i in 1:MM
            s[i] ~ augb(bern[i], omega)
        end
    end
end

# mtbh: full occasion + RE + behavioural logit + augmentation marginal.
function mtbh_s()
    M = MtbhModelExample
    return @slic (; Y=M.MTBH_Y, Yprev=M.MTBH_YPREV, s=M.MTBH_S, T=M.MTBH_T, MM=M.MTBH_M) begin
        omega ~ flat(; lower=0.0, upper=1.0)
        mean_p::vector[T] ~ flat(; lower=0.0, upper=1.0)
        gamma ~ normal(0, 10)
        sigma ~ flat(; lower=0.0, upper=3.0)
        eps_raw::vector[MM] ~ normal(0, 1)
        eps = sigma * eps_raw
        bern = mtbh_bern(Y, Yprev, mean_p, eps, gamma)
        @plate for i in 1:MM
            s[i] ~ augb(bern[i], omega)
        end
    end
end

# lsat: Rasch 1PL on the 5x32 pattern matrix (2-D @plate; the flat binding
# reshapes question-outer/student-inner with zero transcription). beta > 0
# half-normal, PLAIN (no log2).
function lsat_s()
    M = LsatExample
    rmat = reshape(M.LSAT_RESP_FLAT, (5, 32))
    return @slic (; r=rmat, T=5, N=32) begin
        alpha::vector[T] ~ normal(0, 100)
        theta::vector[N] ~ normal(0, 1)
        beta ~ normal(0, 100; lower=0.0)
        @plate for k in 1:T, j in 1:N
            r[k, j] ~ bernoulli_logit(beta * theta[j] - alpha[k])
        end
    end
end

# irt_2pl: a[i]*(theta[j]-b[i]) on the IxJ matrix (2-D @plate); Cauchy
# hyperpriors PLAIN (no log2), a lognormal on vector<lower=0>.
function irt_2pl_s()
    M = Irt2plExample
    y = Int.(M.IRT_2PL_Y)
    I, J = size(y)
    return @slic (; y=y, I=I, J=J) begin
        sigma_theta ~ cauchy(0, 2; lower=0.0)
        theta::vector[J] ~ normal(0, sigma_theta)
        sigma_a ~ cauchy(0, 2; lower=0.0)
        a::vector[I] ~ lognormal(0, sigma_a; lower=0.0)
        mu_b ~ normal(0, 5)
        sigma_b ~ cauchy(0, 2; lower=0.0)
        b::vector[I] ~ normal(mu_b, sigma_b)
        @plate for i in 1:I, j in 1:J
            y[i, j] ~ bernoulli_logit(a[i] * (theta[j] - b[i]))
        end
    end
end

# Batch 5: hierarchical 2PL IRT + Dogs + regularized horseshoe + structural
# DLM + Prophet.
#
# Gather-eta helpers (long-form IRT linear predictors via concrete-index
# gathers — plain Stan loops, exactly the source's gather structure) and the
# structural-DLM transition densities (custom joint families: transition
# priors are densities over param slices, which have no `~` spelling).
# (@brm refused: gathers, LKJ item hierarchies, slab, recurrences.)
@deffun begin
    irt_eta_2pl(alpha::vector[I], beta::vector[I], theta::vector[J], ii::int[N], jj::int[N])::vector[N] = begin
        out::vector[N]
        for n in 1:N
            out[n] = alpha[ii[n]] * (theta[jj[n]] - beta[ii[n]])
        end
        out
    end
    # Twopl sum-to-zero beta prior: N(0,s) over all I incl. derived beta[I].
    # (A prior on the COMPUTED beta has no `~` spelling — the bare-decl
    # pre-binding lesson — so the free vector carries the full prior mass.)
    @lpxf s2z_beta_lpdf(bf::vector[F], s::real)::real = begin
        rv = normal_lpdf(-sum(bf), 0.0, s)
        for i in 1:F
            rv += normal_lpdf(bf[i], 0.0, s)
        end
        rv
    end
    s2z_beta_lpdfs(bf::vector[F], s::real)::real = s2z_beta_lpdf(bf, s)
    s2z_beta_rng(vector[F], s::real)::vector[F] = begin
        out::vector[F]
        for i in 1:F
            out[i] = normal_rng(0.0, s)
        end
        out
    end
    lat_eta(alpha::vector[I], bf::vector[F], theta::vector[J], ii::int[N], jj::int[N])::vector[N] = begin
        bI = -sum(bf)
        out::vector[N]
        for n in 1:N
            bi = ii[n] == I ? bI : bf[ii[n]]
            out[n] = alpha[ii[n]] * theta[jj[n]] - bi
        end
        out
    end
    @lpxf rw_level_lpdf(mu::vector[n], sigma2::real)::real = begin
        rv = 0.0
        for t in 2:n
            rv += normal_lpdf(mu[t], mu[t - 1], sigma2)
        end
        rv
    end
    rw_level_lpdfs(mu::vector[n], sigma2::real)::real = rw_level_lpdf(mu, sigma2)
    rw_level_rng(vector[n], sigma2::real)::vector[n] = begin
        out::vector[n]
        out[1] = normal_rng(0.0, 1.0)
        for t in 2:n
            out[t] = normal_rng(out[t - 1], sigma2)
        end
        out
    end
    @lpxf seas_window_lpdf(se::vector[n], sigma1::real)::real = begin
        rv = 0.0
        for t in 12:n
            acc = 0.0
            for k in (t - 11):t
                acc += se[k]
            end
            rv += normal_lpdf(acc, 0.0, sigma1)
        end
        rv
    end
    seas_window_lpdfs(se::vector[n], sigma1::real)::real = seas_window_lpdf(se, sigma1)
    seas_window_rng(vector[n], sigma1::real)::vector[n] = begin
        out::vector[n]
        for t in 1:n
            out[t] = normal_rng(0.0, sigma1)
        end
        out
    end
end

# Verbatim port of the twopl source's _obtain_W_adj (data-only; same Julia,
# same ops — including the dead range-branch quirk for J > 1).
function _twopl_w_adj(W::Matrix{Float64})
    J, K = size(W)
    W_adj = similar(W)
    for k in 1:K
        col = @view W[:, k]
        if k == 1
            a1, a2 = 0.0, 1.0
        else
            mn, mx = minimum(col), maximum(col)
            a1 = sum(col) / J
            minmax_count = 0
            for j in 1:J
                minmax_count = (((minmax_count + col[j]) == mn) || (col[j] == mx)) ? 1 : 0
            end
            sd = sqrt(sum(abs2, col .- a1) / (J - 1))
            a2 = minmax_count == J ? (mx - mn) : 2 * sd
        end
        @views @. W_adj[:, k] = (col - a1) / a2
    end
    W_adj
end

# Dogs running-count design in matrix form (same recurrence as _dogs_design).
function _dogs_mats(y::AbstractMatrix)
    J, T = size(y)
    ps = zeros(Float64, J, T)
    pa = zeros(Float64, J, T)
    for j in 1:J, t in 1:T
        t == 1 && continue
        ps[j, t] = ps[j, t - 1] + y[j, t - 1]
        pa[j, t] = pa[j, t - 1] + (1 - y[j, t - 1])
    end
    ps, pa
end

# hier_2pl: LKJ item hierarchy via the EXACT conditional decomposition
# (xi1 ~ N(mu1,tau1); xi2|xi1 normal — same density as the joint
# multi_normal_cholesky up to ~1ulp; array-of-vector cells have no @slic
# spelling); long-form eta via the gather helper.
function hier_2pl_s()
    M = Hier2plExample
    y = Int.(M.HIER_2PL_Y)
    I, J = M.HIER_2PL_I, M.HIER_2PL_J
    return @slic (; y=y, ii=M.HIER_2PL_II, jj=M.HIER_2PL_JJ, I=I, J=J, N=length(y)) begin
        theta::vector[J] ~ normal(0, 1)
        mu1 ~ normal(0, 1)
        mu2 ~ normal(0, 5)
        tau::vector[2] ~ exponential(0.1; lower=0.0)
        L::cholesky_factor_corr[2] ~ lkj_corr_cholesky(4)
        xi1::vector[I] ~ normal(mu1, tau[1])
        mu2c = mu2 + (tau[2] * L[2, 1] / tau[1]) * (xi1 - mu1)
        xi2::vector[I] ~ normal(mu2c, tau[2] * L[2, 2])
        alpha = exp(xi1)
        eta = irt_eta_2pl(alpha, xi2, theta, ii, jj)
        y ~ bernoulli_logit(eta)
    end
end

# twopl_latent_reg: 2PL + latent regression + sum-to-zero beta (custom prior
# family, see above); W_adj host-ported verbatim from the source recipe.
function twopl_latent_reg_s()
    M = TwoplLatentRegIrtExample
    y = Int.(M.TWOPL_LR_Y)
    I = M.TWOPL_LR_I
    Wadj = _twopl_w_adj(M.TWOPL_LR_W)
    J, K = size(Wadj)
    return @slic (; y=y, ii=M.TWOPL_LR_II, jj=M.TWOPL_LR_JJ, Wadj=Wadj,
        I=I, J=J, K=K, N=length(y)) begin
        alpha::vector[I] ~ lognormal(1, 1; lower=0.0)
        beta_free::vector[I - 1] ~ s2z_beta(3.0)
        lambda_adj::vector[K] ~ student_t(3, 0, 1)
        mu_theta = Wadj * lambda_adj
        theta::vector[J] ~ normal(mu_theta, 1)
        eta = lat_eta(alpha, beta_free, theta, ii, jj)
        y ~ bernoulli_logit(eta)
    end
end

# dogs_hierarchical: multiplicative a/b Bernoulli, a/b implicit-uniform
# (flat, Jacobian-only); vectorized probability form.
function dogs_hierarchical_s()
    M = DogsHierarchicalExample
    yf = Int.(M.DOGS_HIER_Y_FLAT)
    return @slic (; y=yf, ps=M.DOGS_HIER_PREV_SHOCK, pa=M.DOGS_HIER_PREV_AVOID) begin
        a ~ flat(; lower=0.0, upper=1.0)
        b ~ flat(; lower=0.0, upper=1.0)
        p = exp(ps * log(a) + pa * log(b))
        y ~ bernoulli(p)
    end
end

# dogs_nonhierarchical: correlated per-dog logit-normal (LKJ cholesky,
# flop-exact per-component form with precomputed sL21/sL22) + multiplicative
# Bernoulli on the 2-D dog/trial grid.
function dogs_nonhierarchical_s()
    M = DogsNonhierarchicalExample
    y = Int.(M.DOGS_NH_Y)
    J, T = size(y)
    ps, pa = _dogs_mats(y)
    return @slic (; y=y, ps=ps, pa=pa, J=J, T=T) begin
        mu::vector[2] ~ logistic(0, 1)
        sg::vector[2] ~ normal(0, 1; lower=0.0)
        L::cholesky_factor_corr[2] ~ lkj_corr_cholesky(2)
        z1::vector[J] ~ normal(0, 1)
        z2::vector[J] ~ normal(0, 1)
        sL21 = sg[2] * L[2, 1]
        sL22 = sg[2] * L[2, 2]
        logit_a = mu[1] + z1 * sg[1] + z2 * sL21
        logit_b = mu[2] + z2 * sL22
        @plate for j in 1:J, t in 1:T
            y[j, t] ~ bernoulli(exp(ps[j, t] * (-log1p_exp(-logit_a[j])) + pa[j, t] * (-log1p_exp(-logit_b[j]))))
        end
    end
end

# logistic_rhs: regularized (Finnish) horseshoe with the c/caux slab
# (@brm refused: Horseshoe has no slab spelling); half-t priors PLAIN.
function logistic_rhs_s()
    M = LogisticRegressionRHSExample
    H = M.LOGISTIC_RHS_HYPER
    y = Int.(M.LOGISTIC_RHS_Y)
    d = size(M.LOGISTIC_RHS_X, 2)
    return @slic (; x=M.LOGISTIC_RHS_X, y=y, d=d, scale_icept=H.scale_icept,
        scale_global=H.scale_global, nu_global=H.nu_global, nu_local=H.nu_local,
        slab_scale=H.slab_scale, slab_df=H.slab_df) begin
        beta0 ~ normal(0, scale_icept)
        z::vector[d] ~ normal(0, 1)
        tau ~ student_t(nu_global, 0, scale_global * 2; lower=0.0)
        lambda::vector[d] ~ student_t(nu_local, 0, 1; lower=0.0)
        caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df; lower=0.0)
        c = slab_scale * sqrt(caux)
        c2 = c * c
        tau2 = tau * tau
        lambda2 = lambda .^ 2
        lambda_tilde = sqrt(c2 * lambda2 ./ (c2 .+ tau2 * lambda2))
        beta = (z .* lambda_tilde) * tau
        f = beta0 + x * beta
        y ~ bernoulli_logit(f)
    end
end

# state_space: UK-drivers structural DLM. Level ~ rw_level (custom joint
# family WITH data-directed decl bounds — kwargs compose with families),
# seasonal ~ seas_window, beta/lambda flat, sigma positive_ordered ~ t(4).
# (lobnd/upbnd: `lower`/`upper` are Stan reserved words.)
function state_space_s()
    M = StateSpaceStochasticExample
    y, x, w = M.STATE_SPACE_Y, M.STATE_SPACE_X, M.STATE_SPACE_W
    n = length(y)
    ybar = sum(y) / n
    ysd = sqrt(sum((y .- ybar) .^ 2) / (n - 1))
    lobnd, upbnd = ybar - 3 * ysd, ybar + 3 * ysd
    return @slic (; y=y, x=x, w=w, n=n, lobnd=lobnd, upbnd=upbnd) begin
        sg::positive_ordered[3] ~ student_t(4, 0, 1)
        mu::vector[n] ~ rw_level(sg[2]; lower=lobnd, upper=upbnd)
        seasonal::vector[n] ~ seas_window(sg[1])
        beta ~ flat()
        lambda ~ flat()
        yhat = mu + beta * x + lambda * w
        y ~ normal(yhat + seasonal, sg[3])
    end
end

# prophet: piecewise trend x multiplicative + additive seasonality (LINEAR
# mode only — the data selects trend_indicator == 0; logistic is a
# separately-scoped deliverable on the RK side too). Laplace is Stan's
# double_exponential.
function prophet_s()
    M = ProphetExample
    t, t_change = M.PROPHET_T, M.PROPHET_T_CHANGE
    A = (t .>= t_change') .* 1.0
    @assert M.PROPHET_TREND_INDICATOR == 0 "logistic trend out of scope"
    return @slic (; y=M.PROPHET_Y, t=t, A=A, t_change=t_change, X=M.PROPHET_X,
        sigmas=M.PROPHET_SIGMAS, tau=M.PROPHET_TAU, s_a=M.PROPHET_S_A,
        s_m=M.PROPHET_S_M, S=length(t_change), K=size(M.PROPHET_X, 2)) begin
        k ~ normal(0, 5)
        m ~ normal(0, 5)
        delta::vector[S] ~ double_exponential(0, tau)
        sigma_obs ~ normal(0, 0.5; lower=0.0)
        beta::vector[K] ~ normal(0, sigmas)
        Ad = A * delta
        td = t_change .* delta
        trend = (k + Ad) .* t + (m - A * td)
        seasonal_mult = 1.0 + X * (beta .* s_m)
        seasonal_add = X * (beta .* s_a)
        mean_response = trend .* seasonal_mult + seasonal_add
        y ~ normal(mean_response, sigma_obs)
    end
end

# Batch 6: marginalized/exact GPs + brms HSGP/splines + BYM2 spatial Poisson.
#
# GP covariance helpers (loop-form exp-quad + nugget/jitter, then Cholesky).
# (@brm refused: latent-`gp` only, no marginalized/GP-algebra surface; note
# StanBlocks' `gp_exp_quad_cov` call returns an UNTYPED matrix, so even the
# simple case spells the kernel out — same formula, ~1ulp vs BLAS blocking.)
@deffun begin
    gp_regr_chol(x::vector[N], alpha::real, rho::real, sigma::real)::matrix[N,N] = begin
        K::matrix[N,N]
        for i in 1:N
            for j in 1:N
                d = x[i] - x[j]
                sq = d * d
                K[i, j] = alpha * alpha * exp(-0.5 * sq / (rho * rho)) + (i == j ? sigma : 0.0)
            end
        end
        cholesky_decompose(K)
    end
    hgp_cov(sq::matrix[Y,Y], s_long::real, l_long::real, s_short::real, l_short::real)::matrix[Y,Y] = begin
        out::matrix[Y,Y]
        for i in 1:Y
            for j in 1:Y
                out[i, j] = s_long * s_long * exp(-0.5 * sq[i, j] / (l_long * l_long)) +
                    s_short * s_short * exp(-0.5 * sq[i, j] / (l_short * l_short)) +
                    (i == j ? 1e-6 : 0.0)
            end
        end
        out
    end
    hgp_gpmat(cov::matrix[Y,Y], std::matrix[Y,K])::matrix[Y,K] = begin
        cholesky_decompose(cov) * std
    end
    hgp_state_re(vars::vector[V], sr_ind::int[St], state_std::vector[St])::vector[St] = begin
        out::vector[St]
        for i in 1:St
            out[i] = sqrt(vars[2 + sr_ind[i]]) * state_std[i]
        end
        out
    end
    hgp_obs_mu(mu::real, year_re::vector[Yo], state_re::vector[St], region_re::vector[Rg], GPr::matrix[Y,Rg], GPs::matrix[Y,St], yi::int[N], si::int[N], ri::int[N])::vector[N] = begin
        out::vector[N]
        for n in 1:N
            out[n] = mu + year_re[yi[n]] + state_re[si[n]] + region_re[ri[n]] +
                GPr[yi[n], ri[n]] + GPs[yi[n], si[n]]
        end
        out
    end
    # Kronecker eigenspace marginal (whiten -> scale -> rotate back ->
    # contract; column-major accumulation to match the source; the Julia-eigen
    # vs Stan-eigen decompositions differ ~1ulp internally — inside any parity
    # tolerance, noted here). The _rng is a GQ-only zero draw (unused).
    @lpxf kron_marg_lpdf(yf::vector[V], x1::vector[n1], var1::real, bw1::real, sigma1::real, L::cholesky_factor_corr[n2])::real = begin
        Y::matrix[n2,n1]
        for j in 1:n1
            for i in 1:n2
                Y[i, j] = yf[(j - 1) * n2 + i]
            end
        end
        S1::matrix[n1,n1]
        for i in 1:n1
            for j in 1:n1
                d = x1[i] - x1[j]
                xd = -(d * d)
                S1[i, j] = var1 * exp(xd * bw1) + (i == j ? 1e-5 : 0.0)
            end
        end
        Q1 = eigenvectors_sym(S1)
        R1 = eigenvalues_sym(S1)
        Lam = L * transpose(L)
        Q2 = eigenvectors_sym(Lam)
        R2 = eigenvalues_sym(Lam)
        E::matrix[n2,n1]
        for j in 1:n1
            for i in 1:n2
                E[i, j] = R2[i] * R1[j] + sigma1
            end
        end
        W1 = transpose(Q2) * Y
        W2 = transpose(Q1) * transpose(W1)
        Wh = transpose(W2)
        Sc::matrix[n2,n1]
        for j in 1:n1
            for i in 1:n2
                Sc[i, j] = Wh[i, j] / E[i, j]
            end
        end
        Rt1 = Q2 * Sc
        Rt2 = Q1 * transpose(Rt1)
        Rt = transpose(Rt2)
        acc = 0.0
        lacc = 0.0
        for j in 1:n1
            for i in 1:n2
                acc += Y[i, j] * Rt[i, j]
                lacc += log(E[i, j])
            end
        end
        -0.5 * acc - 0.5 * lacc
    end
    kron_marg_lpdfs(yf::vector[V], x1::vector[n1], var1::real, bw1::real, sigma1::real, L::cholesky_factor_corr[n2])::real =
        kron_marg_lpdf(yf, x1, var1, bw1, sigma1, L)
    kron_marg_rng(vector[V], x1::vector[n1], var1::real, bw1::real, sigma1::real, L::cholesky_factor_corr[n2])::vector[V] =
        rep_vector(0.0, V)
    # brms 1-D HSGP contribution: gp = Xgp * (sqrt(spd) .* zgp) with the
    # exp-quad spectral density at the Laplacian sqrt-eigenvalues. (`pi` is
    # StanBlocks' bare constant — `pi()` is rejected.)
    accel_contrib(Xgp::matrix[N,NB], slambda::vector[NB], sdgp::real, lscale::real, zgp::vector[NB])::vector[N] = begin
        w::vector[NB]
        for m in 1:NB
            spd = sdgp * sdgp * sqrt(2 * pi) * lscale * exp(-0.5 * lscale * lscale * slambda[m] * slambda[m])
            w[m] = sqrt(spd) * zgp[m]
        end
        out::vector[N]
        for i in 1:N
            acc = 0.0
            for m in 1:NB
                acc += Xgp[i, m] * w[m]
            end
            out[i] = acc
        end
        out
    end
    # BYM2: ICAR pairwise-difference prior + soft sum-to-zero over phi.
    @lpxf icar_s2z_lpdf(phi::vector[N], node1::int[E], node2::int[E], s2z_scale::real)::real = begin
        acc = 0.0
        for e in 1:E
            d = phi[node1[e]] - phi[node2[e]]
            acc += d * d
        end
        -0.5 * acc + normal_lpdf(sum(phi), 0.0, s2z_scale)
    end
    icar_s2z_lpdfs(phi::vector[N], node1::int[E], node2::int[E], s2z_scale::real)::real =
        icar_s2z_lpdf(phi, node1, node2, s2z_scale)
    icar_s2z_rng(vector[N], node1::int[E], node2::int[E], s2z_scale::real)::vector[N] = begin
        out::vector[N]
        for i in 1:N
            out[i] = normal_rng(0.0, 1.0)
        end
        out
    end
end

# gp_regr: marginalized exact GP (exp-quad + sigma nugget — sigma, NOT
# sigma^2), Gamma/Normal/Normal hyperpriors PLAIN.
function gp_regr_s()
    M = GPRegrExample
    y = M.GP_REGR_Y
    return @slic (; x=M.GP_REGR_X, y=y, N=length(y)) begin
        rho ~ gamma(25, 4; lower=0.0)
        alpha ~ normal(0, 2; lower=0.0)
        sigma ~ normal(0, 1; lower=0.0)
        L = gp_regr_chol(x, alpha, rho, sigma)
        y ~ multi_normal_cholesky(rep_vector(0.0, N), L)
    end
end

# hierarchical_gp: per-year region/state latent GPs (long+short exp-quad,
# Cholesky-noncentered) + REs + Dirichlet variance split. GP coefficient
# matrices sampled via 2-D @plate (Stan-native matrix params).
function hierarchical_gp_s()
    M = HierarchicalGPExample
    Y, Rg, St, Yo = M.HGP_N_YEARS, M.HGP_N_REGIONS, M.HGP_N_STATES, M.HGP_N_YEARS_OBS
    yrs = collect(1.0:Y)
    sq = [(yrs[i] - yrs[j])^2 for i in 1:Y, j in 1:Y]
    return @slic (; y=M.HGP_Y, year_ind=M.HGP_YEAR_IND, state_ind=M.HGP_STATE_IND,
        region_ind=M.HGP_REGION_IND, state_region_ind=M.HGP_STATE_REGION_IND,
        sq=sq, Y=Y, Rg=Rg, St=St, Yo=Yo, N=length(M.HGP_Y)) begin
        @plate for i in 1:Y, r in 1:Rg
            GP_region_std[i, r] ~ normal(0, 1)
        end
        @plate for i in 1:Y, s in 1:St
            GP_state_std[i, s] ~ normal(0, 1)
        end
        year_std::vector[Yo] ~ normal(0, 1)
        state_std::vector[St] ~ normal(0, 1)
        region_std::vector[Rg] ~ normal(0, 1)
        tot_var ~ gamma(3, 3; lower=0.0)
        prop_var::simplex[17] ~ dirichlet(rep_vector(2.0, 17))
        mu ~ normal(0.5, 0.5)
        len_rl ~ weibull(30, 8; lower=0.0)
        len_sl ~ weibull(30, 8; lower=0.0)
        len_rs ~ weibull(30, 3; lower=0.0)
        len_ss ~ weibull(30, 3; lower=0.0)
        vars = 17.0 * tot_var * prop_var
        year_re = sqrt(vars[1]) * year_std
        region_re = sqrt(vars[2]) * region_std
        state_re = hgp_state_re(vars, state_region_ind, state_std)
        cov_region = hgp_cov(sq, sqrt(vars[13]), len_rl, sqrt(vars[15]), len_rs)
        cov_state = hgp_cov(sq, sqrt(vars[14]), len_sl, sqrt(vars[16]), len_ss)
        GP_region = hgp_gpmat(cov_region, GP_region_std)
        GP_state = hgp_gpmat(cov_state, GP_state_std)
        obs_mu = hgp_obs_mu(mu, year_re, state_re, region_re, GP_region, GP_state,
            year_ind, state_ind, region_ind)
        y ~ normal(obs_mu, sqrt(vars[17]))
    end
end

# kronecker_gp: RBF x LKJ-corr Kronecker marginal via both margins'
# eigendecompositions (see family note on eigen parity).
function kronecker_gp_s()
    M = KroneckerGpExample
    yf = vec(M.KRON_Y)
    n1 = length(M.KRON_X1)
    n2 = size(M.KRON_Y, 1)
    @assert size(M.KRON_Y, 2) == n1
    return @slic (; yf=yf, x1=M.KRON_X1, n2=n2) begin
        var1 ~ lognormal(0, 1; lower=0.0)
        bw1 ~ cauchy(0, 2.5; lower=0.0)
        sigma1 ~ lognormal(0, 1; lower=1e-5)
        L::cholesky_factor_corr[n2] ~ lkj_corr_cholesky(2)
        yf ~ kron_marg(x1, var1, bw1, sigma1, L)
    end
end

# accel_gp: brms HSGP on mu + log-sigma; sdgp half-Student-t NORMALIZED
# (truncated form: lpdf - log(0.5)). prior_only must be 0.
function accel_gp_s()
    M = AccelGPExample
    @assert !M.ACCEL_GP_PRIOR_ONLY
    y = M.ACCEL_GP_Y
    nb1 = size(M.ACCEL_GP_XGP, 2)
    nbs = size(M.ACCEL_GP_XGP_SIGMA, 2)
    return @slic (; Y=y, Xgp_1=M.ACCEL_GP_XGP, slambda_1=M.ACCEL_GP_SLAMBDA,
        Xgp_sigma_1=M.ACCEL_GP_XGP_SIGMA, slambda_sigma_1=M.ACCEL_GP_SLAMBDA_SIGMA,
        nb1=nb1, nbs=nbs) begin
        intercept ~ student_t(3, -13, 36)
        sdgp_1 ~ truncated(student_t, 3, 0, 36; lower=0.0)
        lscale_1 ~ inv_gamma(1.124909, 0.0177; lower=0.0)
        zgp_1::vector[nb1] ~ normal(0, 1)
        intercept_sigma ~ student_t(3, 0, 10)
        sdgp_s ~ truncated(student_t, 3, 0, 36; lower=0.0)
        lscale_s ~ inv_gamma(1.124909, 0.0177; lower=0.0)
        zgp_s::vector[nbs] ~ normal(0, 1)
        gp_mu = accel_contrib(Xgp_1, slambda_1, sdgp_1, lscale_1, zgp_1)
        mu = intercept + gp_mu
        gp_logsigma = accel_contrib(Xgp_sigma_1, slambda_sigma_1, sdgp_s, lscale_s, zgp_s)
        sigma = exp(intercept_sigma + gp_logsigma)
        Y ~ normal(mu, sigma)
    end
end

# accel_splines: brms penalized splines on mu + log-sigma; sds half-t
# NORMALIZED, linear effects flat.
function accel_splines_s()
    M = AccelSplinesExample
    @assert !M.ACCEL_PRIOR_ONLY
    Ks = size(M.ACCEL_XS, 2)
    knots_1 = size(M.ACCEL_ZS_1_1, 2)
    Ks_sigma = size(M.ACCEL_XS_SIGMA, 2)
    knots_sigma_1 = size(M.ACCEL_ZS_SIGMA_1_1, 2)
    return @slic (; Y=M.ACCEL_Y, Xs=M.ACCEL_XS, Zs_1_1=M.ACCEL_ZS_1_1,
        Xs_sigma=M.ACCEL_XS_SIGMA, Zs_sigma_1_1=M.ACCEL_ZS_SIGMA_1_1,
        Ks=Ks, knots_1=knots_1, Ks_sigma=Ks_sigma, knots_sigma_1=knots_sigma_1) begin
        Intercept ~ student_t(3, -13, 36)
        bs::vector[Ks] ~ flat()
        zs_1_1::vector[knots_1] ~ normal(0, 1)
        sds_1_1 ~ truncated(student_t, 3, 0, 36; lower=0.0)
        Intercept_sigma ~ student_t(3, 0, 10)
        bs_sigma::vector[Ks_sigma] ~ flat()
        zs_sigma_1_1::vector[knots_sigma_1] ~ normal(0, 1)
        sds_sigma_1_1 ~ truncated(student_t, 3, 0, 36; lower=0.0)
        s_1_1 = sds_1_1 * zs_1_1
        s_sigma_1_1 = sds_sigma_1_1 * zs_sigma_1_1
        mu = Intercept + Xs * bs + Zs_1_1 * s_1_1
        sigma = exp(Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * s_sigma_1_1)
        Y ~ normal(mu, sigma)
    end
end

# bym2: Morris/Riebler BYM2 spatial Poisson; ICAR + soft s2z via the custom
# joint family (densities over param gathers have no `~` spelling); rho ~
# Beta(0.5,0.5), sigma half-normal PLAIN.
function bym2_s()
    M = Bym2OffsetOnlyExample
    @assert all(M.BYM2_E .> 0)
    N = length(M.BYM2_Y)
    return @slic (; y=M.BYM2_Y, node1=M.BYM2_NODE1, node2=M.BYM2_NODE2,
        log_E=log.(M.BYM2_E), scaling_factor=M.BYM2_SCALING_FACTOR,
        s2z_scale=0.001 * N, N=N) begin
        beta0 ~ normal(0, 1)
        sigma ~ normal(0, 1; lower=0.0)
        rho ~ beta(0.5, 0.5; lower=0.0, upper=1.0)
        theta::vector[N] ~ normal(0, 1)
        phi::vector[N] ~ icar_s2z(node1, node2, s2z_scale)
        convolved = sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi
        eta = log_E + beta0 + convolved * sigma
        y ~ poisson_log(eta)
    end
end

# Batch 7: time series + actuarial + occupancy + renewal + MNIST.
#
# Recurrence/marginal families (the scan/recurrence bodies are plain-Stan
# loops inside @deffun — @slic model bodies admit no `for`; @deffun ranges
# must ASCEND, hence the index-arithmetic buffer shift in covid_edmat).
# (@brm refused throughout: recurrences, augmentation marginals, renewal
# scans, reference-logit multinomials.)
@deffun begin
    # ARMA(1,1): exact error recursion from the fixed (mu, 0) seed.
    @lpxf arma11_lpdf(y::vector[T], mu::real, phi::real, theta::real, sigma::real)::real = begin
        acc = 0.0
        y_prev = mu
        err_prev = 0.0
        for t in 1:T
            nu = mu + phi * y_prev + theta * err_prev
            e = y[t] - nu
            acc += normal_lpdf(e, 0.0, sigma)
            y_prev = y[t]
            err_prev = e
        end
        acc
    end
    arma11_lpdfs(y::vector[T], mu::real, phi::real, theta::real, sigma::real)::real =
        arma11_lpdf(y, mu, phi, theta, sigma)
    arma11_rng(vector[T], mu::real, phi::real, theta::real, sigma::real)::vector[T] = begin
        out::vector[T]
        y_prev = mu
        err_prev = 0.0
        for t in 1:T
            nu = mu + phi * y_prev + theta * err_prev
            e = normal_rng(0.0, sigma)
            out[t] = nu + e
            y_prev = out[t]
            err_prev = e
        end
        out
    end
    # GARCH(1,1) conditional-sd path (explicit squares — no `^` in @deffun).
    garch_sigma(y::vector[T], sigma1::real, mu::real, alpha0::real, alpha1::real, beta1::real)::vector[T] = begin
        out::vector[T]
        out[1] = sigma1
        sprev = sigma1
        for t in 2:T
            dm = y[t - 1] - mu
            st = sqrt(alpha0 + alpha1 * dm * dm + beta1 * sprev * sprev)
            out[t] = st
            sprev = st
        end
        out
    end
    # SiS-lob growth curve, both closed forms live behind the data flag.
    loss_gf(t_value::vector[T], omega::real, theta::real, gid::int)::vector[T] = begin
        out::vector[T]
        for i in 1:T
            t = t_value[i]
            if (gid == 1)
                out[i] = 1.0 - exp(-((t / theta)^omega))
            else
                out[i] = exp(-log1p_exp(omega * log(theta / t)))
            end
        end
        out
    end
    loss_lm(LR::vector[C], gf::vector[T], prem::vector[D], cohort_id::int[D], t_idx::int[D])::vector[D] = begin
        out::vector[D]
        for i in 1:D
            out[i] = LR[cohort_id[i]] * prem[i] * gf[t_idx[i]]
        end
        out
    end
    # Multi-occupancy: the (rho+1)/2 ~ Beta(2,2) density on a [-1,1] param
    # (kwargs bounds compose with custom families).
    @lpxf rho_beta2_lpdf(rho::real)::real = beta_lpdf((rho + 1.0) / 2.0, 2.0, 2.0)
    rho_beta2_lpdfs(rho::real)::real = rho_beta2_lpdf(rho)
    rho_beta2_rng()::real = 2.0 * beta_rng(2.0, 2.0) - 1.0
    # Multi-occupancy joint marginal: detected/undetected site terms over the
    # n x J grid + the never-detected augmentation tail (declared temps are
    # function-scoped — a second loop cannot reuse first-loop Stan locals).
    @lpxf occ_marg_lpmf(Xflat::int[M], spec::int[M], uv1::vector[S], uv2::vector[S], alpha::real, beta_::real, Omega::real, K::int, J::int, n::int)::real = begin
        acc = n * log(Omega)
        psi = 0.0
        theta = 0.0
        for m in 1:M
            psi = uv1[spec[m]] + alpha
            theta = uv2[spec[m]] + beta_
            x = Xflat[m]
            lp_obs = log_inv_logit(psi) + binomial_logit_lpmf(x, K, theta)
            lp_unobs = log_sum_exp(log_inv_logit(psi) + K * log_inv_logit(-theta), log_inv_logit(-psi))
            if (x > 0)
                acc += lp_obs
            else
                acc += lp_unobs
            end
        end
        lo = log(Omega)
        l1o = log1m(Omega)
        S = length(uv1)
        for i in (n + 1):S
            psi = uv1[i] + alpha
            theta = uv2[i] + beta_
            lu = log_sum_exp(log_inv_logit(psi) + K * log_inv_logit(-theta), log_inv_logit(-psi))
            acc += log_sum_exp(l1o, lo + J * lu)
        end
        acc
    end
    occ_marg_lpmfs(Xflat::int[M], spec::int[M], uv1::vector[S], uv2::vector[S], alpha::real, beta_::real, Omega::real, K::int, J::int, n::int)::real =
        occ_marg_lpmf(Xflat, spec, uv1, uv2, alpha, beta_, Omega, K, J, n)
    occ_marg_rng(int[M], spec::int[M], uv1::vector[S], uv2::vector[S], alpha::real, beta_::real, Omega::real, K::int, J::int, n::int)::int[M] = begin
        out::int[M]
        for m in 1:M
            out[m] = binomial_rng(K, 0.5)
        end
        out
    end
    # Covid E_deaths grid: day-1 special case + closed-form imputation days +
    # renewal scan with an explicit shift-register buffer.
    covid_edmat(Rt::matrix[N2,M], SI::vector[N2], fmat::matrix[N2,M], pop::vector[M], y::vector[M], ifr::vector[M], M::int, N0::int, N2::int)::matrix[N2,M] = begin
        ed::matrix[N2,M]
        for m in 1:M
            ed[1, m] = 1e-15 * y[m]
            fp = 0.0
            for i in 2:N0
                fp += fmat[i - 1, m]
                ed[i, m] = ifr[m] * y[m] * fp
            end
        end
        buf::matrix[N2,M]
        for m in 1:M
            for d in 1:N2
                if (d <= N0)
                    buf[d, m] = y[m]
                else
                    buf[d, m] = 0.0
                end
            end
        end
        cum::vector[M]
        for m in 1:M
            cum[m] = (N0 - 1) * y[m]
        end
        for i in (N0 + 1):N2
            for m in 1:M
                conv = 0.0
                conv_f = 0.0
                for d in 1:N2
                    conv += buf[d, m] * SI[d]
                    conv_f += buf[d, m] * fmat[d, m]
                end
                cum[m] += buf[1, m]
                susc = (pop[m] - cum[m]) / pop[m]
                pred = susc * Rt[i, m] * conv
                ed[i, m] = ifr[m] * conv_f
                for k in 1:(N2 - 1)
                    buf[N2 - k + 1, m] = buf[N2 - k, m]
                end
                buf[1, m] = pred
            end
        end
        ed
    end
    # Covid joint marginal: hierarchical Rt + E_deaths grid + the VERBATIM
    # neg_binomial_2 expansion (same terms, same masked accumulation order).
    @lpxf covid_marg_lpmf(dflat::int[G], XX::matrix[G,P], ES::int[M], Ns::int[M], SI::vector[N2], fmat::matrix[N2,M], pop::vector[M], M::int, P::int, N0::int, N2::int, mu::vector[M], alpha_hier::vector[P], kappa::real, y::vector[M], phi::real, tau::real, ifr::vector[M])::real = begin
        alpha::vector[P]
        for p in 1:P
            alpha[p] = alpha_hier[p] - log(1.05) / 6.0
        end
        Lv = XX * alpha
        L = to_matrix(Lv, N2, M)
        Rt::matrix[N2,M]
        for m in 1:M
            for i in 1:N2
                Rt[i, m] = mu[m] * exp(-L[i, m])
            end
        end
        ed = covid_edmat(Rt, SI, fmat, pop, y, ifr, M, N0, N2)
        acc = 0.0
        lg_phi = lgamma(phi)
        phi_log_phi = phi * log(phi)
        for m in 1:M
            for i in ES[m]:Ns[m]
                d = dflat[(m - 1) * N2 + i]
                e = ed[i, m]
                lpd = log(phi + e)
                acc += lgamma(d + phi) - lg_phi - lgamma(d + 1) + phi_log_phi - phi * lpd - d * lpd
                if (d > 0)
                    acc += d * log(e)
                end
            end
        end
        acc
    end
    covid_marg_lpmfs(dflat::int[G], XX::matrix[G,P], ES::int[M], Ns::int[M], SI::vector[N2], fmat::matrix[N2,M], pop::vector[M], M::int, P::int, N0::int, N2::int, mu::vector[M], alpha_hier::vector[P], kappa::real, y::vector[M], phi::real, tau::real, ifr::vector[M])::real =
        covid_marg_lpmf(dflat, XX, ES, Ns, SI, fmat, pop, M, P, N0, N2, mu, alpha_hier, kappa, y, phi, tau, ifr)
    covid_marg_rng(int[G], XX::matrix[G,P], ES::int[M], Ns::int[M], SI::vector[N2], fmat::matrix[N2,M], pop::vector[M], M::int, P::int, N0::int, N2::int, mu::vector[M], alpha_hier::vector[P], kappa::real, y::vector[M], phi::real, tau::real, ifr::vector[M])::int[G] =
        rep_array(0, G)
    # MNIST reference-logit multinomial (explicit loops; ~1ulp vs BLAS).
    @lpxf mnist_cat_lpmf(y::int[N], W::matrix[Cm,F], Xt::matrix[F,N], b::vector[Cm], C::int)::real = begin
        acc = 0.0
        for n in 1:N
            lg::vector[C]
            lg[1] = 0.0
            for i in 1:Cm
                e = b[i]
                for f in 1:F
                    e += W[i, f] * Xt[f, n]
                end
                lg[i + 1] = e
            end
            acc += categorical_logit_lpmf(y[n], lg)
        end
        acc
    end
    mnist_cat_lpmfs(y::int[N], W::matrix[Cm,F], Xt::matrix[F,N], b::vector[Cm], C::int)::real =
        mnist_cat_lpmf(y, W, Xt, b, C)
    mnist_cat_rng(int[N], W::matrix[Cm,F], Xt::matrix[F,N], b::vector[Cm], C::int)::int[N] = begin
        out::int[N]
        for n in 1:N
            out[n] = categorical_rng(rep_vector(1.0 / C, C))
        end
        out
    end
end

# arma11: Gaussian ARMA(1,1) with the exact error recursion; sigma
# half-Cauchy NORMALIZED (truncated form carries the +log 2).
function arma11_s()
    M = ARMA11Example
    y = M.ARMA_SERIES
    return @slic (; y=y) begin
        mu ~ normal(0, 10)
        phi ~ normal(0, 2)
        theta ~ normal(0, 2)
        sigma ~ truncated(cauchy, 0, 2.5; lower=0.0)
        y ~ arma11(mu, phi, theta, sigma)
    end
end

# garch11: flat mu/alpha0/alpha1 + beta1 on [0, 1-alpha1] (param-dependent
# decl bounds ARE supported); deterministic sd path, native observation.
function garch11_s()
    M = GARCH11Example
    return @slic (; y=M.GARCH11_Y, sigma1=M.GARCH11_SIGMA1) begin
        mu ~ flat()
        alpha0 ~ flat(; lower=0.0)
        alpha1 ~ flat(; lower=0.0, upper=1.0)
        beta1 ~ flat(; lower=0.0, upper=1 - alpha1)
        sg = garch_sigma(y, sigma1, mu, alpha0, alpha1, beta1)
        y ~ normal(mu, sg)
    end
end

# losscurve: SiS-lob triangle; LR/mu_LR/sd_LR lognormal hierarchy, both
# growth forms live behind the data flag, premium gathered host-side.
function losscurve_s()
    M = LosscurveSislobExample
    prem_datum = M.LOSSCURVE_PREMIUM[M.LOSSCURVE_COHORT_ID]
    return @slic (; gid=M.LOSSCURVE_GROWTHMODEL_ID, cohort_id=M.LOSSCURVE_COHORT_ID,
        t_idx=M.LOSSCURVE_T_IDX, t_value=M.LOSSCURVE_T_VALUE,
        prem_datum=prem_datum, loss=M.LOSSCURVE_LOSS, C=length(M.LOSSCURVE_PREMIUM)) begin
        omega ~ lognormal(0, 0.5; lower=0.0)
        theta ~ lognormal(0, 0.5; lower=0.0)
        mu_LR ~ normal(0, 0.5)
        sd_LR ~ lognormal(0, 0.5; lower=0.0)
        LR::vector[C] ~ lognormal(mu_LR, sd_LR)
        loss_sd ~ lognormal(0, 0.7; lower=0.0)
        gf = loss_gf(t_value, omega, theta, gid)
        lm = loss_lm(LR, gf, prem_datum, cohort_id, t_idx)
        loss ~ normal(lm, loss_sd * prem_datum)
    end
end

# multi_occupancy: Dorazio-Royle multispecies occupancy. Correlated (uv1,uv2)
# via the EXACT conditional decomposition (hier_2pl precedent); Beta(2,2) on
# (rho+1)/2 as a custom density on the [-1,1] param; full joint marginal.
function multi_occupancy_s()
    M = MultiOccupancyExample
    Xflat = vec(M.MULTI_OCC_X)
    spec = repeat(1:M.MULTI_OCC_N, M.MULTI_OCC_J)
    S = M.MULTI_OCC_S
    return @slic (; Xflat=Xflat, spec=spec, K=M.MULTI_OCC_K, J=M.MULTI_OCC_J,
        n=M.MULTI_OCC_N, S=S) begin
        alpha ~ cauchy(0, 2.5)
        beta_ ~ cauchy(0, 2.5)
        s1 ~ cauchy(0, 2.5; lower=0.0)
        s2 ~ cauchy(0, 2.5; lower=0.0)
        rho ~ rho_beta2(; lower=-1.0, upper=1.0)
        Omega ~ beta(2, 2; lower=0.0, upper=1.0)
        uv2::vector[S] ~ normal(0, s2)
        uv1::vector[S] ~ normal(rho * s1 / s2 * uv2, s1 * sqrt(1 - rho * rho))
        Xflat ~ occ_marg(spec, uv1, uv2, alpha, beta_, Omega, K, J, n)
    end
end

# covid19: Imperial renewal model. The scan is a plain-Stan shift-register
# loop; the NB2 is the verbatim expansion; X arrives preprocessed host-side
# (3-D arrays have no @slic spelling — same Julia ops, same bytes).
function covid19_s()
    M = Covid19ImperialExample
    XX = reshape(permutedims(M.COVID19IMPERIAL_X, (2, 1, 3)),
        M.COVID19IMPERIAL_N2 * M.COVID19IMPERIAL_M, M.COVID19IMPERIAL_P)
    dflat = vec(M.COVID19IMPERIAL_DEATHS)
    return @slic (; dflat=dflat, XX=XX, ES=M.COVID19IMPERIAL_EPIDEMICSTART,
        Ns=M.COVID19IMPERIAL_N, SI=M.COVID19IMPERIAL_SI, fmat=M.COVID19IMPERIAL_F,
        pop=M.COVID19IMPERIAL_POP, M=M.COVID19IMPERIAL_M, P=M.COVID19IMPERIAL_P,
        N0=M.COVID19IMPERIAL_N0, N2=M.COVID19IMPERIAL_N2) begin
        kappa ~ normal(0, 0.5; lower=0.0)
        tau ~ exponential(0.03; lower=0.0)
        mu::vector[M] ~ normal(3.28, kappa)
        alpha_hier::vector[P] ~ gamma(0.1667, 1; lower=0.0)
        y::vector[M] ~ exponential(1 / tau; lower=0.0)
        phi ~ normal(0, 5; lower=0.0)
        ifr::vector[M] ~ normal(1.0, 0.1; lower=0.0)
        dflat ~ covid_marg(XX, ES, Ns, SI, fmat, pop, M, P, N0, N2, mu,
            alpha_hier, kappa, y, phi, tau, ifr)
    end
end

# mnist_logistic: reference-logit softmax on the 8-image fixture; W sampled
# via 2-D @plate (Stan-native matrix, column-major q order matches the
# source's packed layout); X transposed host-side.
function mnist_logistic_s()
    M = MNISTLogisticExample
    Xt = Matrix(transpose(M.MNIST_LOGISTIC_X))
    F = size(Xt, 1)
    Cm = M.NUM_CLASSES - 1
    return @slic (; y=M.MNIST_LOGISTIC_Y, Xt=Xt, F=F, Cm=Cm, C=M.NUM_CLASSES,
        N=length(M.MNIST_LOGISTIC_Y)) begin
        @plate for i in 1:Cm, f in 1:F
            W[i, f] ~ normal(0, 1)
        end
        b::vector[Cm] ~ normal(0, 1)
        y ~ mnist_cat(W, Xt, b, C)
    end
end

# Batch 8: the one portable model among RK examples/ (6) — brm_hsgp (the
# other five are sampler/stats/AD infrastructure with no posterior, hence
# documented no-counterparts: manual_derivative_rule, nutpie adaptation,
# nuts workflow + runtime, online_stats).
#
# HSGP weight family: the z-prior (weights pulled back through the
# centering) plus the centering Jacobian live on the fresh-sampled v, since
# densities over the computed z have no `~` spelling.
@deffun begin
    @lpxf hsgp_v_lpdf(v::vector[K], ls::vector[K], c::vector[K])::real = begin
        rv = 0.0
        for k in 1:K
            z = v[k] * exp(-c[k] * ls[k])
            rv += normal_lpdf(z, 0.0, 1.0) - c[k] * ls[k]
        end
        rv
    end
    hsgp_v_lpdfs(v::vector[K], ls::vector[K], c::vector[K])::real =
        hsgp_v_lpdf(v, ls, c)
    hsgp_v_rng(vector[K], ls::vector[K], c::vector[K])::vector[K] = begin
        out::vector[K]
        for k in 1:K
            out[k] = normal_rng(0.0, 1.0)
        end
        out
    end
end

# Motorcycle data prep mirroring BRMHSGPExample.motorcycle_data exactly
# (same bytes: sha-pinned CSV under the committed RK pin).
function _mcycle_data()
    path = joinpath(@__DIR__, ".bootstrap", "reactivekernels-bb2a2e0fe058",
        "examples", "data", "mcycle.csv")
    bytes2hex(sha256(read(path))) ==
        "b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801" ||
        error("mcycle data hash mismatch")
    lines = readlines(path)
    first(lines) == "rownames,times,accel" || error("unexpected mcycle header")
    rows = split.(lines[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    length(times) == 133 || error("expected 133 observations")
    lo, hi = extrema(times)
    x = @. -1 + 2 * (times - lo) / (hi - lo)
    y = accel ./ std(accel)
    modes = collect(1.0:20.0)
    half_width = 1.5
    freq = modes .* (pi / (2 * half_width))
    basis = sin.((x .+ half_width) * transpose(freq)) ./ sqrt(half_width)
    (; y=y, basis=basis, fsq=freq .^ 2)
end

# brm_hsgp: heteroscedastic HSGP with online-selected centeredness, at a
# FIXED centering (the source's live HAVE c rides as data; the two endpoint
# cases pin both centerings). LogNormal(0,4) hypers native (Stan adds the
# Jacobian exactly as the source's hand term).
function _brm_hsgp_s(cfill::Float64)
    d = _mcycle_data()
    cmu = fill(cfill, 20)
    csg = fill(cfill, 20)
    return @slic (; y=d.y, basis=d.basis, fsq=d.fsq, cmu=cmu, csg=csg,
        K=20, N=length(d.y)) begin
        rho_mu ~ lognormal(0, 4; lower=0.0)
        sd_mu ~ lognormal(0, 4; lower=0.0)
        rho_sg ~ lognormal(0, 4; lower=0.0)
        sd_sg ~ lognormal(0, 4; lower=0.0)
        ls_mu = log(sd_mu) + 0.5 * log(rho_mu) + 0.25 * log(2 * pi) -
            0.25 * exp(2 * log(rho_mu)) * fsq
        ls_sg = log(sd_sg) + 0.5 * log(rho_sg) + 0.25 * log(2 * pi) -
            0.25 * exp(2 * log(rho_sg)) * fsq
        v_mu::vector[K] ~ hsgp_v(ls_mu, cmu)
        v_sg::vector[K] ~ hsgp_v(ls_sg, csg)
        w_mu = v_mu .* exp((1 .- cmu) .* ls_mu)
        w_sg = v_sg .* exp((1 .- csg) .* ls_sg)
        mu = basis * w_mu
        lsg = basis * w_sg
        y ~ normal(mu, exp(lsg))
    end
end

brm_hsgp_nc_s() = _brm_hsgp_s(0.0)
brm_hsgp_c_s() = _brm_hsgp_s(1.0)

