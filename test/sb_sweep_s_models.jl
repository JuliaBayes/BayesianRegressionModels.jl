# test/sb_sweep_s_models.jl — @slic port builders for the v1 SB-parity sweep S driver.
#
# Included (not standalone) by test/sb_sweep_s.jl AFTER the
# ReactiveKernelsPPLExamples using block (builders consume the M.* data
# bindings). Split out so transpile probes can include the builders without
# running the record loop.

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

