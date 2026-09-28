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

