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
# N(0,10). The precision-chol variant forms Omega = Lc*Lc' host-side (same
# IEEE mults either side would do; ~1e-14 vs a direct-triangular path —
# inside any parity tolerance; the record pins the exact program).
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

