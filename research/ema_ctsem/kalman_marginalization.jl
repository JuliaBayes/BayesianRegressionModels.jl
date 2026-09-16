# Marginalizing the latent states via a Kalman filter — "yes, it's just a @deffun".
#
# Companion to the EMA translation (ema_brm.jl). The EMA/SSM models there SAMPLE
# the latent states (innovations are parameters; dim grows as N*T*processes — the
# "expressive but not competitive" regime). This file shows the competitive
# alternative for the LINEAR-GAUSSIAN case: a Kalman filter as a custom `@lpxf`
# log-density that integrates the states OUT, so HMC samples only the physical
# parameters — and the innovation funnel disappears.
#
# VERIFIED (strato2, StanBlocks bec23bc3c523), T=30 scalar AR(1) state space:
#   sampled (innovations are parameters) : dim 34  (3 params + x0 + 30 innovations)
#   marginalized (Kalman @lpxf)          : dim 3   (states integrated out)
# both stanc + BridgeStan finite log-density + gradient.
#
# StanBlocks already ships marginal-likelihood recurrences of exactly this shape
# (`car_normal_lpdf`, `garch11_lpdf`, `arma11_lpdf`, `ar1_recurse`) — a Kalman
# filter is the same pattern.
#
# BOUNDARY FINDING — the @lpxf family works directly at the @slic level, but the
# @brm formula surface REJECTS a custom @lpxf distribution as a top-level response:
#   "distribution `kalman_ar1` has no Stan translation; define `_sb_stan_dist_name`
#    or `_sb_stan_distribution_call`".
# So making Kalman marginalization a first-class @brm response family needs a
# BRM-side registration hook (a small BRM change); at the @slic level it is
# already usable today.
#
# WHERE THE NONLINEAR EMA WOULD NEED MORE: exact Kalman is linear-Gaussian only.
# The EMA drift is state-dependent and one indicator is Bernoulli, so the filter
# becomes an EXTENDED KF (per-step Jacobian linearization) + quadrature/Laplace
# for the binary indicator — same @deffun SHAPE, but approximate and numerically
# more delicate (positive-definite covariance recursion, AD through solve/logdet).
#
# Run: julia --project=test research/ema_ctsem/kalman_marginalization.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

# ── The Kalman marginal log-density as a custom @lpxf distribution ────────────
# Scalar linear-Gaussian filter: transition phi, process var q, obs var r, prior
# x0 ~ N(m0, P0). Returns log p(y | phi, q, r, m0, P0), latent trajectory
# integrated out. Triad: _lpdf / _lpdfs / _rng. `1.8378770664093453` = log(2*pi)
# inlined (a Julia const cannot be referenced inside a @deffun body).
StanBlocks.@deffun begin
    @lhs @lpxf kalman_ar1_lpdf(y::vector[T], phi::real, q::real, r::real,
                               m0::real, P0::real)::real = begin
        m = m0; P = P0
        ll = 0.0
        for t in 1:T
            if t > 1
                m = phi * m
                P = phi * phi * P + q
            end
            v = y[t] - m
            S = P + r
            ll = ll - 0.5 * (1.8378770664093453 + log(S) + v * v / S)
            Kg = P / S
            m = m + Kg * v
            P = (1.0 - Kg) * P
        end
        ll
    end
    kalman_ar1_lpdfs(y::vector[T], phi::real, q::real, r::real,
                     m0::real, P0::real)::vector[T] = begin
        out::vector[T]
        m = m0; P = P0
        for t in 1:T
            if t > 1
                m = phi * m
                P = phi * phi * P + q
            end
            v = y[t] - m
            S = P + r
            out[t] = -0.5 * (1.8378770664093453 + log(S) + v * v / S)
            Kg = P / S
            m = m + Kg * v
            P = (1.0 - Kg) * P
        end
        out
    end
    kalman_ar1_rng(vector[T], phi::real, q::real, r::real,
                   m0::real, P0::real)::vector[T] = begin
        out::vector[T]
        x = normal_rng(m0, sqrt(P0))
        for t in 1:T
            if t > 1
                x = normal_rng(phi * x, sqrt(q))
            end
            out[t] = normal_rng(x, sqrt(r))
        end
        out
    end
end

# SAMPLED: the AR(1) latent trajectory is built from sampled innovations.
StanBlocks.@deffun begin
    ar1_traj(z::vector[T], phi::real, sq::real, x0::real)::vector[T] = begin
        x::vector[T]
        cur = x0
        x[1] = cur
        for t in 2:T
            cur = phi * cur + sq * z[t]
            x[t] = cur
        end
        x
    end
end
sampled(y) = @slic (; y = y) begin
    phi ~ normal(0.5, 0.3)
    sq  ~ exponential(1.0)        # process SD
    sr  ~ exponential(1.0)        # obs SD
    x0  ~ normal(0.0, 1.0)
    z::vector[dims(y)[1]] ~ std_normal()
    x = ar1_traj(z, phi, sq, x0)
    y ~ normal(x, sr)
end

# MARGINALIZED: states integrated out by the Kalman filter (dim = params only).
marginalized(y) = @slic (; y = y) begin
    phi ~ normal(0.5, 0.3)
    q   ~ exponential(1.0)        # process VARIANCE
    r   ~ exponential(1.0)        # obs VARIANCE
    y ~ kalman_ar1(phi, q, r, 0.0, 10.0)
end

# @brm surface: a custom @lpxf family is NOT accepted as a top-level response
# without a BRM `_sb_stan_dist_name` registration (documented boundary).
brm_marginalized(data) = @brm data begin
    phi ~ Normal(0.5, 0.3)
    q ~ Exponential(1.0)
    r ~ Exponential(1.0)
    y ~ kalman_ar1(phi, q, r, 0.0, 10.0)
end

function gate(name, model)
    print(rpad(name, 34))
    local code
    try code = StanBlocks.stan_code(model) catch e
        println("transpile FAIL: ", first(sprint(showerror, e), 100)); return end
    r = StanBlocks.stanc_check(code; warn_pedantic=false)
    r.ok || (println("stanc FAIL: ", first(r.output, 200)); return)
    cache = joinpath(tempdir(), "brm-kalman"); isdir(cache)||mkpath(cache)
    prob = StanBlocks.stan_instantiate(model; path=joinpath(cache, string(hash(code))*".stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.1*((i%5)-2) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("OK  dim=", dim, "  lp=", round(lp;digits=3), "  finite_grad=", all(isfinite,g))
end

function main()
    y = [sin(t/3) + 0.1*(t%5-2) for t in 1:30]
    println("T = ", length(y))
    gate("sampled (innovations sampled)", sampled(y))
    gate("marginalized (Kalman @lpxf)", marginalized(y))
    println("── @brm wiring (expected to reject: needs _sb_stan_dist_name) ──")
    try
        sb = SBBRMI(brm_marginalized((; y = y)); mod=@__MODULE__)
        gate("@brm y ~ kalman_ar1(...)", sb.model)
    catch e
        println("@brm rejected: ", first(sprint(showerror, e), 140))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
