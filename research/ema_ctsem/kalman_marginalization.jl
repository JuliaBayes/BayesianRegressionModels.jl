# Marginalizing the latent states of a state-space model with ONE general,
# DIMENSION-GENERIC Kalman-filter function.
#
#   kalman(y, A, Q, C, R, m0, P0)   integrates the states out of ANY
#   linear-Gaussian state-space model:
#       x[t] = A x[t-1] + N(0, Q)     (state, dim K — inferred from A)
#       y[t] = C x[t]   + N(0, R)     (obs,   dim M — inferred from y, C)
#   K and M come from the matrix argument sizes, so the SAME function marginalizes
#   the EMA's 2 coupled states (K=M=2) and a 1-D AR(1) (K=M=1).
#
# NB: this is NOT a higher-order function — it takes linear-algebra objects
# (A, Q, C, R), not functions. A TRUE marginalization HOF takes user predict/
# update functions and covers Kalman / EKF / HMM-forward uniformly; that broader
# "sequential-marginalization" combinator family is StanBlocks-level work
# (decision 14doey5, option B — delegated to StanBlocks).
#
# This is the LINEAR-GAUSSIAN version of the EMA (the softplus drift replaced by a
# free linear drift matrix A; the binary smoking indicator dropped) — exactly the
# case where marginalization is exact. The faithful nonlinear/non-Gaussian EMA
# (softplus drift + Bernoulli indicator) needs an Extended Kalman filter — see
# ema_ekf.jl.
#
# VERIFIED (strato2, StanBlocks bec23bc3c523), T=30, stanc + BridgeStan finite
# log-density + gradient:
#   EMA 2 states  sampled (2·T innovations) : dim 68
#                 marginalized (Kalman HOF)  : dim 8    <- states integrated out
#   AR(1)         marginalized (Kalman HOF)  : dim 3    <- same function, K=1
#
# Run: julia --project=test research/ema_ctsem/kalman_marginalization.jl

using StanBlocks
using LogDensityProblems

# ── the HOF: one general multivariate Kalman marginal log-density ─────────────
# 1.8378770664093453 = log(2π), inlined (a Julia const can't be used in @deffun).
StanBlocks.@deffun begin
    @lhs @lpxf kalman_lpdf(y::matrix[M, T], A::matrix[K, K], Q::matrix[K, K],
                           C::matrix[M, K], R::matrix[M, M],
                           m0::vector[K], P0::matrix[K, K])::real = begin
        m::vector[K] = m0
        P::matrix[K, K] = P0
        ll = 0.0
        for t in 1:T
            if t > 1
                m = A * m
                P = A * P * A' + Q
            end
            v::vector[M] = y[:, t] - C * m
            S::matrix[M, M] = C * P * C' + R
            Sinv::matrix[M, M] = inverse(S)
            ll = ll - 0.5 * (M * 1.8378770664093453 + log_determinant(S) + quad_form(Sinv, v))
            G::matrix[K, M] = P * C' * Sinv
            m = m + G * v
            P = P - G * C * P
        end
        ll
    end
    kalman_lpdfs(y::matrix[M, T], A::matrix[K, K], Q::matrix[K, K],
                 C::matrix[M, K], R::matrix[M, M],
                 m0::vector[K], P0::matrix[K, K])::vector[T] = begin
        out::vector[T]
        m::vector[K] = m0
        P::matrix[K, K] = P0
        for t in 1:T
            if t > 1
                m = A * m
                P = A * P * A' + Q
            end
            v::vector[M] = y[:, t] - C * m
            S::matrix[M, M] = C * P * C' + R
            Sinv::matrix[M, M] = inverse(S)
            out[t] = -0.5 * (M * 1.8378770664093453 + log_determinant(S) + quad_form(Sinv, v))
            G::matrix[K, M] = P * C' * Sinv
            m = m + G * v
            P = P - G * C * P
        end
        out
    end
    kalman_rng(matrix[M, T], A::matrix[K, K], Q::matrix[K, K],
               C::matrix[M, K], R::matrix[M, M],
               m0::vector[K], P0::matrix[K, K])::matrix[M, T] = begin
        out::matrix[M, T]
        x::vector[K] = multi_normal_rng(m0, P0)
        for t in 1:T
            if t > 1
                x = multi_normal_rng(A * x, Q)
            end
            out[:, t] = multi_normal_rng(C * x, R)
        end
        out
    end
end

# matrix-assembly helpers (build A/Q/R from scalar params) + the sampled trajectory
StanBlocks.@deffun begin
    mat2(a::real, b::real, c::real, d::real)::matrix[2, 2] = begin
        M::matrix[2, 2]; M[1,1]=a; M[1,2]=b; M[2,1]=c; M[2,2]=d; M
    end
    mat1(a::real)::matrix[1, 1] = begin
        M::matrix[1, 1]; M[1,1]=a; M
    end
    lg_traj(zs::vector[T], zm::vector[T], a11::real, a12::real, a21::real, a22::real,
            sq1::real, sq2::real, s10::real, s20::real)::matrix[2, T] = begin
        out::matrix[2, T]
        s1 = s10; s2 = s20
        out[1,1] = s1; out[2,1] = s2
        for t in 2:T
            ns1 = a11*s1 + a12*s2 + sq1*zs[t]
            ns2 = a21*s1 + a22*s2 + sq2*zm[t]
            s1 = ns1; s2 = ns2
            out[1,t] = s1; out[2,t] = s2
        end
        out
    end
end

# EMA 2 states — MARGINALIZED (states integrated out by the Kalman HOF).
ema2_marginalized(y, I2, m0, P0) = @slic (; y=y, I2=I2, m0=m0, P0=P0) begin
    a11 ~ normal(-0.5,0.3); a12 ~ normal(0,0.3); a21 ~ normal(0,0.3); a22 ~ normal(-0.5,0.3)
    q1 ~ exponential(1.0); q2 ~ exponential(1.0); r1 ~ exponential(1.0); r2 ~ exponential(1.0)
    y ~ kalman(mat2(a11,a12,a21,a22), mat2(q1,0.0,0.0,q2), I2, mat2(r1,0.0,0.0,r2), m0, P0)
end

# EMA 2 states — SAMPLED (the 2·T latent states are parameters).
ema2_sampled(ys, ym) = @slic (; ys=ys, ym=ym) begin
    a11 ~ normal(-0.5,0.3); a12 ~ normal(0,0.3); a21 ~ normal(0,0.3); a22 ~ normal(-0.5,0.3)
    sq1 ~ exponential(1.0); sq2 ~ exponential(1.0); sr1 ~ exponential(1.0); sr2 ~ exponential(1.0)
    s10 ~ normal(0,1); s20 ~ normal(0,1)
    zs::vector[dims(ys)[1]] ~ std_normal()
    zm::vector[dims(ys)[1]] ~ std_normal()
    x = lg_traj(zs, zm, a11, a12, a21, a22, sq1, sq2, s10, s20)
    ys ~ normal(x[1,:], sr1)
    ym ~ normal(x[2,:], sr2)
end

# The SAME kalman HOF on a 1-D AR(1) (K=M=1) — "applicable to a bunch of things".
ar1_marginalized(y, m0, P0) = @slic (; y=y, m0=m0, P0=P0) begin
    phi ~ normal(0.5,0.3); q ~ exponential(1.0); r ~ exponential(1.0)
    y ~ kalman(mat1(phi), mat1(q), mat1(1.0), mat1(r), m0, P0)
end

function gate(name, model)
    print(rpad(name, 38))
    local code
    try code = StanBlocks.stan_code(model) catch e
        println("transpile FAIL: ", first(sprint(showerror, e), 150)); return end
    r = StanBlocks.stanc_check(code; warn_pedantic=false)
    r.ok || (println("stanc FAIL"); return)
    cache = joinpath(tempdir(), "brm-khof"); isdir(cache)||mkpath(cache)
    prob = StanBlocks.stan_instantiate(model; path=joinpath(cache, string(hash(code))*".stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.1*((i%5)-2) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("OK  dim=", dim, "  finite_grad=", all(isfinite,g))
end

function main()
    T = 30
    ys = [sin(t/3) + 0.1*(t%5-2) for t in 1:T]
    ym = [cos(t/4) + 0.1*(t%3-1) for t in 1:T]
    y2 = permutedims(hcat(ys, ym)); y1 = permutedims(reshape(ys, T, 1))
    I2 = [1.0 0.0; 0.0 1.0]; m0_2 = [0.0, 0.0]; P0_2 = [10.0 0.0; 0.0 10.0]
    m0_1 = [0.0]; P0_1 = reshape([10.0], 1, 1)
    println("T = ", T, "  (one kalman HOF)")
    gate("EMA 2 states — sampled",       ema2_sampled(ys, ym))
    gate("EMA 2 states — marginalized",  ema2_marginalized(y2, I2, m0_2, P0_2))
    gate("AR(1) — marginalized (K=1)",   ar1_marginalized(y1, m0_1, P0_1))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
