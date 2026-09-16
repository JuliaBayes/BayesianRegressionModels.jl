# FAITHFUL marginalization of the REAL EMA model — an Extended Kalman Filter.
#
# The EMA is nonlinear (state-dependent softplus drift, input-dependent
# diffusion) and non-Gaussian (a Bernoulli smoking indicator), so the exact
# Kalman filter of kalman_marginalization.jl does NOT apply. The EKF integrates
# the two latent states out APPROXIMATELY, without sampling them:
#   - propagate the mean through the TRUE nonlinear Euler drift
#   - propagate the covariance through the drift's Jacobian F (linearization)
#   - Gaussian update for the two continuous indicators (C = I)
#   - EKF / statistical-linearization update for the Bernoulli indicator
# HMC then touches only the physical parameters — no per-timepoint innovations.
#
# VERIFIED (strato2, StanBlocks bec23bc3c523), T=30, stanc + BridgeStan finite
# log-density + gradient:
#   full EMA  sampled (2·T innovations) : dim 76
#            EKF-marginalized            : dim 14   <- states integrated out
#
# EKF is an APPROXIMATION (it linearizes the drift and Gaussian-approximates the
# Bernoulli update); it is not exact the way the linear-Gaussian Kalman is. It is
# what ctsem itself uses for nonlinear/non-Gaussian models. A UKF or higher-order
# assumed-density filter would trade cost for accuracy in the same @deffun shape.
#
# Run: julia --project=test research/ema_ctsem/ema_ekf.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

StanBlocks.@deffun begin
    @lhs @lpxf ema_ekf_lpdf(y::matrix[2, T], smoked::int[T],
            workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::real = begin
        ms = ms0; mm = mm0
        p11 = P0; p12 = 0.0; p22 = P0
        ll = 0.0
        for t in 1:T
            if t > 1
                wl = workload[t-1]; d = dt[t]
                sp = log1p_exp(b0 + bm*mm)              # softplus = -DRIFT[1,1] coefficient
                sig = inv_logit(b0 + bm*mm)             # d(softplus)/darg
                nms = ms + (-sp*ms + a12*mm + wls*wl)*d # nonlinear mean prediction (Euler)
                nmm = mm + (a21*ms + a22*mm + cm)*d
                f11 = 1 + d*(-sp);  f12 = d*(a12 - bm*sig*ms)   # Jacobian F = I + dt·∂drift/∂x
                f21 = d*a21;        f22 = 1 + d*a22
                gs = exp(q0 + qw*wl)                    # input-dependent diffusion
                qs = gs*gs*d;  qm = diffm*diffm*d       # Q = diag(gs²·dt, diffm²·dt)
                fp11 = f11*p11+f12*p12; fp12 = f11*p12+f12*p22  # P' = F P F' + Q
                fp21 = f21*p11+f22*p12; fp22 = f21*p12+f22*p22
                np11 = fp11*f11+fp12*f12+qs
                np12 = fp11*f21+fp12*f22
                np22 = fp21*f21+fp22*f22+qm
                ms = nms; mm = nmm; p11 = np11; p12 = np12; p22 = np22
            end
            # update: two continuous Gaussian indicators (C = I)
            v1 = y[1,t]-ms;  v2 = y[2,t]-mm
            s11 = p11+r1;  s12 = p12;  s22 = p22+r2
            det = s11*s22 - s12*s12
            si11 = s22/det;  si12 = -s12/det;  si22 = s11/det
            quad = v1*(si11*v1+si12*v2) + v2*(si12*v1+si22*v2)
            ll = ll - 0.5*(2*1.8378770664093453 + log(det) + quad)
            k11 = p11*si11+p12*si12; k12 = p11*si12+p12*si22
            k21 = p12*si11+p22*si12; k22 = p12*si12+p22*si22
            ms = ms + k11*v1+k12*v2;  mm = mm + k21*v1+k22*v2
            g11 = (1-k11)*p11 - k12*p12;  g12 = (1-k11)*p12 - k12*p22
            g21 = -k21*p11 + (1-k22)*p12; g22 = -k21*p12 + (1-k22)*p22
            p11 = g11; p12 = 0.5*(g12+g21); p22 = g22
            # update: binary Bernoulli indicator (EKF / statistical linearization)
            logit = l31*ms + thr
            p = inv_logit(logit)
            ll = ll + (smoked[t]*logit - log1p_exp(logit))     # Bernoulli-logit log-lik
            pv = p*(1-p);  h1 = pv*l31                          # H = [∂p/∂stress, 0]
            sb = h1*p11*h1 + pv                                 # H P H' + Var(Bernoulli)
            kb1 = (p11*h1)/sb;  kb2 = (p12*h1)/sb               # gain (2x1)
            resid = smoked[t] - p
            ms = ms + kb1*resid;  mm = mm + kb2*resid
            b11 = p11 - kb1*h1*p11;  b12 = p12 - kb1*h1*p12     # P - Kb H P
            b21 = p12 - kb2*h1*p11;  b22 = p22 - kb2*h1*p12
            p11 = b11; p12 = 0.5*(b12+b21); p22 = b22
        end
        ll
    end
    ema_ekf_lpdfs(y::matrix[2, T], smoked::int[T], workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::vector[T] = begin
        out::vector[T]
        ms = ms0; mm = mm0; p11 = P0; p12 = 0.0; p22 = P0
        for t in 1:T
            if t > 1
                wl = workload[t-1]; d = dt[t]
                sp = log1p_exp(b0 + bm*mm);  sig = inv_logit(b0 + bm*mm)
                nms = ms + (-sp*ms + a12*mm + wls*wl)*d
                nmm = mm + (a21*ms + a22*mm + cm)*d
                f11 = 1 + d*(-sp);  f12 = d*(a12 - bm*sig*ms);  f21 = d*a21;  f22 = 1 + d*a22
                gs = exp(q0 + qw*wl);  qs = gs*gs*d;  qm = diffm*diffm*d
                fp11 = f11*p11+f12*p12; fp12 = f11*p12+f12*p22
                fp21 = f21*p11+f22*p12; fp22 = f21*p12+f22*p22
                np11 = fp11*f11+fp12*f12+qs; np12 = fp11*f21+fp12*f22; np22 = fp21*f21+fp22*f22+qm
                ms = nms; mm = nmm; p11 = np11; p12 = np12; p22 = np22
            end
            v1 = y[1,t]-ms; v2 = y[2,t]-mm
            s11 = p11+r1; s12 = p12; s22 = p22+r2
            det = s11*s22 - s12*s12
            si11 = s22/det; si12 = -s12/det; si22 = s11/det
            quad = v1*(si11*v1+si12*v2) + v2*(si12*v1+si22*v2)
            lg = -0.5*(2*1.8378770664093453 + log(det) + quad)
            k11 = p11*si11+p12*si12; k12 = p11*si12+p12*si22
            k21 = p12*si11+p22*si12; k22 = p12*si12+p22*si22
            ms = ms + k11*v1+k12*v2; mm = mm + k21*v1+k22*v2
            g11 = (1-k11)*p11 - k12*p12; g12 = (1-k11)*p12 - k12*p22
            g21 = -k21*p11 + (1-k22)*p12; g22 = -k21*p12 + (1-k22)*p22
            p11 = g11; p12 = 0.5*(g12+g21); p22 = g22
            logit = l31*ms + thr; p = inv_logit(logit)
            lb = smoked[t]*logit - log1p_exp(logit)
            pv = p*(1-p); h1 = pv*l31; sb = h1*p11*h1 + pv
            kb1 = (p11*h1)/sb; kb2 = (p12*h1)/sb; resid = smoked[t]-p
            ms = ms + kb1*resid; mm = mm + kb2*resid
            b11 = p11 - kb1*h1*p11; b12 = p12 - kb1*h1*p12
            b21 = p12 - kb2*h1*p11; b22 = p22 - kb2*h1*p12
            p11 = b11; p12 = 0.5*(b12+b21); p22 = b22
            out[t] = lg + lb
        end
        out
    end
    ema_ekf_rng(matrix[2, T], smoked::int[T], workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::matrix[2, T] = begin
        out::matrix[2, T]
        s = normal_rng(ms0, sqrt(P0));  m = normal_rng(mm0, sqrt(P0))
        for t in 1:T
            if t > 1
                wl = workload[t-1]; d = dt[t]
                gs = exp(q0 + qw*wl)
                ns = s + (-log1p_exp(b0+bm*m)*s + a12*m + wls*wl)*d + gs*sqrt(d)*normal_rng(0.,1.)
                nm = m + (a21*s + a22*m + cm)*d + diffm*sqrt(d)*normal_rng(0.,1.)
                s = ns; m = nm
            end
            out[1,t] = normal_rng(s, sqrt(r1));  out[2,t] = normal_rng(m, sqrt(r2))
        end
        out
    end
    ema_traj(zs::vector[T], zm::vector[T], workload::vector[T], dt::vector[T],
             b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
             q0::real, qw::real, diffm::real, s0::real, m0::real)::matrix[2, T] = begin
        out::matrix[2, T]
        s = s0; m = m0
        out[1,1] = s; out[2,1] = m
        for t in 2:T
            wl = workload[t-1]; d = dt[t]
            gs = exp(q0 + qw*wl)
            ns = s + (-log1p_exp(b0+bm*m)*s + a12*m + wls*wl)*d + gs*sqrt(d)*zs[t]
            nm = m + (a21*s + a22*m + cm)*d + diffm*sqrt(d)*zm[t]
            s = ns; m = nm
            out[1,t] = s; out[2,t] = m
        end
        out
    end
end

# MARGINALIZED: the full nonlinear/non-Gaussian EMA, states integrated out (EKF).
ekf_marginalized(y, smoked, workload, dt) = @slic (; y=y, smoked=smoked, workload=workload, dt=dt) begin
    bm ~ normal(0,0.5); a12 ~ normal(0,0.5); a21 ~ normal(0,0.5); a22 ~ normal(-0.5,0.3)
    cm ~ normal(0,0.5); wls ~ normal(0,0.5); b0 ~ normal(0,0.5)
    q0 ~ normal(0,0.5); qw ~ normal(0,0.5); diffm ~ exponential(1.0)
    l31 ~ normal(0,1); thr ~ normal(0,1); r1 ~ exponential(1.0); r2 ~ exponential(1.0)
    y ~ ema_ekf(smoked, workload, dt, b0, bm, a12, a21, a22, cm, wls,
                q0, qw, diffm, l31, thr, r1, r2, 0.0, 0.0, 10.0)
end

# SAMPLED: the same nonlinear EMA with the states sampled (innovations as params).
ekf_sampled(ys, ym, smoked, workload, dt) = @slic (; ys=ys, ym=ym, smoked=smoked, workload=workload, dt=dt) begin
    bm ~ normal(0,0.5); a12 ~ normal(0,0.5); a21 ~ normal(0,0.5); a22 ~ normal(-0.5,0.3)
    cm ~ normal(0,0.5); wls ~ normal(0,0.5); b0 ~ normal(0,0.5)
    q0 ~ normal(0,0.5); qw ~ normal(0,0.5); diffm ~ exponential(1.0)
    l31 ~ normal(0,1); thr ~ normal(0,1); r1 ~ exponential(1.0); r2 ~ exponential(1.0)
    s0 ~ normal(0,1); m0 ~ normal(0,1)
    zs::vector[dims(ys)[1]] ~ std_normal()
    zm::vector[dims(ys)[1]] ~ std_normal()
    x = ema_traj(zs, zm, workload, dt, b0, bm, a12, a21, a22, cm, wls, q0, qw, diffm, s0, m0)
    ys ~ normal(x[1,:], r1)
    ym ~ normal(x[2,:], r2)
    smoked ~ bernoulli_logit(l31 * x[1,:] + thr)
end

# The full nonlinear EMA as a @brm MODEL, states integrated out by the EKF.
# `ema_ekf` is a StanBlocks @lpxf family; BRM now accepts a custom @lpxf family
# as a @brm response (src/sbimpl.jl), so this is a plain `y ~ ema_ekf(...)`.
brm_ekf_marginalized(data) = @brm data begin
    bm ~ Normal(0,0.5); a12 ~ Normal(0,0.5); a21 ~ Normal(0,0.5); a22 ~ Normal(-0.5,0.3)
    cm ~ Normal(0,0.5); wls ~ Normal(0,0.5); b0 ~ Normal(0,0.5)
    q0 ~ Normal(0,0.5); qw ~ Normal(0,0.5); diffm ~ Exponential(1.0)
    l31 ~ Normal(0,1); thr ~ Normal(0,1); r1 ~ Exponential(1.0); r2 ~ Exponential(1.0)
    y ~ ema_ekf(smoked, workload, dt, b0, bm, a12, a21, a22, cm, wls,
                q0, qw, diffm, l31, thr, r1, r2, 0.0, 0.0, 10.0)
end

function gate(name, model)
    print(rpad(name, 40))
    local code
    try code = StanBlocks.stan_code(model) catch e
        println("transpile FAIL: ", first(sprint(showerror, e), 160)); return end
    r = StanBlocks.stanc_check(code; warn_pedantic=false)
    r.ok || (println("stanc FAIL"); return)
    cache = joinpath(tempdir(), "brm-ekf"); isdir(cache)||mkpath(cache)
    prob = StanBlocks.stan_instantiate(model; path=joinpath(cache, string(hash(code))*".stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.05*((i%7)-3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("OK  dim=", dim, "  finite_grad=", all(isfinite,g))
end

function main()
    T = 30
    ys = [0.5*sin(t/3) for t in 1:T]; ym = [0.5*cos(t/4) for t in 1:T]
    smoked = [ys[t] > 0.1 ? 1 : 0 for t in 1:T]
    workload = [0.3*sin(t/2) for t in 1:T]; dt = [0.5 + 0.3*abs(sin(t)) for t in 1:T]
    y2 = permutedims(hcat(ys, ym))
    println("T = ", T)
    gate("full EMA — sampled (@slic)", ekf_sampled(ys, ym, smoked, workload, dt))
    gate("full EMA — EKF marginalized (@slic)", ekf_marginalized(y2, smoked, workload, dt))
    sb = SBBRMI(brm_ekf_marginalized((; y=y2, smoked, workload, dt)); mod=@__MODULE__)
    gate("full EMA — EKF marginalized (@brm)", sb.model)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
