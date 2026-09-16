# The full ctsem EMA as a HIERARCHICAL @brm model whose per-subject latent
# states are MARGINALIZED (integrated out) INSIDE the kernel cell.
#
# This is the per-subject / hierarchical marginalization — NOT a single-series
# simplification. Each subject's 2 coupled continuous-time latent states
# (stress, mood) are integrated out by an Extended Kalman filter that runs as a
# custom StanBlocks `@lpxf` family, invoked once per subject in the kernel(...)
# do-block. The population lives entirely on the @brm formula surface:
# per-subject parameters are ordinary formula LPs WITH covariates + random
# effects (age on the stress-drift baseline, treatment on the mood intercept,
# `(1|subject)` throughout), exactly as in any hierarchical @brm model.
#
# Faithful to the nonlinear/non-Gaussian EMA (see ema_brm.jl / ema_ekf.jl):
#   - stress drift has a softplus self-decay  -log1p_exp(b0 + bm*mood)*stress
#   - workload enters the stress drift as a time-varying covariate
#   - process noise on stress is state-dependent  exp(q0 + qw*workload)
#   - 2 continuous indicators (stressReport, moodReport): Gaussian EKF update
#   - 1 binary indicator (smoked): Bernoulli-logit, statistical-linearization
#     update (the states stay marginalized; no per-occasion latent is sampled).
#
# Contrast with the SAMPLED full model (ema_brm.jl, dim 175): there the 2*T
# innovations per subject are free parameters. Here the ENTIRE latent path of
# every subject is integrated out, so only the population/observation params and
# the per-subject random effects remain.
#
# VERIFIED (strato2, StanBlocks bec23bc3c523): SBBRMI -> stan_code ->
# stanc_check -> stan_instantiate + LogDensityProblems.logdensity_and_gradient,
# finite log-density + gradient. 5 subjects x 10 occasions -> dim 46.
#
# Run: julia --project=test research/ema_ctsem/ema_kernel_marginalized.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

# ── the per-subject marginalizer: an EKF as a custom @lpxf family ─────────────
# ys is the LHS (stressReport); moodReport / smoked / workload / dt are per-cell
# vector args (the kernel slices vectors, not matrices). 1.8378770664093453 =
# log(2pi), via a zero-arg @deffun (a Julia const can't be used inside @deffun).
StanBlocks.@deffun begin
    l2pi()::real = 1.8378770664093453
    @lhs @lpxf ema_ekf_lpdf(ys::vector[T], ym::vector[T], smoked::int[T],
            workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::real = begin
        ms=ms0; mm=mm0; p11=P0; p12=0.0; p22=P0; ll=0.0
        for t in 1:T
            if t>1
                wl=workload[t-1]; d=dt[t]
                sp=log1p_exp(b0+bm*mm); sig=inv_logit(b0+bm*mm)
                nms=ms+(-sp*ms+a12*mm+wls*wl)*d; nmm=mm+(a21*ms+a22*mm+cm)*d
                f11=1-d*sp; f12=d*(a12-bm*sig*ms); f21=d*a21; f22=1+d*a22
                gs=exp(q0+qw*wl); qs=gs*gs*d; qm=diffm*diffm*d
                fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22; np22=fp21*f21+fp22*f22+qm
                ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
            end
            # Gaussian update (2 continuous indicators)
            v1=ys[t]-ms; v2=ym[t]-mm; s11=p11+r1; s12=p12; s22=p22+r2
            det=s11*s22-s12*s12; si11=s22/det; si12=-s12/det; si22=s11/det
            quad=v1*(si11*v1+si12*v2)+v2*(si12*v1+si22*v2)
            ll=ll-0.5*(2*l2pi()+log(det)+quad)
            k11=p11*si11+p12*si12; k12=p11*si12+p12*si22; k21=p12*si11+p22*si12; k22=p12*si12+p22*si22
            ms=ms+k11*v1+k12*v2; mm=mm+k21*v1+k22*v2
            g11=(1-k11)*p11-k12*p12; g12=(1-k11)*p12-k12*p22; g21=-k21*p11+(1-k22)*p12; g22=-k21*p12+(1-k22)*p22
            p11=g11; p12=0.5*(g12+g21); p22=g22
            # Bernoulli-logit update (binary indicator), statistical linearization
            logit=l31*ms+thr; p=inv_logit(logit)
            ll=ll+(smoked[t]*logit-log1p_exp(logit))
            pv=p*(1-p); h1=pv*l31; sb=h1*p11*h1+pv; kb1=(p11*h1)/sb; kb2=(p12*h1)/sb; resid=smoked[t]-p
            ms=ms+kb1*resid; mm=mm+kb2*resid
            b11=p11-kb1*h1*p11; b12=p12-kb1*h1*p12; b21=p12-kb2*h1*p11; b22=p22-kb2*h1*p12
            p11=b11; p12=0.5*(b12+b21); p22=b22
        end
        ll
    end
    # per-occasion contributions (auto posterior-predictive twin)
    ema_ekf_lpdfs(ys::vector[T], ym::vector[T], smoked::int[T], workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::vector[T] = begin
        out::vector[T]; ms=ms0; mm=mm0; p11=P0; p12=0.0; p22=P0
        for t in 1:T
            if t>1
                wl=workload[t-1]; d=dt[t]; sp=log1p_exp(b0+bm*mm); sig=inv_logit(b0+bm*mm)
                nms=ms+(-sp*ms+a12*mm+wls*wl)*d; nmm=mm+(a21*ms+a22*mm+cm)*d
                f11=1-d*sp; f12=d*(a12-bm*sig*ms); f21=d*a21; f22=1+d*a22
                gs=exp(q0+qw*wl); qs=gs*gs*d; qm=diffm*diffm*d
                fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22; np22=fp21*f21+fp22*f22+qm
                ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
            end
            v1=ys[t]-ms; v2=ym[t]-mm; s11=p11+r1; s12=p12; s22=p22+r2
            det=s11*s22-s12*s12; si11=s22/det; si12=-s12/det; si22=s11/det
            quad=v1*(si11*v1+si12*v2)+v2*(si12*v1+si22*v2); lg=-0.5*(2*l2pi()+log(det)+quad)
            k11=p11*si11+p12*si12; k12=p11*si12+p12*si22; k21=p12*si11+p22*si12; k22=p12*si12+p22*si22
            ms=ms+k11*v1+k12*v2; mm=mm+k21*v1+k22*v2
            g11=(1-k11)*p11-k12*p12; g12=(1-k11)*p12-k12*p22; g21=-k21*p11+(1-k22)*p12; g22=-k21*p12+(1-k22)*p22
            p11=g11; p12=0.5*(g12+g21); p22=g22
            logit=l31*ms+thr; p=inv_logit(logit); lb=smoked[t]*logit-log1p_exp(logit)
            pv=p*(1-p); h1=pv*l31; sb=h1*p11*h1+pv; kb1=(p11*h1)/sb; kb2=(p12*h1)/sb; resid=smoked[t]-p
            ms=ms+kb1*resid; mm=mm+kb2*resid
            b11=p11-kb1*h1*p11; b12=p12-kb1*h1*p12; b21=p12-kb2*h1*p11; b22=p22-kb2*h1*p12
            p11=b11; p12=0.5*(b12+b21); p22=b22; out[t]=lg+lb
        end
        out
    end
    ema_ekf_rng(vector[T], ym::vector[T], smoked::int[T], workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::vector[T] = begin
        out::vector[T]; s=normal_rng(ms0,sqrt(P0)); m=normal_rng(mm0,sqrt(P0))
        for t in 1:T
            if t>1
                wl=workload[t-1]; d=dt[t]; gs=exp(q0+qw*wl)
                s=s+(-log1p_exp(b0+bm*m)*s+a12*m+wls*wl)*d+gs*sqrt(d)*normal_rng(0.,1.)
                m=m+(a21*s+a22*m+cm)*d+diffm*sqrt(d)*normal_rng(0.,1.)
            end
            out[t]=normal_rng(s,sqrt(r1))
        end
        out
    end
end

# ── synthetic EMA panel: per-subject covariates + ragged obs ──────────────────
function fixture(; n=5, nt=10, seed=1)
    rng=seed; rnd()=(rng=(1103515245*rng+12345)%2^31; rng/2^31)
    subject=String[]; age=Float64[]; treatment=Float64[]
    stressReport=Vector{Float64}[]; moodReport=Vector{Float64}[]
    smoked=Vector{Int}[]; workload=Vector{Float64}[]; dt=Vector{Float64}[]
    for i in 1:n
        push!(subject,"s$i"); push!(age,(i-n/2)/n); push!(treatment, i % 2 == 0 ? 1.0 : 0.0)
        s=0.0; m=0.0; a=Float64[]; b=Float64[]; c=Int[]; w=Float64[]; d=Float64[]
        for _ in 1:nt
            dv=0.5+0.3*rnd(); push!(d,dv); ww=rnd()-0.5; push!(w,ww)
            s=0.7s+0.3m+0.5ww+0.1*(rnd()-0.5); m=0.6m+0.2s+0.1*(rnd()-0.5)
            push!(a,s+0.1*(rnd()-0.5)); push!(b,m+0.1*(rnd()-0.5)); push!(c, s>0.2 ? 1 : 0)
        end
        push!(stressReport,a); push!(moodReport,b); push!(smoked,c); push!(workload,w); push!(dt,d)
    end
    (; subject, age, treatment, stressReport, moodReport, smoked, workload, dt)
end
data = fixture()

# ── the model: population on the formula surface, states marginalized in-cell ─
ema_kernel_marginalized(d) = @brm d begin
    # global drift / coupling / observation params
    bm    ~ Normal(0, 0.5)                 # mood -> stress softplus modulation
    a12   ~ Normal(0, 0.5)                 # mood -> stress
    a21   ~ Normal(0, 0.5)                 # stress -> mood
    a22   ~ Normal(-0.5, 0.3)              # mood self-decay
    wls   ~ Normal(0, 0.5)                 # workload -> stress
    qw    ~ Normal(0, 0.5)                 # workload -> stress process-noise
    diffm ~ Exponential(1.0)              # mood process-noise sd
    l31   ~ Normal(0, 1)                   # stress -> smoking loading
    thr   ~ Normal(0, 1)                   # smoking threshold
    r1    ~ Exponential(1.0)              # stressReport meas. sd
    r2    ~ Exponential(1.0)              # moodReport meas. sd

    # per-subject parameters: covariates + random effects on the FORMULA surface
    b0    ~ 1 + age + (1 | subject)        # stress-drift baseline (age covariate)
    q0    ~ 1 + (1 | subject)              # stress process-noise baseline
    cm    ~ 1 + treatment + (1 | subject)  # mood continuous intercept (treatment covariate)
    s0    ~ 1 + (1 | subject)              # stress initial mean
    m0    ~ 1 + (1 | subject)              # mood initial mean

    # per subject: integrate out THIS subject's entire latent path with the EKF
    pred ~ kernel(dt, workload, stressReport, moodReport, smoked,
                  b0, q0, cm, s0, m0) do dti, wli, ys, ym, smk, lb0, lq0, lcm, ls0, lm0
        ys ~ ema_ekf(ym, smk, wli, dti,
                     lb0, bm, a12, a21, a22, lcm, wls, lq0, qw, diffm, l31, thr, r1, r2,
                     ls0, lm0, 10.0)
        ys
    end
end

function main()
    sb = SBBRMI(ema_kernel_marginalized(data); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @assert StanBlocks.stanc_check(code; warn_pedantic=false).ok "stanc failed"
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_kernel_marg.stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.05*((i % 7) - 3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("hierarchical kernel-cell EKF-marginalized EMA")
    println("  subjects = ", length(data.subject), "  occasions = ", length(data.dt[1]))
    println("  dim = ", dim, "  lp = ", round(lp; digits=2), "  finite_grad = ", all(isfinite, g))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
