# The EMA as a HIERARCHICAL @brm model whose per-subject latent states are
# marginalized EXACTLY by a Kalman filter INSIDE the kernel cell.
#
# This is the LINEAR-GAUSSIAN rung of the SSM hierarchy: the softplus stress
# self-decay of the full EMA is replaced by a FREE linear drift coefficient
# (a11) and the binary indicator is dropped, so the model is linear-Gaussian and
# the Kalman filter integrates the states out EXACTLY (no approximation). It is
# the exact-marginalization counterpart of ema_kernel_marginalized.jl (which uses
# an EKF for the true nonlinear/non-Gaussian EMA) and of ema_brm.jl (which
# SAMPLES the same continuous-time coupled states).
#
# Same shape as the EKF file: the population lives on the @brm formula surface
# (per-subject params are ordinary LPs with covariates + random effects), and
# each subject's entire latent path is integrated out in the kernel(...) do-block
# by a custom StanBlocks @lpxf Kalman family invoked once per subject.
#
# VERIFIED (strato2, StanBlocks bec23bc3c523): SBBRMI -> stan_code -> stanc_check
# -> stan_instantiate + LogDensityProblems.logdensity_and_gradient, finite.
#
# Run: julia --project=test research/ema_ctsem/ema_kernel_kalman.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

# ── the per-subject exact marginalizer: a 2-state Kalman filter as an @lpxf ────
# Continuous-time linear-Gaussian dynamics; ys is the LHS (stressReport),
# ym / workload / dt are per-cell vector args (the kernel slices vectors).
# 1.8378770664093453 = log(2pi), via a zero-arg @deffun (no @deffun const).
StanBlocks.@deffun begin
    l2pi()::real = 1.8378770664093453
    @lhs @lpxf kalman2_lpdf(ys::vector[T], ym::vector[T], workload::vector[T], dt::vector[T],
            a11::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q1::real, q2::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::real = begin
        ms=ms0; mm=mm0; p11=P0; p12=0.0; p22=P0; ll=0.0
        for t in 1:T
            if t>1
                wl=workload[t-1]; d=dt[t]
                nms=ms+(a11*ms+a12*mm+wls*wl)*d; nmm=mm+(a21*ms+a22*mm+cm)*d
                f11=1+a11*d; f12=a12*d; f21=a21*d; f22=1+a22*d
                qs=q1*q1*d; qm=q2*q2*d
                fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22; np22=fp21*f21+fp22*f22+qm
                ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
            end
            v1=ys[t]-ms; v2=ym[t]-mm; s11=p11+r1; s12=p12; s22=p22+r2
            det=s11*s22-s12*s12; si11=s22/det; si12=-s12/det; si22=s11/det
            quad=v1*(si11*v1+si12*v2)+v2*(si12*v1+si22*v2)
            ll=ll-0.5*(2*l2pi()+log(det)+quad)
            k11=p11*si11+p12*si12; k12=p11*si12+p12*si22; k21=p12*si11+p22*si12; k22=p12*si12+p22*si22
            ms=ms+k11*v1+k12*v2; mm=mm+k21*v1+k22*v2
            g11=(1-k11)*p11-k12*p12; g12=(1-k11)*p12-k12*p22; g21=-k21*p11+(1-k22)*p12; g22=-k21*p12+(1-k22)*p22
            p11=g11; p12=0.5*(g12+g21); p22=g22
        end
        ll
    end
    kalman2_lpdfs(ys::vector[T], ym::vector[T], workload::vector[T], dt::vector[T],
            a11::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q1::real, q2::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::vector[T] = begin
        out::vector[T]; ms=ms0; mm=mm0; p11=P0; p12=0.0; p22=P0
        for t in 1:T
            if t>1
                wl=workload[t-1]; d=dt[t]
                nms=ms+(a11*ms+a12*mm+wls*wl)*d; nmm=mm+(a21*ms+a22*mm+cm)*d
                f11=1+a11*d; f12=a12*d; f21=a21*d; f22=1+a22*d; qs=q1*q1*d; qm=q2*q2*d
                fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22; np22=fp21*f21+fp22*f22+qm
                ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
            end
            v1=ys[t]-ms; v2=ym[t]-mm; s11=p11+r1; s12=p12; s22=p22+r2
            det=s11*s22-s12*s12; si11=s22/det; si12=-s12/det; si22=s11/det
            quad=v1*(si11*v1+si12*v2)+v2*(si12*v1+si22*v2); out[t]=-0.5*(2*l2pi()+log(det)+quad)
            k11=p11*si11+p12*si12; k12=p11*si12+p12*si22; k21=p12*si11+p22*si12; k22=p12*si12+p22*si22
            ms=ms+k11*v1+k12*v2; mm=mm+k21*v1+k22*v2
            g11=(1-k11)*p11-k12*p12; g12=(1-k11)*p12-k12*p22; g21=-k21*p11+(1-k22)*p12; g22=-k21*p12+(1-k22)*p22
            p11=g11; p12=0.5*(g12+g21); p22=g22
        end
        out
    end
    kalman2_rng(vector[T], ym::vector[T], workload::vector[T], dt::vector[T],
            a11::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q1::real, q2::real, r1::real, r2::real,
            ms0::real, mm0::real, P0::real)::vector[T] = begin
        out::vector[T]; s=normal_rng(ms0,sqrt(P0)); m=normal_rng(mm0,sqrt(P0))
        for t in 1:T
            if t>1
                wl=workload[t-1]; d=dt[t]
                s=s+(a11*s+a12*m+wls*wl)*d+q1*sqrt(d)*normal_rng(0.,1.)
                m=m+(a21*s+a22*m+cm)*d+q2*sqrt(d)*normal_rng(0.,1.)
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
    workload=Vector{Float64}[]; dt=Vector{Float64}[]
    for i in 1:n
        push!(subject,"s$i"); push!(age,(i-n/2)/n); push!(treatment, i % 2 == 0 ? 1.0 : 0.0)
        s=0.0; m=0.0; a=Float64[]; b=Float64[]; w=Float64[]; d=Float64[]
        for _ in 1:nt
            dv=0.5+0.3*rnd(); push!(d,dv); ww=rnd()-0.5; push!(w,ww)
            s=0.7s+0.3m+0.5ww+0.1*(rnd()-0.5); m=0.6m+0.2s+0.1*(rnd()-0.5)
            push!(a,s+0.1*(rnd()-0.5)); push!(b,m+0.1*(rnd()-0.5))
        end
        push!(stressReport,a); push!(moodReport,b); push!(workload,w); push!(dt,d)
    end
    (; subject, age, treatment, stressReport, moodReport, workload, dt)
end
data = fixture()

# ── the model: population on the formula surface, states marginalized in-cell ─
ema_kernel_kalman(d) = @brm d begin
    # global drift / coupling / noise params
    a12 ~ Normal(0, 0.5)                   # mood -> stress
    a21 ~ Normal(0, 0.5)                   # stress -> mood
    a22 ~ Normal(-0.5, 0.3)                # mood self-decay
    wls ~ Normal(0, 0.5)                   # workload -> stress
    q1  ~ Exponential(1.0)                # stress process-noise sd
    q2  ~ Exponential(1.0)                # mood process-noise sd
    r1  ~ Exponential(1.0)                # stressReport meas. sd
    r2  ~ Exponential(1.0)                # moodReport meas. sd

    # per-subject parameters: covariates + random effects on the FORMULA surface
    a11 ~ 1 + age       + (1 | subject)    # stress self-decay (age covariate)
    cm  ~ 1 + treatment + (1 | subject)    # mood intercept (treatment covariate)
    s0  ~ 1             + (1 | subject)     # stress initial mean
    m0  ~ 1             + (1 | subject)     # mood initial mean

    # per subject: integrate out THIS subject's entire latent path EXACTLY
    pred ~ kernel(dt, workload, stressReport, moodReport,
                  a11, cm, s0, m0) do dti, wli, ys, ym, la11, lcm, ls0, lm0
        ys ~ kalman2(ym, wli, dti, la11, a12, a21, a22, lcm, wls, q1, q2, r1, r2, ls0, lm0, 10.0)
        ys
    end
end

function main()
    sb = SBBRMI(ema_kernel_kalman(data); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @assert StanBlocks.stanc_check(code; warn_pedantic=false).ok "stanc failed"
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_kernel_kalman.stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.05*((i % 7) - 3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("hierarchical kernel-cell KALMAN-marginalized (exact, linear-Gaussian) EMA")
    println("  subjects = ", length(data.subject), "  occasions = ", length(data.dt[1]))
    println("  dim = ", dim, "  lp = ", round(lp; digits=2), "  finite_grad = ", all(isfinite, g))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
