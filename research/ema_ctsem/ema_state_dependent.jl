# Charles Driver's SECOND demo (`fitDemo.Rmd`): a state-dependent continuous-time
# model, translated to @brm with the latent states EKF-marginalized in the kernel.
#
# What makes this HARDER than the first EMA demo (ema_brm.jl / ema_kernel_*.jl):
# there THREE matrix cells are functions of the LATENT STATE, not of an input:
#   1. DRIFT[1,1] = -log1p(exp(b0 + bm*mood))       stress recovers faster in good mood
#   2. DIFFUSION[1,1] = exp(qd0 + qd1*mood)          stress is MORE VOLATILE in good mood
#   3. DIFFUSION[2,1] = fisher-z corr = tanh(cz*stress)   the two shocks COUPLE MORE
#                                                          tightly the more stressed
# The first demo's diffusion was INPUT-dependent (on workload) and DIAGONAL; here
# it is STATE-dependent AND correlated, so the process-noise covariance Q depends
# on the very states being integrated out. The EKF evaluates the drift Jacobian
# and Q at the running state estimate each step.
#
# Generating truth (fitDemo.Rmd, ctsem `type='ct'`, covmattransform='z'):
#   DRIFT   = [ -log1p(exp(0.5 + 0.4*mood))  -0.25 ;  -0.30  -0.60 ]
#   DIFFUSION (sd / fisher-z corr) = [ exp(-0.2 + 0.3*mood)  0 ;  0.7*stress  0.6 ]
#   CINT    = [0 ; 0.3]              (cint_mood only)
#   LAMBDA  = [1 0 ; 0 1 ; 1.2 0]    (smoked loads 1.2 on stress)
#   MANIFESTMEANS = [0 ; 0 ; -1]     (binary threshold -1; no continuous intercepts)
#   MANIFESTVAR   = diag(.3, .3, 0)  (SD form: meas. sd 0.3; binary via filter)
#   T0MEANS = [0 ; 0.5]   T0VAR = diag(.6, .5)  (SD form: T0 sds 0.6, 0.5)
#   ctsem covariance-type matrices are in SD / fisher-z form (UcorSDtoCov): the
#   actual variance is the cell SQUARED -- cf. `[merr_stress]^2` in the rendered
#   Theta and Charles's own `truecov()` in fitDemo.Rmd. T0VAR is FREE in the fit.
#   NO covariates, NO time-dependent predictors, NO between-subject random effects.
#
# VERIFIED (strato2, StanBlocks bec23bc3c523): SBBRMI -> stan_code -> stanc_check
# -> stan_instantiate + LogDensityProblems.logdensity_and_gradient, finite.
#
# Run: julia --project=test research/ema_ctsem/ema_state_dependent.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

# ── per-subject EKF marginalizer with STATE-DEPENDENT, CORRELATED diffusion ────
# ys = stressReport (LHS); ym = moodReport; smoked = binary; dt = intervals.
StanBlocks.@deffun begin
    l2pi()::real = 1.8378770664093453
    @lhs @lpxf ema_sd_lpdf(ys::vector[T], ym::vector[T], smoked::int[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cintm::real,
            qd0::real, qd1::real, cz::real, sdm::real,
            l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int)::real = begin
        ms=ms0; mm=mm0; p11=t0sd1*t0sd1; p12=tanh(t0z)*t0sd1*t0sd2; p22=t0sd2*t0sd2; ll=0.0
        for t in 1:T
            if t>1
                h=dt[t]/nsub                       # SUBSTEPPED continuous-time predict
                for st in 1:nsub
                    sp=log1p_exp(b0+bm*mm); sig=inv_logit(b0+bm*mm)
                    nms=ms+(-sp*ms+a12*mm)*h; nmm=mm+(a21*ms+a22*mm+cintm)*h
                    f11=1-h*sp; f12=h*(a12-bm*sig*ms); f21=h*a21; f22=1+h*a22
                    sds=exp(qd0+qd1*mm)            # stress sd depends on MOOD
                    corr=tanh(cz*ms)               # shock corr depends on STRESS (fisher-z)
                    qs=sds*sds*h; qc=corr*sds*sdm*h; qm=sdm*sdm*h
                    fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                    np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22+qc; np22=fp21*f21+fp22*f22+qm
                    ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
                end
            end
            # Gaussian update (2 continuous indicators, loadings [1;1])
            v1=ys[t]-ms; v2=ym[t]-mm; s11=p11+r1*r1; s12=p12; s22=p22+r2*r2
            det=s11*s22-s12*s12; si11=s22/det; si12=-s12/det; si22=s11/det
            quad=v1*(si11*v1+si12*v2)+v2*(si12*v1+si22*v2)
            ll=ll-0.5*(2*l2pi()+log(det)+quad)
            k11=p11*si11+p12*si12; k12=p11*si12+p12*si22; k21=p12*si11+p22*si12; k22=p12*si12+p22*si22
            ms=ms+k11*v1+k12*v2; mm=mm+k21*v1+k22*v2
            g11=(1-k11)*p11-k12*p12; g12=(1-k11)*p12-k12*p22; g21=-k21*p11+(1-k22)*p12; g22=-k21*p12+(1-k22)*p22
            p11=g11; p12=0.5*(g12+g21); p22=g22
            # Bernoulli-logit update (loading l31 on stress, threshold thr)
            logit=l31*ms+thr; p=inv_logit(logit)
            ll=ll+(smoked[t]*logit-log1p_exp(logit))
            pv=p*(1-p); h1=pv*l31; sb=h1*p11*h1+pv; kb1=(p11*h1)/sb; kb2=(p12*h1)/sb; resid=smoked[t]-p
            ms=ms+kb1*resid; mm=mm+kb2*resid
            b11=p11-kb1*h1*p11; b12=p12-kb1*h1*p12; b21=p12-kb2*h1*p11; b22=p22-kb2*h1*p12
            p11=b11; p12=0.5*(b12+b21); p22=b22
        end
        ll
    end
    ema_sd_lpdfs(ys::vector[T], ym::vector[T], smoked::int[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cintm::real,
            qd0::real, qd1::real, cz::real, sdm::real,
            l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int)::vector[T] = begin
        out::vector[T]; ms=ms0; mm=mm0; p11=t0sd1*t0sd1; p12=tanh(t0z)*t0sd1*t0sd2; p22=t0sd2*t0sd2
        for t in 1:T
            if t>1
                h=dt[t]/nsub
                for st in 1:nsub
                    sp=log1p_exp(b0+bm*mm); sig=inv_logit(b0+bm*mm)
                    nms=ms+(-sp*ms+a12*mm)*h; nmm=mm+(a21*ms+a22*mm+cintm)*h
                    f11=1-h*sp; f12=h*(a12-bm*sig*ms); f21=h*a21; f22=1+h*a22
                    sds=exp(qd0+qd1*mm); corr=tanh(cz*ms); qs=sds*sds*h; qc=corr*sds*sdm*h; qm=sdm*sdm*h
                    fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                    np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22+qc; np22=fp21*f21+fp22*f22+qm
                    ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
                end
            end
            v1=ys[t]-ms; v2=ym[t]-mm; s11=p11+r1*r1; s12=p12; s22=p22+r2*r2
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
    ema_sd_rng(vector[T], ym::vector[T], smoked::int[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cintm::real,
            qd0::real, qd1::real, cz::real, sdm::real,
            l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int)::vector[T] = begin
        out::vector[T]; z01=normal_rng(0.,1.); z02=normal_rng(0.,1.); r0=tanh(t0z)
        s=ms0+t0sd1*z01; m=mm0+t0sd2*(r0*z01+sqrt(1-r0*r0)*z02)
        for t in 1:T
            if t>1
                h=dt[t]/nsub
                for st in 1:nsub
                    sds=exp(qd0+qd1*m); corr=tanh(cz*s)
                    zs=normal_rng(0.,1.); zc=normal_rng(0.,1.); z2=corr*zs+sqrt(1-corr*corr)*zc
                    ds=(-log1p_exp(b0+bm*m)*s+a12*m)*h+sds*sqrt(h)*zs
                    dm=(a21*s+a22*m+cintm)*h+sdm*sqrt(h)*z2
                    s=s+ds; m=m+dm
                end
            end
            out[t]=normal_rng(s,r1)
        end
        out
    end
end

# ── synthetic panel from the generating truth (state-dependent SDE) ────────────
function fixture(; n=8, nt=15, seed=20260916)
    rng=seed; rnd()=(rng=(1103515245*rng+12345)%2^31; rng/2^31)
    randn2()=(u1=max(rnd(),1e-9); u2=rnd(); sqrt(-2*log(u1))*cos(6.283185307*u2))
    subject=String[]; stressReport=Vector{Float64}[]; moodReport=Vector{Float64}[]
    smoked=Vector{Int}[]; dt=Vector{Float64}[]
    for i in 1:n
        push!(subject,"s$i"); s=0.0+0.6*randn2(); m=0.5+0.5*randn2()   # T0 ~ N(T0MEANS, T0VAR), SD form
        a=Float64[]; b=Float64[]; c=Int[]; d=Float64[]
        for _ in 1:80                                   # ctGenerate(burnin = 10): 10 time units, h = 0.125
            hh=0.125; sds=exp(-0.2+0.3*m); corr=tanh(0.7*s)
            zs=randn2(); zc=randn2(); z2=corr*zs+sqrt(max(1-corr*corr,0.0))*zc
            ds=(-log(1+exp(0.5+0.4*m))*s-0.25*m)*hh+sds*sqrt(hh)*zs
            dm=(-0.30*s-0.60*m+0.3)*hh+0.6*sqrt(hh)*z2
            s=s+ds; m=m+dm
        end
        for t in 1:nt
            dv = t==1 ? 0.0 : exp(0.3*randn2())        # log-normal irregular intervals, median 1
            push!(d, t==1 ? 1.0 : dv)
            if t>1
                ng=8; hh=dv/ng                          # fine-grid Euler-Maruyama, matched to the EKF's nsub
                for _ in 1:ng
                    sds=exp(-0.2+0.3*m); corr=tanh(0.7*s)
                    zs=randn2(); zc=randn2(); z2=corr*zs+sqrt(max(1-corr*corr,0.0))*zc
                    ds=(-log(1+exp(0.5+0.4*m))*s-0.25*m)*hh+sds*sqrt(hh)*zs
                    dm=(-0.30*s-0.60*m+0.3)*hh+0.6*sqrt(hh)*z2
                    s=s+ds; m=m+dm
                end
            end
            push!(a, s+0.3*randn2()); push!(b, m+0.3*randn2())   # MANIFESTVAR .3 is an SD
            p=1/(1+exp(-(1.2*s-1))); push!(c, rnd()<p ? 1 : 0)
        end
        push!(stressReport,a); push!(moodReport,b); push!(smoked,c); push!(dt,d)
    end
    (; subject, stressReport, moodReport, smoked, dt)
end
data = fixture()

# ── the @brm model: multi-subject, in the KERNEL, FAITHFUL to Charles's fit
#    (`indvarying = FALSE` — ALL parameters shared, NO random effects). Each
#    subject is an independent series; with no `|ID|` bucket to derive from, the
#    kernel takes the subject COUNT from the pre-grouped `Vector{Vector}` columns'
#    common length (BRM `28e914d4`). Each subject's latent path is marginalized by
#    the state-dependent-diffusion EKF in the cell. No per-subject parameters. ────
ema_state_dependent(d) = @brm d begin
    b0    ~ Normal(0.5, 0.5)               # softplus offset
    bm    ~ Normal(0.4, 0.5)               # mood -> stress recovery modulation
    a12   ~ Normal(-0.25, 0.5)             # mood -> stress
    a21   ~ Normal(-0.30, 0.5)             # stress -> mood
    a22   ~ Normal(-0.60, 0.3)             # mood self-decay
    cintm ~ Normal(0.3, 0.5)               # cint_mood
    qd0   ~ Normal(-0.2, 0.5)              # stress log-sd offset
    qd1   ~ Normal(0.3, 0.5)               # stress volatility on MOOD (state dependent)
    cz    ~ Normal(0.7, 0.5)               # shock-correlation on STRESS (state dependent)
    sdm   ~ Exponential(1.0)              # mood diffusion sd
    l31   ~ Normal(1.2, 0.5)              # smoked loading on stress
    thr   ~ Normal(-1.0, 0.5)             # smoking threshold
    r1    ~ Exponential(1.0)              # merr_stress (an SD; squared in the filter)
    r2    ~ Exponential(1.0)              # merr_mood   (an SD)
    s0    ~ Normal(0.0, 1.0)              # T0MEANS stress
    m0    ~ Normal(0.5, 1.0)              # T0MEANS mood
    t0sd1 ~ Exponential(1.0)              # T0VAR (free in the fit): stress sd
    t0sd2 ~ Exponential(1.0)              #                          mood sd
    t0z   ~ Normal(0.0, 0.5)              #                          fisher-z correlation
    pred ~ kernel(dt, stressReport, moodReport, smoked) do dti, ys, ym, smk
        ys ~ ema_sd(ym, smk, dti, b0, bm, a12, a21, a22, cintm, qd0, qd1, cz, sdm,
                    l31, thr, r1, r2, s0, m0, t0sd1, t0sd2, t0z, 8)   # nsub=8 substeps per interval
        ys
    end
end

function main()
    sb = SBBRMI(ema_state_dependent(data); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @assert StanBlocks.stanc_check(code; warn_pedantic=false).ok "stanc failed"
    # content-hashed path so a differently-shaped model never reuses a stale .stan/.so
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_sd_$(hash(code)).stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.03*((i % 7) - 3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("fitDemo: state-dependent-diffusion EMA (multi-subject, EKF in kernel, indvarying=FALSE)")
    println("  subjects = ", length(data.subject), "  occasions = ", length(data.dt[1]))
    println("  dim = ", dim, "  lp = ", round(lp; digits=2), "  finite_grad = ", all(isfinite, g))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
