# A continuous-time state-space model whose dynamics depend on the LATENT STATE, with the
# states marginalized inside the `kernel(...)` cell.
#
# The model follows a demonstration model of ctsem (Charles Driver's R package for
# hierarchical continuous-time dynamic modelling, https://github.com/cdriveraus/ctsem).
# Three cells of the system matrices are functions of the latent state, not of an input:
#   1. DRIFT[1,1]     = -softplus(b0 + bm*mood)      stress recovers faster in good mood
#   2. DIFFUSION[1,1] = exp(qd0 + qd1*mood)          stress is MORE VOLATILE in good mood
#   3. DIFFUSION[2,1] = tanh(cz*stress)              the two shocks couple more tightly
#                                                    the more stressed (a fisher-z correlation)
# So the process-noise covariance depends on the very states being integrated out.
#
# Generating values (ctsem matrix names; covariance-type cells are in sd / fisher-z form,
# i.e. the variance is the cell SQUARED):
#   DRIFT     = [ -softplus(0.5 + 0.4*mood)  -0.25 ;  -0.30  -0.60 ]
#   DIFFUSION = [ exp(-0.2 + 0.3*mood)  0 ;  0.7*stress  0.6 ]
#   CINT      = [0 ; 0.3]
#   LAMBDA    = [1 0 ; 0 1 ; 1.2 0]        the binary indicator loads 1.2 on stress
#   MANIFESTMEANS = [0 ; 0 ; -1]           binary threshold -1
#   MANIFESTVAR   = diag(.3, .3, 0)        measurement sd 0.3
#   T0MEANS = [0 ; 0.5]   T0VAR = diag(.6, .5)
# All parameters are shared across subjects: no covariates, no random effects.
#
# Run: julia --project=test research/ema_ctsem/ema_state_dependent.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

# The per-subject marginalizer, a custom `@lpxf` family: a substepped continuous-discrete
# Gaussian filter with STATE-DEPENDENT, CORRELATED diffusion. `ys` (stressReport) is the
# left-hand side of `ys ~ ema_sd(...)`; ym = moodReport, smoked = binary, dt = intervals.
# Its PRECISION is two integers, both DATA (so one compiled model serves every setting --
# see `with_filter`):
#   nsub    Euler substeps per observation interval.
#   gh = 0  first-order predict (EKF): drift Jacobian + Q evaluated PLUG-IN at the filtered
#           mean -- the order ctsem's filter uses.
#   gh = K  (3 or 5) MOMENT-MATCHED predict: the Euler map and Q(x) are averaged over the
#           state uncertainty N(m,P) with a KxK Gauss-Hermite rule. The first-order filter
#           uses tanh(cz*E[stress]) where the process carries E[tanh(cz*stress)] (attenuated
#           toward 0); moment matching integrates that instead of plugging in.
# The binary indicator is integrated by Gauss-Hermite quadrature over the latent predictor
# (5 nodes) with a moment-matched state update -- as ctsem's `_binary_moments` does -- not
# linearised.
StanBlocks.@deffun begin
    l2pi()::real = 1.8378770664093453          # log(2 pi)
    # probabilists' Gauss-Hermite node / weight i of a K-point rule (K = 3 or 5)
    ghx(i::int, K::int)::real = begin
        x=0.0
        if K==3; x=(i-2)*1.7320508075688772; end
        if K==5
            if i==1; x=-2.8569700138728056; end
            if i==2; x=-1.3556261799742659; end
            if i==4; x=1.3556261799742659; end
            if i==5; x=2.8569700138728056; end
        end
        x
    end
    ghw(i::int, K::int)::real = begin
        w=0.6666666666666666
        if K==3
            if i!=2; w=0.16666666666666666; end
        end
        if K==5
            w=0.5333333333333333
            if i==1 || i==5; w=0.011257411327720689; end
            if i==2 || i==4; w=0.22207592200561263; end
        end
        w
    end
    ema_sd_lpdfs(ys::vector[T], ym::vector[T], smoked::int[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cintm::real,
            qd0::real, qd1::real, cz::real, sdm::real,
            l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int, gh::int)::vector[T] = begin
        out::vector[T]; ms=ms0; mm=mm0; p11=t0sd1*t0sd1; p12=tanh(t0z)*t0sd1*t0sd2; p22=t0sd2*t0sd2
        for t in 1:T
            if t>1
                h=dt[t]/nsub                       # SUBSTEPPED continuous-time predict
                for st in 1:nsub
                    if gh>0
                        l11=sqrt(p11); l21=p12/l11; l22=sqrt(p22-l21*l21+1e-12)
                        ey1=0.0; ey2=0.0; c11=0.0; c12=0.0; c22=0.0; q11=0.0; q12=0.0
                        for a in 1:gh
                            xa=ghx(a,gh)
                            for b in 1:gh
                                xb=ghx(b,gh); w=ghw(a,gh)*ghw(b,gh)
                                xs=ms+l11*xa; xm=mm+l21*xa+l22*xb
                                y1=xs+h*(-log1p_exp(b0+bm*xm)*xs+a12*xm)
                                y2=xm+h*(a21*xs+a22*xm+cintm)
                                sdx=exp(qd0+qd1*xm)            # stress sd depends on MOOD
                                ey1=ey1+w*y1; ey2=ey2+w*y2
                                c11=c11+w*y1*y1; c12=c12+w*y1*y2; c22=c22+w*y2*y2
                                q11=q11+w*sdx*sdx
                                q12=q12+w*tanh(cz*xs)*sdx*sdm  # shock corr depends on STRESS
                            end
                        end
                        p11=c11-ey1*ey1+h*q11; p12=c12-ey1*ey2+h*q12; p22=c22-ey2*ey2+h*sdm*sdm
                        ms=ey1; mm=ey2
                    else
                        sp=log1p_exp(b0+bm*mm); sig=inv_logit(b0+bm*mm)
                        nms=ms+(-sp*ms+a12*mm)*h; nmm=mm+(a21*ms+a22*mm+cintm)*h
                        f11=1-h*sp; f12=h*(a12-bm*sig*ms); f21=h*a21; f22=1+h*a22
                        sds=exp(qd0+qd1*mm); corr=tanh(cz*ms)
                        qs=sds*sds*h; qc=corr*sds*sdm*h; qm=sdm*sdm*h
                        fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                        np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22+qc; np22=fp21*f21+fp22*f22+qm
                        ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
                    end
                end
            end
            # Gaussian update (2 continuous indicators, loadings [1;1]); r1, r2 are measurement sds
            v1=ys[t]-ms; v2=ym[t]-mm; s11=p11+r1*r1; s12=p12; s22=p22+r2*r2
            det=s11*s22-s12*s12; si11=s22/det; si12=-s12/det; si22=s11/det
            quad=v1*(si11*v1+si12*v2)+v2*(si12*v1+si22*v2); lg=-0.5*(2*l2pi()+log(det)+quad)
            k11=p11*si11+p12*si12; k12=p11*si12+p12*si22; k21=p12*si11+p22*si12; k22=p12*si12+p22*si22
            ms=ms+k11*v1+k12*v2; mm=mm+k21*v1+k22*v2
            g11=(1-k11)*p11-k12*p12; g12=(1-k11)*p12-k12*p22; g21=-k21*p11+(1-k22)*p12; g22=-k21*p12+(1-k22)*p22
            p11=g11; p12=0.5*(g12+g21); p22=g22
            # Binary indicator: Gauss-Hermite (5 nodes) over eta ~ N(l31*stress+thr, l31^2*p11)
            etabar=l31*ms+thr; s2=l31*l31*p11+1e-12; sde=sqrt(s2); z0=0.0; z1=0.0; z2=0.0
            for i in 1:5
                xi=ghx(i,5); wi=ghw(i,5)
                eta=etabar+sde*xi; pr=1-inv_logit(eta)
                if smoked[t]==1; pr=inv_logit(eta); end
                z0=z0+wi*pr; z1=z1+wi*pr*eta; z2=z2+wi*pr*eta*eta
            end
            eeta=z1/z0; veta=z2/z0-eeta*eeta; kb1=p11*l31/s2; kb2=p12*l31/s2; shr=s2-veta
            ms=ms+kb1*(eeta-etabar); mm=mm+kb2*(eeta-etabar)
            n11=p11-kb1*kb1*shr; n12=p12-kb1*kb2*shr; n22=p22-kb2*kb2*shr
            p11=n11; p12=n12; p22=n22
            out[t]=lg+log(z0)
        end
        out
    end
    @lhs @lpxf ema_sd_lpdf(ys::vector[T], ym::vector[T], smoked::int[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cintm::real,
            qd0::real, qd1::real, cz::real, sdm::real,
            l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int, gh::int)::real = begin
        sum(ema_sd_lpdfs(ys, ym, smoked, dt, b0, bm, a12, a21, a22, cintm, qd0, qd1, cz, sdm, l31, thr, r1, r2, ms0, mm0, t0sd1, t0sd2, t0z, nsub, gh))
    end
    ema_sd_rng(vector[T], ym::vector[T], smoked::int[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cintm::real,
            qd0::real, qd1::real, cz::real, sdm::real,
            l31::real, thr::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int, gh::int)::vector[T] = begin
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

"""
Synthetic panel drawn from the generating values above. `ng`: Euler-Maruyama steps per
observation interval in the generator (finer = closer to the SDE). `rng`: an `AbstractRNG`
to draw from; the default is a dependency-free LCG + Box-Muller stream.
"""
function ema_state_dependent_fixture(; n=8, nt=15, seed=20260916, ng=8, rng=nothing)
    state=seed; lcg()=(state=(1103515245*state+12345)%2^31; state/2^31)
    rnd()=rng === nothing ? lcg() : rand(rng)
    randn2()=rng === nothing ? (u1=max(lcg(),1e-9); u2=lcg(); sqrt(-2*log(u1))*cos(6.283185307*u2)) : randn(rng)
    subject=String[]; stressReport=Vector{Float64}[]; moodReport=Vector{Float64}[]
    smoked=Vector{Int}[]; dt=Vector{Float64}[]
    for i in 1:n
        push!(subject,"s$i"); s=0.0+0.6*randn2(); m=0.5+0.5*randn2()   # initial state ~ N(T0MEANS, T0VAR)
        a=Float64[]; b=Float64[]; c=Int[]; d=Float64[]
        for _ in 1:80                                   # burn-in: 10 time units at h = 0.125
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
                hh=dv/ng                                # fine-grid Euler-Maruyama
                for _ in 1:ng
                    sds=exp(-0.2+0.3*m); corr=tanh(0.7*s)
                    zs=randn2(); zc=randn2(); z2=corr*zs+sqrt(max(1-corr*corr,0.0))*zc
                    ds=(-log(1+exp(0.5+0.4*m))*s-0.25*m)*hh+sds*sqrt(hh)*zs
                    dm=(-0.30*s-0.60*m+0.3)*hh+0.6*sqrt(hh)*z2
                    s=s+ds; m=m+dm
                end
            end
            push!(a, s+0.3*randn2()); push!(b, m+0.3*randn2())   # measurement sd 0.3
            p=1/(1+exp(-(1.2*s-1))); push!(c, rnd()<p ? 1 : 0)
        end
        push!(stressReport,a); push!(moodReport,b); push!(smoked,c); push!(dt,d)
    end
    (; subject, stressReport, moodReport, smoked, dt)
end

# The filter's precision rides in the DATA (scalar ints read inside the kernel cell), so the
# emitted Stan -- and the compiled model -- is the same for every setting, and draws from one
# setting are valid points for another. That is what lets a cheap-filter posterior be
# importance-weighted toward a more precise one (ema_state_dependent_psis.jl).
#   gh = 0: first-order plug-in;  gh = 3 / 5: moment-matched predict.
with_filter(d; nsub=8, gh=3) = merge(d, (; nsub, gh))

# Every subject is an independent series sharing all parameters. With no random-effect
# grouping to derive it from, the kernel takes the subject count from the common length of
# the pre-grouped (vector-of-vectors) columns.
function ema_state_dependent_model(data = with_filter(ema_state_dependent_fixture()))
    @brm data begin
        b0    ~ Normal(0.5, 0.5)               # stress recovery: softplus offset
        bm    ~ Normal(0.4, 0.5)               #                  modulation by mood
        a12   ~ Normal(-0.25, 0.5)             # mood -> stress
        a21   ~ Normal(-0.30, 0.5)             # stress -> mood
        a22   ~ Normal(-0.60, 0.3)             # mood self-decay
        cintm ~ Normal(0.3, 0.5)               # mood intercept
        qd0   ~ Normal(-0.2, 0.5)              # stress log-sd: offset
        qd1   ~ Normal(0.3, 0.5)               #                dependence on MOOD
        cz    ~ Normal(0.7, 0.5)               # shock correlation: dependence on STRESS
        sdm   ~ Exponential(1.0)               # mood diffusion sd
        l31   ~ Normal(1.2, 0.5)               # binary indicator: loading on stress
        thr   ~ Normal(-1.0, 0.5)              #                   threshold
        r1    ~ Exponential(1.0)               # measurement sd, stressReport
        r2    ~ Exponential(1.0)               # measurement sd, moodReport
        s0    ~ Normal(0.0, 1.0)               # initial stress mean
        m0    ~ Normal(0.5, 1.0)               # initial mood mean
        t0sd1 ~ Exponential(1.0)               # initial covariance: stress sd
        t0sd2 ~ Exponential(1.0)               #                     mood sd
        t0z   ~ Normal(0.0, 0.5)               #                     fisher-z correlation
        pred ~ kernel(dt, stressReport, moodReport, smoked) do dti, ys, ym, smk
            ys ~ ema_sd(ym, smk, dti, b0, bm, a12, a21, a22, cintm, qd0, qd1, cz, sdm,
                        l31, thr, r1, r2, s0, m0, t0sd1, t0sd2, t0z, nsub, gh)   # filter precision: DATA
            ys
        end
    end
end

function main()
    data = with_filter(ema_state_dependent_fixture())
    sb = SBBRMI(ema_state_dependent_model(data); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @assert StanBlocks.stanc_check(code; warn_pedantic=false).ok "stanc failed"
    # content-hashed path so a differently-shaped model never reuses a stale .stan/.so
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_sd_$(hash(code)).stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.03*((i % 7) - 3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("state-dependent-diffusion model, latent states marginalized in the kernel cell")
    println("  subjects = ", length(data.subject), "  occasions = ", length(data.dt[1]))
    println("  dim = ", dim, "  lp = ", round(lp; digits=2), "  finite_grad = ", all(isfinite, g))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
