# The full ctsem EMA (Demo 1) as a HIERARCHICAL @brm model whose per-subject
# latent states are MARGINALIZED (integrated out) INSIDE the kernel cell by an
# Extended Kalman filter — a FAITHFUL translation of the `ctModelLatex` spec
# (/tmp/ctsemBrmsDemo.pdf, /tmp/ctsemREADME).
#
# This version closes the parameterization gaps of the earlier draft so the model
# matches Charles Driver's spec cell for cell:
#   * DRIFT  -log1p(exp(b0+bm*mood))*stress + a12*mood ;  a21*stress + a22*mood + cint_mood
#   * TDPREDEFFECT wl_stress*workload enters as an IMPULSE at each observation (ctsem's
#            chi(t) is a sum of Dirac deltas): stress jumps by wl_stress*workload[t] at row t.
#   * DIFFUSION (covmattransform='z'): stress sd = exp(q0+qw*tdpreds[rowi]) (input-
#            dependent, read at the CURRENT row), mood sd = diffm, and a FREE process-noise CORRELATION
#            diff21 (fisher-z) — the off-diagonal the earlier draft dropped.
#   * LAMBDA  [1;1;l31 on stress]; MANIFESTMEANS mm_stress, mm_mood, smoke_threshold
#            (the continuous manifest intercepts the earlier draft dropped).
#   * Θ       Gaussian meas. error on the two continuous (merr = SDs); the binary indicator is
#            integrated by Gauss-Hermite quadrature within the filter, as ctsem's julia engine does.
#   * Subject-varying b0, q0, cint_mood, wl_stress share ONE correlated random-
#            effect block (brms `(1|p|subject)` = ctsem's free 4x4 rawPCov), with
#            b0 regressed on age+treatment, cint_mood on treatment (the rendered beta).
#   * T0MEANS + a FREE T0VAR (sd / fisher-z form), and merr as SDs (Theta = [merr]^2):
#            ctsem covariance-type matrices are all in SD form (UcorSDtoCov).
#   * Continuous-time transition solved by a SUBSTEPPED Euler mesh over each
#            irregular Δt (NSUB substeps), not a single step.
#
# The remaining non-faithful item is deliberately out of scope here: the block-
# diagonal-from-Laplace METRIC (ctsem README point 4) is WarmupHMC/StanBlocks work,
# not a model-specification detail.
#
# VERIFIED (strato2, StanBlocks bec23bc3c523): SBBRMI -> stan_code -> stanc_check
# -> stan_instantiate + LogDensityProblems.logdensity_and_gradient, finite.
#
# Run: julia --project=test research/ema_ctsem/ema_kernel_marginalized.jl

using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential

# ── the per-subject marginalizer: a SUBSTEPPED continuous-discrete EKF @lpxf ───
# ys = stressReport (LHS); ym = moodReport; smoked; workload; dt. Between each
# observed occasion the state/covariance are propagated by NSUB Euler substeps;
# at the occasion a Gaussian update (2 continuous) and a Gauss-Hermite-integrated
# binary update run. 1.8378770664093453 = log(2pi) (a Julia const can't be @deffun'd).
StanBlocks.@deffun begin
    l2pi()::real = 1.8378770664093453
    ema_ekf_lpdfs(ys::vector[T], ym::vector[T], smoked::int[T], workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, diff21::real, l31::real, thr::real,
            mm_s::real, mm_m::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int)::vector[T] = begin
        out::vector[T]; ms=ms0; mm=mm0; p11=t0sd1*t0sd1; p12=tanh(t0z)*t0sd1*t0sd2; p22=t0sd2*t0sd2
        for t in 1:T
            if t>1
                wl=workload[t]; h=dt[t]/nsub              # DIFFUSION uses tdpreds[rowi]: the CURRENT row
                for st in 1:nsub
                    sp=log1p_exp(b0+bm*mm); sig=inv_logit(b0+bm*mm)
                    nms=ms+(-sp*ms+a12*mm)*h; nmm=mm+(a21*ms+a22*mm+cm)*h
                    f11=1-h*sp; f12=h*(a12-bm*sig*ms); f21=h*a21; f22=1+h*a22
                    sds=exp(q0+qw*wl); corr=tanh(diff21); qs=sds*sds*h; qc=corr*sds*diffm*h; qm=diffm*diffm*h
                    fp11=f11*p11+f12*p12; fp12=f11*p12+f12*p22; fp21=f21*p11+f22*p12; fp22=f21*p12+f22*p22
                    np11=fp11*f11+fp12*f12+qs; np12=fp11*f21+fp12*f22+qc; np22=fp21*f21+fp22*f22+qm
                    ms=nms; mm=nmm; p11=np11; p12=np12; p22=np22
                end
            end
            ms=ms+wls*workload[t]                    # TDPREDEFFECT = IMPULSE at the observation (ctsem)
            v1=ys[t]-(ms+mm_s); v2=ym[t]-(mm+mm_m); s11=p11+r1*r1; s12=p12; s22=p22+r2*r2
            det=s11*s22-s12*s12; si11=s22/det; si12=-s12/det; si22=s11/det
            quad=v1*(si11*v1+si12*v2)+v2*(si12*v1+si22*v2); lg=-0.5*(2*l2pi()+log(det)+quad)
            k11=p11*si11+p12*si12; k12=p11*si12+p12*si22; k21=p12*si11+p22*si12; k22=p12*si12+p22*si22
            ms=ms+k11*v1+k12*v2; mm=mm+k21*v1+k22*v2
            g11=(1-k11)*p11-k12*p12; g12=(1-k11)*p12-k12*p22; g21=-k21*p11+(1-k22)*p12; g22=-k21*p12+(1-k22)*p22
            p11=g11; p12=0.5*(g12+g21); p22=g22
            # Binary indicator: Gauss-Hermite (5 nodes) over eta ~ N(l31*stress+thr, l31^2*p11) with a
            # moment-matched state update -- as ctsem's `_binary_moments` does -- not linearised.
            etabar=l31*ms+thr; s2=l31*l31*p11+1e-12; sde=sqrt(s2); z0=0.0; z1=0.0; z2=0.0
            for i in 1:5
                xi=0.0; wi=0.5333333333333333
                if i==1; xi=-2.8569700138728056; wi=0.011257411327720689; end
                if i==2; xi=-1.3556261799742659; wi=0.22207592200561263; end
                if i==4; xi=1.3556261799742659; wi=0.22207592200561263; end
                if i==5; xi=2.8569700138728056; wi=0.011257411327720689; end
                eta=etabar+sde*xi; pr=1-inv_logit(eta)
                if smoked[t]==1; pr=inv_logit(eta); end
                z0=z0+wi*pr; z1=z1+wi*pr*eta; z2=z2+wi*pr*eta*eta
            end
            eeta=z1/z0; veta=z2/z0-eeta*eeta; kb1=p11*l31/s2; kb2=p12*l31/s2; shr=s2-veta
            ms=ms+kb1*(eeta-etabar); mm=mm+kb2*(eeta-etabar)
            n11=p11-kb1*kb1*shr; n12=p12-kb1*kb2*shr; n22=p22-kb2*kb2*shr
            p11=n11; p12=n12; p22=n22; out[t]=lg+log(z0)
        end
        out
    end
    @lhs @lpxf ema_ekf_lpdf(ys::vector[T], ym::vector[T], smoked::int[T],
            workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, diff21::real, l31::real, thr::real,
            mm_s::real, mm_m::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int)::real = begin
        sum(ema_ekf_lpdfs(ys, ym, smoked, workload, dt, b0, bm, a12, a21, a22, cm, wls, q0, qw, diffm, diff21,
                          l31, thr, mm_s, mm_m, r1, r2, ms0, mm0, t0sd1, t0sd2, t0z, nsub))
    end
    ema_ekf_rng(vector[T], ym::vector[T], smoked::int[T], workload::vector[T], dt::vector[T],
            b0::real, bm::real, a12::real, a21::real, a22::real, cm::real, wls::real,
            q0::real, qw::real, diffm::real, diff21::real, l31::real, thr::real,
            mm_s::real, mm_m::real, r1::real, r2::real,
            ms0::real, mm0::real, t0sd1::real, t0sd2::real, t0z::real, nsub::int)::vector[T] = begin
        out::vector[T]; z01=normal_rng(0.,1.); z02=normal_rng(0.,1.); r0=tanh(t0z)
        s=ms0+t0sd1*z01; m=mm0+t0sd2*(r0*z01+sqrt(1-r0*r0)*z02)
        for t in 1:T
            if t>1
                wl=workload[t]; h=dt[t]/nsub; corr=tanh(diff21)
                for st in 1:nsub
                    sds=exp(q0+qw*wl); z1=normal_rng(0.,1.); z2=corr*z1+sqrt(1-corr*corr)*normal_rng(0.,1.)
                    ds=(-log1p_exp(b0+bm*m)*s+a12*m)*h+sds*sqrt(h)*z1
                    dm=(a21*s+a22*m+cm)*h+diffm*sqrt(h)*z2
                    s=s+ds; m=m+dm
                end
            end
            s=s+wls*workload[t]                       # impulse at the observation
            out[t]=normal_rng(s+mm_s,r1)
        end
        out
    end
end

# ── synthetic EMA panel: per-subject covariates + ragged obs ──────────────────
function fixture(; n=6, nt=10, seed=1)
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
    bm    ~ Normal(0, 0.5)                  # mood -> stress softplus modulation
    a12   ~ Normal(0, 0.5)                  # mood -> stress
    a21   ~ Normal(0, 0.5)                  # stress -> mood
    a22   ~ Normal(-0.5, 0.3)               # mood self-decay
    qw    ~ Normal(0, 0.5)                  # workload -> stress process-noise
    diffm ~ Exponential(1.0)               # mood process-noise sd
    diff21 ~ Normal(0, 0.5)                 # process-noise CORRELATION (fisher-z)
    l31   ~ Normal(0, 1)                    # stress -> smoking loading
    thr   ~ Normal(0, 1)                    # smoking threshold
    mm_s  ~ Normal(0, 0.5)                  # manifest mean, stressReport
    mm_m  ~ Normal(0, 0.5)                  # manifest mean, moodReport
    r1    ~ Exponential(1.0)               # merr_stress (an SD; Theta = [merr]^2)
    r2    ~ Exponential(1.0)               # merr_mood   (an SD)
    s0    ~ Normal(0, 1)                    # T0MEANS stress (population)
    m0    ~ Normal(0, 1)                    # T0MEANS mood (population)
    t0sd1 ~ Exponential(1.0)               # T0VAR (free): stress sd
    t0sd2 ~ Exponential(1.0)               #               mood sd
    t0z   ~ Normal(0, 0.5)                  #               fisher-z correlation (T0cov_2_1)

    # the FOUR subject-varying params share ONE correlated block (ctsem rawPCov),
    # with b0 and cint_mood regressed on the covariates.
    b0  ~ 1 + age + treatment + (1 | p | subject)   # stress-drift baseline
    q0  ~ 1 +                   (1 | p | subject)    # stress process-noise baseline
    cm  ~ 1 +       treatment + (1 | p | subject)   # cint_mood: the rendered beta has raw_cint_mood_treatment only
                                                     #   (its age cell renders `raw_0`, i.e. not a named free effect)
    wls ~ 1 +                   (1 | p | subject)    # wl_stress (now per-subject)

    # per subject: integrate out THIS subject's entire latent path with the EKF
    pred ~ kernel(dt, workload, stressReport, moodReport, smoked,
                  b0, q0, cm, wls) do dti, wli, ys, ym, smk, lb0, lq0, lcm, lwls
        ys ~ ema_ekf(ym, smk, wli, dti,
                     lb0, bm, a12, a21, a22, lcm, lwls, lq0, qw, diffm, diff21, l31, thr,
                     mm_s, mm_m, r1, r2, s0, m0, t0sd1, t0sd2, t0z, 4)
        ys
    end
end

function main()
    sb = SBBRMI(ema_kernel_marginalized(data); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @assert StanBlocks.stanc_check(code; warn_pedantic=false).ok "stanc failed"
    prob = StanBlocks.stan_instantiate(sb.model; path=joinpath(tempdir(), "ema_kernel_marg_$(hash(code)).stan"))
    dim = LogDensityProblems.dimension(prob)
    q = [0.05*((i % 7) - 3) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
    println("FAITHFUL hierarchical kernel-cell EKF-marginalized EMA (Demo 1)")
    println("  subjects = ", length(data.subject), "  occasions = ", length(data.dt[1]))
    println("  dim = ", dim, "  lp = ", round(lp; digits=2), "  finite_grad = ", all(isfinite, g))
    println("  correlated |p| block: ", occursin("lkj_corr_cholesky", code) || occursin("multi_normal_cholesky", code))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
