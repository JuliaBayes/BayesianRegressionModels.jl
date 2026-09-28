# test/sb_sweep_b2.jl — v1 SB-parity sweep, B2 @brm-probe driver.
#
# Run: julia --project=test test/sb_sweep_b2.jl
#
# The four @brm-viable edge cases (priors transcribed EXACTLY from the RK
# SOURCES at pin `8a51702`): rats (flat RE scales), glmm1 (implicit-uniform
# RE scale via absorbed hypermean), election88 (5 crossed RE blocks), and
# gp_pois_regr (exact-GP term match). Each case is a fail-fast probe of one
# @brm surface edge; a refusal reclassifies the case to the S driver.
# (The other B2 rows — pilots, glmm_poisson, sum_to_zero, mvnormal,
# accel_gp, accel_splines, mnist — were routed to S statically; see the
# exotic-leaf notes. kilpisjarvi + dogs_log already ran in the core driver.)
#
# Records append to $SB_SWEEP_OUT (default: tempdir()/sb-sweep-b2.jsonl).
# Crash-resume: cases already in OUT are skipped, never duplicated.

include(joinpath(@__DIR__, "sb_sweep_common.jl"))

using Distributions: Normal, Uniform, Gamma
using ReactiveKernelsPPLExamples:
    RatsModelExample, GLMM1ModelExample, Election88FullExample, GPPoisRegrExample

const OUT = get(ENV, "SB_SWEEP_OUT",
    joinpath(tempdir(), "sb-sweep-b2.jsonl"))
const _DONE = _sweep_done_cases(OUT)

# rats: BUGS growth curve (8-rat representative subset, all 5 ages).
# alpha_j ~ N(mu_alpha, sigma_alpha), beta_j ~ N(mu_beta, sigma_beta) with
# mu_* ~ N(0,100); all three scales FLAT improper. Two independent single-
# margin blocks (intercept + slope) so no LKJ enters; centered (RK samples
# alpha/beta directly). PROBE: bare `Flat()` sd spelling.
function rats_sb()
    M = RatsModelExample
    df = (; y=M.RATS_Y, rat=M.RATS_RAT, xc=M.RATS_X .- M.RATS_XBAR)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + xc + (1 | pa | rat) + (0 + xc | pb | rat)
        effect(mu, Intercept) ~ Normal(0, 100)
        effect(mu, xc) ~ Normal(0, 100)
        sd(:, pa) ~ Flat()
        sd(:, pb) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:rat])
end

# glmm1: Poisson-log GLMM, 235 site effects. The RK likelihood uses
# alpha[site] directly with alpha ~ N(mu_alpha, sd_alpha): absorbed here as
# mu = Intercept + a[site] with Intercept ~ N(0,10), which is IDENTICAL
# (Intercept ≡ mu_alpha). sd_alpha implicit-U[0,5] on the RK side is spelled
# explicit with the stated -log(5) offset.
function glmm1_sb()
    M = GLMM1ModelExample
    df = (; c=M.GLMM1_OBS, site=M.GLMM1_OBSSITE)
    builder = @brm begin
        mu ~ 1 + (1 | g1 | site)
        effect(mu, Intercept) ~ Normal(0, 10)
        sd(:, g1) ~ Uniform(0, 5)
        c ~ Poisson(exp(mu))
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:site])
end

# election88: 5 crossed zero-mean RE blocks + beta[5] fixed effects
# (beta1 = intercept), N = 11566. sd_* implicit-U[0,100] spelled explicit
# (-log(100) each). AUDIT POINT: 2 states + region 5 are unobserved — the
# run's stan_names must still size 51 + 5 (else the case moves to S).
function election88_sb()
    M = Election88FullExample
    df = (; y=M.E88_Y, black=M.E88_BLACK, female=M.E88_FEMALE,
        inter=M.E88_FEMALE .* M.E88_BLACK, v_prev=M.E88_VPREV,
        age=M.E88_AGE, edu=M.E88_EDU, age_edu=M.E88_AGE_EDU,
        state=M.E88_STATE, region=M.E88_REGION)
    builder = @brm begin
        mu ~ 1 + black + female + inter + v_prev +
            (1 | pa | age) + (1 | pb | edu) + (1 | pc | age_edu) +
            (1 | pd | state) + (1 | pe | region)
        effect(mu, Intercept) ~ Normal(0, 100)
        effect(mu, black) ~ Normal(0, 100)
        effect(mu, female) ~ Normal(0, 100)
        effect(mu, inter) ~ Normal(0, 100)
        effect(mu, v_prev) ~ Normal(0, 100)
        sd(:, pa) ~ Uniform(0, 100)
        sd(:, pb) ~ Uniform(0, 100)
        sd(:, pc) ~ Uniform(0, 100)
        sd(:, pd) ~ Uniform(0, 100)
        sd(:, pe) ~ Uniform(0, 100)
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__,
        centered_groups=[:age, :edu, :age_edu, :state, :region])
end

# gp_pois_regr: latent exact-GP Poisson regression (noncentered f = L f_tilde,
# exp-quad + 1e-10 jitter, rho ~ Gamma(25,4), alpha ~ N(0,2) UNNORMALIZED).
# PROBE: exact-`gp` term match (jitter kwarg + Gamma/Normal hyperpriors).
function gp_pois_sb()
    M = GPPoisRegrExample
    df = (; k=M.GP_POIS_K, x=M.GP_POIS_X)
    builder = @brm begin
        mu ~ gp(x; jitter=1e-10)
        length_scale(:, gp(x)) ~ Gamma(25, 4)
        sd(:, gp(x)) ~ Normal(0, 2)
        k ~ Poisson(exp(mu))
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _B2_CASES = (
    ("rats", rats_sb, 0.0, ""),
    ("glmm1", glmm1_sb, -log(5),
        "explicit Uniform(0,5) RE sd vs implicit-uniform .stan (0)"),
    ("election88", election88_sb, -5 * log(100),
        "explicit Uniform(0,100) RE sds x5 vs implicit-uniform .stan (0)"),
    ("gp_pois_regr", gp_pois_sb, 0.0, ""),
)

open(OUT, "a") do io
    for t in _B2_CASES
        (case, build, offset, reason) = t[1:4]
        if (case, "zeros") in _DONE && (case, "seeded") in _DONE
            println("SB_SWEEP skip $case (already recorded)")
            continue
        end
        recs = sweep_case(io, case, build; offset=offset, offset_reason=reason)
        for r in recs
            println("SB_SWEEP case=$(r["case"]) label=$(r["label"]) lp=$(r["lp"]) offset=$(r["offset"])")
        end
    end
end
println("SB_SWEEP b2 done -> $OUT")
