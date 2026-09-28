# test/sb_sweep_core.jl — v1 SB-parity sweep, CORE families driver.
#
# Run: julia --project=test test/sb_sweep_core.jl
#
# @brm counterparts for the directly-expressible inventory slice, with priors
# transcribed EXACTLY (constants included) from the RK SOURCES at pin
# `8a51702`. Each case compiles via BridgeStan and records the unconstrained
# posterior at the two sweep-defined parity points. See sb_sweep_common.jl.
#
# Records append to $SB_SWEEP_OUT (default: tempdir()/sb-sweep-records.jsonl).

include(joinpath(@__DIR__, "sb_sweep_common.jl"))

using Distributions: Normal, Cauchy, Uniform, truncated, Beta, Gamma, TDist, LocationScale, Binomial
using Statistics: mean, std
using ReactiveKernelsPPLExamples:
    RadonCountyExample, RadonCountyInterceptExample,
    RadonHierarchicalInterceptCenteredExample,
    RadonHierarchicalInterceptNoncenteredExample,
    RadonPartiallyPooledCenteredExample, RadonPartiallyPooledNoncenteredExample,
    RadonVariableInterceptCenteredExample,
    RadonVariableInterceptNoncenteredExample,
    RadonVariableInterceptSlopeCenteredExample,
    RadonVariableInterceptSlopeNoncenteredExample,
    RadonVariableSlopeCenteredExample, RadonVariableSlopeNoncenteredExample,
    KidscoreInteractionExample, KidscoreInteractionCExample,
    KidscoreInteractionC2Example, KidscoreInteractionZExample,
    KidscoreMomWorkExample, KidscoreMomhsExample, KidscoreMomhsiqExample,
    KidscoreMomiqExample, WellsDaaeCExample, WellsDaeCExample,
    WellsDaeInterExample, WellsDaeExample, WellsDistExample,
    WellsDist100Example, WellsDist100arsExample, WellsInteractionCExample,
    WellsInteractionExample, Log10earnHeightExample, LogearnHeightExample,
    LogearnHeightMaleExample, LogearnInteractionExample,
    LogearnInteractionZExample, LogearnLogheightMaleExample, MesquiteExample,
    LogmesquiteExample, LogmesquiteLogvaExample, LogmesquiteLogvasExample,
    LogmesquiteLogvashExample, LogmesquiteLogvolumeExample, DogsExample,
    DogsLogExample, EightSchoolsExample, EightSchoolsNoncenteredExample,
    Rate1Example, Rate2Example, Rate3Example, Rate4Example, Rate5Example,
    LinearRegressionExample, BoundRegressionExample, BetaBinomialExample,
    PoissonGammaExample, GLMBinomialExample, GLMPoissonExample, NESExample,
    NesLogitExample, DiamondsExample, KilpisjarviExample,
    SesameOnePredAExample, ARKExample, SeedsStanifiedExample

const OUT = get(ENV, "SB_SWEEP_OUT",
    joinpath(tempdir(), "sb-sweep-records.jsonl"))

# radon_mn-radon_county: mu_a ~ N(0,1); a_j ~ N(mu_a, sigma_a);
# sigma_a/sigma_y implicit-uniform[0,100] (0 contribution). SB spells the
# bounds explicitly (-log(100) each): stated offset -2*log(100).
function radon_county_sb()
    M = RadonCountyExample
    df = (; y=M.RADON_COUNTY_LOG, county=M.RADON_COUNTY_IDX)
    builder = @brm begin
        sigma ~ Uniform(0, 100)
        mu ~ 1 + (1 | rc | county)
        effect(mu, Intercept) ~ Normal(0, 1)
        sd(:, rc) ~ Uniform(0, 100)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# radon_mn-radon_county_intercept: per-county FIXED alpha_j ~ N(0,10) (no
# hyperprior); beta ~ N(0,10); sigma_y ~ N(0,1) plain.
function radon_county_intercept_sb()
    M = RadonCountyInterceptExample
    df = (; y=M.RADON_CI_LOG, county=M.RADON_CI_COUNTY, floor=M.RADON_CI_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 0 + county + floor
        effect(mu, county) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# radon_mn-radon_hierarchical_intercept_centered: alpha_j ~ N(mu_alpha,
# sigma_alpha) centered; mu_alpha/beta[1:2] ~ N(0,10); sigmas ~ N(0,1) plain.
function radon_hic_sb()
    M = RadonHierarchicalInterceptCenteredExample
    df = (; y=M.RADON_HIC_LOG, county=M.RADON_HIC_COUNTY,
        log_uppm=M.RADON_HIC_UPPM, floor=M.RADON_HIC_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + log_uppm + floor + (1 | hic | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, log_uppm) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, hic) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:county])
end

function radon_hin_sb()
    M = RadonHierarchicalInterceptNoncenteredExample
    df = (; y=M.RADON_HIN_LOG, county=M.RADON_HIN_COUNTY,
        log_uppm=M.RADON_HIN_UPPM, floor=M.RADON_HIN_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + log_uppm + floor + (1 | hin | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, log_uppm) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, hin) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# radon_mn-radon_partially_pooled_{centered,noncentered}: mu_alpha ~ N(0,10);
# sigmas ~ N(0,1) plain; alpha_j ~ N(mu_alpha, sigma_alpha).
function radon_ppc_sb()
    M = RadonPartiallyPooledCenteredExample
    df = (; y=M.RADON_PP_LOG, county=M.RADON_PP_COUNTY)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + (1 | ppc | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        sd(:, ppc) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:county])
end

function radon_ppn_sb()
    M = RadonPartiallyPooledNoncenteredExample
    df = (; y=M.RADON_PP_LOG, county=M.RADON_PP_COUNTY)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + (1 | ppn | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        sd(:, ppn) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# radon_mn-radon_variable_intercept_{centered,noncentered}: + shared floor
# slope beta ~ N(0,10).
function radon_vic_sb()
    M = RadonVariableInterceptCenteredExample
    df = (; y=M.RADON_VI_LOG, county=M.RADON_VI_COUNTY, floor=M.RADON_VI_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + floor + (1 | vic | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, vic) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:county])
end

function radon_vin_sb()
    M = RadonVariableInterceptNoncenteredExample
    df = (; y=M.RADON_VIN_LOG, county=M.RADON_VIN_COUNTY, floor=M.RADON_VIN_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + floor + (1 | vin | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, vin) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# radon_mn-radon_variable_intercept_slope_{centered,noncentered}: INDEPENDENT
# intercept + slope REs; mu_alpha/mu_beta ~ N(0,10); 3 sigmas ~ N(0,1).
# Population intercept/floor-slope carry the hyper-means.
function radon_visc_sb()
    M = RadonVariableInterceptSlopeCenteredExample
    df = (; y=M.RADON_VISC_LOG, county=M.RADON_VISC_COUNTY,
        floor=M.RADON_VISC_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + floor + (1 | via | county) + (0 + floor | vib | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, via) ~ Normal(0, 1)
        sd(:, vib) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:county])
end

function radon_visn_sb()
    M = RadonVariableInterceptSlopeNoncenteredExample
    df = (; y=M.RADON_VISN_LOG, county=M.RADON_VISN_COUNTY,
        floor=M.RADON_VISN_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + floor + (1 | vja | county) + (0 + floor | vjb | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, vja) ~ Normal(0, 1)
        sd(:, vjb) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# radon_mn-radon_variable_slope_{centered,noncentered}: shared intercept
# alpha ~ N(0,10) + per-county slope (hyper-mean mu_beta ~ N(0,10)).
function radon_vsc_sb()
    M = RadonVariableSlopeCenteredExample
    df = (; y=M.RADON_VSC_LOG, county=M.RADON_VSC_COUNTY, floor=M.RADON_VSC_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + floor + (0 + floor | vsc | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, vsc) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:county])
end

function radon_vsn_sb()
    M = RadonVariableSlopeNoncenteredExample
    df = (; y=M.RADON_VSN_LOG, county=M.RADON_VSN_COUNTY, floor=M.RADON_VSN_FLOOR)
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + floor + (0 + floor | vsn | county)
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        sd(:, vsn) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _RADON_CASES = (
    ("radon_county", radon_county_sb, -2 * log(100),
        "explicit Uniform(0,100) on sigma + RE sd vs implicit-uniform .stan (0); -log(100) each"),
    ("radon_county_intercept", radon_county_intercept_sb, 0.0, ""),
    ("radon_hierarchical_intercept_centered", radon_hic_sb, 0.0, ""),
    ("radon_hierarchical_intercept_noncentered", radon_hin_sb, 0.0, ""),
    ("radon_partially_pooled_centered", radon_ppc_sb, 0.0, ""),
    ("radon_partially_pooled_noncentered", radon_ppn_sb, 0.0, ""),
    ("radon_variable_intercept_centered", radon_vic_sb, 0.0, ""),
    ("radon_variable_intercept_noncentered", radon_vin_sb, 0.0, ""),
    ("radon_variable_intercept_slope_centered", radon_visc_sb, 0.0, ""),
    ("radon_variable_intercept_slope_noncentered", radon_visn_sb, 0.0, ""),
    ("radon_variable_slope_centered", radon_vsc_sb, 0.0, ""),
    ("radon_variable_slope_noncentered", radon_vsn_sb, 0.0, ""),
)

function _run_batch(cases)
    open(OUT, "a") do io
        for t in cases
            (case, build, offset, reason) = t[1:4]
            seeded = length(t) >= 5 ? t[5] : nothing
            recs = sweep_case(io, case, build; offset=offset, offset_reason=reason,
                seeded_q=seeded)
            for r in recs
                println("SB_SWEEP case=$(r["case"]) label=$(r["label"]) lp=$(r["lp"]) offset=$(r["offset"])")
            end
        end
    end
end

_run_batch(_RADON_CASES)
println("SB_SWEEP core/radon done -> $OUT")

# ---- kidscore (8): ARM Gaussian regressions; transforms replicated host-side.
# interaction/momhs/momhsiq/momiq: beta flat, sigma ~ Cauchy(0,2.5) plain;
# _c/_c2/_z/mom_work: fully flat.

function kidscore_interaction_sb()
    M = KidscoreInteractionExample
    hs, iq = M.INTERACTION_MOM_HS, M.INTERACTION_MOM_IQ
    df = (; y=M.INTERACTION_KID_SCORE, hs, iq, hsiq=hs .* iq)
    builder = @brm begin
        sigma ~ Cauchy(0, 2.5; lower=0.0)
        mu ~ 1 + hs + iq + hsiq
        effect(mu, Intercept) ~ Flat()
        effect(mu, hs) ~ Flat()
        effect(mu, iq) ~ Flat()
        effect(mu, hsiq) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kidscore_interaction_c_sb()
    M = KidscoreInteractionCExample
    hs, iq = M.INTERACTION_C_MOM_HS, M.INTERACTION_C_MOM_IQ
    chs, ciq = hs .- mean(hs), iq .- mean(iq)
    df = (; y=M.INTERACTION_C_KID_SCORE, chs, ciq, chsiq=chs .* ciq)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + chs + ciq + chsiq
        effect(mu, Intercept) ~ Flat()
        effect(mu, chs) ~ Flat()
        effect(mu, ciq) ~ Flat()
        effect(mu, chsiq) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kidscore_interaction_c2_sb()
    M = KidscoreInteractionC2Example
    hs, iq = M.INTERACTION_C2_MOM_HS, M.INTERACTION_C2_MOM_IQ
    chs, ciq = hs .- 0.5, iq .- 100.0
    df = (; y=M.INTERACTION_C2_KID_SCORE, chs, ciq, chsiq=chs .* ciq)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + chs + ciq + chsiq
        effect(mu, Intercept) ~ Flat()
        effect(mu, chs) ~ Flat()
        effect(mu, ciq) ~ Flat()
        effect(mu, chsiq) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kidscore_interaction_z_sb()
    M = KidscoreInteractionZExample
    hs, iq = M.INTERACTION_Z_MOM_HS, M.INTERACTION_Z_MOM_IQ
    zhs = (hs .- mean(hs)) ./ (2 * std(hs))
    ziq = (iq .- mean(iq)) ./ (2 * std(iq))
    df = (; y=M.INTERACTION_Z_KID_SCORE, zhs, ziq, zhsiq=zhs .* ziq)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + zhs + ziq + zhsiq
        effect(mu, Intercept) ~ Flat()
        effect(mu, zhs) ~ Flat()
        effect(mu, ziq) ~ Flat()
        effect(mu, zhsiq) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kidscore_mom_work_sb()
    M = KidscoreMomWorkExample
    df = (; y=M.MOM_WORK_KID_SCORE, w2=M.MOM_WORK_WORK2,
        w3=M.MOM_WORK_WORK3, w4=M.MOM_WORK_WORK4)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + w2 + w3 + w4
        effect(mu, Intercept) ~ Flat()
        effect(mu, w2) ~ Flat()
        effect(mu, w3) ~ Flat()
        effect(mu, w4) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kidscore_momhs_sb()
    M = KidscoreMomhsExample
    df = (; y=M.MOMHS_KID_SCORE, hs=M.MOMHS_MOM_HS)
    builder = @brm begin
        sigma ~ Cauchy(0, 2.5; lower=0.0)
        mu ~ 1 + hs
        effect(mu, Intercept) ~ Flat()
        effect(mu, hs) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kidscore_momhsiq_sb()
    M = KidscoreMomhsiqExample
    df = (; y=M.MOMHSIQ_KID_SCORE, hs=M.MOMHSIQ_MOM_HS, iq=M.MOMHSIQ_MOM_IQ)
    builder = @brm begin
        sigma ~ Cauchy(0, 2.5; lower=0.0)
        mu ~ 1 + hs + iq
        effect(mu, Intercept) ~ Flat()
        effect(mu, hs) ~ Flat()
        effect(mu, iq) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kidscore_momiq_sb()
    M = KidscoreMomiqExample
    df = (; y=M.MOMIQ_KID_SCORE, iq=M.MOMIQ_MOM_IQ)
    builder = @brm begin
        sigma ~ Cauchy(0, 2.5; lower=0.0)
        mu ~ 1 + iq
        effect(mu, Intercept) ~ Flat()
        effect(mu, iq) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _KIDSCORE_CASES = (
    ("kidscore_interaction", kidscore_interaction_sb, 0.0, ""),
    ("kidscore_interaction_c", kidscore_interaction_c_sb, 0.0, ""),
    ("kidscore_interaction_c2", kidscore_interaction_c2_sb, 0.0, ""),
    ("kidscore_interaction_z", kidscore_interaction_z_sb, 0.0, ""),
    ("kidscore_mom_work", kidscore_mom_work_sb, 0.0, ""),
    ("kidscore_momhs", kidscore_momhs_sb, 0.0, ""),
    ("kidscore_momhsiq", kidscore_momhsiq_sb, 0.0, ""),
    ("kidscore_momiq", kidscore_momiq_sb, 0.0, ""),
)

_run_batch(_KIDSCORE_CASES)
println("SB_SWEEP core/kidscore done -> $OUT")

# ---- wells (9): Bernoulli-logit GLMs, all flat. Rescales/centering host-side.

function wells_daae_c_sb()
    M = WellsDaaeCExample
    dist, ars = M.WELLS_DAAE_C_DIST, M.WELLS_DAAE_C_ARSENIC
    cdist = (dist .- mean(dist)) ./ 100.0
    cars = ars .- mean(ars)
    educ = M.WELLS_DAAE_C_EDUC ./ 4
    df = (; y=M.WELLS_DAAE_C_SWITCHED, cdist, cars,
        inter=cdist .* cars, assoc=M.WELLS_DAAE_C_ASSOC, educ)
    builder = @brm begin
        mu ~ 1 + cdist + cars + inter + assoc + educ
        effect(mu, Intercept) ~ Flat()
        effect(mu, cdist) ~ Flat()
        effect(mu, cars) ~ Flat()
        effect(mu, inter) ~ Flat()
        effect(mu, assoc) ~ Flat()
        effect(mu, educ) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_dae_c_sb()
    M = WellsDaeCExample
    dist, ars = M.WELLS_DAE_C_DIST, M.WELLS_DAE_C_ARSENIC
    cdist = (dist .- mean(dist)) ./ 100.0
    cars = ars .- mean(ars)
    educ = M.WELLS_DAE_C_EDUC ./ 4
    df = (; y=M.WELLS_DAE_C_SWITCHED, cdist, cars,
        inter=cdist .* cars, educ)
    builder = @brm begin
        mu ~ 1 + cdist + cars + inter + educ
        effect(mu, Intercept) ~ Flat()
        effect(mu, cdist) ~ Flat()
        effect(mu, cars) ~ Flat()
        effect(mu, inter) ~ Flat()
        effect(mu, educ) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_dae_inter_sb()
    M = WellsDaeInterExample
    dist, ars, educ = M.WELLS_DAE_INTER_DIST, M.WELLS_DAE_INTER_ARSENIC,
        M.WELLS_DAE_INTER_EDUC
    cdist = (dist .- mean(dist)) ./ 100.0
    cars = ars .- mean(ars)
    ceduc = (educ .- mean(educ)) ./ 4.0
    df = (; y=M.WELLS_DAE_INTER_SWITCHED, cdist, cars, ceduc,
        i_da=cdist .* cars, i_de=cdist .* ceduc, i_ae=cars .* ceduc)
    builder = @brm begin
        mu ~ 1 + cdist + cars + ceduc + i_da + i_de + i_ae
        effect(mu, Intercept) ~ Flat()
        effect(mu, cdist) ~ Flat()
        effect(mu, cars) ~ Flat()
        effect(mu, ceduc) ~ Flat()
        effect(mu, i_da) ~ Flat()
        effect(mu, i_de) ~ Flat()
        effect(mu, i_ae) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_dae_sb()
    M = WellsDaeExample
    df = (; y=M.WELLS_DAE_SWITCHED, dist100=M.WELLS_DAE_DIST ./ 100,
        arsenic=M.WELLS_DAE_ARSENIC, educ4=M.WELLS_DAE_EDUC ./ 4)
    builder = @brm begin
        mu ~ 1 + dist100 + arsenic + educ4
        effect(mu, Intercept) ~ Flat()
        effect(mu, dist100) ~ Flat()
        effect(mu, arsenic) ~ Flat()
        effect(mu, educ4) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_dist_sb()
    M = WellsDistExample
    df = (; y=M.WELLS_DIST_SWITCHED, dist=M.WELLS_DIST_DIST)
    builder = @brm begin
        mu ~ 1 + dist
        effect(mu, Intercept) ~ Flat()
        effect(mu, dist) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_dist100_sb()
    M = WellsDist100Example
    df = (; y=M.WELLS_DIST100_SWITCHED, dist100=M.WELLS_DIST100_DIST ./ 100)
    builder = @brm begin
        mu ~ 1 + dist100
        effect(mu, Intercept) ~ Flat()
        effect(mu, dist100) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_dist100ars_sb()
    M = WellsDist100arsExample
    df = (; y=M.WELLS_DIST100ARS_SWITCHED,
        dist100=M.WELLS_DIST100ARS_DIST ./ 100,
        arsenic=M.WELLS_DIST100ARS_ARSENIC)
    builder = @brm begin
        mu ~ 1 + dist100 + arsenic
        effect(mu, Intercept) ~ Flat()
        effect(mu, dist100) ~ Flat()
        effect(mu, arsenic) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_interaction_c_sb()
    M = WellsInteractionCExample
    dist, ars = M.WELLS_INTERACTION_C_DIST, M.WELLS_INTERACTION_C_ARSENIC
    cdist = (dist .- mean(dist)) ./ 100.0
    cars = ars .- mean(ars)
    df = (; y=M.WELLS_INTERACTION_C_SWITCHED, cdist, cars,
        inter=cdist .* cars)
    builder = @brm begin
        mu ~ 1 + cdist + cars + inter
        effect(mu, Intercept) ~ Flat()
        effect(mu, cdist) ~ Flat()
        effect(mu, cars) ~ Flat()
        effect(mu, inter) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function wells_interaction_sb()
    M = WellsInteractionExample
    dist100 = M.WELLS_INTERACTION_DIST ./ 100
    df = (; y=M.WELLS_INTERACTION_SWITCHED, dist100,
        arsenic=M.WELLS_INTERACTION_ARSENIC,
        inter=dist100 .* M.WELLS_INTERACTION_ARSENIC)
    builder = @brm begin
        mu ~ 1 + dist100 + arsenic + inter
        effect(mu, Intercept) ~ Flat()
        effect(mu, dist100) ~ Flat()
        effect(mu, arsenic) ~ Flat()
        effect(mu, inter) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _WELLS_CASES = (
    ("wells_daae_c_model", wells_daae_c_sb, 0.0, ""),
    ("wells_dae_c_model", wells_dae_c_sb, 0.0, ""),
    ("wells_dae_inter_model", wells_dae_inter_sb, 0.0, ""),
    ("wells_dae_model", wells_dae_sb, 0.0, ""),
    ("wells_dist", wells_dist_sb, 0.0, ""),
    ("wells_dist100_model", wells_dist100_sb, 0.0, ""),
    ("wells_dist100ars_model", wells_dist100ars_sb, 0.0, ""),
    ("wells_interaction_c_model", wells_interaction_c_sb, 0.0, ""),
    ("wells_interaction_model", wells_interaction_sb, 0.0, ""),
)

_run_batch(_WELLS_CASES)
println("SB_SWEEP core/wells done -> $OUT")

# ---- logearn/earn (6; earn_height rides the pilot): Gaussian, all flat.

function log10earn_height_sb()
    M = Log10earnHeightExample
    df = (; y=log10.(M.LOG10EARN_HEIGHT_EARN), height=M.LOG10EARN_HEIGHT_HEIGHT)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + height
        effect(mu, Intercept) ~ Flat()
        effect(mu, height) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logearn_height_sb()
    M = LogearnHeightExample
    df = (; y=log.(M.LOGEARN_HEIGHT_EARN), height=M.LOGEARN_HEIGHT_HEIGHT)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + height
        effect(mu, Intercept) ~ Flat()
        effect(mu, height) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logearn_height_male_sb()
    M = LogearnHeightMaleExample
    df = (; y=log.(M.LEHM_EARN), height=M.LEHM_HEIGHT, male=M.LEHM_MALE)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + height + male
        effect(mu, Intercept) ~ Flat()
        effect(mu, height) ~ Flat()
        effect(mu, male) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logearn_interaction_sb()
    M = LogearnInteractionExample
    h, ml = M.LOGEARN_INTERACTION_HEIGHT, M.LOGEARN_INTERACTION_MALE
    df = (; y=log.(M.LOGEARN_INTERACTION_EARN), height=h, male=ml,
        inter=h .* ml)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + height + male + inter
        effect(mu, Intercept) ~ Flat()
        effect(mu, height) ~ Flat()
        effect(mu, male) ~ Flat()
        effect(mu, inter) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logearn_interaction_z_sb()
    M = LogearnInteractionZExample
    h, ml = M.LEIZ_HEIGHT, M.LEIZ_MALE
    zh = (h .- mean(h)) ./ std(h)
    df = (; y=log.(M.LEIZ_EARN), zh, male=ml, inter=zh .* ml)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + zh + male + inter
        effect(mu, Intercept) ~ Flat()
        effect(mu, zh) ~ Flat()
        effect(mu, male) ~ Flat()
        effect(mu, inter) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logearn_logheight_male_sb()
    M = LogearnLogheightMaleExample
    df = (; y=log.(M.LELHM_EARN), logheight=log.(M.LELHM_HEIGHT),
        male=M.LELHM_MALE)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + logheight + male
        effect(mu, Intercept) ~ Flat()
        effect(mu, logheight) ~ Flat()
        effect(mu, male) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _EARN_CASES = (
    ("log10earn_height", log10earn_height_sb, 0.0, ""),
    ("logearn_height", logearn_height_sb, 0.0, ""),
    ("logearn_height_male", logearn_height_male_sb, 0.0, ""),
    ("logearn_interaction", logearn_interaction_sb, 0.0, ""),
    ("logearn_interaction_z", logearn_interaction_z_sb, 0.0, ""),
    ("logearn_logheight_male", logearn_logheight_male_sb, 0.0, ""),
)

_run_batch(_EARN_CASES)
println("SB_SWEEP core/earn done -> $OUT")

# ---- mesquite (6): Gaussian, all flat. Log/derived columns host-side.

function mesquite_sb()
    M = MesquiteExample
    df = (; y=M.MESQ_WEIGHT, d1=M.MESQ_DIAM1, d2=M.MESQ_DIAM2,
        ch=M.MESQ_CANOPY_HEIGHT, th=M.MESQ_TOTAL_HEIGHT,
        den=M.MESQ_DENSITY, g=M.MESQ_GROUP)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + d1 + d2 + ch + th + den + g
        effect(mu, Intercept) ~ Flat()
        effect(mu, d1) ~ Flat()
        effect(mu, d2) ~ Flat()
        effect(mu, ch) ~ Flat()
        effect(mu, th) ~ Flat()
        effect(mu, den) ~ Flat()
        effect(mu, g) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logmesquite_sb()
    M = LogmesquiteExample
    df = (; y=M.LOGMESQ_LOG_WEIGHT, d1=log.(M.LOGMESQ_DIAM1),
        d2=log.(M.LOGMESQ_DIAM2), ch=log.(M.LOGMESQ_CANOPY_HEIGHT),
        th=log.(M.LOGMESQ_TOTAL_HEIGHT), den=log.(M.LOGMESQ_DENSITY),
        g=M.LOGMESQ_GROUP)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + d1 + d2 + ch + th + den + g
        effect(mu, Intercept) ~ Flat()
        effect(mu, d1) ~ Flat()
        effect(mu, d2) ~ Flat()
        effect(mu, ch) ~ Flat()
        effect(mu, th) ~ Flat()
        effect(mu, den) ~ Flat()
        effect(mu, g) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logmesquite_logva_sb()
    M = LogmesquiteLogvaExample
    d1, d2, ch = M.LOGVA_DIAM1, M.LOGVA_DIAM2, M.LOGVA_CANOPY_HEIGHT
    df = (; y=M.LOGVA_LOG_WEIGHT, vol=log.(d1 .* d2 .* ch),
        area=log.(d1 .* d2), g=M.LOGVA_GROUP)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + vol + area + g
        effect(mu, Intercept) ~ Flat()
        effect(mu, vol) ~ Flat()
        effect(mu, area) ~ Flat()
        effect(mu, g) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logmesquite_logvas_sb()
    M = LogmesquiteLogvasExample
    d1, d2, ch = M.LOGVAS_DIAM1, M.LOGVAS_DIAM2, M.LOGVAS_CANOPY_HEIGHT
    df = (; y=M.LOGVAS_LOG_WEIGHT, vol=log.(d1 .* d2 .* ch),
        area=log.(d1 .* d2), shape=log.(d1 ./ d2),
        th=log.(M.LOGVAS_TOTAL_HEIGHT), den=log.(M.LOGVAS_DENSITY),
        g=M.LOGVAS_GROUP)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + vol + area + shape + th + den + g
        effect(mu, Intercept) ~ Flat()
        effect(mu, vol) ~ Flat()
        effect(mu, area) ~ Flat()
        effect(mu, shape) ~ Flat()
        effect(mu, th) ~ Flat()
        effect(mu, den) ~ Flat()
        effect(mu, g) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logmesquite_logvash_sb()
    M = LogmesquiteLogvashExample
    d1, d2, ch = M.LOGVASH_DIAM1, M.LOGVASH_DIAM2, M.LOGVASH_CANOPY_HEIGHT
    df = (; y=M.LOGVASH_LOG_WEIGHT, vol=log.(d1 .* d2 .* ch),
        area=log.(d1 .* d2), shape=log.(d1 ./ d2),
        th=log.(M.LOGVASH_TOTAL_HEIGHT), g=M.LOGVASH_GROUP)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + vol + area + shape + th + g
        effect(mu, Intercept) ~ Flat()
        effect(mu, vol) ~ Flat()
        effect(mu, area) ~ Flat()
        effect(mu, shape) ~ Flat()
        effect(mu, th) ~ Flat()
        effect(mu, g) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function logmesquite_logvolume_sb()
    M = LogmesquiteLogvolumeExample
    d1, d2, ch = M.LOGVOL_DIAM1, M.LOGVOL_DIAM2, M.LOGVOL_CANOPY_HEIGHT
    df = (; y=M.LOGVOL_LOG_WEIGHT, vol=log.(d1 .* d2 .* ch))
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + vol
        effect(mu, Intercept) ~ Flat()
        effect(mu, vol) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _MESQUITE_CASES = (
    ("mesquite", mesquite_sb, 0.0, ""),
    ("logmesquite", logmesquite_sb, 0.0, ""),
    ("logmesquite_logva", logmesquite_logva_sb, 0.0, ""),
    ("logmesquite_logvas", logmesquite_logvas_sb, 0.0, ""),
    ("logmesquite_logvash", logmesquite_logvash_sb, 0.0, ""),
    ("logmesquite_logvolume", logmesquite_logvolume_sb, 0.0, ""),
)

_run_batch(_MESQUITE_CASES)
println("SB_SWEEP core/mesquite done -> $OUT")

# ---- dogs (2 of 4; hierarchical/nonhierarchical ride the S driver).

function dogs_sb()
    M = DogsExample
    df = (; y=M.DOGS_Y_FLAT, na=M.DOGS_N_AVOID, ns=M.DOGS_N_SHOCK)
    builder = @brm begin
        mu ~ 1 + na + ns
        effect(mu, Intercept) ~ Normal(0, 100)
        effect(mu, na) ~ Normal(0, 100)
        effect(mu, ns) ~ Normal(0, 100)
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function dogs_log_sb()
    M = DogsLogExample
    df = (; y=M.DOGS_LOG_Y_FLAT, na=M.DOGS_LOG_N_AVOID, ns=M.DOGS_LOG_N_SHOCK)
    builder = @brm begin
        mu ~ 0 + na + ns
        effect(mu, na) ~ Uniform(-100, 0)
        effect(mu, ns) ~ Uniform(0, 100)
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _DOGS_CASES = (
    ("dogs", dogs_sb, 0.0, ""),
    # Seeded point hand-placed interior ([na, ns] order): the coefficients
    # are declared unbounded with Uniform bounds enforced by a rejecting
    # _lpdf, so a generic 0.25*randn draw lands outside [-100,0]x[0,100]
    # 75% of the time (-Inf lp). [-1, 1] is safely interior.
    ("dogs_log", dogs_log_sb, 0.0, "", [-1.0, 1.0]),
)

_run_batch(_DOGS_CASES)
println("SB_SWEEP core/dogs done -> $OUT")

# ---- eight_schools (2): KNOWN observation scales; hierarchical on schools.
# Centered tau carries +log(2): attempted via truncated Cauchy; falls back
# to a stated offset if the spelling is refused (see offset below).

function eight_schools_sb()
    M = EightSchoolsExample
    df = (; y=M.EIGHT_SCHOOLS_Y, se=M.EIGHT_SCHOOLS_SIGMA, school=1:8)
    builder = @brm begin
        mu ~ 1 + (1 | es | school)
        effect(mu, Intercept) ~ Normal(0, 5)
        sd(:, es) ~ truncated(Cauchy(0, 5); lower=0.0)
        y ~ Normal(mu, se)
    end
    return SBBRMI(builder(df); mod=@__MODULE__, centered_groups=[:school])
end

function eight_schools_noncentered_sb()
    M = EightSchoolsNoncenteredExample
    df = (; y=M.ES_NC_Y, se=M.ES_NC_SIGMA, school=1:8)
    builder = @brm begin
        mu ~ 1 + (1 | esn | school)
        effect(mu, Intercept) ~ Normal(0, 5)
        sd(:, esn) ~ Cauchy(0, 5)
        y ~ Normal(mu, se)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _EIGHT_CASES = (
    ("eight_schools", eight_schools_sb, 0.0, ""),
    ("eight_schools_noncentered", eight_schools_noncentered_sb, 0.0, ""),
)

_run_batch(_EIGHT_CASES)
println("SB_SWEEP core/eight done -> $OUT")

# ---- rate (5): bare Beta/Binomial parameters. Scalar-data spelling is
# runtime-decided: on refusal these move to the S driver (trivial @slic).

function rate_1_sb()
    M = Rate1Example
    df = (; k=M.RATE1_K, n=M.RATE1_N)
    builder = @brm begin
        theta ~ Beta(1, 1)
        k ~ Binomial(n, theta)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function rate_2_sb()
    M = Rate2Example
    df = (; k=[M.RATE2_K1, M.RATE2_K2], n=[M.RATE2_N1, M.RATE2_N2])
    builder = @brm begin
        theta1 ~ Beta(1, 1)
        theta2 ~ Beta(1, 1)
        k ~ Binomial(n, [theta1, theta2])
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function rate_3_sb()
    M = Rate3Example
    df = (; k=[M.RATE3_K1, M.RATE3_K2], n=[M.RATE3_N1, M.RATE3_N2])
    builder = @brm begin
        theta ~ Beta(1, 1)
        k ~ Binomial(n, theta)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function rate_4_sb()
    M = Rate4Example
    df = (; k=M.RATE4_K, n=M.RATE4_N)
    builder = @brm begin
        theta ~ Beta(1, 1)
        thetaprior ~ Beta(1, 1)
        k ~ Binomial(n, theta)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function rate_5_sb()
    M = Rate5Example
    df = (; k=[M.RATE5_K1, M.RATE5_K2], n=[M.RATE5_N1, M.RATE5_N2])
    builder = @brm begin
        theta ~ Beta(1, 1)
        k ~ Binomial(n, theta)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _RATE_CASES = (
    ("rate_1", rate_1_sb, 0.0, ""),
    ("rate_2", rate_2_sb, 0.0, ""),
    ("rate_3", rate_3_sb, 0.0, ""),
    ("rate_4", rate_4_sb, 0.0, ""),
    ("rate_5", rate_5_sb, 0.0, ""),
)

_run_batch(_RATE_CASES)
println("SB_SWEEP core/rate done -> $OUT")

# ---- glm/misc B1 remainder.

function linear_regression_sb()
    M = LinearRegressionExample
    df = (; y=M.LINREG_Y, x=vec(M.LINREG_X))
    builder = @brm begin
        sigma ~ truncated(Normal(0, 5); lower=0.0)
        mu ~ 1 + x
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, x) ~ Normal(0, 10)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function bound_regression_sb()
    M = BoundRegressionExample
    X = M.BOUND_RAW_X
    n = size(X, 1)
    cm = vec(sum(X; dims=1) ./ n)
    csd = vec(sqrt.(sum(abs2, X .- cm'; dims=1) ./ n))
    Z = (X .- cm') ./ csd'
    df = (; y=M.BOUND_Y, z1=Z[:, 1], z2=Z[:, 2])
    builder = @brm begin
        sigma ~ truncated(Normal(0, 5); lower=0.0)
        mu ~ 1 + z1 + z2
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, z1) ~ Normal(0, 5)
        effect(mu, z2) ~ Normal(0, 5)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function beta_binomial_sb()
    M = BetaBinomialExample
    df = (; k=M.BETA_BINOMIAL_SUCCESSES, n=M.BETA_BINOMIAL_TRIALS)
    builder = @brm begin
        rate ~ Beta(2, 2)
        k ~ Binomial(n, rate)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function poisson_gamma_sb()
    M = PoissonGammaExample
    df = (; y=M.POISSON_COUNTS)
    builder = @brm begin
        rate ~ Gamma(2, 1)
        y ~ Poisson(rate)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function glm_binomial_sb()
    M = GLMBinomialExample
    yr = M.GLM_BINOMIAL_YEAR
    df = (; c=M.GLM_BINOMIAL_C, n=M.GLM_BINOMIAL_N, year=yr, year2=yr .^ 2)
    builder = @brm begin
        mu ~ 1 + year + year2
        effect(mu, Intercept) ~ Normal(0, 100)
        effect(mu, year) ~ Normal(0, 100)
        effect(mu, year2) ~ Normal(0, 100)
        c ~ BinomialLogit(n, mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function glm_poisson_sb()
    M = GLMPoissonExample
    yr = M.GLM_POISSON_YEAR
    df = (; c=M.GLM_POISSON_C, year=yr, year2=yr .^ 2, year3=yr .^ 3)
    builder = @brm begin
        mu ~ 1 + year + year2 + year3
        effect(mu, Intercept) ~ Uniform(-20, 20)
        effect(mu, year) ~ Uniform(-10, 10)
        effect(mu, year2) ~ Uniform(-10, 10)
        effect(mu, year3) ~ Uniform(-10, 10)
        c ~ Poisson(exp(mu))
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function nes_sb()
    M = NESExample
    X = M.NES_X
    cols = ntuple(j -> X[:, j], size(X, 2))
    df = (; y=M.NES_PARTYID7, ntuple(j -> Symbol(:x, j) => cols[j], 9)...)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
        effect(mu, x1) ~ Flat()
        effect(mu, x2) ~ Flat()
        effect(mu, x3) ~ Flat()
        effect(mu, x4) ~ Flat()
        effect(mu, x5) ~ Flat()
        effect(mu, x6) ~ Flat()
        effect(mu, x7) ~ Flat()
        effect(mu, x8) ~ Flat()
        effect(mu, x9) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function nes_logit_sb()
    M = NesLogitExample
    df = (; y=M.NES_LOGIT_VOTE, income=M.NES_LOGIT_INCOME)
    builder = @brm begin
        mu ~ 1 + income
        effect(mu, Intercept) ~ Flat()
        effect(mu, income) ~ Flat()
        y ~ BernoulliLogit(mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function diamonds_sb()
    M = DiamondsExample
    X = M.DIAMONDS_X
    Xnoint = X[:, 2:size(X, 2)]
    cm = vec(sum(Xnoint; dims=1) ./ size(Xnoint, 1))
    Xc = Xnoint .- cm'
    cols = ntuple(j -> Xc[:, j], size(Xc, 2))
    df = (; y=M.DIAMONDS_Y, ntuple(j -> Symbol(:x, j) => cols[j], 24)...)
    builder = @brm begin
        # Normalized half-t: Stan `T[0,]` adds +log(2), matching the RK
        # model's explicit `log(2) + student_t(3,0,10).logpdf(sigma)`.
        sigma ~ truncated(LocationScale(0, 10, TDist(3)); lower=0.0)
        mu ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 +
            x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 +
            x21 + x22 + x23 + x24
        effect(mu, Intercept) ~ LocationScale(8, 10, TDist(3))
        effect(mu, :) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function kilpisjarvi_sb()
    M = KilpisjarviExample
    df = (; y=M.KILPISJARVI_Y, x=M.KILPISJARVI_X)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + x
        effect(mu, Intercept) ~ Normal(9.31290322580645, 100.0)
        effect(mu, x) ~ Normal(0.0, 0.0333333333333333)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function sesame_sb()
    M = SesameOnePredAExample
    df = (; y=M.SESAME_WATCHED, e=M.SESAME_ENCOURAGED)
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + e
        effect(mu, Intercept) ~ Flat()
        effect(mu, e) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

function ark_sb()
    M = ARKExample
    Ylag = M.ARK_YLAG
    @assert M.ARK_K == 5
    df = (; y=M.ARK_YT, ntuple(j -> Symbol(:lag, j) => Ylag[:, j], 5)...)
    builder = @brm begin
        sigma ~ Cauchy(0, 2.5; lower=0.0)
        mu ~ 1 + lag1 + lag2 + lag3 + lag4 + lag5
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, lag1) ~ Normal(0, 10)
        effect(mu, lag2) ~ Normal(0, 10)
        effect(mu, lag3) ~ Normal(0, 10)
        effect(mu, lag4) ~ Normal(0, 10)
        effect(mu, lag5) ~ Normal(0, 10)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _MISC_B1_CASES = (
    ("linear_regression", linear_regression_sb, 0.0, ""),
    ("bound_regression", bound_regression_sb, 0.0, ""),
    ("beta_binomial", beta_binomial_sb, 0.0, ""),
    ("poisson_gamma", poisson_gamma_sb, 0.0, ""),
    ("glm_binomial", glm_binomial_sb, 0.0, ""),
    ("glm_poisson", glm_poisson_sb,
        -(log(40) + 3 * log(20)),
        "explicit Uniform box vs implicit-uniform .stan (0); -log(width) each"),
    ("nes", nes_sb, 0.0, ""),
    ("nes_logit", nes_logit_sb, 0.0, ""),
    ("diamonds", diamonds_sb, 0.0, ""),
    ("kilpisjarvi", kilpisjarvi_sb, 0.0, ""),
    ("sesame_one_pred_a", sesame_sb, 0.0, ""),
    ("ark", ark_sb, 0.0, ""),
)

_run_batch(_MISC_B1_CASES)
println("SB_SWEEP core/misc done -> $OUT")

# ---- seeds_stanified: binomial GLMM with OBSERVATION-level RE (one b per
# row); alpha* ~ N(0,1), sigma ~ Cauchy(0,1) plain.

function seeds_stanified_sb()
    M = SeedsStanifiedExample
    x1, x2 = M.SEEDS_STANIFIED_X1, M.SEEDS_STANIFIED_X2
    n = length(M.SEEDS_STANIFIED_COUNTS)
    df = (; c=M.SEEDS_STANIFIED_COUNTS, ntot=M.SEEDS_STANIFIED_TOTALS,
        x1, x2, inter=x1 .* x2, obs=collect(1:n))
    builder = @brm begin
        mu ~ 1 + x1 + x2 + inter + (1 | ss | obs)
        effect(mu, Intercept) ~ Normal(0, 1)
        effect(mu, x1) ~ Normal(0, 1)
        effect(mu, x2) ~ Normal(0, 1)
        effect(mu, inter) ~ Normal(0, 1)
        sd(:, ss) ~ Cauchy(0, 1)
        c ~ BinomialLogit(ntot, mu)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

const _SEEDS_CASES = (("seeds_stanified_model", seeds_stanified_sb, 0.0, ""),)

_run_batch(_SEEDS_CASES)
println("SB_SWEEP core/seeds done -> $OUT")
