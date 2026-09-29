# test/varyingsource_fixture.jl — synthetic varyingsource-twin model for the
# RK extraction tests (no Bruno dependency).
#
# Included (into Main) by test/rk_emitter.jl (plan), test/rk_ast.jl (AST),
# and test/rk_parity.jl (publication tripwire). Builds a twin-shaped
# `@brm` model from a generated string (like the twin itself): 13 subject
# LPs sharing one correlated `|p|` block, 3 dose-axis LPs, 2 placebo HSGP
# LPs, a `dose_concentration_gp` effectiveness submodel, per-assay
# Exponential scales, and the `varyingsource_subject_locs` kernel cell.
#
# The twin-only call heads are Main stubs (never called — the planner
# matches them by name; the stubs exist in case `@brm` resolves cell
# heads at construction).
function dose_concentration_gp(args...; kws...)
    nothing
end
function varyingsource_subject_locs(args...; kws...)
    nothing
end
function censored_addpropnormal(args...; kws...)
    nothing
end
function assay_scale(args...; kws...)
    nothing
end
# Wrong-head submodel for the fail-closed battery (must resolve at
# construction so the planner — not `@brm` — rejects it by name).
function other_gp(args...; kws...)
    nothing
end

const _VS_FIX_SUBJECT_LPS = (:lp1, :lp2, :lp3, :lp4, :lp5, :lp6,
    :lp7, :lp8, :lp9, :lp10, :lp11, :lp12, :lp13)
const _VS_FIX_DOSE_LPS = (:d1, :d2, :d3)
const _VS_FIX_PLACEBO_LPS = (:p1, :p2)

# Reference stan_data dict (bridge input): 3 subjects, 7 obs over 3
# assays, 4 dose rows, 3 discretization lags.
function vs_test_stan_data()
    Dict{Symbol,Any}(
        :subject => [1, 1, 2, 2, 2, 3, 3],
        :ts => [0.0, 4.0, 0.0, 2.0, 8.0, 0.0, 6.0],
        :assay => [1, 2, 1, 3, 2, 1, 3],
        :obs => [8.0, 120.0, 6.0, 55.0, 90.0, 7.0, 60.0],
        :lloq => [1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0],
        :dosing_subject => [1, 2, 2, 3],
        :dosing_times => [0.0, 0.0, 4.0, 0.0],
        :doses => [50.0, 50.0, 20.0, 50.0],
        :treatment => [1, 2, 4, 1],
        :dosing_diet => [1, 2, 3, 4],
        :discretization_times => [1.0, 2.0, 4.0],
        :placebo_lo_time => 0.0,
        :placebo_hi_time => 24.0,
    )
end

# Twin-shaped data container: subject/obs/dose/placebo axes +
# precomputed design columns (dropped by the planner).
function vs_test_data(stan=vs_test_stan_data())
    (;
        subject=[1, 2, 3],
        diseased=[0.0, 1.0, 0.0],
        male=[1.0, 0.0, 1.0],
        age_std=[-0.5, 0.2, 0.9],
        weight_std=[0.1, -0.3, 0.4],
        treatment_map=[101, 202, 104],
        pk_unique_dts=[7, 7, 7],
        obs_subject=stan[:subject],
        obs=stan[:obs],
        obs_assay=stan[:assay],
        obs_lloq=stan[:lloq],
        dose_subject=stan[:dosing_subject],
        dose_amount=stan[:doses],
        vessel_bottle=[1.0, 0.0, 0.0, 1.0],
        vessel_bottle_20=[0.0, 1.0, 0.0, 0.0],
        vessel_tablet=[0.0, 0.0, 1.0, 0.0],
        vessel_tablet_60=[0.0, 0.0, 0.0, 0.0],
        diet=categorical(stan[:dosing_diet]; levels=1:4, ordered=true),
        placebo_subject=[1, 1, 2, 2, 3],
        placebo_time=[-1.0, 0.0, -0.5, 0.5, 0.0],
        placebo_time_csf=[-0.8, 0.2, -0.3, 0.6, 0.1],
    )
end

# Twin-shaped formula block (k_gp = 4, k_placebo = 3 — distinct valid
# ranks; assay scales carry distinct literals per assay).
function vs_test_body()
    cov_rhs = "1 + diseased + male + age_std + weight_std + (1 | p | subject)"
    lean_rhs = "1 + diseased + (1 | p | subject)"
    subject_formulas = join(
        ["$lp ~ $((i <= 2 ? cov_rhs : lean_rhs))"
            for (i, lp) in enumerate(_VS_FIX_SUBJECT_LPS)], "\n")
    dose_formulas = join(
        ["$lp ~ 0 + vessel_bottle + vessel_bottle_20 + vessel_tablet + " *
            "vessel_tablet_60 + mo(diet)" for lp in _VS_FIX_DOSE_LPS], "\n")
    placebo_formulas = join([
        "p1 ~ 0 + hsgp(placebo_time; k = 3, domain = (-1.5, 1.5))",
        "p2 ~ 0 + hsgp(placebo_time_csf; k = 3, domain = (-1.5, 1.5))",
    ], "\n")
    subj_args = join(_VS_FIX_SUBJECT_LPS, ", ")
    subj_cells = join(["$(lp)_i" for lp in _VS_FIX_SUBJECT_LPS], ", ")
    priors = join([
        ["effect($lp, Intercept) ~ Normal($(0.1 * i), 1.0)"
            for (i, lp) in enumerate(_VS_FIX_SUBJECT_LPS)]...,
        "effect(:, diseased) ~ Normal(0.0, 1.0)",
        "effect(:, male) ~ Normal(0.0, 2.0)",
        "effect(:, age_std) ~ Normal(0.0, 2.0)",
        "effect(:, weight_std) ~ Normal(0.0, 2.0)",
        "sd(:, p) ~ Exponential(0.5)",
        "cor(:, p) ~ LKJCholesky(13, 2.0)",
        ["effect($lp, :) ~ Normal(0.0, 0.5)" for lp in _VS_FIX_DOSE_LPS]...,
        "length_scale(:, hsgp(placebo_time)) ~ Uniform(0.5, 2.0)",
        "sd(:, hsgp(placebo_time)) ~ LogNormal(0.0, 1.0)",
        "length_scale(:, hsgp(placebo_time_csf)) ~ Uniform(0.7, 1.8)",
        "sd(:, hsgp(placebo_time_csf)) ~ LogNormal(0.1, 0.9)",
    ], "\n")
    """
    $subject_formulas
    $dose_formulas
    $placebo_formulas
    effectiveness ~ dose_concentration_gp(; k = 4, dose_slope_scale = 1.5,
        conc_slope_scale = 2.5, amplitude_scale = 2.0)
    a1 ~ Exponential(1.0)
    a2 ~ Exponential(2.0)
    a3 ~ Exponential(3.0)
    r1 ~ Exponential(4.0)
    r2 ~ Exponential(5.0)
    r3 ~ Exponential(6.0)
    loc ~ kernel(ragged(obs, obs_subject), ragged(obs_assay, obs_subject),
        ragged(obs_lloq, obs_subject), ragged(dose_amount, dose_subject),
        treatment_map, pk_unique_dts, ragged(d1, dose_subject),
        ragged(d2, dose_subject), ragged(d3, dose_subject),
        ragged(p1, placebo_subject), ragged(p2, placebo_subject),
        $subj_args) do y_i, assay_i, lloq_i, dose_i, tm_i, pkd_i,
            d1_i, d2_i, d3_i, p1_i, p2_i, $subj_cells
        mu = varyingsource_subject_locs(assay_i, dose_i, tm_i, pkd_i,
            d1_i, d2_i, d3_i, p1_i, p2_i, effectiveness, $subj_cells)
        y_i ~ censored_addpropnormal(mu, assay_scale(assay_i, a1, a2, a3),
            assay_scale(assay_i, r1, r2, r3), lloq_i)
        mu
    end
    $priors
    """
end

vs_test_brmi(data=vs_test_data(), body=vs_test_body()) =
    Core.eval(Main, BRM._brm(body; df=data))
vs_test_raw(stan=vs_test_stan_data()) = rk_varyingsource_raw(stan)
vs_test_plan(brmi=vs_test_brmi(), raw=vs_test_raw()) =
    BRM._brm_rk_plan(brmi; centered_groups=[:subject], varyingsource_raw=raw)

# Public 3-subject source-equivalent twin used by the peer's
# Stan-reference benchmark: 13 centered margins, k=4 GP, k=10 placebo
# HSGPs, original covariates/priors, and no disease term on either
# absorption margin. Its 232 coordinates match the peer's 232-coordinate
# Stan comparison.
const _VS_REFERENCE_SUBJECT_LPS = (:lp1, :lp2, :lp3, :lp4, :lp5, :lp6,
    :lp7, :lp8, :lp9, :lp10, :lp11, :lp12, :lp13)
const _VS_REFERENCE_LOC0 = vcat(
    log.([10.0, 0.01, sqrt(0.01 * 50), sqrt(0.001 * 5)]) .+
        [0.0, 3.2, 0.0, 0.0],
    [log(130.0) - 1, -4.0, -4.0, 2.0, log(80.0) - 1, -4.0, 2.0],
    [log(1 / 8), 0.0])
const _VS_REFERENCE_SCALE0 = [0.8, 0.8, 2.0, 2.0, 0.8, 0.8, 0.8, 0.8,
    0.8, 0.8, 0.8, 0.8, 0.8]
const _VS_REFERENCE_RHO_LOWER = (6 / pi) *
    sqrt(log(100.0) / ((10^2) - 1))

function vs_reference_data(cols)
    (;
        subject=cols[:sid], diseased=cols[:diseased], male=cols[:male],
        age_std=cols[:age_std], weight_std=cols[:weight_std],
        obs_subject=cols[:subj], obs=cols[:dv], obs_assay=cols[:assay],
        obs_lloq=cols[:lloq], treatment_map=[101, 202, 104],
        pk_unique_dts=fill(length(cols[:disc]), length(cols[:sid])),
        dose_subject=cols[:dsubj], dose_amount=cols[:damt],
        vessel_bottle=cols[:bottle], vessel_bottle_20=cols[:bottle_20],
        vessel_tablet=cols[:tablet], vessel_tablet_60=cols[:tablet_60],
        diet=categorical(cols[:diet]; levels=1:4, ordered=true),
        placebo_subject=[1, 1, 2, 2, 3],
        placebo_time=[-1.0, 0.0, -0.5, 0.5, 0.0],
        placebo_time_csf=[-0.8, 0.2, -0.3, 0.6, 0.1],
    )
end

function vs_reference_stan_data(cols)
    Dict{Symbol,Any}(
        :subject=>cols[:subj], :ts=>cols[:time], :assay=>cols[:assay],
        :obs=>cols[:dv], :lloq=>cols[:lloq],
        :dosing_subject=>cols[:dsubj], :dosing_times=>cols[:dtime],
        :doses=>cols[:damt], :treatment=>cols[:vessel],
        :dosing_diet=>cols[:diet],
        :discretization_times=>cols[:disc],
        :placebo_lo_time=>0.0, :placebo_hi_time=>24.0)
end

function vs_reference_body()
    subject_rhs(i) = i <= 2 ?
        "1 + diseased + male + age_std + weight_std + (1 | p | subject)" :
        (i <= 11 ? "1 + diseased + (1 | p | subject)" :
            "1 + (1 | p | subject)")
    subject_formulas = join(["$lp ~ $(subject_rhs(i))"
        for (i, lp) in enumerate(_VS_REFERENCE_SUBJECT_LPS)], "\n")
    dose_formulas = join([
        "$lp ~ 0 + vessel_bottle + vessel_bottle_20 + vessel_tablet + " *
            "vessel_tablet_60 + mo(diet)" for lp in (:d1, :d2, :d3)], "\n")
    placebo_formulas = join([
        "p1 ~ 0 + hsgp(placebo_time; k = 10, domain = (-1.5, 1.5))",
        "p2 ~ 0 + hsgp(placebo_time_csf; k = 10, domain = (-1.5, 1.5))",
    ], "\n")
    subj_args = join(_VS_REFERENCE_SUBJECT_LPS, ", ")
    subj_cells = join(["$(lp)_i" for lp in _VS_REFERENCE_SUBJECT_LPS], ", ")
    intercept_priors = join([
        "effect($lp, Intercept) ~ Normal($(_VS_REFERENCE_LOC0[i]), " *
            "$(_VS_REFERENCE_SCALE0[i]))"
        for (i, lp) in enumerate(_VS_REFERENCE_SUBJECT_LPS)], "\n")
    priors = join([
        intercept_priors,
        "effect(:, diseased) ~ Normal(0.0, 1.0)",
        "effect(:, male) ~ Normal(0.0, 0.1)",
        "effect(:, age_std) ~ Normal(0.0, 0.1)",
        "effect(:, weight_std) ~ Normal(0.0, 0.1)",
        "sd(:, p) ~ Exponential(0.6666666666666666)",
        "cor(:, p) ~ LKJCholesky(13, 2.0)",
        ["effect($lp, :) ~ Normal(0.0, 0.5)" for lp in (:d1, :d2, :d3)]...,
        "length_scale(:, hsgp(placebo_time)) ~ " *
            "Uniform($(_VS_REFERENCE_RHO_LOWER), 2.0)",
        "sd(:, hsgp(placebo_time)) ~ LogNormal(0.0, 1.0)",
        "length_scale(:, hsgp(placebo_time_csf)) ~ " *
            "Uniform($(_VS_REFERENCE_RHO_LOWER), 2.0)",
        "sd(:, hsgp(placebo_time_csf)) ~ LogNormal(0.0, 1.0)",
    ], "\n")
    """
    $subject_formulas
    $dose_formulas
    $placebo_formulas
    effectiveness ~ dose_concentration_gp(; k = 4, dose_slope_scale = 1.0,
        conc_slope_scale = 1.0, amplitude_scale = 1.0)
    a1 ~ Exponential(0.25)
    a2 ~ Exponential(0.25)
    a3 ~ Exponential(0.25)
    r1 ~ Exponential(0.25)
    r2 ~ Exponential(0.25)
    r3 ~ Exponential(0.25)
    loc ~ kernel(ragged(obs, obs_subject), ragged(obs_assay, obs_subject),
        ragged(obs_lloq, obs_subject), ragged(dose_amount, dose_subject),
        treatment_map, pk_unique_dts, ragged(d1, dose_subject),
        ragged(d2, dose_subject), ragged(d3, dose_subject),
        ragged(p1, placebo_subject), ragged(p2, placebo_subject),
        $subj_args) do y_i, assay_i, lloq_i, dose_i, tm_i, pkd_i,
            d1_i, d2_i, d3_i, p1_i, p2_i, $subj_cells
        mu = varyingsource_subject_locs(assay_i, dose_i, tm_i, pkd_i,
            d1_i, d2_i, d3_i, p1_i, p2_i, effectiveness, $subj_cells)
        y_i ~ censored_addpropnormal(mu, assay_scale(assay_i, a1, a2, a3),
            assay_scale(assay_i, r1, r2, r3), lloq_i)
        mu
    end
    $priors
    """
end

vs_reference_brmi(cols) = Core.eval(Main, BRM._brm(vs_reference_body();
    df=vs_reference_data(cols)))
function vs_reference_plan(cols)
    brmi = vs_reference_brmi(cols)
    BRM._brm_rk_plan(brmi; centered_groups=[:subject],
        varyingsource_raw=rk_varyingsource_raw(
            vs_reference_stan_data(cols)))
end
