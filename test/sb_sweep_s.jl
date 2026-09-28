# test/sb_sweep_s.jl — v1 SB-parity sweep, S (@slic-port) driver.
#
# Run: julia --project=test test/sb_sweep_s.jl
#
# @slic ports of inventory rows with no exact @brm counterpart (priors and
# transforms transcribed EXACTLY from the RK SOURCES at pin `8a51702`).
# Same record schema + parity-point convention as the @brm drivers; the
# Stan program (not the authoring surface) is what the parity leg pins.
#
# Records append to $SB_SWEEP_OUT (default: tempdir()/sb-sweep-s.jsonl).
# Crash-resume: cases already in OUT are skipped, never duplicated.

include(joinpath(@__DIR__, "sb_sweep_common.jl"))

using ReactiveKernelsPPLExamples: Rate2Example, Rate4Example, DugongsGrowthExample,
    PilotsExample, GLMMPoissonExample, SumToZeroExample, MVNormalRegressionExample,
    EightSchoolsExample, NormalMixtureExample, NormalMixtureKExample,
    LowDimGaussMixExample, LowDimGaussMixCollapseExample, SeedsExample,
    SeedsCenteredExample, SurgicalExample, SurveyModelExample, M0Example,
    MbExample, MhExample, MtExample, MthModelExample, MtbhModelExample,
    LsatExample, Irt2plExample, Hier2plExample, TwoplLatentRegIrtExample,
    DogsHierarchicalExample, DogsNonhierarchicalExample,
    LogisticRegressionRHSExample, StateSpaceStochasticExample, ProphetExample,
    GPRegrExample, HierarchicalGPExample, KroneckerGpExample, AccelGPExample,
    AccelSplinesExample, Bym2OffsetOnlyExample

# Builders + custom families live in sb_sweep_s_models.jl so transpile probes
# can include them without running the record loop below.
include(joinpath(@__DIR__, "sb_sweep_s_models.jl"))
const OUT = get(ENV, "SB_SWEEP_OUT",
    joinpath(tempdir(), "sb-sweep-s.jsonl"))
const _DONE = _sweep_done_cases(OUT)

# SlicModel parallel of sweep_case (kept separate so the green @brm drivers
# never churn): compile via StanBlocks directly, same schema out.
function sweep_scase(io::IO, case::AbstractString, build::Function;
        offset::Float64=0.0, offset_reason::String="")
    model = build()
    code = StanBlocks.stan_code(model)
    stan_sha = _short_sha(code)
    mkpath(_SWEEP_CACHE)
    path = joinpath(_SWEEP_CACHE, "$(case)-$(stan_sha).stan")
    isfile(path) || write(path, code)
    problem = Base.invokelatest(StanBlocks.stan_instantiate, model; path=path)
    names = BS.param_unc_names(problem.model)
    data_hash = _short_sha(_canonical_data(StanBlocks.stan_data(model)))
    recs = []
    for (label, q) in sweep_points(case, length(names))
        grad = zeros(length(q))
        lp, _ = BS.log_density_gradient!(problem.model, Vector{Float64}(q), grad;
            propto=false, jacobian=true)
        @assert isfinite(lp) "non-finite lp for $case/$label at q=$q " *
            "(names=$(names)): parity point outside model support?"
        rec = Dict(
            "case" => String(case), "label" => String(label),
            "brm_tip" => _brm_tip(), "stan_sha" => stan_sha,
            "data_hash" => data_hash, "stan_names" => names,
            "q" => Vector{Float64}(q), "lp" => lp,
            "offset" => offset, "offset_reason" => offset_reason,
        )
        println(io, JSON.json(rec))
        flush(io)
        push!(recs, rec)
    end
    return recs
end

const _S_BATCH1 = (
    ("rate_2", rate_2_s, 0.0, ""),
    ("rate_4", rate_4_s, 0.0, ""),
    ("dugongs", dugongs_s, 0.0, ""),
)

const _S_BATCH2 = (
    ("pilots", pilots_s, -3 * log(100),
        "explicit U[0,100] sa/sb/sy vs implicit-uniform .stan (0)"),
    ("glmm_poisson", glmm_poisson_s, 0.0, ""),
    ("sum_to_zero", sum_to_zero_s, 0.0, ""),
    ("mvnormal_cov", mvnormal_cov_s, 0.0, ""),
    ("mvnormal_chol", mvnormal_chol_s, 0.0, ""),
    ("mvnormal_prec", mvnormal_prec_s, 0.0, ""),
    ("mvnormal_prec_chol", mvnormal_prec_chol_s, 0.0, ""),
)

const _S_BATCH3 = (
    ("normal_mixture", normal_mixture_s, 0.0, ""),
    ("normal_mixture_k", normal_mixture_k_s, 0.0, ""),
    ("low_dim_gauss_mix", low_dim_gauss_mix_s, 0.0, ""),
    ("low_dim_gauss_mix_collapse", low_dim_gauss_mix_collapse_s, 0.0, ""),
    ("seeds", seeds_s, 0.0, ""),
    ("seeds_centered", seeds_centered_s, 0.0, ""),
    ("surgical", surgical_s, 0.0, ""),
    ("survey", survey_s, 0.0, ""),
)

const _S_BATCH4 = (
    ("m0", m0_s, 0.0, ""),
    ("mb", mb_s, 0.0, ""),
    ("mh", mh_s, 0.0, ""),
    ("mt", mt_s, 0.0, ""),
    ("mth", mth_s, 0.0, ""),
    ("mtbh", mtbh_s, 0.0, ""),
    ("lsat", lsat_s, 0.0, ""),
    ("irt_2pl", irt_2pl_s, 0.0, ""),
)

const _S_BATCH5 = (
    ("hier_2pl", hier_2pl_s, 0.0, ""),
    ("twopl_latent_reg", twopl_latent_reg_s, 0.0, ""),
    ("dogs_hierarchical", dogs_hierarchical_s, 0.0, ""),
    ("dogs_nonhierarchical", dogs_nonhierarchical_s, 0.0, ""),
    ("logistic_rhs", logistic_rhs_s, 0.0, ""),
    ("state_space", state_space_s, 0.0, ""),
    ("prophet", prophet_s, 0.0, ""),
)

const _S_BATCH6 = (
    ("gp_regr", gp_regr_s, 0.0, ""),
    ("hierarchical_gp", hierarchical_gp_s, 0.0, ""),
    ("kronecker_gp", kronecker_gp_s, 0.0, ""),
    ("accel_gp", accel_gp_s, 0.0, ""),
    ("accel_splines", accel_splines_s, 0.0, ""),
    ("bym2", bym2_s, 0.0, ""),
)

for (_label, _cases) in (("s/batch1", _S_BATCH1), ("s/batch2", _S_BATCH2),
        ("s/batch3", _S_BATCH3), ("s/batch4", _S_BATCH4),
        ("s/batch5", _S_BATCH5), ("s/batch6", _S_BATCH6))
    open(OUT, "a") do io
        for t in _cases
            (case, build, offset, reason) = t[1:4]
            if (case, "zeros") in _DONE && (case, "seeded") in _DONE
                println("SB_SWEEP skip $case (already recorded)")
                continue
            end
            recs = sweep_scase(io, case, build; offset=offset, offset_reason=reason)
            for r in recs
                println("SB_SWEEP case=$(r["case"]) label=$(r["label"]) lp=$(r["lp"]) offset=$(r["offset"])")
            end
        end
    end
    println("SB_SWEEP $_label done -> $OUT")
end
