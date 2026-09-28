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

using ReactiveKernelsPPLExamples: Rate2Example, Rate4Example, DugongsGrowthExample

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

# rate_2: two independent Beta-Binomials. (@brm refused: vector-of-sampled-
# params likelihood arg cannot lift to Stan.)
function rate_2_s()
    M = Rate2Example
    return @slic (; k1=M.RATE2_K1, n1=M.RATE2_N1, k2=M.RATE2_K2, n2=M.RATE2_N2) begin
        theta1 ~ beta(1, 1)
        theta2 ~ beta(1, 1)
        k1 ~ binomial(n1, theta1)
        k2 ~ binomial(n2, theta2)
    end
end

# rate_4: rate + prior-only thetaprior. (@brm refused: SBBRMI demotes the
# prior-only param to generated quantities.) StanBlocks demotes
# observation-unreached params in @slic too, so thetaprior rides `0 *` into
# the likelihood — density- and gradient-exact, keeping both sampled.
function rate_4_s()
    M = Rate4Example
    return @slic (; k=M.RATE4_K, n=M.RATE4_N) begin
        theta ~ beta(1, 1)
        thetaprior ~ beta(1, 1)
        theta_eff = theta + 0 * thetaprior
        k ~ binomial(n, theta_eff)
    end
end

# dugongs (BUGS form, inventory): length = α − β·λ^age with (α, β, u_λ,
# log_τ); λ ∈ (0.5,1) lub-mapped (Jacobian log(0.5)+log(s)+log(1−s), exactly
# Stan's), τ = exp(log_τ) (Jacobian log_τ); α/β ~ N(0,1000),
# λ ~ Uniform(0.5,1), τ ~ Gamma(1e-4,1e-4). (The partner's DUG *probe* is the
# von-Bertalanffy form — a different model, ported in sb_sweep_probes.jl.)
function dugongs_s()
    M = DugongsGrowthExample
    return @slic (; ages=M.DUGONGS_AGE, lengths=M.DUGONGS_LENGTH) begin
        alpha ~ normal(0, 1000)
        beta ~ normal(0, 1000)
        lambda ~ uniform(0.5, 1.0; lower=0.5, upper=1.0)
        tau ~ gamma(1e-4, 1e-4)
        sigma = 1 / sqrt(tau)
        mu = alpha - beta * exp(log(lambda) * ages)
        lengths ~ normal(mu, sigma)
    end
end

const _S_BATCH1 = (
    ("rate_2", rate_2_s, 0.0, ""),
    ("rate_4", rate_4_s, 0.0, ""),
    ("dugongs", dugongs_s, 0.0, ""),
)

open(OUT, "a") do io
    for t in _S_BATCH1
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
println("SB_SWEEP s/batch1 done -> $OUT")
