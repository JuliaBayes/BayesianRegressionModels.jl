# test/sb_sweep_probes.jl — SB parity probes REQUESTED by the RK half.
#
# (ReactiveKernels:brm:kernel:everything, test_sweep_replicate.jl header on
# lane kb-impl/ReactiveKernels-brm-kernel-everything @ c5f6c1af: "REQUEST to
# BayesianRegressionModels:rk:kernel:everything (SB parity probes; Stan
# propto=false with Jacobians, BridgeStan AD grads)".)
#
# Partner-specified data + CONSTRAINED probes; each probe maps to
# unconstrained via BridgeStan param_unconstrain on the compiled SB model
# (exact, no hand transcription) and reports lp + AD grad. Every probe
# carries an independent Distributions.jl hand oracle (transcribed from the
# partner's own want formulas) asserted at rtol 1e-9.
#
# R1/R3/ARK now. R2/R4/DUG follow with the S driver: R2's vector-of-params
# likelihood arg is refused at Stan lifting, R4's prior-only thetaprior is
# demoted to generated quantities, and DUG is nonlinear — all three need
# @slic ports (see sb_sweep_core.jl MOVED notes).
#
# Run: julia --project=test test/sb_sweep_probes.jl
# Records append to $SB_PROBE_OUT (default: tempdir()/sb-sweep-probes.jsonl).

include(joinpath(@__DIR__, "sb_sweep_common.jl"))

using Distributions: Beta, Binomial, Normal, Cauchy

const PROBE_OUT = get(ENV, "SB_PROBE_OUT",
    joinpath(tempdir(), "sb-sweep-probes.jsonl"))

# Compile (cached by Stan hash); return model + unconstrained names.
function _probe_model(case::AbstractString, sb::BRM.SBBRMI)
    code = BRM.stan_code(sb)
    stan_sha = _short_sha(code)
    mkpath(_SWEEP_CACHE)
    path = joinpath(_SWEEP_CACHE, "probe-$(case)-$(stan_sha).stan")
    isfile(path) || write(path, code)
    problem = Base.invokelatest(StanBlocks.stan_instantiate, sb.model; path=path)
    return problem.model, BS.param_unc_names(problem.model), stan_sha, code
end

# Evaluate at a CONSTRAINED probe (name => value); assert the names, map via
# BridgeStan, and cross-check the independent oracle.
function probe_case(io::IO, probe::AbstractString, sb::BRM.SBBRMI,
        x_phys::AbstractVector, oracle::Float64,
        expect_names::AbstractVector{<:AbstractString})
    model, names, stan_sha, _ = _probe_model(probe, sb)
    @assert names == expect_names "probe $probe: names $(names) != " *
        "expected $(expect_names) — physical probe mis-mapped?"
    u = BS.param_unconstrain(model, Vector{Float64}(x_phys))
    grad = zeros(length(u))
    lp, _ = BS.log_density_gradient!(model, Vector{Float64}(u), grad;
        propto=false, jacobian=true)
    @assert isfinite(lp) "non-finite lp for probe $probe"
    match = isapprox(lp, oracle; rtol=1e-9)
    @assert match "probe $probe: SB lp $lp != oracle $oracle"
    data = BRM.stan_data(sb)
    rec = Dict(
        "probe" => String(probe), "brm_tip" => _brm_tip(),
        "stan_sha" => stan_sha, "data_hash" => _short_sha(_canonical_data(data)),
        "stan_names" => names, "x_phys" => Vector{Float64}(x_phys),
        "u" => Vector{Float64}(u), "lp" => lp, "grad" => grad,
        "oracle" => oracle,
    )
    println(io, JSON.json(rec))
    println("SB_PROBE probe=$probe lp=$lp oracle=$oracle grad=$grad")
    return rec
end

const DL = Distributions.logpdf

# R1: theta ~ Beta(1,1), k ~ Binomial(n, theta); k=[6], n=[10]; theta=0.6.
function probe_r1()
    df = (; k=[6], n=[10])
    builder = @brm begin
        theta ~ Beta(1, 1)
        k ~ Binomial(n, theta)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    oracle = DL(Beta(1.0, 1.0), 0.6) + DL(Binomial(10, 0.6), 6) + log(0.6 * 0.4)
    return sb, [0.6], oracle
end

# R3: same model; k=[6,8], n=[10,12]; theta=0.6.
function probe_r3()
    df = (; k=[6, 8], n=[10, 12])
    builder = @brm begin
        theta ~ Beta(1, 1)
        k ~ Binomial(n, theta)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    oracle = DL(Beta(1.0, 1.0), 0.6) + DL(Binomial(10, 0.6), 6) +
        DL(Binomial(12, 0.6), 8) + log(0.6 * 0.4)
    return sb, [0.6], oracle
end

# ARK: alpha/b ~ Normal(0,10), sigma ~ truncated-Cauchy(0,2.5) NORMALIZED
# (+log2, per the partner's Stan-faithful oracle); K=2 synthetic design.
function probe_ark()
    yt = [1.0, 2.0, 1.5]
    l1 = [0.5, 1.0, 2.0]
    l2 = [0.2, 0.5, 1.0]
    df = (; y=yt, lag1=l1, lag2=l2)
    builder = @brm begin
        sigma ~ truncated(Cauchy(0, 2.5); lower=0.0)
        mu ~ 1 + lag1 + lag2
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, lag1) ~ Normal(0, 10)
        effect(mu, lag2) ~ Normal(0, 10)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    # x_phys in [sigma, pop.1, pop.2, pop.3] order (asserted at runtime).
    x = [1.5, 1.0, 0.5, -0.25]
    mu = 1.0 .+ 0.5 .* l1 .+ (-0.25) .* l2
    oracle = sum(DL(Normal(m, 1.5), y) for (m, y) in zip(mu, yt)) +
        sum(DL(Normal(0.0, 10.0), c) for c in (1.0, 0.5, -0.25)) +
        DL(Cauchy(0.0, 2.5), 1.5) + log(2) + log(1.5)
    return sb, x, oracle
end

open(PROBE_OUT, "a") do io
    for (probe, build, names) in (
        ("R1", probe_r1, ["theta"]),
        ("R3", probe_r3, ["theta"]),
        ("ARK", probe_ark,
            ["sigma", "pop_mu_beta_pop.1", "pop_mu_beta_pop.2", "pop_mu_beta_pop.3"]),
    )
        sb, x, oracle = build()
        probe_case(io, probe, sb, x, oracle, names)
    end
end
println("SB_PROBE wrote $PROBE_OUT")
