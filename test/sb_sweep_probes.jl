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
# R1/R3/ARK via @brm; R2/R4/DUG via inline @slic (R2's vector-of-params
# likelihood arg is refused at Stan lifting, R4's prior-only thetaprior is
# demoted to GQ, DUG is nonlinear — see sb_sweep_core.jl MOVED notes).
# DUG data discrepancy: the header REQUEST names ages=[0.5,1,2],
# lengths=[0.8,1.6,2.1] but _sr_dug_cols() uses [1,5,9]/[1,2,1.5] — pinned
# BOTH (DUG0/DUG1 on code data, DUGh on header data).
#
# Run: julia --project=test test/sb_sweep_probes.jl
# Records append to $SB_PROBE_OUT (default: tempdir()/sb-sweep-probes.jsonl).

include(joinpath(@__DIR__, "sb_sweep_common.jl"))

using Distributions: Beta, Binomial, Normal, Cauchy, Exponential
import Distributions

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

# R2: two independent Beta-Binomials; probe (0.6, 0.7).
function probe_r2()
    df = (; k1=[6], n1=[10], k2=[8], n2=[12])
    model = @slic (; k1=df.k1, n1=df.n1, k2=df.k2, n2=df.n2) begin
        theta1 ~ beta(1, 1)
        theta2 ~ beta(1, 1)
        k1 ~ binomial(n1, theta1)
        k2 ~ binomial(n2, theta2)
    end
    oracle = DL(Beta(1.0, 1.0), 0.6) + DL(Binomial(10, 0.6), 6) +
        log(0.6 * 0.4) + DL(Beta(1.0, 1.0), 0.7) +
        DL(Binomial(12, 0.7), 8) + log(0.7 * 0.3)
    return model, [0.6, 0.7], oracle
end

# R4: R1 + prior-only thetaprior; probe (0.6, 0.4). StanBlocks demotes
# observation-unreached params to GQ (§34 activity analysis), so thetaprior
# rides `0 *` into the likelihood: density- and gradient-exact (0*x == 0 for
# finite x), keeping it a sampled parameter with exactly its Beta terms.
function probe_r4()
    df = (; k=[6], n=[10])
    model = @slic (; k=df.k, n=df.n) begin
        theta ~ beta(1, 1)
        thetaprior ~ beta(1, 1)
        theta_eff = theta + 0 * thetaprior
        k ~ binomial(n, theta_eff)
    end
    oracle = DL(Beta(1.0, 1.0), 0.6) + DL(Binomial(10, 0.6), 6) +
        log(0.6 * 0.4) + DL(Beta(1.0, 1.0), 0.4) + log(0.4 * 0.6)
    return model, [0.6, 0.4], oracle
end

# DUG: von-Bertalanffy (the partner's probe model — NOT the inventory
# BUGS-form dugongs). mu = Linf*(1-exp(-kk*(age-t0))), y ~ Normal(mu,sigma).
function _dug_model(y, age)
    return @slic (; y=y, age=age) begin
        Linf ~ normal(2.0, 1.0)
        kk ~ normal(0.0, 1.0)
        t0 ~ normal(0.0, 1.0)
        sigma ~ exponential(1.0)
        mu = Linf * (1 - exp(-kk * (age - t0)))
        y ~ normal(mu, sigma)
    end
end

function _dug_oracle(y, age, Linf, kk, t0, sigma, u_sigma)
    mu = Linf .* (1 .- exp.(-kk .* (age .- t0)))
    return sum(DL(Normal(m, sigma), v) for (m, v) in zip(mu, y)) +
        DL(Normal(2.0, 1.0), Linf) + DL(Normal(0.0, 1.0), kk) +
        DL(Normal(0.0, 1.0), t0) + DL(Exponential(1.0), sigma) + u_sigma
end

# probe_case for SlicModel probes (mirrors the SBBRMI twin).
function probe_scase(io::IO, probe::AbstractString, model,
        x_phys::AbstractVector, oracle::Float64,
        expect_names::AbstractVector{<:AbstractString})
    code = StanBlocks.stan_code(model)
    stan_sha = _short_sha(code)
    mkpath(_SWEEP_CACHE)
    path = joinpath(_SWEEP_CACHE, "probe-$(probe)-$(stan_sha).stan")
    isfile(path) || write(path, code)
    problem = Base.invokelatest(StanBlocks.stan_instantiate, model; path=path)
    names = BS.param_unc_names(problem.model)
    @assert names == expect_names "probe $probe: names $(names) != " *
        "expected $(expect_names) — physical probe mis-mapped?"
    u = BS.param_unconstrain(problem.model, Vector{Float64}(x_phys))
    grad = zeros(length(u))
    lp, _ = BS.log_density_gradient!(problem.model, Vector{Float64}(u), grad;
        propto=false, jacobian=true)
    @assert isfinite(lp) "non-finite lp for probe $probe"
    match = isapprox(lp, oracle; rtol=1e-9)
    @assert match "probe $probe: SB lp $lp != oracle $oracle"
    data = StanBlocks.stan_data(model)
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
    for (probe, build, names) in (
        ("R2", probe_r2, ["theta1", "theta2"]),
        ("R4", probe_r4, ["theta", "thetaprior"]),
    )
        model, x, oracle = build()
        probe_scase(io, probe, model, x, oracle, names)
    end
    # DUG on code data (u0 + u1) and header data (u1 physical).
    dug_names = ["Linf", "kk", "t0", "sigma"]
    let y = [1.0, 2.0, 1.5], age = [1.0, 5.0, 9.0]
        probe_scase(io, "DUG0", _dug_model(y, age), [0.0, 0.0, 0.0, 1.0],
            _dug_oracle(y, age, 0.0, 0.0, 0.0, 1.0, 0.0), dug_names)
        probe_scase(io, "DUG1", _dug_model(y, age), [1.0, 1.0, 1.0, exp(1.0)],
            _dug_oracle(y, age, 1.0, 1.0, 1.0, exp(1.0), 1.0), dug_names)
    end
    let y = [0.8, 1.6, 2.1], age = [0.5, 1.0, 2.0]
        probe_scase(io, "DUGh", _dug_model(y, age), [1.0, 1.0, 1.0, exp(1.0)],
            _dug_oracle(y, age, 1.0, 1.0, 1.0, exp(1.0), 1.0), dug_names)
    end
end
println("SB_PROBE wrote $PROBE_OUT")
