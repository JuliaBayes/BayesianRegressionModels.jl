# Bambi replication: https://bambinos.github.io/bambi/notebooks/alternative_links_binary.html
# Run: julia --project=. links.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi alternative_links_binary: beetle mortality, logit/probit/cloglog.
# probit/cloglog come through BRM's user-link path: the LHS head only needs an
# InverseFunctions inverse with a Stan-known name (Phi / inv_cloglog).
using Random, Statistics, Distributions, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: logit
import InverseFunctions

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "links")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

probit(p) = quantile(Normal(), p)
Phi(x) = cdf(Normal(), x)
InverseFunctions.inverse(::typeof(probit)) = Phi
cloglog(p) = log(-log1p(-p))
inv_cloglog(x) = 1 - exp(-exp(x))
InverseFunctions.inverse(::typeof(cloglog)) = inv_cloglog

function sample_fit(sb, problem, seed; draws=800)
    rp = try
        BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    catch
        problem
    end
    fit = adaptive_warmup_mcmc(
        Xoshiro(seed), rp; init=missing, n_draws=draws,
        nonlinear_adapt=false, monitor_ess=false,
    )
    P = fit.posterior_position
    con_names = BridgeStan.param_names(problem.model)
    C = permutedims(hcat([BridgeStan.param_constrain(problem.model, collect(P[:, i])) for i in axes(P, 2)]...))
    rows = map(eachindex(con_names)) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows,
        "names" => string.(con_names), "draws" => [C[:, i] for i in eachindex(con_names)])
end

x0 = [1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]
nvec = [59, 60, 62, 56, 63, 59, 62, 60]
yvec = [6, 13, 18, 28, 52, 53, 61, 60]
mx = sum(x0) / length(x0)
sx = sqrt(sum((x0 .- mx) .^ 2) / length(x0))
xz = (x0 .- mx) ./ sx
open(joinpath(OUT, "links_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => mx, "std" => sx))
end

b_logit = @brm begin
    logit(p) ~ 1 + xz
    effect(p, Intercept) ~ Normal(0, 5)
    effect(p, xz) ~ Normal(0, 2.5)
    y ~ Binomial(nvec, p)
end
sb = SBBRMI(b_logit((; y=yvec, xz=xz, nvec=nvec)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "link_logit.stan"))
f = sample_fit(sb, problem, 5001)
open(joinpath(OUT, "link_logit.json"), "w") do io
    JSON.print(io, f)
end
println("LINK_LOGIT_DONE")
flush(stdout)

b_probit = @brm begin
    probit(p) ~ 1 + xz
    effect(p, Intercept) ~ Normal(0, 5)
    effect(p, xz) ~ Normal(0, 2.5)
    y ~ Binomial(nvec, p)
end
sb = SBBRMI(b_probit((; y=yvec, xz=xz, nvec=nvec)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "link_probit.stan"))
f = sample_fit(sb, problem, 5002)
open(joinpath(OUT, "link_probit.json"), "w") do io
    JSON.print(io, f)
end
println("LINK_PROBIT_DONE")
flush(stdout)

# cloglog through the user-link path (native inv_cloglog tracer rule;
# snag brm-tracer-inv-c-a19794bc landed). A 2-level ordinal Cumulative()+
# CloglogLink() is loglog-shaped, not cloglog -- verified in brm_ordinal_lpmf.
b_cll = @brm begin
    cloglog(p) ~ 1 + xz
    effect(p, Intercept) ~ Normal(0, 5)
    effect(p, xz) ~ Normal(0, 2.5)
    y ~ Binomial(nvec, p)
end
sb = SBBRMI(b_cll((; y=yvec, xz=xz, nvec=nvec)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "link_cloglog.stan"))
f = sample_fit(sb, problem, 5003)
open(joinpath(OUT, "link_cloglog.json"), "w") do io
    JSON.print(io, f)
end
println("LINK_CLOGLOG_DONE")
flush(stdout)
