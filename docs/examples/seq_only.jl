# Bambi replication: https://bambinos.github.io/bambi/notebooks/ordinal_regression.html
# Run: julia --project=. seq_only.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi ordinal_regression replication with BRM (SBBRMI).
# cumulative Trolley (thresholds-only + predictors) + stopping-ratio attrition.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "ordinal")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

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
    keep = findall(n -> (occursin("beta_pop", string(n)) ||
                         (occursin("_thresholds", string(n)) && !occursin("beta", string(n)))),
                   con_names)
    rows = map(keep) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows,
        "keep_names" => string.(con_names[keep]), "keep_draws" => [C[:, i] for i in keep])
end

summ(v) = Dict("mean" => mean(v), "sd" => std(v),
    "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))

# Posterior predictive category counts from joint draws.
# each draw: (thresholds::Vector, eta::Vector) -> probabilities per row (n, K)
function ppc_counts(probmat, y, levels, seed; n_keep=200)
    rng = Xoshiro(seed)
    idx = unique(Int.(round.(range(1, length(probmat); length=n_keep))))
    counts = [zeros(Int, length(levels)) for _ in idx]
    for (j, d) in enumerate(idx)
        Pm = probmat[d]
        U = rand(rng, size(Pm, 1))
        cat = [searchsortedfirst(cumsum(Pm[i, :]), U[i]) for i in axes(Pm, 1)]
        for c in cat
            counts[j][c] += 1
        end
    end
    obs = [sum(y .== l) for l in levels]
    Dict("levels" => levels, "observed" => obs,
        "q05" => [quantile([c[k] for c in counts], 0.05) for k in eachindex(levels)],
        "q50" => [quantile([c[k] for c in counts], 0.5) for k in eachindex(levels)],
        "q95" => [quantile([c[k] for c in counts], 0.95) for k in eachindex(levels)])
end

const LOGISTIC = x -> 1 / (1 + exp(-x))

function cum_probmat(thr_draws, eta_draws)
    # thr_draws: list of K-1 vectors; eta_draws: list of n-vectors
    map(thr_draws, eta_draws) do c, e
        cp = hcat([LOGISTIC.(c[k] .- e) for k in eachindex(c)]...)
        K = length(c) + 1
        P = zeros(length(e), K)
        P[:, 1] = cp[:, 1]
        for k in 2:K-1
            P[:, k] = cp[:, k] - cp[:, k-1]
        end
        P[:, K] = 1 .- cp[:, K-1]
        P
    end
end

function seq_probmat(thr_draws, eta_draws)
    map(thr_draws, eta_draws) do c, e
        K = length(c) + 1
        Q = hcat([LOGISTIC.(c[k] .- e) for k in eachindex(c)]...)
        P = zeros(length(e), K)
        surv = ones(length(e))
        for k in 1:K-1
            P[:, k] = surv .* Q[:, k]
            surv = surv .* (1 .- Q[:, k])
        end
        P[:, K] = surv
        P
    end
end

# Per-draw threshold vectors from kept columns (threshold cols come first).
function thr_matrix(f)
    ti = findall(n -> occursin("_thresholds", n), f["keep_names"])
    cols = [f["keep_draws"][i] for i in ti]
    [[cols[k][d] for k in eachindex(cols)] for d in 1:f["n_draws"]]
end
# Per-draw eta vectors from beta_pop columns and a row-major design matrix.
function eta_matrix(f, X)
    bi = findall(n -> occursin("beta_pop", n), f["keep_names"])
    B = hcat([f["keep_draws"][i] for i in bi]...)  # draws x p
    [vec(X * B[d, :]) for d in 1:f["n_draws"]]
end
# --- model 3: stopping-ratio attrition (YearsAtCompany ~ TotalWorkingYears) ---
adf = CSV.read(joinpath(SCRATCH, "data", "hr_employee_attrition.tsv.txt"), DataFrame; delim='\t')
adf = adf[adf.Attrition .== "No", :]
ya = Int.(adf.YearsAtCompany)
twy0 = Float64.(adf.TotalWorkingYears)
twy = (twy0 .- sum(twy0) / length(twy0)) ./ sqrt(sum((twy0 .- sum(twy0) / length(twy0)) .^ 2) / length(twy0))
open(joinpath(OUT, "seq_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => sum(twy0) / length(twy0),
        "std" => sqrt(sum((twy0 .- sum(twy0) / length(twy0)) .^ 2) / length(twy0)),
        "n" => length(ya)))
end
levels_ya = sort(unique(ya))
b_ord3 = @brm begin
    eta ~ 0 + twy
    effect(eta, twy) ~ Normal(0, 2.5)
    YearsAtCompany ~ Ordinal(StoppingRatio(), LogitLink(), eta)
end
sb3 = SBBRMI(b_ord3((; YearsAtCompany=ya, twy=twy)); mod=@__MODULE__)
problem3 = StanBlocks.stan_instantiate(sb3.model; path=joinpath(SCRATCH, ".out", "stan", "ord_seq.stan"))
f3 = sample_fit(sb3, problem3, 1236)
T3 = thr_matrix(f3)
E3 = eta_matrix(f3, reshape(twy, length(twy), 1))
P3 = seq_probmat(T3, E3)
pmean3 = vec(sum(P3) ./ length(P3))
# posterior-mean stop probabilities expit(c) averaged over draws (notebook cell-24 analog)
stopmean = vec(sum(hcat([LOGISTIC.(c) for c in T3]...); dims=1) ./ length(T3))
open(joinpath(OUT, "seq.json"), "w") do io
    JSON.print(io, Dict("fit" => Dict("n_draws" => f3["n_draws"],
        "divergences" => f3["divergences"], "params" => f3["params"]),
        "levels" => levels_ya, "prob_mean" => pmean3, "stopprob_mean" => stopmean,
        "ppc" => ppc_counts(P3, ya, levels_ya, 779)))
end
println("ORD_SEQ_DONE")
flush(stdout)
