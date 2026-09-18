# Bambi replication: https://bambinos.github.io/bambi/notebooks/categorical_regression.html
# Run: julia --project=. categorical.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# Bambi categorical_regression replication with BRM (SBBRMI).
# Bambi family="categorical": one formula per non-reference level.
# BRM: K-1 explicit linear predictors under CategoricalLogit.
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "categorical")
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
    keep = findall(n -> occursin("beta_pop", string(n)), con_names)
    rows = map(keep) do i
        v = C[:, i]
        Dict("name" => string(con_names[i]), "mean" => mean(v), "sd" => std(v),
            "q05" => quantile(v, 0.05), "q50" => quantile(v, 0.5), "q95" => quantile(v, 0.95))
    end
    popidx = findall(n -> occursin("beta_pop", string(n)), con_names)
    Dict("n_draws" => size(P, 2), "divergences" => fit.n_divergent_samples, "params" => rows,
        "pop_names" => string.(con_names[popidx]), "pop_draws" => [C[:, i] for i in popidx])
end

# --- model 1: simulated A/B/C separation on x (notebook seed 1234) ---
rng = Xoshiro(1234)
x1raw = vcat([randn(rng, 50) .* s .+ m for (m, s) in ((-2.5, 1.2), (0.0, 0.5), (2.5, 1.2))]...)
x1 = (x1raw .- sum(x1raw) / length(x1raw)) ./ sqrt(sum((x1raw .- sum(x1raw) / length(x1raw)) .^ 2) / length(x1raw))
open(joinpath(OUT, "sim_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => sum(x1raw) / length(x1raw),
        "std" => sqrt(sum((x1raw .- sum(x1raw) / length(x1raw)) .^ 2) / length(x1raw))))
end
y1 = vcat(fill(1, 50), fill(2, 50), fill(3, 50))
b_sim = @brm begin
    eta2 ~ 1 + x
    eta3 ~ 1 + x
    effect(eta2, Intercept) ~ Normal(0, 2.5)
    effect(eta2, x) ~ Normal(0, 2.5)
    effect(eta3, Intercept) ~ Normal(0, 2.5)
    effect(eta3, x) ~ Normal(0, 2.5)
    y ~ CategoricalLogit(eta2, eta3)
end
sb = SBBRMI(b_sim((; y=y1, x=x1)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "cat_sim.stan"))
open(joinpath(OUT, "sim.json"), "w") do io
    JSON.print(io, sample_fit(sb, problem, 1234))
end
println("CAT_SIM_DONE")
flush(stdout)

# --- model 2: iris (4 predictors, 3 species) ---
df = CSV.read(joinpath(SCRATCH, "data", "iris.csv"), DataFrame)
codes(v) = (u = sort(unique(v)); m = Dict(x => i for (i, x) in enumerate(u)); [m[x] for x in v])
sp = codes(string.(df.Species))  # setosa=1, versicolor=2, virginica=3 (sorted)
zsc(v) = (v .- sum(v) / length(v)) ./ sqrt(sum((v .- sum(v) / length(v)) .^ 2) / length(v))
sl0 = Float64.(df.var"Sepal.Length"); sw0 = Float64.(df.var"Sepal.Width")
pl0 = Float64.(df.var"Petal.Length"); pw0 = Float64.(df.var"Petal.Width")
stds = Dict(:sl => sqrt(sum((sl0 .- sum(sl0)/length(sl0)) .^ 2) / length(sl0)),
    :sw => sqrt(sum((sw0 .- sum(sw0)/length(sw0)) .^ 2) / length(sw0)),
    :pl => sqrt(sum((pl0 .- sum(pl0)/length(pl0)) .^ 2) / length(pl0)),
    :pw => sqrt(sum((pw0 .- sum(pw0)/length(pw0)) .^ 2) / length(pw0)))
means = Dict(:sl => sum(sl0) / length(sl0), :sw => sum(sw0) / length(sw0),
    :pl => sum(pl0) / length(pl0), :pw => sum(pw0) / length(pw0))
data_iris = (; y=sp, sl=zsc(sl0), sw=zsc(sw0), pl=zsc(pl0), pw=zsc(pw0))
open(joinpath(OUT, "iris_scale.json"), "w") do io
    JSON.print(io, Dict("stds" => stds, "means" => means))
end
b_iris = @brm begin
    eta_versicolor ~ 1 + sl + sw + pl + pw
    eta_virginica ~ 1 + sl + sw + pl + pw
    effect(eta_versicolor, Intercept) ~ Normal(0, 2.5)
    effect(eta_versicolor, sl) ~ Normal(0, 2.5)
    effect(eta_versicolor, sw) ~ Normal(0, 2.5)
    effect(eta_versicolor, pl) ~ Normal(0, 2.5)
    effect(eta_versicolor, pw) ~ Normal(0, 2.5)
    effect(eta_virginica, Intercept) ~ Normal(0, 2.5)
    effect(eta_virginica, sl) ~ Normal(0, 2.5)
    effect(eta_virginica, sw) ~ Normal(0, 2.5)
    effect(eta_virginica, pl) ~ Normal(0, 2.5)
    effect(eta_virginica, pw) ~ Normal(0, 2.5)
    y ~ CategoricalLogit(eta_versicolor, eta_virginica)
end
sb = SBBRMI(b_iris(data_iris); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "cat_iris.stan"))
open(joinpath(OUT, "iris.json"), "w") do io
    JSON.print(io, sample_fit(sb, problem, 1235))
end
println("CAT_IRIS_DONE")
flush(stdout)

# --- model 3: alligator choice (full inline data from the notebook, 63 rows) ---
g_length = [
    1.30, 1.32, 1.32, 1.40, 1.42, 1.42, 1.47, 1.47, 1.50, 1.52, 1.63, 1.65, 1.65, 1.65, 1.65,
    1.68, 1.70, 1.73, 1.78, 1.78, 1.80, 1.85, 1.93, 1.93, 1.98, 2.03, 2.03, 2.31, 2.36, 2.46,
    3.25, 3.28, 3.33, 3.56, 3.58, 3.66, 3.68, 3.71, 3.89, 1.24, 1.30, 1.45, 1.45, 1.55, 1.60,
    1.60, 1.65, 1.78, 1.78, 1.80, 1.88, 2.16, 2.26, 2.31, 2.36, 2.39, 2.41, 2.44, 2.56, 2.67,
    2.72, 2.79, 2.84,
]
g_choice = ["I", "F", "F", "F", "I", "F", "I", "F", "I", "I", "I", "O", "O", "I", "F", "F",
    "I", "O", "F", "O", "F", "F", "I", "F", "I", "F", "F", "F", "F", "F", "O", "O",
    "F", "F", "F", "F", "O", "F", "F", "I", "I", "I", "O", "I", "I", "I", "F", "I",
    "O", "I", "I", "F", "F", "F", "F", "F", "F", "F", "O", "F", "I", "F", "F"]
# Treatment contrast as an explicit 0/1 float column (BRM drops integer-coded
# predictors, so Male/Female is coded 0/1 by hand instead of relying on dummies).
g_sex = map(s -> s == "Male" ? 1.0 : 0.0, vcat(fill("Male", 32), fill("Female", 31)))
len0 = Float64.(g_length)
g_len = (len0 .- sum(len0) / length(len0)) ./ sqrt(sum((len0 .- sum(len0) / length(len0)) .^ 2) / length(len0))
open(joinpath(OUT, "choice_scale.json"), "w") do io
    JSON.print(io, Dict("mean" => sum(len0) / length(len0),
        "std" => sqrt(sum((len0 .- sum(len0) / length(len0)) .^ 2) / length(len0))))
end
# Reference = Other (notebook level order); codes 1=Other, 2=Invertebrates, 3=Fish.
g_y = [c == "O" ? 1 : c == "I" ? 2 : 3 for c in g_choice]
b_choice = @brm begin
    eta_inv ~ 1 + len + sex
    eta_fish ~ 1 + len + sex
    effect(eta_inv, Intercept) ~ Normal(0, 2.5)
    effect(eta_inv, len) ~ Normal(0, 2.5)
    effect(eta_inv, sex) ~ Normal(0, 2.5)
    effect(eta_fish, Intercept) ~ Normal(0, 2.5)
    effect(eta_fish, len) ~ Normal(0, 2.5)
    effect(eta_fish, sex) ~ Normal(0, 2.5)
    y ~ CategoricalLogit(eta_inv, eta_fish)
end
sb = SBBRMI(b_choice((; y=g_y, len=g_len, sex=g_sex)); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(SCRATCH, ".out", "stan", "cat_choice.stan"))
open(joinpath(OUT, "choice.json"), "w") do io
    JSON.print(io, sample_fit(sb, problem, 1236))
end
println("CAT_CHOICE_DONE")
flush(stdout)
