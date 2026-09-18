# Bambi replication: https://bambinos.github.io/bambi/notebooks/survival_model.html
# Run: julia --project=. surv_model.jl   (from docs/examples/, after Pkg.instantiate())
# Inputs: data/ (vendored). Outputs: .out/results/<model>/ + .out/stan/ (gitignored).
# bambi survival_model: Austin cats exponential AFT, intercept-only + by-color.
# Censoring via censored(Exponential(mu); upper=u), months scale (days/31).
# TWO encodings: as-written (adopted -> 'right', matches the notebook's 0.502
# exactly) and corrected (adopted -> observed; MLE receipt in the brief).
using Random, Statistics, Distributions, CSV, DataFrames, JSON
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
SCRATCH = @__DIR__  # docs/examples root (data/ vendored, outputs under .out/)
OUT = joinpath(SCRATCH, ".out", "results", "survmodel")
mkpath(OUT)
mkpath(joinpath(SCRATCH, ".out", "stan"))

function sample_fit(sb, problem, seed; draws=800)
    rp = try
        BRM.adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
    catch
        problem
    end
    fit = adaptive_warmup_mcmc(
        Xoshiro(seed), rp; init=0.2, n_draws=draws,
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

cdf = CSV.read(joinpath(SCRATCH, "data", "AustinCats.csv"), DataFrame; delim=';')
t = Float64.(cdf.days_to_event) ./ 31
adopted = string.(cdf.out_event) .== "Adoption"
black = Float64.(string.(cdf.color) .== "Black")
other = 1.0 .- black
println("CATS_N=", length(t), " ADOPTED=", sum(adopted), " BLACK=", sum(black))
HUGE = 1.0e300
# as-written: adopted -> censored-at-t; corrected: adopted -> observed
u_written = [adopted[i] ? t[i] : HUGE for i in eachindex(t)]
u_fixed = [adopted[i] ? HUGE : t[i] for i in eachindex(t)]

function fit_config(tag, u, bycolor, seed)
    b = if !bycolor
        @brm begin
            log(mu) ~ 1
            effect(mu, Intercept) ~ Normal(0, 2.5)
            time ~ censored(Exponential(mu); upper=u)
        end
    else
        @brm begin
            log(mu) ~ 0 + black + other
            effect(mu, black) ~ Normal(0, 1)
            effect(mu, other) ~ Normal(0, 1)
            time ~ censored(Exponential(mu); upper=u)
        end
    end
    sb = SBBRMI(b((; time=t, u=u, black=black, other=other)); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(SCRATCH, ".out", "stan", "sm_$tag.stan"))
    f = sample_fit(sb, problem, seed)
    open(joinpath(OUT, "$tag.json"), "w") do io
        JSON.print(io, f)
    end
    println("SM_$(uppercase(tag))_DONE")
    flush(stdout)
    return f
end

f_w0 = fit_config("written_int", u_written, false, 33001)
f_wc = fit_config("written_col", u_written, true, 33002)
f_f0 = fit_config("fixed_int", u_fixed, false, 33003)
f_fc = fit_config("fixed_col", u_fixed, true, 33004)

# survival curves S(m) = exp(-m/mu) per draw, by color, both encodings
mgrid = collect(range(0, 12; length=61))
function curves(f)
    bi = findall(n -> occursin("beta_pop", n), f["names"])
    B = hcat([f["draws"][i] for i in bi]...)
    iscol = size(B, 2) == 2
    # NOTE: column order is declaration order (black, other); verified post-fit
    # against f["names"] before briefing.
    map(mgrid) do m
        mu_o = exp.(iscol ? B[:, 2] : B[:, 1])
        mu_b = exp.(iscol ? B[:, 1] : B[:, 1])
        Dict("month" => m,
            "black" => mean(exp.(-m ./ mu_b)),
            "other" => mean(exp.(-m ./ mu_o)))
    end
end
open(joinpath(OUT, "sm_curves.json"), "w") do io
    JSON.print(io, Dict("written" => curves(f_wc), "fixed" => curves(f_fc)))
end
println("SM_CURVES_DONE")
flush(stdout)
