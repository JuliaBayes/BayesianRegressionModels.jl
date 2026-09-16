# RBesT gMAP as an adaptive-centering case study: BRM builders, compiled targets,
# coordinate layouts, initial values and hand-written reference densities.
# Source pins and the model derivation are in README.md.
module RBesTCentering
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using Distributions, LinearAlgebra, Statistics, Random, JSON, Serialization, DelimitedFiles
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "..", "pupil_total_effects", "stan_target.jl"))   # BrmsPupilProblem: counted BridgeStan target
const ENZYME_BACKEND = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)

# gMAP with the documented AS priors: beta.prior = 2 (Normal(0, 2) on the logit intercept),
# tau.dist = "HalfNormal", tau.prior = 1.  RBesT: inst/stan/gMAP.stan lines 205, 211-212, 237-238.
const AS_MODEL = @brm begin
    eta ~ 1 + (1 | map | study)
    effect(eta, Intercept) ~ Normal(0, 2)
    sd(:, map) ~ Normal(0, 1)
    r ~ BinomialLogit(n, eta)
end
# gMAP with the documented crohn priors: beta.prior = cbind(0, 88), HalfNormal tau.prior = 44,
# known SE y.se = 88 / sqrt(n).  RBesT: R/crohn.R example, gMAP.stan line 235-236.
const CROHN_MODEL = @brm begin
    theta ~ 1 + (1 | map | study)
    effect(theta, Intercept) ~ Normal(0, 88)
    sd(:, map) ~ Normal(0, 44)
    y ~ Normal(theta, y_se)
end

function read_dataset(name)
    table, header = readdlm(joinpath(@__DIR__, "reference", "datasets", "$name.tsv"), '\t'; header=true)
    cols = Dict(String(h) => table[:, i] for (i, h) in enumerate(vec(header)))
    cols
end

"""Case record: builder, linear-predictor name, family, RBesT prior scales, BRM data, labels."""
function load_case(name)
    cols = read_dataset(name)
    labels = String.(cols["study"]); H = length(labels)
    if name == "AS"
        n = Int.(cols["n"]); r = Int.(cols["r"])
        @assert H == 8 && sum(n) == 513 && sum(r) == 127
        return (; name, builder=AS_MODEL, lp=:eta, response=:r, family=:binomial, beta_sd=2.0, tau_sd=1.0,
            data=(; study=collect(1:H), n, r), H, labels, n, r)
    elseif name == "crohn"
        n = Int.(cols["n"]); y = Float64.(cols["y"]); y_se = 88 ./ sqrt.(n)
        @assert H == 6
        return (; name, builder=CROHN_MODEL, lp=:theta, response=:y, family=:gaussian, beta_sd=88.0, tau_sd=44.0,
            data=(; study=collect(1:H), y, y_se), H, labels, n, y, y_se)
    end
    error("Unknown case $name")
end

"""Per-trial empirical link-scale estimates (RBesT's `theta.strat`, gMAP.R lines 616-651) used only for initialization."""
empirical_totals(c) = c.family == :binomial ? log.((c.r .+ 0.5) ./ (c.n .- c.r .+ 0.5)) : copy(c.y)

"""Automatic exact-total model: one scalar block, Normal population intercept integrated out."""
function total_model(c, out)
    mkpath(out)
    sb = SBBRMI(c.builder(c.data); mod=@__MODULE__)
    block = only(total_effect_blocks(sb)); @assert block.predictor == c.lp
    p = Base.invokelatest(StanBlocks.stan_instantiate, sb.model; path=joinpath(out, "automatic_totals.stan"))
    names = BridgeStan.param_unc_names(p.model)
    coords = BRM._total_coordinates(sb, block, names)
    @assert isempty(coords.mixture) && size(coords.totals) == (c.H, 1) && length(coords.scales) == 1
    @assert block.A ≈ ones(1, 1) && block.location ≈ [0.0] && block.precision ≈ [1 / c.beta_sd^2]
    @assert length(names) == c.H + 1 "unexpected total coordinates: $names"
    (; sb, block, model=p.model, names, coords, kind=:total)
end

"""Conventional model (no totals): BRM's default non-centered block, or the centered block.
BRM's `centered_groups` samples the group DEVIATION `b_j` with its `N(0, tau)` prior (centered on the
scale hierarchy only); the population intercept stays a separate coefficient and `theta = beta + b`.
That is `c = 1` of the adaptive wrapper. It is not RBesT's legacy centered form, which samples
`theta_j` itself (the intercept folded in); that geometry corresponds to BRM's exact totals at `c = 1`."""
function ordinary_model(c, out; centered=false)
    mkpath(out)
    sb = SBBRMI(c.builder(c.data); mod=@__MODULE__, total_groups=(),
        centered_groups=centered ? Set([:study]) : Set{Symbol}())
    label = centered ? "ordinary_cp" : "ordinary_ncp"
    p = Base.invokelatest(StanBlocks.stan_instantiate, sb.model; path=joinpath(out, label * ".stan"))
    names = BridgeStan.param_unc_names(p.model)
    population = only(findall(==("pop_$(c.lp)_beta_pop.1"), names))
    block = only(BRM.adaptive_centering_blocks(sb, names))
    @assert size(block.effects) == (1, c.H) && length(block.log_scales) == 1
    effects = vec(block.effects); log_scale = only(block.log_scales)
    @assert length(names) == c.H + 2 "unexpected ordinary coordinates: $names"
    desc = brm_descriptor(sb)
    tp_names = BridgeStan.param_names(p.model; include_tp=true, include_gq=false)
    lp_cols = brm_output_coordinates(desc, c.lp, tp_names); @assert length(lp_cols) == c.H
    beta_col = only(brm_population_effect_coordinates(desc, c.lp, tp_names; coefficient=:Intercept).coordinates)
    tau_col = only(brm_ranef_sd_coordinates(desc, c.lp, tp_names; id=:map, coefficient=:Intercept).coordinates)
    (; sb, model=p.model, names, population, effects, log_scale, block, desc, tp_names, lp_cols, beta_col, tau_col,
        centered, kind=centered ? :ordinary_cp : :ordinary_ncp)
end

function total_initial(c, m)
    T = empirical_totals(c); q = zeros(length(m.names))
    q[vec(m.coords.totals)] = T; q[only(m.coords.scales)] = log(max(std(T), 0.05)); q
end
function ordinary_initial(c, m)
    T = empirical_totals(c); beta = mean(T); tau = max(std(T), 0.05); q = zeros(length(m.names))
    q[m.population] = beta; q[m.log_scale] = log(tau)
    q[m.effects] = m.centered ? T .- beta : (T .- beta) ./ tau; q
end

logistic(x) = inv(1 + exp(-x))
loglik(c, theta) = c.family == :binomial ?
    sum(logpdf(Binomial(c.n[h], logistic(theta[h])), c.r[h]) for h in 1:c.H) :
    sum(logpdf(Normal(theta[h], c.y_se[h]), c.y[h]) for h in 1:c.H)
half_normal(x, s) = logpdf(Normal(0, s), x) + log(2)

"""Exact log density of the marginalized (total) posterior in BRM's unconstrained coordinates."""
function total_reference(c, m, q)
    T = q[vec(m.coords.totals)]; ltau = q[only(m.coords.scales)]; tau = exp(ltau)
    Sigma = Symmetric(tau^2 * I(c.H) + c.beta_sd^2 * ones(c.H, c.H))
    loglik(c, T) + logpdf(MvNormal(zeros(c.H), Sigma), T) + half_normal(tau, c.tau_sd) + ltau
end
"""Exact log density of the conventional posterior in BRM's unconstrained coordinates (NCP or CP frame)."""
function ordinary_reference(c, m, q)
    beta = q[m.population]; ltau = q[m.log_scale]; tau = exp(ltau); u = q[m.effects]
    theta = m.centered ? beta .+ u : beta .+ tau .* u
    prior = m.centered ? sum(logpdf(Normal(0, tau), x) for x in u) : sum(logpdf(Normal(), x) for x in u)
    loglik(c, theta) + logpdf(Normal(0, c.beta_sd), beta) + half_normal(tau, c.tau_sd) + ltau + prior
end
"""Physical quantities (beta, tau, H totals) for a matrix of ordinary-model draws (coordinates x draws)."""
function ordinary_physical(c, m, positions)
    hcat([begin
        p = BridgeStan.param_constrain(m.model, collect(q); include_tp=true)
        vcat(p[m.beta_col], p[m.tau_col], p[m.lp_cols])
    end for q in eachcol(positions)]...)
end
function finite_gradient(f, q)
    [begin
        h = 1e-5 * max(1, abs(q[k])); plus = copy(q); minus = copy(q)
        plus[k] += h; minus[k] -= h; (f(plus) - f(minus)) / (2h)
    end for k in eachindex(q)]
end
end
