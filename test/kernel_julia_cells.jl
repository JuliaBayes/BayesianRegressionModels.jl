# Julia-first kernel cells: one authored Julia body runs on RKBRMI as written,
# and SBBRMI gives StanBlocks the Stan spellings of its Julia callables and
# Distributions observation constructors.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LogExpFunctions

# A caller's own function that happens to share a Julia name keeps its
# meaning: callables are matched by identity, never by spelling.
module OwnLogistic
logistic(x) = 2x
end

# One helper name, two implementations: StanBlocks traces the `@deffun`, the
# RK backend calls the Julia method.
module DualHelper
using BayesianRegressionModels, Distributions, LogExpFunctions, StanBlocks
StanBlocks.@deffun saturating(x::vector[n], s::real) = 2.0 * inv_logit(s * x) - 1.0
saturating(x::AbstractVector, s::Real) = 2.0 .* logistic.(s .* x) .- 1.0
function build(data)
    @brm data begin
        sigma ~ Exponential(1)
        base ~ Normal(0, 1)
        log_s ~ Normal(0, 0.5)
        amp ~ Normal(0, 1)
        @plate for i in eachindex(c)
            loc[i] = base .+ amp .* saturating(c[i], exp(log_s))
            y[i] ~ Normal(loc[i], sigma)
        end
    end
end
end

const STAN = BRM._sb_cell_statements

@stestset "cell callables take their Stan spelling by identity" begin
    scope = @__MODULE__
    spell(ex) = only(STAN(Any[ex], scope))
    @test spell(:(e = 2.0 .* logistic.(s .* x) .- 1.0)) ==
        :(e = 2.0 .* inv_logit.(s .* x) .- 1.0)
    @test spell(:(z = cumsum(x))) == :(z = cumulative_sum(x))
    @test spell(:(z = logsumexp(x) .+ log1pexp.(x))) ==
        :(z = log_sum_exp(x) .+ log1p_exp.(x))
    @test spell(:(z = LogExpFunctions.logistic(a))) == :(z = inv_logit(a))
    @test spell(:(z = fill(a, n) .+ zeros(n))) ==
        :(z = rep_vector(a, n) .+ rep_vector(0.0, n))
    @test spell(:(z = ones(m, n))) == :(z = rep_matrix(1.0, m, n))
    @test spell(:(z = vcat(a, b, c))) == :(z = append_row(append_row(a, b), c))
    # Stan spellings, Julia names Stan shares, and unbound names pass through.
    for ex in (:(e = 2.0 * inv_logit(s * x) - 1.0), :(z = rep_vector(a, 3)),
            :(z = exp.(x) .+ sum(x)), :(z = my_helper(x; k = 2)))
        @test spell(ex) == ex
    end
    @test only(STAN(Any[:(z = logistic.(x))], OwnLogistic)) == :(z = logistic.(x))
end

@stestset "cell observations take Stan families" begin
    scope = @__MODULE__
    spell(ex) = only(STAN(Any[ex], scope))
    @test spell(:(y ~ Normal(mu, sigma))) == :(y ~ normal(mu, sigma))
    @test spell(:(y ~ LogNormal(mu, sigma))) == :(y ~ lognormal(mu, sigma))
    # Distributions parameterizations follow the top-level translation table.
    @test spell(:(y ~ Exponential(theta))) == :(y ~ exponential(1.0 ./ theta))
    @test spell(:(y ~ censored(Normal(mu, s); lower = lo))) ==
        :(y ~ censored(normal(mu, s); lower = lo))
    @test spell(:(y ~ weighted(Normal(logistic.(mu), s), w))) ==
        :(y ~ weighted(normal(inv_logit.(mu), s), w))
    for ex in (:(y ~ normal(mu, sigma)), :(y ~ weighted(normal, w, mu, sigma)),
            :(y ~ logistic(mu, s)))
        @test spell(ex) == ex
    end
end

const DATA = (; c=[[0.1, 0.5, 1.0], [0.2, 0.4], [0.0, 0.3, 0.6, 0.9]],
    y=[[0.2, 0.1, -0.3], [0.4, 0.0], [0.1, 0.2, 0.3, -0.1]])

julia_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    log_s ~ Normal(0, 0.5)
    amp ~ Normal(0, 1)
    @plate for i in eachindex(c)
        effect = 2.0 .* logistic.(exp(log_s) .* c[i]) .- 1.0
        loc[i] = base .+ amp .* effect
        y[i] ~ Normal(loc[i], sigma)
    end
end

stan_spelled_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    log_s ~ Normal(0, 0.5)
    amp ~ Normal(0, 1)
    @plate for i in eachindex(c)
        effect = 2.0 .* inv_logit.(exp(log_s) .* c[i]) .- 1.0
        loc[i] = base .+ amp .* effect
        y[i] ~ normal(loc[i], sigma)
    end
end

function saturating_oracle(data, names)
    index = Dict(n => findfirst(==(n), names) for n in (:sigma, :base, :log_s, :amp))
    u -> begin
        sigma, base, log_s, amp = exp(u[index[:sigma]]), u[index[:base]],
            u[index[:log_s]], u[index[:amp]]
        logpdf(Exponential(1), sigma) + u[index[:sigma]] + logpdf(Normal(0, 1), base) +
            logpdf(Normal(0, 0.5), log_s) + logpdf(Normal(0, 1), amp) +
            sum(sum(logpdf.(Normal.(base .+ amp .* (2.0 .* logistic.(exp(log_s) .* c) .- 1.0),
                sigma), y)) for (c, y) in zip(data.c, data.y))
    end
end

const MAPPING = [:sigma => "sigma", :base => "base", :log_s => "log_s", :amp => "amp"]

@stestset "one Julia cell runs on RK and compiles to the Stan cell" begin
    saved = deepcopy(DATA)
    julia, stan_spelled = julia_cell(DATA), stan_spelled_cell(DATA)
    julia_code = BRM.stan_code(SBBRMI(julia; mod=@__MODULE__, total_groups=()))
    @test julia_code == BRM.stan_code(SBBRMI(stan_spelled; mod=@__MODULE__, total_groups=()))
    @test occursin("inv_logit", julia_code) && !occursin("logistic", julia_code)
    backend, problem = consumer_problem(julia)
    names = coordinate_names(backend.model.layout)
    @test sort(names) == [:amp, :base, :log_s, :sigma]
    oracle = saturating_oracle(DATA, names)
    stan = consumer_stan(julia, "julia-first-cell")
    # The StanBlocks spelling stays accepted on RK as a compatibility layer.
    stan_backend, stan_problem = consumer_problem(stan_spelled)
    @test coordinate_names(stan_backend.model.layout) == names
    for u in ([0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 0.4], [-0.4, 0.25, -0.3, 0.2])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, MAPPING, backend, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_value, stan_gradient = LogDensityProblems.logdensity_and_gradient(stan_problem, u)
        @test stan_value ≈ value atol=1e-12 rtol=1e-12
        @test stan_gradient ≈ gradient atol=1e-12 rtol=1e-12
    end
    @test isequal(DATA, saved)
end

prims_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    log_s ~ Normal(0, 0.5)
    amp ~ Normal(0, 1)
    @plate for i in eachindex(c)
        z = vcat(0.0, cumsum(abs2.(c[i])))
        w = fill(base, length(c[i])) .+ zeros(length(c[i]))
        loc[i] = w .+ amp .* z[2:(length(c[i]) + 1)] .+
            exp(log_s) .* logsumexp(c[i]) .+ 0.1 .* log1pexp.(c[i])
        y[i] ~ Normal(loc[i], sigma)
    end
end

@stestset "Julia constructors and reductions agree on RK and Stan" begin
    brmi = prims_cell(DATA)
    backend, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "julia-first-prims")
    code = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
    for name in ("append_row", "cumulative_sum", "rep_vector", "log_sum_exp", "log1p_exp")
        @test occursin(name, code)
    end
    for u in ([0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 0.4], [-0.4, 0.25, -0.3, 0.2])
        check_consumer_stan(problem, stan, MAPPING, backend, u)
    end
end

@stestset "one helper name dispatches to each backend's implementation" begin
    brmi = DualHelper.build(DATA)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    oracle = saturating_oracle(DATA, names)
    stan = consumer_stan(brmi, "julia-first-helper"; mod=DualHelper)
    @test occursin("saturating", BRM.stan_code(SBBRMI(brmi; mod=DualHelper, total_groups=())))
    for u in ([0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 0.4])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, MAPPING, backend, u)
    end
end
