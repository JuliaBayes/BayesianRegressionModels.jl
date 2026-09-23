# Run with --project=test; no sampling.
#
# A lambda handed to a higher-order `@deffun` in a `@brm` top-level assignment --
# written inline as the first argument, or as a trailing `do` block -- reaches
# StanBlocks as a closure. The body may read sampled parameters and shared data
# vectors. Before this seam the lifter refused it (`cannot lift to Stan
# expression: Expr: l->…`), so a function-valued argument could only be a name.

using Test, Random
using BayesianRegressionModels, StanBlocks
using Distributions: Normal, Poisson
import LogDensityProblems

# The callee lives in ANOTHER module, as a package's operators would.
module ClosureOps
using StanBlocks
StanBlocks.@deffun begin
    # Y[t] = sum_{l = 0}^{L - 1} f(l) * x[t - l]; x before row 1 counts as 0
    lagged(f, x::vector[T], L::int)::vector[T] = begin
        Y::vector[T]
        for t in 1:T
            acc = 0.0
            for l in 0:(L - 1)
                if t - l >= 1
                    acc += f(l) * x[t - l]
                end
            end
            Y[t] = acc
        end
        Y
    end
    # the same filter with its weights as a data vector
    lagged_weights(w::vector[L], x::vector[T])::vector[T] = begin
        Y::vector[T]
        for t in 1:T
            acc = 0.0
            for l in 0:(L - 1)
                if t - l >= 1
                    acc += w[l + 1] * x[t - l]
                end
            end
            Y[t] = acc
        end
        Y
    end
end
end # module
using .ClosureOps: lagged, lagged_weights

const closure_df = (; time=collect(1.0:12), y=[3, 4, 6, 5, 8, 9, 12, 11, 15, 18, 17, 21],
                      w=[0.5, 0.3, 0.2])

stan(brmi; kwargs...) = BayesianRegressionModels.stan_code(SBBRMI(brmi; mod=@__MODULE__, kwargs...))
stanc_ok(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok
function density(brmi, q)
    problem = StanBlocks.stan_instantiate(SBBRMI(brmi; mod=@__MODULE__).model)
    @assert LogDensityProblems.dimension(problem) == length(q)
    LogDensityProblems.logdensity_and_gradient(problem, q)
end

inline_lambda = @brm closure_df begin
    rate ~ Normal(1.0, 0.5; lower=0.0)
    log_mu ~ 1 + rw(time)
    Y = lagged(l -> exp(-rate * l), exp(log_mu), 3)
    y ~ Poisson(Y)
end
do_block = @brm closure_df begin
    rate ~ Normal(1.0, 0.5; lower=0.0)
    log_mu ~ 1 + rw(time)
    Y = lagged(exp(log_mu), 3) do l
        exp(-rate * l)
    end
    y ~ Poisson(Y)
end
# the body reads a shared DATA vector (`w` has 3 elements, not one per row)
captures_data = @brm closure_df begin
    log_mu ~ 1 + rw(time)
    Y = lagged(exp(log_mu), 3) do l
        w[l + 1]
    end
    y ~ Poisson(Y)
end
weights_vector = @brm closure_df begin
    log_mu ~ 1 + rw(time)
    Y = lagged_weights(w, exp(log_mu))
    y ~ Poisson(Y)
end

@testset "a lambda argument transpiles, inline and as a do block" begin
    inline_code, do_code = stan(inline_lambda), stan(do_block)
    @test stanc_ok(inline_code)
    # `do` is only another spelling of the first positional argument
    @test do_code == inline_code
    # StanBlocks lifts the closure into a Stan function and passes the captured
    # PARAMETER as an argument -- it is sampled, not frozen into the program
    @test occursin("lagged_closure_1(rate, exp(log_mu), 3)", inline_code)
    @test occursin("real<lower=0.0> rate;", inline_code)
end

@testset "the body may capture a shared data vector" begin
    code = stan(captures_data)
    @test stanc_ok(code)
    @test occursin(r"vector\[w_n\] w;", code)                 # registered as Stan data
    # same model as the data-vector spelling: identical density and gradient
    q = 0.1 .* randn(Xoshiro(3), 13)
    lp_closure, grad_closure = density(captures_data, q)
    lp_vector, grad_vector = density(weights_vector, q)
    @test isfinite(lp_closure) && all(isfinite, grad_closure)
    @test lp_closure ≈ lp_vector
    @test grad_closure ≈ grad_vector
end

@testset "a sampled parameter inside the body is differentiated" begin
    q = 0.1 .* randn(Xoshiro(4), 14)
    lp, grad = density(do_block, q)
    @test isfinite(lp) && all(isfinite, grad)
    # coordinate 1 is `rate` (declared first): the likelihood depends on it through the closure
    shifted = copy(q); shifted[1] += 0.5
    @test density(do_block, shifted)[1] != lp
    @test grad[1] != 0.0
end

@testset "prior-only build, printing, and the other backends" begin
    # The prior spelling keeps the model identical and omits the response
    # column: the closure-backed observation forward-simulates.
    do_block_prior = @brm (; time=closure_df.time) begin
        rate ~ Normal(1.0, 0.5; lower=0.0)
        log_mu ~ 1 + rw(time)
        Y = lagged(exp(log_mu), 3) do l
            exp(-rate * l)
        end
        y ~ Poisson(Y)
    end
    @test stanc_ok(stan(do_block_prior))
    @test occursin("lagged", sprint(show, do_block))          # the statement prints
    # the callee is a Stan function: Turing refuses by name rather than lowering row-wise
    @test_throws "Y" TuringBRMI(do_block)
end

@testset "anything else that is not a lambda still refuses loudly" begin
    bad = @brm closure_df begin
        log_mu ~ 1 + rw(time)
        Y = lagged_weights(w, exp(log_mu))
        y ~ Poisson(Y)
    end
    @test stanc_ok(stan(bad))                                 # control: the vector form is fine
    @test_throws "cannot lift to Stan expression" BayesianRegressionModels._sb_scalar_expr(:(a + b), Dict{Symbol,Any}())
end
