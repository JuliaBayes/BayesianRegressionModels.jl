# test/prune_responses.jl — `prune_responses`: a model restricted to the
# responses it keeps, before any backend sees it.
#
# The load-bearing check compares emitted programs: pruning a joint model must
# give exactly the program of the formula one would write without the pruned
# responses (Stan source, RK source), and with `marginalize = true` exactly that
# formula with the marginal correlation prior LKJ(a, eta + (K - a)/2) spelled out.
#
# Run: julia --project=test test/prune_responses.jl

using Test
using BayesianRegressionModels
using Random
using Distributions: Exponential, Normal, LKJCholesky

const BRM = BayesianRegressionModels

# A joint model with a shared `|p|` block: `mu` holds two of its margins, `nu`
# the third.
shared = @brm begin
    sigma  ~ Exponential(1)
    sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    nu ~ 1 + (1 | p | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, sigma2)
end
shared_cor = @brm begin
    sigma  ~ Exponential(1)
    sigma2 ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    nu ~ 1 + (1 | p | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, sigma2)
    cor(:, p) ~ LKJCholesky(3, 2.0)
end
# The same model written without `z`, with each correlation prior pruning gives.
only_y = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
end
only_y_lkj15 = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
    cor(:, p) ~ LKJCholesky(2, 1.5)
end
only_y_lkj20 = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
    cor(:, p) ~ LKJCholesky(2, 2.0)
end
only_y_lkj25 = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
    cor(:, p) ~ LKJCholesky(2, 2.5)
end
only_z = @brm begin
    sigma2 ~ Exponential(1)
    nu ~ 1 + (1 | p | subject)
    z ~ Normal(nu, sigma2)
end

rng = MersenneTwister(3)
n = 24
covariates = (; x = randn(rng, n), subject = repeat(11:16; inner = 4))
observed = merge(covariates, (; y = randn(rng, n)))       # `z` omitted
both = merge(observed, (; z = randn(rng, n)))

stan(b) = BRM.stan_code(SBBRMI(b; mod = @__MODULE__, total_groups = ()))
rk(b) = (a = BRM.emit_rk_artifact(b; case_id = "prune"); (a.ast, a.defs))

@testset "pruning gives the program of the formula without the pruned responses" begin
    for (label, (pruned, reference)) in (
            "default: the block's own LKJ at the kept width" =>
                (prune_responses(shared(observed)), only_y(observed)),
            "marginalize: the default LKJ(3, 1) restricted to two margins" =>
                (prune_responses(shared(observed); marginalize = true),
                 only_y_lkj15(observed)),
            "an authored LKJ(3, 2) at the kept width" =>
                (prune_responses(shared_cor(observed)), only_y_lkj20(observed)),
            "an authored LKJ(3, 2) restricted to two margins" =>
                (prune_responses(shared_cor(observed); marginalize = true),
                 only_y_lkj25(observed)),
            "a prior program names the responses it keeps" =>
                (prune_responses(shared(covariates); keep = :y), only_y(covariates)),
            "an omitted response can be kept on its own" =>
                (prune_responses(shared(observed); keep = :z), only_z(covariates)))
        @testset "$label" begin
            @test stan(pruned) == stan(reference)
            @test rk(pruned) == rk(reference)
        end
    end
    # The pruned responses' parts are not in the model at all.
    @test !any(in(keys(prune_responses(shared(observed)).operations)), (:nu, :z, :sigma2))
end

@testset "keeping every response changes nothing" begin
    @test stan(prune_responses(shared(both))) == stan(shared(both))
    @test stan(prune_responses(shared_cor(both); keep = (:y, :z), marginalize = true)) ==
          stan(shared_cor(both))
end

@testset "descriptor and description show only the kept model" begin
    unpruned = brm_descriptor(SBBRMI(shared(observed); mod = @__MODULE__, total_groups = ()))
    pruned = brm_descriptor(SBBRMI(prune_responses(shared(observed)); mod = @__MODULE__,
                                   total_groups = ()))
    logical(d) = Set(o.logical for o in d.outputs if !isnothing(o.logical))
    @test issubset((:nu, :z), logical(unpruned))
    @test isdisjoint((:nu, :z, :sigma2), logical(pruned))
    owners = Set(c.provenance.owner for c in brm_description_components(brm_description(pruned))
                 if hasproperty(c.provenance, :owner))
    @test !(:z in owners) && :y in owners
end

kernel_model = @brm begin
    sigma ~ Exponential(1)
    sigma2 ~ Exponential(1)
    log_cl ~ 1 + weight + (1 | p | subject)
    eta ~ 1 + (1 | p | subject)
    pred ~ kernel(bins, log_cl) do ys, lcl
        mu = sum(ys) .* exp(lcl)
        ys ~ normal(mu, sigma)
        mu
    end
    w ~ Normal(eta, sigma2)
end
kernel_only = @brm begin
    sigma ~ Exponential(1)
    log_cl ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(bins, log_cl) do ys, lcl
        mu = sum(ys) .* exp(lcl)
        ys ~ normal(mu, sigma)
        mu
    end
end
kernel_data = (; subject = ["s1", "s2", "s3"], weight = [60.0, 75.0, 90.0],
               bins = [[0.0, 1.0], [1.0], [0.0, 1.0, 1.0]], w = [0.1, -0.2, 0.3])

@testset "a kernel cell response is kept or pruned with its cell" begin
    # By its data column or by the cell's formal.
    for keep in (:bins, :ys)
        @test stan(prune_responses(kernel_model(kernel_data); keep)) ==
              stan(kernel_only(kernel_data))
    end
    pruned = prune_responses(kernel_model(kernel_data); keep = :w)
    @test !any(in(keys(pruned.operations)), (:pred, :log_cl, :sigma))
end

@testset "refused shapes" begin
    # refused: a kept name must be a response, or the closure would silently
    # drop every likelihood (fail closed).
    @test_throws "`sigma` is not a response" prune_responses(shared(observed); keep = :sigma)
    # refused: a prior program binds no response, so the default keeps nothing.
    @test_throws "no response column is bound" prune_responses(shared(covariates))
    # refused: a block-wide R2D2 budget has no exact restriction to fewer margins.
    r2d2_model = @brm begin
        sigma  ~ Exponential(1)
        sigma2 ~ Exponential(1)
        mu ~ 1 + (1 | p | subject)
        nu ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(mean_R2 = 0.5, prec_R2 = 2, concentration = 1,
                        reference_scale = sigma)
        y ~ Normal(mu, sigma)
        z ~ Normal(nu, sigma2)
    end
    @test_throws "joint R2D2 budget" prune_responses(r2d2_model(observed))
    # refused: a kept kernel cell cannot lose the observation in its body.
    coupled = @brm begin
        sigma ~ Exponential(1)
        sigma2 ~ Exponential(1)
        log_cl ~ 1 + weight + (1 | subject)
        pred ~ kernel(bins, log_cl) do ys, lcl
            mu = sum(ys) .* exp(lcl)
            ys ~ normal(mu, sigma)
            mu
        end
        w ~ Normal(pred, sigma2)
    end
    @test_throws "pruning cannot remove a statement from a cell body" prune_responses(
        coupled(kernel_data); keep = :w)
end
