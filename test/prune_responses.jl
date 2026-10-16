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
end

# One `@plate` cell computes a shared `level` once and observes three
# responses. `b` reads two predictors, a data index and a scale that nothing
# else reads; `c` reads one predictor and its scale; `a` reads the shared
# computation alone. `report` is an output the cell reports.
plate_abc = @brm begin
    sigma_a ~ Exponential(1)
    sigma_b ~ Exponential(1)
    sigma_c ~ Exponential(1)
    log_k ~ 1 + x + (1 | p | subject)
    base  ~ 1 + (1 | p | subject)
    slope ~ 1 + (1 | p | subject)
    rate  ~ 1 + (1 | p | subject)
    @plate for i in eachindex(t)
        level = exp(log_k[i]) .* t[i]
        report = level ./ 2
        b_loc = base[i] .+ slope[i] .* level[b_idx[i]]
        b[i] ~ normal(b_loc, sigma_b)
        c_loc = rate[i] .* level
        c[i] ~ normal(c_loc, sigma_c)
        a_loc[i] = level
        a[i] ~ normal(a_loc[i], sigma_a)
    end
end
# The same model written without `b`, then without `b` and `c`.
plate_ac = @brm begin
    sigma_a ~ Exponential(1)
    sigma_c ~ Exponential(1)
    log_k ~ 1 + x + (1 | p | subject)
    rate  ~ 1 + (1 | p | subject)
    @plate for i in eachindex(t)
        level = exp(log_k[i]) .* t[i]
        report = level ./ 2
        c_loc = rate[i] .* level
        c[i] ~ normal(c_loc, sigma_c)
        a_loc[i] = level
        a[i] ~ normal(a_loc[i], sigma_a)
    end
end
plate_ac_lkj20 = @brm begin
    sigma_a ~ Exponential(1)
    sigma_c ~ Exponential(1)
    log_k ~ 1 + x + (1 | p | subject)
    rate  ~ 1 + (1 | p | subject)
    @plate for i in eachindex(t)
        level = exp(log_k[i]) .* t[i]
        report = level ./ 2
        c_loc = rate[i] .* level
        c[i] ~ normal(c_loc, sigma_c)
        a_loc[i] = level
        a[i] ~ normal(a_loc[i], sigma_a)
    end
    cor(:, p) ~ LKJCholesky(2, 2.0)
end
plate_a = @brm begin
    sigma_a ~ Exponential(1)
    log_k ~ 1 + x + (1 | p | subject)
    @plate for i in eachindex(t)
        level = exp(log_k[i]) .* t[i]
        report = level ./ 2
        a_loc[i] = level
        a[i] ~ normal(a_loc[i], sigma_a)
    end
end
# A pruned observation after the cell's last output: the pruned cell returns
# that output's value directly, as the formula written without it does.
plate_tail = @brm begin
    sigma_a ~ Exponential(1)
    sigma_b ~ Exponential(1)
    log_k ~ 1 + x + (1 | p | subject)
    base  ~ 1 + (1 | p | subject)
    @plate for i in eachindex(t)
        level = exp(log_k[i]) .* t[i]
        a[i] ~ normal(level, sigma_a)
        a_loc[i] = level .* 2
        b[i] ~ normal(base[i] .+ a_loc[i][b_idx[i]], sigma_b)
    end
end
plate_tail_a = @brm begin
    sigma_a ~ Exponential(1)
    log_k ~ 1 + x + (1 | p | subject)
    @plate for i in eachindex(t)
        level = exp(log_k[i]) .* t[i]
        a[i] ~ normal(level, sigma_a)
        a_loc[i] = level .* 2
    end
end

plate_data = (; subject = ["s1", "s2", "s3"], x = [0.1, -0.4, 0.7],
              t = [[0.5, 1.0], [1.0], [0.25, 0.5, 2.0]],
              b_idx = [[2], [1], [1, 3]],
              a = [[0.1, 0.2], [0.3], [0.0, 0.1, 0.4]],
              b = [[1.1], [0.9], [1.2, 1.4]],
              c = [[0.2, 0.5], [0.4], [0.1, 0.3, 0.9]])
plate_ac_data = Base.structdiff(plate_data, NamedTuple{(:b,)})   # `b` omitted
plate_a_data = Base.structdiff(plate_ac_data, NamedTuple{(:c,)})
rk_source(b) = Base.remove_linenums!.(rk(b))

@testset "a shared @plate cell keeps its kept responses and loses the pruned ones" begin
    for (label, (pruned, reference)) in (
            "pruning `b`: its statements, inputs, predictors and margins go" =>
                (prune_responses(plate_abc(plate_data); keep = (:a, :c)),
                 plate_ac(plate_ac_data)),
            "the default keeps the bound responses" =>
                (prune_responses(plate_abc(plate_ac_data)), plate_ac(plate_ac_data)),
            "marginalize: LKJ(4, 1) restricted to two margins" =>
                (prune_responses(plate_abc(plate_ac_data); marginalize = true),
                 plate_ac_lkj20(plate_ac_data)),
            "pruning `b` and `c` leaves one margin and no correlation" =>
                (prune_responses(plate_abc(plate_data); keep = :a), plate_a(plate_a_data)),
            "a pruned observation after the last output" =>
                (prune_responses(plate_tail(plate_data); keep = :a),
                 plate_tail_a(plate_a_data)))
        @testset "$label" begin
            @test stan(pruned) == stan(reference)
            @test rk_source(pruned) == rk_source(reference)
        end
    end
    pruned = prune_responses(plate_abc(plate_data); keep = (:a, :c))
    @test !any(in(keys(pruned.operations)), (:base, :slope, :sigma_b))
    outputs(b) = Set(o.logical for o in brm_descriptor(SBBRMI(b; mod = @__MODULE__,
                     total_groups = ())).outputs if !isnothing(o.logical))
    # The descriptor names what the formula written without `b` names, and
    # none of the parts only `b` read (which the joint model's does name).
    @test outputs(pruned) == outputs(plate_ac(plate_ac_data))
    @test issubset((:base, :slope), outputs(plate_abc(plate_data)))
    @test isdisjoint((:b, :b_loc, :base, :slope, :sigma_b), outputs(pruned))
    # Keeping every response changes nothing.
    @test stan(prune_responses(plate_abc(plate_data))) == stan(plate_abc(plate_data))
end

@testset "what only a pruned observation reaches goes, outputs computed from it too" begin
    # `shift` reports a value computed from `b_loc`, which only the pruned
    # observation of `b` reaches, so it goes with `b`.
    reported = @brm begin
        sigma_a ~ Exponential(1)
        sigma_b ~ Exponential(1)
        log_k ~ 1 + x + (1 | p | subject)
        base  ~ 1 + (1 | p | subject)
        @plate for i in eachindex(t)
            level = exp(log_k[i]) .* t[i]
            b_loc = base[i] .+ level[b_idx[i]]
            shift = b_loc .- 1
            b[i] ~ normal(b_loc, sigma_b)
            a_loc[i] = level
            a[i] ~ normal(a_loc[i], sigma_a)
        end
    end
    reported_a = @brm begin
        sigma_a ~ Exponential(1)
        log_k ~ 1 + x + (1 | p | subject)
        @plate for i in eachindex(t)
            level = exp(log_k[i]) .* t[i]
            a_loc[i] = level
            a[i] ~ normal(a_loc[i], sigma_a)
        end
    end
    pruned = prune_responses(reported(plate_data); keep = :a)
    @test stan(pruned) == stan(reported_a(plate_a_data))
    @test rk_source(pruned) == rk_source(reported_a(plate_a_data))
    # A kept value that reads a pruned response keeps its observation, as a
    # formula-level likelihood reading a response does.
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
    @test stan(prune_responses(coupled(kernel_data); keep = :w)) == stan(coupled(kernel_data))
end

@testset "an observation-only cell loses its pruned observations" begin
    # The cell is named after its first observed response (`a_cell`), so the
    # formula written without `b` names it the same. SBBRMI does not build
    # observation-only cells yet (StanBlocks snag `result-less-do-f-3076f098`),
    # so this compares the RK source.
    observed = @brm begin
        sigma_a ~ Exponential(1)
        sigma_b ~ Exponential(1)
        log_k ~ 1 + x + (1 | p | subject)
        base  ~ 1 + (1 | p | subject)
        @plate for i in eachindex(t)
            level = exp(log_k[i]) .* t[i]
            a[i] ~ normal(level, sigma_a)
            b[i] ~ normal(base[i] .+ level[b_idx[i]], sigma_b)
        end
    end
    observed_a = @brm begin
        sigma_a ~ Exponential(1)
        log_k ~ 1 + x + (1 | p | subject)
        @plate for i in eachindex(t)
            level = exp(log_k[i]) .* t[i]
            a[i] ~ normal(level, sigma_a)
        end
    end
    pruned = prune_responses(observed(plate_data); keep = :a)
    @test rk_source(pruned) == rk_source(observed_a(plate_a_data))
    @test !any(in(keys(pruned.operations)), (:base, :sigma_b))
end
