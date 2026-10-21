using Test, Random, LinearAlgebra, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan
const BRM = BayesianRegressionModels

# Exact totals beside `offset(...)` summands (snag exact-totals-ref-b3bbb608): an
# offset adds a known term to the predictor and shares no coefficient with the
# totals, so integrating the population coefficients stays exact.
const OFFSET_DATA = let rng = Xoshiro(20261020)
    J, n = 7, 4
    g = repeat(1:J; inner=n)
    x = randn(rng, J * n)
    w = log.(rand(rng, 1:20, J * n))
    y = rand.(rng, Poisson.(exp.(0.3 .+ 0.2 .* x .+ w .+ 0.4 .* randn(rng, J)[g])))
    (; g, x, w, y)
end

data_offset = @brm begin
    log(mu) ~ 1 + x + offset(w) + (1 | a | g)
    effect(mu, Intercept) ~ Normal(1, 2)
    effect(mu, x) ~ Normal(0, 1)
    sd(:, a) ~ Gamma(2, 0.5)
    phi ~ Exponential(1)
    y ~ NegativeBinomial2(mu, phi)
end
model_offset = @brm begin
    theta ~ Normal(0, 0.5)
    log(mu) ~ 1 + x + offset(theta * w) + (1 | g)
    effect(mu, Intercept) ~ Normal(1, 2)
    phi ~ Exponential(1)
    y ~ NegativeBinomial2(mu, phi)
end

# log p_totals(x) = log p_conventional(beta, z) - J log(tau) - log p(beta | totals)
# for every beta: the totals integrate beta, and z = (total - A beta) / tau.
function conventional_marginal(sb, conventional, x, names)
    block = only(total_effect_blocks(sb))
    coords = BRM._total_coordinates(sb, block, names)
    conditional = BRM._total_conditional(block, coords, x)
    beta = conditional.mean .+ 0.3
    original = StanBlocks.stan_instantiate(conventional.model;
        path=joinpath(mktempdir(), "conventional.stan"))
    original_names = BridgeStan.param_unc_names(original.model)
    ranef = only(adaptive_centering_blocks(conventional, original_names))
    q = fill(NaN, length(original_names))
    # `phi` and `theta` have the same names in both programs.
    for (i, name) in pairs(names)
        startswith(name, "pop_") && continue
        j = findfirst(==(name), original_names)
        isnothing(j) || (q[j] = x[i])
    end
    q[ranef.log_scales] = x[coords.scales]
    q[vec(ranef.effects)] = (conditional.totals[:, 1] .- (block.A * beta)[1]) ./ conditional.tau[1]
    # Conventional (Intercept, x) against the totals' recovered Intercept and sampled x.
    population = findall(n -> startswith(n, "pop_"), original_names)
    @test length(population) == 2
    q[population] = [only(beta), x[only(findall(n -> startswith(n, "pop_"), names))]]
    @test !any(isnan, q)
    J = size(conditional.totals, 1)
    BridgeStan.log_density(original.model, q; propto=false) - J * sum(log, conditional.tau) -
        logpdf(MvNormal(conditional.mean, inv(conditional.factor)), beta)
end

@testset "exact totals beside offsets" begin
    for (label, builder) in (("data offset", data_offset), ("model-valued offset", model_offset))
        @testset "$label" begin
            brmi = builder(OFFSET_DATA)
            sb = SBBRMI(brmi; mod=@__MODULE__)
            block = only(total_effect_blocks(sb))
            @test block.predictor === :mu && block.group === :g
            @test block.population_columns == (:Intercept,)
            code = BRM.stan_code(sb)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            @test occursin("rows_dot_product(total_mu", code)
            conventional = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
            @test isempty(total_effect_blocks(conventional))
            problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(mktempdir(), "totals.stan"))
            names = BridgeStan.param_unc_names(problem.model)
            for trial in 1:4
                x = 0.4 .* randn(Xoshiro(trial), length(names))
                @test BridgeStan.log_density(problem.model, x; propto=false) ≈
                    conventional_marginal(sb, conventional, x, names) atol=1e-9
            end
            # The offset stays in the predictor: dropping it changes the density.
            no_offset = SBBRMI(@brm(OFFSET_DATA, begin
                log(mu) ~ 1 + x + (1 | a | g)
                effect(mu, Intercept) ~ Normal(1, 2)
                effect(mu, x) ~ Normal(0, 1)
                sd(:, a) ~ Gamma(2, 0.5)
                phi ~ Exponential(1)
                y ~ NegativeBinomial2(mu, phi)
            end); mod=@__MODULE__)
            @test BRM.stan_code(no_offset) != code
            # Replay and the population program keep the offset.
            @test brm_descriptor(sb) isa BRMDescriptor
            shifted = (; OFFSET_DATA..., w=OFFSET_DATA.w .+ 0.5)
            replay = reprocess(sb, shifted)
            @test BRM.stan_code(replay) == code
            @test replay.data[:w] == shifted.w
            pop = population_model(sb; groups=:g)
            @test StanBlocks.stanc_check(BRM.stan_code(pop); warn_pedantic=false).ok
            @test isempty(setdiff(Set(names), Set(BridgeStan.param_unc_names(
                StanBlocks.stan_instantiate(pop.model; path=joinpath(mktempdir(), "pop.stan")).model))))
        end
    end
end

@testset "the description covers the offset beside the totals" begin
    # A Normal twin: NegativeBinomial2 observations have no scientific
    # description yet, with or without totals (owner todo).
    m = @brm OFFSET_DATA begin
        mu ~ 1 + x + offset(w) + (1 | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    auto = SBBRMI(m; mod=@__MODULE__)
    @test length(total_effect_blocks(auto)) == 1
    r, c = brm_description(auto), brm_description(SBBRMI(m; mod=@__MODULE__, total_groups=()))
    @test r.complete && c.complete
    predictor_ids(d) = Set(x.id for x in d.coverage if first(x.id) === :predictor)
    @test predictor_ids(r) == predictor_ids(c)
end

@testset "an explicit total_groups request covers every block of its group" begin
    data = (; g=OFFSET_DATA.g, x=OFFSET_DATA.x, y=OFFSET_DATA.x .+ 0.1)
    # `sigma`'s block on `g` is ineligible (an unsupported population prior).
    mixed = @brm begin
        mu ~ 1 + x + (1 | g)
        log(sigma) ~ 1 + (1 | g)
        effect(sigma, Intercept) ~ Cauchy(0, 1)
        y ~ Normal(mu, sigma)
    end
    @test isempty(total_effect_blocks(SBBRMI(mixed(data); mod=@__MODULE__)))
    @test_throws "`sigma`" SBBRMI(mixed(data); mod=@__MODULE__, total_groups=[:g])
    @test_throws "Gaussian grouping structure" SBBRMI(mixed(data); mod=@__MODULE__, total_groups=:g)
    @test_throws "no random-effect block" SBBRMI(mixed(data); mod=@__MODULE__, total_groups=[:h])
    # Positive control: both blocks eligible, both integrated.
    eligible = @brm begin
        mu ~ 1 + x + (1 | g)
        log(sigma) ~ 1 + (1 | g)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(eligible(data); mod=@__MODULE__, total_groups=[:g])
    @test Set(b.predictor for b in total_effect_blocks(sb)) == Set([:mu, :sigma])
end
