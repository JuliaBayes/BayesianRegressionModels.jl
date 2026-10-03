# Generated random effects coexist with fitted covariate coordinates when
# endpoint observations are absent. Use the program's carrier roles, never
# absence from a caller-supplied name list, to decide which blocks to transport.
using Test
using BayesianRegressionModels
using StanBlocks
using BridgeStan
using Distributions
using Random

const TOY = (; subject=[1, 1, 2, 2],
    w=Union{Missing,Float64}[0.4, missing, 1.8, -0.3],
    v=[0.2, 0.9, 1.4, -0.1], design=[0.1, 0.2, 0.3, 0.4])
const BUILDER = @brm begin
    location ~ Normal(0, 1)
    mi(w) ~ Normal(location, 0.7)
    v ~ Normal(w, 0.6)
    mu ~ 1 + standardize(w) + offset(design) + (1 + design | p | subject)
    y ~ Normal(mu, 1)
end
const PRIOR = SBBRMI(BUILDER(TOY); mod=@__MODULE__)
const FIT_DATA = merge(TOY, (; y=[0.1, 0.8, 1.2, -0.4]))
const FIT = SBBRMI(BUILDER(FIT_DATA); mod=@__MODULE__)
const FUTURE = merge(TOY, (; design=[1.0, 1.2, 1.4, 1.6]))
const REPLAY = reprocess(PRIOR, FUTURE; freeze_constants=true)
const CV = reprocess(PRIOR, TOY; freeze_constants=true, resample_groups=[:subject])

@testset "generated roles come from the traced carrier" begin
    for model in (PRIOR, generative_plan(PRIOR), REPLAY, CV)
        block = only(ranef_blocks(model))
        @test block.generated
        @test (block.n_terms, block.n_groups) == (2, 2)
        @test block.levels == [1, 2]
        outputs = Dict(o.name => o.kind for o in StanBlocks.stan_descriptor(model.model).outputs)
        @test outputs[block.z] === :generated_quantity
        # Generated quantities have no sampler coordinates to resolve.
        @test_throws "generated quantities" ranef_coordinates(block, String[])
    end
    @test !only(ranef_blocks(FIT)).generated
    cv_fit = SBBRMI(BUILDER(FIT_DATA); mod=@__MODULE__, cv_groups=[:subject])
    @test !only(ranef_blocks(cv_fit)).generated
end

function named_values(sb, problem, q, logical, seed)
    values = BridgeStan.param_constrain(problem.model, q;
        include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, seed))
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    values[brm_output_coordinates(brm_descriptor(sb), logical, names)]
end

@testset "nonzero prior transport retains fitted completion and frozen design" begin
    source = StanBlocks.stan_instantiate(PRIOR.model)
    names = BridgeStan.param_unc_names(source.model)
    @test BridgeStan.param_unc_num(source.model) == 2
    @test Set(names) == Set(["location", "w_y_mis.1"])
    draws = [0.15 -0.2; -0.3 0.7; 0.4 0.1]
    source_copy = copy(draws)
    prior_w = named_values(PRIOR, source, draws[1, :], :w, 41)
    @test prior_w == [0.4, -0.2, 1.8, -0.3]
    # Independent normalized covariate likelihood: endpoint-only priors have
    # become RNG draws and contribute no density to the retained coordinates.
    for row in axes(draws, 1)
        at = Dict(name => draws[row, j] for (j, name) in enumerate(names))
        complete = [0.4, at["w_y_mis.1"], 1.8, -0.3]
        expected = logpdf(Normal(), at["location"]) +
            sum(logpdf.(Normal(at["location"], 0.7), complete)) +
            sum(logpdf.(Normal.(complete, 0.6), TOY.v))
        @test BridgeStan.log_density(source.model, draws[row, :]; propto=false) ≈ expected
    end
    for target in (PRIOR, REPLAY, CV)
        problem = StanBlocks.stan_instantiate(target.model)
        target_names = BridgeStan.param_unc_names(problem.model)
        moved = transport_draws(generative_plan(PRIOR), target, draws, names, target_names)
        @test size(moved) == (size(draws, 1), length(target_names))
        for (j, name) in enumerate(target_names)
            @test moved[:, j] == draws[:, findfirst(==(name), names)]
        end
        @test target.data[:standardize_w_mean] == PRIOR.data[:standardize_w_mean]
        @test target.data[:standardize_w_scale] == PRIOR.data[:standardize_w_scale]
        for row in axes(draws, 1), seed in (41, 42)
            @test named_values(target, problem, moved[row, :], :w, seed) ==
                  named_values(PRIOR, source, draws[row, :], :w, seed)
            @test isfinite(BridgeStan.log_density(problem.model, moved[row, :]))
        end
        # GQ randomness stays in Stan; multiple seeds redraw structural effects.
        names_gq = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        block = only(ranef_blocks(target))
        output = only(o for o in brm_descriptor(target).outputs if o.name === block.z)
        indices = brm_output_coordinates(output, names_gq)
        gq(seed) = BridgeStan.param_constrain(problem.model, moved[1, :];
            include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, seed))
        @test gq(41)[indices] != gq(42)[indices]
    end
    @test draws == source_copy
    problem = StanBlocks.stan_instantiate(REPLAY.model)
    @test named_values(PRIOR, source, draws[1, :], :mu, 41) !=
          named_values(REPLAY, problem, draws[1, :], :mu, 41)
end

@testset "sampled posterior and CV transport keep strict coordinate checks" begin
    source = StanBlocks.stan_instantiate(FIT.model)
    names = BridgeStan.param_unc_names(source.model)
    draws = reshape(collect(0.02:0.02:2length(names)*0.02), 2, length(names))
    @test transport_draws(FIT, FIT, draws, names, names) == draws
    cv = reprocess(FIT, FIT_DATA; resample_groups=[:subject])
    target = StanBlocks.stan_instantiate(cv.model)
    target_names = BridgeStan.param_unc_names(target.model)
    moved = transport_draws(FIT, cv, draws, names, target_names)
    for (j, name) in enumerate(target_names)
        @test moved[:, j] == draws[:, findfirst(==(name), names)]
    end
    # Missing one sampled cell means misaligned draws, not a generated block.
    cell = first(ranef_coordinates(only(ranef_blocks(FIT)), names))
    shortened = names[setdiff(eachindex(names), [cell])]
    @test_throws "expects unconstrained coordinates" transport_draws(FIT, FIT, draws, names, shortened)
    @test_throws "expects unconstrained coordinates" transport_draws(FIT, FIT,
        draws[:, setdiff(eachindex(names), [cell])], shortened, names)
end
