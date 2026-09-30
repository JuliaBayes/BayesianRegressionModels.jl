# Run in a fresh process: julia --threads=4 --project=test test/sbimpl_generation_concurrency.jl
using Test, BayesianRegressionModels, Distributions
import StanBlocks

const BRM = BayesianRegressionModels
include("concurrent_builds.jl")
include("testset_filter.jl")

const GENERATION_DATA = (; x=[-1.0, 0.0, 1.0], z=[-1.0, 0.0, 1.0],
                         y=[-0.4, 0.1, 0.7])
const VECTOR_X = @brm begin
    mu ~ 1 + x
    effect(mu, Intercept) ~ TDist(7)
    effect(mu, x) ~ Normal(0, 3)
    y ~ Normal(mu, 0.9)
end

same_owner_prior(location, scale) = Normal(location, scale)
BRM.brm_distribution_type(::typeof(same_owner_prior)) = Normal
BRM._sb_stan_dist_name(::typeof(same_owner_prior)) = :same_owner_family

function check_same_named_modules()
    # Serial fixture declarations simulate separate consumers with the same
    # printed module name. Only model construction below runs concurrently.
    owners = map((0.25, 1.75)) do shift
        mod = Module(:RepeatedPriorOwner)
        Core.eval(mod, :(using StanBlocks))
        Core.eval(mod, quote
            StanBlocks.@deffun begin
                same_owner_family_rng(location::real, scale::real)::real =
                    normal_rng(location + $shift, scale)
                @lpxf same_owner_family_lpdf(y::real, location::real,
                                            scale::real)::real =
                    normal_lpdf(y, location + $shift, scale)
            end
        end)
        mod
    end
    @test owners[1] !== owners[2]
    @test string(owners[1]) == string(owners[2])
    priors = [BRM.ExprColumn(same_owner_prior, 0.0, 1.0),
              BRM.ExprColumn(Normal, 0.0, 2.0)]
    families = concurrent_builds(repeat(collect(owners), 4)) do owner
        first(BRM._sb_vector_prior_family(priors; positive=false, mod=owner))
    end
    @test families[1] !== families[2]
    @test all(parentmodule(family) === owner
              for (family, owner) in zip(families, repeat(collect(owners), 4)))
    @test all(family === families[1] for family in families[1:2:end])
    @test all(family === families[2] for family in families[2:2:end])

    builder = @brm begin
        mu ~ 1 + x
        effect(mu, Intercept) ~ same_owner_prior(0, 1)
        effect(mu, x) ~ Normal(0, 2)
        y ~ Normal(mu, 0.9)
    end
    codes = concurrent_builds(repeat(collect(owners), 4)) do owner
        BRM.stan_code(SBBRMI(builder(GENERATION_DATA); mod=owner))
    end
    @test all(code == codes[1] for code in codes[1:2:end])
    @test all(code == codes[2] for code in codes[2:2:end])
    @test occursin("0.25", codes[1]) && !occursin("1.75", codes[1])
    @test occursin("1.75", codes[2]) && !occursin("0.25", codes[2])
    @test StanBlocks.stanc_check(codes[1]; warn_pedantic=false).ok
    @test StanBlocks.stanc_check(codes[2]; warn_pedantic=false).ok
end
const VECTOR_Z = @brm begin
    mu ~ 1 + z
    effect(mu, Intercept) ~ Normal(0, 2)
    effect(mu, z) ~ TDist(9)
    y ~ Normal(mu, 0.9)
end

const MIXTURE_TWO = @brm begin
    mu1 ~ Normal(-1, 1)
    mu2 ~ Normal(1, 1)
    y ~ MixtureModel([Normal(mu1, 0.9), Normal(mu2, 0.9)], [0.4, 0.6])
end
const MIXTURE_THREE = @brm begin
    mu1 ~ Normal(-1, 1)
    mu2 ~ Normal(1, 1)
    y ~ MixtureModel([Normal(mu1, 0.9), Normal(0, 0.9), Normal(mu2, 0.9)],
                    [0.2, 0.3, 0.5])
end
const HORSESHOE_X = @brm begin
    mu ~ 1 + x
    effect(mu, x) ~ Horseshoe()
    y ~ Normal(mu, 0.9)
end
const HORSESHOE_Z = @brm begin
    mu ~ 1 + z
    effect(mu, z) ~ Horseshoe()
    y ~ Normal(mu, 0.9)
end
const GENERATION_BUILDERS = (VECTOR_X, VECTOR_Z, MIXTURE_TWO, MIXTURE_THREE,
                             HORSESHOE_X, HORSESHOE_Z)

generation_result(index) = generation_result_for(GENERATION_BUILDERS[index])

@noinline function generation_result_for(builder)
    # Construction and consumption deliberately share a compiled caller. Tasks
    # waiting at the barrier predate other tasks' first family registration.
    sb = SBBRMI(builder(GENERATION_DATA); mod=@__MODULE__)
    descriptor = brm_descriptor(sb)
    # Symbolic dimensions such as num_elements(y) carry fresh trace objects;
    # their default equality compares those objects, not the expressions.
    # Compare expression spelling alongside exact emitted Stan and data.
    outputs = [(; output.name, output.kind, output.role, output.logical,
                output.type, size=repr(output.size), output.constraints, output.labels,
                output.segments)
               for output in brm_outputs(descriptor)]
    (; code=BRM.stan_code(sb), data=BRM.stan_data(sb), outputs)
end

function generation_mismatches(jobs, results, references)
    mismatches = []
    for (index, result) in zip(jobs, results)
        reference = references[index]
        result == reference && continue
        data_keys = [key for key in union(keys(result.data), keys(reference.data))
                     if !isequal(get(result.data, key, missing),
                                 get(reference.data, key, missing))]
        push!(mismatches, (; index, code=result.code == reference.code,
            data=result.data == reference.data, outputs=result.outputs == reference.outputs,
            data_isequal=isequal(result.data, reference.data),
            outputs_isequal=isequal(result.outputs, reference.outputs), data_keys))
    end
    mismatches
end

@stestset "cold and warm SBBRMI construction preserves code, data and outputs" begin
    input_snapshot = deepcopy(GENERATION_DATA)
    # One warm key overlaps other first-use keys; repeats also contend on each
    # initially cold key. The serial oracle is built AFTER that contention.
    warm = generation_result(1)
    jobs = repeat(collect(eachindex(GENERATION_BUILDERS)), 4)
    results = concurrent_builds(generation_result, jobs)
    references = map(generation_result, eachindex(GENERATION_BUILDERS))
    @test warm == references[1]
    @test generation_mismatches(jobs, results, references) == []
    @test GENERATION_DATA == input_snapshot

    # Identical prior layouts with different coefficient labels must preserve
    # each model's parameter names, regardless of construction order.
    @test occursin("hs_2_x_raw", references[5].code)
    @test !occursin("hs_2_z_raw", references[5].code)
    @test occursin("hs_2_z_raw", references[6].code)
    @test !occursin("hs_2_x_raw", references[6].code)
    for reference in references
        @test StanBlocks.stanc_check(reference.code; warn_pedantic=false).ok
    end

    # Reversed warm-cache order must not change any model artifact either.
    reversed = reverse(jobs)
    repeated = concurrent_builds(generation_result, reversed)
    @test generation_mismatches(reversed, repeated, references) == []
    @test GENERATION_DATA == input_snapshot
end

@stestset "same-named custom modules retain distinct family identities" begin
    check_same_named_modules()
end

@stestset "horseshoe submodels own their mutable syntax and data" begin
    specs = (; labels=[:Intercept, :x], hs=[nothing, (1.0, 1.0)])
    overrides = [nothing, nothing]
    originals = deepcopy((specs, overrides))
    models = concurrent_builds(_ -> BRM._sb_horseshoe_popefs_model(
        specs, overrides), 1:16)
    snapshot = deepcopy(first(models).model)
    first(first(models).model.args).args[end] = :(normal(123.0, 1.0))
    first(models).data[:caller_only] = true
    @test all(model.model == snapshot for model in models[2:end])
    @test all(isempty(model.data) for model in models[2:end])
    @test BRM._sb_horseshoe_popefs_model(specs, overrides).model == snapshot
    @test (specs, overrides) == originals
end

@stestset "invalid construction leaves later builds usable" begin
    # Empty vectors fail validation before family registration. Repeated
    # invalid input must keep failing while valid concurrent builds succeed.
    for _ in 1:2
        @test_throws ErrorException BRM._sb_vector_prior_family(BRM.ExprColumn[])
    end
    priors = [BRM.ExprColumn(Exponential, 1.5), BRM.ExprColumn(Normal, 0.0, 2.0)]
    families = concurrent_builds(_ -> first(BRM._sb_vector_prior_family(priors)), 1:8)
    @test all(family === first(families) for family in families)
end
