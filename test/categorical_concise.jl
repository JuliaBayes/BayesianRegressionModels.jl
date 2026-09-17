# test/categorical_concise.jl — explicit nested predictor-formula contracts.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/categorical_concise.jl

using Test
using BayesianRegressionModels
using Distributions
using LogDensityProblems
using StanBlocks
import CategoricalArrays as CA

const BRM = BayesianRegressionModels
const RUN_BRIDGESTAN = get(ENV, "BRM_CATEGORICAL_RUNTIME", "1") != "0"
const CACHE = joinpath(tempdir(), "brm-categorical-concise")

module FormulaSecurityProbe
    include(joinpath(@__DIR__, "..", "web-macro", "src", "security.jl"))
end

struct KeywordLikelihood end

df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    prog=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    y_cat=[20, 10, 30, 20, 30, 10],
    y=[-0.8, -0.2, 0.1, 0.7, 1.0, 1.4],
)

concise_builder = @brm begin
    y_cat ~ CategoricalLogit(@brm(1 + x + prog))
end

explicit_builder = @brm begin
    cat_eta2 ~ 1 + x + prog
    cat_eta3 ~ 1 + x + prog
    y_cat ~ CategoricalLogit(cat_eta2, cat_eta3)
end

distributional_builder = @brm begin
    y ~ Normal(@brm(1 + x), exp(@brm(1)))
end

keyword_builder = @brm begin
    y ~ KeywordLikelihood(; location=@brm(1 + x))
end

@testset "explicit nested @brm normalization" begin
    brmi = concise_builder(df)
    lp_names = (:y_cat_nested_arg1_class2, :y_cat_nested_arg1_class3)

    @test keys(brmi.operations) == (:x, :prog, lp_names..., :y_cat)
    @test [x.name for x in linear_predictors(brmi)] == collect(lp_names)
    @test popcoefnames(brmi, lp_names[1]) == [:Intercept, :x, :prog]
    @test popcoefnames(brmi, lp_names[2]) == [:Intercept, :x, :prog]
    @test predictors(brmi, lp_names[1]) ==
          (; intercept=true, continuous=[:x, :prog], categorical=Symbol[],
             re_terms=NamedTuple[])
    @test dependencies(brmi, lp_names[2]) ==
          (; data=[:x, :prog], intermediates=Symbol[])

    outcome = only(outcomes(brmi))
    @test outcome.response === :y_cat
    @test outcome.family === CategoricalLogit
    @test [x.link_lp for x in linear_predictor_args(outcome)] == collect(lp_names)

    # Row order does not affect generated names or plain-vector level order.
    reverse_df = (; (k => reverse(v) for (k, v) in pairs(df))...)
    reverse_brmi = concise_builder(reverse_df)
    @test keys(reverse_brmi.operations) == keys(brmi.operations)
    @test [x.name for x in linear_predictors(reverse_brmi)] == collect(lp_names)

    # A CategoricalVector's declared order, including the reference category,
    # controls the class-to-predictor mapping just as it does in explicit form.
    levelled_df = merge(df, (;
        y_cat=CA.categorical(df.y_cat; levels=[30, 20, 10]),
    ))
    levelled_brmi = concise_builder(levelled_df)
    @test keys(levelled_brmi.operations) == keys(brmi.operations)
    levelled_sb = SBBRMI(levelled_brmi; mod=@__MODULE__)
    @test levelled_sb.preproc[:y_cat].const_.levels == [30, 20, 10]

    # The marker is generic: every marker at a structural family-argument path
    # becomes an ordinary named predictor; wrappers such as exp stay ordinary.
    distributional = distributional_builder(df)
    @test keys(distributional.operations) ==
          (:x, :y_nested_arg1, :y_nested_arg2_arg1, :y)
    normal_outcome = only(outcomes(distributional))
    @test [x.link_lp for x in linear_predictor_args(normal_outcome)] ==
          [:y_nested_arg1, :y_nested_arg2_arg1]
    @test [x.link_fn for x in linear_predictor_args(normal_outcome)] ==
          [identity, exp]
    @test popcoefnames(distributional, :y_nested_arg1) == [:Intercept, :x]
    @test popcoefnames(distributional, :y_nested_arg2_arg1) == [:Intercept]

    keyword_brmi = keyword_builder(df)
    @test haskey(keyword_brmi.operations, :y_nested_kw_location)
    @test popcoefnames(keyword_brmi, :y_nested_kw_location) == [:Intercept, :x]
    _, keyword_rhs = getargs(parent(keyword_brmi.operations.y), 2)
    @test getkwargs(keyword_rhs) ==
          (; location=keyword_brmi.operations.y_nested_kw_location)
end

@testset "unmarked arguments and ambiguous placements stay explicit" begin
    explicit = explicit_builder(df)
    @test !any(contains(string(k), "_nested_") for k in keys(explicit.operations))
    @test SBBRMI(explicit; mod=@__MODULE__) isa SBBRMI

    unmarked_expression = @brm begin
        y_cat ~ CategoricalLogit(1 + x)
    end
    # Unmarked arguments keep ordinary expression semantics: one expression is
    # one predictor, so three outcome levels against one predictor fails the
    # arity check (the old scalar-LP-reference error is gone by design).
    @test_throws "received 1 non-reference predictors; expected 2" begin
        SBBRMI(unmarked_expression(df); mod=@__MODULE__)
    end

    mixed_categorical = @brm begin
        cat_eta2 ~ 1 + x
        y_cat ~ CategoricalLogit(@brm(1 + x), cat_eta2)
    end
    @test_throws "exactly one direct marker" mixed_categorical(df)

    marker_in_lp = @brm begin
        eta ~ @brm(1 + x)
        y ~ Normal(eta, 1)
    end
    @test_throws "not the linear predictor" marker_in_lp(df)

    marker_in_assignment = @brm begin
        sigma = exp(@brm(1))
        y ~ Normal(0, sigma)
    end
    @test_throws "not inside `sigma = ...`" marker_in_assignment(df)

    direct_marker = @brm begin
        y ~ @brm(1 + x)
    end
    @test_throws "must be an argument of the observed family" direct_marker(df)

    @test_throws "standalone predictor fragment" begin
        macroexpand(@__MODULE__, :(@brm(1 + x)))
    end
    @test_throws "cannot contain another nested `@brm`" begin
        macroexpand(@__MODULE__, :(@brm begin
            y ~ Normal(@brm(1 + @brm(x)), 1)
        end))
    end
end

@testset "web formula allowlist admits only the exact marker" begin
    accepted = Meta.parse("begin\ny_cat ~ CategoricalLogit(@brm(1 + x))\nend")
    @test isnothing(FormulaSecurityProbe.walk(accepted))

    other_macro = Meta.parse("begin\ny ~ Normal(@time(1 + x), 1)\nend")
    @test FormulaSecurityProbe.walk(other_macro) isa
          FormulaSecurityProbe.FormulaSecurityError

    unsafe_payload = Meta.parse("begin\ny ~ Normal(@brm(run(x)), 1)\nend")
    violation = FormulaSecurityProbe.walk(unsafe_payload)
    @test violation isa FormulaSecurityProbe.FormulaSecurityError
    @test occursin("not in the formula allowlist", violation.msg)

    recursive_marker =
        Meta.parse("begin\ny ~ Normal(@brm(1 + @brm(x)), 1)\nend")
    violation = FormulaSecurityProbe.walk(recursive_marker)
    @test violation isa FormulaSecurityProbe.FormulaSecurityError
    @test occursin("cannot contain another nested", violation.msg)
end

@testset "concise categorical reaches Stan and BridgeStan" begin
    concise = concise_builder(df)
    concise_sb = SBBRMI(concise; mod=@__MODULE__)
    concise_code = BRM.stan_code(concise_sb)
    @test StanBlocks.stanc_check(concise_code; warn_pedantic=false).ok
    @test occursin("y_cat ~ categorical_logit(y_cat_categorical_logits);",
                   concise_code)
    @test occursin("y_cat_nested_arg1_class2", concise_code)
    @test occursin("y_cat_nested_arg1_class3", concise_code)
    @test concise_sb.data[:y_cat] == [2, 1, 3, 2, 3, 1]
    @test concise_sb.preproc[:y_cat].const_.levels == [10, 20, 30]

    distributional_sb = SBBRMI(distributional_builder(df); mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(distributional_sb);
                                 warn_pedantic=false).ok

    replayed = reprocess(concise_sb, merge(df, (;
        x=df.x .+ 0.25,
        prog=reverse(df.prog),
    )))
    @test replayed.data[:y_cat] == concise_sb.data[:y_cat]
    unseen = merge(df, (; y_cat=[10, 20, 30, 40, 10, 20]))
    @test_throws "not a training level" reprocess(concise_sb, unseen)

    if RUN_BRIDGESTAN
        isdir(CACHE) || mkpath(CACHE)
        problem = StanBlocks.stan_instantiate(
            concise_sb.model;
            path=joinpath(CACHE, string(hash(concise_code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        @test dimension == 6
        q = zeros(dimension)
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping concise categorical BridgeStan gate (BRM_CATEGORICAL_RUNTIME=0)"
    end
end
