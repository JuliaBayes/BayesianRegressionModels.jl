# test/interaction_effect_priors.jl — `effect(lp, a & b)` whole-interaction priors.
#
# Run: julia --project=. test/interaction_effect_priors.jl
#
# An interaction term is addressable the way the formula spells it:
# `effect(mu, gen & mathz)` sets one shared Normal prior over every beta_pop
# column the term emits, in either operand order. SBBRMI resolves the address
# against the traced `&` terms; the backend-neutral design (Turing/RK path)
# carries the same key as the columns' shared block.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels

# counts.jl-style: Float64 0/1 dummies + standardized math.
const INT_N = 24
const INT_ACAD = Float64.(mod.(1:INT_N, 3) .== 0)
const INT_GEN = Float64.(mod.(1:INT_N, 3) .== 1)
const INT_VOC = Float64.(mod.(1:INT_N, 3) .== 2)
const INT_MATH0 = Float64.(collect(1:INT_N))
const INT_MATHZ =
    (INT_MATH0 .- sum(INT_MATH0) / INT_N) ./
    sqrt(sum((INT_MATH0 .- sum(INT_MATH0) / INT_N) .^ 2) / INT_N)
const INT_DAYSABS = collect(1:INT_N)
const INT_GCODE = Int.(mod.(1:INT_N, 3) .+ 1)
const INT_COUNT_DF =
    (; daysabs=INT_DAYSABS, acad=INT_ACAD, gen=INT_GEN, voc=INT_VOC,
     mathz=INT_MATHZ)

# A macro-expansion failure (where `@brm` address errors raise) escapes a
# surrounding try/catch, so refusal cases eval from string.
function brm_parse_message(formula::String)
    try
        eval(Meta.parse(formula))
        nothing
    catch err
        sprint(showerror, err)
    end
end

@testset "parse: & address accepted, order-insensitive" begin
    brmi = (@brm begin
        log(mu) ~ 0 + acad + gen + voc + mathz + gen & mathz + voc & mathz
        effect(mu, gen & mathz) ~ Normal(0, 2.5)
        phi ~ Exponential(1)
        daysabs ~ NegativeBinomial2(mu, phi)
    end)(INT_COUNT_DF)
    specs = effect_priors(brmi)
    @test length(specs) == 1
    @test specs[1].predictor === :mu
    @test specs[1].coefficient === Symbol("gen&mathz")
    @test specs[1].family <: Normal
    @test popcoefnames(brmi, :mu) ==
          [:acad, :gen, :voc, :mathz, :int_gen_x_mathz, :int_voc_x_mathz]

    # Flipped surface order: same key, order-following labels.
    flipped = (@brm begin
        log(mu) ~ 0 + acad + gen + voc + mathz + mathz & gen + voc & mathz
        effect(mu, mathz & gen) ~ Normal(0, 2.5)
        phi ~ Exponential(1)
        daysabs ~ NegativeBinomial2(mu, phi)
    end)(INT_COUNT_DF)
    @test only(effect_priors(flipped)).coefficient === Symbol("gen&mathz")
    @test popcoefnames(flipped, :mu) ==
          [:acad, :gen, :voc, :mathz, :int_mathz_x_gen, :int_voc_x_mathz]
end

@testset "parse: & address refusals stay loud" begin
    transformed = brm_parse_message("""
        @brm begin
            log(mu) ~ 1 + mathz + gen & mathz
            effect(mu, (mathz + 0.0) & gen) ~ Normal(0, 1)
            phi ~ Exponential(1)
            daysabs ~ NegativeBinomial2(mu, phi)
        end""")
    @test !isnothing(transformed) && occursin("two plain columns", transformed)

    nested = brm_parse_message("""
        @brm begin
            log(mu) ~ 1 + acad + gen + voc
            effect(mu, acad & gen & voc) ~ Normal(0, 1)
            phi ~ Exponential(1)
            daysabs ~ NegativeBinomial2(mu, phi)
        end""")
    @test !isnothing(nested) && occursin("two plain columns", nested)

    predictor_slot = brm_parse_message("""
        @brm begin
            log(mu) ~ 1 + gen + mathz
            effect(gen & mathz, mathz) ~ Normal(0, 1)
            phi ~ Exponential(1)
            daysabs ~ NegativeBinomial2(mu, phi)
        end""")
    @test !isnothing(predictor_slot) &&
          occursin("COEFFICIENT slot", predictor_slot)

    wrong_head = brm_parse_message("""
        @brm begin
            log(mu) ~ 1 + gen + mathz
            sd(mu, gen & mathz) ~ Exponential(1)
            phi ~ Exponential(1)
            daysabs ~ NegativeBinomial2(mu, phi)
        end""")
    @test !isnothing(wrong_head) &&
          occursin("through `effect`", wrong_head)

    duplicate = brm_parse_message("""
        @brm begin
            log(mu) ~ 1 + gen + mathz + gen & mathz
            effect(mu, gen & mathz) ~ Normal(0, 1)
            effect(mu, mathz & gen) ~ Normal(0, 2)
            phi ~ Exponential(1)
            daysabs ~ NegativeBinomial2(mu, phi)
        end""")
    @test !isnothing(duplicate) && occursin("duplicate", duplicate)
end

@testset "SBBRMI: reporter's negative-binomial spelling builds" begin
    brmi = (@brm begin
        log(mu) ~ 0 + acad + gen + voc + mathz + gen & mathz + voc & mathz
        effect(mu, acad) ~ Normal(0, 2.5)
        effect(mu, gen) ~ Normal(0, 2.5)
        effect(mu, voc) ~ Normal(0, 2.5)
        effect(mu, mathz) ~ Normal(0, 2.5)
        effect(mu, gen & mathz) ~ Normal(0, 2.5)
        effect(mu, voc & mathz) ~ Normal(0, 2.5)
        phi ~ Exponential(1)
        daysabs ~ NegativeBinomial2(mu, phi)
    end)(INT_COUNT_DF)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("[2.5, 2.5, 2.5, 2.5, 2.5, 2.5]'", code)
end

@testset "SBBRMI: & prior lands on exactly its columns" begin
    brmi = (@brm begin
        log(mu) ~ 0 + acad + gen + voc + mathz + gen & mathz + voc & mathz
        effect(mu, gen & mathz) ~ Normal(0, 2.5)
        phi ~ Exponential(1)
        daysabs ~ NegativeBinomial2(mu, phi)
    end)(INT_COUNT_DF)
    overrides = BRM._sb_effect_prior_overrides(brmi)
    @test Set(keys(overrides)) == Set([:mu])
    @test length(overrides[:mu].pop) == 6
    @test [isnothing(p) for p in overrides[:mu].pop] ==
          [true, true, true, true, false, true]
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("[1.0, 1.0, 1.0, 1.0, 2.5, 1.0]'", code)
end

@testset "SBBRMI: cont×cat fan-out + per-level refinement" begin
    df = (; daysabs=INT_DAYSABS, gcode=INT_GCODE, mathz=INT_MATHZ)
    brmi = (@brm begin
        log(mu) ~ 1 + gcode + mathz + gcode & mathz
        effect(mu, gcode & mathz) ~ Normal(0, 2.5)
        effect(mu, int_mathz_x_gcode_lvl_3) ~ Normal(1, 0.5)
        phi ~ Exponential(1)
        daysabs ~ NegativeBinomial2(mu, phi)
    end)(df)
    @test popcoefnames(brmi, :mu) == [:Intercept, :mathz,
        :int_mathz_x_gcode_lvl_2, :int_mathz_x_gcode_lvl_3]
    overrides = BRM._sb_effect_prior_overrides(brmi)
    @test [isnothing(p) for p in overrides[:mu].pop] ==
          [true, true, false, false]
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # lvl_2 carries the fanned-out (0, 2.5); lvl_3 the refining (1, 0.5).
    @test occursin("[1.0, 1.0, 2.5, 0.5]'", code)
end

@testset "SBBRMI: : fan-out reaches the owning predictor" begin
    brmi = (@brm begin
        mu ~ 1 + gen & mathz
        log(sigma) ~ 1 + acad
        effect(:, gen & mathz) ~ Normal(0, 2.5)
        y ~ Normal(mu, sigma)
    end)((; y=Float64.(INT_DAYSABS), gen=INT_GEN, mathz=INT_MATHZ,
          acad=INT_ACAD))
    overrides = BRM._sb_effect_prior_overrides(brmi)
    @test Set(keys(overrides)) == Set([:mu])
    @test [isnothing(p) for p in overrides[:mu].pop] == [true, false]
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("[1.0, 2.5]'", code)
    # `log(sigma)` keeps the plain default emission (no `_popefs_normal`
    # scales literal at all) — the keys assertion above is the fan-out proof.
    @test !occursin("[1.0, 1.0]'", code)
end

@testset "SBBRMI: unknown operands fail loudly" begin
    brmi = (@brm begin
        log(mu) ~ 0 + acad + gen + voc + mathz + gen & mathz
        effect(mu, gen & nosuch) ~ Normal(0, 1)
        phi ~ Exponential(1)
        daysabs ~ NegativeBinomial2(mu, phi)
    end)(INT_COUNT_DF)
    msg = try
        SBBRMI(brmi; mod=@__MODULE__)
        nothing
    catch err
        sprint(showerror, err)
    end
    @test !isnothing(msg) && occursin("not a population coefficient", msg)
    @test occursin("Interaction term(s)", msg)
    @test occursin("gen & mathz", msg)
end

@testset "neutral design: cont×cont fan-out" begin
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 0 + acad + gen + voc + mathz + gen & mathz + voc & mathz
        effect(mu, gen & mathz) ~ Normal(0, 2.5)
        y ~ Normal(mu, sigma)
    end)((; y=Float64.(INT_DAYSABS), acad=INT_ACAD, gen=INT_GEN,
          voc=INT_VOC, mathz=INT_MATHZ))
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    @test Tuple(c.label for c in design.columns) ==
          (:acad, :gen, :voc, :mathz, :int_gen_x_mathz, :int_voc_x_mathz)
    @test design.columns[5].effect_addresses ==
          (:int_gen_x_mathz, Symbol("gen&mathz"))
    @test design.columns[5].effect_block === Symbol("gen&mathz")
    overrides = BRM._brm_simple_population_effect_overrides(brmi, design)
    location, scale = BRM._brm_materialize_normal_effect_priors(
        overrides, length(design.columns))
    @test location == zeros(6)
    @test scale == [1.0, 1.0, 1.0, 1.0, 2.5, 1.0]
end

@testset "neutral design: cont×cat fan-out + refinement" begin
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + gcode + mathz + gcode & mathz
        effect(mu, gcode & mathz) ~ Normal(0, 2.5)
        effect(mu, int_mathz_x_gcode_lvl_3) ~ Normal(1, 0.5)
        y ~ Normal(mu, sigma)
    end)((; y=Float64.(INT_DAYSABS), gcode=INT_GCODE, mathz=INT_MATHZ))
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    @test Tuple(c.label for c in design.columns) == (:Intercept, :gcode_lvl_2,
        :gcode_lvl_3, :mathz, :int_mathz_x_gcode_lvl_2,
        :int_mathz_x_gcode_lvl_3)
    @test design.columns[5].effect_block === Symbol("gcode&mathz")
    @test design.columns[6].effect_block === Symbol("gcode&mathz")
    overrides = BRM._brm_simple_population_effect_overrides(brmi, design)
    location, scale = BRM._brm_materialize_normal_effect_priors(
        overrides, length(design.columns))
    @test location == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    @test scale == [1.0, 1.0, 1.0, 1.0, 2.5, 0.5]
end

@testset "neutral design: unknown key fails loudly" begin
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 0 + acad + gen + voc + mathz + gen & mathz
        effect(mu, gen & nosuch) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end)((; y=Float64.(INT_DAYSABS), acad=INT_ACAD, gen=INT_GEN,
          voc=INT_VOC, mathz=INT_MATHZ))
    context = BRM._brm_backend_context(brmi)
    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    msg = try
        BRM._brm_simple_population_effect_overrides(brmi, design)
        nothing
    catch err
        sprint(showerror, err)
    end
    @test !isnothing(msg) && occursin("not a population coefficient", msg)
    @test occursin("gen&mathz", msg)
end
