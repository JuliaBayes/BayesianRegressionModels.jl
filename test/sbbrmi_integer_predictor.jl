# test/sbbrmi_integer_predictor.jl — SBBRMI bare-integer-predictor warning gate.
#
# Snag brm-int-predicto-b36205e3: a bare integer column in a population
# formula (`eta ~ 1 + len + sex` with `sex::Vector{Int}`) is silently
# treatment-coded into its own `cat_<lp>_<col>` block and is ABSENT from
# `beta_pop` — the Integer-means-categorical rule. A consumer summarizing
# `beta_pop` reads that as "dropped" (the bambi alligator replication cost a
# full refit cycle to exactly that misdiagnosis). SBBRMI now warns once per
# predictor at lowering time, naming the reinterpretation, the emitted block,
# and both silencing spellings (`protect(col)` for a numeric slope,
# `factor(col)` / `CategoricalVector` for explicit categorical intent).
#
# The warning is the ONLY change: emission is byte-identical, so this gate
# also pins the cat-block layout and the `protect` numeric escape hatch.
# Transpile + stanc (mirrors the sbbrmi_bernoulli SBBRMI testset); no
# BridgeStan compile.
#
# NOTE on predictor names: the warning fires once per predictor per session
# (`maxlog=1`, predictor-keyed `_id`), so every warn-asserting case below
# uses a FRESH linear-predictor name. Reusing a name across cases would let
# session suppression eat the second warning.

using Test
using Logging
using BayesianRegressionModels
using StanBlocks
import CategoricalArrays as CA
import StanBlocks.stan: transpiles

intpred_data(; sex=[1, 2, 1, 2, 1, 2]) = (;
    y=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    len=[1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
    sex=sex)

# Build an SBBRMI capturing every Warn+-level log alongside the value.
intpred_logged(builder, df) =
    Test.collect_test_logs(() -> SBBRMI(builder(df); mod=@__MODULE__);
                           min_level=Logging.Warn)

@testset "bare integer predictor warns and names the cat block" begin
    builder = df -> (@brm begin
        eta ~ 1 + len + sex
        y ~ BernoulliLogit(eta)
    end)(df)
    logs, sb = intpred_logged(builder, intpred_data())
    @test length(logs) == 1
    @test logs[1].level == Logging.Warn
    @test occursin("`sex`", logs[1].message)
    @test occursin("cat_eta_sex", logs[1].message)
    @test occursin("beta_pop", logs[1].message)
    @test occursin("protect(sex)", logs[1].message)
    @test occursin("factor(sex)", logs[1].message)

    # Emission is unchanged: sex is treatment-coded, absent from beta_pop.
    @test popcoefnames(sb.parent, :eta) == [:Intercept, :len]
    code = BayesianRegressionModels.stan_code(sb)
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("cat_eta_sex_beta", code)
    @test occursin("eta = (pop_eta + cat_eta_sex)", code)
end

@testset "several bare integer columns share one warning" begin
    builder = df -> (@brm begin
        mu2 ~ 1 + len + sex + grp
        y ~ BernoulliLogit(mu2)
    end)(merge(df, (; grp=[1, 1, 2, 2, 1, 2])))
    logs, sb = intpred_logged(builder, intpred_data())
    @test length(logs) == 1
    @test occursin("cat_mu2_sex", logs[1].message)
    @test occursin("cat_mu2_grp", logs[1].message)
    @test popcoefnames(sb.parent, :mu2) == [:Intercept, :len]
end

@testset "explicit categorical spellings stay silent" begin
    # `factor(sex)` declares the categorical intent explicitly.
    factor_builder = df -> (@brm begin
        eta_f ~ 1 + len + factor(sex)
        y ~ BernoulliLogit(eta_f)
    end)(df)
    factor_logs, factor_sb = intpred_logged(factor_builder, intpred_data())
    @test isempty(factor_logs)
    @test popcoefnames(factor_sb.parent, :eta_f) == [:Intercept, :len]

    # A `CategoricalVector` column is likewise explicit.
    cat_builder = df -> (@brm begin
        eta_c ~ 1 + len + sex
        y ~ BernoulliLogit(eta_c)
    end)(df)
    cat_logs, cat_sb = intpred_logged(
        cat_builder, intpred_data(; sex=CA.categorical([1, 2, 1, 2, 1, 2])))
    @test isempty(cat_logs)
    @test popcoefnames(cat_sb.parent, :eta_c) == [:Intercept, :len]
end

@testset "protect keeps the numeric slope and stays silent" begin
    builder = df -> (@brm begin
        eta_p ~ 1 + len + protect(sex)
        y ~ BernoulliLogit(eta_p)
    end)(df)
    logs, sb = intpred_logged(builder, intpred_data())
    @test isempty(logs)
    names = popcoefnames(sb.parent, :eta_p)
    @test length(names) == 3
    @test names[1:2] == [:Intercept, :len]
    code = BayesianRegressionModels.stan_code(sb)
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test !occursin("cat_eta_p_sex", code)
end

@testset "bare Bool predictor warns (Bool <: Integer)" begin
    builder = df -> (@brm begin
        eta_b ~ 1 + len + flag
        y ~ BernoulliLogit(eta_b)
    end)(merge(df, (; flag=[true, false, true, false, true, false])))
    logs, sb = intpred_logged(builder, intpred_data())
    @test length(logs) == 1
    @test occursin("cat_eta_b_flag", logs[1].message)
    @test popcoefnames(sb.parent, :eta_b) == [:Intercept, :len]
end
