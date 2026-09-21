# test/reprocess_single_level_factor.jl — REGRESSION for snag reprocess-formul-d5e1fb67.
#
# A single-level (K == 1) categorical predictor inside a DISTRIBUTIONAL scale
# submodel (`log_sigma ~ 1 + ecg_source`) must replay through `reprocess` —
# frozen and via `resample_groups` — exactly like a K >= 2 factor: the derived
# `<c>_idx` codes regenerate from the new frame's raw column against the fitted
# level set while `<c>_n_levels` stays frozen and the fitted Stan is preserved.
#
# Root cause (shared with handled snag `one-level-nomina-1ff57ba5`): the old
# `_sb_emit_cat!` had an `if n_levels < 2` branch that wrote a literal
# `data[:cat_<c>]` zero column and returned BEFORE recording any `PreprocEntry`,
# so `reprocess` rejected the model with "data key `cat_<c>` is a derived
# vector with no preprocessing record". The landed fix (`83176f5`, widened by
# `4151172`) deletes that branch: the uniform `_sb_cat` path degenerates on its
# own at K == 1 AND records `PreprocEntry(:factor, …)`, so no `cat_`-prefixed
# data key exists anymore. This file locks that in for the distributional +
# resample shape the sibling snag's (unlanded) plain-predictor test does not
# cover, and pins the legacy-artifact symptom: an SBBRMI built before the fix
# still carries `cat_<c>` and must fail loud until it is rebuilt.
#
# Run: julia --project=test test/reprocess_single_level_factor.jl
using Test
using BayesianRegressionModels
using StanBlocks
import CategoricalArrays as CA

builder = @brm begin
    mu ~ 1 + x + (1 | subject)
    log_sigma ~ 1 + ecg_source
    y ~ Normal(mu, exp(log_sigma))
end
train = (;
    subject = repeat([1, 2, 3], inner = 2),
    x = [0.5, -0.3, 1.1, 0.2, -0.7, 0.9],
    ecg_source = CA.categorical(fill("A", 6); levels = ["A"]),
    y = [0.2, 1.1, -0.4, 0.7, 0.3, 1.2],
)
# `total_groups=()` keeps the conventional artifact the `resample_groups`
# geometry check requires (totals resample through the plan/transport route).
fit = SBBRMI(builder(train); mod = @__MODULE__, total_groups = ())

@testset "single-level factor emission carries :factor provenance" begin
    @test !any(k -> startswith(String(k), "cat_"), keys(fit.data))
    @test haskey(fit.preproc, :ecg_source_idx)
    @test fit.preproc[:ecg_source_idx].kind === :factor
    @test fit.preproc[:ecg_source_idx].raw_ref === :ecg_source
    @test fit.data[:ecg_source_idx] == ones(Int, 6)
    @test fit.data[:ecg_source_n_levels] == 1
end

future = (;
    subject = repeat([1, 2], inner = 2),
    x = [0.1, 0.4, -0.2, 0.8],
    ecg_source = CA.categorical(fill("A", 4); levels = ["A"]),
    y = zeros(4),
)

@testset "frozen replay regenerates the lone level, Stan byte-identical" begin
    rp = reprocess(fit, future; freeze_constants = true)
    @test rp.data[:ecg_source_idx] == ones(Int, 4)
    @test rp.data[:ecg_source_n_levels] == 1
    @test rp.preproc[:ecg_source_idx].kind === :factor
    @test StanBlocks.stan_code(rp.model) == StanBlocks.stan_code(fit.model)
end

@testset "resample replay keeps the frozen factor (reporter call shape)" begin
    newpop = merge(future, (; subject = repeat([101, 203], inner = 2)))
    to = reprocess(fit, newpop; freeze_constants = true,
                   resample_groups = [:subject])
    @test to.data[:ecg_source_idx] == ones(Int, 4)
    @test to.data[:ecg_source_n_levels] == 1
    @test StanBlocks.stanc_check(StanBlocks.stan_code(to.model);
                                 warn_pedantic = false).ok
end

@testset "unseen level fails loud; freeze=false grows K=1 -> K=2" begin
    unseen = merge(future, (; ecg_source = CA.categorical(["A", "A", "B", "B"];
                                                          levels = ["A", "B"])))
    @test_throws "not a training level" reprocess(fit, unseen;
                                                  freeze_constants = true)
    grown = reprocess(fit, unseen; freeze_constants = false)
    @test grown.data[:ecg_source_idx] == [1, 1, 2, 2]
    @test grown.data[:ecg_source_n_levels] == 2
end

@testset "pre-fix artifact shape still fails with the reporter symptom" begin
    legacy_data = copy(fit.data)
    legacy_data[:cat_ecg_source] = zeros(6)
    legacy = SBBRMI(fit.parent, fit.model, legacy_data, fit.preproc,
                    copy(fit.held_out), fit.bindings)
    @test_throws "derived vector with no" reprocess(legacy, future;
                                                    freeze_constants = true)
end
